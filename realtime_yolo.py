# realtime_yolo.py
#
# Real-time (or video) YOLO person detection + centroid tracking +
# Haar face detection + optional face collection.
#
# Usage examples:
#   python realtime_yolo.py --video testvideo.mp4
#   python realtime_yolo.py --video testvideo.mp4 --collect-label keanu
#   python realtime_yolo.py --video testvideo.mp4 --no-collect
#
# This is now a small "app" instead of a hard-coded script.

import argparse
import csv
import logging
import os
import time
from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# Optional deps (only needed if you enable the flags)
try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception:
    DeepSort = None


# ----------------- BBOX / MATH HELPERS -----------------
def clamp_bbox(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W - 1, int(x1)))
    y1 = max(0, min(H - 1, int(y1)))
    x2 = max(0, min(W - 1, int(x2)))
    y2 = max(0, min(H - 1, int(y2)))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2


def expand_bbox(x1, y1, x2, y2, pad, W, H):
    return clamp_bbox(x1 - pad, y1 - pad, x2 + pad, y2 + pad, W, H)


def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def xywh_to_xyxy(x, y, w, h):
    return (int(x), int(y), int(x + w), int(y + h))


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def now_stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def softmax(x):
    x = x.astype(np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    if s <= 1e-9:
        return np.ones_like(x) / float(len(x))
    return ex / s


# ----------------- ANONYMIZATION -----------------
def pixelate(img, x1, y1, x2, y2, blocks=14):
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    bw = max(1, w // blocks)
    bh = max(1, h // blocks)
    small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = pix


# ----------------- YUNET LOADER -----------------
def create_yunet(model_path, score_thresh=0.6, nms_thresh=0.3, top_k=5000):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YuNet model not found at: {model_path}")

    if hasattr(cv2, "FaceDetectorYN_create"):
        return cv2.FaceDetectorYN_create(model_path, "", (320, 320), score_thresh, nms_thresh, top_k)

    if hasattr(cv2, "FaceDetectorYN") and hasattr(cv2.FaceDetectorYN, "create"):
        return cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_thresh, nms_thresh, top_k)

    raise RuntimeError("Your OpenCV build does not expose FaceDetectorYN / YuNet.")


# ----------------- EMBEDDINGS (ARCFACE via OpenCV DNN) -----------------
class ArcFaceEmbedder:
    """
    Minimal ArcFace ONNX wrapper using OpenCV DNN.
    - Input: BGR face crop
    - Output: L2-normalized embedding vector (1D)
    """
    def __init__(self, onnx_path, input_size=112):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ArcFace ONNX not found at: {onnx_path}")
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.input_size = int(input_size)

    def _preprocess(self, face_bgr):
        face = cv2.resize(face_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        blob = face.astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]  # NCHW
        return blob

    def embed(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            return None
        blob = self._preprocess(face_bgr)
        self.net.setInput(blob)
        out = self.net.forward()
        vec = out.reshape(-1).astype(np.float32)
        n = np.linalg.norm(vec)
        if n > 1e-9:
            vec = vec / n
        return vec


# ----------------- EXPRESSION (FER+ via ONNX Runtime, AUTO LAYOUT) -----------------
FERPLUS_LABELS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

class FerplusEmotion:
    """
    Robust FER+ runner:
      - uses onnxruntime
      - auto-detects input layout (NCHW vs NHWC)
      - safer preprocessing + output handling
    """
    def __init__(self, model_path, norm="-11"):
        if ort is None:
            raise RuntimeError("onnxruntime not installed. pip install onnxruntime")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Emotion model not found at: {model_path}")

        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0]
        self.in_name = self.input.name
        self.in_shape = self.input.shape  # may contain None
        self.norm = norm

        # layout guess
        # common: [1,1,64,64] or [1,64,64,1]
        self.layout = "NCHW"
        if len(self.in_shape) == 4:
            # if channel-last
            if self.in_shape[-1] == 1 and (self.in_shape[1] in (64, None)):
                self.layout = "NHWC"
            # if channel-first
            if self.in_shape[1] == 1:
                self.layout = "NCHW"

        outs = self.sess.get_outputs()
        self.out_names = [o.name for o in outs]

    def _prep(self, face_bgr):
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        x = gray.astype(np.float32) / 255.0

        # normalization options
        # "-11": map [0,1] -> [-1,1]
        if self.norm == "-11":
            x = (x - 0.5) / 0.5
        # "01": keep [0,1]

        if self.layout == "NCHW":
            x = x[None, None, :, :]  # (1,1,64,64)
        else:
            x = x[None, :, :, None]  # (1,64,64,1)
        return x

    def infer(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            return None, 0.0, None

        inp = self._prep(face_bgr)

        outs = self.sess.run(None, {self.in_name: inp})
        if not outs:
            return None, 0.0, None

        scores = np.array(outs[0]).reshape(-1).astype(np.float32)
        if scores.size < 8:
            # some exports wrap as [1,1,8] etc, reshape already handles it
            return None, 0.0, None

        probs = softmax(scores[:8])
        k = int(np.argmax(probs))
        return FERPLUS_LABELS[k], float(probs[k]), probs


# ----------------- HEAD POSE + GAZE (MEDIAPIPE FACE MESH) -----------------
class FaceMeshPoseGaze:
    """
    Experimental:
      - head pose via solvePnP from a few face landmarks
      - gaze via iris center vs eye corners (rough)
    """
    def __init__(self):
        if mp is None:
            raise RuntimeError("mediapipe not installed. pip install mediapipe")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,   # enables iris landmarks
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # landmark indices (MediaPipe FaceMesh)
        self.IDX_NOSE_TIP = 1
        self.IDX_CHIN = 152
        self.IDX_L_EYE = 33
        self.IDX_R_EYE = 263
        self.IDX_L_MOUTH = 61
        self.IDX_R_MOUTH = 291

        # Iris landmarks
        self.IDX_L_IRIS = [468, 469, 470, 471]
        self.IDX_R_IRIS = [473, 474, 475, 476]

        self.IDX_L_EYE_CORNER_IN = 133
        self.IDX_L_EYE_CORNER_OUT = 33
        self.IDX_R_EYE_CORNER_IN = 362
        self.IDX_R_EYE_CORNER_OUT = 263

    def _lm_xy(self, lm, w, h):
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)

    def estimate(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            return None

        h, w = face_bgr.shape[:2]
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None

        lms = res.multi_face_landmarks[0].landmark

        # 2D image points
        image_points = np.array([
            self._lm_xy(lms[self.IDX_NOSE_TIP], w, h),
            self._lm_xy(lms[self.IDX_CHIN], w, h),
            self._lm_xy(lms[self.IDX_L_EYE], w, h),
            self._lm_xy(lms[self.IDX_R_EYE], w, h),
            self._lm_xy(lms[self.IDX_L_MOUTH], w, h),
            self._lm_xy(lms[self.IDX_R_MOUTH], w, h),
        ], dtype=np.float32)

        # Generic 3D model points (approx, in mm-ish units)
        model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (0.0, -63.6, -12.5),    # Chin
            (-43.3, 32.7, -26.0),   # Left eye corner
            (43.3, 32.7, -26.0),    # Right eye corner
            (-28.9, -28.9, -24.1),  # Left mouth corner
            (28.9, -28.9, -24.1),   # Right mouth corner
        ], dtype=np.float32)

        focal = w
        center = (w / 2.0, h / 2.0)
        cam_mtx = np.array([
            [focal, 0, center[0]],
            [0, focal, center[1]],
            [0, 0, 1],
        ], dtype=np.float32)
        dist = np.zeros((4, 1), dtype=np.float32)

        ok, rvec, tvec = cv2.solvePnP(model_points, image_points, cam_mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None

        R, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[1, 0], R[0, 0])
        else:
            pitch = np.arctan2(-R[1, 2], R[1, 1])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = 0

        # degrees
        pitch = float(pitch * 180.0 / np.pi)
        yaw = float(yaw * 180.0 / np.pi)
        roll = float(roll * 180.0 / np.pi)

        # Gaze (rough): iris center position normalized within eye corners
        def iris_center(idxs):
            pts = np.stack([self._lm_xy(lms[i], w, h) for i in idxs], axis=0)
            return pts.mean(axis=0)

        li = iris_center(self.IDX_L_IRIS)
        ri = iris_center(self.IDX_R_IRIS)

        l_in = self._lm_xy(lms[self.IDX_L_EYE_CORNER_IN], w, h)
        l_out = self._lm_xy(lms[self.IDX_L_EYE_CORNER_OUT], w, h)
        r_in = self._lm_xy(lms[self.IDX_R_EYE_CORNER_IN], w, h)
        r_out = self._lm_xy(lms[self.IDX_R_EYE_CORNER_OUT], w, h)

        def gaze_1d(iris, a, b):
            # returns 0..1 position along eye corner line
            denom = np.linalg.norm(b - a) + 1e-6
            return float(np.dot(iris - a, b - a) / (denom * denom))

        gaze_l = gaze_1d(li, l_out, l_in)
        gaze_r = gaze_1d(ri, r_in, r_out)

        # map to text
        gaze_x = (gaze_l + gaze_r) * 0.5
        if gaze_x < 0.42:
            gaze_txt = "LEFT"
        elif gaze_x > 0.58:
            gaze_txt = "RIGHT"
        else:
            gaze_txt = "CENTER"

        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "gaze": gaze_txt,
        }


# ----------------- DETECTION STRUCT -----------------
@dataclass
class Detection:
    bbox_xyxy: tuple
    conf: float
    from_detector: bool
    emb: Optional[np.ndarray] = None
    cam_id: int = 0


# ----------------- KALMAN TRACK (BOX) -----------------
class KalmanBoxTrack:
    """
    State: [x, y, w, h, vx, vy, vw, vh]
    """
    def __init__(self, track_id, init_bbox_xyxy):
        self.id = track_id

        # bookkeeping
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.last_meas_from_detector = True
        self._last_pred = None
        self._last_meas_frame = 0

        # identity / reid
        self.emb_ema = None
        self.label = "UNKNOWN"
        self.label_sim = 0.0
        self.global_id = None

        # expression
        self.expr_probs_ema = None
        self.expr_label = None
        self.expr_prob = 0.0
        self.expr_last_frame = 0

        # pose/gaze
        self.yaw = None
        self.pitch = None
        self.roll = None
        self.gaze = None
        self.pose_last_frame = 0

        # Kalman
        self.kf = cv2.KalmanFilter(8, 4)
        dt = 1.0

        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, dt, 0,  0,  0],
                [0, 1, 0, 0, 0,  dt, 0,  0],
                [0, 0, 1, 0, 0,  0,  dt, 0],
                [0, 0, 0, 1, 0,  0,  0,  dt],
                [0, 0, 0, 0, 1,  0,  0,  0],
                [0, 0, 0, 0, 0,  1,  0,  0],
                [0, 0, 0, 0, 0,  0,  1,  0],
                [0, 0, 0, 0, 0,  0,  0,  1],
            ],
            dtype=np.float32,
        )

        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 2e-1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 1.0

        x, y, w, h = xyxy_to_xywh(init_bbox_xyxy)
        self.kf.statePost = np.array([[x], [y], [w], [h], [0], [0], [0], [0]], dtype=np.float32)

    def predict(self):
        pred = self.kf.predict()
        self._last_pred = pred
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox_xyxy, meas_from_detector=True, frame_idx=0):
        x, y, w, h = xyxy_to_xywh(bbox_xyxy)
        meas = np.array([[x], [y], [w], [h]], dtype=np.float32)
        self.kf.correct(meas)

        self.hits += 1
        self.time_since_update = 0
        self.last_meas_from_detector = meas_from_detector
        self._last_pred = None

        if frame_idx > 0:
            self._last_meas_frame = frame_idx

    def _state_to_bbox(self, state_vec):
        x = float(state_vec[0])
        y = float(state_vec[1])
        w = float(state_vec[2])
        h = float(state_vec[3])
        return xywh_to_xyxy(x, y, max(1.0, w), max(1.0, h))

    def get_pred_bbox_xyxy(self):
        if self._last_pred is None:
            return self._state_to_bbox(self.kf.statePost)
        return self._state_to_bbox(self._last_pred)

    def get_current_bbox_xyxy(self):
        if self.time_since_update == 0:
            return self._state_to_bbox(self.kf.statePost)
        return self.get_pred_bbox_xyxy()

    def update_embedding(self, emb, ema_alpha=0.85):
        if emb is None:
            return
        if self.emb_ema is None:
            self.emb_ema = emb.astype(np.float32)
        else:
            self.emb_ema = (ema_alpha * self.emb_ema + (1.0 - ema_alpha) * emb).astype(np.float32)
            n = np.linalg.norm(self.emb_ema)
            if n > 1e-9:
                self.emb_ema = self.emb_ema / n

    def update_expression(self, probs, frame_idx, ema_alpha=0.80):
        if probs is None:
            return
        probs = probs.astype(np.float32)
        if self.expr_probs_ema is None:
            self.expr_probs_ema = probs
        else:
            self.expr_probs_ema = (ema_alpha * self.expr_probs_ema + (1.0 - ema_alpha) * probs).astype(np.float32)

        k = int(np.argmax(self.expr_probs_ema))
        self.expr_label = FERPLUS_LABELS[k]
        self.expr_prob = float(self.expr_probs_ema[k])
        self.expr_last_frame = int(frame_idx)

    def update_pose(self, pose_dict, frame_idx):
        if not pose_dict:
            return
        self.yaw = pose_dict.get("yaw")
        self.pitch = pose_dict.get("pitch")
        self.roll = pose_dict.get("roll")
        self.gaze = pose_dict.get("gaze")
        self.pose_last_frame = int(frame_idx)


# ----------------- KALMAN MULTI-TRACKER (per camera) -----------------
class KalmanMultiTracker:
    def __init__(self, max_age=18, iou_threshold=0.2, appearance_weight=0.35, min_app_sim=0.25):
        self.next_id = 0
        self.tracks = OrderedDict()  # id -> KalmanBoxTrack
        self.max_age = int(max_age)
        self.iou_threshold = float(iou_threshold)
        self.appearance_weight = float(appearance_weight)
        self.min_app_sim = float(min_app_sim)

    def _register(self, bbox_xyxy, frame_idx=0):
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = KalmanBoxTrack(tid, bbox_xyxy)
        self.tracks[tid]._last_meas_frame = frame_idx

    def _deregister(self, tid):
        if tid in self.tracks:
            del self.tracks[tid]

    def update(self, detections: list[Detection], frame_idx=0):
        track_ids = list(self.tracks.keys())

        for tid in track_ids:
            self.tracks[tid].predict()

        if len(self.tracks) == 0:
            for det in detections:
                self._register(det.bbox_xyxy, frame_idx=frame_idx)
            return {tid: self.tracks[tid].get_current_bbox_xyxy() for tid in self.tracks}, []

        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(len(detections)))
        matches = []

        pairs = []
        for tid in track_ids:
            pb = self.tracks[tid].get_pred_bbox_xyxy()
            t_emb = self.tracks[tid].emb_ema

            for j, det in enumerate(detections):
                iou = iou_xyxy(pb, det.bbox_xyxy)
                if iou <= 0.0:
                    continue
                score = iou

                if t_emb is not None and det.emb is not None:
                    sim = cosine_sim(t_emb, det.emb)
                    if sim >= self.min_app_sim:
                        score = (1.0 - self.appearance_weight) * iou + self.appearance_weight * sim

                pairs.append((score, iou, tid, j))

        pairs.sort(reverse=True, key=lambda x: x[0])

        for score, iou, tid, j in pairs:
            if tid not in unmatched_tracks or j not in unmatched_dets:
                continue
            if iou < self.iou_threshold:
                continue
            matches.append((tid, j))
            unmatched_tracks.remove(tid)
            unmatched_dets.remove(j)

        for tid, j in matches:
            det = detections[j]
            self.tracks[tid].update(det.bbox_xyxy, meas_from_detector=det.from_detector, frame_idx=frame_idx)
            if det.emb is not None:
                self.tracks[tid].update_embedding(det.emb)

        to_delete = []
        for tid in unmatched_tracks:
            if self.tracks[tid].time_since_update > self.max_age:
                to_delete.append(tid)
        for tid in to_delete:
            self._deregister(tid)

        for j in unmatched_dets:
            self._register(detections[j].bbox_xyxy, frame_idx=frame_idx)

        return {tid: self.tracks[tid].get_current_bbox_xyxy() for tid in self.tracks}, matches


# ----------------- DEEPSORT WRAPPER (per camera) -----------------
class DeepSortTrackerWrapper:
    """
    Tracks face boxes using DeepSORT (modern tracking).
    We still use YuNet to produce face detections; DeepSORT handles association + motion.
    """
    def __init__(self, max_age=30):
        if DeepSort is None:
            raise RuntimeError("deep_sort_realtime not installed. pip install deep_sort_realtime")
        self.ds = DeepSort(max_age=max_age)

        # emulate KalmanMultiTracker API:
        self.tracks = OrderedDict()  # tid -> KalmanBoxTrack-ish state (we will store minimal)
        self._last_seen = {}

    def update(self, detections: list[Detection], frame, frame_idx=0):
        ds_in = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            ds_in.append(([x1, y1, x2 - x1, y2 - y1], det.conf, "face"))

        tracks = self.ds.update_tracks(ds_in, frame=frame)  # deep_sort_realtime supports passing frame for better embeddings; optional

        out = {}
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = int(t.track_id)
            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            out[tid] = (x1, y1, x2, y2)

            # Maintain a tiny track record similar to KalmanBoxTrack
            if tid not in self.tracks:
                kb = KalmanBoxTrack(tid, (x1, y1, x2, y2))
                self.tracks[tid] = kb
            else:
                kb = self.tracks[tid]
                kb.update((x1, y1, x2, y2), meas_from_detector=True, frame_idx=frame_idx)

        # clean old
        # (DeepSORT handles internal cleanup; we keep ours small)
        return out, []


# ----------------- MULTI-CAMERA REID MANAGER -----------------
class GlobalReID:
    """
    Identity fusion across cameras:
      - each local track has an embedding EMA
      - we assign / merge into global IDs by cosine similarity
    """
    def __init__(self, sim_thresh=0.55, ttl_frames=120):
        self.sim_thresh = float(sim_thresh)
        self.ttl = int(ttl_frames)
        self.next_gid = 1
        self.global_db = {}  # gid -> {"emb": vec, "last_frame": int}
        self.map_local_to_global = {}  # (cam_id, local_id) -> gid

    def _new_gid(self, emb, frame_idx):
        gid = self.next_gid
        self.next_gid += 1
        self.global_db[gid] = {"emb": emb.copy(), "last_frame": frame_idx}
        return gid

    def _update_gid(self, gid, emb, frame_idx, ema=0.90):
        ref = self.global_db[gid]["emb"]
        ref = (ema * ref + (1.0 - ema) * emb).astype(np.float32)
        n = np.linalg.norm(ref)
        if n > 1e-9:
            ref = ref / n
        self.global_db[gid]["emb"] = ref
        self.global_db[gid]["last_frame"] = frame_idx

    def cleanup(self, frame_idx):
        dead = [gid for gid, v in self.global_db.items() if (frame_idx - v["last_frame"]) > self.ttl]
        for gid in dead:
            del self.global_db[gid]
        # prune mappings that point to dead gids
        dead_set = set(dead)
        if dead_set:
            self.map_local_to_global = {k: v for k, v in self.map_local_to_global.items() if v not in dead_set}

    def assign(self, cam_id, local_id, emb, frame_idx):
        if emb is None:
            return None

        key = (cam_id, local_id)
        if key in self.map_local_to_global and self.map_local_to_global[key] in self.global_db:
            gid = self.map_local_to_global[key]
            self._update_gid(gid, emb, frame_idx)
            return gid

        best_gid = None
        best_sim = -1.0
        for gid, v in self.global_db.items():
            s = cosine_sim(emb, v["emb"])
            if s > best_sim:
                best_sim = s
                best_gid = gid

        if best_gid is not None and best_sim >= self.sim_thresh:
            self.map_local_to_global[key] = best_gid
            self._update_gid(best_gid, emb, frame_idx)
            return best_gid

        gid = self._new_gid(emb, frame_idx)
        self.map_local_to_global[key] = gid
        return gid


# ----------------- PROFILE PANEL -----------------
def render_profile_card(face_bgr, lines, width=420, height=260):
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    if face_bgr is not None and face_bgr.size > 0:
        face = cv2.resize(face_bgr, (160, 160), interpolation=cv2.INTER_AREA)
        panel[20:180, 20:180] = face
        cv2.rectangle(panel, (20, 20), (180, 180), (0, 255, 255), 2)

    y = 30
    x = 200
    for i, t in enumerate(lines):
        cv2.putText(panel, t, (x, y + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    return panel


def select_primary_track(trackers_by_cam, tracks_out_by_cam):
    best = None  # (cam_id, tid)
    best_score = -1.0

    for cam_id, tracks_out in tracks_out_by_cam.items():
        trk_mgr = trackers_by_cam[cam_id]
        for tid, bbox in tracks_out.items():
            trk = trk_mgr.tracks.get(tid)
            if trk is None:
                continue

            x1, y1, x2, y2 = bbox
            area = max(0, x2 - x1) * max(0, y2 - y1)
            measured_now = (trk.time_since_update == 0) and trk.last_meas_from_detector
            score = area + (1e9 if measured_now else 0.0)
            if score > best_score:
                best_score = score
                best = (cam_id, tid)

    return best


# ----------------- ARGPARSE -----------------
def build_arg_parser():
    p = argparse.ArgumentParser(
        description="AVIS: YOLO person → head ROI → YuNet faces → tracking (Kalman or DeepSORT) + expression + pose/gaze + multi-cam ReID."
    )

    # input
    p.add_argument("--video", type=str, default=None, help="Path to input video file (single stream).")
    p.add_argument("--camera", action="append", type=int, default=[], help="Camera index. You can pass multiple: --camera 0 --camera 1")
    p.add_argument("--target-width", type=int, default=960, help="Resize frames so width <= this value.")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold (person detections).")

    # tracking
    p.add_argument("--tracker", type=str, default="kalman", choices=["kalman", "deepsort"], help="Tracking backend.")
    p.add_argument("--max-age", type=int, default=18, help="Max frames to keep a track without updates.")
    p.add_argument("--iou-thresh", type=float, default=0.20, help="IoU threshold for Kalman matching.")

    # detector cadence
    p.add_argument("--detect-every", type=int, default=1, help="Run YuNet every N frames.")
    p.add_argument("--yunet-model", type=str, default="models/yunet/face_detection_yunet_2023mar.onnx", help="YuNet ONNX path.")
    p.add_argument("--yunet-score", type=float, default=0.60, help="YuNet score threshold.")
    p.add_argument("--yunet-nms", type=float, default=0.30, help="YuNet NMS threshold.")

    # collection
    p.add_argument("--collect-label", type=str, default="keanu", help="Label folder for saved faces.")
    p.add_argument("--no-collect", action="store_true", help="Disable saving face crops.")
    p.add_argument("--save-every", type=int, default=2, help="Save a face crop every N frames (MEAS only).")

    # anonymize
    p.add_argument("--anonymize", type=str, default="none", choices=["none", "all", "unknown"], help="Anonymize mode.")
    p.add_argument("--pixel-blocks", type=int, default=14, help="Pixelation strength.")

    # expression
    p.add_argument("--expression", action="store_true", help="Enable FER+ expression + profile overlay.")
    p.add_argument("--emotion-model", type=str, default="models/emotion/emotion-ferplus-8.onnx", help="FER+ ONNX path.")
    p.add_argument("--emotion-every", type=int, default=2, help="Run emotion inference every N frames (MEAS only).")
    p.add_argument("--emotion-norm", type=str, default="-11", choices=["-11", "01"], help="FER+ input normalization.")

    # pose + gaze
    p.add_argument("--pose", action="store_true", help="Enable head pose (yaw/pitch/roll) (experimental).")
    p.add_argument("--gaze", action="store_true", help="Enable gaze (LEFT/CENTER/RIGHT) (experimental).")
    p.add_argument("--pose-every", type=int, default=3, help="Run pose/gaze every N frames (MEAS only).")

    # ReID / multi-cam fusion
    p.add_argument("--reid", action="store_true", help="Enable multi-camera global ID via ArcFace embeddings.")
    p.add_argument("--arcface-model", type=str, default=None, help="ArcFace ONNX path (required for --reid).")
    p.add_argument("--reid-every", type=int, default=3, help="Compute embeddings every N frames (MEAS only).")
    p.add_argument("--reid-thresh", type=float, default=0.55, help="Cosine similarity threshold for global ID match.")
    p.add_argument("--fusion", action="store_true", help="Show identity-fusion HUD (enterprise-ish multi-cam view).")

    # UI
    p.add_argument("--profile-key", type=str, default="p", help="Key to toggle profile window.")
    p.add_argument("--snapshot-key", type=str, default="s", help="Key to save snapshot + open profile.")
    p.add_argument("--snapshots-dir", type=str, default="snapshots", help="Snapshot output directory.")

    # logs
    p.add_argument("--log-interval", type=int, default=30, help="Log every N frames (0 disables).")

    return p


# ----------------- MAIN -----------------
def main():
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if not args.camera and not args.video:
        args.camera = [0]  # default single cam

    if args.tracker == "deepsort" and DeepSort is None:
        raise RuntimeError("DeepSORT requested but deep_sort_realtime is not installed. pip install deep_sort_realtime")

    # models
    yolo = YOLO("yolov8n.pt")
    yunet = create_yunet(args.yunet_model, score_thresh=args.yunet_score, nms_thresh=args.yunet_nms, top_k=5000)

    # expression model
    emotion = None
    if args.expression:
        if ort is None:
            raise RuntimeError("Expression enabled but onnxruntime not installed. pip install onnxruntime")
        emotion = FerplusEmotion(args.emotion_model, norm=args.emotion_norm)

    # pose/gaze model
    pose_gaze = None
    if args.pose or args.gaze:
        if mp is None:
            raise RuntimeError("pose/gaze enabled but mediapipe not installed. pip install mediapipe")
        pose_gaze = FaceMeshPoseGaze()

    # ReID embedder + global fusion
    embedder = None
    global_reid = None
    if args.reid:
        if not args.arcface_model:
            raise RuntimeError("--reid requires --arcface-model path/to/arcface.onnx")
        embedder = ArcFaceEmbedder(args.arcface_model, input_size=112)
        global_reid = GlobalReID(sim_thresh=args.reid_thresh, ttl_frames=180)

    # collection
    collect_faces = not args.no_collect
    save_dir = None
    saved_count = 0
    if collect_faces:
        save_dir = ensure_dir(os.path.join("dataset", args.collect_label))
        logging.info(f"[COLLECT] enabled -> {save_dir}")

    snapshots_dir = ensure_dir(args.snapshots_dir)

    # capture(s)
    caps = {}
    if args.camera:
        for cam_id, cam_idx in enumerate(args.camera):
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera index {cam_idx}")
            caps[cam_id] = cap
        logging.info(f"[INPUT] cameras: {args.camera}")
    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {args.video}")
        caps[0] = cap
        logging.info(f"[INPUT] video: {args.video}")

    # per-camera trackers
    trackers = {}
    for cam_id in caps.keys():
        if args.tracker == "kalman":
            trackers[cam_id] = KalmanMultiTracker(max_age=args.max_age, iou_threshold=args.iou_thresh)
        else:
            trackers[cam_id] = DeepSortTrackerWrapper(max_age=max(30, args.max_age))

    # profile UI state
    profile_open = False
    profile_pick = None  # (cam_id, tid)

    profile_key = (args.profile_key or "p").lower()
    snapshot_key = (args.snapshot_key or "s").lower()

    frame_idx = 0
    last_time = time.time()
    fps = 0.0

    while True:
        frames = {}
        # read all cameras
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                frames[cam_id] = None
            else:
                frames[cam_id] = frame

        # if single video ended
        if len(frames) == 1 and list(frames.values())[0] is None:
            logging.info("End of stream.")
            break

        frame_idx += 1

        # fps
        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 1e-6:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

        tracks_out_by_cam = {}

        # process each camera independently
        for cam_id, frame in frames.items():
            if frame is None:
                continue

            h, w = frame.shape[:2]
            if w > args.target_width:
                s = args.target_width / w
                frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            frames[cam_id] = frame
            H, W = frame.shape[:2]

            # YOLO persons
            yres = yolo(frame, conf=args.conf, classes=[0])[0]
            people = []
            for b in yres.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, W, H)
                ar = (x2 - x1) / max(1, (y2 - y1))
                if ar > 3.0:
                    continue
                if (y2 - y1) < 90 or (x2 - x1) < 90:
                    continue
                people.append((x1, y1, x2, y2))

            detections = []
            run_det = (args.detect_every > 0) and (frame_idx % args.detect_every == 0)
            if run_det:
                for (px1, py1, px2, py2) in people:
                    box_h = py2 - py1
                    head_y2 = py1 + int(box_h * 0.75)
                    hx1, hy1, hx2, hy2 = clamp_bbox(px1, py1, px2, head_y2, W, H)
                    hx1, hy1, hx2, hy2 = expand_bbox(hx1, hy1, hx2, hy2, pad=12, W=W, H=H)

                    head_roi = frame[hy1:hy2, hx1:hx2]
                    rh, rw = head_roi.shape[:2]
                    if rh < 16 or rw < 16:
                        continue

                    yunet.setInputSize((rw, rh))
                    _, faces = yunet.detect(head_roi)
                    if faces is None or len(faces) == 0:
                        continue

                    best = max(faces, key=lambda f: float(f[2]) * float(f[3]))
                    fx, fy, fw, fh = best[:4]

                    fx1 = hx1 + int(fx)
                    fy1 = hy1 + int(fy)
                    fx2 = fx1 + int(fw)
                    fy2 = fy1 + int(fh)

                    pad = int(0.15 * max(float(fw), float(fh)))
                    fx1, fy1, fx2, fy2 = expand_bbox(fx1, fy1, fx2, fy2, pad, W, H)

                    conf = float(best[14]) if len(best) > 14 else 0.9  # best-effort

                    emb = None
                    if args.reid and embedder is not None and args.reid_every > 0 and (frame_idx % args.reid_every == 0):
                        crop = frame[fy1:fy2, fx1:fx2]
                        emb = embedder.embed(crop)

                    detections.append(Detection(bbox_xyxy=(fx1, fy1, fx2, fy2), conf=conf, from_detector=True, emb=emb, cam_id=cam_id))

            # bootstrap if no tracks and detector missed
            if len(detections) == 0 and len(trackers[cam_id].tracks) == 0:
                for (px1, py1, px2, py2) in people:
                    box_h = py2 - py1
                    face_h = int(box_h * 0.42)
                    face_w = int(face_h * 0.95)
                    cx = px1 + (px2 - px1) // 2
                    fx1 = cx - face_w // 2
                    fx2 = cx + face_w // 2
                    fy1 = py1 + int(box_h * 0.06)
                    fy2 = fy1 + face_h
                    fx1, fy1, fx2, fy2 = clamp_bbox(fx1, fy1, fx2, fy2, W, H)
                    detections.append(Detection(bbox_xyxy=(fx1, fy1, fx2, fy2), conf=0.2, from_detector=False, emb=None, cam_id=cam_id))
                    break

            tracks_out, _ = trackers[cam_id].update(detections, frame=frame, frame_idx=frame_idx)
            tracks_out_by_cam[cam_id] = tracks_out

            # expression + pose/gaze updates (MEAS only)
            for tid, bbox in tracks_out.items():
                trk = trackers[cam_id].tracks.get(tid)
                if trk is None:
                    continue

                measured_now = (trk.time_since_update == 0) and trk.last_meas_from_detector
                if not measured_now:
                    continue

                x1, y1, x2, y2 = clamp_bbox(*bbox, W, H)
                face_crop = frame[y1:y2, x1:x2]

                if args.expression and emotion is not None and args.emotion_every > 0 and (frame_idx % args.emotion_every == 0):
                    lbl, prob, probs = emotion.infer(face_crop)
                    if probs is not None:
                        trk.update_expression(probs, frame_idx)

                if (args.pose or args.gaze) and pose_gaze is not None and args.pose_every > 0 and (frame_idx % args.pose_every == 0):
                    pose = pose_gaze.estimate(face_crop)
                    if pose:
                        trk.update_pose(pose, frame_idx)

                # assign global ID (multi-cam)
                if args.reid and global_reid is not None and trk.emb_ema is not None:
                    gid = global_reid.assign(cam_id, tid, trk.emb_ema, frame_idx)
                    trk.global_id = gid

                # update embedding EMA from detection emb when available (Kalman tracker stores it)
                # (DeepSORT wrapper uses KalmanBoxTrack internally too)
                # We'll attach it here using the best detection overlap
                if args.reid and embedder is not None and (frame_idx % args.reid_every == 0):
                    # if we didn't embed earlier (e.g. DeepSORT / or det missing), embed now
                    if trk.emb_ema is None:
                        e = embedder.embed(face_crop)
                        if e is not None:
                            trk.update_embedding(e)

            # cleanup global DB
            if args.reid and global_reid is not None:
                global_reid.cleanup(frame_idx)

            # draw overlays
            meas_tracks = 0
            pred_tracks = 0

            for tid, bbox in tracks_out.items():
                x1, y1, x2, y2 = clamp_bbox(*bbox, W, H)
                trk = trackers[cam_id].tracks.get(tid)
                measured_now = (trk is not None) and (trk.time_since_update == 0) and trk.last_meas_from_detector

                if measured_now:
                    meas_tracks += 1
                else:
                    pred_tracks += 1

                status = "MEAS" if measured_now else "PRED"
                color = (0, 255, 0) if measured_now else (0, 255, 255)

                if args.anonymize == "all":
                    pixelate(frame, x1, y1, x2, y2, blocks=args.pixel_blocks)

                # save faces (MEAS only)
                if collect_faces and measured_now and (args.save_every > 0) and (frame_idx % args.save_every == 0):
                    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    out_path = os.path.join(save_dir, f"cam{cam_id}_img_{saved_count:05d}.jpg")
                    cv2.imwrite(out_path, gray)
                    saved_count += 1

                label = f"C{cam_id} ID{tid} {status}"
                if trk is not None and trk.global_id is not None:
                    label += f" | G{trk.global_id}"

                if trk is not None and args.expression and trk.expr_label is not None:
                    label += f" | {trk.expr_label}({trk.expr_prob:.2f})"

                if trk is not None and (args.pose or args.gaze) and trk.yaw is not None:
                    label += f" | yaw{trk.yaw:.0f} pitch{trk.pitch:.0f}"
                    if args.gaze and trk.gaze:
                        label += f" gaze:{trk.gaze}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            cv2.putText(
                frame,
                f"cam={cam_id} fps={fps:.1f} people={len(people)} tracks={len(tracks_out)} MEAS={meas_tracks} PRED={pred_tracks}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # multi-camera fusion HUD (identity-level)
        if args.fusion and args.reid and global_reid is not None:
            # draw on cam0 if available
            if 0 in frames and frames[0] is not None:
                overlay = frames[0]
                y = 45
                cv2.putText(overlay, "FUSION (Global IDs):", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                y += 24

                # show top few active gids
                items = sorted(global_reid.global_db.items(), key=lambda kv: kv[1]["last_frame"], reverse=True)[:6]
                for gid, v in items:
                    cv2.putText(overlay, f"G{gid} last_frame={v['last_frame']}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
                    y += 20

        # show windows
        for cam_id, frame in frames.items():
            if frame is None:
                continue
            cv2.imshow(f"AVIS Cam {cam_id}", frame)

        # profile window logic
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            ch = chr(key).lower()
            if ch == "q":
                break
            if ch == profile_key:
                profile_open = not profile_open
                if not profile_open:
                    profile_pick = None
                    try:
                        cv2.destroyWindow("AVIS Profile")
                    except Exception:
                        pass
            if ch == snapshot_key:
                profile_open = True

                # pick best track
                if tracks_out_by_cam:
                    profile_pick = select_primary_track(trackers, tracks_out_by_cam)

                if profile_pick is not None:
                    cam_id, tid = profile_pick
                    if cam_id in frames and frames[cam_id] is not None:
                        bbox = tracks_out_by_cam.get(cam_id, {}).get(tid)
                        if bbox:
                            H, W = frames[cam_id].shape[:2]
                            x1, y1, x2, y2 = clamp_bbox(*bbox, W, H)
                            crop = frames[cam_id][y1:y2, x1:x2]
                            out_path = os.path.join(snapshots_dir, f"cam{cam_id}_id{tid}_{now_stamp()}.jpg")
                            cv2.imwrite(out_path, crop)
                            logging.info(f"[SNAP] saved: {out_path}")

        if profile_open and tracks_out_by_cam:
            if profile_pick is None:
                profile_pick = select_primary_track(trackers, tracks_out_by_cam)

            if profile_pick is not None:
                cam_id, tid = profile_pick
                frame = frames.get(cam_id)
                bbox = tracks_out_by_cam.get(cam_id, {}).get(tid)
                trk = trackers[cam_id].tracks.get(tid)

                if frame is not None and bbox and trk is not None:
                    H, W = frame.shape[:2]
                    x1, y1, x2, y2 = clamp_bbox(*bbox, W, H)
                    crop = frame[y1:y2, x1:x2]

                    measured_now = (trk.time_since_update == 0) and trk.last_meas_from_detector
                    status = "MEAS" if measured_now else "PRED"

                    lines = [
                        f"Cam: {cam_id}",
                        f"Track ID: {tid}",
                        f"Status: {status}",
                        f"Hits: {trk.hits}",
                        f"Age: {trk.age}",
                    ]
                    if trk.global_id is not None:
                        lines.append(f"Global ID: G{trk.global_id}")

                    if args.expression:
                        if trk.expr_label is None:
                            lines.append("Emotion: (no data yet)")
                        else:
                            lines.append(f"Emotion: {trk.expr_label} ({trk.expr_prob:.2f})")

                    if args.pose or args.gaze:
                        if trk.yaw is None:
                            lines.append("Pose: (no data yet)")
                        else:
                            lines.append(f"Yaw: {trk.yaw:.0f} Pitch: {trk.pitch:.0f} Roll: {trk.roll:.0f}")
                            if args.gaze and trk.gaze:
                                lines.append(f"Gaze: {trk.gaze}")

                    panel = render_profile_card(crop, lines)
                    cv2.imshow("AVIS Profile", panel)

        if args.log_interval > 0 and (frame_idx % args.log_interval == 0):
            logging.info(f"Frame {frame_idx} | fps={fps:.1f}")

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    logging.info("Shutting down cleanly.")


if __name__ == "__main__":
    main()
