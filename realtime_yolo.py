import argparse
import csv
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional, Dict, Tuple, List

import cv2
import numpy as np
from ultralytics import YOLO

from avis.actions import Presence
from avis.camera import open_source
from avis.config import apply_config
from avis.events import EventLog
from avis.learning import FaceDB, IdentityModel
from avis.liveness import AntiSpoof

CHALLENGES = ["peace", "thumbs_up", "open", "fist", "point"]   # most reliable gestures to read

# Optional features (--expression, --pose/--gaze, --tracker deepsort) import their
# libraries only when turned on. See requirements-optional.txt.


# ----------------- UTIL -----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path):
    """Resolve relative model paths against the script folder, so it runs from any working directory."""
    if path is None or os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(SCRIPT_DIR, path)
    return candidate if os.path.exists(candidate) else path



def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


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


def softmax(x):
    x = x.astype(np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    if s <= 1e-9:
        return np.ones_like(x) / float(len(x))
    return ex / s


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


# ----------------- MODELS -----------------
def create_yunet(model_path, score_thresh=0.6, nms_thresh=0.3, top_k=5000):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YuNet model not found at: {model_path}")

    if hasattr(cv2, "FaceDetectorYN_create"):
        return cv2.FaceDetectorYN_create(model_path, "", (320, 320), score_thresh, nms_thresh, top_k)

    if hasattr(cv2, "FaceDetectorYN") and hasattr(cv2.FaceDetectorYN, "create"):
        return cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_thresh, nms_thresh, top_k)

    raise RuntimeError("Your OpenCV build does not expose FaceDetectorYN / YuNet.")


class ArcFaceEmbedder:
    """
    ArcFace ONNX via OpenCV DNN.
    Input: BGR face crop
    Output: L2-normalized embedding
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


def create_sface(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SFace model not found at: {model_path}")
    if hasattr(cv2, "FaceRecognizerSF_create"):
        return cv2.FaceRecognizerSF_create(model_path, "")
    return cv2.FaceRecognizerSF.create(model_path, "")


def open_recorder(path, fps, size):
    """MP4 writer. Prefers H.264 on Windows (plays on phones/Snapchat), falls back to mp4v."""
    ensure_dir(os.path.dirname(path) or ".")
    attempts = [(cv2.CAP_MSMF, "H264")] if sys.platform == "win32" else []
    attempts.append((cv2.CAP_ANY, "mp4v"))
    for api, cc in attempts:
        w = cv2.VideoWriter(path, api, cv2.VideoWriter_fourcc(*cc), float(fps), size)
        if w.isOpened():
            return w, cc
        w.release()
    return None, None


def next_trial_path(folder):
    """recordings/AVIS Trial 1.mp4, AVIS Trial 2.mp4, ... (never overwrites)."""
    ensure_dir(folder)
    nums = [int(m.group(1)) for f in os.listdir(folder)
            for m in [re.match(r"AVIS Trial (\d+)\.mp4$", f)] if m]
    return os.path.join(folder, f"AVIS Trial {max(nums, default=0) + 1}.mp4")


def sanitize_name(name):
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_")
    return name[:40]



FERPLUS_LABELS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]


class FerplusEmotion:
    """
    FER+ ONNXRuntime runner (auto layout).
    """
    def __init__(self, model_path, norm="-11"):
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError("--expression needs onnxruntime: pip install -r requirements-optional.txt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Emotion model not found at: {model_path}")

        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.in_name = inp.name
        self.in_shape = inp.shape
        self.norm = norm

        self.layout = "NCHW"
        if len(self.in_shape) == 4:
            if self.in_shape[-1] == 1 and (self.in_shape[1] in (64, None)):
                self.layout = "NHWC"
            if self.in_shape[1] == 1:
                self.layout = "NCHW"

    def _prep(self, face_bgr):
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        x = gray.astype(np.float32) / 255.0
        if self.norm == "-11":
            x = (x - 0.5) / 0.5
        if self.layout == "NCHW":
            x = x[None, None, :, :]
        else:
            x = x[None, :, :, None]
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
            return None, 0.0, None
        probs = softmax(scores[:8])
        k = int(np.argmax(probs))
        return FERPLUS_LABELS[k], float(probs[k]), probs


class FaceMeshPoseGaze:
    """
    MediaPipe FaceMesh pose + rough gaze.
    """
    def __init__(self):
        try:
            import mediapipe as mp  # pyright: ignore[reportMissingImports]
        except ImportError:
            raise RuntimeError("--pose/--gaze needs mediapipe: pip install mediapipe")

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.IDX_NOSE_TIP = 1
        self.IDX_CHIN = 152
        self.IDX_L_EYE = 33
        self.IDX_R_EYE = 263
        self.IDX_L_MOUTH = 61
        self.IDX_R_MOUTH = 291

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

        image_points = np.array([
            self._lm_xy(lms[self.IDX_NOSE_TIP], w, h),
            self._lm_xy(lms[self.IDX_CHIN], w, h),
            self._lm_xy(lms[self.IDX_L_EYE], w, h),
            self._lm_xy(lms[self.IDX_R_EYE], w, h),
            self._lm_xy(lms[self.IDX_L_MOUTH], w, h),
            self._lm_xy(lms[self.IDX_R_MOUTH], w, h),
        ], dtype=np.float32)

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ], dtype=np.float32)

        focal = w
        center = (w / 2.0, h / 2.0)
        cam_mtx = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype=np.float32)
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

        pitch = float(pitch * 180.0 / np.pi)
        yaw = float(yaw * 180.0 / np.pi)
        roll = float(roll * 180.0 / np.pi)

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
            denom = np.linalg.norm(b - a) + 1e-6
            return float(np.dot(iris - a, b - a) / (denom * denom))

        gaze_l = gaze_1d(li, l_out, l_in)
        gaze_r = gaze_1d(ri, r_in, r_out)
        gaze_x = (gaze_l + gaze_r) * 0.5

        if gaze_x < 0.42:
            gaze_txt = "LEFT"
        elif gaze_x > 0.58:
            gaze_txt = "RIGHT"
        else:
            gaze_txt = "CENTER"

        return {"yaw": yaw, "pitch": pitch, "roll": roll, "gaze": gaze_txt}


# ----------------- DATA -----------------
@dataclass
class Detection:
    bbox_xyxy: Tuple[int, int, int, int]
    conf: float
    from_detector: bool
    emb: Optional[np.ndarray] = None
    cam_id: int = 0
    face_emb: Optional[np.ndarray] = None
    aligned: Optional[np.ndarray] = None
    face_box: Optional[Tuple[int, int, int, int]] = None
    quality_ok: bool = True


# ----------------- TRACK STATE (shared fields) -----------------
class TrackState:
    """
    Lightweight state we keep for both Kalman and DeepSORT so CSV/UI is consistent.
    """
    def __init__(self, tid: int, init_bbox_xyxy: Tuple[int, int, int, int]):
        self.id = tid
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.last_meas_from_detector = True

        # ReID
        self.emb_ema: Optional[np.ndarray] = None
        self.global_id: Optional[int] = None

        # Emotion
        self.expr_probs_ema: Optional[np.ndarray] = None
        self.expr_label: Optional[str] = None
        self.expr_prob: float = 0.0
        self.expr_last_frame: int = 0

        # Pose/Gaze
        self.yaw = None
        self.pitch = None
        self.roll = None
        self.gaze = None
        self.pose_last_frame: int = 0

        # Identity (enrolled face recognition)
        self.name: Optional[str] = None
        self.name_sim: float = 0.0
        self.closest: Optional[str] = None

        # Liveness (spoof protection) + event counters
        self.live_ema: Optional[float] = None
        self.live: Optional[bool] = None      # None = still checking
        self.live_n = 0
        self.spoof_hits = 0
        self.unknown_hits = 0

        # Gesture verification + gesture password
        self.gesture_ok = False
        self.challenge: Optional[str] = None
        self.last_challenge: Optional[str] = None
        self.ch_t0 = 0.0
        self.ch_fails = 0
        self.pw_idx = 0
        self.pw_last: Optional[str] = None
        self.pw_fails = 0
        self.unlocked = False

        self._bbox = init_bbox_xyxy

    def set_bbox(self, bbox_xyxy, measured=True, from_detector=True):
        self._bbox = bbox_xyxy
        self.age += 1
        if measured:
            self.hits += 1
            self.time_since_update = 0
        else:
            self.time_since_update += 1
        self.last_meas_from_detector = bool(from_detector)

    def get_bbox(self):
        return self._bbox

    def update_live(self, score, thresh, alpha=0.8):
        if score is None:
            return
        self.live_n += 1
        self.live_ema = score if self.live_ema is None else alpha * self.live_ema + (1 - alpha) * score
        if self.live_n < 6:
            return                                   # need 6 looks (~0.6 s) before deciding
        if self.live is None:
            self.live = self.live_ema >= thresh
        elif self.live and self.live_ema < thresh - 0.15:  # a confirmed-live face needs sustained evidence to flip
            self.live = False
        elif not self.live and self.live_ema >= thresh:
            self.live = True

    def update_embedding(self, emb, ema_alpha=0.85):
        if emb is None:
            return
        emb = emb.astype(np.float32)
        if self.emb_ema is None:
            self.emb_ema = emb
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


# ----------------- KALMAN TRACKER -----------------
class KalmanBoxTrack(TrackState):
    """
    State: [x, y, w, h, vx, vy, vw, vh]
    """
    def __init__(self, tid: int, init_bbox_xyxy: Tuple[int, int, int, int]):
        super().__init__(tid, init_bbox_xyxy)

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

        self._last_pred = None

    def predict(self):
        pred = self.kf.predict()
        self._last_pred = pred
        self.age += 1
        self.time_since_update += 1

    def correct(self, bbox_xyxy, meas_from_detector=True):
        x, y, w, h = xyxy_to_xywh(bbox_xyxy)
        meas = np.array([[x], [y], [w], [h]], dtype=np.float32)
        self.kf.correct(meas)

        self.hits += 1
        self.time_since_update = 0
        self.last_meas_from_detector = bool(meas_from_detector)
        self._last_pred = None

        self._bbox = bbox_xyxy

    def _state_to_bbox(self, state_vec):
        v = np.asarray(state_vec).reshape(-1)
        x, y, w, h = float(v[0]), float(v[1]), float(v[2]), float(v[3])
        return xywh_to_xyxy(x, y, max(1.0, w), max(1.0, h))

    def get_bbox(self):
        if self.time_since_update == 0:
            return self._bbox
        if self._last_pred is None:
            return self._bbox
        return self._state_to_bbox(self._last_pred)


class KalmanMultiTracker:
    def __init__(self, max_age=18, iou_threshold=0.2):
        self.next_id = 0
        self.tracks: "OrderedDict[int, KalmanBoxTrack]" = OrderedDict()
        self.max_age = int(max_age)
        self.iou_threshold = float(iou_threshold)

    def _register(self, bbox_xyxy):
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = KalmanBoxTrack(tid, bbox_xyxy)

    def _deregister(self, tid):
        if tid in self.tracks:
            del self.tracks[tid]

    def update(self, detections: List[Detection], frame_bgr=None, frame_idx=0):
        track_ids = list(self.tracks.keys())

        # predict all
        for tid in track_ids:
            self.tracks[tid].predict()

        # no tracks yet
        if len(self.tracks) == 0:
            for det in detections:
                self._register(det.bbox_xyxy)
                if det.emb is not None:
                    self.tracks[self.next_id - 1].update_embedding(det.emb)
            return {tid: self.tracks[tid].get_bbox() for tid in self.tracks}, {"matches": []}

        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(len(detections)))
        matches = []

        # greedy IoU matching
        pairs = []
        for tid in track_ids:
            pb = self.tracks[tid].get_bbox()
            for j, det in enumerate(detections):
                iou = iou_xyxy(pb, det.bbox_xyxy)
                if iou <= 0:
                    continue
                pairs.append((iou, tid, j))

        pairs.sort(reverse=True, key=lambda x: x[0])

        for iou, tid, j in pairs:
            if tid not in unmatched_tracks or j not in unmatched_dets:
                continue
            if iou < self.iou_threshold:
                continue
            matches.append((tid, j))
            unmatched_tracks.remove(tid)
            unmatched_dets.remove(j)

        for tid, j in matches:
            det = detections[j]
            self.tracks[tid].correct(det.bbox_xyxy, meas_from_detector=det.from_detector)
            if det.emb is not None:
                self.tracks[tid].update_embedding(det.emb)

        # delete old
        to_delete = [tid for tid in unmatched_tracks if self.tracks[tid].time_since_update > self.max_age]
        for tid in to_delete:
            self._deregister(tid)

        # register new dets
        for j in unmatched_dets:
            self._register(detections[j].bbox_xyxy)
            if detections[j].emb is not None:
                self.tracks[self.next_id - 1].update_embedding(detections[j].emb)

        return {tid: self.tracks[tid].get_bbox() for tid in self.tracks}, {"matches": matches}


# ----------------- DEEPSORT WRAPPER -----------------
class DeepSortWrapper:
    """
    DeepSORT wrapper that matches the rest of the app API.

    Key fixes:
    - n_init=1 so tracks confirm fast
    - draw boxes even if not confirmed
    - do NOT reference self.ds.max_age (DeepSort doesn't expose it)
    - accept frame or frame_bgr to avoid signature mismatch
    """
    def __init__(self, max_age=18, n_init=1):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError:
            raise RuntimeError("--tracker deepsort needs deep_sort_realtime: pip install -r requirements-optional.txt")

        self.max_age = int(max_age)
        self.ds = DeepSort(max_age=self.max_age, n_init=int(n_init))

        # mirror minimal per-track state so the rest of the pipeline keeps working
        self.tracks = OrderedDict()      # tid -> KalmanBoxTrack (used as a generic state container)
        self._last_seen = {}             # tid -> last frame_idx

    def update(self, detections, frame=None, frame_bgr=None, frame_idx=0):
        if frame_bgr is None:
            frame_bgr = frame

        ds_in = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            ds_in.append(([x1, y1, x2 - x1, y2 - y1], float(det.conf), "face"))

        ds_tracks = self.ds.update_tracks(ds_in, frame=frame_bgr)

        out = {}
        meta = {}

        for trk in ds_tracks:
            tid = int(trk.track_id)

            # always draw, even if not confirmed yet
            l, t, r, b = map(int, trk.to_ltrb())
            out[tid] = (l, t, r, b)

            tsu = int(getattr(trk, "time_since_update", 0))
            confirmed = bool(getattr(trk, "is_confirmed", lambda: True)())
            measured_now = (tsu == 0)

            meta[tid] = {
                "confirmed": confirmed,
                "time_since_update": tsu,
                "measured_now": measured_now,
            }

            # keep a generic state object so ReID / emotion / pose can attach to it
            if tid not in self.tracks:
                self.tracks[tid] = KalmanBoxTrack(tid, (l, t, r, b))

            kb = self.tracks[tid]
            kb.age += 1
            kb.time_since_update = tsu
            kb.last_meas_from_detector = measured_now
            if measured_now:
                kb.hits += 1

            # store latest bbox (we use DeepSORT's box, not the Kalman filter one)
            kb._ds_bbox = (l, t, r, b)

            self._last_seen[tid] = int(frame_idx)

        # prune our mirror state (DeepSORT prunes internally; this keeps our dict clean)
        dead = [tid for tid, last in self._last_seen.items() if (frame_idx - last) > (self.max_age + 5)]
        for tid in dead:
            self._last_seen.pop(tid, None)
            self.tracks.pop(tid, None)

        return out, meta


# ----------------- GLOBAL REID -----------------
class GlobalReID:
    def __init__(self, sim_thresh=0.55, ttl_frames=180):
        self.sim_thresh = float(sim_thresh)
        self.ttl = int(ttl_frames)
        self.next_gid = 1
        self.global_db: Dict[int, Dict] = {}  # gid -> {"emb": vec, "last_frame": int}
        self.map_local_to_global: Dict[Tuple[int, int], int] = {}  # (cam_id, local_id) -> gid

    def _new_gid(self, emb, frame_idx):
        gid = self.next_gid
        self.next_gid += 1
        self.global_db[gid] = {"emb": emb.copy(), "last_frame": int(frame_idx)}
        return gid

    def _update_gid(self, gid, emb, frame_idx, ema=0.90):
        ref = self.global_db[gid]["emb"]
        ref = (ema * ref + (1.0 - ema) * emb).astype(np.float32)
        n = np.linalg.norm(ref)
        if n > 1e-9:
            ref = ref / n
        self.global_db[gid]["emb"] = ref
        self.global_db[gid]["last_frame"] = int(frame_idx)

    def cleanup(self, frame_idx):
        dead = [gid for gid, v in self.global_db.items() if (frame_idx - v["last_frame"]) > self.ttl]
        for gid in dead:
            del self.global_db[gid]
        dead_set = set(dead)
        if dead_set:
            self.map_local_to_global = {k: v for k, v in self.map_local_to_global.items() if v not in dead_set}

    def assign(self, cam_id, local_id, emb, frame_idx):
        if emb is None:
            return None

        key = (int(cam_id), int(local_id))
        if key in self.map_local_to_global:
            gid = self.map_local_to_global[key]
            if gid in self.global_db:
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


# ----------------- CSV METRICS (MOST IMPORTANT) -----------------
class MetricsCSV:
    """
    Always-safe CSV writer:
    - never crashes the app if a field is missing
    - writes consistent rows across Kalman/DeepSORT and any optional modules
    """
    def __init__(self, path: str):
        ensure_dir(os.path.dirname(path) or ".")
        self.path = path
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow([
            "ts", "frame",
            "cam", "tracker",
            "local_id", "global_id",
            "x1", "y1", "x2", "y2",
            "measured_now", "from_detector",
            "people_count", "detections_count",
            "emotion", "emotion_conf",
            "yaw", "pitch", "roll", "gaze",
            "fps", "name"
        ])
        self.f.flush()

    def log_track(self, ts, frame_idx, cam_id, tracker_name, tid, gid, bbox,
                  measured_now, from_detector, people_count, det_count,
                  emotion_label, emotion_conf,
                  yaw, pitch, roll, gaze,
                  fps, name=None):
        try:
            x1, y1, x2, y2 = bbox
        except Exception:
            x1 = y1 = x2 = y2 = None

        self.w.writerow([
            ts, frame_idx,
            cam_id, tracker_name,
            tid, (gid if gid is not None else ""),
            x1, y1, x2, y2,
            int(bool(measured_now)), int(bool(from_detector)),
            int(people_count), int(det_count),
            (emotion_label if emotion_label else ""),
            (float(emotion_conf) if emotion_conf is not None else ""),
            (float(yaw) if yaw is not None else ""),
            (float(pitch) if pitch is not None else ""),
            (float(roll) if roll is not None else ""),
            (gaze if gaze else ""),
            (float(fps) if fps is not None else ""),
            (name if name else "")
        ])

    def flush(self):
        try:
            self.f.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass


class SessionLog:
    """
    Clean per-frame log for cam 0 -> runs/<session>.csv (analyze with analyze_runs.py).
    One row per frame, fixed columns, numbers only (names column is text),
    empty cell only where a value genuinely doesn't exist (no face -> no similarity).
    """
    COLUMNS = ["time_s", "frame", "frame_ms", "fps_avg", "people", "faces", "tracks",
               "tracks_measured", "tracks_predicted", "known", "unknown",
               "best_similarity", "match_thresh", "names", "spoof", "verified"]

    def __init__(self, path, match_thresh):
        ensure_dir(os.path.dirname(path) or ".")
        self.path = path
        self.thresh = float(match_thresh)
        self.t0 = None
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow(self.COLUMNS)
        self.f.flush()

    def log(self, ts, frame_idx, frame_ms, fps, people, faces, tracks, meas, pred, names, sims, verified=0):
        if self.t0 is None:
            self.t0 = ts
        known = sum(1 for n in names if n not in ("Unknown", "Spoof"))
        spoof = sum(1 for n in names if n == "Spoof")
        self.w.writerow([
            round(ts - self.t0, 3), int(frame_idx), round(float(frame_ms), 2), round(float(fps), 2),
            int(people), int(faces), int(tracks), int(meas), int(pred),
            int(known), int(sum(1 for n in names if n == "Unknown")),
            (round(max(sims), 4) if sims else ""), self.thresh,
            ";".join(sorted(n for n in names if n not in ("Unknown", "Spoof"))),
            int(spoof), int(verified),
        ])
        if frame_idx % 30 == 0:
            self.f.flush()

    def close(self):
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass


# ----------------- CLI -----------------
def build_arg_parser():
    p = argparse.ArgumentParser(
        description="AVIS: YOLO person -> head ROI -> YuNet faces -> Kalman/DeepSORT + Emotion + Pose/Gaze + ReID + Fusion + CSV"
    )

    p.add_argument("--config", type=str, default="config.yaml", help="Settings file (command-line flags override it).")

    # input
    p.add_argument("--source", type=str, default=None,
                   help="Webcam index (0), IP camera URL (rtsp://user:pass@ip:554/stream), or video file.")
    p.add_argument("--device", type=str, default="auto", help="auto / cuda / cpu for YOLO.")
    p.add_argument("--yolo-every", type=int, default=1, help="Run person detection every N frames (reuse boxes between).")
    p.add_argument("--video", type=str, default=None, help="Path to input video file (single stream).")
    p.add_argument("--camera", action="append", type=int, default=[], help="Camera index (repeatable). Example: --camera 0 --camera 1")
    p.add_argument("--target-width", type=int, default=960, help="Resize frames so width <= this value.")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold for person detections.")

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
    p.add_argument("--collect", action="store_true", help="Opt-in: save face crops to dataset/<label>/.")
    p.add_argument("--no-collect", action="store_true", help="(Default) Do not save face crops. Kept for old commands.")
    p.add_argument("--save-every", type=int, default=2, help="Save a face crop every N frames (MEAS only).")

    # anonymize
    p.add_argument("--anonymize", type=str, default="none", choices=["none", "all", "unknown"],
                   help="Pixelate faces: none, all, or unknown (everyone except recognized enrolled people).")
    p.add_argument("--pixel-blocks", type=int, default=14, help="Pixelation strength.")

    # expression
    p.add_argument("--expression", action="store_true", help="Enable FER+ emotion inference.")
    p.add_argument("--emotion-model", type=str, default="models/emotion/emotion-ferplus-8.onnx", help="FER+ ONNX path.")
    p.add_argument("--emotion-every", type=int, default=2, help="Run emotion inference every N frames (MEAS only).")
    p.add_argument("--emotion-norm", type=str, default="-11", choices=["-11", "01"], help="FER+ input normalization.")

    # pose + gaze
    p.add_argument("--pose", action="store_true", help="Enable head pose estimation (experimental).")
    p.add_argument("--gaze", action="store_true", help="Enable gaze estimation (experimental).")
    p.add_argument("--pose-every", type=int, default=3, help="Run pose/gaze every N frames (MEAS only).")

    # ReID / multi-cam fusion
    p.add_argument("--reid", action="store_true", help="Enable ArcFace embeddings + global IDs.")
    p.add_argument("--arcface-model", type=str, default=None, help="ArcFace ONNX path (required for --reid).")
    p.add_argument("--reid-every", type=int, default=4, help="Compute embeddings every N frames (MEAS only).")
    p.add_argument("--reid-thresh", type=float, default=0.55, help="Cosine similarity threshold for global ID match.")
    p.add_argument("--fusion", action="store_true", help="Show fusion HUD on cam0.")

    # identity: enroll + recognize
    p.add_argument("--enroll", type=str, default=None, help="Enroll this person's face (e.g. --enroll KJ). Only one face should be in view.")
    p.add_argument("--enroll-samples", type=int, default=20, help="Face samples to capture when enrolling.")
    p.add_argument("--sface-model", type=str, default="models/sface/face_recognition_sface_2021dec.onnx", help="SFace ONNX path.")
    p.add_argument("--faces-db", type=str, default="faces_db", help="Folder where enrolled faces are stored.")
    p.add_argument("--match-thresh", type=float, default=0.363, help="Cosine similarity needed to call a face a known person.")
    p.add_argument("--recognize-every", type=int, default=3, help="Run recognition every N frames.")
    p.add_argument("--no-recognize", action="store_true", help="Disable face recognition.")
    p.add_argument("--min-face-score", type=float, default=0.8, help="Face detector confidence needed to judge identity/liveness.")
    p.add_argument("--min-face-px", type=int, default=56, help="Smallest face (pixels) to judge identity/liveness.")

    # spoof protection
    BO = argparse.BooleanOptionalAction
    p.add_argument("--verify", type=str, default="gesture", choices=["gesture", "model", "none"],
                   help="How a recognized face proves it's live: gesture challenge, the ML spoof model, or none.")
    p.add_argument("--challenge-timeout-s", type=float, default=10, help="Seconds to answer a gesture challenge.")
    p.add_argument("--challenge-fails", type=int, default=3, help="Missed challenges before a 'check failed' alert.")
    p.add_argument("--verify-grace-s", type=float, default=60, help="Stay verified this long after looking away.")
    p.add_argument("--gesture-password", type=str, default="", help="Secret gesture sequence, e.g. 'peace,fist,thumbs_up'.")
    p.add_argument("--on-unlock-command", type=str, default="", help="Command to run when the gesture password is entered.")
    p.add_argument("--liveness", action=BO, default=True, help="Spoof protection (reject photos/screens).")
    p.add_argument("--liveness-model", type=str, default="models/antispoof/AntiSpoofing_bin_1.5_128.onnx")
    p.add_argument("--liveness-thresh", type=float, default=0.45, help="Real-face probability needed to count as live.")
    p.add_argument("--liveness-min-face-px", type=int, default=90,
                   help="Only run the spoof check on faces at least this wide (far faces are too blurry).")
    p.add_argument("--liveness-warmup-s", type=float, default=2.0, help="Skip spoof checks while the webcam adjusts exposure.")
    p.add_argument("--enroll-liveness", action=BO, default=False,
                   help="Also require the spoof check while enrolling (off: you supervise enrollment).")
    p.add_argument("--spoof-hits", type=int, default=5, help="Failed checks in a row before raising a spoof event.")

    # learning / ML
    p.add_argument("--learn", action=BO, default=True, help="Add confident, live sightings to the person's profile.")
    p.add_argument("--learn-min-sim", type=float, default=0.55, help="Similarity needed before a sighting is learned.")
    p.add_argument("--max-samples", type=int, default=300, help="Max samples per person profile.")
    p.add_argument("--identity-model", type=str, default="models/identity/identity_mlp.pt",
                   help="Trained identity model (train_identity.py). Used if it exists.")
    p.add_argument("--model-min-prob", type=float, default=0.6, help="Model confidence needed to confirm a match.")
    p.add_argument("--save-photos", action=BO, default=True,
                   help="Save face photos of recognized enrolled people, one folder per run (dataset/<name>/<date time run>/).")
    p.add_argument("--photo-every-s", type=float, default=1.0, help="At most one photo per person every N seconds.")
    p.add_argument("--max-photos-per-session", type=int, default=150, help="Cap per person per run.")
    p.add_argument("--auto-train", action=BO, default=True,
                   help="After each session, retrain the identity model on the GPU in the background.")

    # events + alerts
    p.add_argument("--events", action=BO, default=True, help="Write runs/events.csv + snapshots.")
    p.add_argument("--ntfy-topic", type=str, default="", help="ntfy topic or URL for phone alerts (empty = off).")
    p.add_argument("--discord-webhook", type=str, default="", help="Discord webhook URL for alerts (empty = off).")
    p.add_argument("--alert-cooldown-s", type=float, default=60, help="Min seconds between alerts of the same kind.")
    p.add_argument("--seen-cooldown-s", type=float, default=300, help="Min seconds between 'X recognized' log entries.")
    p.add_argument("--unknown-hits", type=int, default=3, help="Unknown checks in a row before an unknown-person event.")
    p.add_argument("--snapshot-days", type=int, default=7, help="Delete snapshots older than this (0 = keep).")

    # actions
    p.add_argument("--owner", type=str, default="", help="Your enrolled name (for auto-lock).")
    p.add_argument("--auto-lock", action=BO, default=False, help="Lock Windows when the owner is away.")
    p.add_argument("--lock-after-s", type=float, default=60, help="Seconds without seeing the owner before locking.")
    p.add_argument("--on-recognized-command", type=str, default="", help="Command to run when someone is recognized ({name}).")
    p.add_argument("--command-cooldown-s", type=float, default=300)

    # recording
    p.add_argument("--record", nargs="?", const="auto", default=None,
                   help="Save the video as MP4 (default: recordings/AVIS Trial N.mp4). Uses the clean overlay.")
    p.add_argument("--debug", action="store_true", help="Show debug overlay (tracker stats, IDs, MEAS/PRED, scores).")

    p.add_argument("--no-log", action="store_true", help="Don't write the per-frame session CSV in runs/.")

    # metrics
    p.add_argument("--eval-csv", type=str, default=None, help="Write per-track metrics rows to this CSV.")

    # logs
    p.add_argument("--log-interval", type=int, default=30, help="Log every N frames (0 disables).")

    return p


def start_auto_train(face_db, args):
    """Retrain the identity network in the background (own process) so AVIS closes instantly."""
    mp = resolve_path(args.identity_model) if args.identity_model else None
    changed = True    # new photos / learned samples every run -> retrain (takes seconds on the GPU)
    if not changed or mp is None:
        return
    unknown_dir = os.path.join(SCRIPT_DIR, "eval", "unknown")
    has_negatives = os.path.isdir(unknown_dir) and any(os.scandir(unknown_dir))
    if len(face_db.people) + int(has_negatives) < 2:
        logging.info("[ML] auto-train waiting: needs a 2nd enrolled person or an eval/unknown clip "
                     "(Run Task > AVIS: Record Test Clip > unknown).")
        return
    import subprocess
    log_path = os.path.join(SCRIPT_DIR, "runs", "identity_training.log")
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
    with open(log_path, "w", encoding="utf-8") as log:
        subprocess.Popen([sys.executable, os.path.join(SCRIPT_DIR, "train_identity.py"), "--out", mp],
                         cwd=SCRIPT_DIR, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    logging.info(f"[ML] retraining identity model on the GPU in the background -> {os.path.relpath(log_path, SCRIPT_DIR)}")


# ----------------- MAIN -----------------
def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = build_arg_parser()
    pre, _ = parser.parse_known_args()
    cfg_path = resolve_path(pre.config) if pre.config else None
    if cfg_path and not os.path.isabs(cfg_path):
        cfg_path = os.path.join(SCRIPT_DIR, cfg_path)
    if apply_config(parser, cfg_path):
        logging.info(f"[CONFIG] loaded {cfg_path}")
    args = parser.parse_args()

    if args.source is not None and str(args.source).strip() != "":
        sources = [args.source]
    elif args.video:
        sources = [args.video]
    else:
        sources = args.camera or [0]

    # compute device (GPU if available)
    device, half = "cpu", False
    try:
        import torch
        if args.device in ("auto", "cuda") and torch.cuda.is_available():
            device, half = 0, True
            logging.info(f"[GPU] YOLO on {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda}, half precision)")
        elif args.device == "cuda":
            logging.warning("[GPU] CUDA requested but not available; using CPU.")
    except ImportError:
        pass
    if device == "cpu":
        logging.info("[GPU] YOLO on CPU")

    # init models
    yolo = YOLO(resolve_path("yolov8n.pt"))
    yunet = create_yunet(resolve_path(args.yunet_model), score_thresh=args.yunet_score, nms_thresh=args.yunet_nms, top_k=5000)

    # identity
    sface = None
    face_db = None
    enroll_name = None
    enroll_embs: List[np.ndarray] = []
    if not args.no_recognize or args.enroll:
        face_db = FaceDB(args.faces_db if os.path.isabs(args.faces_db) else os.path.join(SCRIPT_DIR, args.faces_db),
                         max_samples=args.max_samples)
        sface = create_sface(resolve_path(args.sface_model))
        logging.info(f"[ID] known people: {sorted(face_db.people.keys()) or 'none yet'}")
    if args.verify == "model":
        args.liveness = True
    antispoof = None
    if sface is not None and args.liveness:
        try:
            antispoof = AntiSpoof(resolve_path(args.liveness_model))
        except (RuntimeError, FileNotFoundError) as e:
            logging.warning(f"[LIVE] spoof protection OFF: {e}")

    identity_model = None
    model_path = resolve_path(args.identity_model) if args.identity_model else None
    if sface is not None and model_path and os.path.exists(model_path):
        try:
            identity_model = IdentityModel(model_path)
            missing = sorted(set(face_db.people) - set(identity_model.classes))
            if missing:
                logging.warning(f"[ML] model doesn't know {missing}; retrain with train_identity.py. Ignoring model.")
                identity_model = None
        except Exception as e:
            logging.warning(f"[ML] could not load identity model: {e}")

    events = None
    if args.events:
        events = EventLog(os.path.join(SCRIPT_DIR, "runs"), ntfy_topic=args.ntfy_topic,
                          discord_webhook=args.discord_webhook, alert_cooldown_s=args.alert_cooldown_s,
                          snapshot_days=args.snapshot_days)
    presence = Presence(owner=args.owner, auto_lock=args.auto_lock, lock_after_s=args.lock_after_s,
                        command=args.on_recognized_command, command_cooldown_s=args.command_cooldown_s,
                        events=events)
    enroll_note = ""

    # hand tracking for gesture verification / password
    password = [g.strip().lower() for g in (args.gesture_password or "").split(",") if g.strip()]
    hand_tracker, hand_stable = None, {}
    if sface is not None and (args.verify == "gesture" or password):
        try:
            from avis.hands import HandTracker
            hand_tracker = HandTracker()
            logging.info(f"[VERIFY] gesture check on ({args.challenge_timeout_s:.0f}s to answer)"
                         + (f", gesture password of {len(password)} steps" if password else ""))
        except Exception as e:
            logging.warning(f"[VERIFY] hand tracking unavailable ({e}); gesture check OFF")
            if args.verify == "gesture":
                args.verify = "none"
    verified_until: Dict[str, float] = {}
    pstate: Dict[str, dict] = {}     # per PERSON (survives the face being re-detected): fails, password progress

    def person(name):
        st = pstate.setdefault(name, {"fails": 0, "fail_t": 0.0, "pw_idx": 0, "pw_last": None, "pw_fails": 0, "unlocked_until": 0.0})
        if time.time() - st["fail_t"] > 120:          # forget old misses after 2 minutes
            st["fails"] = 0
        return st

    def verified_ok(t):
        if t is None:
            return False
        if args.verify == "gesture":
            return bool(t.gesture_ok)
        if args.verify == "model":
            return antispoof is None or bool(t.live)
        return True

    if args.enroll:
        enroll_name = sanitize_name(args.enroll)
        if not enroll_name:
            raise RuntimeError("--enroll needs a name with letters or numbers.")
        logging.info(f"[ID] enrolling '{enroll_name}': look at the camera, turn your head a little.")

    emotion = None
    if args.expression:
        emotion = FerplusEmotion(resolve_path(args.emotion_model), norm=args.emotion_norm)

    pose_gaze = None
    if args.pose or args.gaze:
        pose_gaze = FaceMeshPoseGaze()

    embedder = None
    global_reid = None
    if args.reid:
        if not args.arcface_model:
            raise RuntimeError("--reid requires --arcface-model path/to/arcface.onnx")
        embedder = ArcFaceEmbedder(resolve_path(args.arcface_model), input_size=112)
        global_reid = GlobalReID(sim_thresh=args.reid_thresh, ttl_frames=180)

    # metrics
    metrics = MetricsCSV(args.eval_csv) if args.eval_csv else None
    if metrics:
        logging.info(f"[METRICS] writing -> {args.eval_csv}")

    # collection
    collect_faces = args.collect and not args.no_collect
    session = now_stamp()
    save_dir = None
    saved_count = 0
    if collect_faces:
        save_dir = ensure_dir(os.path.join("dataset", args.collect_label))
        logging.info(f"[COLLECT] enabled -> {save_dir}")

    # capture(s): webcam index, IP camera URL, or file
    caps: Dict[int, object] = {}
    for cam_id, src in enumerate(sources):
        caps[cam_id] = open_source(src)
        shown = re.sub(r"//[^@/]*@", "//***@", str(src))   # hide camera passwords in logs
        logging.info(f"[INPUT] cam{cam_id}: {shown}")

    # trackers per cam
    trackers: Dict[int, object] = {}
    for cam_id in caps.keys():
        if args.tracker == "kalman":
            trackers[cam_id] = KalmanMultiTracker(max_age=args.max_age, iou_threshold=args.iou_thresh)
        else:
            trackers[cam_id] = DeepSortWrapper(max_age=args.max_age, n_init=1)

    # recording (cam 0 only); opened after a few frames so the video plays at real speed
    recorder = None
    record_path = None
    if args.record in (True, "true", "True"):
        args.record = "auto"
    if args.record in (False, "false", "False", "", "none"):
        args.record = None
    if args.record:
        record_path = args.record if args.record != "auto" else next_trial_path(os.path.join(SCRIPT_DIR, "recordings"))
        logging.info(f"[REC] will record to {record_path}")

    clean_hud = not args.debug

    # one photo folder per run: dataset/<name>/<YYYY-MM-DD HH-MM-SS> <run name>/
    run_label = (os.path.splitext(os.path.basename(record_path))[0] if record_path
                 else ("Enrollment" if args.enroll else "Live session"))
    session_folder = f"{time.strftime('%Y-%m-%d %H-%M-%S')} {run_label}"
    photo_state: Dict[str, dict] = {}

    def save_face_photo(name, aligned, emb=None):
        """Save one aligned face crop to this run's folder; skips near-duplicates. Returns True if saved."""
        st = photo_state.setdefault(name, {"n": 0, "last_t": 0.0, "last_emb": None})
        if aligned is None or st["n"] >= args.max_photos_per_session:
            return False
        if emb is not None and st["last_emb"] is not None and float(st["last_emb"] @ emb) > 0.97:
            return False
        folder = ensure_dir(os.path.join(SCRIPT_DIR, "dataset", name, session_folder))
        st["n"] += 1
        cv2.imwrite(os.path.join(folder, f"{time.strftime('%H-%M-%S')}_{st['n']:03d}.jpg"), aligned)
        st["last_t"] = time.time()
        st["last_emb"] = emb
        return True

    session_log = None
    if not args.no_log:
        base = os.path.splitext(os.path.basename(record_path))[0] if record_path else time.strftime("Session %Y-%m-%d %H-%M-%S")
        session_log = SessionLog(os.path.join(SCRIPT_DIR, "runs", base + ".csv"), args.match_thresh)
        logging.info(f"[LOG] stats -> {session_log.path}")

    frame_idx = 0
    last_time = time.time()
    fps = 0.0
    last_people: Dict[int, list] = {}
    shown_windows: set = set()
    session_t0 = time.time()

    try:
        while True:
            frames: Dict[int, Optional[np.ndarray]] = {}

            for cam_id, cap in caps.items():
                ret, frame = cap.read()
                frames[cam_id] = frame if ret else None

            if all(f is None for f in frames.values()):
                logging.info("End of stream.")
                break

            frame_idx += 1

            now = time.time()
            dt = now - last_time
            frame_ms = max(dt * 1000.0, 0.01)
            last_time = now
            if dt > 1e-6:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            ts = time.time()

            # per-cam processing
            for cam_id, frame in frames.items():
                if frame is None:
                    continue

                h, w = frame.shape[:2]
                if w > args.target_width:
                    s = args.target_width / w
                    frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                frames[cam_id] = frame
                clean = frame.copy()
                H, W = frame.shape[:2]

                # YOLO persons (every yolo_every frames; reuse boxes in between)
                if args.yolo_every <= 1 or frame_idx % args.yolo_every == 0 or cam_id not in last_people:
                    yres = yolo(frame, conf=args.conf, classes=[0], verbose=False, device=device, half=half)[0]
                    yboxes = [b.xyxy[0].tolist() for b in yres.boxes]
                else:
                    yboxes = last_people[cam_id]
                last_people[cam_id] = yboxes
                people = []
                for xyxy in yboxes:
                    x1, y1, x2, y2 = clamp_bbox(*xyxy, W, H)

                    # basic filters so you don't spam tiny boxes
                    ar = (x2 - x1) / max(1, (y2 - y1))
                    if ar > 3.0:
                        continue
                    if (y2 - y1) < 90 or (x2 - x1) < 90:
                        continue
                    people.append((x1, y1, x2, y2))

                detections: List[Detection] = []
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

                        # choose biggest face in head ROI
                        best = max(faces, key=lambda f: float(f[2]) * float(f[3]))
                        fx, fy, fw, fh = best[:4]

                        fx1 = hx1 + int(fx)
                        fy1 = hy1 + int(fy)
                        fx2 = fx1 + int(fw)
                        fy2 = fy1 + int(fh)
                        raw_box = (fx1, fy1, fx2, fy2)

                        pad = int(0.15 * max(float(fw), float(fh)))
                        fx1, fy1, fx2, fy2 = expand_bbox(fx1, fy1, fx2, fy2, pad, W, H)

                        conf = float(best[14]) if len(best) > 14 else 0.9

                        # face quality gate: only judge identity / liveness on a clearly visible face
                        # (cap brim, looking down, turned away, too small -> "can't see face", not "spoof")
                        eye_dist = float(np.hypot(best[6] - best[4], best[7] - best[5]))
                        quality_ok = (conf >= args.min_face_score and min(fw, fh) >= args.min_face_px
                                      and eye_dist >= 0.28 * float(fw))

                        # face embedding for identity (enroll / recognize)
                        face_emb = None
                        aligned = None
                        want_id = sface is not None and (
                            enroll_name is not None
                            or (face_db.people and args.recognize_every > 0 and frame_idx % args.recognize_every == 0)
                        )
                        if want_id:
                            row = np.array(best[:15], dtype=np.float32).copy()
                            row[0] += hx1
                            row[1] += hy1
                            row[4:14:2] += hx1
                            row[5:14:2] += hy1
                            try:
                                aligned = sface.alignCrop(clean, row)
                                face_emb = sface.feature(aligned).reshape(-1).astype(np.float32)
                                n = np.linalg.norm(face_emb)
                                face_emb = face_emb / n if n > 1e-9 else None
                            except cv2.error:
                                face_emb = None

                        emb = None
                        if args.reid and embedder is not None and args.reid_every > 0 and (frame_idx % args.reid_every == 0):
                            crop = frame[fy1:fy2, fx1:fx2]
                            emb = embedder.embed(crop)

                        detections.append(Detection(
                            bbox_xyxy=(fx1, fy1, fx2, fy2),
                            conf=conf,
                            from_detector=True,
                            emb=emb,
                            cam_id=cam_id,
                            face_emb=face_emb,
                            aligned=aligned,
                            face_box=raw_box,
                            quality_ok=quality_ok,
                        ))

                # hands -> stable gestures (for gesture verification / password)
                hand_list = []
                if hand_tracker is not None and cam_id == 0:
                    from avis.hands import features as hand_features, classify as classify_gesture, StableGesture
                    seen_keys = set()
                    for i, (lab, hpts) in enumerate(hand_tracker(clean)):
                        hf = hand_features(hpts)
                        key = f"{lab}{i}"
                        seen_keys.add(key)
                        sg = hand_stable.setdefault(key, StableGesture(4))
                        hand_list.append((hpts, hf, sg.update(classify_gesture(hf, sg.state))))
                    for key in [k for k in hand_stable if k not in seen_keys]:
                        del hand_stable[key]          # hand left the frame: forget its last gesture

                # enrollment: needs exactly one face in view
                id_dets = [d for d in detections if d.face_emb is not None and d.quality_ok]
                if enroll_name is not None and cam_id == 0:
                    live_ok = True
                    if len(id_dets) == 1 and frame_idx % 3 == 0 and antispoof is not None and args.enroll_liveness:
                        ls = antispoof.score(clean, id_dets[0].face_box)
                        live_ok = ls is not None and ls >= args.liveness_thresh
                        enroll_note = "" if live_ok else "  (not a live face - photo/screen?)"
                    if len(id_dets) == 1 and frame_idx % 3 == 0 and live_ok:
                        enroll_embs.append(id_dets[0].face_emb)
                        photo_state.setdefault(enroll_name, {"n": 0, "last_t": 0.0, "last_emb": None})["last_emb"] = None
                        save_face_photo(enroll_name, id_dets[0].aligned)
                    if len(enroll_embs) >= args.enroll_samples:
                        face_db.add(enroll_name, enroll_embs)
                        logging.info(f"[ID] saved {len(enroll_embs)} samples for '{enroll_name}' -> {face_db.folder}")
                        enroll_name = None
                        enroll_embs = []

                # tracker update (UNIFIED API)
                tracks_out, _meta = trackers[cam_id].update(detections, frame_bgr=frame, frame_idx=frame_idx)

                # update per-track optional modules (MEAS only)
                meas_tracks = 0
                pred_tracks = 0
                frame_names: List[str] = []
                frame_sims: List[float] = []
                frame_verified = set()          # known people who passed the spoof check this frame

                for tid, bbox in tracks_out.items():
                    x1, y1, x2, y2 = clamp_bbox(*bbox, W, H)
                    trk_state = trackers[cam_id].tracks.get(tid)

                    measured_now = False
                    from_detector = False
                    if trk_state is not None:
                        measured_now = (trk_state.time_since_update == 0) and trk_state.last_meas_from_detector
                        from_detector = bool(trk_state.last_meas_from_detector)

                    if measured_now:
                        meas_tracks += 1
                    else:
                        pred_tracks += 1

                    # collection (MEAS only)
                    if collect_faces and measured_now and args.save_every > 0 and (frame_idx % args.save_every == 0):
                        gray = cv2.cvtColor(clean[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                        out_path = os.path.join(save_dir, f"{session}_cam{cam_id}_img_{saved_count:05d}.jpg")
                        cv2.imwrite(out_path, gray)
                        saved_count += 1

                    face_crop = clean[y1:y2, x1:x2]

                    # emotion update
                    if measured_now and args.expression and emotion is not None and args.emotion_every > 0 and (frame_idx % args.emotion_every == 0):
                        _lbl, _prob, probs = emotion.infer(face_crop)
                        if trk_state is not None and probs is not None:
                            trk_state.update_expression(probs, frame_idx)

                    # pose/gaze update
                    if measured_now and (args.pose or args.gaze) and pose_gaze is not None and args.pose_every > 0 and (frame_idx % args.pose_every == 0):
                        pose = pose_gaze.estimate(face_crop)
                        if trk_state is not None and pose:
                            trk_state.update_pose(pose, frame_idx)

                    # ensure embedding EMA exists (MEAS only)
                    if measured_now and args.reid and embedder is not None and args.reid_every > 0 and (frame_idx % args.reid_every == 0):
                        if trk_state is not None:
                            if trk_state.emb_ema is None:
                                e = embedder.embed(face_crop)
                                if e is not None:
                                    trk_state.update_embedding(e)

                    # assign global id
                    if measured_now and args.reid and global_reid is not None and trk_state is not None and trk_state.emb_ema is not None:
                        gid = global_reid.assign(cam_id, tid, trk_state.emb_ema, frame_idx)
                        trk_state.global_id = gid

                    # identity: match this track to the closest face detection, then look it up
                    if measured_now and trk_state is not None and face_db is not None and face_db.people and id_dets:
                        best_det = max(id_dets, key=lambda d: iou_xyxy((x1, y1, x2, y2), d.bbox_xyxy))
                        if iou_xyxy((x1, y1, x2, y2), best_det.bbox_xyxy) > 0.3:
                            who, sim, closest = face_db.match(best_det.face_emb, args.match_thresh)

                            # spoof protection
                            live_score = None
                            fb = best_det.face_box
                            big_enough = fb is not None and (fb[2] - fb[0]) >= args.liveness_min_face_px
                            warm = (time.time() - session_t0) >= args.liveness_warmup_s   # webcam exposure settling
                            if antispoof is not None and big_enough and warm:  # small/far faces are too blurry to judge
                                live_score = antispoof.score(clean, fb)
                                trk_state.update_live(live_score, args.liveness_thresh)

                            # ML model must agree with the similarity match
                            if who and identity_model is not None:
                                m_label, m_prob = identity_model.predict(best_det.face_emb)
                                if m_label != who or m_prob < args.model_min_prob:
                                    who = None

                            trk_state.name_sim = sim
                            trk_state.closest = closest
                            if antispoof is not None and trk_state.live is None and big_enough and warm:
                                trk_state.name = None                      # close enough to check: wait for verdict
                            elif antispoof is not None and trk_state.live is False:
                                trk_state.name = "Spoof"
                                trk_state.spoof_hits += 1
                                trk_state.unknown_hits = 0
                            else:
                                trk_state.name = who or "Unknown"
                                trk_state.spoof_hits = 0
                                trk_state.unknown_hits = 0 if who else trk_state.unknown_hits + 1

                            # photos: verified enrolled people, one folder per run (for training / review)
                            if (args.save_photos and who and trk_state.name == who
                                    and verified_ok(trk_state)
                                    and time.time() - photo_state.get(who, {}).get("last_t", 0) >= args.photo_every_s):
                                save_face_photo(who, best_det.aligned, best_det.face_emb)

                            # learn: confident + live sightings improve the profile over time
                            if args.learn and who and trk_state.name == who and verified_ok(trk_state):
                                face_db.maybe_learn(who, best_det.face_emb, sim, min_sim=args.learn_min_sim)

                            # events / alerts
                            if events is not None:
                                if trk_state.name == "Spoof" and trk_state.spoof_hits >= args.spoof_hits:
                                    events.emit("spoof", closest if sim >= args.match_thresh else "", sim, live_score,
                                                clean, best_det.bbox_xyxy, key=("spoof",))
                                elif trk_state.name == "Unknown" and trk_state.unknown_hits >= args.unknown_hits:
                                    events.emit("unknown", "", sim, live_score, clean, best_det.bbox_xyxy, key=("unknown",))
                                elif who and trk_state.name == who and verified_ok(trk_state):
                                    events.emit("seen", who, sim, live_score, key=("seen", who),
                                                cooldown_s=args.seen_cooldown_s)

                    # gesture verification + gesture password (for recognized enrolled people)
                    if (hand_tracker is not None and trk_state is not None and measured_now
                            and trk_state.name not in (None, "Unknown", "Spoof")):
                        now_t = time.time()
                        fwid, fcx = max(1, x2 - x1), (x1 + x2) / 2
                        mine = [h for h in hand_list if abs(h[0][0][0] - fcx) < 3.2 * fwid and h[0][0][1] > y1 - fwid]
                        hand_g = min(mine, key=lambda h: abs(h[0][0][0] - fcx))[2] if mine else None
                        if args.verify == "gesture" and not trk_state.gesture_ok:
                            if verified_until.get(trk_state.name, 0) > now_t:          # verified a moment ago
                                trk_state.gesture_ok = True
                            else:
                                if trk_state.challenge is None:
                                    trk_state.challenge = random.choice([c for c in CHALLENGES if c != trk_state.last_challenge])
                                    trk_state.ch_t0 = now_t
                                if hand_g == trk_state.challenge:
                                    trk_state.gesture_ok, trk_state.challenge = True, None
                                    if events is not None:
                                        events.emit("verified", trk_state.name, trk_state.name_sim, key=("verified", trk_state.name),
                                                    cooldown_s=args.seen_cooldown_s)
                                elif now_t - trk_state.ch_t0 > args.challenge_timeout_s:
                                    ps = person(trk_state.name)
                                    ps["fails"] += 1
                                    ps["fail_t"] = now_t
                                    trk_state.last_challenge, trk_state.challenge = trk_state.challenge, None
                                    if ps["fails"] >= args.challenge_fails and events is not None:
                                        events.emit("challenge_failed", trk_state.name, trk_state.name_sim, None, clean,
                                                    (x1, y1, x2, y2), key=("challenge_failed",))
                        if trk_state.gesture_ok:
                            verified_until[trk_state.name] = now_t + args.verify_grace_s
                        ps = person(trk_state.name)
                        trk_state.unlocked = ps["unlocked_until"] > now_t
                        if password and verified_ok(trk_state) and not trk_state.unlocked:
                            if hand_g in (None, "none", "other"):
                                ps["pw_last"] = None                     # hand lowered: next gesture can register
                            elif hand_g != ps["pw_last"]:
                                ps["pw_last"] = hand_g
                                if hand_g == password[ps["pw_idx"]]:
                                    ps["pw_idx"] += 1
                                    if ps["pw_idx"] == len(password):
                                        ps["unlocked_until"] = now_t + args.verify_grace_s
                                        ps["pw_idx"] = 0
                                        trk_state.unlocked = True
                                        if events is not None:
                                            events.emit("unlocked", trk_state.name, cooldown_s=0)
                                        if args.on_unlock_command:
                                            import subprocess
                                            subprocess.Popen(args.on_unlock_command.replace("{name}", trk_state.name), shell=True)
                                else:
                                    ps["pw_idx"] = 1 if hand_g == password[0] else 0
                                    ps["pw_fails"] += 1
                                    if ps["pw_fails"] >= 3 and events is not None:
                                        events.emit("password_failed", trk_state.name, cooldown_s=60)

                    # log identity for every measured face (keeps its last result between checks)
                    if measured_now and trk_state is not None and trk_state.name:
                        frame_names.append(trk_state.name)
                        frame_sims.append(float(trk_state.name_sim))
                        if trk_state.name not in ("Unknown", "Spoof") and verified_ok(trk_state):
                            frame_verified.add(trk_state.name)

                    tname = trk_state.name if trk_state is not None else None
                    known_person = bool(tname) and tname not in ("Unknown", "Spoof")

                    # draw box (GREEN if measured_now else YELLOW)
                    status = "MEAS" if measured_now else "PRED"
                    color = (0, 255, 0) if measured_now else (0, 255, 255)

                    label = f"C{cam_id} ID{tid} {status}"
                    if trk_state is not None and trk_state.name:
                        label = f"{trk_state.name} ({trk_state.name_sim:.2f}) | " + label
                        if known_person:
                            color = (255, 200, 0)  # known person = blue
                        elif trk_state.name == "Spoof":
                            color = (0, 0, 255)
                    if trk_state is not None and trk_state.live_ema is not None:
                        label += f" | live {trk_state.live_ema:.2f}"
                    if trk_state is not None and trk_state.global_id is not None:
                        label += f" | G{trk_state.global_id}"

                    if trk_state is not None and args.expression and trk_state.expr_label is not None:
                        label += f" | {trk_state.expr_label}({trk_state.expr_prob:.2f})"

                    if trk_state is not None and (args.pose or args.gaze) and trk_state.yaw is not None:
                        label += f" | yaw{trk_state.yaw:.0f} pitch{trk_state.pitch:.0f}"
                        if args.gaze and trk_state.gaze:
                            label += f" gaze:{trk_state.gaze}"

                    # privacy: pixelate faces (all, or everyone who isn't a recognized enrolled person)
                    if args.anonymize == "all" or (args.anonymize == "unknown" and not known_person):
                        pixelate(frame, x1, y1, x2, y2, blocks=args.pixel_blocks)

                    if clean_hud:
                        # recording look: box + name tag only
                        who = tname
                        if who == "Spoof":
                            color, tag = (0, 0, 255), "SPOOF?"
                        elif who is None and antispoof is not None and face_db is not None and face_db.people:
                            color, tag = (160, 160, 160), ""        # still checking
                        elif known_person and not verified_ok(trk_state):
                            color = (0, 215, 255) if args.verify == "gesture" else (190, 160, 90)
                            tag = who.replace("_", " ") + "?"                      # recognized, not verified yet
                        else:
                            color = (255, 200, 0) if known_person else (0, 255, 0)
                            tag = who.replace("_", " ") if who else ""
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        if tag:
                            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            ty = max(th + 10, y1)
                            cv2.rectangle(frame, (x1 - 1, ty - th - 10), (x1 + tw + 10, ty), color, -1)
                            cv2.putText(frame, tag, (x1 + 5, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 0, 0), 2, cv2.LINE_AA)
                        # prompts under the face: gesture challenge / password progress
                        prompt, pcol = None, (0, 215, 255)
                        if known_person and trk_state.challenge and not trk_state.gesture_ok:
                            from avis.hands import GESTURE_TEXT
                            left = max(0.0, args.challenge_timeout_s - (time.time() - trk_state.ch_t0))
                            prompt = f"SHOW: {GESTURE_TEXT.get(trk_state.challenge, trk_state.challenge)}  {left:.0f}s"
                        elif known_person and password and verified_ok(trk_state):
                            prompt, pcol = (("UNLOCKED", (80, 255, 80)) if trk_state.unlocked
                                            else (f"PASSWORD {person(trk_state.name)['pw_idx']}/{len(password)}", (255, 200, 0)))
                        if prompt:
                            (pw_, ph_), _ = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            py = min(H - 6, y2 + ph_ + 12)
                            cv2.rectangle(frame, (x1 - 1, py - ph_ - 8), (x1 + pw_ + 10, py + 4), (0, 0, 0), -1)
                            cv2.putText(frame, prompt, (x1 + 5, py - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pcol, 2, cv2.LINE_AA)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, max(10, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

                    # CSV row (ALWAYS WORKS)
                    if metrics and trk_state is not None:
                        metrics.log_track(
                            ts=ts,
                            frame_idx=frame_idx,
                            cam_id=cam_id,
                            tracker_name=args.tracker,
                            tid=tid,
                            gid=trk_state.global_id,
                            bbox=(x1, y1, x2, y2),
                            measured_now=measured_now,
                            from_detector=from_detector,
                            people_count=len(people),
                            det_count=len(detections),
                            emotion_label=(trk_state.expr_label if args.expression else None),
                            emotion_conf=(trk_state.expr_prob if args.expression else None),
                            yaw=(trk_state.yaw if (args.pose or args.gaze) else None),
                            pitch=(trk_state.pitch if (args.pose or args.gaze) else None),
                            roll=(trk_state.roll if (args.pose or args.gaze) else None),
                            gaze=(trk_state.gaze if args.gaze else None),
                            fps=fps,
                            name=trk_state.name,
                        )

                presence.update(frame_verified)
                if face_db is not None and frame_idx % 300 == 0:
                    face_db.save()

                if session_log is not None and cam_id == 0:
                    session_log.log(ts, frame_idx, frame_ms, fps, len(people), len(detections),
                                    len(tracks_out), meas_tracks, pred_tracks, frame_names, frame_sims,
                                    verified=len(frame_verified))

                # cleanup global reid
                if args.reid and global_reid is not None:
                    global_reid.cleanup(frame_idx)

                # fusion HUD on cam0
                if args.fusion and args.reid and global_reid is not None and cam_id == 0:
                    y = 45
                    cv2.putText(frame, "FUSION (Global IDs):", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                    y += 24
                    items = sorted(global_reid.global_db.items(), key=lambda kv: kv[1]["last_frame"], reverse=True)[:6]
                    for gid, v in items:
                        cv2.putText(frame, f"G{gid} last_frame={v['last_frame']}", (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
                        y += 20

                if hand_list and any(getattr(t, "challenge", None) or (password and not getattr(t, "unlocked", True))
                                     for t in trackers[cam_id].tracks.values()):
                    for hpts, _, hg in hand_list:
                        for q in hpts[[0, 4, 8, 12, 16, 20]]:
                            cv2.circle(frame, (int(q[0]), int(q[1])), 4, (0, 215, 255), -1, cv2.LINE_AA)

                # footer (debug stats; hidden in clean/recording mode)
                if not clean_hud:
                  cv2.putText(
                    frame,
                    f"tracker={args.tracker} cam={cam_id} fps={fps:.1f} people={len(people)} dets={len(detections)} tracks={len(tracks_out)} MEAS={meas_tracks} PRED={pred_tracks}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if cam_id == 0:
                    if enroll_name is not None:
                        msg = f"ENROLLING {enroll_name}: {len(enroll_embs)}/{args.enroll_samples}{enroll_note}"
                        if len(id_dets) != 1:
                            msg += "  (need ONE clear face: look at the camera, no cap brim/hand over it)"
                        cv2.putText(frame, msg, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    elif sface is not None and not args.record and args.debug:
                        cv2.putText(frame, "E = enroll a face   Q = quit", (10, H - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

                if args.record and cam_id == 0:
                    if recorder is None and record_path and frame_idx >= 20:
                        rec_fps = max(5, min(30, int(round(fps))))
                        recorder, codec = open_recorder(record_path, rec_fps, (W, H))
                        if recorder is None:
                            logging.error("[REC] could not open a video writer; recording disabled.")
                            record_path = None
                        else:
                            logging.info(f"[REC] recording at {rec_fps} fps ({codec}). Press Q to stop and save.")
                    if recorder is not None:
                        recorder.write(frame)
                        # REC badge on the preview only (not burned into the video)
                        frame = frame.copy()
                        cv2.circle(frame, (W - 70, 22), 7, (0, 0, 255), -1)
                        cv2.putText(frame, "REC", (W - 58, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow(f"AVIS Cam {cam_id}", frame)
                shown_windows.add(f"AVIS Cam {cam_id}")

            if metrics:
                metrics.flush()

            key = cv2.waitKey(1) & 0xFF
            if key != 255 and chr(key).lower() == "q":
                break
            # closing the window with its X = quit cleanly (so recordings/logs are saved)
            if any(cv2.getWindowProperty(wn, cv2.WND_PROP_VISIBLE) < 1 for wn in shown_windows):
                logging.info("Window closed.")
                break
            if key != 255 and chr(key).lower() == "e" and sface is not None and enroll_name is None:
                typed = input("\n[AVIS] Type the name to enroll, then press Enter: ")
                enroll_name = sanitize_name(typed) or None
                enroll_embs = []
                if enroll_name:
                    logging.info(f"[ID] enrolling '{enroll_name}': look at the camera, turn your head a little.")

            if args.log_interval > 0 and (frame_idx % args.log_interval == 0):
                logging.info(f"Frame {frame_idx} | fps={fps:.1f}")

    finally:
        for cap in caps.values():
            try:
                cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        if recorder is not None:
            recorder.release()
            logging.info(f"[REC] saved video -> {record_path}")
        if face_db is not None:
            face_db.save()
            for nm, k in face_db.learned_this_session.items():
                logging.info(f"[ML] learned {k} new samples of {nm} this session (profile: {len(face_db.people.get(nm, []))})")
            for nm, st in photo_state.items():
                if st["n"]:
                    logging.info(f"[PHOTOS] saved {st['n']} photos of {nm} -> dataset/{nm}/{session_folder}/")
            if args.auto_train:
                start_auto_train(face_db, args)
        if events is not None:
            events.close()
        if hand_tracker is not None:
            hand_tracker.close()
        if session_log is not None:
            session_log.close()
            logging.info(f"[LOG] saved stats -> {session_log.path}  (graph it: python analyze_runs.py)")
        if metrics:
            metrics.close()
        logging.info("Shutting down cleanly.")


if __name__ == "__main__":
    main()
