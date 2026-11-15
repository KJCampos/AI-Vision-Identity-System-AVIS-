"""
Shared hand tracking + gesture recognition (MediaPipe HandLandmarker, LIVE_STREAM mode).

Used by AVIS (gesture challenge / gesture password) and hand3d.py (drawing, touchless control).
Tracking runs on MediaPipe's own thread on a small copy of the frame, so the video stays smooth.

Gestures (named, stable after a few frames):
  open (5 fingers), fist, one..four (finger counts), peace, thumbs_up, point, pinch, l_shape
"""
import math
import os
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAND_MODEL = os.path.join(HERE, "models", "hands", "hand_landmarker.task")

TIPS, PIPS = (8, 12, 16, 20), (6, 10, 14, 18)
PALM = [0, 5, 9, 13, 17]
GESTURE_TEXT = {"open": "OPEN HAND", "fist": "FIST", "one": "1 FINGER", "two": "2 FINGERS", "three": "3 FINGERS",
                "four": "4 FINGERS", "peace": "PEACE SIGN", "thumbs_up": "THUMBS UP", "point": "POINT",
                "pinch": "PINCH", "l_shape": "L SHAPE"}


class HandTracker:
    def __init__(self, model_path=HAND_MODEL, num_hands=2, track_width=640):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"hand model missing: {model_path} (run Task 'AVIS: Hand 3D Setup')")
        self.mp = mp
        self.track_width = int(track_width)
        self._lock = threading.Lock()
        self._latest = []
        self.updates = 0
        self.size = (1, 1)
        self.ts = 0
        opts = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.LIVE_STREAM, num_hands=num_hands,
            min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5,
            result_callback=self._on_result)
        self.lm = vision.HandLandmarker.create_from_options(opts)

    def _on_result(self, res, image, ts):
        w, h = self.size
        out = []
        for i, lms in enumerate(res.hand_landmarks):
            label = "H"
            if res.handedness and i < len(res.handedness) and res.handedness[i]:
                label = res.handedness[i][0].category_name or "H"
            out.append((label, np.array([[p.x * w, p.y * h] for p in lms], np.float64)))
        with self._lock:
            self._latest = out
            self.updates += 1

    def __call__(self, frame):
        """Submit a frame; returns the newest [(label, 21x2 points in frame pixels)] (may lag ~1 frame)."""
        h, w = frame.shape[:2]
        self.size = (w, h)
        small = frame if w <= self.track_width else cv2.resize(
            frame, (self.track_width, int(h * self.track_width / w)), interpolation=cv2.INTER_AREA)
        self.ts = max(self.ts + 1, int(time.monotonic() * 1000))
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        self.lm.detect_async(img, self.ts)
        with self._lock:
            return list(self._latest)

    def close(self):
        self.lm.close()


def features(pts):
    palm_w = float(np.linalg.norm(pts[5] - pts[17]))
    palm_l = float(np.linalg.norm(pts[0] - pts[9]))
    unit = max(palm_w, palm_l * 0.8, 1.0)
    ext = [float(np.linalg.norm(pts[t] - pts[0])) > float(np.linalg.norm(pts[p] - pts[0])) + 0.12 * unit
           for t, p in zip(TIPS, PIPS)]
    tdir, idir = pts[4] - pts[2], pts[8] - pts[5]
    cos_ti = float(tdir @ idir) / max(float(np.linalg.norm(tdir) * np.linalg.norm(idir)), 1e-6)
    thumb_far = float(np.linalg.norm(pts[4] - pts[5])) / unit > 0.6
    thumb_out = thumb_far and float(np.linalg.norm(pts[4] - pts[9])) > float(np.linalg.norm(pts[3] - pts[9]))
    return {"palm_w": palm_w, "palm_l": palm_l, "unit": unit, "ext": ext, "thumb_out": thumb_out,
            "l_shape": thumb_far and cos_ti < math.cos(math.radians(40)),
            "thumb_up": float(pts[2][1] - pts[4][1]) > 0.45 * unit,
            "pinch_d": float(np.linalg.norm(pts[4] - pts[8])) / unit,
            "pinch_pt": (pts[4] + pts[8]) / 2, "palm_c": pts[PALM].mean(axis=0),
            "roll": math.atan2(float(pts[17][1] - pts[5][1]), float(pts[17][0] - pts[5][0]))}


def classify(f, prev=None):
    """Instantaneous gesture name for one hand."""
    if f["pinch_d"] < 0.33 or (prev == "pinch" and f["pinch_d"] < 0.48):
        return "pinch"
    ext = f["ext"]
    n4 = sum(ext)
    if n4 == 0:
        if f["thumb_out"] and f["thumb_up"]:
            return "thumbs_up"
        return "fist"
    if ext == [True, False, False, False]:
        return "l_shape" if f["l_shape"] else "point"
    if ext == [True, True, False, False]:
        return "three" if f["thumb_out"] else "peace"
    if n4 == 4:
        return "open" if f["thumb_out"] else "four"
    if ext == [True, True, True, False]:
        return "three" if not f["thumb_out"] else "four"
    return "other"


class StableGesture:
    """A gesture only 'counts' after it has been seen for `frames` frames in a row."""
    def __init__(self, frames=4):
        self.frames = frames
        self.cand, self.n, self.state = None, 0, "none"

    def update(self, g):
        if g == self.cand:
            self.n += 1
        else:
            self.cand, self.n = g, 1
        if self.n >= self.frames:
            self.state = g
        return self.state
