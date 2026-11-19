"""
hand3d.py - AVIS Hand 3D: draw 3D shapes in the air with your finger, then grab,
move, spin and stretch them with your hands. Shows each shape's real-world size.

Gestures
  POINT + HOLD STILL (~half a second)      a ring fills at your fingertip, then you're drawing.
                                          Lower the finger to finish:
        circle -> SPHERE   square/rectangle -> BOX   triangle -> PYRAMID   straight line -> CYLINDER
  PINCH (thumb + index) on a shape        grab + move it. Twist your wrist to rotate it.
                                          Let go while moving sideways = flick it into a spin.
  "L" HAND on a shape (thumb + index out)  spread fingertips = bigger, bring them closer = smaller.
                                          Pull your hand away to keep that size.
  SWIPE a fingertip across a shape        spins it with real physics: faster swipe = faster spin,
                                          bigger shapes are heavier, friction slows it to a stop
  PINCH ONE SHAPE WITH BOTH HANDS         pull apart / push together to resize, twist to rotate
  OPEN PALM under a shape                 it floats above your hand and spins
  FIST on a shape (hold ~half a second)   delete it


Keys: T drawing on/off   C clear all   R stop/start recording   S screenshot   M mirror   D hand skeleton   Q quit

Every session is recorded automatically to recordings/Hand3D/Hand3D Session N.mp4 (separate from AVIS Trials).
The video shows the shapes and labels but not the on-screen help text. Quit with Q (or the window's X) to save.

Sizes are estimated from your hand (average palm = 8 cm across). Calibrate with --palm-cm.
Runs in AVIS's main environment (.venv), which now includes MediaPipe.
"""
import argparse
import math
import os
import re
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from avis.camera import open_source  # noqa: E402

HAND_MODEL = os.path.join(HERE, "models", "hands", "hand_landmarker.task")
YUNET = os.path.join(HERE, "models", "yunet", "face_detection_yunet_2023mar.onnx")
SFACE = os.path.join(HERE, "models", "sface", "face_recognition_sface_2021dec.onnx")

NEON = {"sphere": (255, 200, 40), "box": (80, 255, 180), "pyramid": (60, 170, 255), "cylinder": (255, 110, 230)}
TIPS, PIPS = (8, 12, 16, 20), (6, 10, 14, 18)
DWELL_S = 0.45            # hold a pointing finger still this long to start drawing
PALM = [0, 5, 9, 13, 17]
SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


# ---------------- math ----------------
def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


TILT = rx(math.radians(-22)) @ ry(math.radians(28))

# spin physics: d(omega)/dt = -DRAG*omega - FRICTION*unit(omega)   (air drag + rolling friction -> comes to a full stop)
DRAG, FRICTION = 0.25, 0.55


def rot_vec(w, dt):
    """Rotation matrix for angular velocity vector w (rad/s) over dt (Rodrigues)."""
    n = float(np.linalg.norm(w))
    th = n * dt
    if th < 1e-9:
        return np.eye(3)
    k = w / n
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)   # starting 3/4 view so new shapes look 3D


# ---------------- shapes ----------------
class Shape:
    def __init__(self, kind, center, dims_px, cm_per_px, R=None):
        self.kind = kind
        self.c = np.array(center, np.float64)
        self.d = dict(dims_px)            # pixel dimensions at scale 1
        self.cm_per_px = float(cm_per_px)  # frozen at creation so sizes don't jitter
        self.R = TILT.copy() if R is None else R
        self.scale = 1.0
        self.omega = np.array([0.0, 0.9, 0.0])   # angular velocity (rad/s); pops in with a little spin
        self.born = time.time()
        self._mesh = None

    def mesh(self):
        if self._mesh is not None:
            return self._mesh
        k, d = self.kind, self.d
        if k == "sphere":
            r, nl, nm = d["r"], 10, 16
            V = [(r * math.sin(t) * math.cos(p), r * math.cos(t), r * math.sin(t) * math.sin(p))
                 for t in np.linspace(0, math.pi, nl + 1) for p in np.linspace(0, 2 * math.pi, nm + 1)]
            F = [(i * (nm + 1) + j, (i + 1) * (nm + 1) + j, (i + 1) * (nm + 1) + j + 1, i * (nm + 1) + j + 1)
                 for i in range(nl) for j in range(nm)]
        elif k == "box":
            w, h, z = d["w"] / 2, d["h"] / 2, d["d"] / 2
            V = [(x, y, q) for x in (-w, w) for y in (-h, h) for q in (-z, z)]
            F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        elif k == "pyramid":
            b, h = d["b"] / 2, d["h"] / 2
            V = [(-b, h, -b), (b, h, -b), (b, h, b), (-b, h, b), (0, -h, 0)]
            F = [(0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
        else:  # cylinder along local x
            L, r, n = d["L"] / 2, d["r"], 20
            ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
            V = [(-L, r * math.cos(a), r * math.sin(a)) for a in ang] + [(L, r * math.cos(a), r * math.sin(a)) for a in ang]
            F = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)] + [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
        self._mesh = (np.array(V, np.float64), F)
        return self._mesh

    def heaviness(self):
        """0..1 how easily a swipe spins it: bigger shapes have more rotational inertia."""
        return 1.0 / (1.0 + (self.radius_px() / 180.0) ** 2)

    def step_physics(self, dt):
        w = self.omega
        n = float(np.linalg.norm(w))
        if n < 1e-6:
            self.omega = np.zeros(3)
            return
        self.R = rot_vec(w, dt) @ self.R
        n_new = max(0.0, n * math.exp(-DRAG * dt) - FRICTION * dt)
        self.omega = w * (n_new / n) if n_new > 0.02 else np.zeros(3)

    def radius_px(self):
        d, s = self.d, self.scale
        if self.kind == "sphere":
            return d["r"] * s
        if self.kind == "box":
            return 0.5 * math.sqrt(d["w"] ** 2 + d["h"] ** 2 + d["d"] ** 2) * s
        if self.kind == "pyramid":
            return 0.5 * math.sqrt(2 * d["b"] ** 2 + d["h"] ** 2) * s
        return 0.5 * d["L"] * s

    def measurements(self):
        k = self.cm_per_px * self.scale
        d = {key: v * k for key, v in self.d.items()}
        if self.kind == "sphere":
            r = d["r"]
            return "SPHERE", [f"diameter {2 * r:.1f} cm", f"volume {4 / 3 * math.pi * r ** 3:,.0f} cm3"]
        if self.kind == "box":
            return "BOX", [f"{d['w']:.1f} x {d['h']:.1f} x {d['d']:.1f} cm", f"volume {d['w'] * d['h'] * d['d']:,.0f} cm3"]
        if self.kind == "pyramid":
            return "PYRAMID", [f"base {d['b']:.1f} cm  height {d['h']:.1f} cm", f"volume {d['b'] ** 2 * d['h'] / 3:,.0f} cm3"]
        return "CYLINDER", [f"length {d['L']:.1f} cm  dia {2 * d['r']:.1f} cm",
                            f"volume {math.pi * d['r'] ** 2 * d['L']:,.0f} cm3"]


def recognize_stroke(pts, cm_per_px):
    """Turn a drawn stroke into a Shape (or None if it isn't clear enough)."""
    if len(pts) < 12:
        return None, "too short"
    P = np.array(pts, np.float32)
    k = 5   # smooth hand jitter
    if len(P) > k:
        P = np.stack([np.convolve(P[:, i], np.ones(k) / k, mode="valid") for i in (0, 1)], axis=1).astype(np.float32)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    path = float(seg.sum())
    x, y, w, h = cv2.boundingRect(P)
    size = max(w, h)
    if path < 60 or size < 40 or size * cm_per_px < 5.0:
        return None, "too small - draw it bigger"
    gap = float(np.linalg.norm(P[0] - P[-1]))
    straight = gap / max(path, 1e-6)

    if straight > 0.85:                                    # straight line -> cylinder
        a, b = P[0], P[-1]
        L = float(np.linalg.norm(b - a))
        if L * cm_per_px < 8.0:
            return None, "line too short for a cylinder"
        ang = math.atan2(float(b[1] - a[1]), float(b[0] - a[0]))
        r = max(10.0, 0.09 * L)
        return Shape("cylinder", (a + b) / 2, {"L": L, "r": r}, cm_per_px, R=rx(math.radians(-20)) @ rz(ang)), None

    if gap > 0.45 * size:
        return None, "close the shape (end near where you started)"

    hull = cv2.convexHull(P)
    area = float(cv2.contourArea(hull))
    perim = float(cv2.arcLength(hull, True))
    circ = 4 * math.pi * area / max(perim ** 2, 1e-6)
    M = cv2.moments(hull)
    c = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]]) if M["m00"] else np.array([x + w / 2, y + h / 2])

    # how much of its tightest rectangle / triangle the shape fills:
    # square ~0.95, circle ~0.79, triangle ~0.5 of the rectangle (but ~1.0 of its triangle)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(hull)
    rect_fill = area / max(rw * rh, 1e-6)
    tri_area, _ = cv2.minEnclosingTriangle(hull)
    tri_fill = area / max(float(tri_area), 1e-6)
    if tri_fill > 0.86 and rect_fill < 0.72:
        bx, by, bw, bh = cv2.boundingRect(hull)
        return Shape("pyramid", (bx + bw / 2, by + bh / 2), {"b": float(bw), "h": float(bh)}, cm_per_px,
                     R=rx(math.radians(-18)) @ ry(math.radians(30))), None
    if rect_fill < 0.875 and circ >= 0.75:
        return Shape("sphere", c, {"r": math.sqrt(area / math.pi)}, cm_per_px), None
    if rw < rh:
        rw, rh, ang = rh, rw, ang + 90
    return Shape("box", (cx, cy), {"w": rw, "h": rh, "d": min(rw, rh)}, cm_per_px,
                 R=TILT @ rz(math.radians(ang))), None


# ---------------- rendering ----------------
LIGHT = np.array([-0.4, -0.7, -0.6]) / np.linalg.norm([-0.4, -0.7, -0.6])


def draw_shapes(frame, shapes, focal=900.0):
    if not shapes:
        return frame
    H, W = frame.shape[:2]
    faces, lines, boxes = [], [], []
    for s in shapes:
        V, F = s.mesh()
        P = (s.R @ V.T).T * s.scale
        k = focal / np.maximum(focal + P[:, 2], 60.0)
        xy = s.c + P[:, :2] * k[:, None]
        s.bbox2d = (*xy.min(axis=0), *xy.max(axis=0))
        boxes.append(s.bbox2d)
        col = np.array(NEON[s.kind], np.float64)
        order = sorted(range(len(F)), key=lambda i: -float(np.mean(P[list(F[i]), 2])))   # far faces first
        for i in order:
            f = list(F[i])
            n = np.cross(P[f[1]] - P[f[0]], P[f[2]] - P[f[0]])
            nn = np.linalg.norm(n)
            shade = 0.25 + 0.75 * abs(float(n @ LIGHT) / nn) if nn > 1e-9 else 0.5
            poly = np.round(xy[f]).astype(np.int32)
            faces.append((poly, tuple(float(v) for v in col * shade * 0.8)))
            lines.append((poly, tuple(int(v) for v in col)))
    # work only inside the region the shapes cover (+ glow margin): much cheaper than the full frame
    m = 30
    x0 = int(max(0, min(b[0] for b in boxes) - m)); y0 = int(max(0, min(b[1] for b in boxes) - m))
    x1 = int(min(W, max(b[2] for b in boxes) + m)); y1 = int(min(H, max(b[3] for b in boxes) + m))
    out = frame.copy()
    if x1 <= x0 or y1 <= y0:
        return out
    roi = out[y0:y1, x0:x1]
    off = np.array([x0, y0], np.int32)
    fill = roi.copy()
    for poly, col in faces:
        cv2.fillConvexPoly(fill, poly - off, col, cv2.LINE_AA)
    blended = cv2.addWeighted(fill, 0.55, roi, 0.45, 0)
    glow = np.zeros_like(roi)
    for poly, col in lines:
        cv2.polylines(glow, [poly - off], True, col, 3, cv2.LINE_AA)
    gs = cv2.resize(glow, (max(1, glow.shape[1] // 2), max(1, glow.shape[0] // 2)), interpolation=cv2.INTER_AREA)
    gs = cv2.resize(cv2.GaussianBlur(gs, (0, 0), 3), (glow.shape[1], glow.shape[0]), interpolation=cv2.INTER_LINEAR)
    blended = cv2.add(blended, gs)
    for poly, col in lines:
        cv2.polylines(blended, [poly - off], True, (255, 255, 255), 1, cv2.LINE_AA)
    out[y0:y1, x0:x1] = blended
    return out


def draw_label(img, s, placed):
    title, rows = s.measurements()
    w_spin = float(np.linalg.norm(s.omega))
    if w_spin > 0.05:
        rows = rows + [f"spin {w_spin * 60 / (2 * math.pi):.0f} rpm"]
    H, W = img.shape[:2]
    col = NEON[s.kind]
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for t in rows + [title]) + 16
    h = 22 + 20 * len(rows)
    x1, y1, x2, y2 = getattr(s, "bbox2d", (s.c[0] - 30, s.c[1] - 30, s.c[0] + 30, s.c[1] + 30))
    # try: right of shape, left, above, below - first spot that is on screen and doesn't overlap
    cands = [((x2 + 14, (y1 + y2) / 2 - h / 2), (x2, (y1 + y2) / 2)), ((x1 - 14 - w, (y1 + y2) / 2 - h / 2), (x1, (y1 + y2) / 2)),
             (((x1 + x2) / 2 - w / 2, y1 - 14 - h), ((x1 + x2) / 2, y1)), (((x1 + x2) / 2 - w / 2, y2 + 14), ((x1 + x2) / 2, y2))]
    best = None
    for (bx, by), anchor in cands:
        bx, by = int(min(max(bx, 4), W - w - 4)), int(min(max(by, 60), H - h - 36))
        rect = (bx, by, bx + w, by + h)
        clash = sum(1 for r in placed if not (rect[2] < r[0] or rect[0] > r[2] or rect[3] < r[1] or rect[1] > r[3]))
        if best is None or clash < best[0]:
            best = (clash, bx, by, anchor)
        if clash == 0:
            break
    _, bx, by, anchor = best
    placed.append((bx, by, bx + w, by + h))
    near = (min(max(int(anchor[0]), bx), bx + w), min(max(int(anchor[1]), by), by + h))
    cv2.line(img, (int(anchor[0]), int(anchor[1])), near, col, 1, cv2.LINE_AA)
    sub = img[by:by + h, bx:bx + w]
    if sub.size:
        img[by:by + h, bx:bx + w] = (sub * 0.35).astype(np.uint8)
    cv2.rectangle(img, (bx, by), (bx + w, by + h), col, 1, cv2.LINE_AA)
    cv2.putText(img, title, (bx + 8, by + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    for i, t in enumerate(rows):
        cv2.putText(img, t, (bx + 8, by + 37 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)


# ---------------- hands ----------------
class MPHands:
    """MediaPipe 1.x HandLandmarker in LIVE_STREAM mode: runs on its own thread, so the window keeps the
    camera's frame rate. Tracks on a small copy (track_width px wide); landmarks are scaled back to full size."""
    def __init__(self, model_path, num_hands=2, track_width=640):
        import threading
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        self.mp = mp
        self.track_width = int(track_width)
        self._lock = threading.Lock()
        self._latest = []
        self.updates = 0
        opts = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.LIVE_STREAM, num_hands=num_hands,
            min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5,
            result_callback=self._on_result)
        self.lm = vision.HandLandmarker.create_from_options(opts)
        self.ts = 0
        self.size = (1, 1)

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
        h, w = frame.shape[:2]
        self.size = (w, h)
        small = frame if w <= self.track_width else cv2.resize(
            frame, (self.track_width, int(h * self.track_width / w)), interpolation=cv2.INTER_AREA)
        self.ts = max(self.ts + 1, int(time.monotonic() * 1000))
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        self.lm.detect_async(img, self.ts)          # returns immediately; result arrives via callback
        with self._lock:
            return list(self._latest)

    def close(self):
        self.lm.close()


class HandState:
    def __init__(self):
        self.cand, self.count, self.state = None, 0, "none"
        self.tip = None
        self.stroke = []
        self.off_frames = 0
        self.grab = None            # (shape, offset, last_roll)
        self.hist = []              # (t, x, y) of pinch point
        self.carry = None
        self.fist_t = 0.0
        self.missing = 0
        self.resize = None          # (shape, start span, start scale)
        self.prev_tip = None
        self.vel = np.zeros(2)      # fingertip velocity px/s (smoothed)
        self.dwell_t = 0.0          # how long the pointing fingertip has been held still
        self.dwell_anchor = None


def hand_features(pts):
    palm_w = float(np.linalg.norm(pts[5] - pts[17]))
    palm_l = float(np.linalg.norm(pts[0] - pts[9]))
    unit = max(palm_w, palm_l * 0.8, 1.0)
    ext = [float(np.linalg.norm(pts[t] - pts[0])) > float(np.linalg.norm(pts[p] - pts[0])) + 0.12 * unit
           for t, p in zip(TIPS, PIPS)]
    pinch_d = float(np.linalg.norm(pts[4] - pts[8])) / unit
    tdir, idir = pts[4] - pts[2], pts[8] - pts[5]
    cos_ti = float(tdir @ idir) / max(float(np.linalg.norm(tdir) * np.linalg.norm(idir)), 1e-6)
    thumb_out = float(np.linalg.norm(pts[4] - pts[5])) / unit > 0.6 and cos_ti < math.cos(math.radians(40))   # an "L", not a relaxed thumb
    return {"thumb_out": thumb_out, "palm_w": palm_w, "palm_l": palm_l, "ext": ext, "pinch_d": pinch_d,
            "pinch_pt": (pts[4] + pts[8]) / 2, "palm_c": pts[PALM].mean(axis=0),
            "roll": math.atan2(float(pts[17][1] - pts[5][1]), float(pts[17][0] - pts[5][0]))}


def raw_gesture(f, prev):
    if f["pinch_d"] < 0.33 or (prev == "pinch" and f["pinch_d"] < 0.48):
        return "pinch"
    ext = f["ext"]
    if ext[0] and not ext[1] and not ext[2] and not ext[3]:
        return "spread" if f["thumb_out"] else "point"
    if all(ext):
        return "open"
    if not any(ext):
        return "fist"
    return "other"


# ---------------- face label (optional) ----------------
class FaceLabel:
    def __init__(self):
        self.ok = os.path.exists(YUNET)
        self.rec, self.db = None, None
        try:
            self.det = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), 0.8, 0.3, 50) if self.ok else None
        except Exception as e:                      # different OpenCV build: run without the face label
            print(f"(face label off: {e})")
            self.ok, self.det = False, None
        if self.ok and os.path.exists(SFACE):
            try:
                from avis.learning import FaceDB
                self.db = FaceDB(os.path.join(HERE, "faces_db"))
                self.rec = cv2.FaceRecognizerSF.create(SFACE, "") if self.db.people else None
            except Exception:
                self.rec = None
        self.box, self.name, self.n = None, "", 0

    def update(self, frame):
        if not self.ok:
            return
        try:
            self._update(frame)
        except cv2.error:
            self.ok = False

    def _update(self, frame):
        self.n += 1
        if self.n % 6:
            return
        small = cv2.resize(frame, None, fx=0.5, fy=0.5)
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.det.detect(small)
        if faces is None or not len(faces):
            self.box = None
            return
        f = max(faces, key=lambda r: r[2] * r[3]).copy()
        f[:14] *= 2
        self.box = tuple(int(v) for v in (f[0], f[1], f[0] + f[2], f[1] + f[3]))
        if self.rec is not None and (self.n % 15 == 0 or not self.name):
            e = self.rec.feature(self.rec.alignCrop(frame, f)).reshape(-1).astype(np.float32)
            e /= max(np.linalg.norm(e), 1e-9)
            who, _, _ = self.db.match(e, 0.363)
            self.name = who.replace("_", " ") if who else "Unknown"

    def draw(self, img):
        if not self.box:
            return
        x1, y1, x2, y2 = self.box
        L, col = max(10, (x2 - x1) // 5), (200, 200, 200)
        for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (px, py), (px + dx * L, py), col, 2, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py + dy * L), col, 2, cv2.LINE_AA)
        if self.name:
            cv2.putText(img, self.name, (x1, max(16, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)


# ---------------- app ----------------
class App:
    def __init__(self, tracker, palm_cm=8.0, mirror=True, face=True):
        self.tracker = tracker
        self.palm_cm = palm_cm
        self.mirror = mirror
        self.shapes = []
        self.hands = {}
        self.toasts = []
        self.poofs = []
        self.debug = False
        self.cm_per_px = 0.05
        self.face = FaceLabel() if face else None
        self.draw_enabled = True
        self.frame_hands = []
        self.last_t = time.time()
        self.last_detected = []

    def toast(self, text, secs=2.2):
        self.toasts.append((text, time.time() + secs))

    def hit(self, p, pad=1.15):
        best, bd = None, 1e9
        for s in self.shapes:
            d = float(np.linalg.norm(p - s.c))
            if d < s.radius_px() * pad + 18 and d < bd:
                best, bd = s, d
        return best

    def finish_stroke(self, hs):
        if len(hs.stroke) >= 3:
            shape, why = recognize_stroke(hs.stroke, self.cm_per_px)
            if shape is not None:
                self.shapes.append(shape)
                title, rows = shape.measurements()
                self.toast(f"{title}  {rows[0]}")
            elif why:
                self.toast(f"Didn't catch that shape ({why})", 1.6)
        hs.stroke = []

    def release(self, hs, now):
        if hs.grab is not None:
            shape = hs.grab[0]
            pts = [h for h in hs.hist if now - h[0] < 0.15]
            if len(pts) >= 2 and pts[-1][0] > pts[0][0]:
                T = pts[-1][0] - pts[0][0]
                vx, vy = (pts[-1][1] - pts[0][1]) / T, (pts[-1][2] - pts[0][2]) / T
                if math.hypot(vx, vy) > 250:            # flick on release: sideways -> spin around vertical axis
                    R = max(shape.radius_px(), 20.0)
                    shape.omega = shape.omega + 0.8 * shape.heaviness() * np.array([vy / R, -vx / R, 0.0])
            hs.grab = None
            hs.hist = []
        hs.carry = None

    def step(self, frame):
        now = time.time()
        dt = min(0.1, now - self.last_t)
        self.last_t = now
        if self.mirror:
            frame = cv2.flip(frame, 1)
        detected = self.tracker(frame)
        self.last_detected = detected
        if self.face is not None:
            self.face.update(frame)

        seen = set()
        feats = {}
        for i, (label, pts) in enumerate(detected):
            key = label if label not in seen else f"{label}{i}"
            seen.add(key)
            f = hand_features(pts)
            feats[key] = (f, pts)
            px_per_cm = max(f["palm_w"] / self.palm_cm, f["palm_l"] / (self.palm_cm * 1.19))
            if px_per_cm > 1:
                self.cm_per_px = 0.9 * self.cm_per_px + 0.1 * (1.0 / px_per_cm)

        for key in list(self.hands):
            if key not in feats:
                hs = self.hands[key]
                hs.missing += 1
                if hs.missing > 5:
                    self.finish_stroke(hs)
                    self.release(hs, now)
                    del self.hands[key]

        for key, (f, pts) in feats.items():
            hs = self.hands.setdefault(key, HandState())
            hs.missing = 0
            raw = raw_gesture(f, hs.state)
            if raw == hs.cand:
                hs.count += 1
            else:
                hs.cand, hs.count = raw, 1
            prev = hs.state
            if hs.count >= 3:
                hs.state = raw
            st = hs.state

            # fingertip velocity (for swipes)
            tip = pts[8]
            if hs.prev_tip is not None and dt > 0:
                hs.vel = 0.5 * hs.vel + 0.5 * (tip - hs.prev_tip) / dt
            hs.prev_tip = tip.copy()

            # RESIZE with one hand: thumb + index "L" over a shape; fingertip spread sets the size
            span = f["pinch_d"]
            if hs.resize is None and st == "spread" and hs.grab is None:
                s = self.hit(f["pinch_pt"], pad=1.3)
                if s is not None:
                    hs.resize = (s, max(span, 0.15), s.scale)
                    s.omega = np.zeros(3)
            if hs.resize is not None:
                s, d0, s0 = hs.resize
                away = float(np.linalg.norm(f["pinch_pt"] - s.c)) > s.radius_px() * 1.8 + 60
                if s not in self.shapes or away or st in ("open", "fist"):
                    if s in self.shapes:
                        t, rows = s.measurements()
                        self.toast(f"{t}  {rows[0]}", 1.6)
                    hs.resize = None
                else:
                    s.scale = float(np.clip(s0 * max(span, 0.05) / d0, 0.15, 8.0))
                    hs.resize_line = (pts[4].copy(), pts[8].copy(), s.scale / s0)

            # SWIPE to spin: fingertip moving fast across a shape drags its surface (friction coupling)
            if st not in ("pinch", "fist") and hs.resize is None and hs.grab is None and float(np.linalg.norm(hs.vel)) > 250:
                for s in self.shapes:
                    rel = tip - s.c
                    Rr = s.radius_px()
                    if float(np.linalg.norm(rel)) < Rr and not any(h.grab is not None and h.grab[0] is s for h in self.hands.values()):
                        r3 = np.array([rel[0], rel[1], -math.sqrt(max(Rr * Rr - float(rel @ rel), 0.0))])
                        v_surf = np.cross(s.omega, r3)
                        dv = np.array([hs.vel[0] - v_surf[0], hs.vel[1] - v_surf[1], 0.0])
                        k = min(1.0, 10.0 * dt) * s.heaviness()
                        s.omega = s.omega + k * np.cross(r3, dv) / (Rr * Rr)
                        s.omega = np.clip(s.omega, -25, 25)

            # DRAW (strokes that start on top of a shape are swipes, not drawings)
            hs.tip = tip if hs.tip is None else 0.55 * hs.tip + 0.45 * tip
            can_draw = self.draw_enabled and st == "point" and hs.grab is None and hs.resize is None
            if can_draw and not hs.stroke:
                # deliberate start: hold the pointing fingertip still for DWELL_S (casual pointing never draws)
                still = hs.dwell_anchor is not None and float(np.linalg.norm(hs.tip - hs.dwell_anchor)) < 0.22 * f["palm_w"] + 4
                if self.hit(hs.tip, pad=1.0) is not None:
                    hs.dwell_t, hs.dwell_anchor = 0.0, None          # on a shape = swipe, not draw
                elif still:
                    hs.dwell_t += dt
                    if hs.dwell_t >= DWELL_S:
                        hs.stroke = [hs.tip.copy()]
                        hs.off_frames = 0
                else:
                    hs.dwell_t, hs.dwell_anchor = 0.0, hs.tip.copy()
            elif not can_draw:
                hs.dwell_t, hs.dwell_anchor = 0.0, None
            if can_draw and hs.stroke:
                hs.off_frames = 0
                if np.linalg.norm(hs.tip - hs.stroke[-1]) > 2.5:
                    hs.stroke.append(hs.tip.copy())
            elif hs.stroke:
                hs.off_frames += 1
                if hs.off_frames >= 4:
                    self.finish_stroke(hs)

            # GRAB / MOVE / ROTATE
            pp = f["pinch_pt"]
            if st == "pinch" and prev != "pinch" and hs.grab is None and hs.resize is None:
                s = self.hit(pp)
                if s is not None:
                    hs.grab = (s, s.c - pp, f["roll"])
                    s.omega = np.zeros(3)
            if st == "pinch" and raw == "pinch" and hs.grab is not None:   # follow only a real pinch
                s, off, last_roll = hs.grab
                s.c = pp + off
                droll = wrap(f["roll"] - last_roll)
                s.R = rz(droll) @ s.R
                hs.grab = (s, off, f["roll"])
                hs.hist = (hs.hist + [(now, float(pp[0]), float(pp[1]))])[-8:]
            if hs.grab is not None and raw != "pinch" and hs.cand == raw and hs.count >= 2:
                self.release(hs, now)                                     # let go promptly

            # OPEN PALM: shape floats + spins on your hand
            if st == "open" and hs.grab is None:
                if hs.carry is None and float(np.linalg.norm(hs.vel)) < 200:
                    above = f["palm_c"] - np.array([0.0, 1.6 * f["palm_w"]])
                    hs.carry = self.hit(above, pad=1.6)
                if hs.carry is not None and hs.carry in self.shapes:
                    s = hs.carry
                    target = f["palm_c"] - np.array([0.0, s.radius_px() + 0.6 * f["palm_w"]])
                    s.c += (target - s.c) * 0.35
                    s.omega = s.omega + (np.array([0.0, 2.8, 0.0]) - s.omega) * min(1.0, 3.0 * dt)
            elif st != "open":
                hs.carry = None

            # FIST: hold on a shape to delete
            if st == "fist":
                s = self.hit(f["palm_c"], pad=1.0)
                hs.fist_t = hs.fist_t + dt if s is not None else 0.0
                if s is not None and hs.fist_t > 0.55:
                    self.shapes.remove(s)
                    self.poofs.append((s.c.copy(), s.radius_px(), now, NEON[s.kind]))
                    self.toast(f"{s.kind.upper()} deleted", 1.4)
                    hs.fist_t = 0.0
            else:
                hs.fist_t = 0.0

        # two hands pinching the same shape -> scale + rotate
        grabbers = [(k, hs) for k, hs in self.hands.items() if hs.grab is not None and k in feats]
        if len(grabbers) == 2 and grabbers[0][1].grab[0] is grabbers[1][1].grab[0]:
            s = grabbers[0][1].grab[0]
            pa, pb = feats[grabbers[0][0]][0]["pinch_pt"], feats[grabbers[1][0]][0]["pinch_pt"]
            dist, ang = float(np.linalg.norm(pa - pb)), math.atan2(pb[1] - pa[1], pb[0] - pa[0])
            if not hasattr(s, "_two") or s._two is None:
                s._two = (dist, s.scale, ang)
            d0, s0, a0 = s._two
            s.scale = float(np.clip(s0 * dist / max(d0, 1), 0.2, 6))
            s.R = rz(wrap(ang - a0)) @ s.R
            s._two = (d0, s0, ang)
            s.c = (pa + pb) / 2
        else:
            for s in self.shapes:
                s._two = None

        self.frame_hands = [(f, pts, self.hands[k].state) for k, (f, pts) in feats.items() if k in self.hands]
        self.physics(dt)
        return frame

    def physics(self, dt):
        held = {hs.grab[0] for hs in self.hands.values() if hs.grab is not None}
        held |= {hs.resize[0] for hs in self.hands.values() if hs.resize is not None}
        for s in self.shapes:
            if s not in held:
                s.step_physics(dt)

    def render(self, frame):
        out = draw_shapes(frame, self.shapes)
        placed = []
        for s in self.shapes:
            draw_label(out, s, placed)
        now = time.time()
        for (c, r, t0, col) in list(self.poofs):
            a = (now - t0) / 0.45
            if a >= 1:
                self.poofs.remove((c, r, t0, col))
                continue
            cv2.circle(out, tuple(int(v) for v in c), int(r * (1 + a)), col, max(1, int(4 * (1 - a))), cv2.LINE_AA)
        for key, hs in self.hands.items():
            if hs.resize is not None and getattr(hs, "resize_line", None) is not None:
                a, b, ratio = hs.resize_line
                cv2.line(out, tuple(int(v) for v in a), tuple(int(v) for v in b), (255, 255, 255), 2, cv2.LINE_AA)
                for q in (a, b):
                    cv2.circle(out, tuple(int(v) for v in q), 7, (255, 220, 120), 2, cv2.LINE_AA)
                m = (a + b) / 2
                cv2.putText(out, f"{ratio * 100:.0f}%", (int(m[0]) + 10, int(m[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            if hs.dwell_t > 0 and not hs.stroke and hs.tip is not None:     # arming ring at the fingertip
                c = tuple(int(v) for v in hs.tip)
                cv2.circle(out, c, 16, (90, 90, 90), 1, cv2.LINE_AA)
                cv2.ellipse(out, c, (16, 16), -90, 0, 360 * min(1.0, hs.dwell_t / DWELL_S), (255, 255, 255), 3, cv2.LINE_AA)
            if len(hs.stroke) > 1:
                pts = np.array(hs.stroke, np.int32)
                glow = np.zeros_like(out)
                cv2.polylines(glow, [pts], False, (255, 255, 0), 8, cv2.LINE_AA)
                out = cv2.add(out, cv2.GaussianBlur(glow, (0, 0), 7))
                cv2.polylines(out, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)
        if self.face is not None:
            self.face.draw(out)
        return out

    def draw_hands(self, img, detected):
        for _, pts in detected:
            for a, b in SKELETON:
                cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), (200, 200, 200), 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, tuple(int(v) for v in p), 3, (0, 255, 255), -1, cv2.LINE_AA)

    def hud(self, img, fps, recording):
        H, W = img.shape[:2]
        cv2.putText(img, "AVIS HAND 3D", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 120), 2, cv2.LINE_AA)
        hz = getattr(self, "track_hz", 0.0)
        cv2.putText(img, f"{fps:.0f} fps  |  hands {hz:.0f}/s  |  {W}x{H}  |  {len(self.shapes)} shapes",
                    (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        for i, (key, hs) in enumerate(sorted(self.hands.items())):
            cv2.putText(img, f"{key}: {hs.state}", (14, 72 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 255, 160), 1, cv2.LINE_AA)
        bar = ("DRAW: point + hold still" if self.draw_enabled else "DRAW: off (T)") + \
              "   GRAB: pinch   RESIZE: thumb+index L   SPIN: swipe across   FLOAT: open palm   DELETE: hold fist"
        fs = 0.42
        while fs > 0.25 and cv2.getTextSize(bar, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0] > W - 20:
            fs -= 0.02
        img[H - 30:H] = (img[H - 30:H] * 0.4).astype(np.uint8)
        cv2.putText(img, bar, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, fs, (230, 230, 230), 1, cv2.LINE_AA)
        if recording:
            cv2.circle(img, (W - 70, 22), 7, (0, 0, 255), -1)
            cv2.putText(img, "REC", (W - 58, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        now = time.time()
        self.toasts = [t for t in self.toasts if t[1] > now]
        for i, (text, _) in enumerate(self.toasts[-3:]):
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            x, y = (W - tw) // 2, H - 48 - 34 * i
            cv2.rectangle(img, (x - 12, y - th - 10), (x + tw + 12, y + 10), (30, 30, 30), -1)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 230, 150), 2, cv2.LINE_AA)


def next_path(folder, stem, ext):
    os.makedirs(folder, exist_ok=True)
    nums = [int(m.group(1)) for f in os.listdir(folder) for m in [re.match(rf"{re.escape(stem)} (\d+){re.escape(ext)}$", f)] if m]
    return os.path.join(folder, f"{stem} {max(nums, default=0) + 1}{ext}")


def open_writer(path, fps, size):
    for api, cc in ([(cv2.CAP_MSMF, "H264")] if sys.platform == "win32" else []) + [(cv2.CAP_ANY, "mp4v")]:
        w = cv2.VideoWriter(path, api, cv2.VideoWriter_fourcc(*cc), float(fps), size)
        if w.isOpened():
            return w
        w.release()
    return None


def main():
    ap = argparse.ArgumentParser(description="AVIS Hand 3D")
    ap.add_argument("--source", default="0", help="webcam index, IP camera URL, or video file")
    ap.add_argument("--palm-cm", type=float, default=8.0, help="your palm width in cm (index knuckle to pinky knuckle)")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--no-face", action="store_true")
    ap.add_argument("--no-record", action="store_true", help="don't auto-record this session (R still starts one)")
    ap.add_argument("--width", type=int, default=1280, help="requested camera width (HD = sharper)")
    ap.add_argument("--height", type=int, default=720)

    a = ap.parse_args()

    if not os.path.exists(HAND_MODEL):
        sys.exit("Hand model missing. Run Task > 'AVIS: Hand 3D Setup' first.")
    try:
        tracker = MPHands(HAND_MODEL)
    except ImportError:
        sys.exit("MediaPipe isn't installed in this Python. Run Task > 'AVIS: Hand 3D Setup' first.")

    app = App(tracker, palm_cm=a.palm_cm, mirror=not a.no_mirror, face=not a.no_face)
    cam = open_source(a.source, width=a.width, height=a.height, fps=30)
    win = "AVIS Hand 3D"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    sized = False
    rec_dir = os.path.join(HERE, "recordings", "Hand3D")
    writer, rec_path, want_rec = None, None, not a.no_record
    frames_seen = 0
    fps, t_prev, shown = 0.0, time.time(), False
    hz_t0, hz_n0 = time.time(), 0
    print(__doc__.split("Run setup")[0])
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            frame = app.step(frame)
            out = app.render(frame)
            if app.debug:
                app.draw_hands(out, app.last_detected)
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3) if fps else 1 / max(now - t_prev, 1e-3)
            t_prev = now
            if now - hz_t0 >= 1.0:
                app.track_hz = (tracker.updates - hz_n0) / (now - hz_t0)
                hz_t0, hz_n0 = now, tracker.updates
            frames_seen += 1
            if want_rec and writer is None and frames_seen >= 30:          # wait ~1 s so the video's fps is right
                rec_path = next_path(rec_dir, "Hand3D Session", ".mp4")
                writer = open_writer(rec_path, max(10, min(30, round(fps or 20))), (out.shape[1], out.shape[0]))
                app.toast(f"Recording {os.path.basename(rec_path)}" if writer else "Could not start recording")
                want_rec = writer is not None
            clean = out.copy()
            app.hud(out, fps, writer is not None)
            if writer is not None:
                writer.write(clean)
            if not sized:
                cv2.resizeWindow(win, out.shape[1], out.shape[0])     # show at the camera's real resolution
                sized = True
            cv2.imshow(win, out)
            shown = True
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if shown and cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("c"), ord("C")):
                app.shapes.clear()
                app.toast("Cleared")
            elif key in (ord("t"), ord("T")):
                app.draw_enabled = not app.draw_enabled
                app.toast("Drawing " + ("on" if app.draw_enabled else "off"), 1.2)
            elif key in (ord("m"), ord("M")):
                app.mirror = not app.mirror
            elif key in (ord("d"), ord("D")):
                app.debug = not app.debug
            elif key in (ord("s"), ord("S")):
                p = next_path(rec_dir, "Hand3D Snapshot", ".png")
                cv2.imwrite(p, clean)
                app.toast(f"Saved {os.path.basename(p)}")
            elif key in (ord("r"), ord("R")):
                if writer is not None:
                    writer.release()
                    writer, want_rec = None, False
                    app.toast(f"Saved {os.path.basename(rec_path)}  (R = new recording)")
                    print(f"Saved {rec_path}")
                else:
                    want_rec, frames_seen = True, 30      # start a new numbered session file right away
    finally:
        cam.release()
        if writer is not None:
            writer.release()
            print(f"Saved {rec_path}")
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
