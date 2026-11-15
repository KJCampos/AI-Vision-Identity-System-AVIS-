"""Offline face pipeline for evaluate.py / train_identity.py: detect -> align -> embed (+ liveness)."""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YUNET = os.path.join(HERE, "models", "yunet", "face_detection_yunet_2023mar.onnx")
SFACE = os.path.join(HERE, "models", "sface", "face_recognition_sface_2021dec.onnx")
ANTISPOOF = os.path.join(HERE, "models", "antispoof", "AntiSpoofing_bin_1.5_128.onnx")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")


class FacePipe:
    def __init__(self, liveness=True):
        self.det = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), 0.6, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(SFACE, "")
        self.spoof = None
        if liveness and os.path.exists(ANTISPOOF):
            from avis.liveness import AntiSpoof
            self.spoof = AntiSpoof(ANTISPOOF)

    def embed_aligned(self, img):
        """Embedding of an already-aligned face photo (the crops AVIS saves in dataset/)."""
        if img is None or img.size == 0:
            return None
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        emb = self.rec.feature(img).reshape(-1).astype(np.float32)
        return emb / max(np.linalg.norm(emb), 1e-9)

    def largest_face(self, frame):
        """-> (embedding, live_score or None, box) for the biggest face, or None."""
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda r: float(r[2]) * float(r[3]))
        if f[2] < 40 or f[3] < 40:
            return None
        emb = self.rec.feature(self.rec.alignCrop(frame, f)).reshape(-1).astype(np.float32)
        emb /= max(np.linalg.norm(emb), 1e-9)
        box = (int(f[0]), int(f[1]), int(f[0] + f[2]), int(f[1] + f[3]))
        live = self.spoof.score(frame, box) if self.spoof is not None else None
        return emb, live, box


def iter_photos(folder):
    """All .jpg/.png under folder, including the per-run subfolders."""
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(IMAGE_EXT):
                img = cv2.imread(os.path.join(root, f))
                if img is not None:
                    yield os.path.relpath(os.path.join(root, f), folder), img


def iter_media(folder, step=3):
    """Yield (file, frame) for every image and every `step`-th video frame in folder."""
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXT:
            img = cv2.imread(path)
            if img is not None:
                yield name, img
        elif ext in VIDEO_EXT:
            cap = cv2.VideoCapture(path)
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % step == 0:
                    yield name, frame
                i += 1
            cap.release()
