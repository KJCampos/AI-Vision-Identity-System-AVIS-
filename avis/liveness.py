"""
Spoof protection: is this a real face, or a photo / phone screen held up to the camera?

Model: MiniFASNet-style CNN trained on CelebA-Spoof (hairymax/Face-AntiSpoofing,
AntiSpoofing_bin_1.5_128.onnx). Input = 1.5x square crop around the face, 128x128 RGB.
Output = [real, spoof] probabilities.
"""
import logging
import os

import cv2
import numpy as np


def _increased_crop(img, bbox, inc=1.5):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    side = max(w, h) * inc
    xc, yc = x1 + w / 2.0, y1 + h / 2.0
    x, y = int(xc - side / 2), int(yc - side / 2)
    s = int(side)
    cx1, cy1, cx2, cy2 = max(0, x), max(0, y), min(W, x + s), min(H, y + s)
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    return cv2.copyMakeBorder(crop, cy1 - y, (y + s) - cy2, cx1 - x, (x + s) - cx2,
                              cv2.BORDER_CONSTANT, value=[0, 0, 0])


class AntiSpoof:
    def __init__(self, model_path, size=128):
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError("Spoof protection needs onnxruntime: pip install onnxruntime")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Anti-spoof model not found at {model_path}")
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(model_path, providers=prov)
        self.inp = self.sess.get_inputs()[0].name
        self.size = int(size)
        logging.info("[LIVE] anti-spoof model loaded (%s)", self.sess.get_providers()[0])

    def score(self, frame_bgr, face_box):
        """Probability (0-1) that the face in face_box (x1,y1,x2,y2) is a real, live face."""
        crop = _increased_crop(frame_bgr, face_box)
        if crop is None:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        r = self.size / max(h, w)
        nh, nw = max(1, int(h * r)), max(1, int(w * r))
        rgb = cv2.resize(rgb, (nw, nh))
        dh, dw = self.size - nh, self.size - nw
        rgb = cv2.copyMakeBorder(rgb, dh // 2, dh - dh // 2, dw // 2, dw - dw // 2,
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
        x = rgb.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
        logits = self.sess.run(None, {self.inp: x})[0].reshape(-1).astype(np.float64)
        e = np.exp(logits - logits.max())
        return float((e / e.sum())[0])
