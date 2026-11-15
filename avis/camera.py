"""
Video sources: webcam index, IP camera URL (rtsp://, http://), or a video file.

Live sources (webcams, IP cams) are read on a background thread that always keeps
only the newest frame, so a slow frame never builds up lag. IP cameras reconnect
automatically if the stream drops. Files are read frame by frame (nothing skipped).
"""
import logging
import sys
import threading
import time

import cv2


def _webcam_opener(index, width=None, height=None, fps=None):
    def opener():
        if sys.platform == "win32":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif sys.platform == "darwin":
            backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]
        for be in backends:
            cap = cv2.VideoCapture(index, be)
            if cap.isOpened():
                if width and height:
                    # MJPG lets USB webcams deliver HD at full frame rate (raw YUY2 is often capped at 640x480)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
                if fps:
                    cap.set(cv2.CAP_PROP_FPS, float(fps))
                ok, _ = cap.read()
                if ok:
                    return cap
            cap.release()
        return None
    return opener


def _url_opener(url):
    def opener():
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return opener


class LiveCapture:
    def __init__(self, opener, name, reconnect=False, read_timeout=5.0):
        self.opener = opener
        self.name = name
        self.reconnect = reconnect
        self.read_timeout = read_timeout
        self.cap = opener()
        if self.cap is None:
            raise RuntimeError(f"Could not open {name}")
        self._lock = threading.Condition()
        self._frame = None
        self._seq = 0
        self._last_read = 0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        fails = 0
        while not self._stop:
            ok, frame = self.cap.read() if self.cap is not None else (False, None)
            if ok:
                fails = 0
                with self._lock:
                    self._frame = frame
                    self._seq += 1
                    self._lock.notify_all()
                continue
            fails += 1
            if self.reconnect and fails >= 30:
                logging.warning("[CAM] %s dropped; reconnecting...", self.name)
                try:
                    if self.cap is not None:
                        self.cap.release()
                except Exception:
                    pass
                time.sleep(2.0)
                self.cap = self.opener()
                fails = 0
            else:
                time.sleep(0.01)

    def read(self):
        """Newest frame not returned before (waits for it). (False, None) on timeout."""
        deadline = time.time() + self.read_timeout
        with self._lock:
            while self._seq == self._last_read and not self._stop:
                left = deadline - time.time()
                if left <= 0:
                    return False, None
                self._lock.wait(left)
            self._last_read = self._seq
            return True, (self._frame.copy() if self._frame is not None else None)

    def release(self):
        self._stop = True
        self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


class FileCapture:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


def open_source(src, width=None, height=None, fps=None):
    """src: int / '0' (webcam), 'rtsp://...' / 'http://...' (IP camera), or a file path.
    width/height/fps: requested webcam mode (the camera picks the closest it supports)."""
    s = str(src).strip()
    if s.lstrip("-").isdigit():
        idx = int(s)
        try:
            cap = LiveCapture(_webcam_opener(idx, width, height, fps), f"camera {idx}", reconnect=False, read_timeout=5.0)
            got = (int(cap.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            logging.info("[CAM] camera %d opened at %dx%d", idx, *got)
            return cap
        except RuntimeError:
            raise RuntimeError(
                f"Could not open camera {idx}. Close other apps using the webcam (Zoom, Discord, Teams, "
                f"Camera app), check Windows Settings > Privacy > Camera, or try source: 1 in config.yaml."
            )
    if "://" in s:
        return LiveCapture(_url_opener(s), "IP camera", reconnect=True, read_timeout=20.0)
    return FileCapture(s)
