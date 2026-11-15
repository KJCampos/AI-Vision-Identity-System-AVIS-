"""
control.py - AVIS Touchless Control: run slides, music and volume with hand gestures.

Gestures (sent as real key presses to whatever app is in front: PowerPoint, Google Slides,
YouTube, Spotify, VLC...):
  SWIPE LEFT  (open hand, quick sweep)   next slide   (Right arrow)
  SWIPE RIGHT                            previous slide (Left arrow)
  PINCH + HOLD half a second             play / pause (media key)
  "L" HAND, move it UP / DOWN            volume up / down (media keys)
  THUMBS UP (hold 1 s)                   mute / unmute
  FIST (hold 1.5 s)                      pause / resume gesture control (so random moves do nothing)

A small always-on-top preview shows what it sees. Click your presentation/player once after
starting so the key presses go to it. Q in the preview window quits.
Uses the webcam, so close AVIS Live Camera / Hand 3D first.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from avis.camera import open_source  # noqa: E402
from avis.hands import HandTracker, StableGesture, classify, features  # noqa: E402

VK = {"next": 0x27, "prev": 0x25, "play_pause": 0xB3, "vol_up": 0xAF, "vol_down": 0xAE, "mute": 0xAD}
LABEL = {"next": "NEXT >", "prev": "< PREVIOUS", "play_pause": "PLAY / PAUSE", "vol_up": "VOLUME +",
         "vol_down": "VOLUME -", "mute": "MUTE", "pause": "GESTURES PAUSED", "resume": "GESTURES ON"}


def send_key(action):
    if sys.platform != "win32":
        return
    import ctypes
    vk = VK[action]
    ext = 0x1 if vk in (0x25, 0x27) else 0           # arrows are "extended" keys
    ctypes.windll.user32.keybd_event(vk, 0, ext, 0)
    ctypes.windll.user32.keybd_event(vk, 0, ext | 0x2, 0)


class Controller:
    def __init__(self, sender=send_key):
        self.send = sender
        self.armed = True
        self.hist = []                 # (t, x, y) palm positions while hand is open
        self.last_swipe = 0.0
        self.stable = {}
        self.hold = {}                 # gesture -> since
        self.fired = set()             # holds that already fired (until released)
        self.vol_anchor = None
        self.toast = ("GESTURES ON", time.time() + 1.5)
        self.log = []

    def act(self, action, now):
        self.log.append(action)
        self.toast = (LABEL[action], now + 1.0)
        if action in VK:
            self.send(action)

    def held(self, g, now, secs, key=None):
        key = key or g
        t0 = self.hold.setdefault(key, now)
        if now - t0 >= secs and key not in self.fired:
            self.fired.add(key)
            return True
        return False

    def update(self, hands, W, H, now=None):
        """hands: list of (label, 21x2 points). Uses the most prominent (largest) hand."""
        now = time.time() if now is None else now
        if not hands:
            self.hold.clear(); self.fired.clear(); self.hist.clear(); self.vol_anchor = None
            return None
        label, pts = max(hands, key=lambda h: float(np.linalg.norm(h[1][5] - h[1][17])))
        f = features(pts)
        sg = self.stable.setdefault("main", StableGesture(3))
        g = sg.update(classify(f, sg.state))
        for k in list(self.hold):              # release holds of other gestures
            if k != g:
                self.hold.pop(k, None); self.fired.discard(k)

        if g == "fist" and self.held("fist", now, 1.5):
            self.armed = not self.armed
            self.act("resume" if self.armed else "pause", now)
        if not self.armed:
            return g

        # swipe: open hand travels fast sideways
        if g == "open":
            self.hist = [h for h in self.hist if now - h[0] < 0.35] + [(now, float(f["palm_c"][0]), float(f["palm_c"][1]))]
            dx = self.hist[-1][1] - self.hist[0][1]
            dy = self.hist[-1][2] - self.hist[0][2]
            if abs(dx) > 0.28 * W and abs(dx) > 2.0 * abs(dy) and now - self.last_swipe > 0.8:
                self.act("next" if dx < 0 else "prev", now)   # mirrored view: dx<0 = hand moved to your left
                self.last_swipe, self.hist = now, []
        else:
            self.hist.clear()

        if g == "pinch" and self.held("pinch", now, 0.5):
            self.act("play_pause", now)
        if g == "thumbs_up" and self.held("thumbs_up", now, 1.0):
            self.act("mute", now)

        # volume: L-shaped hand, move up/down; one key press per step
        if g == "l_shape":
            y = float(f["pinch_pt"][1])
            if self.vol_anchor is None:
                self.vol_anchor = y
            step = 0.05 * H
            while y < self.vol_anchor - step:
                self.act("vol_up", now); self.vol_anchor -= step
            while y > self.vol_anchor + step:
                self.act("vol_down", now); self.vol_anchor += step
        else:
            self.vol_anchor = None
        return g


def main():
    ap = argparse.ArgumentParser(description="AVIS Touchless Control")
    ap.add_argument("--source", default="0")
    ap.add_argument("--preview-width", type=int, default=480)
    a = ap.parse_args()
    print(__doc__)
    tracker = HandTracker()
    ctl = Controller()
    cam = open_source(a.source, width=1280, height=720, fps=30)
    win = "AVIS Touchless Control"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    sized = False
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]
            hands = tracker(frame)
            g = ctl.update(hands, W, H)
            view = cv2.resize(frame, (a.preview_width, int(H * a.preview_width / W)))
            s = a.preview_width / W
            for _, pts in hands:
                for q in pts[[0, 4, 8, 12, 16, 20]] * s:
                    cv2.circle(view, (int(q[0]), int(q[1])), 4, (0, 215, 255), -1, cv2.LINE_AA)
            col = (80, 255, 80) if ctl.armed else (80, 80, 255)
            cv2.rectangle(view, (0, 0), (view.shape[1], 28), (20, 20, 20), -1)
            cv2.putText(view, ("ACTIVE" if ctl.armed else "PAUSED (hold fist to resume)") + (f"  |  {g}" if g else ""),
                        (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            text, until = ctl.toast
            if time.time() < until:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                x, y = (view.shape[1] - tw) // 2, view.shape[0] - 20
                cv2.rectangle(view, (x - 10, y - th - 10), (x + tw + 10, y + 8), (0, 0, 0), -1)
                cv2.putText(view, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 230, 150), 2, cv2.LINE_AA)
            if not sized:
                cv2.resizeWindow(win, view.shape[1], view.shape[0])
                cv2.moveWindow(win, 20, 20)
                try:
                    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
                except cv2.error:
                    pass
                sized = True
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cam.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
