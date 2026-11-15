"""
Event log + alerts.

Every event goes to runs/events.csv. Unknown-person and spoof events also save a
snapshot to runs/snapshots/ and, if you set one up in config.yaml, are pushed to
your phone (ntfy) or a Discord channel. Sending happens on a background thread so
the video never waits on the network. Old snapshots are deleted automatically.
"""
import csv
import json
import logging
import os
import queue
import threading
import time
import urllib.request
import uuid

import cv2

ALERT_KINDS = {"unknown", "spoof", "auto_lock", "challenge_failed", "password_failed"}
SNAPSHOT_KINDS = {"unknown", "spoof", "challenge_failed"}


class EventLog:
    def __init__(self, runs_dir, ntfy_topic="", discord_webhook="", alert_cooldown_s=60, snapshot_days=7):
        self.runs_dir = runs_dir
        self.snap_dir = os.path.join(runs_dir, "snapshots")
        os.makedirs(self.snap_dir, exist_ok=True)
        self.path = os.path.join(runs_dir, "events.csv")
        new = not os.path.exists(self.path)
        self.f = open(self.path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(["time", "event", "name", "similarity", "live_score", "snapshot"])
            self.f.flush()
        self.ntfy = (ntfy_topic or "").strip()
        self.discord = (discord_webhook or "").strip()
        self.cooldown = float(alert_cooldown_s)
        self._last = {}
        self._q = queue.Queue()
        self._worker = threading.Thread(target=self._send_loop, daemon=True)
        self._worker.start()
        self._purge_old(snapshot_days)
        if self.ntfy or self.discord:
            logging.info("[EVENTS] alerts on: %s", ", ".join(x for x, on in (("ntfy", self.ntfy), ("Discord", self.discord)) if on))

    def _purge_old(self, days):
        if not days or days <= 0:
            return
        cutoff = time.time() - days * 86400
        removed = 0
        for f in os.listdir(self.snap_dir):
            p = os.path.join(self.snap_dir, f)
            try:
                if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                    os.remove(p)
                    removed += 1
            except OSError:
                pass
        if removed:
            logging.info("[EVENTS] deleted %d snapshots older than %d days", removed, days)

    def emit(self, kind, name="", sim=None, live=None, frame=None, box=None, key=None, cooldown_s=None):
        """Log an event. Same key within cooldown_s is ignored (prevents spam)."""
        now = time.time()
        key = key or (kind, name)
        cd = self.cooldown if cooldown_s is None else cooldown_s
        if now - self._last.get(key, 0) < cd:
            return False
        self._last[key] = now

        snap = ""
        if kind in SNAPSHOT_KINDS and frame is not None:
            img = frame.copy()
            if box is not None:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            snap = os.path.join(self.snap_dir, time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{kind}.jpg")
            cv2.imwrite(snap, img)

        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.w.writerow([stamp, kind, name or "",
                         ("" if sim is None else round(float(sim), 4)),
                         ("" if live is None else round(float(live), 4)),
                         (os.path.relpath(snap, self.runs_dir) if snap else "")])
        self.f.flush()
        who = (name or "").replace("_", " ")
        text = {
            "seen": f"{who} recognized",
            "unknown": "Unknown person in view",
            "spoof": f"Possible spoof attempt{(' (looks like ' + who + ')') if who else ''} - photo or screen?",
            "left": f"{who} left",
            "auto_lock": f"PC locked: {who} away",
            "verified": f"{who} verified (gesture check passed)",
            "challenge_failed": f"Gesture check failed for '{who}' - photo or screen?",
            "unlocked": f"{who} unlocked with gesture password",
            "password_failed": f"Wrong gesture password attempts ({who})",
        }.get(kind, kind)
        logging.info("[EVENT] %s", text)
        if kind in ALERT_KINDS and (self.ntfy or self.discord):
            self._q.put((text, snap))
        return True

    # ---------- senders ----------
    def _send_loop(self):
        while True:
            text, snap = self._q.get()
            if self.ntfy:
                self._try(self._send_ntfy, text, snap)
            if self.discord:
                self._try(self._send_discord, text, snap)

    @staticmethod
    def _try(fn, text, snap):
        try:
            fn(text, snap)
        except Exception as e:
            logging.warning("[EVENTS] alert failed (%s): %s", fn.__name__, e)

    def _send_ntfy(self, text, snap):
        url = self.ntfy if "://" in self.ntfy else f"https://ntfy.sh/{self.ntfy}"
        if snap and os.path.exists(snap):
            data = open(snap, "rb").read()
            headers = {"Title": "AVIS", "Message": text, "Filename": os.path.basename(snap), "Tags": "rotating_light"}
        else:
            data, headers = text.encode("utf-8"), {"Title": "AVIS", "Tags": "rotating_light"}
        urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers, method="POST"), timeout=10)

    def _send_discord(self, text, snap):
        boundary = uuid.uuid4().hex
        parts = [(f'--{boundary}\r\nContent-Disposition: form-data; name="payload_json"\r\n'
                  f'Content-Type: application/json\r\n\r\n{json.dumps({"content": "**AVIS:** " + text})}\r\n').encode()]
        if snap and os.path.exists(snap):
            parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="files[0]"; '
                          f'filename="{os.path.basename(snap)}"\r\nContent-Type: image/jpeg\r\n\r\n').encode()
                         + open(snap, "rb").read() + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        req = urllib.request.Request(self.discord, data=b"".join(parts), method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                                              "User-Agent": "AVIS"})
        urllib.request.urlopen(req, timeout=10)

    def close(self):
        try:
            deadline = time.time() + 5
            while not self._q.empty() and time.time() < deadline:
                time.sleep(0.1)
            self.f.close()
        except Exception:
            pass
