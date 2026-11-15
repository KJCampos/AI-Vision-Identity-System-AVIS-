"""
record_clip.py - record a short RAW clip (no boxes/labels) for testing accuracy or training.

    python record_clip.py KJ          -> eval/KJ/clip_N.mp4        (an enrolled person)
    python record_clip.py unknown     -> eval/unknown/clip_N.mp4   (someone NOT enrolled, with their OK)
    python record_clip.py spoof       -> eval/spoof/clip_N.mp4     (hold up a photo / phone of an enrolled person)

Counts down 3 seconds, records 10 seconds (change with --seconds). Q stops early.
"""
import argparse
import os
import re
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avis.camera import open_source  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", help="enrolled name, 'unknown', or 'spoof'")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--source", default="0")
    a = ap.parse_args()
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", a.label.strip()).strip("_") or "unknown"
    if label.lower() in ("unknown", "spoof"):
        label = label.lower()
    folder = os.path.join(HERE, "eval", label)
    os.makedirs(folder, exist_ok=True)
    n = 1 + max([int(m.group(1)) for f in os.listdir(folder) for m in [re.match(r"clip_(\d+)\.mp4$", f)] if m], default=0)
    out = os.path.join(folder, f"clip_{n}.mp4")

    cam = open_source(a.source)
    writer, t_start, frames = None, None, 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            left = 3 - (time.time() - t0)
            view = frame.copy()
            if left > 0:
                cv2.putText(view, f"{label}: recording in {left:.0f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
            else:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
                    t_start = time.time()
                writer.write(frame)
                frames += 1
                el = time.time() - t_start
                cv2.circle(view, (30, 40), 10, (0, 0, 255), -1)
                cv2.putText(view, f"REC {label} {el:.0f}/{a.seconds:.0f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                if el >= a.seconds:
                    break
            cv2.imshow("AVIS clip recorder", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cam.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    print(f"Saved {frames} frames -> {out}" if frames else "Nothing recorded.")


if __name__ == "__main__":
    main()
