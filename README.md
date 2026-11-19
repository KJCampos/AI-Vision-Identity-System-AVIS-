# AVIS — AI Vision Identity System

AVIS is a real-time computer-vision system I built in Python. It runs locally on a webcam (or an IP camera / video file), detects people, recognizes enrolled faces by name, verifies they're live with a hand-gesture challenge, and keeps learning each person's face over time with a model trained on the GPU.

The same project also ships two hand-tracking apps: **Hand 3D** (draw, measure and spin 3D shapes in the air) and **Touchless Control** (run slides, music and volume with gestures).

Everything runs on your own machine. No face data leaves the PC, and none of it is in this repo.

---

## Features

**Detection and recognition**
- YOLOv8 person detection on the GPU (CUDA, half precision), with YuNet face detection on a head region of each person
- Face recognition with SFace embeddings and a quality gate (face size, detector score, both eyes visible) so blurry or side-on faces aren't misjudged
- A second check from a small identity network trained with PyTorch: AVIS only names someone when both agree, which cuts false matches
- Unknown faces can be blurred or pixelated automatically

**Spoof protection**
- **Gesture challenge:** when AVIS recognizes someone, it asks for a random gesture (peace sign, thumbs up, open hand, fist, point). A printed photo or phone screen can't answer, and repeated failures raise an alert with a snapshot
- **Optional gesture password:** a secret sequence of gestures that unlocks a command of your choice
- Optional anti-spoofing model (off by default)

**Self-learning**
- High-confidence, verified sightings are added to that person's face profile
- Photos are saved into per-session folders (`dataset/<name>/<date time> <run>/`)
- The identity network retrains automatically on the GPU after each session

**Events, data and tools**
- Event log with optional phone (ntfy) or Discord alerts and snapshots
- Optional actions: lock Windows when the owner leaves, run a command when someone is recognized
- Per-session CSV logs, cleaned and graphed with matplotlib (`analyze_runs.py`)
- Accuracy report on your own labeled clips: FAR, FRR, EER and spoof error rates, with recommended thresholds (`evaluate.py`)
- Local web dashboard (127.0.0.1 only, token-protected) for people, events, graphs and settings
- Recording to readable file names (`recordings/AVIS Trial N.mp4`)
- Optional extras: emotion, head pose / gaze, ArcFace ReID, DeepSORT tracking

**Hand apps (MediaPipe hand tracking)**
- **Hand 3D** (`hand3d.py`): point and hold to draw; circles become spheres, squares boxes, triangles pyramids. Pinch to grab, spread thumb and index to resize, swipe to spin with real momentum and friction, fist to delete. Shows each shape's real-world size. Sessions record to `recordings/Hand3D/`
- **Touchless Control** (`control.py`): swipe for next/previous slide, pinch-hold to play/pause, "L" hand up/down for volume, thumbs up to mute, hold a fist to pause gesture control

---

## Project layout

```
realtime_yolo.py      main AVIS app (live camera, recognition, verification, recording)
avis/                 camera sources, face pipeline, learning, hands, events, actions, config
config.yaml           all settings (command-line flags override it)
dashboard.py          local web dashboard
train_identity.py     GPU training for the identity network
evaluate.py           accuracy report on labeled clips
record_clip.py        record labeled test clips into eval/
analyze_runs.py       clean session CSVs and draw graphs
hand3d.py             Hand 3D app
control.py            Touchless Control app
setup_and_run.bat     one-click setup + launch (Windows)
tools_*.bat           GPU diagnostics, CUDA PyTorch install
```

---

## Setup (Windows)

Requirements: Windows 10/11, Python 3.11+, a webcam. An NVIDIA GPU is optional but recommended.

1. Clone the repo and open the folder.
2. Run **`setup_and_run.bat`**. It creates a `.venv`, installs `requirements.txt` and starts AVIS. To only install, run `setup_and_run.bat --setup-only`.
3. For GPU speed, run **`tools_install_cuda.bat`** to install the CUDA build of PyTorch, then **`tools_diag.bat`** to confirm the GPU is detected.

Model files (YOLOv8n, YuNet, SFace, MediaPipe hand landmarker) are downloaded or placed under `models/`. Large and personal model files are git-ignored.

> If pip hangs on a dead package index, your global pip config is the cause. The batch files already work around it by ignoring that config.

---

## Usage

Activate the environment first: `.venv\Scripts\activate`

| What | Command |
|---|---|
| Live camera | `python realtime_yolo.py` |
| Enroll a person (with their consent) | `python realtime_yolo.py --enroll "Name"` |
| Live camera + record | `python realtime_yolo.py --record` |
| Use an IP camera or file | `python realtime_yolo.py --source rtsp://...` |
| Graph the last session | `python analyze_runs.py` |
| Dashboard | `python dashboard.py` |
| Record a labeled test clip | `python record_clip.py <Name\|unknown\|spoof>` |
| Accuracy report | `python evaluate.py` (add `--apply` to save recommended thresholds) |
| Train identity network | `python train_identity.py` |
| Hand 3D | `python hand3d.py` |
| Touchless Control | `python control.py` |

Only one app can use the webcam at a time.

**Settings:** edit `config.yaml` or use the dashboard. Useful ones:
- `spoof_protection.verify`: `gesture` (default), `model` or `none`
- `spoof_protection.gesture_password`: e.g. `"peace,fist,thumbs_up"`
- `privacy.anonymize`: blur unknown faces
- `events.ntfy_topic` / `events.discord_webhook`: alerts (leave empty to stay offline)
- `actions.auto_lock`: lock Windows when the owner leaves

---

## Privacy

- Recognition runs entirely on the local machine. AVIS does not look anyone up online; people it doesn't know stay "Unknown".
- Only enroll people who agree to it. The dashboard's **Forget** button deletes a person's face data.
- Face profiles (`faces_db/`), photos (`dataset/`), test clips (`eval/`), recordings, logs and trained models are excluded by `.gitignore` and never committed.
- The dashboard only listens on 127.0.0.1 and needs a per-session token to change anything.

---

## Tech stack

Python · OpenCV (YuNet, SFace) · Ultralytics YOLOv8 · PyTorch (CUDA) · ONNX Runtime · MediaPipe · NumPy · matplotlib · PyYAML
