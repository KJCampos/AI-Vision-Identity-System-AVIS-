README.md
# AVIS — AI Vision Identity System (Local, Real-Time)

AVIS is a **local-only, real-time** computer vision pipeline for **person detection → face detection → tracking → (optional) opt-in identity recognition → anonymization**.

This is built as a **portfolio-grade system** (not a toy): configurable CLI, real-time behavior controls, tracking state, privacy toggles, and evaluation logging.

## What it does

- **Real-time YOLOv8 person detection** (Ultralytics)
- **Head ROI cropping** to reduce false positives and speed up face detection
- **Face detection using YuNet (OpenCV FaceDetectorYN)** (more reliable than Haar)
- **Tracking with Kalman (default)**  
  - Track persistence across missed detections (occlusion tolerance)
  - Clear tracking state: **MEAS** (measured) vs **PRED** (predicted)
- **Optional embeddings + opt-in recognition**
  - ArcFace embeddings (ONNX via OpenCV DNN)
  - Cosine similarity matching vs enrolled templates
- **Privacy controls**
  - `--anonymize all` or `--anonymize unknown` (pixelation)
- **Dataset collection (opt-in only)**
  - Save face crops only on strong measurements
- **Evaluation harness**
  - Per-frame metrics logging to CSV

## Ethics / Safety

AVIS is designed for **ethical-only, opt-in** use:
- No scraping real people.
- No covert surveillance features.
- Enrollment and face datasets are **explicitly user-provided**.
- Anonymization modes are included to support privacy-first workflows.

## Requirements

- macOS + Python **3.12.x**
- `ultralytics`, `opencv-python`, `numpy`
- YuNet ONNX face detector model file

Optional:
- ArcFace ONNX model (for embeddings / ReID-style matching)

## Install

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install ultralytics opencv-python numpy


Note: If you use MediaPipe later, it requires numpy<2. Keep AVIS in its own venv.

Models
YuNet (Face Detector)

Place the YuNet ONNX here (default path used by the script):

models/yunet/face_detection_yunet_2023mar.onnx


Run with:

python3 realtime_yolo.py --camera 0 --no-collect

ArcFace (Optional Embeddings)

If using embeddings, provide an ArcFace ONNX file and point to it:

python3 realtime_yolo.py --camera 0 --no-collect --embeddings \
  --arcface-model /full/path/to/arcface.onnx \
  --enroll-dir enrollments

Quickstart Commands
Webcam (default tracking)
python3 realtime_yolo.py --camera 0 --no-collect

Video file
python3 realtime_yolo.py --video testvideo.mp4 --no-collect

Improve stability / speed tradeoffs

Run face detector less often (saves CPU, more prediction frames):

python3 realtime_yolo.py --camera 0 --no-collect --detect-every 2


Lower resize width (saves CPU):

python3 realtime_yolo.py --camera 0 --no-collect --target-width 800

Enable anonymization

Pixelate everyone:

python3 realtime_yolo.py --camera 0 --no-collect --anonymize all


Pixelate unknown only (best privacy-first demo):

python3 realtime_yolo.py --camera 0 --no-collect --anonymize unknown \
  --embeddings --arcface-model /full/path/to/arcface.onnx --enroll-dir enrollments

Opt-in face collection

Saves crops only on strong measurements (MEAS) at your chosen cadence:

python3 realtime_yolo.py --camera 0 --collect-label keanu --save-every 2

Metrics CSV (evaluation)

Writes per-frame stats (fps, tracks, MEAS/PRED counts, etc.):

python3 realtime_yolo.py --camera 0 --no-collect --eval-csv runs/metrics.csv

CLI Reference (most useful flags)

Input:

--camera 0 : webcam

--video path.mp4 : video file

Performance:

--target-width 960

--detect-every 1 : face detector cadence

Tracking:

--max-age 18

--iou-thresh 0.20

Collection:

--collect-label name

--no-collect

--save-every 2

Embeddings / Identity (opt-in):

--embeddings

--arcface-model /path/to/arcface.onnx

--enroll-dir enrollments

--id-thresh 0.45

Privacy:

--anonymize none|all|unknown

--pixel-blocks 14

Logging:

--log-interval 30

Repo Structure (suggested)
.
├── realtime_yolo.py
├── models/
│   └── yunet/
│       └── face_detection_yunet_2023mar.onnx
├── enrollments/               # opt-in user-provided face sets
├── dataset/                   # opt-in collected face crops
├── runs/                      # metrics outputs (csv)
└── README.md

Notes / Known Issues

If you add DeepSORT or MediaPipe, pin numpy accordingly (separate venv recommended).

Recognition is intended for opt-in enrolled identities only.

Emotions CLI feature is bugged. 