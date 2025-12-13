AVIS (AI Vision Identity System)

AVIS is a local, real-time computer vision project I built in Python as a CS student.
It detects people, finds faces in the upper head region, tracks them over time, and can optionally attach extra signals like emotion, head pose/gaze, and ReID (ArcFace) for multi-camera identity continuity.

The point of this repo isn’t “look I ran YOLO once” — it’s to show system design, real-time behavior, clean CLI tooling, and features you’d actually build in a real pipeline.

---

## What it does

Pipeline (high level):
1) YOLOv8 detects people
2) AVIS crops a head ROI from each person box (top portion)
3) YuNet detects the face + 5 landmarks in that ROI
4) A tracker maintains IDs across frames (Kalman or DeepSORT)
5) Optional modules add:
   - emotion label (FER+ via ONNXRuntime)
   - head pose (yaw/pitch/roll) + simple gaze direction
   - ReID embeddings (ArcFace ONNX) + fusion across cameras
6) Optional CSV metrics logging for evaluation

---

## Features

- Real-time YOLOv8 person detection (Ultralytics)
- Head ROI cropping (keeps face detection focused + faster)
- Face detection via YuNet (OpenCV FaceDetectorYN)
- Tracking
  - Kalman tracker (fast, stable)
  - DeepSORT (appearance-based tracking)
  - Track state output: MEAS (measured) vs PRED (predicted)
- Expression / Emotion (optional)
  - FER+ model via ONNXRuntime
- Head pose + gaze (optional)
  - yaw/pitch/roll estimate + left/center/right gaze approximation
- ReID + Fusion (optional)
  - ArcFace embeddings (ONNX)
  - multi-camera global ID linking using cosine similarity
- Privacy mode
  - pixelate all faces or only unknown faces (when identity logic is enabled)
- Opt-in face crop collection (for building datasets)
- Metrics CSV logging
  - FPS, detections, tracks, etc.

---

## Ethics / Privacy

This is built for ethical, opt-in use only.
- No covert surveillance
- No scraping real people
- Any collection / identity-related features are meant for consenting users
- Includes anonymization modes to support privacy-first demos

---

## Requirements

- macOS or Windows 11+
- Python 3.12.x
- Webcam (optional)

---

## Setup

1) Create venv + install deps
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
Heads up: OpenCV 4.12+ on Python 3.12 expects NumPy 2.x.
If you previously installed MediaPipe, it may conflict (MediaPipe wants NumPy < 2). If needed:

bash
Copy code
python3 -m pip uninstall -y mediapipe
Models
YuNet (face detector)
Expected at:
models/yunet/face_detection_yunet_2023mar.onnx

ArcFace (ReID / Fusion)
Download it:

bash
Copy code
mkdir -p models/arcface
curl -L -o models/arcface/arcface.onnx \
  "https://huggingface.co/FoivosPar/Arc2Face/resolve/main/arcface.onnx"
Emotion model (FER+)
Expected at:
models/emotion/emotion-ferplus-8.onnx

If your file is actually HTML by mistake, re-download it. ONNX files should not show as “HTML document” in file output.

Quick usage
Basic (single camera)

bash
Copy code
python3 realtime_yolo.py --camera 0 --no-collect
Video file

bash
Copy code
python3 realtime_yolo.py --video testvideo.mp4 --no-collect
Lower CPU (run face detector less often)

bash
Copy code
python3 realtime_yolo.py --camera 0 --no-collect --detect-every 2
Main demo commands (with CSV output)
Kalman tracking (fast baseline)

bash
Copy code
python3 realtime_yolo.py \
  --camera 0 \
  --tracker kalman \
  --conf 0.25 \
  --detect-every 1 \
  --no-collect \
  --eval-csv runs/metrics_kalman.csv
DeepSORT tracking

bash
Copy code
python3 realtime_yolo.py \
  --camera 0 \
  --tracker deepsort \
  --conf 0.25 \
  --detect-every 1 \
  --no-collect \
  --eval-csv runs/metrics_deepsort.csv
Emotion / expression (FER+)

bash
Copy code
python3 realtime_yolo.py \
  --camera 0 \
  --tracker kalman \
  --conf 0.25 \
  --detect-every 1 \
  --expression \
  --emotion-model models/emotion/emotion-ferplus-8.onnx \
  --emotion-every 2 \
  --no-collect \
  --eval-csv runs/metrics_emotion.csv
Pose + gaze

bash
Copy code
python3 realtime_yolo.py \
  --camera 0 \
  --tracker kalman \
  --conf 0.25 \
  --detect-every 1 \
  --pose --gaze \
  --pose-every 2 \
  --no-collect \
  --eval-csv runs/metrics_pose_gaze.csv
ReID (ArcFace embeddings)

bash
Copy code
python3 realtime_yolo.py \
  --camera 0 \
  --tracker kalman \
  --conf 0.25 \
  --detect-every 2 \
  --reid --arcface-model models/arcface/arcface.onnx \
  --reid-every 4 \
  --no-collect \
  --eval-csv runs/metrics_reid.csv
Fusion (multi-cam global IDs)
Note: macOS may only expose one webcam index by default. If camera 1 fails, run single-cam.

bash
Copy code
python3 realtime_yolo.py \
  --camera 0 --camera 1 \
  --tracker deepsort \
  --conf 0.25 \
  --detect-every 2 \
  --reid --arcface-model models/arcface/arcface.onnx \
  --fusion --reid-thresh 0.45 \
  --reid-every 4 \
  --no-collect \
  --eval-csv runs/metrics_fusion.csv
All combined (DeepSORT + ReID + Fusion + Emotion + Pose/Gaze)

bash
Copy code
python3 realtime_yolo.py \
  --camera 0 \
  --tracker deepsort \
  --conf 0.25 \
  --detect-every 1 \
  --pose --gaze --pose-every 2 \
  --expression --emotion-model models/emotion/emotion-ferplus-8.onnx --emotion-every 2 \
  --reid --arcface-model models/arcface/arcface.onnx --reid-every 4 \
  --fusion --reid-thresh 0.45 \
  --no-collect \
  --eval-csv runs/metrics_all.csv
Project layout
realtime_yolo.py
collect_faces.py
collect_faces_from_video.py
models/
dataset/ (local opt-in data)
runs/ (metrics CSVs, snapshots, etc.)