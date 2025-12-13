# AVIS (AI Vision Identity System)

AVIS is a local, real-time computer vision project I built in Python. It detects people, finds faces inside the upper head region, tracks faces over time, and can optionally do opt-in recognition and anonymization. The goal of this project is to show strong system design, real-time behavior, and clean CLI tooling (not just a demo script).

## Features

- Real-time YOLOv8 person detection (Ultralytics)
- Head ROI cropping to focus face detection on the top portion of the person box
- Face detection using YuNet (OpenCV FaceDetectorYN)
- Tracking with Kalman filtering
  - Track persistence across short detection dropouts
  - Track state shows MEAS (measured) vs PRED (predicted)
- Optional identity recognition (opt-in only)
  - ArcFace embeddings (ONNX using OpenCV DNN)
  - Cosine similarity matching against enrolled templates
- Privacy mode
  - Pixelate all faces or only unknown faces
- Optional face crop collection (opt-in only)
- Optional metrics logging to CSV for evaluation

## Ethics

This project is designed for ethical, opt-in use only.
- No scraping real people
- No covert surveillance tools
- Datasets are user-provided with explicit consent
- Anonymization exists to support privacy-first usage

## Requirements

- macOS or Windows 11+
- Python 3.12.x
- OpenCV, NumPy, Ultralytics

Install:
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install ultralytics opencv-python numpy
Models
YuNet face detector ONNX (default path used by the script):

models/yunet/face_detection_yunet_2023mar.onnx

ArcFace embeddings ONNX (optional, you provide the file path):

Use --arcface-model /full/path/to/arcface.onnx

Quick usage
Webcam:

bash
Copy code
python3 realtime_yolo.py --camera 0 --no-collect
Video file:

bash
Copy code
python3 realtime_yolo.py --video testvideo.mp4 --no-collect
Lower CPU usage (runs face detector less often):

bash
Copy code
python3 realtime_yolo.py --camera 0 --no-collect --detect-every 2
Anonymize everyone:

bash
Copy code
python3 realtime_yolo.py --camera 0 --no-collect --anonymize all
Anonymize unknown faces only (best demo if embeddings are enabled):

bash
Copy code
python3 realtime_yolo.py --camera 0 --no-collect --anonymize unknown --embeddings \
  --arcface-model /full/path/to/arcface.onnx --enroll-dir enrollments
Opt-in face collection:

bash
Copy code
python3 realtime_yolo.py --camera 0 --collect-label keanu --save-every 2
Write metrics CSV:

bash
Copy code
python3 realtime_yolo.py --camera 0 --no-collect --eval-csv runs/metrics.csv
Helpful flags
--camera 0 or --video path.mp4

--target-width 960

--conf 0.25

--detect-every 1

--max-age 18

--iou-thresh 0.20

--anonymize none|all|unknown

--embeddings --arcface-model ... --enroll-dir ... --id-thresh 0.45

--eval-csv runs/metrics.csv

Project layout (recommended)
realtime_yolo.py

models/

dataset/ (local, opt-in data)

enrollments/ (local, opt-in enrolled identities)

runs/ (local outputs like metrics CSV)