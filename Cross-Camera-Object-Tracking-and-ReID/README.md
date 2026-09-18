# Real-Time Multi-Object Tracking + Cross-Camera Re-Identification

Computer Vision pipeline using **YOLOv8 + DeepSORT + ResNet-50 ReID**.

## Structure

```text
SHRISTY CV GIT/
├── cli.py
├── configs/
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── detector.py
│   ├── tracker.py
│   ├── reid.py
│   ├── cross_camera_matcher.py
│   └── utils.py
├── tests/
│   └── test_basic.py
├── scripts/
│   └── download_weights.sh
├── data/raw/
├── data/processed/
├── weights/
├── outputs/
├── requirements.txt
├── .gitignore
└── README_cv.md
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads YOLOv8 and ResNet-50 weights if they are not already cached.

## Run

Single camera:

```bash
python cli.py --config configs/config.yaml --inputs data/raw/cam1.mp4 --camera-ids cam1 --output-dir outputs/run1
```

Two cameras:

```bash
python cli.py --config configs/config.yaml --inputs data/raw/cam1.mp4 data/raw/cam2.mp4 --camera-ids cam1 cam2 --output-dir outputs/run2
```

Quick test:

```bash
python cli.py --config configs/config.yaml --inputs data/raw/cam1.mp4 --camera-ids cam1 --output-dir outputs/smoke --max-frames 20
```

Run tests:

```bash
pytest tests/ -v
```

## Important

The ReID gallery is appearance-based. A global ID is a matching hypothesis, not a guaranteed real-world identity. For a convincing cross-camera demonstration, use two videos containing the same subjects and similar capture conditions.
