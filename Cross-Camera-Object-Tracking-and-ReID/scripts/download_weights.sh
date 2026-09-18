#!/usr/bin/env bash
set -euo pipefail

python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python -c "import torchvision.models as m; m.resnet50(weights=m.ResNet50_Weights.DEFAULT)"
echo "YOLOv8 and ResNet-50 weights downloaded to the libraries' normal cache locations."
