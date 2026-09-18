"""YOLOv8 object detection wrapper."""
from dataclasses import dataclass
from typing import List
import numpy as np
from ultralytics import YOLO

@dataclass
class Detection:
    xyxy: np.ndarray
    confidence: float
    class_id: int

class ObjectDetector:
    def __init__(self, model_name: str, confidence_threshold: float,
                 iou_threshold: float, classes: List[int], device: str = "cpu"):
        self.model = YOLO(model_name)
        self.conf = float(confidence_threshold)
        self.iou = float(iou_threshold)
        self.classes = classes
        self.device = device

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame, conf=self.conf, iou=self.iou,
            classes=self.classes, device=self.device, verbose=False,
        )
        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                detections.append(Detection(
                    xyxy=box.xyxy[0].detach().cpu().numpy().astype(float),
                    confidence=float(box.conf[0].detach().cpu().item()),
                    class_id=int(box.cls[0].detach().cpu().item()),
                ))
        return detections
