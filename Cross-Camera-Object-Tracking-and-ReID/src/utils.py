"""Shared configuration, video, drawing, and CSV helpers."""
import csv
import os
from typing import List, Tuple, Optional
import cv2
import numpy as np
import yaml

CLASS_NAMES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def open_video_writer(output_path: str, fps: float, width: int, height: int):
    ensure_dir(os.path.dirname(output_path) or ".")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")
    return writer

def draw_tracks(frame: np.ndarray, tracks: List[Tuple], global_ids: Optional[dict] = None) -> np.ndarray:
    annotated = frame.copy()
    for track_id, x1, y1, x2, y2, cls_id in tracks:
        gid = global_ids.get(track_id, track_id) if global_ids else track_id
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
        cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        label = f"Global:{gid} Local:{track_id} {cls_name}"
        cv2.putText(annotated, label, (p1[0], max(p1[1] - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return annotated

def init_csv_writer(csv_path: str):
    ensure_dir(os.path.dirname(csv_path) or ".")
    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["frame_idx", "camera_id", "track_id", "global_id", "x1", "y1", "x2", "y2", "class_id"])
    return f, writer

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
