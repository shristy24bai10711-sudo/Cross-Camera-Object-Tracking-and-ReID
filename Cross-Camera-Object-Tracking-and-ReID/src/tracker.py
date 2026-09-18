"""DeepSORT single-camera multi-object tracking wrapper."""
from typing import List, Tuple
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from src.detector import Detection

class MultiObjectTracker:
    def __init__(self, max_age: int, n_init: int, nms_max_overlap: float,
                 max_iou_distance: float, embedder: str = "mobilenet"):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=nms_max_overlap,
            max_iou_distance=max_iou_distance,
            embedder=embedder,
            half=False,
            bgr=True,
        )

    def update(self, frame: np.ndarray, detections: List[Detection]) -> List[Tuple]:
        formatted = []
        for det in detections:
            x1, y1, x2, y2 = map(float, det.xyxy)
            formatted.append(([x1, y1, x2 - x1, y2 - y1], float(det.confidence), int(det.class_id)))

        tracks = self.tracker.update_tracks(formatted, frame=frame)
        results = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            x1, y1, x2, y2 = track.to_ltrb()
            cls_id = track.get_det_class()
            if cls_id is None:
                cls_id = -1
            results.append((track.track_id, x1, y1, x2, y2, int(cls_id)))
        return results
