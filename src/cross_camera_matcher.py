"""Appearance-based global identity matching across camera streams."""
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.spatial.distance import cosine

class CrossCameraMatcher:
    def __init__(self, match_threshold: float = 0.65):
        self.match_threshold = float(match_threshold)
        self.gallery: Dict[int, Tuple[int, np.ndarray]] = {}  # gid -> (class_id, embedding)
        self._next_global_id = 1

    @staticmethod
    def _similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - cosine(a, b))

    def match_or_register(self, embedding: np.ndarray, class_id: int = -1) -> Tuple[int, float]:
        best_id: Optional[int] = None
        best_score = -1.0
        for gid, (gallery_class, gallery_emb) in self.gallery.items():
            # Never match a person to a vehicle (or one vehicle class to another).
            if class_id != -1 and gallery_class != -1 and gallery_class != class_id:
                continue
            score = self._similarity(embedding, gallery_emb)
            if score > best_score:
                best_id, best_score = gid, score

        if best_id is not None and best_score >= self.match_threshold:
            gallery_class, gallery_emb = self.gallery[best_id]
            updated = 0.9 * gallery_emb + 0.1 * embedding
            updated /= max(np.linalg.norm(updated), 1e-12)
            self.gallery[best_id] = (gallery_class, updated)
            return best_id, best_score

        new_id = self._next_global_id
        self.gallery[new_id] = (class_id, embedding.copy())
        self._next_global_id += 1
        return new_id, 1.0
