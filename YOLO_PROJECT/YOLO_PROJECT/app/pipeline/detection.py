from pathlib import Path
from typing import List, Tuple
import numpy as np
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, image: np.ndarray, conf: float = 0.25) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        results = self.model.predict(image, conf=conf, verbose=False)
        out = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls = int(b.cls.item()) if b.cls is not None else -1
                score = float(b.conf.item()) if b.conf is not None else 0.0
                xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
                x1, y1, x2, y2 = xyxy
                out.append((cls, score, (x1, y1, x2, y2)))
        return out
