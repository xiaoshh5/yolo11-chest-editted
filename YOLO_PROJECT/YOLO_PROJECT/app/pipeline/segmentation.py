from typing import Tuple, Optional
import numpy as np
from ultralytics import SAM, YOLO


class Segmenter:
    def __init__(self, model_path: str, mode: str = "sam"):
        self.mode = mode
        if mode == "sam":
            self.model = SAM(model_path)
        else:
            self.model = YOLO(model_path)

    def segment(self, image: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        if self.mode == "sam":
            x1, y1, x2, y2 = box
            res = self.model.predict(image, bboxes=[[x1, y1, x2, y2]], verbose=False)
            for r in res:
                if hasattr(r, "masks") and r.masks is not None and len(r.masks.data) > 0:
                    m = r.masks.data[0].cpu().numpy().astype(np.uint8)
                    return m
            return None
        else:
            res = self.model.predict(image, verbose=False)
            for r in res:
                if hasattr(r, "masks") and r.masks is not None and len(r.masks.data) > 0:
                    m = r.masks.data[0].cpu().numpy().astype(np.uint8)
                    return m
            return None
