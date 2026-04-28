from pathlib import Path
import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor


class MedSAMSegmenter:
    def __init__(self, weights_path: str, model_type: str = "vit_b", device: str = "cpu"):
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"MedSAM weights not found: {weights_path}")
        if model_type not in sam_model_registry:
            raise ValueError(f"Unknown model type '{model_type}'. Available: {list(sam_model_registry.keys())}")

        self.device = device
        self.weights_path = str(weights_path)
        self.model = sam_model_registry[model_type](checkpoint=self.weights_path)
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)

    def predict(self, img_bgr: np.ndarray, box_xyxy: np.ndarray):
        """Run SAM prediction on a region.

        Args:
            img_bgr: BGR image (H, W, 3).
            box_xyxy: [x1, y1, x2, y2] detection box.

        Returns:
            Binary mask (H, W) as uint8.
        """
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(f"img_bgr must be (H, W, 3), got {img_bgr.shape}")

        box = np.asarray(box_xyxy, dtype=np.float32).flatten()
        if box.shape != (4,):
            raise ValueError(f"box_xyxy must be [x1, y1, x2, y2], got {box_xyxy}")

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        masks, scores, _logits = self.predictor.predict(box=box, multimask_output=True)

        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(np.uint8)
