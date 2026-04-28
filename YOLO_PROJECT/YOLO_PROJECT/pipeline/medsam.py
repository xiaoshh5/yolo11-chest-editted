import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor

class MedSAMSegmenter:
    def __init__(self, weights_path: str, model_type: str = "vit_b", device: str = "cpu"):
        self.device = device
        self.weights_path = weights_path
        self.model = sam_model_registry[model_type](checkpoint=weights_path)
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)

    def predict(self, img_bgr: np.ndarray, box_xyxy: np.ndarray):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Ensure box is float32 and correct shape for SAM
        box = box_xyxy.astype(np.float32)
        
        self.predictor.set_image(rgb)
        # Use multimask_output=True as it proved more stable on XPU in tests
        masks, scores, logits = self.predictor.predict(box=box, multimask_output=True)
        
        # Pick the best mask
        best_idx = np.argmax(scores)
        m = masks[best_idx].astype(np.uint8)
        return m
