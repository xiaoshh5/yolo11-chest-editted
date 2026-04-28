
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor

def test_sam_cpu():
    print("Testing SAM on CPU...")
    device = "cpu"
    
    checkpoint = r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\runs\medsam_train_cpu\weights\best.pth"
    print(f"Loading checkpoint: {checkpoint}")
    
    model = sam_model_registry["vit_b"](checkpoint=checkpoint)
    model.to(device=device)
    predictor = SamPredictor(model)
    print("Model loaded.")
    
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    print("Setting image...")
    predictor.set_image(img)
    print("Image set successfully.")
    
    box = np.array([100, 100, 200, 200])
    print("Predicting...")
    masks, _, _ = predictor.predict(box=box, multimask_output=False)
    print(f"Predicted. Mask shape: {masks.shape}")

if __name__ == "__main__":
    test_sam_cpu()
