
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

def test_sam_simple():
    print("Testing SAM simple...")
    device = "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu"
    print(f"Device: {device}")
    
    # Use any of the .pth files we found
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
    test_sam_simple()
