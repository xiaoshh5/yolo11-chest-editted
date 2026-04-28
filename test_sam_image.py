
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor

def test_sam_image():
    print("Testing SAM with real image...")
    device = "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu"
    print(f"Device: {device}")
    
    checkpoint = r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\runs\medsam_train_cpu\weights\best.pth"
    print(f"Loading checkpoint: {checkpoint}")
    
    model = sam_model_registry["vit_b"](checkpoint=checkpoint)
    model.to(device=device)
    predictor = SamPredictor(model)
    print("Model loaded.")
    
    # Create a dummy BGR image that looks like a real one
    img_bgr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    print(f"Image shape: {rgb.shape}, dtype: {rgb.dtype}")
    print("Calling predictor.set_image(rgb)...")
    try:
        predictor.set_image(rgb)
        print("predictor.set_image(rgb) finished.")
    except Exception as e:
        print(f"Error in set_image: {e}")
        return

    box = np.array([100, 100, 200, 200])
    print("Calling predictor.predict...")
    try:
        masks, _, _ = predictor.predict(box=box, multimask_output=False)
        print(f"Predict finished. Mask shape: {masks.shape}")
    except Exception as e:
        print(f"Error in predict: {e}")

if __name__ == "__main__":
    test_sam_image()
