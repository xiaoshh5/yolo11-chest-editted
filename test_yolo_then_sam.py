
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

def test_yolo_then_sam():
    print("Testing YOLO then SAM on XPU...")
    device = "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu"
    print(f"Device: {device}")
    
    # 1. Init YOLO
    yolo_path = r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\runs\lung_nodule_lung1_v6_cpu\weights\best.pt"
    print(f"Loading YOLO: {yolo_path}")
    det = YOLO(yolo_path)
    # Run a dummy prediction on CPU
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    print("Running YOLO dummy predict (CPU)...")
    det.predict(img, device='cpu', verbose=False)
    print("YOLO dummy predict finished.")
    
    # 2. Init SAM
    sam_checkpoint = r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\runs\medsam_train_cpu\weights\best.pth"
    print(f"Loading SAM: {sam_checkpoint}")
    model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    model.to(device=device)
    predictor = SamPredictor(model)
    print("SAM loaded.")
    
    print("Calling SAM predictor.set_image...")
    predictor.set_image(img[:512, :512, :])
    print("SAM set_image finished.")
    
    box = np.array([100, 100, 200, 200])
    print("Calling SAM predictor.predict...")
    masks, _, _ = predictor.predict(box=box, multimask_output=False)
    print(f"SAM predict finished. Mask shape: {masks.shape}")

if __name__ == "__main__":
    test_yolo_then_sam()
