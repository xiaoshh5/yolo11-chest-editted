import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import time

def test_yolo_sam():
    print("YOLO + SAM test starting...", flush=True)
    
    # 1. YOLO
    yolo_path = r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\runs\lung_nodule_lung1_v6_cpu\weights\best.pt"
    print(f"Loading YOLO on CPU...", flush=True)
    yolo_model = YOLO(yolo_path)
    
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    print("Running YOLO prediction...", flush=True)
    yolo_model.predict(dummy_img, device="cpu", verbose=False)
    print("YOLO finished.", flush=True)
    
    # 2. SAM
    sam_checkpoint = r"G:\project\yolo11-chest_editted\sam_b.pt"
    device = "xpu"
    print(f"Loading SAM on {device}...", flush=True)
    sam_model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam_model.to(device=device)
    predictor = SamPredictor(sam_model)
    print("SAM loaded.", flush=True)
    
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    print("Calling SAM set_image...", flush=True)
    start = time.time()
    predictor.set_image(img)
    print(f"SAM set_image finished in {time.time() - start:.2f}s", flush=True)
    
    print("Success!", flush=True)

if __name__ == "__main__":
    test_yolo_sam()
