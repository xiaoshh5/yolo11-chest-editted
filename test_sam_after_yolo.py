import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

def test():
    print("Testing SAM after YOLO loading...", flush=True)
    
    # 1. Load YOLO and RUN it
    yolo_path = r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\runs\lung_nodule_lung1_v6_cpu\weights\best.pt"
    print(f"Loading YOLO: {yolo_path}", flush=True)
    det = YOLO(yolo_path)
    print("YOLO loaded. Running predict...", flush=True)
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    det.predict(dummy_img, device="cpu", verbose=False)
    print("YOLO predict finished.", flush=True)
    
    # 2. Load SAM on XPU
    sam_checkpoint = r"G:\project\yolo11-chest_editted\sam_b.pt"
    device = "xpu"
    print(f"Loading SAM on {device}: {sam_checkpoint}", flush=True)
    model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    model.to(device=device)
    predictor = SamPredictor(model)
    print("SAM loaded.", flush=True)
    
    # 3. Predict
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    print("Calling set_image...", flush=True)
    predictor.set_image(img)
    print("set_image finished.", flush=True)
    
    print("Calling predict...", flush=True)
    box = np.array([100, 100, 200, 200])
    predictor.predict(box=box, multimask_output=False)
    print("predict finished.", flush=True)

if __name__ == "__main__":
    test()
