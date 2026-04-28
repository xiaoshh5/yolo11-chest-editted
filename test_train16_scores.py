
import sys
import os
import torch
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

# Set PYTHONPATH to include the project directory
project_root = r"G:\project\yolo11-chest_editted\EXPORT_GG0_PROJECT\YOLO_PROJECT"
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from the actual app structure
from pipeline.dicom import load_series, window_normalize

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    mn, mx = np.min(a), np.max(a)
    if mx > mn:
        a = (a - mn) / (mx - mn)
    else:
        a = np.zeros_like(a)
    a = (a * 255.0).clip(0, 255).astype(np.uint8)
    return a

def test():
    # 1. Paths
    dicom_dir = r"G:\project\yolo11-chest_editted\lung_1\sysucc lung cancer more than 4\10015239+20150119+CT\a_2mm"
    yolo_path = r"G:\project\yolo11-chest_editted\EXPORT_GG0_PROJECT\weights\yolo_best.pt"
    
    print(f"Checking yolo_path: {os.path.exists(yolo_path)}")
    print(f"Checking dicom_dir: {os.path.exists(dicom_dir)}")
    
    print(f"Loading DICOM from: {dicom_dir}")
    arr, spacing, _ = load_series(dicom_dir)
    if arr is None:
        print("Failed to load DICOM")
        return
    
    # Slice 105 is known to have a nodule in some previous tests
    idx = 105
    hu = arr[idx]
    
    # 2. Prepare images with different normalization
    img_window = window_normalize(hu)
    bgr_window = cv2.cvtColor(img_window, cv2.COLOR_GRAY2BGR)
    
    img_minmax = normalize_to_uint8(hu)
    bgr_minmax = cv2.cvtColor(img_minmax, cv2.COLOR_GRAY2BGR)
    
    # 3. Load Model
    print(f"Loading YOLO model from: {yolo_path}")
    model = YOLO(yolo_path)
    
    # 4. Test Window Normalization
    print("\n--- Testing with Window Normalization (-1000 to 400) ---")
    results_w = model.predict(source=[bgr_window], imgsz=640, conf=0.01, device='cpu', verbose=False)
    boxes_w = results_w[0].boxes.xyxy.cpu().numpy() if len(results_w) and len(results_w[0].boxes) else []
    print(f"Detected {len(boxes_w)} boxes.")
    for b in boxes_w:
        print(f"  Box: {b}")

    # 5. Test Min-Max Normalization
    print("\n--- Testing with Min-Max Normalization (Per Slice) ---")
    results_m = model.predict(source=[bgr_minmax], imgsz=640, conf=0.01, device='cpu', verbose=False)
    boxes_m = results_m[0].boxes.xyxy.cpu().numpy() if len(results_m) and len(results_m[0].boxes) else []
    print(f"Detected {len(boxes_m)} boxes.")
    for b in boxes_m:
        print(f"  Box: {b}")

    # 6. Try different slices if none found
    if len(boxes_w) == 0 and len(boxes_m) == 0:
        print("\nNo boxes found in slice 105. Searching other slices...")
        for i in range(0, arr.shape[0], 10):
            hu_i = arr[i]
            img_i = normalize_to_uint8(hu_i)
            bgr_i = cv2.cvtColor(img_i, cv2.COLOR_GRAY2BGR)
            res_i = model.predict(source=[bgr_i], imgsz=640, conf=0.1, device='cpu', verbose=False)
            if len(res_i[0].boxes) > 0:
                print(f"Found something in slice {i} with Min-Max!")
                break

if __name__ == "__main__":
    test()
