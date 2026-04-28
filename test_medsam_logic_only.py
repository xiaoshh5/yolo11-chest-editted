import os
import sys
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

# Add project root to sys.path
sys.path.append(r"G:\project\yolo11-chest_editted\YOLO_PROJECT")

# import SimpleITK as sitk
from ultralytics import YOLO

def load_series_local(path: str):
    import SimpleITK as sitk
    from pathlib import Path
    p = Path(path)
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(p))
    if not series_ids:
        files = sorted([str(x) for x in p.glob("*.dcm")])
        if not files:
            return None, None, None
        reader.SetFileNames(files)
    else:
        files = reader.GetGDCMSeriesFileNames(str(p), series_ids[0])
        reader.SetFileNames(files)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    return arr, spacing, {"origin": origin, "direction": direction}

def window_normalize_local(arr: np.ndarray, wl_low: int = -1000, wl_high: int = 400):
    a = np.clip(arr, wl_low, wl_high)
    a = (a - wl_low) / (wl_high - wl_low)
    a = (a * 255.0).astype(np.uint8)
    return a

def test_logic():
    print("Starting MedSAM logic test (no PyQt)...", flush=True)
    # import torch
    # torch.set_num_threads(1)
    # print("Set torch threads to 1", flush=True)
    
    # 1. Load Real DICOM
    dicom_dir = r"G:\project\yolo11-chest_editted\lung_1\sysucc lung cancer more than 4\10015239+20150119+CT\a_2mm"
    print(f"Loading DICOM from: {dicom_dir}", flush=True)
    series, spacing, meta = load_series_local(dicom_dir)
    print(f"DICOM loaded: {series.shape}", flush=True)
    
    idx = 118
    hu = series[idx]
    
    # NEW: Delete series to save memory
    del series
    import gc
    gc.collect()
    
    img = window_normalize_local(hu)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"Image processed: {rgb.shape}, dtype={rgb.dtype}", flush=True)
    
    # 2. YOLO Detection (CPU)
    yolo_path = r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\runs\lung_nodule_lung1_v6_cpu\weights\best.pt"
    print(f"Loading YOLO: {yolo_path}", flush=True)
    det = YOLO(yolo_path)
    r = det.predict(source=bgr, imgsz=640, conf=0.01, device="cpu", verbose=False)
    boxes = r[0].boxes.xyxy.cpu().numpy() if len(r) and len(r[0].boxes) else np.empty((0, 4))
    
    if boxes.shape[0] == 0:
        print("No boxes found! Using dummy box.", flush=True)
        box = np.array([170, 340, 178, 349], dtype=np.float32)
    else:
        box = boxes[0].astype(np.float32)
    
    print(f"Using box: {box}", flush=True)
    
    # 3. MedSAM Segmentation
    medsam_path = r"G:\project\yolo11-chest_editted\sam_b.pt"
    device = "xpu"
    
    print(f"\n--- Testing on {device.upper()} ---", flush=True)
    print(f"Loading SAM on {device}...", flush=True)
    model = sam_model_registry["vit_b"](checkpoint=medsam_path)
    model.to(device=device)
    predictor = SamPredictor(model)
    print("MedSAM loaded.", flush=True)
    
    print("Calling predictor.set_image...", flush=True)
    predictor.set_image(rgb)
    print("predictor.set_image finished.", flush=True)
    
    print(f"Calling predictor.predict with box: {box}", flush=True)
    masks, scores, logits = predictor.predict(box=box, multimask_output=True)
    print("predictor.predict finished.", flush=True)
    
    mask = masks[0].astype(np.uint8)
    print(f"Success on {device}! Mask sum: {np.sum(mask)}", flush=True)

if __name__ == "__main__":
    test_logic()
