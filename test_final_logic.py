import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add project root to sys.path
sys.path.append(r"G:\project\yolo11-chest_editted\YOLO_PROJECT")

from ultralytics import YOLO
from YOLO_PROJECT.pipeline.dicom import load_series, window_normalize
from YOLO_PROJECT.pipeline.medsam import MedSAMSegmenter

def test_final_logic():
    print("Starting final logic test...", flush=True)
    
    # 1. Load DICOM
    dicom_dir = r"G:\project\yolo11-chest_editted\lung_1\sysucc lung cancer more than 4\10015239+20150119+CT\a_2mm"
    print(f"Loading DICOM from: {dicom_dir}", flush=True)
    series, spacing, meta = load_series(dicom_dir)
    print(f"DICOM loaded: {series.shape}", flush=True)
    
    idx = 118
    hu = series[idx]
    
    # Save a slice to verify later if needed
    img = window_normalize(hu)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Memory optimization
    del series
    import gc
    gc.collect()
    
    # 2. YOLO (CPU)
    yolo_path = r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\runs\lung_nodule_lung1_v6_cpu\weights\best.pt"
    print(f"Loading YOLO: {yolo_path}", flush=True)
    det = YOLO(yolo_path)
    r = det.predict(source=[bgr], imgsz=640, conf=0.01, device="cpu", verbose=False)
    
    if len(r) and len(r[0].boxes):
        confs = r[0].boxes.conf.cpu().numpy()
        print(f"Detected {len(confs)} boxes with confidences: {confs}", flush=True)
        boxes = r[0].boxes.xyxy.cpu().numpy().astype(np.float32)
    else:
        print("No detections even at 0.01 conf.", flush=True)
        return
        
    box = boxes[0]
    print(f"Detected box: {box}", flush=True)
    
    # 3. MedSAM (XPU)
    medsam_path = r"G:\project\yolo11-chest_editted\sam_b.pt"
    print(f"Loading MedSAM on XPU: {medsam_path}", flush=True)
    seg = MedSAMSegmenter(medsam_path, model_type="vit_b", device="xpu")
    
    print("Running MedSAM prediction...", flush=True)
    mask = seg.predict(bgr, box)
    print(f"MedSAM finished. Mask sum: {np.sum(mask)}", flush=True)
    print("FINAL LOGIC SUCCESS!", flush=True)

if __name__ == "__main__":
    test_final_logic()
