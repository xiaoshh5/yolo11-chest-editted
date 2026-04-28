
import sys
import os
import torch
import numpy as np
import cv2
from pathlib import Path

# Set PYTHONPATH
sys.path.append(r"G:\project\yolo11-chest_editted\YOLO_PROJECT")

from ultralytics import YOLO
from YOLO_PROJECT.pipeline.dicom import load_series, window_normalize
from YOLO_PROJECT.pipeline.medsam import MedSAMSegmenter
from YOLO_PROJECT.pipeline.ctr import ctr_ratio

def test_pipeline():
    print("Starting Pipeline Test...")
    
    # 1. Load DICOM
    dicom_dir = r"G:\project\yolo11-chest_editted\lung_1\sysucc lung cancer more than 4\10015239+20150119+CT\a_2mm"
    print(f"Loading DICOM from: {dicom_dir}")
    arr, spacing, meta = load_series(dicom_dir)
    if arr is None:
        print("Failed to load DICOM series.")
        return
    print(f"Loaded series with shape: {arr.shape}, spacing: {spacing}")
    
    # 2. Prepare Image (middle slice)
    idx = arr.shape[0] // 2
    hu = arr[idx]
    img = window_normalize(hu)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    print(f"Prepared image from slice {idx}. Shape: {bgr.shape}, Dtype: {bgr.dtype}, Min: {bgr.min()}, Max: {bgr.max()}", flush=True)
    
    # 3. Load YOLO
    yolo_path = r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\runs\lung_nodule_lung1_v6_cpu\weights\best.pt"
    print(f"Loading YOLO model: {yolo_path}")
    device = "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu"
    print(f"Using device: {device}")
    det = YOLO(yolo_path)
    
    # 4. Predict
    print(f"Running YOLO detection on CPU (avoiding XPU hang)...")
    r = det.predict(source=[bgr], imgsz=640, conf=0.01, device='cpu', verbose=True)
    print("YOLO prediction call finished.")
    boxes = r[0].boxes.xyxy.cpu().numpy().astype(int) if len(r) and len(r[0].boxes) else np.empty((0, 4), dtype=int)
    
    if boxes.shape[0] == 0:
        print("No nodules detected.")
        return
    
    print(f"Detected {len(boxes)} nodules. Picking the first one: {boxes[0]}")
    box = boxes[0]
    
    # 5. Load MedSAM
    medsam_path = r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\runs\medsam_train_cpu\weights\best.pth"
    print(f"Loading MedSAM model: {medsam_path}")
    print(f"Using device: {device}")
    seg = MedSAMSegmenter(medsam_path, model_type="vit_b", device=device)
    
    # 6. Segment
    print("Running MedSAM segmentation (predictor.set_image)...", flush=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    seg.predictor.set_image(rgb)
    print("Running MedSAM segmentation (predictor.predict)...", flush=True)
    masks, scores, _ = seg.predictor.predict(box=box, multimask_output=False)
    m = masks[0].cpu().numpy().astype(np.uint8) if hasattr(masks[0], "cpu") else masks[0].astype(np.uint8)
    print(f"Segmentation mask created. Sum of pixels: {np.sum(m)}", flush=True)
    
    # 7. CTR Ratio
    print("Calculating CTR ratio...", flush=True)
    c = ctr_ratio(hu, m, solid_threshold=-300.0, spacing=(spacing[0], spacing[1]))
    print(f"Calculated CTR: {c:.4f}", flush=True)
    
    print("Pipeline test completed successfully!", flush=True)

if __name__ == "__main__":
    test_pipeline()
