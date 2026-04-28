import sys
import os
import cv2
import numpy as np
from pathlib import Path
import torch

# Add the project path to sys.path
project_root = r"G:\project\yolo11-chest_editted\YOLO_PROJECT"
if project_root not in sys.path:
    sys.path.append(project_root)

sys.path.append(os.path.join(project_root, "YOLO_PROJECT"))

from app.pipeline.detection import Detector
from app.pipeline.segmentation import Segmenter

def verify():
    # Force CPU
    device = "cpu"
    
    model_path = r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\runs\lung_nodule_lung1_v6_cpu\weights\best.pt"
    seg_model_path = r"G:\project\yolo11-chest_editted\sam_b.pt"
    image_path = r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\datasets\lung_1_YOLO_3\train\images\10015239+20150119+CT_0105.jpg"

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    if not os.path.exists(seg_model_path):
        print(f"Segmentation model not found at {seg_model_path}")
        return
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        return

    print(f"Loading detector with model: {model_path}")
    detector = Detector(model_path)
    detector.model.to(device)
    
    print(f"Loading segmenter with model: {seg_model_path}")
    segmenter = Segmenter(seg_model_path, mode="sam")
    segmenter.model.to(device)

    print(f"Reading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image")
        return

    print("Running detection...")
    results = detector.predict(img, conf=0.01)

    if not results:
        print("No detections found.")
    else:
        print(f"Found {len(results)} detections:")
        for i, (cls, score, box) in enumerate(results):
            print(f"  [{i}] Class: {cls}, Score: {score:.4f}, Box: {box}")
            
            print(f"  Running segmentation for detection {i}...")
            # Ensure the segmenter uses the right device
            mask = segmenter.segment(img, box)
            if mask is not None:
                print(f"  Mask found! Mask shape: {mask.shape}, Sum: {np.sum(mask)}")
            else:
                print(f"  No mask found for detection {i}")

if __name__ == "__main__":
    verify()
