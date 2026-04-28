
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from YOLO_PROJECT.pipeline.dicom import load_series, window_normalize
from YOLO_PROJECT.pipeline.medsam import MedSAMSegmenter

def main():
    dicom_dir = r"G:\project\yolo11-chest_editted\lung_1\sysucc lung cancer more than 4\10015239+20150119+CT\a_2mm"
    weights = r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\runs\medsam_final_prod_xpu_aug_v2\weights\best.pth"  # MedSAM weights path
    
    print(f"Loading series from {dicom_dir}...")
    arr, spacing, _ = load_series(dicom_dir)
    if arr is None:
        print("Failed to load series")
        return
        
    # Pick a slice (e.g., 105)
    idx = 105
    hu = arr[idx]
    img = window_normalize(hu)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Define a manual box around the nodule (based on the label info roughly)
    # The label was around x=0.36, y=0.67 -> x=184, y=343
    # Let's use a 40x40 box around it
    box = np.array([164, 323, 204, 363]) 
    
    device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading MedSAM from {weights}...")
    seg = MedSAMSegmenter(weights, model_type="vit_b", device=device)
    
    print("Predicting...")
    mask = seg.predict(bgr, box)
    print(f"Mask sum: {np.sum(mask)}")
    
    # Save result
    save_dir = Path(r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\runs\verification")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    overlay = bgr.copy()
    overlay[mask > 0] = (0.3 * overlay[mask > 0] + 0.7 * np.array([0, 0, 255])).astype(np.uint8)
    cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    
    out_path = save_dir / "verify_slice_105.jpg"
    cv2.imwrite(str(out_path), overlay)
    print(f"Saved result to {out_path}")

if __name__ == "__main__":
    main()
