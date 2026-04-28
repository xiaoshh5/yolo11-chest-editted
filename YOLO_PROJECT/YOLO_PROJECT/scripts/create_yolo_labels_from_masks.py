
import os
import cv2
import numpy as np
from pathlib import Path

def mask_to_yolo(mask_path, label_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = mask.shape
    labels = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 5: # Skip very small noise
            continue
            
        # Get bounding box
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        # YOLO format: class x_center y_center width height (normalized)
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        width = bw / w
        height = bh / h
        
        labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    if labels:
        with open(label_path, "w") as f:
            f.write("\n".join(labels))
        return True
    return False

def main():
    dataset_root = Path(r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\datasets\lung_1_YOLO_3\train")
    mask_dir = dataset_root / "masks"
    label_dir = dataset_root / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    
    mask_files = list(mask_dir.glob("*.png"))
    print(f"Found {len(mask_files)} mask files.")
    
    count = 0
    for mask_path in mask_files:
        label_path = label_dir / (mask_path.stem + ".txt")
        if mask_to_yolo(mask_path, label_path):
            count += 1
            
    print(f"Created {count} YOLO label files.")

if __name__ == "__main__":
    main()
