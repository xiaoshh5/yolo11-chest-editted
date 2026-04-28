import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from YOLO_PROJECT.pipeline.dicom import load_series, window_normalize
from YOLO_PROJECT.pipeline.medsam import MedSAMSegmenter
from YOLO_PROJECT.pipeline.ctr import ctr_ratio

def pick_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dicom_dir", required=True)
    p.add_argument("--yolo_weights", required=True)
    p.add_argument("--medsam_weights", required=True)
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--save_dir", default=str(Path(__file__).resolve().parent.parent / "runs" / "medsam_test"))
    args = p.parse_args()
    arr, spacing, _ = load_series(args.dicom_dir)
    if arr is None:
        return
    z = arr.shape[0] // 2
    hu = arr[z]
    img = window_normalize(hu)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    device = pick_device()
    print(f"[test] Loading YOLO from {args.yolo_weights}...")
    det = YOLO(args.yolo_weights)
    print(f"[test] Predicting on slice {z}...")
    r = det.predict(source=[bgr], device=device, imgsz=640, conf=0.1, verbose=True)
    
    if not len(r) or len(r[0].boxes) == 0:
        print("[test] No detections found by YOLO even with conf=0.1. Pipeline stopped.")
        return
    
    boxes = r[0].boxes.xyxy.cpu().numpy().astype(int)
    conf = r[0].boxes.conf.cpu().numpy()
    print(f"[test] Found {len(boxes)} boxes. Best conf: {conf[0]:.4f}")
    
    box = boxes[0]
    print(f"[test] Loading MedSAM from {args.medsam_weights} on CPU...")
    seg = MedSAMSegmenter(args.medsam_weights, model_type=args.model_type, device="cpu")
    print(f"[test] Running MedSAM prediction...")
    m = seg.predict(bgr, box)
    print(f"[test] MedSAM finished. Mask sum: {np.sum(m)}")
    c = ctr_ratio(hu, m, solid_threshold=-300.0)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    overlay = bgr.copy()
    overlay[m > 0] = (0.2 * overlay[m > 0] + 0.8 * np.array([0, 0, 255])).astype(np.uint8)
    cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.putText(overlay, f"CTR:{c:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    out = Path(args.save_dir) / "overlay.jpg"
    cv2.imwrite(str(out), overlay)

if __name__ == "__main__":
    main()
