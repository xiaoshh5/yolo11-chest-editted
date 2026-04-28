import os
import argparse
from pathlib import Path
import cv2
import numpy as np


def write_seg_label(txt_path: Path, contours, img_w: int, img_h: int, cls_id: int = 0):
    lines = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        pts = cnt.reshape(-1, 2).astype(np.float32)
        pts[:, 0] = pts[:, 0] / img_w
        pts[:, 1] = pts[:, 1] / img_h
        coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in pts])
        lines.append(f"{cls_id} {coords}")
    txt_path.write_text("\n".join(lines))


def process_split(root: Path, split: str):
    img_dir = root / split / "images"
    mask_dir = root / split / "masks"
    lbl_dir = root / split / "labels"
    if not img_dir.exists() or not mask_dir.exists():
        return
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        name = img_path.stem
        mask_path = mask_dir / f"{name}.png"
        if not mask_path.exists():
            # skip if mask missing
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            continue
        h, w = img.shape
        binm = (m > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        txt_path = lbl_dir / f"{name}.txt"
        write_seg_label(txt_path, contours, w, h, cls_id=0)


def main():
    ap = argparse.ArgumentParser(description="Generate YOLO segmentation labels (.txt with polygon coords) from PNG masks")
    ap.add_argument("--root", required=True, help="Dataset root with train/images, train/masks, val/images, val/masks")
    args = ap.parse_args()
    root = Path(args.root)
    process_split(root, "train")
    process_split(root, "val")
    print(f"Generated segmentation labels under {root}")


if __name__ == "__main__":
    main()
