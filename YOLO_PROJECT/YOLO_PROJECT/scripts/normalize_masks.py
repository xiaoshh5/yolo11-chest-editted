import os
import argparse
from pathlib import Path
import cv2
import numpy as np


def normalize_dir(mask_dir: Path):
    if not mask_dir.exists():
        return
    for f in mask_dir.iterdir():
        if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        bin_mask = (img > 0).astype(np.uint8)  # foreground -> 1, background -> 0
        cv2.imwrite(str(f), bin_mask)


def main():
    ap = argparse.ArgumentParser(description="Normalize segmentation masks to class index values (0 background, 1 foreground)")
    ap.add_argument("--root", required=True, help="Dataset root with train/masks and val/masks")
    args = ap.parse_args()
    root = Path(args.root)
    normalize_dir(root / "train" / "masks")
    normalize_dir(root / "val" / "masks")
    print(f"Normalized masks under {root}")


if __name__ == "__main__":
    main()
