import os
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm

def split_dataset(root, images_subdir="images", labels_subdir="labels", train_ratio=0.8, img_exts=(".jpg", ".png", ".jpeg")):
    images_dir = os.path.join(root, images_subdir)
    labels_dir = os.path.join(root, labels_subdir)
    if not os.path.exists(images_dir):
        print(f"Error: Images dir not found: {images_dir}")
        return
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(img_exts)]
    if not images:
        print("Warning: no images found for splitting")
    random.shuffle(images)
    train_count = int(len(images) * train_ratio)
    dirs = [
        os.path.join(root, "train", "images"),
        os.path.join(root, "train", "labels"),
        os.path.join(root, "val", "images"),
        os.path.join(root, "val", "labels"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"Total images: {len(images)}")
    print(f"Splitting -> Train: {train_count}, Val: {len(images) - train_count}")
    for i, img_name in enumerate(tqdm(images)):
        label_name = os.path.splitext(img_name)[0] + ".txt"
        src_img = os.path.join(images_dir, img_name)
        src_label = os.path.join(labels_dir, label_name)
        dst_root = os.path.join(root, "train" if i < train_count else "val")
        shutil.move(src_img, os.path.join(dst_root, "images", img_name))
        if os.path.exists(src_label):
            shutil.move(src_label, os.path.join(dst_root, "labels", label_name))
    print("Dataset split complete!")

def parse_args():
    default_root = Path(__file__).resolve().parents[1] / "datasets" / "lung_1_YOLO_3"
    ap = argparse.ArgumentParser(description="Split a flat images/labels dataset into train/val for YOLO")
    ap.add_argument("--root", default=str(default_root), help="Dataset root containing images/ and labels/")
    ap.add_argument("--images_subdir", default="images", help="Images subdirectory under root")
    ap.add_argument("--labels_subdir", default="labels", help="Labels subdirectory under root")
    ap.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio (0-1)")
    ap.add_argument("--img_exts", default=".jpg,.png,.jpeg", help="Comma-separated image extensions to include")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    img_exts = tuple(x.strip().lower() for x in args.img_exts.split(",") if x.strip())
    split_dataset(
        root=args.root,
        images_subdir=args.images_subdir,
        labels_subdir=args.labels_subdir,
        train_ratio=args.train_ratio,
        img_exts=img_exts or (".jpg",),
    )
