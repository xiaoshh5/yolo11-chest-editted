#!/usr/bin/env python3
"""Helper: convert a simple dataset into basic YOLO format.

This is a conservative helper that expects a source directory with two subdirs
`train` and `val`, each containing images and optionally labels. It will
create the structure expected by ultralytics/YOLO:

  <dst>/train/images/
  <dst>/train/labels/
  <dst>/val/images/
  <dst>/val/labels/

If label files are not present, it will create empty .txt label files so you
can manually annotate or use another tool.
"""
import os
import argparse
import shutil
from pathlib import Path


def is_image(fname):
    return fname.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def prepare(src, dst):
    src = Path(src)
    dst = Path(dst)
    for split in ('train', 'val'):
        src_images = src / split / 'images'
        src_labels = src / split / 'labels'

        out_images = dst / split / 'images'
        out_labels = dst / split / 'labels'
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        if not src_images.exists():
            print(f"Warning: {src_images} does not exist, skipping {split}")
            continue

        for f in src_images.iterdir():
            if not is_image(f):
                continue
            shutil.copy2(f, out_images / f.name)
            label_name = f.with_suffix('.txt').name
            src_label = src_labels / label_name
            dst_label = out_labels / label_name
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
            else:
                # create empty label file placeholder
                dst_label.write_text('')

    print(f'Prepared YOLO dataset in {dst}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='source dataset root (with train/val subfolders)')
    p.add_argument('--dst', required=True, help='destination dataset root (will be created)')
    args = p.parse_args()
    prepare(args.src, args.dst)


if __name__ == '__main__':
    main()
