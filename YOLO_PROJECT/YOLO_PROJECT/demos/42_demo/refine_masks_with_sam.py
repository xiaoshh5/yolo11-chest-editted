"""
refine_masks_with_sam.py

流程：
1. 使用已有 YOLO 检测模型（指定 path_to_detector，默认为 runs/*/best.pt）对图片进行检测，得到 bbox。
2. 使用 Segment Anything (SAM) 以 bbox 作为 prompt 生成多个掩码，选择得分最高的 mask。
3. 将生成的精细掩码保存到 `masks_sam/<split>/` 中，并输出叠加可视化到 `masks_sam_vis/<split>/`。
4. 输出 CSV 统计：image, box_id, lesion_pixels, bbox_pixels, image_pixels, ratio_bbox, ratio_image

运行示例：
    python 42_demo\refine_masks_with_sam.py --data_root "G:/project/yolo11-chest/YOLO_PROJECT/lung_1_YOLO_3" --det_weights "runs/train16/best.pt" --sam_checkpoint "sam_vit_h_4b8939.pth"

依赖：
    pip install ultralytics pillow numpy pandas
    pip install git+https://github.com/facebookresearch/segment-anything.git
    或按照 Segment Anything 官方安装说明安装

注意：如果没有 SAM checkpoint，请先下载合适的权重并指定 `--sam_checkpoint`。
"""

import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd

# ultralytics
from ultralytics import YOLO

# try import segment_anything
try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception as e:
    raise ImportError(
        "segment_anything not installed or import failed.\n"
        "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git\n"
        "Then download a SAM checkpoint (e.g., sam_vit_h_4b8939.pth) and pass --sam_checkpoint."
    ) from e


def find_default_detector():
    # search runs/*/best.pt
    runs_dir = Path('runs')
    if not runs_dir.exists():
        return None
    for d in sorted(runs_dir.iterdir(), reverse=True):
        p = d / 'best.pt'
        if p.exists():
            return str(p)
    return None


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def bbox_area(box):
    x1,y1,x2,y2 = box
    return max(0, int((x2-x1))) * max(0, int((y2-y1)))


def save_mask(mask, out_path):
    # mask: HxW boolean or 0/1
    m = (mask.astype(np.uint8) * 255)
    Image.fromarray(m).save(out_path)


def save_overlay(img_np, mask, out_path):
    img = Image.fromarray(img_np.astype(np.uint8)).convert('RGBA')
    mask_img = Image.fromarray((mask.astype(np.uint8)*255)).convert('L')
    red = Image.new('RGBA', img.size, (255,0,0,120))
    img.paste(red, (0,0), mask_img)
    img.convert('RGB').save(out_path)


def main(args):
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    # splits
    splits = ['train', 'val']

    # detector
    det_weights = args.det_weights or find_default_detector()
    if det_weights is None:
        raise FileNotFoundError('Cannot find detector weights automatically; provide --det_weights')
    print('Using detector:', det_weights)
    det_model = YOLO(det_weights)

    # SAM
    if not args.sam_checkpoint:
        raise FileNotFoundError('Please provide --sam_checkpoint for SAM weights')
    sam_checkpoint = args.sam_checkpoint
    # choose model type by filename
    if 'vit_h' in sam_checkpoint:
        model_type='vit_h'
    elif 'vit_b' in sam_checkpoint:
        model_type='vit_b'
    else:
        model_type='vit_l'  # fallback
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)

    out_root = Path(args.out_dir or (data_root / 'masks_sam'))
    out_vis_root = Path(args.out_vis_dir or (data_root / 'masks_sam_vis'))
    ensure_dir(out_root)
    ensure_dir(out_vis_root)

    records = []

    for split in splits:
        imgs_dir = data_root / split / 'images'
        if not imgs_dir.exists():
            print('Skip missing split:', split)
            continue
        masks_out_dir = out_root / split
        vis_out_dir = out_vis_root / split
        ensure_dir(masks_out_dir)
        ensure_dir(vis_out_dir)

        img_files = sorted([p for p in imgs_dir.iterdir() if p.suffix.lower() in ['.jpg','.png','.jpeg']])
        print(f'Processing {len(img_files)} images in {split}...')
        for img_p in img_files:
            img_np = np.array(Image.open(img_p).convert('RGB'))
            h,w = img_np.shape[:2]
            # run detector
            results = det_model.predict(source=str(img_p), conf=args.conf, max_det=args.max_det, verbose=False)
            res = results[0]
            # boxes in xyxy
            try:
                boxes = res.boxes.xyxy.cpu().numpy()
            except Exception:
                boxes = np.empty((0,4))
            if boxes.size == 0:
                # no detections -> skip or create empty file
                continue
            predictor.set_image(img_np)
            for i,box in enumerate(boxes):
                x1,y1,x2,y2 = box.astype(int)
                # clamp
                x1, y1 = max(0,x1), max(0,y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)
                if x2<=x1 or y2<=y1:
                    continue
                input_box = np.array([x1,y1,x2,y2]).reshape(1,4)
                masks, scores, logits = predictor.predict(box=input_box, multimask_output=True)
                # masks: (3, H, W) boolean
                if masks is None or masks.shape[0]==0:
                    continue
                # choose mask with largest area intersect bbox
                best_idx = 0
                best_score = -1
                for j in range(masks.shape[0]):
                    mask = masks[j]
                    # compute intersection with bbox
                    mask_crop = mask[y1:y2+1, x1:x2+1]
                    inter = mask_crop.sum()
                    if inter > best_score:
                        best_score = inter
                        best_idx = j
                best_mask = masks[best_idx]

                # save mask
                stem = img_p.stem + f"__box{i}"
                mask_out = masks_out_dir / (stem + '.png')
                save_mask(best_mask, mask_out)

                # save overlay
                vis_out = vis_out_dir / (stem + '.jpg')
                save_overlay(img_np, best_mask, vis_out)

                lesion_pixels = int(best_mask.sum())
                bbox_pixels = bbox_area([x1,y1,x2,y2])
                image_pixels = h*w
                ratio_bbox = lesion_pixels / bbox_pixels if bbox_pixels>0 else 0
                ratio_image = lesion_pixels / image_pixels
                records.append({
                    'image': str(img_p.relative_to(data_root)),
                    'box_id': i,
                    'mask_path': str(mask_out.relative_to(data_root)),
                    'vis_path': str(vis_out.relative_to(data_root)),
                    'lesion_pixels': lesion_pixels,
                    'bbox_pixels': bbox_pixels,
                    'image_pixels': image_pixels,
                    'ratio_bbox': ratio_bbox,
                    'ratio_image': ratio_image,
                })

    # save csv
    df = pd.DataFrame(records)
    csv_out = out_root / 'sam_refined_stats.csv'
    df.to_csv(csv_out, index=False)
    print('Done. saved masks to', out_root)
    print('Stats csv:', csv_out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--det_weights', type=str, default=None)
    p.add_argument('--sam_checkpoint', type=str, default=None)
    p.add_argument('--out_dir', type=str, default=None)
    p.add_argument('--out_vis_dir', type=str, default=None)
    p.add_argument('--conf', type=float, default=0.2)
    p.add_argument('--max_det', type=int, default=100)
    args = p.parse_args()
    main(args)
