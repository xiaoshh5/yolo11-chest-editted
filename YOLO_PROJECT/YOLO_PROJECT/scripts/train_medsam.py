import argparse
from pathlib import Path
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

class LungSegDataset(Dataset):
    def __init__(self, root: str, split: str = "train", img_size: int = 1024, augment: bool = True):
        self.root = Path(root)
        self.img_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"
        self.names = [p.stem for p in self.img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg", ".bmp"}]
        self.transform = ResizeLongestSide(img_size)
        self.img_size = img_size
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
        self.augment = augment
    def __len__(self):
        return len(self.names)
    def __getitem__(self, i):
        name = self.names[i]
        ip = str(self.img_dir / f"{name}.jpg")
        if not os.path.exists(ip):
            ip = str(self.img_dir / f"{name}.png")
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        mp = str(self.mask_dir / f"{name}.png")
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            m = np.zeros((h, w), dtype=np.uint8)
        if self.augment:
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
                m = m[:, ::-1]
            if np.random.rand() < 0.5:
                img = img[::-1, :]
                m = m[::-1, :]
            if np.random.rand() < 0.3:
                a = 1.0 + (np.random.rand() - 0.5) * 0.3
                b = (np.random.rand() - 0.5) * 30.0
                img = cv2.convertScaleAbs(img, alpha=a, beta=b)
            if np.random.rand() < 0.2:
                ang = np.random.uniform(-10.0, 10.0)
                M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), ang, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                m = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        # box from mask tight rectangle
        ys, xs = np.where(m > 0)
        if len(xs) == 0 or len(ys) == 0:
            box = np.array([0, 0, w - 1, h - 1], dtype=np.float32)
        else:
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
        # SAM preprocess
        im_tr = self.transform.apply_image(img)
        im_t = torch.from_numpy(im_tr).permute(2, 0, 1).float().unsqueeze(0)
        im_t = (im_t - self.pixel_mean) / self.pixel_std
        # scale box to transformed coords
        scale_w = im_tr.shape[1] / w
        scale_h = im_tr.shape[0] / h
        b = box.copy()
        b[0] *= scale_w; b[2] *= scale_w
        b[1] *= scale_h; b[3] *= scale_h
        # mask downsample to 256x256
        m_t = torch.from_numpy((m > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        m_rs = F.interpolate(m_t, size=(256, 256), mode="nearest")
        return im_t.squeeze(0), torch.from_numpy(b), m_rs.squeeze(0)

def pick_device(dev_arg: str):
    if isinstance(dev_arg, str) and "xpu" in dev_arg and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

def dice_loss(pred_logits, gt_mask, eps=1e-6):
    pred = torch.sigmoid(pred_logits)
    inter = (pred * gt_mask).sum()
    union = pred.sum() + gt_mask.sum() + eps
    return 1.0 - (2.0 * inter / union)

def bce_loss(pred_logits, gt_mask):
    return F.binary_cross_entropy_with_logits(pred_logits, gt_mask)

def train(args):
    device = pick_device(args.device)
    print(f"[medsam] device={device}", flush=True)
    ds = LungSegDataset(args.data, split="train", img_size=1024, augment=bool(args.augment))
    print(f"[medsam] dataset_size={len(ds)} root={args.data}", flush=True)
    # Use batch_size=1 to avoid shape mismatches inside SAM mask decoder
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    print(f"[medsam] loading model {args.model_type}...", flush=True)
    model = sam_model_registry[args.model_type](checkpoint=args.checkpoint if args.checkpoint else None)
    print(f"[medsam] moving model to {device}...", flush=True)
    model.to(device=device)
    print(f"[medsam] model ready", flush=True)
    model.train()
    # Only mask decoder is trained, freeze others to save memory/speed
    model.image_encoder.eval()
    model.prompt_encoder.eval()
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    for param in model.prompt_encoder.parameters():
        param.requires_grad = False
        
    opt = torch.optim.AdamW(model.mask_decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    # write args for traceability
    try:
        with open(run_dir / "args.yaml", "w", encoding="utf-8") as f:
            f.write(f"device: {args.device}\n")
            f.write(f"epochs: {args.epochs}\n")
            f.write(f"batch: 1\n")
            f.write(f"model_type: {args.model_type}\n")
            f.write(f"checkpoint: {args.checkpoint}\n")
            f.write(f"lr: {args.lr}\n")
            f.write(f"data: {args.data}\n")
            f.write(f"project: {args.project}\n")
            f.write(f"name: {args.name}\n")
    except Exception as e:
        print(f"[medsam] failed to write args.yaml: {e}", flush=True)
    out_dir = run_dir / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[medsam] out_dir={out_dir}", flush=True)
    print(f"[medsam] dl length: {len(dl)}", flush=True)
    # touch a marker file so we can verify directory creation
    try:
        with open(run_dir / "started.txt", "w", encoding="utf-8") as f:
            f.write("ok")
        with open(out_dir / "started.txt", "w", encoding="utf-8") as f:
            f.write("ok")
    except Exception as e:
        print(f"[medsam] failed to write started.txt: {e}", flush=True)
    for epoch in range(args.epochs):
        print(f"[medsam] starting epoch {epoch+1}...", flush=True)
        total = 0.0
        steps = 0
        for im_t, box, m_rs in dl:
            try:
                im_t = im_t.to(device)
                box = box.to(device).float()
                m_rs = m_rs.to(device).float()
                
                with torch.no_grad():
                    emb = model.image_encoder(im_t)
                    sparse, dense = model.prompt_encoder(points=None, boxes=box.unsqueeze(1), masks=None)
                
                low_res_masks, _ = model.mask_decoder(
                    image_embeddings=emb,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False
                )
                l1 = bce_loss(low_res_masks, m_rs)
                l2 = dice_loss(low_res_masks, m_rs)
                loss = l1 + l2
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                steps += 1
                if steps % 1 == 0:
                    print(f"[medsam] epoch={epoch+1} step={steps} loss={loss.item():.4f}", flush=True)
                
                # Help XPU memory management
                if device.type == "xpu":
                    torch.xpu.empty_cache()

                if args.max_steps and steps >= int(args.max_steps):
                    print(f"[medsam] reached max_steps {args.max_steps}, breaking epoch", flush=True)
                    break
            except Exception as e:
                print(f"[medsam] step error: {e}", flush=True)
                break
        ck = out_dir / f"epoch{epoch+1}.pth"
        try:
            print(f"[medsam] saving epoch {epoch+1} weights to {ck}...", flush=True)
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state, str(ck))
            print(f"[medsam] saved {ck}", flush=True)
        except Exception as e:
            print(f"[medsam] save epoch failed: {e}", flush=True)
    
    print(f"[medsam] training loop finished", flush=True)
    final = out_dir / "best.pth"
    try:
        print(f"[medsam] saving final weights to {final}...", flush=True)
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state, str(final))
        print(f"[medsam] saved {final}", flush=True)
    except Exception as e:
        print(f"[medsam] save final failed: {e}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path("YOLO_PROJECT/YOLO_PROJECT/datasets/lung_1_YOLO_3").resolve()))
    ap.add_argument("--project", default=str(Path("YOLO_PROJECT/YOLO_PROJECT/runs").resolve()))
    ap.add_argument("--name", default="medsam_train")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--model_type", default="vit_b")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--augment", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=200)
    args = ap.parse_args()
    train(args)

if __name__ == "__main__":
    main()
