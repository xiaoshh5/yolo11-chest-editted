from ultralytics import YOLO
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--data", default=str(Path("configs/sample_data.yaml").resolve()))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default=str(Path("runs").resolve()))
    p.add_argument("--name", default="lung_nodule_exp")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--cache", type=bool, default=True)
    p.add_argument("--amp", type=bool, default=True)
    p.add_argument("--exist_ok", type=bool, default=True)
    args = p.parse_args()
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        amp=args.amp,
        project=args.project,
        name=args.name,
        patience=args.patience,
        exist_ok=args.exist_ok,
    )
    print(str(results.save_dir))

if __name__ == '__main__':
    main()
