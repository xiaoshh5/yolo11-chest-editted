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
    default_project = Path(__file__).resolve().parent.parent / "runs"
    p.add_argument("--project", default=str(default_project))
    p.add_argument("--name", default="lung_nodule_exp")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--cache", action="store_true", help="Cache images for faster training")
    p.add_argument("--no_cache", action="store_false", dest="cache", help="Do not cache images")
    p.set_defaults(cache=True)
    p.add_argument("--amp", action="store_true", help="Use AMP")
    p.add_argument("--no_amp", action="store_false", dest="amp", help="Disable AMP")
    p.set_defaults(amp=False)
    p.add_argument("--exist_ok", action="store_true", help="Allow existing project/name")
    p.add_argument("--no_exist_ok", action="store_false", dest="exist_ok", help="Do not allow existing project/name")
    p.set_defaults(exist_ok=True)
    p.add_argument("--resume", action="store_true", help="Resume training from last.pt")
    p.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    p.add_argument("--warmup_epochs", type=float, default=3.0, help="Warmup epochs")
    p.add_argument("--optimizer", default="auto", help="Optimizer to use")
    args = p.parse_args()

    model = YOLO(args.model)
    
    # 如果是恢复训练，我们显式传递参数而不是依赖 resume=True，
    # 这样可以覆盖原来的 cache, workers, batch 等设置。
    # 只要 project, name, data 保持一致，就会接着训练。
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
        resume=args.resume,
        lr0=args.lr0,
        warmup_epochs=args.warmup_epochs,
        optimizer=args.optimizer,
    )
    print(str(results.save_dir))


if __name__ == "__main__":
    main()
