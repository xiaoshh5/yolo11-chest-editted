import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="One-click YOLO inference")
    p.add_argument("--model", default="yolo11n.pt", help="Path to YOLO model .pt")
    p.add_argument("--source", required=True, help="Image/video file or directory")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--device", default="cpu", help="Device id or 'cpu'")
    default_project = Path(__file__).resolve().parent.parent / "runs"
    p.add_argument("--project", default=str(default_project), help="Project directory for outputs")
    p.add_argument("--name", default="predict", help="Run name under project")
    p.add_argument("--save_txt", action="store_true", help="Save results to *.txt")
    p.add_argument("--save_conf", action="store_true", help="Save confidences in labels")
    p.add_argument("--save", dest="save", action="store_true", help="Save visualized predictions")
    p.add_argument("--nosave", dest="save", action="store_false", help="Do not save images")
    p.set_defaults(save=True)
    args = p.parse_args()

    model = YOLO(args.model)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=args.project,
        name=args.name,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        verbose=False,
    )
    print(f"Results saved to {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()
