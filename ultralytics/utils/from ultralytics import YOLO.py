from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # 或你的模型/配置
results = model.train(
    data="path/to/your_dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="xpu",
    amp=True  # 需要时可启用
)