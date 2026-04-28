
from ultralytics import YOLO
import os

model_path = r'G:\project\yolo11-chest_editted\runs\train16\weights\best.pt'
img_path = r'G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\datasets\lung_1_YOLO_3\train\images\10015239+20150119+CT_0105.jpg'

print(f"Model exists: {os.path.exists(model_path)}")
print(f"Image exists: {os.path.exists(img_path)}")

try:
    model = YOLO(model_path)
    print("Model loaded.")
    results = model.predict(source=img_path, conf=0.001, device='cpu', verbose=True)
    print("Prediction done.")
    print(f"Detected: {len(results[0].boxes)} boxes")
    for box in results[0].boxes:
        print(f"  Box: {box.xyxy}, Conf: {box.conf}")
except Exception as e:
    print(f"Error: {e}")
