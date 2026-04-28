from ultralytics import YOLO
import cv2
import os

model_path = r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\runs\lung_nodule_lung1_v6_cpu\weights\best.pt"
image_path = r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\datasets\lung_1_YOLO_3\train\images\10015239+20150119+CT_0105.jpg"

print("Loading model...")
model = YOLO(model_path)
print("Reading image...")
img = cv2.imread(image_path)
print("Predicting...")
results = model.predict(img, device='cpu')
print("Detections:", len(results[0].boxes))
