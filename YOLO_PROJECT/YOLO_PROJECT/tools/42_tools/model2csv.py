import os

from ultralytics import YOLO
import cv2
import os.path as osp
import numpy as np
data = []
# Load a model
img_folder = ""
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model
image_names = os.listdir(img_folder)
for image_name in image_names:
    image_path = osp.join(img_folder, image_name)
    results = model(image_path)
    result = results[0]
    result_names = result.names
    result_nums = [0 for i in range(0, len(result_names))]
    cls_ids = list(result.boxes.cls.cpu().numpy())
    for cls_id in cls_ids:
        result_nums[int(cls_id)] = result_nums[int(cls_id)] + 1
    data.append([image_name, result_nums[0], result_nums[1], result_nums[2], result_nums[3], result_nums[4]])

np.savetxt("result.csv", np.array(data), fmt="%s", delimiter=",", newline="\n")
