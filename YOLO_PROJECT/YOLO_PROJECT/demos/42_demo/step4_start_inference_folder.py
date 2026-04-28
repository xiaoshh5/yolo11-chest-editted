'''
@Project ：ultralytics-8.2.77
@File    ：start_single_detect.py
@IDE     ：PyCharm
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：用于文件夹格式的预测
@Date    ：2024/8/15 15:15
'''

from ultralytics import YOLO
# Load a pretrained YOLOv8n model
MODEL_PATH = ""r"E:\downloads\xufei\yolov8-42-xufei\42_demo\runs\detect\train8\weights\best.pt"
FOLDER_PATH = r"E:\downloads\xufei\test"
model = YOLO(MODEL_PATH)  # 加载模型路径
model.predict(FOLDER_PATH, save=True, imgsz=640, conf=0.5, save_txt=True, save_conf=False)
