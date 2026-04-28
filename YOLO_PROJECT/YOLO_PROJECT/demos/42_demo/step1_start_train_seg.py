#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import time
import torch
from ultralytics import YOLO

# set dataset yaml absolute path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_yaml = os.path.join(project_root, 'chest_seg_data.yaml')

# load segmentation model (use seg config/weights if available)
# If you have a pretrained seg weight, set it below (e.g., 'yolo11n-seg.pt'), else will use config only.
model = YOLO('yolo11n-seg.yaml').load('yolo11n-seg.pt')

# device selection: prefer Intel XPU if available
try:
    if torch.xpu.is_available():
        device_arg = 'xpu'
        print('Using Intel GPU (XPU)')
    else:
        device_arg = 'cpu'
        print('Intel GPU not available, using CPU')
except Exception:
    device_arg = 'cpu'
    print('XPU check failed, using CPU')

# Important: use a distinct project/name so detection outputs aren't overwritten
results = model.train(
    data=data_yaml,
    task='segment',
    project='./runs',
    name='seg_train1',       # ensures results saved to runs/seg_train1
    epochs=100,
    imgsz=640,
    batch=8,
    device=device_arg,
    cache=True,
    save_period=10,
    patience=20,
)

print('Training finished')

time.sleep(2)
