#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：step3_start_window_track.py 
@File    ：export_onnx.py
@IDE     ：PyCharm 
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：TODO 添加文件描述
@Date    ：2024/12/6 10:15 
'''
import torch
net = torch.load('best.pt', map_location='cpu')
net.eval()
dummpy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(net, dummpy_input, 'yolov8n.onnx', export_params=True,
                  input_names=['input'],
                  output_names=['output'])