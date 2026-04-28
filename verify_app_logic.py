
import sys
import os
import torch
import numpy as np
import cv2
from pathlib import Path

# 设置 PYTHONPATH
sys.path.append(r"G:\project\yolo11-chest_editted\YOLO_PROJECT")

from PyQt6.QtWidgets import QApplication
from YOLO_PROJECT.app.qt_medsam_app import Main

def verify_app_logic():
    print("开始验证 qt_medsam_app.py 内部逻辑...")
    
    # 1. 初始化应用和窗口
    app = QApplication(sys.argv)
    print("QApplication 已创建")
    window = Main()
    print("Main 窗口已实例化")
    
    # 2. 模拟加载 DICOM 序列
    dicom_dir = r"G:\project\yolo11-chest_editted\lung_1\sysucc lung cancer more than 4\10015239+20150119+CT\a_2mm"
    print(f"正在模拟加载 DICOM 目录: {dicom_dir}")
    window.load_series_dir(dicom_dir)
    
    if window.series is not None:
        print(f"DICOM 加载成功: shape={window.series.shape}, spacing={window.spacing}")
    else:
        print("DICOM 加载失败")
        return

    # 3. 验证模型权重加载
    yolo_weight = window.combo_yolo.currentData()
    medsam_weight = window.combo_medsam.currentData()
    print(f"当前选择的 YOLO 权重: {yolo_weight}")
    print(f"当前选择的 MedSAM 权重: {medsam_weight}")
    
    if True: # 强制使用手动注入的权重
        # 尝试手动设置一个已知的路径以便继续测试逻辑
        yolo_weight = r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\runs\lung_nodule_lung1_v6_cpu\weights\best.pt"
        # 使用真实的 MedSAM 权重
        medsam_weight = r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\runs\medsam_train_cpu\weights\best.pth"
        window.combo_yolo.addItem("Manual", yolo_weight)
        window.combo_yolo.setCurrentIndex(window.combo_yolo.count()-1)
        window.combo_medsam.clear()
        window.combo_medsam.addItem("Manual_MedSAM", medsam_weight)
        window.combo_medsam.setCurrentIndex(window.combo_medsam.count()-1)
        print("已手动注入权重路径 (使用真实 MedSAM 模型) 进行逻辑验证", flush=True)

    # 4. 模拟点击“自动标注”按钮
    print("正在模拟执行 '自动标注' (on_run)...")
    try:
        window.device = "cpu"
        print(f"测试强制使用设备: {window.device}")
        
        # 遍历所有切片，寻找能检测到目标的切片
        found = False
        print(f"遍历序列中，总计 {window.series.shape[0]} 个切片...", flush=True)
        # 缩小搜索范围，优先搜索中间切片
        center = window.series.shape[0] // 2
        offsets = [0, 10, -10, 20, -20, 30, -30, 40, -40, 50, -50]
        
        for offset in offsets:
            i = center + offset
            if not (0 <= i < window.series.shape[0]): continue
            
            window.idx = i
            print(f"尝试切片: {i}", flush=True)
            # 模拟执行 on_run 的核心预测部分
            hu = window.series[window.idx]
            from YOLO_PROJECT.pipeline.dicom import window_normalize
            img = window_normalize(hu)
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            yolo_path = window.combo_yolo.currentData()
            if window.det is None:
                from ultralytics import YOLO
                print(f"加载 YOLO 模型: {yolo_path}", flush=True)
                window.det = YOLO(yolo_path)
            
            # 使用更低的置信度
            r = window.det.predict(source=[bgr], imgsz=640, conf=0.01, device="cpu", verbose=False)
            boxes = r[0].boxes.xyxy.cpu().numpy().astype(int) if len(r) and len(r[0].boxes) else np.empty((0, 4), dtype=int)
            
            if boxes.shape[0] > 0:
                print(f"在切片 {i} 发现目标: {boxes[0]}", flush=True)
                # 继续执行分割
                box = boxes[0]
                medsam_path = window.combo_medsam.currentData()
                if window.seg is None:
                    from YOLO_PROJECT.pipeline.medsam import MedSAMSegmenter
                    print(f"正在初始化 MedSAMSegmenter: {medsam_path}", flush=True)
                    # 尝试使用 XPU，因为 test_sam_image.py 在 XPU 上成功了
                    window.seg = MedSAMSegmenter(medsam_path, model_type="vit_b", device="xpu")
                    # 打印模型设备信息
                    print(f"MedSAM model device: {next(window.seg.model.parameters()).device}", flush=True)
                    print("MedSAMSegmenter 初始化完成", flush=True)
                
                print("正在运行 MedSAM 预测...", flush=True)
                # 显式使用 predictor 以便打印更详细的日志
                # 模拟 MedSAM 预处理
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img_1024 = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                print(f"Resized image: {img_1024.shape}", flush=True)
                
                print("Calling predictor.set_image...", flush=True)
                window.seg.predictor.set_image(rgb)
                print("predictor.set_image finished.", flush=True)
                
                print(f"Calling predictor.predict with box: {box}", flush=True)
                masks, scores, _ = window.seg.predictor.predict(box=box, multimask_output=False)
                print("predictor.predict finished.", flush=True)
                
                window.last_mask = masks[0].astype(np.uint8)
                window.last_box = box
                print(f"切片 {i} 自动标注成功! Mask 像素和: {np.sum(window.last_mask)}", flush=True)
                found = True
                break
        
        if not found:
            print("在整个序列采样中均未检测到目标")
            
    except Exception as e:
        print(f"执行 on_run 时出错: {e}")
        import traceback
        traceback.print_exc()

    print("逻辑验证完成。")

if __name__ == "__main__":
    verify_app_logic()
