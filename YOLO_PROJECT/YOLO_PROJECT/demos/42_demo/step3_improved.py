#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import sys
import shutil
import cv2
import numpy as np
import time
import torch
import threading
import os.path as osp
from collections import defaultdict
from PySide6.QtGui import *
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from ultralytics import YOLO, SAM

# ==========================================
# 配置与常量
# ==========================================
WINDOW_TITLE = "Medical AI Workstation - 肺结节智能诊断系统"
DEFAULT_OUTPUT_SIZE = 640
TEMP_DIR = "images/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs("record/img", exist_ok=True)

# 专业医学风格样式表 (Dark Theme)
STYLESHEET = """
QMainWindow {
    background-color: #1e1e1e;
}
QWidget {
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
    color: #e0e0e0;
}
/* 侧边栏背景 */
QFrame#SidePanel {
    background-color: #252526;
    border-left: 1px solid #3e3e42;
}
/* 影像区域背景 */
QLabel#Viewport {
    background-color: #000000;
    border: 1px solid #333333;
}
/* 按钮通用样式 */
QPushButton {
    background-color: #3c3c3c;
    border: 1px solid #555555;
    color: #ffffff;
    padding: 6px 12px;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #505050;
    border-color: #007acc;
}
QPushButton:pressed {
    background-color: #007acc;
    border-color: #007acc;
}
/* 强调按钮 */
QPushButton#PrimaryBtn {
    background-color: #0078d4;
    border-color: #0078d4;
    font-weight: bold;
}
QPushButton#PrimaryBtn:hover {
    background-color: #106ebe;
}
/* 表格样式 */
QTableWidget {
    background-color: #252526;
    border: 1px solid #3e3e42;
    gridline-color: #3e3e42;
}
QTableWidget::item {
    padding: 5px;
}
QHeaderView::section {
    background-color: #333333;
    color: #cccccc;
    padding: 4px;
    border: 1px solid #3e3e42;
}
/* 输入框 */
QLineEdit, QComboBox {
    background-color: #3c3c3c;
    border: 1px solid #555555;
    color: white;
    padding: 4px;
}
/* 分组框 */
QGroupBox {
    border: 1px solid #444444;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    color: #aaaaaa;
}
"""

# ==========================================
# 自定义控件：悬浮分析面板 (模拟截图中的弹窗)
# ==========================================
class AnalysisOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.SubWindow)
        self.setAttribute(Qt.WA_TranslucentBackground) # 透明背景支持
        self.hide()
        
        # 布局容器
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 背景框 (半透明深色)
        self.frame = QFrame()
        self.frame.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 30, 30, 240);
                border: 1px solid #0078d4;
                border-radius: 6px;
            }
            QLabel { background: transparent; }
        """)
        self.frame_layout = QVBoxLayout(self.frame)
        
        # 标题栏
        title_bar = QHBoxLayout()
        self.title_label = QLabel("ROI 分析结果")
        self.title_label.setStyleSheet("font-weight: bold; color: #0078d4; font-size: 14px;")
        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet("border: none; background: transparent; font-size: 16px; color: #aaa;")
        close_btn.clicked.connect(self.hide)
        title_bar.addWidget(self.title_label)
        title_bar.addStretch()
        title_bar.addWidget(close_btn)
        
        # 内容区
        self.content_label = QLabel("正在计算...")
        self.histogram_label = QLabel() # 用于显示直方图图片
        self.histogram_label.setFixedHeight(80)
        self.histogram_label.setAlignment(Qt.AlignCenter)
        self.histogram_label.setStyleSheet("border: 1px dashed #555;")
        
        self.frame_layout.addLayout(title_bar)
        self.frame_layout.addWidget(self.histogram_label)
        self.frame_layout.addWidget(self.content_label)
        
        self.main_layout.addWidget(self.frame)

    def update_data(self, solid_ratio, avg_density, size_info):
        """更新面板数据"""
        # 模拟生成一个直方图 (用OpenCV画一个简单的图)
        hist_img = np.zeros((80, 260, 3), dtype=np.uint8)
        # 画一些随机线条模拟分布
        pts = np.array([[20, 70], [50, 40], [100, 20], [150, 30], [200, 70]], np.int32)
        cv2.fillPoly(hist_img, [pts], (0, 120, 212)) # 蓝色填充
        cv2.polylines(hist_img, [pts], False, (255, 255, 255), 1)
        
        # 转为QPixmap显示
        h, w, c = hist_img.shape
        bytes_per_line = 3 * w
        q_img = QImage(hist_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.histogram_label.setPixmap(QPixmap.fromImage(q_img))

        # 更新文字
        text = (
            f"<b>实性占比 (Solid Ratio):</b> <span style='color:#00ff00'>{solid_ratio:.2f}%</span><br>"
            f"<b>平均密度 (Avg HU):</b> {avg_density:.1f}<br>"
            f"<b>最大直径:</b> {size_info} mm<br>"
            f"<hr>"
            f"<span style='color:#aaa; font-size:11px'>*该结果仅供辅助诊断参考</span>"
        )
        self.content_label.setText(text)

# ==========================================
# 主窗口
# ==========================================
class MedicalMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1400, 900)
        self.setStyleSheet(STYLESHEET)
        
        # 核心变量
        self.img_path = None
        self.model = YOLO("yolov8n.pt") # 默认加载
        try:
            self.sam_model = SAM("sam_b.pt")
        except:
            print("SAM模型未找到，分割功能可能受限")
            self.sam_model = None

        self.conf_thres = 0.25
        self.iou_thres = 0.45
        
        # 初始化界面
        self.init_ui()
        
    def init_ui(self):
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ------------------------------------------------
        # 1. 左侧：影像查看区 (Viewport)
        # ------------------------------------------------
        viewport_container = QWidget()
        viewport_layout = QVBoxLayout(viewport_container)
        viewport_layout.setContentsMargins(10, 10, 10, 10)
        
        # 顶部工具条 (模拟医学软件顶部)
        tool_bar = QHBoxLayout()
        self.info_label = QLabel("Patient ID: ANONYMOUS | Sex: M | Age: 45 | Series: Thorax CT")
        self.info_label.setStyleSheet("color: #888888; font-size: 12px;")
        tool_bar.addWidget(self.info_label)
        tool_bar.addStretch()
        # 模拟工具图标
        for icon_text in ["🔍 Zoom", "🖐️ Pan", "📏 Measure", "🌗 Window"]:
            btn = QPushButton(icon_text)
            btn.setStyleSheet("background: transparent; border: none; color: #aaa;")
            tool_bar.addWidget(btn)
        
        # 图像显示标签
        self.image_label = QLabel()
        self.image_label.setObjectName("Viewport")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("请导入 CT 影像 / 图像\n(Drag & Drop not supported yet)")
        self.image_label.setStyleSheet("color: #555; font-size: 16px;")
        
        viewport_layout.addLayout(tool_bar)
        viewport_layout.addWidget(self.image_label, stretch=1)
        
        # 初始化悬浮分析面板 (添加到 Viewport 中)
        self.overlay = AnalysisOverlay(self.image_label)

        # ------------------------------------------------
        # 2. 右侧：操作与信息面板 (SidePanel)
        # ------------------------------------------------
        side_panel = QFrame()
        side_panel.setObjectName("SidePanel")
        side_panel.setFixedWidth(380)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(15, 20, 15, 20)
        side_layout.setSpacing(15)

        # A. 标题区
        title_lbl = QLabel("AI 辅助诊断控制台")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        side_layout.addWidget(title_lbl)

        # B. 结节检测列表
        list_group = QGroupBox("检测结果 (Findings)")
        list_layout = QVBoxLayout(list_group)
        self.result_table = QTableWidget(0, 4)
        self.result_table.setHorizontalHeaderLabels(["ID", "类别", "置信度", "位置"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers) # 不可编辑
        list_layout.addWidget(self.result_table)
        side_layout.addWidget(list_group, stretch=1)

        # C. 参数设置区
        settings_group = QGroupBox("参数配置 (Configuration)")
        settings_layout = QGridLayout(settings_group)
        
        settings_layout.addWidget(QLabel("置信度阈值:"), 0, 0)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setValue(self.conf_thres)
        self.conf_spin.setSingleStep(0.05)
        settings_layout.addWidget(self.conf_spin, 0, 1)

        settings_layout.addWidget(QLabel("IOU 阈值:"), 1, 0)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setValue(self.iou_thres)
        self.iou_spin.setSingleStep(0.05)
        settings_layout.addWidget(self.iou_spin, 1, 1)
        
        side_layout.addWidget(settings_group)

        # D. 功能按钮区
        btn_layout = QVBoxLayout()
        
        self.btn_upload = QPushButton("📂 导入影像 (Load Image)")
        self.btn_upload.setMinimumHeight(40)
        self.btn_upload.clicked.connect(self.upload_img)
        
        self.btn_detect = QPushButton("⚡ 开始检测 (AI Detection)")
        self.btn_detect.setObjectName("PrimaryBtn") # 蓝色高亮
        self.btn_detect.setMinimumHeight(40)
        self.btn_detect.clicked.connect(self.run_detection)
        
        self.btn_segment = QPushButton("🧬 结节分割与分析 (Segmentation)")
        self.btn_segment.setMinimumHeight(40)
        self.btn_segment.clicked.connect(self.run_segmentation)
        
        btn_layout.addWidget(self.btn_upload)
        btn_layout.addWidget(self.btn_detect)
        btn_layout.addWidget(self.btn_segment)
        
        side_layout.addLayout(btn_layout)
        
        # 底部版权微标 (可选，保持低调)
        footer = QLabel("System Ready. v2.0 Pro")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #444; font-size: 10px;")
        side_layout.addWidget(footer)

        # 将左右两侧加入主布局
        main_layout.addWidget(viewport_container)
        main_layout.addWidget(side_panel)

    # ==========================================
    # 逻辑功能区
    # ==========================================

    def upload_img(self):
        """上传图片"""
        file_name, _ = QFileDialog.getOpenFileName(self, '选择影像', '', 'Image Files (*.jpg *.png *.jpeg *.bmp *.tif)')
        if file_name:
            self.img_path = file_name
            # 显示原始图片
            self.show_image(file_name)
            # 清空之前的结果
            self.result_table.setRowCount(0)
            self.overlay.hide()
            self.info_label.setText(f"File: {os.path.basename(file_name)} | Size: Original")

    def show_image(self, img_source):
        """在 Label 上显示图片，自适应大小"""
        if isinstance(img_source, str):
            cv_img = cv2.imread(img_source)
        else:
            cv_img = img_source
            
        if cv_img is None:
            return

        # 转换为 RGB
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 缩放到 Label 大小 (保持比例)
        # 注意：这里简单的缩放，实际工程中通常保存 pixmap 并在 resizeEvent 中重绘
        pixmap = QPixmap.fromImage(qt_img)
        lbl_w = self.image_label.width()
        lbl_h = self.image_label.height()
        scaled_pixmap = pixmap.scaled(lbl_w, lbl_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def run_detection(self):
        """运行 YOLO 检测"""
        if not self.img_path:
            QMessageBox.warning(self, "提示", "请先加载图片")
            return
        
        # 获取参数
        conf = self.conf_spin.value()
        iou = self.iou_spin.value()
        
        # 运行推理
        results = self.model(self.img_path, conf=conf, iou=iou)
        result = results[0]
        
        # 绘制结果
        plot_img = result.plot() # BGR numpy array
        self.show_image(plot_img) # 更新显示
        
        # 保存临时结果用于分割
        cv2.imwrite(osp.join(TEMP_DIR, "last_detection.jpg"), plot_img)
        
        # 更新右侧表格
        self.update_table(result)
        
        QMessageBox.information(self, "完成", f"检测完成，发现 {len(result.boxes)} 个目标。")

    def update_table(self, result):
        """将检测结果填入表格"""
        self.result_table.setRowCount(0)
        boxes = result.boxes
        names = result.names
        
        for idx, box in enumerate(boxes):
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            
            # 类别 ID
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])
            xywh = box.xywh[0].cpu().numpy() # x, y, w, h
            
            # 填入数据
            self.result_table.setItem(row, 0, QTableWidgetItem(str(idx + 1)))
            self.result_table.setItem(row, 1, QTableWidgetItem(cls_name))
            
            # 置信度带颜色条
            conf_item = QTableWidgetItem(f"{conf:.2f}")
            if conf > 0.8:
                conf_item.setForeground(QColor("#00ff00"))
            elif conf > 0.5:
                conf_item.setForeground(QColor("#ffff00"))
            self.result_table.setItem(row, 2, conf_item)
            
            loc_str = f"X:{int(xywh[0])} Y:{int(xywh[1])}"
            self.result_table.setItem(row, 3, QTableWidgetItem(loc_str))

    def run_segmentation(self):
        """运行 SAM 分割并弹出浮动窗口"""
        if not self.img_path:
            return
            
        if self.sam_model is None:
            QMessageBox.critical(self, "错误", "未加载 SAM 模型 (sam_b.pt)")
            return

        # 简单使用 SAM 全图生成模式 (实际应用中通常基于 YOLO 的 Box 提示进行分割)
        # 这里为了演示效果，我们假设基于最近一次检测结果或者全图
        results = self.sam_model(self.img_path)
        result = results[0]
        
        # 获取绘制后的图像
        plot_img = result.plot()
        self.show_image(plot_img)
        
        # --- 模拟计算医学指标 ---
        # 假设我们获取了掩膜，计算像素强度
        # 在真实场景中，这里需要读取原始 DICOM 的 HU 值
        
        # 模拟数据
        import random
        solid_ratio = random.uniform(20.0, 85.0) # 实性占比
        avg_density = random.uniform(-600, 100)  # 平均密度
        size_info = random.randint(5, 30)        # 直径
        
        # 显示浮动面板
        # 计算面板位置：显示在 label 的右下角或者中央
        panel_x = 50
        panel_y = 50
        self.overlay.move(panel_x, panel_y)
        self.overlay.update_data(solid_ratio, avg_density, size_info)
        self.overlay.show()
        self.overlay.raise_() # 确保在最上层


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用图标 (可选)
    # app.setWindowIcon(QIcon("icon.png"))
    
    window = MedicalMainWindow()
    window.show()
    sys.exit(app.exec())