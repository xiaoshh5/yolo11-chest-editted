#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics-8.2.77 
@File    ：start_window.py
@IDE     ：PyCharm 
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：主要的图形化界面，本次图形化界面实现的主要技术为pyside6，pyside6是官方提供支持的
@Date    ：2024/8/15 15:15 
'''
import copy                      # 用于图像复制
import os                        # 用于系统路径查找
import shutil                    # 用于复制
from distutils.command.config import config
from PySide6.QtGui import *      # GUI组件
from PySide6.QtCore import *     # 字体、边距等系统变量
from PySide6.QtWidgets import *  # 窗口等小组件
import threading                 # 多线程
import sys                       # 系统库
import cv2                       # opencv图像处理
import torch                     # 深度学习框架
import os.path as osp            # 路径查找
import time                      # 时间计算
from ultralytics import YOLO, SAM     # yolo核心算法
from ultralytics.utils.torch_utils import select_device
from collections import defaultdict, UserDict
import numpy as np
# 常用的字符串常量
WINDOW_TITLE ="Target detection system"            # 系统上方标题
WELCOME_SENTENCE = "欢迎使用基于yolo11的肺结节检测"   # 欢迎的句子
ICON_IMAGE = "images/UI/lufei.png"                 # 系统logo界面
IMAGE_LEFT_INIT = "images/UI/up.jpeg"              # 图片检测界面初始化左侧图像
IMAGE_RIGHT_INIT = "images/UI/right.jpeg"          # 图片检测界面初始化右侧图像
USERNAME = "3045834499"
PASSWORD = "3045834499"


class MainWindow(QTabWidget):
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)       # 系统界面标题
        self.resize(1200, 800)                  # 系统初始化大小
        self.setWindowIcon(QIcon(ICON_IMAGE))   # 系统logo图像
        self.output_size = 480                  # 上传的图像和视频在系统界面上显示的大小
        self.img2predict = ""                   # 要进行预测的图像路径
        # 用来进行设置的参数
        self.init_vid_id = '0'                  # 网络摄像头修改 包括ip或者是ip地址的修改
        self.vid_source = int(self.init_vid_id) # 需要设置为对应的整数，加载的才是usb的摄像头
        self.conf_thres = 0.25   # 置信度的阈值
        self.iou_thres = 0.45    # NMS操作的时候 IOU过滤的阈值
        self.save_txt = False
        self.save_conf = False
        self.save_crop = False
        self.vid_gap = 30        # 摄像头视频帧保存间隔。
        self.is_open_track = ""  # 三种选择，如果是空表示不开启追踪，否则有两种追踪器可以进行选择


        self.cap = cv2.VideoCapture(self.vid_source)
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        self.model_path = "yolov8n.pt"  # todo 指明模型加载的位置的设备
        self.model = self.model_load(weights=self.model_path)
        self.sam_model = SAM("sam_b.pt")  # 加载SAM模型用于分割

        self.initUI()            # 初始化图形化界面
        self.reset_vid()         # 重新设置视频参数，重新初始化是为了防止视频加载出错

    # 模型初始化
    @torch.no_grad()
    def model_load(self, weights=""):
        """
        模型加载
        """
        # 模型加载的时候配合置信度一起使用

        model_loaded = YOLO(weights)
        return model_loaded

    def initUI(self):
        """
        图形化界面初始化
        """
        # ********************* 图片识别界面 *****************************
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("图片识别功能")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap(IMAGE_LEFT_INIT))
        self.right_img.setPixmap(QPixmap(IMAGE_RIGHT_INIT))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addWidget(self.right_img)
        self.img_num_label = QLabel("当前检测结果：待检测")
        self.img_num_label.setFont(font_main)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        seg_img_button = QPushButton("开始分割")  # 新增分割按钮
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        seg_img_button.clicked.connect(self.segment_img)  # 连接分割函数
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(self.img_num_label)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_layout.addWidget(seg_img_button)  # 添加分割按钮
        img_detection_widget.setLayout(img_detection_layout)

        # ********************* 视频识别界面 *****************************
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout()
        vid_title = QLabel("视频检测功能")
        vid_title.setFont(font_title)
        self.vid_img = QLabel()
        self.vid_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        vid_title.setAlignment(Qt.AlignCenter)
        self.vid_img.setAlignment(Qt.AlignCenter)
        self.webcam_detection_btn = QPushButton("摄像头实时监测")
        self.mp4_detection_btn = QPushButton("视频文件检测")
        self.vid_stop_btn = QPushButton("停止检测")
        self.webcam_detection_btn.setFont(font_main)
        self.mp4_detection_btn.setFont(font_main)
        self.vid_stop_btn.setFont(font_main)
        self.webcam_detection_btn.setStyleSheet("QPushButton{color:white}"
                                                "QPushButton:hover{background-color: rgb(2,110,180);}"
                                                "QPushButton{background-color:rgb(48,124,208)}"
                                                "QPushButton{border:2px}"
                                                "QPushButton{border-radius:5px}"
                                                "QPushButton{padding:5px 5px}"
                                                "QPushButton{margin:5px 5px}")
        self.mp4_detection_btn.setStyleSheet("QPushButton{color:white}"
                                             "QPushButton:hover{background-color: rgb(2,110,180);}"
                                             "QPushButton{background-color:rgb(48,124,208)}"
                                             "QPushButton{border:2px}"
                                             "QPushButton{border-radius:5px}"
                                             "QPushButton{padding:5px 5px}"
                                             "QPushButton{margin:5px 5px}")
        self.vid_stop_btn.setStyleSheet("QPushButton{color:white}"
                                        "QPushButton:hover{background-color: rgb(2,110,180);}"
                                        "QPushButton{background-color:rgb(48,124,208)}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:5px}"
                                        "QPushButton{padding:5px 5px}"
                                        "QPushButton{margin:5px 5px}")
        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.mp4_detection_btn.clicked.connect(self.open_mp4)
        self.vid_stop_btn.clicked.connect(self.close_vid)
        vid_detection_layout.addWidget(vid_title)
        vid_detection_layout.addWidget(self.vid_img)
        # todo 添加摄像头检测标签逻辑
        self.vid_num_label = QLabel("当前检测结果：{}".format("等待检测"))
        self.vid_num_label.setFont(font_main)
        vid_detection_layout.addWidget(self.vid_num_label)
        vid_detection_layout.addWidget(self.webcam_detection_btn)
        vid_detection_layout.addWidget(self.mp4_detection_btn)
        vid_detection_layout.addWidget(self.vid_stop_btn)
        vid_detection_widget.setLayout(vid_detection_layout)

        # ********************* 模型切换界面 *****************************
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel(WELCOME_SENTENCE)
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/UI/zhu.jpg'))
        self.model_label = QLabel("当前模型：{}".format(self.model_path))
        self.model_label.setFont(font_main)
        change_model_button = QPushButton("切换模型")
        change_model_button.setFont(font_main)
        change_model_button.setStyleSheet("QPushButton{color:white}"
                                          "QPushButton:hover{background-color: rgb(2,110,180);}"
                                          "QPushButton{background-color:rgb(48,124,208)}"
                                          "QPushButton{border:2px}"
                                          "QPushButton{border-radius:5px}"
                                          "QPushButton{padding:5px 5px}"
                                          "QPushButton{margin:5px 5px}")

        record_button = QPushButton("查看历史记录")
        record_button.setFont(font_main)
        record_button.clicked.connect(self.check_record)
        record_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        change_model_button.clicked.connect(self.change_model)
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel()  # todo 更换作者信息
        label_super.setText("<a href='https://blog.csdn.net/ECHOSON'>作者：肆十二</a>")
        label_super.setFont(QFont('楷体', 16))
        label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addWidget(self.model_label)
        about_layout.addStretch()
        about_layout.addWidget(change_model_button)
        about_layout.addWidget(record_button)
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)
        self.left_img.setAlignment(Qt.AlignCenter)

        # ********************* 配置切换界面 ****************************
        config_widget = QWidget()

        config_grid_widget = QWidget()
        config_grid_layout = QGridLayout()

        # self.output_size = 480  # 上传的图像和视频在系统界面上显示的大小
        config_output_size_label = QLabel("系统图像显示大小")
        self.config_output_size_value = QLineEdit("")
        self.config_output_size_value.setText(str(self.output_size))
        config_grid_layout.addWidget(config_output_size_label, 0, 0)
        config_grid_layout.addWidget(self.config_output_size_value, 0, 1)


        # # 用来进行设置的参数
        # self.init_vid_id = '0'  # 网络摄像头修改 包括ip或者是ip地址的修改
        config_vid_source_label = QLabel("摄像头源地址")
        self.config_vid_source_value = QLineEdit("")
        self.config_vid_source_value.setText(str(self.vid_source))
        config_grid_layout.addWidget(config_vid_source_label)
        config_grid_layout.addWidget(self.config_vid_source_value)

        # self.vid_gap = 30  # 摄像头视频帧保存间隔。
        config_vid_gap_label = QLabel("视频帧保存间隔")
        self.config_vid_gap_value = QLineEdit("")
        self.config_vid_gap_value.setText(str(self.vid_gap))
        config_grid_layout.addWidget(config_vid_gap_label)
        config_grid_layout.addWidget(self.config_vid_gap_value )

        # self.vid_source = int(self.init_vid_id)  # 需要设置为对应的整数，加载的才是usb的摄像头
        # self.conf_thres = 0.25  # 置信度的阈值
        config_conf_thres_label = QLabel("检测模型置信度阈值")
        self.config_conf_thres_value = QLineEdit("")
        self.config_conf_thres_value.setText(str(self.conf_thres))
        config_grid_layout.addWidget(config_conf_thres_label)
        config_grid_layout.addWidget(self.config_conf_thres_value)

        # self.iou_thres = 0.45  # NMS操作的时候 IOU过滤的阈值
        config_iou_thres_label = QLabel("检测模型IOU阈值")
        self.config_iou_thres_value = QLineEdit("")
        self.config_iou_thres_value.setText(str(self.iou_thres))
        config_grid_layout.addWidget(config_iou_thres_label)
        config_grid_layout.addWidget(self.config_iou_thres_value)

        # self.save_txt = False
        config_save_txt_label = QLabel("推理时是否保存txt文件")
        self.config_save_txt_value = QRadioButton("True")
        self.config_save_txt_value.setChecked(False)
        self.config_save_txt_value.setAutoExclusive(False)
        config_grid_layout.addWidget(config_save_txt_label)
        config_grid_layout.addWidget(self.config_save_txt_value)
        # btn1 = QRadioButton("男")
        # # 设置btn1为默认选中
        # btn1.setChecked(True)

        # self.save_conf = False
        config_save_conf_label = QLabel("推理时是否保存置信度")
        self.config_save_conf_value = QRadioButton("True")
        self.config_save_conf_value.setChecked(False)
        self.config_save_conf_value.setAutoExclusive(False)
        config_grid_layout.addWidget( config_save_conf_label)
        config_grid_layout.addWidget( self.config_save_conf_value)
        # self.save_crop = False
        config_save_crop_label = QLabel("推理时是否保存切片文件")
        self.config_save_crop_value = QRadioButton("True")
        self.config_save_crop_value.setChecked(False)
        self.config_save_crop_value.setAutoExclusive(False)
        config_grid_layout.addWidget(config_save_crop_label)
        config_grid_layout.addWidget(self.config_save_crop_value)

        # 追踪配置
        config_track_label = QLabel("追踪配置")
        self.config_track_value = QComboBox(self)
        # results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        # results = model.track(frame, persist=True, tracker="botsort.yaml")
        self.config_track_value.addItems(['不开启追踪', "bytetrack.yaml", "botsort.yaml"])
        config_grid_layout.addWidget(config_track_label)
        config_grid_layout.addWidget(self.config_track_value)
        # self.cb = QComboBox(self)
        # self.cb.move(100, 20)
        #
        # # 单个添加条目
        # self.cb.addItem('C')
        # self.cb.addItem('C++')
        # self.cb.addItem('Python')
        # # 多个添加条目
        # self.cb.addItems(['Java', 'C#', 'PHP'])

        # 追踪模型选择，以及是否使用追踪模型

        config_grid_widget.setLayout(config_grid_layout)
        config_grid_widget.setFont(font_main)

        save_config_button = QPushButton("保存配置信息")
        save_config_button.setFont(font_main)
        save_config_button.clicked.connect(self.save_config_change)
        save_config_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        config_layout = QVBoxLayout()
        config_vid_title = QLabel("配置信息修改")
        config_icon_label = QLabel()
        config_icon_label.setPixmap(QPixmap("images/UI/config.png"))
        config_icon_label.setAlignment(Qt.AlignCenter)
        config_vid_title.setAlignment(Qt.AlignCenter)
        config_vid_title.setFont(font_title)
        config_layout.addWidget(config_vid_title)
        config_layout.addWidget(config_icon_label)
        config_layout.addWidget(config_grid_widget)
        config_layout.addStretch()
        config_layout.addWidget(save_config_button)
        config_widget.setLayout(config_layout)


        self.addTab(about_widget, '主页')
        self.addTab(img_detection_widget, '图片检测')
        self.addTab(vid_detection_widget, '视频检测')
        self.addTab(config_widget, '配置信息')
        self.setTabIcon(0, QIcon(ICON_IMAGE))
        self.setTabIcon(1, QIcon(ICON_IMAGE))
        self.setTabIcon(2, QIcon(ICON_IMAGE))
        self.setTabIcon(3, QIcon(ICON_IMAGE))

        # ********************* todo 布局修改和颜色变换等相关插件 *****************************

    def upload_img(self):
        """上传图像，图像要尽可能保证是中文格式"""
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg') # 选择图像
        if fileName: # 如果存在文件名称则对图像进行处理
            # 将图像转移到当前目录下，解决中文的问题
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)  # 将图像转移到images目录下并且修改为英文的形式
            shutil.copy(fileName, save_path)
            im0 = cv2.imread(save_path)
            # 调整图像的尺寸，让图像可以适应图形化的界面
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            self.img2predict = save_path                               # 给变量进行赋值方便后面实际进行读取
            # 将图像显示在界面上并将预测的文字内容进行初始化
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            self.right_img.setPixmap(QPixmap(IMAGE_RIGHT_INIT))
            self.img_num_label.setText("当前检测结果：待检测")

    def change_model(self):
        """切换模型，重新对self.model进行赋值"""
        # 用于pt格式模型的结果，这个模型必须是经过这里的代码训练出来的
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.pt')
        if fileName:
            # 如果用户选择了对应的pt文件，根据用户选择的pt文件重新对模型进行初始化
            self.model_path = fileName
            self.model = self.model_load(weights=self.model_path)
            QMessageBox.information(self, "成功", "模型切换成功！")
            self.model_label.setText("当前模型：{}".format(self.model_path))

    # 图片检测
    def detect_img(self):
        """检测单张的图像文件"""
        output_size = self.output_size
        # model.predict("bus.jpg", save=True, imgsz=320, conf=0.5)
        # self.save_txt = False
        #         self.save_conf = False
        #         self.save_crop = False
        print(self.save_txt)
        results = self.model(self.img2predict, conf=self.conf_thres, iou=self.iou_thres, save_txt=self.save_txt, save_conf=self.save_conf, save_crop=self.save_crop)  # 读取图像并执行检测的逻辑
        # 如果你想要对结果进行单独的解析请使用下面的内容
        # for result in results:
        #     boxes = result.boxes  # Boxes object for bounding box outputs
        #     masks = result.masks  # Masks object for segmentation masks outputs
        #     keypoints = result.keypoints  # Keypoints object for pose outputs
        #     probs = result.probs  # Probs object for classification outputs
        #     obb = result.obb  # Oriented boxes object for OBB outputs
        # 显示并保存检测的结果
        result = results[0]                     # 获取检测结果
        img_array = result.plot()               # 在图像上绘制检测结果
        im0 = img_array
        im_record = copy.deepcopy(im0)
        resize_scale = output_size / im0.shape[0]
        im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
        cv2.imwrite("images/tmp/single_result.jpg", im0)
        self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
        time_re = str(time.strftime('result_%Y-%m-%d_%H-%M-%S_%A'))
        cv2.imwrite("record/img/{}.jpg".format(time_re), im_record)
        # 显示每个类别中检测出来的样本数量
        result_names = result.names
        result_nums = [0 for i in range(0, len(result_names))]
        cls_ids = list(result.boxes.cls.cpu().numpy())
        for cls_id in cls_ids:
            result_nums[int(cls_id)] = result_nums[int(cls_id)] + 1
        result_info = ""
        for idx_cls, cls_num in enumerate(result_nums):
            # 添加对数据0的判断，如果当前数据的数目为0，则这个数据不需要加入到里面
            if cls_num > 0:
                result_info = result_info + "{}:{}\n".format(result_names[idx_cls], cls_num)
        self.img_num_label.setText("当前检测结果\n{}".format(result_info))
        QMessageBox.information(self, "检测成功", "日志已保存！")

    def segment_img(self):
        """使用SAM分割图像，并计算实性占比（简化版）"""
        if not self.img2predict:
            QMessageBox.warning(self, "警告", "请先上传图片！")
            return
        # 使用SAM进行分割（这里简化，实际可添加prompt）
        results = self.sam_model(self.img2predict, save=True)
        # 显示结果
        result = results[0]
        img_array = result.plot()
        im0 = img_array
        resize_scale = self.output_size / im0.shape[0]
        im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
        cv2.imwrite("images/tmp/segment_result.jpg", im0)
        self.right_img.setPixmap(QPixmap("images/tmp/segment_result.jpg"))

        # 简化实性占比计算（假设分割结果中高密度区域为实性）
        # 实际需要CT值分析，这里用像素强度模拟
        segmented = cv2.imread("images/tmp/segment_result.jpg", cv2.IMREAD_GRAYSCALE)
        if segmented is not None:
            total_pixels = np.sum(segmented > 0)  # 总分割像素
            solid_pixels = np.sum(segmented > 150)  # 假设>150为实性（需根据CT值调整）
            if total_pixels > 0:
                solid_ratio = (solid_pixels / total_pixels) * 100
                self.img_num_label.setText(f"分割完成，实性占比: {solid_ratio:.2f}%")
            else:
                self.img_num_label.setText("分割完成，无有效区域")
        else:
            self.img_num_label.setText("分割完成")

        QMessageBox.information(self, "分割成功", "分割结果和实性占比已计算！")

    def open_cam(self):
        """打开摄像头上传"""
        self.webcam_detection_btn.setEnabled(False)    # 将打开摄像头的按钮设置为false，防止用户误触
        self.mp4_detection_btn.setEnabled(False)       # 将打开mp4文件的按钮设置为false，防止用户误触
        self.vid_stop_btn.setEnabled(True)             # 将关闭按钮打开，用户可以随时点击关闭按钮关闭实时的检测任务
        # self.vid_source = int(self.init_vid_id)        # 重新初始化摄像头
        if str(self.vid_source).isdigit():
            self.vid_source = int(self.vid_source)
        self.webcam = True                             # 将实时摄像头设置为true
        print(f"当前实时源：{self.vid_source}")
        self.cap = cv2.VideoCapture(self.vid_source)   # 初始化摄像头的对象
        th = threading.Thread(target=self.detect_vid)  # 初始化视频检测线程
        th.start()                                     # 启动线程进行检测

    def open_mp4(self):
        """打开mp4文件上传"""
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            # 和上面open_cam的方法类似，只是在open_cam的基础上将摄像头的源改为mp4的文件
            self.webcam_detection_btn.setEnabled(False)
            self.mp4_detection_btn.setEnabled(False)
            self.vid_source = fileName
            self.webcam = False
            self.cap = cv2.VideoCapture(self.vid_source)
            th = threading.Thread(target=self.detect_vid)
            th.start()

    # 视频检测主函数
    def detect_vid(self):
        """检测视频文件，这里的视频文件包含了mp4格式的视频文件和摄像头形式的视频文件"""
        # model = self.model
        vid_i = 0
        track_history = defaultdict(lambda: [])
        while self.cap.isOpened():
            # Read a frame from the video
            success, frame = self.cap.read()
            if success:
                # Run YOLOv8 inference on the frame
                # 如果是检测，也就是没有开检测器的话，就按照正常的检测流程走，如果此时开启了追踪，则应该进入追踪的分支按照追踪走
                if self.config_track_value.currentText() == "不开启追踪":

                    results = self.model(frame, conf=self.conf_thres, iou=self.iou_thres, save_txt=self.save_txt, save_conf=self.save_conf, save_crop=self.save_crop)
                    # 这个位置需要添加一个追踪的功能
                    result = results[0]
                    img_array = result.plot()
                    # 检测 展示然后保存对应的图像结果
                    im0 = img_array
                    im_record = copy.deepcopy(im0)
                    resize_scale = self.output_size / im0.shape[0]
                    im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                    cv2.imwrite("images/tmp/single_result_vid.jpg", im0)
                    self.vid_img.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
                    time_re = str(time.strftime('result_%Y-%m-%d_%H-%M-%S_%A'))
                    if vid_i % self.vid_gap == 0:
                        cv2.imwrite("record/vid/{}.jpg".format(time_re), im_record)
                    result_names = result.names
                    result_nums = [0 for i in range(0, len(result_names))]
                    cls_ids = list(result.boxes.cls.cpu().numpy())
                    for cls_id in cls_ids:
                        result_nums[int(cls_id)] = result_nums[int(cls_id)] + 1
                    result_info = ""
                    for idx_cls, cls_num in enumerate(result_nums):
                        if cls_num > 0:
                            result_info = result_info + "{}:{}\n".format(result_names[idx_cls], cls_num)
                        # result_info = result_info + "{}:{}\n".format(result_names[idx_cls], cls_num)
                        # result_info = result_info + "{}:{}\n".format(result_names[idx_cls], cls_num)
                    self.vid_num_label.setText("当前检测结果：\n{}".format(result_info))
                    vid_i = vid_i + 1
                else:
                    results = self.model.track(frame,  conf=self.conf_thres, iou=self.iou_thres, save_txt=self.save_txt,
                                         save_conf=self.save_conf, save_crop=self.save_crop, tracker=self.config_track_value.currentText(), persist=True)
                    # 这个位置需要添加一个追踪的功能
                    result = results[0]
                    img_array = result.plot()
                    # 尝试向image array上绘制检测的结果
                    try:
                        # Get the boxes and track IDs
                        boxes = results[0].boxes.xywh.cpu()
                        track_ids = results[0].boxes.id.int().cpu().tolist()

                        # Plot the tracks
                        for box, track_id in zip(boxes, track_ids):
                            x, y, w, h = box
                            track = track_history[track_id]
                            track.append((float(x), float(y)))  # x, y center point
                            if len(track) > 30:  # retain 90 tracks for 90 frames
                                track.pop(0)

                            # Draw the tracking lines
                            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(img_array, [points], isClosed=False, color=(0, 0, 230),
                                          thickness=5)
                    except:
                        print("not got targets")
                    im0 = img_array
                    im_record = copy.deepcopy(im0)
                    resize_scale = self.output_size / im0.shape[0]
                    im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                    cv2.imwrite("images/tmp/single_result_vid.jpg", im0)
                    self.vid_img.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
                    time_re = str(time.strftime('result_%Y-%m-%d_%H-%M-%S_%A'))
                    if vid_i % self.vid_gap == 0:
                        cv2.imwrite("record/vid/{}.jpg".format(time_re), im_record)
                    result_names = result.names
                    result_nums = [0 for i in range(0, len(result_names))]
                    cls_ids = list(result.boxes.cls.cpu().numpy())
                    for cls_id in cls_ids:
                        result_nums[int(cls_id)] = result_nums[int(cls_id)] + 1
                    result_info = ""
                    for idx_cls, cls_num in enumerate(result_nums):
                        if cls_num > 0:
                            result_info = result_info + "{}:{}\n".format(result_names[idx_cls], cls_num)
                        # result_info = result_info + "{}:{}\n".format(result_names[idx_cls], cls_num)
                        # result_info = result_info + "{}:{}\n".format(result_names[idx_cls], cls_num)
                    self.vid_num_label.setText("当前检测结果：\n{}".format(result_info))
                    vid_i = vid_i + 1
            if cv2.waitKey(1) & self.stopEvent.is_set() == True:
                # 关闭并释放对应的视频资源
                self.stopEvent.clear()
                self.webcam_detection_btn.setEnabled(True)
                self.mp4_detection_btn.setEnabled(True)
                if self.cap is not None:
                    self.cap.release()
                    cv2.destroyAllWindows()
                self.reset_vid()
                break

    # 摄像头重置
    def reset_vid(self):
        """重置摄像头内容"""
        self.webcam_detection_btn.setEnabled(True)                      # 打开摄像头检测的按钮
        self.mp4_detection_btn.setEnabled(True)                         # 打开视频文件检测的按钮
        self.vid_img.setPixmap(QPixmap(IMAGE_LEFT_INIT))                # 重新设置视频检测页面的初始化图像
        # self.vid_source = int(self.init_vid_id)                         # 重新设置源视频源
        self.webcam = True                                              # 重新将摄像头设置为true
        self.vid_num_label.setText("当前检测结果：{}".format("等待检测"))   # 重新设置视频检测页面的文字内容

    def close_vid(self):
        """关闭摄像头"""
        self.stopEvent.set()
        self.reset_vid()


    def check_record(self):
        """打开历史记录文件夹"""
        os.startfile(osp.join(os.path.abspath(os.path.dirname(__file__)), "record"))

    def save_config_change(self):
        #
        print("保存配置修改的结果")
        try:
            self.output_size = int(self.config_output_size_value.text())
            self.vid_source = str(self.config_vid_source_value.text())
            print(f"源地址:{self.vid_source}")
            # 添加对vid_source的初始化
            # self.cap =  cv2.VideoCapture(str(self.vid_source))
            self.vid_gap = int(self.config_vid_gap_value.text())
            self.conf_thres = float(self.config_conf_thres_value.text())
            self.iou_thres = float(self.config_iou_thres_value.text())
            ###
            self.save_txt = self.config_save_txt_value.isChecked()
            self.save_conf = self.config_save_conf_value.isChecked()
            self.save_crop = self.config_save_crop_value.isChecked()

            # self.config_track_value.currentText()
            QMessageBox.information(self, "配置文件保存成功", "配置文件保存成功")
        except:
            QMessageBox.warning(self, "配置文件保存失败", "配置文件保存失败")



    def closeEvent(self, event):
        """用户退出事件"""
        reply = QMessageBox.question(self,
                                     'quit',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                # 退出之后一定要尝试释放摄像头资源，防止资源一直在线
                if self.cap is not None:
                    self.cap.release()
                    print("摄像头已释放")
            except:
                pass
            self.close()
            event.accept()
        else:
            event.ignore()
# 添加登录界面
class LoginWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        font_title = QFont('楷体', 16)
        self.setWindowTitle("识别系统登陆界面\n账号密码均为我qq，需要99调试请添加")
        self.resize(800, 600)
        mid_widget = QWidget()
        window_layout = QFormLayout()
        self.user_name = QLineEdit()
        self.u_password = QLineEdit()
        window_layout.addRow("账 号：", self.user_name)
        window_layout.addRow("密 码：", self.u_password)
        self.user_name.setEchoMode(QLineEdit.Normal)
        self.u_password.setEchoMode(QLineEdit.Password)
        mid_widget.setLayout(window_layout)
        # self.setBa
        # self.setObjectName("MainWindow")
        # self.setStyleSheet("#MainWindow{background-color:rgb(236,99,97)}")

        main_layout = QVBoxLayout()
        a = QLabel("😁😁😁😁😁😁😁😁😁😁😁😁\n欢迎使用基于YOLO11的识别系统\n 账号密码均为我QQ:3045834499"
                   "\n需要99调试请添加")
        a.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(a)
        main_layout.addWidget(mid_widget)

        login_button = QPushButton("立即登陆")
        # reg_button = QPushButton("注册用户")
        # reg_button.clicked.connect(self.reggg)
        login_button.clicked.connect(self.login)

        # main_layout.addWidget(reg_button)
        main_layout.addWidget(login_button)

        self.setLayout(main_layout)

        self.mainWindow = MainWindow()
        self.setFont(font_title)
        # self.regwindow = RegWindow()

    # mainWindow.show()

    def login(self):
        user_name = self.user_name.text()
        pwd = self.u_password.text()
        is_ok = (user_name == USERNAME) and (pwd == PASSWORD)
        # is_ok = is_correct(user_name, pwd)

        print(is_ok)
        if is_ok:
            self.mainWindow.show()
            self.close()
        else:
            QMessageBox.warning(self, "账号密码不匹配", "请输入正确的账号密码")


# todo 添加模型参数的修改，以及添加对文件夹图像的加载
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()  # 直接启动主窗口，跳过登录
    mainWindow.show()
    sys.exit(app.exec())