import sys
from pathlib import Path
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog, QLineEdit, QVBoxLayout, QHBoxLayout, QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from app.pipeline.detection import Detector
from app.pipeline.segmentation import Segmenter
from app.pipeline.radiomics import solid_ratio_otsu, extract_radiomics


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("肺结节检测分割与实性占比")
        self.image = None
        self.detector = None
        self.segmenter = None
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.btn_open = QPushButton("打开图像")
        self.btn_open.clicked.connect(self.open_image)
        self.det_model_edit = QLineEdit()
        self.det_model_edit.setPlaceholderText("检测模型路径 .pt")
        self.seg_model_edit = QLineEdit()
        self.seg_model_edit.setPlaceholderText("分割模型路径 .pt")
        self.seg_mode = QComboBox()
        self.seg_mode.addItems(["sam", "yolo-seg"])
        self.btn_init_models = QPushButton("初始化模型")
        self.btn_init_models.clicked.connect(self.init_models)
        self.btn_run = QPushButton("运行检测分割")
        self.btn_run.clicked.connect(self.run_pipeline)
        self.info_label = QLabel()
        top = QWidget()
        v = QVBoxLayout(top)
        hr = QHBoxLayout()
        hr.addWidget(self.det_model_edit)
        hr.addWidget(self.seg_model_edit)
        hr.addWidget(self.seg_mode)
        hr.addWidget(self.btn_init_models)
        v.addLayout(hr)
        v.addWidget(self.btn_open)
        v.addWidget(self.btn_run)
        v.addWidget(self.image_label)
        v.addWidget(self.info_label)
        self.setCentralWidget(top)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        self.image = img
        self.show_image(img)
        self.info_label.setText("")

    def init_models(self):
        det_path = self.det_model_edit.text().strip()
        seg_path = self.seg_model_edit.text().strip()
        mode = self.seg_mode.currentText()
        if Path(det_path).exists():
            self.detector = Detector(det_path)
        if Path(seg_path).exists():
            self.segmenter = Segmenter(seg_path, mode=mode)

    def run_pipeline(self):
        if self.image is None or self.detector is None or self.segmenter is None:
            return
        boxes = self.detector.predict(self.image)
        if not boxes:
            return
        cls, score, box = boxes[0]
        mask = self.segmenter.segment(self.image, box)
        if mask is None:
            return
        ratio = solid_ratio_otsu(self.image, mask)
        feats = extract_radiomics(self.image, mask) or {}
        vis = self.overlay(self.image, box, mask)
        self.show_image(vis)
        text = f"分类:{cls} 置信度:{score:.3f} 实性占比:{ratio:.3f} 特征:{len(feats)}"
        self.info_label.setText(text)

    def overlay(self, img: np.ndarray, box, mask: np.ndarray) -> np.ndarray:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        m = (mask > 0).astype(np.uint8) * 255
        color = np.zeros_like(vis)
        color[:, :, 2] = m
        vis = cv2.addWeighted(vis, 1.0, color, 0.4, 0)
        return vis

    def show_image(self, img: np.ndarray):
        if len(img.shape) == 2:
            h, w = img.shape
            q = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, _ = img.shape
            q = QImage(img.data, w, h, 3 * w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(q).scaled(self.image_label.width() if self.image_label.width() > 0 else w, self.image_label.height() if self.image_label.height() > 0 else h, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pix)


def run():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1024, 768)
    w.show()
    sys.exit(app.exec_())
