import sys
from pathlib import Path
import numpy as np
import cv2
import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
                             QFileDialog, QVBoxLayout, QHBoxLayout, QComboBox, QMessageBox)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

# Add project root to sys.path to allow imports from YOLO_PROJECT package
# This assumes the script is run from a location where YOLO_PROJECT is importable 
# or we find the root relative to this file.
FILE_PATH = Path(__file__).resolve()
APP_DIR = FILE_PATH.parent
INNER_DIR = APP_DIR.parent           # .../YOLO_PROJECT/YOLO_PROJECT
OUTER_DIR = INNER_DIR.parent         # .../YOLO_PROJECT
ROOT_DIR = OUTER_DIR.parent          # .../ (The actual project root G:\project\yolo11-chest_editted)

# We need OUTER_DIR in sys.path so we can do 'from YOLO_PROJECT.pipeline...'
# because YOLO_PROJECT package corresponds to INNER_DIR
if str(OUTER_DIR) not in sys.path:
    sys.path.append(str(OUTER_DIR))

from ultralytics import YOLO
from YOLO_PROJECT.pipeline.dicom import load_series, window_normalize
from YOLO_PROJECT.pipeline.medsam import MedSAMSegmenter
from YOLO_PROJECT.pipeline.ctr import ctr_ratio
import nibabel as nib

class InferenceWorker(QThread):
    """
    Worker thread for running YOLO detection and MedSAM segmentation
    to keep the UI responsive.
    """
    finished = pyqtSignal(object, object, float, float) # mask, box, ctr, ggo_ratio
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, yolo_path, medsam_path, image_bgr, hu_image, spacing, conf_threshold=0.01, device="cpu"):
        super().__init__()
        self.yolo_path = yolo_path
        self.medsam_path = medsam_path
        self.image_bgr = image_bgr
        self.hu_image = hu_image
        self.spacing = spacing
        self.conf_threshold = conf_threshold
        self.device = device
        self.is_running = True

    def run(self):
        try:
            self.progress.emit("正在加载 YOLO 模型...")
            # Load YOLO
            det = YOLO(self.yolo_path)
            
            self.progress.emit("正在进行目标检测...")
            # Predict with YOLO
            # Note: We remove device="cpu" restriction to allow GPU if available
            r = det.predict(source=[self.image_bgr], imgsz=640, conf=self.conf_threshold, device=self.device, verbose=False)
            
            boxes = np.empty((0, 4), dtype=np.float32)
            if len(r) and len(r[0].boxes):
                confs = r[0].boxes.conf.cpu().numpy()
                # Debug info
                # print(f"Detected {len(confs)} boxes. Max conf: {np.max(confs):.4f}")
                boxes = r[0].boxes.xyxy.cpu().numpy().astype(np.float32)
            
            if boxes.shape[0] == 0:
                self.error.emit(f"未检测到目标 (Conf: {self.conf_threshold})")
                return

            # Pick the best box (highest confidence, assumes sorted or take the first)
            # YOLO results are usually sorted by confidence descending
            box = boxes[0]

            self.progress.emit("正在加载 MedSAM 模型...")
            # Load MedSAM
            # We instantiate it here to avoid passing complex objects across threads, 
            # though caching it would be better for repeated runs. 
            # For now, safe and simple: load in worker.
            seg = MedSAMSegmenter(self.medsam_path, model_type="vit_b", device=self.device)

            self.progress.emit("正在进行图像分割...")
            m = seg.predict(self.image_bgr, box)

            self.progress.emit("正在计算指标...")
            # CTR calculation
            c = ctr_ratio(self.hu_image, m, solid_threshold=-300.0, spacing=(self.spacing[0], self.spacing[1]))
            ggo_ratio = 1.0 - c

            self.finished.emit(m, box, c, ggo_ratio)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(f"运行出错: {str(e)}")

class Viewer(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(512, 512)
        self.img = None
    def set_image(self, bgr: np.ndarray):
        self.img = bgr
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(self.width(), self.height(), Qt.AspectRatioMode.KeepAspectRatio))
    def wheelEvent(self, e):
        w = self.window()
        if hasattr(w, 'on_wheel'):
            w.on_wheel(e.angleDelta().y())

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GGO Assist (Optimized)")
        self.setAcceptDrops(True)
        self.left = Viewer()
        self.right = Viewer()
        self.btn_open = QPushButton("打开DICOM序列")
        self.btn_run = QPushButton("自动标注")
        self.btn_save = QPushButton("保存结果")
        
        self.combo_yolo = QComboBox()
        self.combo_medsam = QComboBox()
        self.btn_refresh = QPushButton("刷新权重")
        
        self.conf_spin = QComboBox()
        for v in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
            self.conf_spin.addItem(f"Conf: {v}", v)
        self.conf_spin.setCurrentIndex(0) # Default to 0.01
        
        l = QHBoxLayout()
        l.addWidget(self.left)
        l.addWidget(self.right)
        
        c = QWidget()
        v = QVBoxLayout()
        v.addLayout(l)
        
        h_weights = QHBoxLayout()
        h_weights.addWidget(QLabel("YOLO:"))
        h_weights.addWidget(self.combo_yolo)
        h_weights.addWidget(QLabel("MedSAM:"))
        h_weights.addWidget(self.combo_medsam)
        h_weights.addWidget(self.conf_spin)
        h_weights.addWidget(self.btn_refresh)
        v.addLayout(h_weights)
        
        v.addWidget(self.btn_open)
        v.addWidget(self.btn_run)
        v.addWidget(self.btn_save)
        c.setLayout(v)
        self.setCentralWidget(c)
        
        self.btn_open.clicked.connect(self.on_open)
        self.btn_run.clicked.connect(self.on_run)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_refresh.clicked.connect(self.refresh_weights)
        
        self.series = None
        self.spacing = None
        self.meta = None
        self.idx = 0
        
        # Results cache
        self.last_mask = None
        self.last_box = None
        
        self.worker = None
        self.device = self.pick_device()
        print(f"Detected Device: {self.device}")
        
        self.refresh_weights()

    def refresh_weights(self):
        # Dynamically find run directories
        # ROOT_DIR is the project root (contains outer YOLO_PROJECT)
        # INNER_DIR is the inner YOLO_PROJECT folder
        
        search_roots = [
            ROOT_DIR / "runs",
            INNER_DIR / "runs",
            ROOT_DIR # In case weights are in root
        ]
        
        self.combo_yolo.clear()
        yolo_files = []
        for root in search_roots:
            if root.exists():
                yolo_files.extend(list(root.glob("**/weights/best.pt")))
        
        # Sort by mtime
        yolo_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        # Deduplicate by path
        seen_paths = set()
        unique_yolo_files = []
        for f in yolo_files:
            if str(f) not in seen_paths:
                seen_paths.add(str(f))
                unique_yolo_files.append(f)

        for f in unique_yolo_files:
            # Show relative path if possible, else name
            try:
                display_name = f.relative_to(ROOT_DIR)
            except ValueError:
                display_name = f.name
            self.combo_yolo.addItem(str(display_name), str(f))
            
        self.combo_medsam.clear()
        medsam_files = []
        for root in search_roots:
            if root.exists():
                medsam_files.extend(list(root.glob("**/weights/best.pth"))) # sometimes .pth
                medsam_files.extend(list(root.glob("**/sam_b.pt")))      # standard SAM
        
        medsam_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
         # Deduplicate
        seen_paths = set()
        unique_medsam_files = []
        for f in medsam_files:
            if str(f) not in seen_paths:
                seen_paths.add(str(f))
                unique_medsam_files.append(f)

        for f in unique_medsam_files:
            try:
                display_name = f.relative_to(ROOT_DIR)
            except ValueError:
                display_name = f.name
            self.combo_medsam.addItem(str(display_name), str(f))

    def pick_device(self):
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
    
    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if not urls:
            return
        d = urls[0].toLocalFile()
        self.load_series_dir(d)

    def on_open(self):
        d = QFileDialog.getExistingDirectory(self, "选择DICOM序列文件夹")
        if not d:
            return
        self.load_series_dir(d)

    def load_series_dir(self, d):
        arr, spacing, meta = load_series(d)
        if arr is None:
            QMessageBox.warning(self, "Load Error", "Failed to load DICOM series.")
            return
        self.series = arr
        self.spacing = spacing
        self.meta = meta
        self.idx = arr.shape[0] // 2
        self.update_display()

    def update_display(self):
        if self.series is None:
            return
        hu = self.series[self.idx]
        img = window_normalize(hu)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.left.set_image(bgr)
        # If we have a cached result for this slice, show it (optional, for now just clear right)
        # Simpler: just clear right or show original
        self.right.set_image(bgr) 

    def on_run(self):
        if self.series is None:
            return
        
        # Get selected weights and confidence
        yolo_path = self.combo_yolo.currentData()
        medsam_path = self.combo_medsam.currentData()
        conf_val = self.conf_spin.currentData()
        
        if not yolo_path:
            # Try to ask user
            f, _ = QFileDialog.getOpenFileName(self, "未找到YOLO权重，请手动选择", filter="*.pt")
            if f: yolo_path = f
            else: return

        if not medsam_path:
             f, _ = QFileDialog.getOpenFileName(self, "未找到MedSAM权重，请手动选择", filter="*.pt;;*.pth")
             if f: medsam_path = f
             else: return

        # Prepare data for worker
        hu = self.series[self.idx]
        img = window_normalize(hu)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Disable UI
        self.btn_run.setEnabled(False)
        self.btn_run.setText("初始化...")

        # Start Worker
        self.worker = InferenceWorker(yolo_path, medsam_path, bgr, hu, self.spacing, conf_val, self.device)
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()

    @pyqtSlot(str)
    def on_worker_progress(self, msg):
        self.btn_run.setText(msg)

    @pyqtSlot(object, object, float, float)
    def on_worker_finished(self, mask, box, c, ggo_ratio):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("自动标注")
        self.last_mask = mask
        self.last_box = box
        
        # Visualization
        hu = self.series[self.idx]
        img = window_normalize(hu)
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Draw Mask
        # Red for general mask
        overlay[mask > 0] = (0.3 * overlay[mask > 0] + 0.7 * np.array([0, 0, 255])).astype(np.uint8)
        
        # Highlight solid part within the mask in a different color (Yellow)
        # Use -300 HU as threshold for solid component
        solid_mask = (hu > -300.0) & (mask > 0)
        overlay[solid_mask] = (0.3 * overlay[solid_mask] + 0.7 * np.array([0, 255, 255])).astype(np.uint8)

        # Draw Box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        info_text = f"CTR: {c:.2f} | GGO: {ggo_ratio*100:.1f}%"
        cv2.putText(overlay, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        self.right.set_image(overlay)

    @pyqtSlot(str)
    def on_worker_error(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("自动标注")
        QMessageBox.critical(self, "Error", msg)

    def on_save(self):
        if self.series is None or self.last_mask is None:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not out_dir:
            return
        p = Path(out_dir) / "mask.npy"
        np.save(str(p), self.last_mask)
        
        # Save NIfTI
        aff = np.eye(4, dtype=np.float32)
        aff[0, 0] = float(self.spacing[0]) if self.spacing else 1.0
        aff[1, 1] = float(self.spacing[1]) if self.spacing else 1.0
        nii = nib.Nifti1Image(self.last_mask.astype(np.uint8), affine=aff)
        nib.save(nii, str(Path(out_dir) / "mask.nii.gz"))
        
        QMessageBox.information(self, "Saved", f"Results saved to {out_dir}")

    def on_wheel(self, delta):
        if self.series is None:
            return
        self.idx = int(np.clip(self.idx + (1 if delta < 0 else -1), 0, self.series.shape[0] - 1))
        self.update_display()

def main():
    app = QApplication(sys.argv)
    w = Main()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
