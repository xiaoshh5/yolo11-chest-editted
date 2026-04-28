import sys
import time
from pathlib import Path
import numpy as np
import cv2
import nibabel as nib
import torch

# --- Path Setup ---
FILE_PATH = Path(__file__).resolve()
APP_DIR = FILE_PATH.parent
INNER_DIR = APP_DIR.parent           # .../YOLO_PROJECT/YOLO_PROJECT
OUTER_DIR = INNER_DIR.parent         # .../YOLO_PROJECT
ROOT_DIR = OUTER_DIR.parent          # .../ (The actual project root)

# We need OUTER_DIR in sys.path so we can do 'from YOLO_PROJECT.pipeline...'
if str(OUTER_DIR) not in sys.path:
    sys.path.append(str(OUTER_DIR))

# PyQt6 Imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
                             QFileDialog, QVBoxLayout, QHBoxLayout, QComboBox, 
                             QProgressBar, QMessageBox, QGroupBox)
from PyQt6.QtGui import QImage, QPixmap, QAction
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

# AI Models
from ultralytics import YOLO
try:
    from YOLO_PROJECT.pipeline.dicom import load_series, window_normalize
    from YOLO_PROJECT.pipeline.medsam import MedSAMSegmenter
except ImportError as e:
    print(f"Warning: Custom YOLO_PROJECT modules import error: {e}")
    # Fallback or exit? For now let it fail to alert user if paths are wrong

# --- 算法核心逻辑 ---

def calculate_ctr_and_visualize(hu_img, mask, bgr_img, solid_thresh=-300.0):
    """
    计算CTR并生成可视化覆盖层
    :param hu_img: 原始CT值 (numpy array)
    :param mask: 分割掩码 (0/1)
    :param bgr_img: 用于绘图的底图
    :param solid_thresh: 实性成分阈值，默认 -300 HU
    :return: (ctr_value, overlay_img, solid_area_pixels, total_area_pixels)
    """
    if mask is None or np.sum(mask) == 0:
        return 0.0, bgr_img, 0, 0

    # 1. 提取感兴趣区域 (ROI)
    # 逻辑：在 mask 为 1 的区域内，HU > -300 的是实性，否则是磨玻璃
    roi_hu = hu_img[mask > 0]
    
    total_area = len(roi_hu)
    solid_pixels = roi_hu[roi_hu > solid_thresh]
    solid_area = len(solid_pixels)
    
    ctr = solid_area / total_area if total_area > 0 else 0.0

    # 2. 可视化绘制
    overlay = bgr_img.copy()
    
    # 定义掩码区域
    mask_bool = mask > 0
    solid_mask = (hu_img > solid_thresh) & mask_bool
    ggo_mask = (hu_img <= solid_thresh) & mask_bool
    
    # 颜色定义 (BGR)
    COLOR_GGO = np.array([0, 0, 255])    # 红色: 纯磨玻璃
    COLOR_SOLID = np.array([0, 255, 255]) # 黄色: 实性部分
    
    # 混合绘制: 磨玻璃部分
    overlay[ggo_mask] = (0.4 * overlay[ggo_mask] + 0.6 * COLOR_GGO).astype(np.uint8)
    # 混合绘制: 实性部分
    overlay[solid_mask] = (0.4 * overlay[solid_mask] + 0.6 * COLOR_SOLID).astype(np.uint8)
    
    return ctr, overlay, solid_area, total_area

# --- 工作线程 (解决界面卡顿) ---

class AnalysisThread(QThread):
    # 信号：发送结果回主界面
    result_ready = pyqtSignal(object) 
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.yolo_model = None
        self.medsam_model = None
        self.current_yolo_path = None
        self.current_medsam_path = None
        
        # 任务参数
        self.img_bgr = None
        self.img_hu = None
        self.conf = 0.25
        self.device = "cpu" # 默认配置
        self.spacing = (1.0, 1.0)

    def set_models(self, yolo_path, medsam_path):
        self.req_yolo_path = yolo_path
        self.req_medsam_path = medsam_path

    def set_data(self, img_bgr, img_hu, conf, device, spacing):
        self.img_bgr = img_bgr
        self.img_hu = img_hu
        self.conf = conf
        self.device = device
        self.spacing = spacing

    def run(self):
        try:
            # 1. 懒加载/热切换模型
            if self.yolo_model is None or self.current_yolo_path != self.req_yolo_path:
                self.status_update.emit(f"正在加载 YOLO: {Path(self.req_yolo_path).name}...")
                self.yolo_model = YOLO(self.req_yolo_path)
                self.current_yolo_path = self.req_yolo_path

            if self.medsam_model is None or self.current_medsam_path != self.req_medsam_path:
                self.status_update.emit(f"正在加载 MedSAM: {Path(self.req_medsam_path).name}...")
                self.medsam_model = MedSAMSegmenter(self.req_medsam_path, model_type="vit_b", device=self.device)
                self.current_medsam_path = self.req_medsam_path
            
            # Ensure MedSAM is on correct device (if it was loaded on CPU before)
            if self.medsam_model.device != self.device:
                self.medsam_model.model.to(self.device)
                self.medsam_model.device = self.device
                self.medsam_model.predictor.model = self.medsam_model.model

            # 2. 运行 YOLO
            self.status_update.emit("正在检测目标...")
            # [OPTIMIZED] Use self.device instead of hardcoded 'cpu'
            results = self.yolo_model.predict(source=[self.img_bgr], imgsz=640, conf=self.conf, device=self.device, verbose=False)
            
            if not results or len(results[0].boxes) == 0:
                self.result_ready.emit(None) # 无目标
                return

            # 取置信度最高的框
            boxes = results[0].boxes.xyxy.cpu().numpy()
            best_box = boxes[0] # [x1, y1, x2, y2]

            # 3. 运行 MedSAM
            self.status_update.emit("正在分割轮廓...")
            mask = self.medsam_model.predict(self.img_bgr, best_box)

            # 4. 计算 CTR
            self.status_update.emit("正在计算实占比...")
            ctr, overlay, solid_area, total_area = calculate_ctr_and_visualize(
                self.img_hu, mask, self.img_bgr, solid_thresh=-300.0
            )

            # 绘制边界框
            x1, y1, x2, y2 = map(int, best_box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 打包结果
            res = {
                "ctr": ctr,
                "mask": mask,
                "box": best_box,
                "overlay": overlay,
                "solid_area": solid_area,
                "total_area": total_area
            }
            self.result_ready.emit(res)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))

# --- UI 组件 ---

class Viewer(QLabel):
    def __init__(self, text="无图像"):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("border: 1px solid #444; background-color: #222; color: #888;")
        self.setText(text)
        self.img = None

    def set_image(self, bgr: np.ndarray):
        self.img = bgr
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.width(), self.height(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))
        
    def resizeEvent(self, event):
        if self.img is not None:
            self.set_image(self.img) # 重新缩放
        super().resizeEvent(event)

    def wheelEvent(self, e):
        w = self.window()
        if hasattr(w, 'on_wheel'):
            w.on_wheel(e.angleDelta().y())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GGO Assist Pro - 智能肺结节分析")
        self.resize(1200, 800)
        self.setAcceptDrops(True)

        # 数据状态
        self.series = None
        self.spacing = None
        self.meta = None
        self.current_idx = 0
        self.last_result = None # 存储mask用于保存
        
        # 初始化界面
        self.init_ui()
        
        # 初始化后台线程
        self.worker = AnalysisThread()
        self.worker.result_ready.connect(self.on_analysis_finished)
        self.worker.error_occurred.connect(self.on_analysis_error)
        self.worker.status_update.connect(self.update_status)

        # 尝试自动扫描权重
        self.scan_weights()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 1. 顶部查看器
        self.left_view = Viewer("原始影像 (DICOM)")
        self.right_view = Viewer("AI分析结果 (Overlay)")
        
        view_layout = QHBoxLayout()
        view_layout.addWidget(self.left_view, 1)
        view_layout.addWidget(self.right_view, 1)
        
        # 2. 控制面板
        control_panel = QGroupBox("控制面板")
        control_layout = QHBoxLayout()
        
        # 权重选择
        self.combo_yolo = QComboBox()
        self.combo_medsam = QComboBox()
        self.btn_refresh = QPushButton("刷新权重")
        
        # 参数配置
        self.combo_conf = QComboBox()
        # 添加更低的置信度选项，因为某些模型（如 train16）可能在推理时得分较低
        conf_values = [0.01, 0.05, 0.1, 0.25, 0.4, 0.6]
        for v in conf_values:
            self.combo_conf.addItem(f"Conf: {v}", v)
        # 默认设为 0.05
        self.combo_conf.setCurrentIndex(1) 
        
        control_layout.addWidget(QLabel("YOLO:"))
        control_layout.addWidget(self.combo_yolo, 1)
        control_layout.addWidget(QLabel("MedSAM:"))
        control_layout.addWidget(self.combo_medsam, 1)
        control_layout.addWidget(self.combo_conf)
        control_layout.addWidget(self.btn_refresh)
        control_panel.setLayout(control_layout)
        
        # 3. 操作按钮
        btn_layout = QHBoxLayout()
        self.btn_open = QPushButton("📂 打开文件夹")
        self.btn_run = QPushButton("⚡ 自动标注 (Auto)")
        self.btn_save = QPushButton("💾 保存结果")
        self.lbl_status = QLabel("就绪")
        
        # 设置大一点的按钮样式
        for btn in [self.btn_open, self.btn_run, self.btn_save]:
            btn.setMinimumHeight(40)
        
        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.lbl_status)
        btn_layout.addStretch()

        # 4. 总体布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(view_layout, 1)
        main_layout.addWidget(control_panel)
        main_layout.addLayout(btn_layout)
        main_widget.setLayout(main_layout)
        
        # 绑定事件
        self.btn_open.clicked.connect(self.on_open_folder)
        self.btn_run.clicked.connect(self.on_run_analysis)
        self.btn_save.clicked.connect(self.on_save_result)
        self.btn_refresh.clicked.connect(self.scan_weights)

    def scan_weights(self):
        """扫描当前目录及子目录下的权重文件"""
        self.combo_yolo.clear()
        self.combo_medsam.clear()
        
        # [OPTIMIZED] Robust path finding
        roots_to_scan = [
            ROOT_DIR / "runs",
            INNER_DIR / "runs",
            ROOT_DIR,
            INNER_DIR
        ]
        
        yolo_files = {} # path_abs -> display_name
        sam_files = {}  # path_abs -> display_name

        def add_weights(root_dir, patterns, target_dict, is_sam=False):
            if not root_dir.exists(): return
            for pattern in patterns:
                for p in root_dir.glob(pattern):
                    # 忽略无关目录
                    if any(part in p.parts for part in ['venv', 'datasets', 'archives', '.git']):
                        continue
                    
                    p_abs = p.resolve()
                    if p_abs in target_dict:
                        continue
                        
                    if is_sam:
                        name_lower = p.name.lower()
                        if not ("sam" in name_lower or "vit" in name_lower or "best" in name_lower):
                            continue

                    # 生成显示名称
                    try:
                        rel_path = p.relative_to(ROOT_DIR) # Relative to actual project root
                        display_name = str(rel_path)
                    except Exception:
                        display_name = p.name
                    
                    target_dict[p_abs] = display_name

        # 1. 扫描 YOLO
        yolo_patterns = ["**/weights/*.pt", "**/best.pt"]
        for r in roots_to_scan:
            add_weights(r, yolo_patterns, yolo_files)
        
        # 2. 扫描 MedSAM
        sam_patterns = ["**/weights/*.pth", "**/sam*.pt", "**/*.pth"]
        for r in roots_to_scan:
            add_weights(r, sam_patterns, sam_files, is_sam=True)

        # 排序：最近修改的在前
        def sort_key(item):
            try:
                mtime = item[0].stat().st_mtime
                return mtime
            except:
                return 0

        # reversed = newest first
        sorted_yolo = sorted(yolo_files.items(), key=sort_key, reverse=True)
        sorted_sam = sorted(sam_files.items(), key=sort_key, reverse=True)

        # 添加到 ComboBox
        for path, name in sorted_yolo:
            self.combo_yolo.addItem(name, str(path))
        for path, name in sorted_sam:
            self.combo_medsam.addItem(name, str(path))

        if not yolo_files: self.combo_yolo.addItem("未找到YOLO权重", None)
        if not sam_files: self.combo_medsam.addItem("未找到MedSAM权重", None)

    def pick_device(self):
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available(): 
            return "cuda:0"
        return "cpu"

    def update_image_views(self):
        if self.series is None: return
        
        hu = self.series[self.current_idx]
        img_gray = window_normalize(hu)
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        
        self.left_view.set_image(img_bgr)
        # 如果切换切片，右侧暂时显示原图，直到再次点击运行
        self.right_view.set_image(img_bgr)
        
        self.setWindowTitle(f"GGO Assist Pro - Slice: {self.current_idx + 1}/{self.series.shape[0]} | {self.pick_device()}")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if not urls:
            return
        d = urls[0].toLocalFile()
        self._load_dicom_folder(d)

    def _load_dicom_folder(self, d):
        self.lbl_status.setText("正在加载DICOM...")
        QApplication.processEvents()
        try:
            arr, spacing, meta = load_series(d)
            if arr is None: raise ValueError("无法读取DICOM")
            self.series = arr
            self.spacing = spacing
            self.meta = meta
            self.current_idx = arr.shape[0] // 2
            self.update_image_views()
            self.lbl_status.setText(f"加载成功: {arr.shape[0]} 层")
            return True
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败: {str(e)}")
            self.lbl_status.setText("加载失败")
            return False

    def on_open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "选择DICOM文件夹")
        if not d: return
        self._load_dicom_folder(d)

    def on_wheel(self, delta):
        if self.series is None: return
        step = 1 if delta < 0 else -1
        new_idx = self.current_idx + step
        if 0 <= new_idx < self.series.shape[0]:
            self.current_idx = new_idx
            self.update_image_views()
            # 切片改变时，清空上次的分析结果，避免误导
            self.last_result = None 

    def on_run_analysis(self):
        if self.series is None:
            QMessageBox.warning(self, "提示", "请先打开DICOM图像")
            return
        
        yolo_path = self.combo_yolo.currentData()
        medsam_path = self.combo_medsam.currentData()
        
        if not yolo_path or not medsam_path:
            QMessageBox.warning(self, "配置缺失", "请确保选择了YOLO和MedSAM的权重文件")
            return

        # 锁定按钮
        self.btn_run.setEnabled(False)
        self.btn_open.setEnabled(False)
        
        # 准备数据
        hu_slice = self.series[self.current_idx]
        img_gray = window_normalize(hu_slice)
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        conf = self.combo_conf.currentData()
        
        # 配置线程并启动
        self.worker.set_models(yolo_path, medsam_path)
        self.worker.set_data(img_bgr, hu_slice, conf, self.pick_device(), self.spacing)
        self.worker.start()

    @pyqtSlot(str)
    def update_status(self, msg):
        self.lbl_status.setText(msg)

    @pyqtSlot(str)
    def on_analysis_error(self, err_msg):
        self.btn_run.setEnabled(True)
        self.btn_open.setEnabled(True)
        self.lbl_status.setText("错误")
        QMessageBox.critical(self, "运行出错", err_msg)

    @pyqtSlot(object)
    def on_analysis_finished(self, result):
        self.btn_run.setEnabled(True)
        self.btn_open.setEnabled(True)
        
        if result is None:
            self.lbl_status.setText("未检测到结节")
            # QMessageBox.information(self, "结果", "当前切片未检测到目标。")
            return
            
        self.lbl_status.setText(f"分析完成 | CTR: {result['ctr']:.2f}")
        self.last_result = result
        
        # 更新右侧视图
        overlay = result['overlay']
        
        # 在图片上写详细信息
        info_text = [
            f"CTR: {result['ctr']:.2f}",
            f"GGO Ratio: {(1-result['ctr'])*100:.1f}%",
            f"Solid Area: {result['solid_area']} px"
        ]
        
        y0 = 30
        for line in info_text:
            cv2.putText(overlay, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlay, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y0 += 25
            
        self.right_view.set_image(overlay)

    def on_save_result(self):
        if self.last_result is None or self.series is None:
            QMessageBox.warning(self, "提示", "没有可保存的结果")
            return
            
        out_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not out_dir: return
        
        ts = int(time.time())
        base_name = f"result_{ts}"
        save_path = Path(out_dir)
        
        try:
            # 保存 NPY
            mask = self.last_result['mask']
            np.save(save_path / f"{base_name}_mask.npy", mask)
            
            # 保存 NIfTI
            aff = np.eye(4)
            if self.spacing:
                aff[0,0], aff[1,1] = self.spacing[0], self.spacing[1]
            
            nii = nib.Nifti1Image(mask.astype(np.uint8), affine=aff)
            nib.save(nii, save_path / f"{base_name}_mask.nii.gz")
            
            # 保存截图
            cv2.imwrite(str(save_path / f"{base_name}_overlay.jpg"), self.last_result['overlay'])
            
            QMessageBox.information(self, "成功", f"文件已保存至:\n{out_dir}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
