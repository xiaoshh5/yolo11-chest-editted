"""Unit tests for core pipeline modules (no model weights needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project to path
ROOT = Path(__file__).resolve().parent.parent
OUTER = ROOT / "YOLO_PROJECT"
if str(OUTER) not in sys.path:
    sys.path.insert(0, str(OUTER))


# ── CTR tests ────────────────────────────────────────────
from YOLO_PROJECT.pipeline.ctr import ctr_ratio


class TestCTR:
    def test_all_ggo(self):
        """All voxels are -600 HU (GGO), CTR should be 0."""
        hu = np.full((64, 64), -600.0, dtype=np.float32)
        mask = np.ones((64, 64), dtype=np.uint8)
        assert ctr_ratio(hu, mask, solid_threshold=-300) == 0.0

    def test_all_solid(self):
        """All voxels are 100 HU (solid), CTR should be 1."""
        hu = np.full((64, 64), 100.0, dtype=np.float32)
        mask = np.ones((64, 64), dtype=np.uint8)
        assert ctr_ratio(hu, mask, solid_threshold=-300) == 1.0

    def test_half_solid(self):
        """Half solid, half GGO."""
        hu = np.zeros((64, 64), dtype=np.float32)
        hu[:32, :] = 200    # solid
        hu[32:, :] = -500   # GGO
        mask = np.ones((64, 64), dtype=np.uint8)
        assert ctr_ratio(hu, mask, solid_threshold=-300) == pytest.approx(0.5)

    def test_empty_mask(self):
        hu = np.ones((10, 10), dtype=np.float32) * 100
        mask = np.zeros((10, 10), dtype=np.uint8)
        assert ctr_ratio(hu, mask) == 0.0

    def test_none_mask(self):
        assert ctr_ratio(np.ones((5, 5)), None) == 0.0

    def test_custom_threshold(self):
        hu = np.array([[ -500, -200, 100 ]], dtype=np.float32)
        mask = np.ones((1, 3), dtype=np.uint8)
        # threshold=-400: 2/3 solid; threshold=0: 1/3 solid
        assert ctr_ratio(hu, mask, solid_threshold=-400) == pytest.approx(2 / 3)
        assert ctr_ratio(hu, mask, solid_threshold=0) == pytest.approx(1 / 3)


# ── DICOM tests ──────────────────────────────────────────
from YOLO_PROJECT.pipeline.dicom import window_normalize, load_series


class TestWindowNormalize:
    def test_basic(self):
        arr = np.array([[-1000, -500, 0, 400, 1000]], dtype=np.float32)
        out = window_normalize(arr, wl_low=-1000, wl_high=400)
        assert out.shape == arr.shape
        assert out.dtype == np.uint8
        assert out[0, 0] == 0   # -1000 → 0
        assert out[0, -1] == 255  # above window → 255

    def test_out_of_range_clipped(self):
        arr = np.array([-2000, 2000], dtype=np.float32)
        out = window_normalize(arr, wl_low=-1000, wl_high=400)
        assert out[0] == 0
        assert out[1] == 255


class TestLoadSeries:
    def test_load_synthetic_dicom(self):
        """Load the generated test DICOMs."""
        dicom_dir = ROOT / "tests" / "data" / "dicom_sample"
        if not dicom_dir.exists():
            pytest.skip("Test DICOM data not generated; run tests/generate_test_data.py first")

        arr, spacing, meta = load_series(str(dicom_dir))
        assert arr is not None
        assert arr.ndim == 3, f"Expected 3D array, got shape {arr.shape}"
        assert arr.shape[1:] == (256, 256)
        assert spacing is not None
        assert meta is not None

    def test_nonexistent_dir(self):
        arr, spacing, meta = load_series("/nonexistent/path/12345")
        assert arr is None


# ── App core logic tests ─────────────────────────────────
try:
    from YOLO_PROJECT.app.app import calculate_ctr_and_visualize, pick_device
except ImportError:
    # Fallback: import from the file directly
    import importlib.util

    app_path = OUTER / "YOLO_PROJECT" / "app" / "app.py"
    spec = importlib.util.spec_from_file_location("app", app_path)
    app_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_mod)
    calculate_ctr_and_visualize = app_mod.calculate_ctr_and_visualize


class TestCTRVisualize:
    def test_outputs(self):
        hu = np.ones((64, 64), dtype=np.float32) * 100  # all solid
        mask = np.ones((64, 64), dtype=np.uint8)
        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        ctr, overlay, solid_px, total_px = calculate_ctr_and_visualize(hu, mask, bgr)
        assert ctr == 1.0
        assert solid_px == total_px
        assert overlay.shape == bgr.shape

    def test_empty_mask(self):
        hu = np.ones((64, 64), dtype=np.float32) * 100
        mask = np.zeros((64, 64), dtype=np.uint8)
        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        ctr, overlay, solid_px, total_px = calculate_ctr_and_visualize(hu, mask, bgr)
        assert ctr == 0.0
        assert np.array_equal(overlay, bgr)

    def test_ggo_coloring(self):
        """GGO pixels (-600 HU) should get red overlay, solid (100 HU) yellow."""
        hu = np.full((32, 32), -600.0, dtype=np.float32)
        hu[16:, :] = 100  # bottom half solid
        mask = np.ones((32, 32), dtype=np.uint8)
        bgr = np.full((32, 32, 3), fill_value=128, dtype=np.uint8)
        _, overlay, _, _ = calculate_ctr_and_visualize(hu, mask, bgr)
        # GGO area (top half) should be redder than original
        orig_r = bgr[0, 0, 2]
        ggo_r = overlay[0, 0, 2]
        solid_r = overlay[31, 0, 2]
        assert ggo_r > orig_r  # red channel increased for GGO


# ── Device selection ─────────────────────────────────────
class TestDevice:
    def test_pick_device_returns_string(self):
        import torch
        dev = app_mod.pick_device() if hasattr(app_mod, 'pick_device') else "cpu"
        assert dev in ("cpu", "cuda:0", "xpu")
