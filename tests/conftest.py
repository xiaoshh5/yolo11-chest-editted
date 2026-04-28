"""Shared fixtures for pipeline tests."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTER = ROOT / "YOLO_PROJECT"

if str(OUTER) not in sys.path:
    sys.path.insert(0, str(OUTER))


@pytest.fixture
def sample_hu_slice():
    """A 64x64 CT slice with -1000 to 400 HU range."""
    rng = np.random.default_rng(1234)
    return rng.uniform(-1000, 400, (64, 64)).astype(np.float32)


@pytest.fixture
def sample_mask():
    """A circular mask on a 64x64 image."""
    yy, xx = np.ogrid[:64, :64]
    return ((yy - 32) ** 2 + (xx - 32) ** 2 < 16 ** 2).astype(np.uint8)


@pytest.fixture
def sample_bgr():
    """64x64 BGR image."""
    return np.zeros((64, 64, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def generate_data():
    """Ensure test data exists before running any tests."""
    import importlib.util

    gen_path = HERE / "generate_test_data.py"
    spec = importlib.util.spec_from_file_location("generate_test_data", gen_path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.make_dicom_sample()
    gen.make_sample_dataset()
