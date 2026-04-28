import os
from pathlib import Path
import numpy as np
import pydicom
import SimpleITK as sitk

def load_series(path: str):
    p = Path(path)
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(p))
    if not series_ids:
        files = sorted([str(x) for x in p.glob("*.dcm")])
        if not files:
            return None, None, None
        reader.SetFileNames(files)
    else:
        files = reader.GetGDCMSeriesFileNames(str(p), series_ids[0])
        reader.SetFileNames(files)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    return arr, spacing, {"origin": origin, "direction": direction}

def to_hu(dcm: pydicom.dataset.FileDataset):
    img = dcm.pixel_array.astype(np.int16)
    intercept = dcm.RescaleIntercept if hasattr(dcm, "RescaleIntercept") else 0
    slope = dcm.RescaleSlope if hasattr(dcm, "RescaleSlope") else 1
    hu = img * slope + intercept
    return hu

def window_normalize(arr: np.ndarray, wl_low: int = -1000, wl_high: int = 400):
    a = np.clip(arr, wl_low, wl_high)
    a = (a - wl_low) / (wl_high - wl_low)
    a = (a * 255.0).astype(np.uint8)
    return a
