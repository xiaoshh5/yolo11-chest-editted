from pathlib import Path
import numpy as np
import SimpleITK as sitk


def load_series(path: str):
    """Load a DICOM series from a directory. Returns (array, spacing, meta) or (None, None, None)."""
    p = Path(path)
    if not p.exists():
        return None, None, None

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


def window_normalize(arr: np.ndarray, wl_low: int = -1000, wl_high: int = 400):
    """Window CT HU values to [0, 255] uint8 for display.

    Raises:
        ValueError: if wl_low >= wl_high.
    """
    if wl_low >= wl_high:
        raise ValueError(f"wl_low ({wl_low}) must be less than wl_high ({wl_high})")
    arr = np.asarray(arr, dtype=np.float32)
    a = np.clip(arr, wl_low, wl_high)
    a = (a - wl_low) / (wl_high - wl_low)
    a = (a * 255.0).astype(np.uint8)
    return a
