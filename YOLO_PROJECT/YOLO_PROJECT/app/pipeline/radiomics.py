from typing import Dict, Optional
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from skimage.filters import threshold_otsu


def solid_ratio_otsu(image: np.ndarray, mask: np.ndarray) -> float:
    roi = image.copy()
    roi[mask == 0] = 0
    vals = roi[mask > 0]
    if vals.size == 0:
        return 0.0
    thr = threshold_otsu(vals.astype(np.float32))
    solid = (vals >= thr).sum()
    total = vals.size
    return float(solid) / float(total)


def extract_radiomics(image: np.ndarray, mask: np.ndarray) -> Optional[Dict[str, float]]:
    img = sitk.GetImageFromArray(image.astype(np.int16))
    m = sitk.GetImageFromArray(mask.astype(np.uint8))
    extractor = featureextractor.RadiomicsFeatureExtractor()
    try:
        features = extractor.execute(img, m)
    except Exception:
        return None
    out = {}
    for k, v in features.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out
