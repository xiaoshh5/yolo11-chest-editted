import numpy as np

def ctr_ratio(hu_slice: np.ndarray, mask: np.ndarray, solid_threshold: float = -300.0):
    """Compute Consolidation-to-Tumor Ratio as pixel-count ratio.

    CTR = solid_area / total_area, where solid_area counts pixels
    within the mask with HU > solid_threshold.
    """
    if mask is None or np.sum(mask) == 0:
        return 0.0
    roi_hu = hu_slice[mask > 0]
    total = roi_hu.size
    solid = np.count_nonzero(roi_hu > solid_threshold)
    return float(solid / total) if total > 0 else 0.0
