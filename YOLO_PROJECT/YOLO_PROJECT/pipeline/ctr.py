import numpy as np
import cv2

def max_diameter(mask: np.ndarray, spacing=(1.0, 1.0)):
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    w, h = rect[1]
    px = max(w, h)
    sx, sy = spacing[0], spacing[1]
    d = px * max(sx, sy)
    return float(d)

def ctr_ratio(hu_slice: np.ndarray, mask: np.ndarray, solid_threshold: float = -300.0, spacing=(1.0, 1.0)):
    full_d = max_diameter(mask, spacing)
    solid = ((hu_slice > solid_threshold) & (mask > 0)).astype(np.uint8)
    solid_d = max_diameter(solid, spacing)
    if full_d <= 0:
        return 0.0
    return float(solid_d / full_d)
