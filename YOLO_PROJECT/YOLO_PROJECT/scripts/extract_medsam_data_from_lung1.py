
import os
import cv2
import numpy as np
import nibabel as nib
from pathlib import Path
from YOLO_PROJECT.pipeline.dicom import load_series, window_normalize

def extract_data():
    lung1_root = Path(r"G:\project\yolo11-chest_editted\lung_1")
    label_dir = lung1_root / "label"
    dicom_base = lung1_root / "sysucc lung cancer more than 4"
    
    # Target directories
    target_root = Path(r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\datasets\lung_1_YOLO_3\train")
    target_img_dir = target_root / "images"
    target_mask_dir = target_root / "masks"
    
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_mask_dir.mkdir(parents=True, exist_ok=True)
    
    nii_files = list(label_dir.glob("*.nii"))
    print(f"Found {len(nii_files)} NIfTI label files.")
    
    total_extracted = 0
    
    for nii_path in nii_files:
        case_id = nii_path.stem
        # Find corresponding DICOM dir
        case_dicom_dir = dicom_base / case_id
        if not case_dicom_dir.exists():
            print(f"DICOM dir not found for {case_id}, skipping.")
            continue
            
        # Search for the subdirectory with DCM files
        dcm_dir = None
        for sub in case_dicom_dir.iterdir():
            if sub.is_dir() and len(list(sub.glob("*.dcm"))) > 10:
                dcm_dir = sub
                break
        
        if dcm_dir is None:
            print(f"No DCM subdirectory found for {case_id}, skipping.")
            continue
            
        print(f"Processing {case_id}...")
        
        try:
            # Load DICOM series
            series, spacing, _ = load_series(str(dcm_dir))
            if series is None:
                continue
            
            # Load NIfTI mask
            nii_img = nib.load(str(nii_path))
            mask_data = nii_img.get_fdata()
            
            # NIfTI data often needs orientation adjustment to match DICOM loading
            # Our load_series sorts by ImagePositionPatient, resulting in (Z, H, W)
            # NIfTI is usually (H, W, Z) or (W, H, Z). We need to match them.
            
            # Basic sanity check on dimensions
            if series.shape[0] != mask_data.shape[2]:
                # Try to see if flipping or rotating helps, but usually medical data needs careful alignment.
                # For this task, we assume the Z-axis matches after proper loading.
                print(f"Dimension mismatch for {case_id}: DICOM={series.shape}, NIfTI={mask_data.shape}. Attempting to proceed...")
            
            num_slices = min(series.shape[0], mask_data.shape[2])
            
            for z in range(num_slices):
                hu_slice = series[z]
                mask_slice = mask_data[:, :, z].T  # Transpose if needed, NIfTI is often (W, H)
                
                # Check if slice has a nodule
                if np.any(mask_slice > 0):
                    # Normalize and convert to BGR
                    img = window_normalize(hu_slice)
                    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    
                    # Ensure mask is uint8 (0 or 255)
                    mask_uint8 = (mask_slice > 0).astype(np.uint8) * 255
                    
                    # Resize mask to match image if they differ (should be 512x512)
                    if bgr.shape[:2] != mask_uint8.shape:
                        mask_uint8 = cv2.resize(mask_uint8, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    # Save
                    fname = f"lung1_{case_id}_slice_{z:03d}"
                    cv2.imwrite(str(target_img_dir / f"{fname}.jpg"), bgr)
                    cv2.imwrite(str(target_mask_dir / f"{fname}.png"), mask_uint8)
                    total_extracted += 1
                    
        except Exception as e:
            print(f"Error processing {case_id}: {e}")
            
    print(f"Finished. Extracted {total_extracted} new high-quality 2D samples.")

if __name__ == "__main__":
    extract_data()
