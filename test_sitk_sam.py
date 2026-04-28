import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import SimpleITK as sitk
import time

def test_sitk_sam():
    print("SimpleITK + SAM test starting...", flush=True)
    
    # 1. SimpleITK
    print("Running SimpleITK...", flush=True)
    dummy_arr = np.zeros((10, 512, 512), dtype=np.float32)
    sitk_img = sitk.GetImageFromArray(dummy_arr)
    _ = sitk.GetArrayFromImage(sitk_img)
    print("SimpleITK finished.", flush=True)
    
    # 2. SAM
    sam_checkpoint = r"G:\project\yolo11-chest_editted\sam_b.pt"
    device = "xpu"
    print(f"Loading SAM on {device}...", flush=True)
    sam_model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam_model.to(device=device)
    predictor = SamPredictor(sam_model)
    print("SAM loaded.", flush=True)
    
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    print("Calling SAM set_image...", flush=True)
    start = time.time()
    predictor.set_image(img)
    print(f"SAM set_image finished in {time.time() - start:.2f}s", flush=True)
    
    print("Success!", flush=True)

if __name__ == "__main__":
    test_sitk_sam()
