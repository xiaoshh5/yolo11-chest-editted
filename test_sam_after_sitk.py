import torch
import numpy as np
import SimpleITK as sitk
from segment_anything import sam_model_registry, SamPredictor

def test():
    print("Testing SAM after SimpleITK...", flush=True)
    
    # 1. Use SimpleITK
    print("Running SimpleITK dummy...", flush=True)
    img = sitk.GetImageFromArray(np.zeros((10, 10), dtype=np.float32))
    sitk.GetArrayFromImage(img)
    print("SimpleITK finished.", flush=True)
    
    # 2. Load SAM on XPU
    sam_checkpoint = r"G:\project\yolo11-chest_editted\sam_b.pt"
    device = "xpu"
    print(f"Loading SAM on {device}: {sam_checkpoint}", flush=True)
    model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    model.to(device=device)
    predictor = SamPredictor(model)
    print("SAM loaded.", flush=True)
    
    # 3. Predict
    img_data = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    print("Calling set_image...", flush=True)
    predictor.set_image(img_data)
    print("set_image finished.", flush=True)
    
    print("Calling predict...", flush=True)
    box = np.array([100, 100, 200, 200])
    predictor.predict(box=box, multimask_output=False)
    print("predict finished.", flush=True)

if __name__ == "__main__":
    test()
