import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import time

def test_minimal():
    print("Minimal SAM test starting...", flush=True)
    sam_checkpoint = r"G:\project\yolo11-chest_editted\sam_b.pt"
    model_type = "vit_b"
    device = "xpu"
    
    print(f"Loading model on {device}...", flush=True)
    model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    model.to(device=device)
    predictor = SamPredictor(model)
    print("Model loaded.", flush=True)
    
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    print("Calling set_image...", flush=True)
    start = time.time()
    predictor.set_image(img)
    print(f"set_image finished in {time.time() - start:.2f}s", flush=True)
    
    box = np.array([100, 100, 200, 200])
    print("Calling predict...", flush=True)
    start = time.time()
    masks, scores, logits = predictor.predict(box=box, multimask_output=False)
    print(f"predict finished in {time.time() - start:.2f}s", flush=True)
    print("Success!", flush=True)

if __name__ == "__main__":
    test_minimal()
