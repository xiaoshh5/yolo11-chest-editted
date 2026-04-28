from ultralytics import YOLO
import torch

model_path = r"G:\project\yolo11-chest_editted\YOLO_PROJECT\YOLO_PROJECT\runs\lung_nodule_lung1_v6_cpu\weights\last.pt"
model = YOLO(model_path)

has_nan = False
for name, param in model.model.named_parameters():
    if torch.isnan(param).any():
        print(f"Parameter {name} contains NaN")
        has_nan = True

if not has_nan:
    print("No NaNs found in model parameters.")
else:
    print("NaNs found in model parameters!")
