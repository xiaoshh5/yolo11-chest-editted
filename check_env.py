import sys
import os
print("Python version:", sys.version)
print("CWD:", os.getcwd())
try:
    import torch
    print("Torch version:", torch.__version__)
    print("XPU available:", hasattr(torch, "xpu") and torch.xpu.is_available())
except ImportError:
    print("Torch not found")
