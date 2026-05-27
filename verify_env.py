import sys
print(f"Python version: {sys.version}")

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU memory: {mem_gb:.1f} GB")

    x = torch.randn(100, 100, device="cuda")
    y = x @ x.T
    print(f"GPU matmul test passed: shape={y.shape}")

a = torch.randn(500, 500)
b = torch.randn(500, 500)
c = a @ b
print(f"CPU matmul test passed: shape={c.shape}")

import numpy as np;           print(f"numpy:       {np.__version__}")
import pandas as pd;          print(f"pandas:      {pd.__version__}")
import matplotlib;            print(f"matplotlib:  {matplotlib.__version__}")
import seaborn;               print(f"seaborn:     {seaborn.__version__}")
import sklearn;               print(f"scikit-learn:{sklearn.__version__}")
import cv2;                   print(f"opencv:      {cv2.__version__}")
import scipy;                 print(f"scipy:       {scipy.__version__}")
import jupyterlab;            print(f"jupyterlab:  {jupyterlab.__version__}")

print("\nAll checks passed. Environment is ready.")
