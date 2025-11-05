#!/bin/bash
# Setup script for Vast.ai cluster
# Run this on the cluster to set up the training environment

echo "=========================================="
echo "DFU Detection - Vast.ai Setup Script"
echo "=========================================="

# Check GPU
echo -e "\n1. Checking GPU..."
nvidia-smi

# Check CUDA
echo -e "\n2. Checking CUDA..."
nvcc --version || echo "nvcc not in PATH (this is OK)"

# Check Python
echo -e "\n3. Checking Python..."
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Create project directory
echo -e "\n4. Creating project directory..."
mkdir -p /workspace/dfu_detection
cd /workspace/dfu_detection

# Install dependencies
echo -e "\n5. Installing dependencies..."
pip install --no-cache-dir \
    albumentations \
    opencv-python-headless \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm \
    lmdb \
    Pillow

# Verify installations
echo -e "\n6. Verifying installations..."
python -c "
import torch
import torchvision
import albumentations
import lmdb
import cv2
import pandas as pd
import numpy as np
print('✓ All packages imported successfully!')
print(f'  PyTorch: {torch.__version__}')
print(f'  Torchvision: {torchvision.__version__}')
print(f'  Albumentations: {albumentations.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo -e "\n=========================================="
echo "Setup complete!"
echo "=========================================="
echo "Next steps:"
echo "1. Transfer your code: scp -P 40535 -r scripts/ root@198.53.64.194:/workspace/dfu_detection/"
echo "2. Transfer LMDB data: scp -P 40535 -r data/*.lmdb root@198.53.64.194:/workspace/dfu_detection/data/"
echo "3. Start training: cd /workspace/dfu_detection/scripts && python train_improved.py"
