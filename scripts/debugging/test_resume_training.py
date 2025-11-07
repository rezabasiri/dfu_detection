"""
Test script to verify resume training functionality
"""

import os
import torch

checkpoint_dir = "../checkpoints"

print("="*60)
print("Resume Training Test")
print("="*60)

# Check for checkpoints
resume_path = os.path.join(checkpoint_dir, "resume_training.pth")
best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

print("\nCheckpoint Status:")
print("-" * 60)

if os.path.exists(resume_path):
    print(f"✓ resume_training.pth found")
    checkpoint = torch.load(resume_path, map_location='cpu')
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Train Loss: {checkpoint.get('train_loss', 'unknown')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'unknown')}")
    print(f"  Next training will resume from epoch {checkpoint.get('epoch', 0) + 1}")
else:
    print(f"✗ resume_training.pth not found")

print()

if os.path.exists(best_model_path):
    print(f"✓ best_model.pth found")
    checkpoint = torch.load(best_model_path, map_location='cpu')
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Train Loss: {checkpoint.get('train_loss', 'unknown')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'unknown')}")
    if not os.path.exists(resume_path):
        print(f"  Next training will resume from epoch {checkpoint.get('epoch', 0) + 1}")
else:
    print(f"✗ best_model.pth not found")

print()

# Check for epoch checkpoints
epoch_checkpoints = []
if os.path.exists(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint_epoch_") and file.endswith(".pth"):
            epoch_num = file.replace("checkpoint_epoch_", "").replace(".pth", "")
            epoch_checkpoints.append(int(epoch_num))

if epoch_checkpoints:
    epoch_checkpoints.sort()
    print(f"✓ Found {len(epoch_checkpoints)} epoch checkpoints:")
    print(f"  Epochs: {epoch_checkpoints}")
    print(f"  Latest: checkpoint_epoch_{max(epoch_checkpoints)}.pth")
    print(f"\n  To resume from a specific epoch:")
    print(f"    cp ../checkpoints/checkpoint_epoch_N.pth ../checkpoints/resume_training.pth")
else:
    print(f"✗ No epoch checkpoints found")

print("\n" + "="*60)
print("What will happen when you run train_improved.py?")
print("="*60)

if os.path.exists(resume_path):
    print("\n✓ Training will RESUME from resume_training.pth")
elif os.path.exists(best_model_path):
    print("\n✓ Training will RESUME from best_model.pth")
else:
    print("\n✓ Training will START FRESH with ImageNet pretrained weights")

print("\n" + "="*60)
