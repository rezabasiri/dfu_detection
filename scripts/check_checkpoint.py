#!/usr/bin/env python3
"""
Check if checkpoint has NaN/Inf values in model weights
Usage: python check_checkpoint.py <checkpoint_path>
"""

import torch
import sys
import os

def check_checkpoint(checkpoint_path):
    """Check checkpoint for NaN/Inf values"""
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return False

    print(f"Checking checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Check model state_dict
    model_state = checkpoint.get('model_state_dict', {})

    nan_params = []
    inf_params = []
    total_params = 0

    for name, param in model_state.items():
        total_params += 1
        if torch.isnan(param).any():
            nan_params.append(name)
        if torch.isinf(param).any():
            inf_params.append(name)

    print(f"\nTotal parameters: {total_params}")
    print(f"Parameters with NaN: {len(nan_params)}")
    print(f"Parameters with Inf: {len(inf_params)}")

    if nan_params:
        print("\nParameters with NaN values:")
        for name in nan_params[:10]:  # Show first 10
            print(f"  - {name}")
        if len(nan_params) > 10:
            print(f"  ... and {len(nan_params) - 10} more")

    if inf_params:
        print("\nParameters with Inf values:")
        for name in inf_params[:10]:
            print(f"  - {name}")
        if len(inf_params) > 10:
            print(f"  ... and {len(inf_params) - 10} more")

    # Check optimizer state
    if 'optimizer_state_dict' in checkpoint:
        print("\nChecking optimizer state...")
        opt_state = checkpoint['optimizer_state_dict']
        if 'state' in opt_state:
            nan_opt_states = 0
            for param_id, state in opt_state['state'].items():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        if torch.isnan(value).any() or torch.isinf(value).any():
                            nan_opt_states += 1
                            break
            print(f"Optimizer states with NaN/Inf: {nan_opt_states}")

    # Print checkpoint metadata
    print("\nCheckpoint metadata:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best composite score: {checkpoint.get('best_composite_score', 'N/A')}")
    print(f"  Learning rate: {checkpoint.get('learning_rate', 'N/A')}")

    # Verdict
    print("\n" + "=" * 60)
    if nan_params or inf_params:
        print("❌ CHECKPOINT IS CORRUPTED (contains NaN/Inf values)")
        print("   DO NOT use this checkpoint for training!")
        print("\n   Solutions:")
        print("   1. Use an earlier checkpoint (before epoch 26)")
        print("   2. Start training from scratch with fixed augmentations")
        return False
    else:
        print("✓ Checkpoint looks healthy (no NaN/Inf values)")
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_checkpoint.py <checkpoint_path>")
        print("\nExamples:")
        print("  python check_checkpoint.py ../checkpoints_b5/best_model.pth")
        print("  python check_checkpoint.py ../checkpoints_b5/resume_training.pth")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    is_healthy = check_checkpoint(checkpoint_path)
    sys.exit(0 if is_healthy else 1)
