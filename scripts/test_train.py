"""
Quick test script for 2-class training system
Tests with minimal resources: b0, 2 epochs, small batch, low resolution
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from train_improved import train_model

if __name__ == "__main__":
    data_dir = "../data"
    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "val.csv")

    # Check for healthy feet image lists
    train_image_list = os.path.join(data_dir, "train_images.csv")
    val_image_list = os.path.join(data_dir, "val_images.csv")

    # Use healthy feet if available
    train_images = train_image_list if os.path.exists(train_image_list) else None
    val_images = val_image_list if os.path.exists(val_image_list) else None

    data_root = "/mnt/c/Users/90rez/OneDrive - University of Toronto/PhDUofT/SideProjects/DFU_Detection_Asem"
    image_folder = os.path.join(data_root, "DFUC2022_train_images")
    healthy_folder = os.path.join(data_root, "HealthyFeet")

    if not all(os.path.exists(f) for f in [train_csv, val_csv]):
        print("Error: CSV files not found. Please run data_preprocessing.py first.")
        exit(1)

    print("\n" + "="*60)
    print("QUICK TEST - 2-CLASS TRAINING SYSTEM")
    print("="*60)
    print("TEST PARAMETERS:")
    print("  - Model: efficientnet_b0 (smallest)")
    print("  - Epochs: 2 (quick validation)")
    print("  - Batch size: 4")
    print("  - Image size: 64x64 (fast)")
    print("  - Data: Full dataset")
    print("="*60 + "\n")

    model, history = train_model(
        train_csv=train_csv,
        val_csv=val_csv,
        image_folder=image_folder,
        num_epochs=2,  # Quick test
        batch_size=4,  # Small batch
        learning_rate=0.0001,
        img_size=64,  # Small images for speed
        backbone="efficientnet_b0",  # Smallest model
        device="cuda",
        checkpoint_dir="../checkpoints_test",
        early_stopping_patience=10,
        use_amp=True,
        train_image_list=train_images,
        val_image_list=val_images,
        healthy_folder=healthy_folder if train_images else None
    )

    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print("If no errors occurred, the 2-class system is working correctly.")
    print("You can now run full training with larger model and more epochs.")
    print("="*60)
