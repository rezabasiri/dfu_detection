"""
Add healthy feet images as negative samples (no DFU annotations)
This improves model performance by reducing false positives
"""

import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

def add_healthy_feet_to_dataset(
    healthy_feet_folder: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = 42
):
    """
    Add healthy feet images (negative samples) to existing dataset

    Args:
        healthy_feet_folder: Path to HealthyFeet folder
        train_csv: Path to existing train.csv
        val_csv: Path to existing val.csv
        test_csv: Path to existing test.csv
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        random_seed: Random seed for reproducibility
    """
    print("="*60)
    print("Adding Healthy Feet (Negative Samples)")
    print("="*60)

    # Load existing datasets
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    print(f"\nExisting dataset:")
    print(f"  Train: {train_df['name'].nunique()} DFU images")
    print(f"  Val:   {val_df['name'].nunique()} DFU images")
    print(f"  Test:  {test_df['name'].nunique()} DFU images")

    # Get list of healthy feet images
    healthy_feet_path = Path(healthy_feet_folder)
    if not healthy_feet_path.exists():
        print(f"\nError: {healthy_feet_folder} does not exist!")
        return

    healthy_images = list(healthy_feet_path.glob('*.jpg')) + \
                     list(healthy_feet_path.glob('*.jpeg')) + \
                     list(healthy_feet_path.glob('*.png'))

    print(f"\nFound {len(healthy_images)} healthy feet images")

    if len(healthy_images) == 0:
        print("No images found in HealthyFeet folder!")
        return

    # Split healthy feet images
    healthy_names = [img.name for img in healthy_images]

    train_healthy, temp_healthy = train_test_split(
        healthy_names,
        test_size=(1 - train_ratio),
        random_state=random_seed
    )

    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_healthy, test_healthy = train_test_split(
        temp_healthy,
        test_size=(1 - val_ratio_adjusted),
        random_state=random_seed
    )

    print(f"\nHealthy feet split:")
    print(f"  Train: {len(train_healthy)} images")
    print(f"  Val:   {len(val_healthy)} images")
    print(f"  Test:  {len(test_healthy)} images")

    # Create empty DataFrames for healthy images (no bboxes)
    # Note: We DON'T add rows to the CSV for negative samples
    # The dataset class will handle images without annotations

    # Instead, create a separate CSV listing all images (positive + negative)
    all_train_images = list(train_df['name'].unique()) + train_healthy
    all_val_images = list(val_df['name'].unique()) + val_healthy
    all_test_images = list(test_df['name'].unique()) + test_healthy

    # Save image lists
    output_dir = os.path.dirname(train_csv)

    pd.DataFrame({'name': all_train_images}).to_csv(
        os.path.join(output_dir, 'train_images.csv'), index=False
    )
    pd.DataFrame({'name': all_val_images}).to_csv(
        os.path.join(output_dir, 'val_images.csv'), index=False
    )
    pd.DataFrame({'name': all_test_images}).to_csv(
        os.path.join(output_dir, 'test_images.csv'), index=False
    )

    print(f"\nUpdated dataset (with healthy feet):")
    print(f"  Train: {len(all_train_images)} total images ({train_df['name'].nunique()} DFU + {len(train_healthy)} healthy)")
    print(f"  Val:   {len(all_val_images)} total images ({val_df['name'].nunique()} DFU + {len(val_healthy)} healthy)")
    print(f"  Test:  {len(all_test_images)} total images ({test_df['name'].nunique()} DFU + {len(test_healthy)} healthy)")

    print(f"\nSaved image lists to:")
    print(f"  {output_dir}/train_images.csv")
    print(f"  {output_dir}/val_images.csv")
    print(f"  {output_dir}/test_images.csv")

    print(f"\nOriginal annotation CSVs (train.csv, val.csv, test.csv) unchanged")
    print(f"Dataset class will use image lists + annotation CSVs together")

    return all_train_images, all_val_images, all_test_images

if __name__ == "__main__":
    # Paths
    data_root = "/mnt/c/Users/90rez/OneDrive - University of Toronto/PhDUofT/SideProjects/DFU_Detection_Asem"
    healthy_feet_folder = os.path.join(data_root, "HealthyFeet")

    data_dir = "../data"
    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "val.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    # Check if annotation CSVs exist
    if not all(os.path.exists(f) for f in [train_csv, val_csv, test_csv]):
        print("Error: Run data_preprocessing.py first to create train/val/test CSVs")
        exit(1)

    # Add healthy feet
    add_healthy_feet_to_dataset(
        healthy_feet_folder=healthy_feet_folder,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv
    )

    print("\n" + "="*60)
    print("Complete!")
    print("="*60)
    print("\nNext: Update dataset.py to use both image lists and annotation CSVs")