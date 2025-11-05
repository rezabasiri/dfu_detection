"""
Data preprocessing script for DFU detection
Handles CSV processing and image validation
"""

import pandas as pd
import os
from pathlib import Path
from PIL import Image
import shutil
from typing import Tuple, List

def validate_and_fix_csv(csv_path: str, image_folder: str) -> pd.DataFrame:
    """
    Load CSV and fix file extensions if needed
    
    Args:
        csv_path: Path to groundtruth.csv
        image_folder: Path to folder containing JPG images
        
    Returns:
        DataFrame with corrected image names
    """
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} annotations from CSV")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")

    # Match filenames regardless of extension and convert all to PNG
    print("\nMatching filenames and standardizing to PNG format...")

    # Build a dictionary of actual files in the folder (basename -> full filename)
    actual_files = {}
    for file in os.listdir(image_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            basename = os.path.splitext(file)[0]
            actual_files[basename] = file

    print(f"Found {len(actual_files)} image files in folder")

    # Match CSV entries to actual files and convert to PNG
    extension_fixes = 0
    conversions = 0
    missing_images = []

    for i, row in df.iterrows():
        csv_name = row['name']
        csv_basename = os.path.splitext(csv_name)[0]

        # Find matching file regardless of extension
        if csv_basename in actual_files:
            actual_filename = actual_files[csv_basename]
            actual_path = os.path.join(image_folder, actual_filename)
            target_name = csv_basename + '.png'
            target_path = os.path.join(image_folder, target_name)

            # If actual file is not PNG, convert it
            if not actual_filename.endswith('.png'):
                try:
                    img = Image.open(actual_path)
                    img = img.convert('RGB')
                    img.save(target_path, 'PNG')
                    conversions += 1
                    # Optionally remove the old file (commented out for safety)
                    # os.remove(actual_path)
                except Exception as e:
                    print(f"Warning: Could not convert {actual_filename} to PNG: {e}")
                    missing_images.append(csv_name)
                    continue

            # Update CSV to use .png extension
            if csv_name != target_name:
                df.at[i, 'name'] = target_name
                extension_fixes += 1
        else:
            # File not found with any extension
            missing_images.append(csv_name)

    if extension_fixes > 0:
        print(f"Updated {extension_fixes} filenames to .png extension")
    if conversions > 0:
        print(f"Converted {conversions} images to PNG format")

    # Validate that images exist (after conversion)
    missing_after_check = []
    for img_name in df['name'].unique():
        img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(img_path):
            missing_after_check.append(img_name)

    missing_images = list(set(missing_images + missing_after_check))
    
    if missing_images:
        print(f"\nWarning: {len(missing_images)} images not found:")
        for img in missing_images[:5]:
            print(f"  - {img}")
        if len(missing_images) > 5:
            print(f"  ... and {len(missing_images) - 5} more")
    else:
        print("\nAll images found!")
    
    # Remove rows with missing images
    df = df[~df['name'].isin(missing_images)]
    print(f"\nFinal dataset: {len(df)} annotations for {df['name'].nunique()} images")
    
    return df

def validate_bboxes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate bounding boxes and remove invalid ones
    
    Args:
        df: DataFrame with bbox coordinates
        
    Returns:
        DataFrame with only valid bboxes
    """
    initial_count = len(df)
    
    # Remove invalid bboxes (where xmin >= xmax or ymin >= ymax)
    df = df[(df['xmax'] > df['xmin']) & (df['ymax'] > df['ymin'])]
    
    # Remove negative coordinates
    df = df[(df['xmin'] >= 0) & (df['ymin'] >= 0)]
    
    removed = initial_count - len(df)
    if removed > 0:
        print(f"\nRemoved {removed} invalid bounding boxes")
    
    return df

def get_image_stats(image_folder: str, df: pd.DataFrame) -> dict:
    """
    Get statistics about the dataset images including all unique dimensions

    Args:
        image_folder: Path to images
        df: DataFrame with annotations

    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_images': df['name'].nunique(),
        'total_annotations': len(df),
        'avg_annotations_per_image': len(df) / df['name'].nunique(),
        'image_sizes': []
    }

    # Scan ALL images to get unique dimensions
    print("Scanning all images for dimensions...")
    all_images = df['name'].unique()
    dimension_counts = {}

    for img_name in all_images:
        img_path = os.path.join(image_folder, img_name)
        try:
            with Image.open(img_path) as img:
                size = img.size  # (width, height)
                stats['image_sizes'].append(size)

                # Count occurrences of each dimension
                dimension_key = f"{size[0]}x{size[1]}"
                dimension_counts[dimension_key] = dimension_counts.get(dimension_key, 0) + 1
        except Exception as e:
            print(f"Error reading {img_name}: {e}")

    if stats['image_sizes']:
        widths = [s[0] for s in stats['image_sizes']]
        heights = [s[1] for s in stats['image_sizes']]
        stats['avg_width'] = sum(widths) / len(widths)
        stats['avg_height'] = sum(heights) / len(heights)
        stats['unique_dimensions'] = dimension_counts

    return stats

def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.18, 
                 random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/val/test sets
    
    Args:
        df: DataFrame with annotations
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set (test = 1 - train - val)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    # Get unique images
    unique_images = df['name'].unique()
    
    # First split: train and temp (val+test)
    train_images, temp_images = train_test_split(
        unique_images, 
        test_size=(1 - train_ratio),
        random_state=random_seed
    )
    
    # Second split: val and test
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_images, test_images = train_test_split(
        temp_images,
        test_size=(1 - val_ratio_adjusted),
        random_state=random_seed
    )
    
    train_df = df[df['name'].isin(train_images)].reset_index(drop=True)
    val_df = df[df['name'].isin(val_images)].reset_index(drop=True)
    test_df = df[df['name'].isin(test_images)].reset_index(drop=True)
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_images)} images, {len(train_df)} annotations")
    print(f"  Val:   {len(val_images)} images, {len(val_df)} annotations")
    print(f"  Test:  {len(test_images)} images, {len(test_df)} annotations")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # Paths
    data_root = "/mnt/c/Users/90rez/OneDrive - University of Toronto/PhDUofT/SideProjects/DFU_Detection_Asem"
    csv_path = os.path.join(data_root, "groundtruth.csv")
    image_folder = os.path.join(data_root, "DFUC2022_train_images")
    
    print("="*60)
    print("DFU Detection - Data Preprocessing")
    print("="*60)
    
    # Load and validate CSV
    df = validate_and_fix_csv(csv_path, image_folder)
    
    # Validate bounding boxes
    df = validate_bboxes(df)
    
    # Get statistics
    print("\nGathering dataset statistics...")
    stats = get_image_stats(image_folder, df)
    print(f"\nDataset Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Total annotations: {stats['total_annotations']}")
    print(f"  Avg annotations per image: {stats['avg_annotations_per_image']:.2f}")
    if 'avg_width' in stats:
        print(f"  Avg image size: {stats['avg_width']:.0f} x {stats['avg_height']:.0f}")

    # Print unique dimensions
    if 'unique_dimensions' in stats:
        print(f"\nUnique Image Dimensions:")
        # Sort by count (descending)
        sorted_dims = sorted(stats['unique_dimensions'].items(), key=lambda x: x[1], reverse=True)
        for dim, count in sorted_dims:
            print(f"  {dim}: {count} images ({count/stats['total_images']*100:.1f}%)")
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(df)
    
    # Save split datasets
    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"\nSaved split datasets to {output_dir}/")
    print("\nPreprocessing complete!")
