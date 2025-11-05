"""
Diagnostic script to understand what happened to the missing images
"""

import pandas as pd
import os

print("="*60)
print("Dataset Diagnostic Report")
print("="*60)

# Paths
data_root = "/mnt/c/Users/90rez/OneDrive - University of Toronto/PhDUofT/SideProjects/DFU_Detection_Asem"
csv_path = os.path.join(data_root, "groundtruth.csv")
image_folder = os.path.join(data_root, "DFUC2022_train_images")

# Check original CSV
print("\n1. ORIGINAL DATASET (groundtruth.csv)")
print("-" * 60)

if os.path.exists(csv_path):
    df_original = pd.read_csv(csv_path)
    print(f"Total annotations in groundtruth.csv: {len(df_original)}")
    print(f"Unique images in groundtruth.csv: {df_original['name'].nunique()}")

    # Check for duplicates
    print(f"\nAnnotations per image:")
    print(f"  Min: {df_original.groupby('name').size().min()}")
    print(f"  Max: {df_original.groupby('name').size().max()}")
    print(f"  Average: {df_original.groupby('name').size().mean():.2f}")
else:
    print(f"groundtruth.csv not found at {csv_path}")
    df_original = None

# Check processed CSVs
print("\n2. PROCESSED SPLITS")
print("-" * 60)

data_dir = "../data"
train_csv = os.path.join(data_dir, "train.csv")
val_csv = os.path.join(data_dir, "val.csv")
test_csv = os.path.join(data_dir, "test.csv")

if all(os.path.exists(f) for f in [train_csv, val_csv, test_csv]):
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    train_images = df_train['name'].nunique()
    val_images = df_val['name'].nunique()
    test_images = df_test['name'].nunique()
    total_processed = train_images + val_images + test_images

    print(f"Train: {train_images} images, {len(df_train)} annotations")
    print(f"Val:   {val_images} images, {len(df_val)} annotations")
    print(f"Test:  {test_images} images, {len(df_test)} annotations")
    print(f"Total: {total_processed} images, {len(df_train) + len(df_val) + len(df_test)} annotations")

    # Calculate split ratios
    if total_processed > 0:
        print(f"\nActual split ratios:")
        print(f"  Train: {train_images/total_processed*100:.1f}%")
        print(f"  Val:   {val_images/total_processed*100:.1f}%")
        print(f"  Test:  {test_images/total_processed*100:.1f}%")
else:
    print("Processed CSV files not found")
    df_train = df_val = df_test = None
    total_processed = 0

# Compare original vs processed
if df_original is not None and df_train is not None:
    print("\n3. MISSING IMAGES ANALYSIS")
    print("-" * 60)

    original_images = set(df_original['name'].unique())
    processed_images = set(df_train['name'].unique()) | set(df_val['name'].unique()) | set(df_test['name'].unique())

    missing_images = original_images - processed_images

    print(f"Original unique images: {len(original_images)}")
    print(f"Processed unique images: {len(processed_images)}")
    print(f"Missing images: {len(missing_images)}")

    if len(missing_images) > 0:
        print(f"\nReasons for missing images:")

        # Check if images exist in folder
        missing_files = []
        for img in list(missing_images)[:10]:  # Check first 10
            img_path = os.path.join(image_folder, img)
            if not os.path.exists(img_path):
                missing_files.append(img)

        if missing_files:
            print(f"  - Files not found on disk: {len(missing_files)} checked (sample)")
            print(f"    Examples: {missing_files[:3]}")

        # Check for invalid bboxes in original
        original_missing = df_original[df_original['name'].isin(missing_images)]
        invalid_bbox = original_missing[
            (original_missing['xmax'] <= original_missing['xmin']) |
            (original_missing['ymax'] <= original_missing['ymin'])
        ]

        if len(invalid_bbox) > 0:
            print(f"  - Invalid bounding boxes: {len(invalid_bbox)} annotations")
            print(f"    Affected images: {invalid_bbox['name'].nunique()}")

    # Check if we're looking at a subset
    print(f"\n4. POSSIBLE EXPLANATIONS")
    print("-" * 60)

    if len(original_images) > 6000:
        print(f"✓ Original dataset has {len(original_images)} images")
        print(f"✓ Processed dataset has {len(processed_images)} images")
        print(f"✗ Lost {len(missing_images)} images ({len(missing_images)/len(original_images)*100:.1f}%)")
        print(f"\nLikely causes:")
        print(f"  1. Images were filtered out during preprocessing")
        print(f"  2. File extension mismatches (.jpg vs .png)")
        print(f"  3. Missing or corrupted image files")
        print(f"  4. Invalid bounding boxes removed")
    else:
        print(f"⚠ You may be working with a subset of the data")

# Check healthy feet
print("\n5. HEALTHY FEET IMAGES")
print("-" * 60)

healthy_folder = os.path.join(data_root, "HealthyFeet")
if os.path.exists(healthy_folder):
    healthy_files = [f for f in os.listdir(healthy_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Healthy feet images in folder: {len(healthy_files)}")

    train_images_csv = os.path.join(data_dir, "train_images.csv")
    if os.path.exists(train_images_csv):
        df_train_images = pd.read_csv(train_images_csv)

        # Count DFU vs healthy
        dfu_in_list = set(df_train['name'].unique())
        total_in_list = set(df_train_images['name'].tolist())
        healthy_in_list = total_in_list - dfu_in_list

        print(f"\nIn train_images.csv:")
        print(f"  Total images: {len(df_train_images)}")
        print(f"  DFU images: {len(dfu_in_list)}")
        print(f"  Healthy images: {len(healthy_in_list)}")
else:
    print("HealthyFeet folder not found")

print("\n" + "="*60)
print("Diagnostic complete!")
print("="*60)
