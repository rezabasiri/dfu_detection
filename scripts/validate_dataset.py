"""
Dataset validation script for DFU detection
Checks for:
- Corrupted or unreadable images
- Invalid bounding boxes (negative coords, zero area, out of bounds)
- Extreme pixel values or unusual distributions
- Duplicate images
"""

import pandas as pd
import os
import numpy as np
from PIL import Image
from pathlib import Path
import hashlib
from collections import defaultdict
from tqdm import tqdm

def check_image_validity(image_path):
    """
    Check if an image can be loaded and get its properties

    Returns:
        dict with keys: valid, error, size, min_val, max_val, mean_val
    """
    result = {
        'valid': False,
        'error': None,
        'size': None,
        'min_val': None,
        'max_val': None,
        'mean_val': None
    }

    try:
        # Try to open and load the image
        img = Image.open(image_path)
        img.load()  # Force loading to catch truncated images
        img = img.convert('RGB')

        # Get image properties
        img_array = np.array(img)
        result['valid'] = True
        result['size'] = img.size  # (width, height)
        result['min_val'] = float(img_array.min())
        result['max_val'] = float(img_array.max())
        result['mean_val'] = float(img_array.mean())

    except Exception as e:
        result['error'] = str(e)

    return result

def compute_image_hash(image_path):
    """Compute hash of image file for duplicate detection"""
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def validate_bbox(bbox, img_width, img_height):
    """
    Validate a bounding box

    Returns:
        dict with keys: valid, issues (list of issue descriptions)
    """
    xmin, ymin, xmax, ymax = bbox
    issues = []

    # Check for negative coordinates
    if xmin < 0 or ymin < 0:
        issues.append(f"Negative coordinates: ({xmin}, {ymin})")

    # Check for inverted coordinates
    if xmin >= xmax:
        issues.append(f"xmin >= xmax: {xmin} >= {xmax}")
    if ymin >= ymax:
        issues.append(f"ymin >= ymax: {ymin} >= {ymax}")

    # Check for out of bounds
    if xmax > img_width:
        issues.append(f"xmax ({xmax}) > image width ({img_width})")
    if ymax > img_height:
        issues.append(f"ymax ({ymax}) > image height ({img_height})")

    # Check for zero or very small area
    width = xmax - xmin
    height = ymax - ymin
    area = width * height

    if area <= 0:
        issues.append(f"Zero or negative area: {area}")
    elif area < 100:  # Less than 10x10 pixels
        issues.append(f"Very small bbox area: {area} pixels ({width}x{height})")

    # Check for extremely large bboxes (likely annotation errors)
    img_area = img_width * img_height
    if area > 0.9 * img_area:
        issues.append(f"Bbox covers >90% of image ({area / img_area * 100:.1f}%)")

    return {
        'valid': len(issues) == 0,
        'issues': issues
    }

def validate_dataset(csv_path, image_folder, output_report=None):
    """
    Main validation function

    Args:
        csv_path: Path to groundtruth CSV
        image_folder: Path to image folder
        output_report: Path to save detailed report (optional)
    """
    print("="*60)
    print("DFU Dataset Validation")
    print("="*60)

    # Load CSV
    print(f"\nLoading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Total annotations: {len(df)}")
    print(f"  Unique images: {df['name'].nunique()}")

    # Validation results
    corrupted_images = []
    invalid_bboxes = []
    extreme_values = []
    missing_images = []
    duplicate_hashes = defaultdict(list)

    # Process each unique image
    unique_images = df['name'].unique()
    print(f"\nValidating {len(unique_images)} images...")

    for img_name in tqdm(unique_images):
        img_path = os.path.join(image_folder, img_name)

        # Check if file exists
        if not os.path.exists(img_path):
            missing_images.append(img_name)
            continue

        # Check image validity
        img_result = check_image_validity(img_path)

        if not img_result['valid']:
            corrupted_images.append({
                'image': img_name,
                'error': img_result['error']
            })
            continue

        # Check for extreme pixel values (could indicate corruption)
        if img_result['min_val'] == img_result['max_val']:
            extreme_values.append({
                'image': img_name,
                'issue': f"Constant pixel value: {img_result['min_val']}"
            })

        # Compute hash for duplicate detection
        img_hash = compute_image_hash(img_path)
        if img_hash:
            duplicate_hashes[img_hash].append(img_name)

        # Validate bounding boxes for this image
        img_annotations = df[df['name'] == img_name]
        img_width, img_height = img_result['size']

        for idx, row in img_annotations.iterrows():
            bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            bbox_result = validate_bbox(bbox, img_width, img_height)

            if not bbox_result['valid']:
                invalid_bboxes.append({
                    'image': img_name,
                    'bbox': bbox,
                    'image_size': (img_width, img_height),
                    'issues': bbox_result['issues']
                })

    # Find actual duplicates (more than one image with same hash)
    duplicates = {hash_val: imgs for hash_val, imgs in duplicate_hashes.items() if len(imgs) > 1}

    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    total_issues = len(missing_images) + len(corrupted_images) + len(invalid_bboxes) + len(extreme_values) + len(duplicates)

    if total_issues == 0:
        print("\n✓ All validation checks passed!")
        print(f"  {len(unique_images)} images validated successfully")
    else:
        print(f"\n⚠ Found {total_issues} issue(s):\n")

        if missing_images:
            print(f"  ❌ Missing images: {len(missing_images)}")
            for img in missing_images[:5]:
                print(f"     - {img}")
            if len(missing_images) > 5:
                print(f"     ... and {len(missing_images) - 5} more")

        if corrupted_images:
            print(f"\n  ❌ Corrupted/unreadable images: {len(corrupted_images)}")
            for item in corrupted_images[:5]:
                print(f"     - {item['image']}: {item['error']}")
            if len(corrupted_images) > 5:
                print(f"     ... and {len(corrupted_images) - 5} more")

        if invalid_bboxes:
            print(f"\n  ❌ Invalid bounding boxes: {len(invalid_bboxes)}")
            for item in invalid_bboxes[:5]:
                print(f"     - {item['image']}:")
                print(f"       bbox: {item['bbox']}, image size: {item['image_size']}")
                for issue in item['issues']:
                    print(f"       • {issue}")
            if len(invalid_bboxes) > 5:
                print(f"     ... and {len(invalid_bboxes) - 5} more")

        if extreme_values:
            print(f"\n  ⚠ Images with extreme values: {len(extreme_values)}")
            for item in extreme_values[:5]:
                print(f"     - {item['image']}: {item['issue']}")
            if len(extreme_values) > 5:
                print(f"     ... and {len(extreme_values) - 5} more")

        if duplicates:
            print(f"\n  ⚠ Duplicate images found: {len(duplicates)} groups")
            for hash_val, imgs in list(duplicates.items())[:3]:
                print(f"     - Duplicates: {', '.join(imgs)}")
            if len(duplicates) > 3:
                print(f"     ... and {len(duplicates) - 3} more groups")

    # Save detailed report if requested
    if output_report:
        print(f"\nSaving detailed report to: {output_report}")
        with open(output_report, 'w') as f:
            f.write("DFU Dataset Validation Report\n")
            f.write("="*60 + "\n\n")

            f.write(f"Total images: {len(unique_images)}\n")
            f.write(f"Total annotations: {len(df)}\n")
            f.write(f"Total issues: {total_issues}\n\n")

            if missing_images:
                f.write(f"\nMissing Images ({len(missing_images)}):\n")
                for img in missing_images:
                    f.write(f"  - {img}\n")

            if corrupted_images:
                f.write(f"\nCorrupted Images ({len(corrupted_images)}):\n")
                for item in corrupted_images:
                    f.write(f"  - {item['image']}: {item['error']}\n")

            if invalid_bboxes:
                f.write(f"\nInvalid Bounding Boxes ({len(invalid_bboxes)}):\n")
                for item in invalid_bboxes:
                    f.write(f"  - {item['image']} {item['bbox']} (image size: {item['image_size']})\n")
                    for issue in item['issues']:
                        f.write(f"    • {issue}\n")

            if extreme_values:
                f.write(f"\nImages with Extreme Values ({len(extreme_values)}):\n")
                for item in extreme_values:
                    f.write(f"  - {item['image']}: {item['issue']}\n")

            if duplicates:
                f.write(f"\nDuplicate Images ({len(duplicates)} groups):\n")
                for hash_val, imgs in duplicates.items():
                    f.write(f"  - {', '.join(imgs)}\n")

    print(f"\n{'='*60}")
    print("Validation complete!")
    print(f"{'='*60}\n")

    return {
        'total_images': len(unique_images),
        'missing': missing_images,
        'corrupted': corrupted_images,
        'invalid_bboxes': invalid_bboxes,
        'extreme_values': extreme_values,
        'duplicates': duplicates
    }

if __name__ == "__main__":
    # Paths
    data_root = "/mnt/c/Users/90rez/OneDrive - University of Toronto/PhDUofT/SideProjects/DFU_Detection_Asem"
    csv_path = os.path.join(data_root, "groundtruth.csv")
    image_folder = os.path.join(data_root, "DFUC2022_train_images")
    output_report = "../results/dataset_validation_report.txt"

    # Create results directory
    os.makedirs("../results", exist_ok=True)

    # Run validation
    results = validate_dataset(csv_path, image_folder, output_report)