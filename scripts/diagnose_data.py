"""
Diagnostic script to identify data quality issues causing NaN losses
"""

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from dataset import DFUDatasetLMDB, get_train_transforms, get_val_transforms, collate_fn

def diagnose_dataset(lmdb_path, mode="train", img_size=640, num_samples=100):
    """
    Diagnose dataset for issues that could cause NaN losses
    """
    print(f"\n{'='*60}")
    print(f"Diagnosing {mode} dataset")
    print(f"{'='*60}")

    transforms = get_train_transforms(img_size) if mode == "train" else get_val_transforms(img_size)

    dataset = DFUDatasetLMDB(
        lmdb_path=lmdb_path,
        transforms=transforms,
        mode=mode
    )

    print(f"\nTotal samples: {len(dataset)}")
    print(f"Checking first {min(num_samples, len(dataset))} samples...\n")

    # Statistics
    stats = {
        "empty_boxes": 0,
        "invalid_boxes": 0,
        "nan_images": 0,
        "inf_images": 0,
        "zero_area_boxes": 0,
        "out_of_bounds_boxes": 0,
        "inverted_boxes": 0,
        "total_boxes": 0,
        "box_areas": []
    }

    problematic_indices = []

    for idx in tqdm(range(min(num_samples, len(dataset)))):
        try:
            image, target = dataset[idx]

            # Check image
            if torch.isnan(image).any():
                stats["nan_images"] += 1
                problematic_indices.append((idx, "NaN in image"))

            if torch.isinf(image).any():
                stats["inf_images"] += 1
                problematic_indices.append((idx, "Inf in image"))

            # Check boxes
            boxes = target['boxes']

            if len(boxes) == 0:
                stats["empty_boxes"] += 1
                problematic_indices.append((idx, "No bounding boxes"))
                continue

            stats["total_boxes"] += len(boxes)

            # Check each box
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box

                # Check for NaN/Inf
                if torch.isnan(box).any() or torch.isinf(box).any():
                    stats["invalid_boxes"] += 1
                    problematic_indices.append((idx, f"Invalid box {i}: {box.tolist()}"))
                    continue

                # Check if inverted
                if x2 <= x1 or y2 <= y1:
                    stats["inverted_boxes"] += 1
                    problematic_indices.append((idx, f"Inverted box {i}: {box.tolist()}"))
                    continue

                # Check area
                area = (x2 - x1) * (y2 - y1)
                if area <= 0:
                    stats["zero_area_boxes"] += 1
                    problematic_indices.append((idx, f"Zero area box {i}: {box.tolist()}"))
                else:
                    stats["box_areas"].append(area.item())

                # Check if out of bounds (assuming normalized or pixel coords)
                # For pixel coords in 640x640 image
                if x1 < 0 or y1 < 0 or x2 > img_size or y2 > img_size:
                    stats["out_of_bounds_boxes"] += 1
                    problematic_indices.append((idx, f"Out of bounds box {i}: {box.tolist()}"))

        except Exception as e:
            problematic_indices.append((idx, f"Exception: {str(e)}"))

    # Print results
    print(f"\n{'='*60}")
    print(f"Diagnosis Results for {mode} set")
    print(f"{'='*60}")
    print(f"\nImage Issues:")
    print(f"  Images with NaN values: {stats['nan_images']}")
    print(f"  Images with Inf values: {stats['inf_images']}")

    print(f"\nBounding Box Issues:")
    print(f"  Samples with no boxes: {stats['empty_boxes']} ({100*stats['empty_boxes']/min(num_samples, len(dataset)):.1f}%)")
    print(f"  Invalid boxes (NaN/Inf): {stats['invalid_boxes']}")
    print(f"  Inverted boxes (x2<=x1 or y2<=y1): {stats['inverted_boxes']}")
    print(f"  Zero area boxes: {stats['zero_area_boxes']}")
    print(f"  Out of bounds boxes: {stats['out_of_bounds_boxes']}")
    print(f"  Total boxes checked: {stats['total_boxes']}")

    if stats['box_areas']:
        print(f"\nBox Area Statistics:")
        print(f"  Mean area: {np.mean(stats['box_areas']):.2f}")
        print(f"  Median area: {np.median(stats['box_areas']):.2f}")
        print(f"  Min area: {np.min(stats['box_areas']):.2f}")
        print(f"  Max area: {np.max(stats['box_areas']):.2f}")

    if problematic_indices:
        print(f"\n{'='*60}")
        print(f"Problematic Samples (first 20):")
        print(f"{'='*60}")
        for idx, issue in problematic_indices[:20]:
            print(f"  Index {idx}: {issue}")

    # Test a batch through DataLoader
    print(f"\n{'='*60}")
    print(f"Testing DataLoader (batch_size=4)")
    print(f"{'='*60}")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Use 0 for debugging
        collate_fn=collate_fn
    )

    batch_stats = {
        "batches_with_empty_samples": 0,
        "total_batches_checked": 0
    }

    for batch_idx, (images, targets) in enumerate(loader):
        if batch_idx >= 10:  # Check first 10 batches
            break

        batch_stats["total_batches_checked"] += 1

        # Check each sample in batch
        for i, target in enumerate(targets):
            if len(target['boxes']) == 0:
                batch_stats["batches_with_empty_samples"] += 1
                print(f"  Batch {batch_idx}, Sample {i}: Empty boxes!")
                break

    print(f"\nBatch Statistics:")
    print(f"  Total batches checked: {batch_stats['total_batches_checked']}")
    print(f"  Batches with empty samples: {batch_stats['batches_with_empty_samples']}")

    return stats, problematic_indices


if __name__ == "__main__":
    data_dir = "../data"
    train_lmdb = os.path.join(data_dir, "train.lmdb")
    val_lmdb = os.path.join(data_dir, "val.lmdb")

    # Check if LMDB exists
    if not os.path.exists(train_lmdb):
        print(f"Error: {train_lmdb} not found!")
        exit(1)

    # Diagnose training set
    train_stats, train_issues = diagnose_dataset(train_lmdb, mode="train", img_size=640, num_samples=500)

    # Diagnose validation set
    val_stats, val_issues = diagnose_dataset(val_lmdb, mode="val", img_size=640, num_samples=100)

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")

    critical_issues = []

    if train_stats["empty_boxes"] > len(train_issues) * 0.1:
        critical_issues.append(f"CRITICAL: {train_stats['empty_boxes']} training samples have no boxes after augmentation!")

    if val_stats["empty_boxes"] > 0:
        critical_issues.append(f"CRITICAL: {val_stats['empty_boxes']} validation samples have no boxes!")

    if train_stats["invalid_boxes"] > 0 or val_stats["invalid_boxes"] > 0:
        critical_issues.append(f"CRITICAL: Invalid boxes detected (NaN/Inf)")

    if critical_issues:
        print("\nCRITICAL ISSUES FOUND:")
        for issue in critical_issues:
            print(f"  - {issue}")
        print("\nThese issues are likely causing NaN losses during training!")
    else:
        print("\nNo critical issues found. The problem may be elsewhere.")
