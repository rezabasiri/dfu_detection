"""
Evaluate YOLO Model with Metrics Compatible with Faster R-CNN/RetinaNet

Computes:
- F1 Score
- Mean IoU
- Precision
- Recall
- Composite Score (0.40×F1 + 0.25×IoU + 0.20×Recall + 0.15×Precision)

This allows direct comparison with Faster R-CNN and RetinaNet results.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import lmdb
import pickle
import cv2
from tqdm import tqdm

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("ERROR: ultralytics package not found. Install with: pip install ultralytics")
    exit(1)


def box_iou(box1, box2):
    """
    Compute IoU between two boxes [xmin, ymin, xmax, ymax]
    """
    # Get intersection coordinates
    xmin_inter = max(box1[0], box2[0])
    ymin_inter = max(box1[1], box2[1])
    xmax_inter = min(box1[2], box2[2])
    ymax_inter = min(box1[3], box2[3])

    # Compute intersection area
    inter_width = max(0, xmax_inter - xmin_inter)
    inter_height = max(0, ymax_inter - ymin_inter)
    inter_area = inter_width * inter_height

    # Compute union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_detection_metrics(all_predictions, all_targets, iou_threshold=0.5):
    """
    Compute detection metrics: F1, IoU, Precision, Recall

    Args:
        all_predictions: List of predictions, each is dict with 'boxes', 'scores', 'labels'
        all_targets: List of ground truth, each is dict with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching predictions to ground truth

    Returns:
        Dictionary with metrics
    """
    total_tp = 0  # True positives
    total_fp = 0  # False positives
    total_fn = 0  # False negatives
    total_iou = 0.0  # Sum of IoUs for matched boxes
    num_matched = 0  # Number of matched boxes

    for pred, target in zip(all_predictions, all_targets):
        pred_boxes = pred['boxes']  # (N, 4)
        pred_scores = pred['scores']  # (N,)
        target_boxes = target['boxes']  # (M, 4)

        if len(pred_boxes) == 0 and len(target_boxes) == 0:
            # True negative (correctly detected no ulcers)
            continue

        if len(pred_boxes) == 0:
            # False negatives (missed all ground truth boxes)
            total_fn += len(target_boxes)
            continue

        if len(target_boxes) == 0:
            # False positives (predicted boxes but no ground truth)
            total_fp += len(pred_boxes)
            continue

        # Match predictions to ground truth using Hungarian algorithm (greedy for simplicity)
        matched_targets = set()

        for pred_box in pred_boxes:
            best_iou = 0.0
            best_target_idx = -1

            # Find best matching target
            for target_idx, target_box in enumerate(target_boxes):
                if target_idx in matched_targets:
                    continue

                iou = box_iou(pred_box, target_box)
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = target_idx

            # Check if match is good enough
            if best_iou >= iou_threshold:
                total_tp += 1
                total_iou += best_iou
                num_matched += 1
                matched_targets.add(best_target_idx)
            else:
                total_fp += 1

        # Remaining unmatched targets are false negatives
        total_fn += len(target_boxes) - len(matched_targets)

    # Compute metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = total_iou / num_matched if num_matched > 0 else 0.0

    # Composite score (same as Faster R-CNN/RetinaNet)
    composite_score = 0.40 * f1 + 0.25 * mean_iou + 0.20 * recall + 0.15 * precision

    return {
        'f1': f1,
        'mean_iou': mean_iou,
        'precision': precision,
        'recall': recall,
        'composite_score': composite_score,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'num_predictions': total_tp + total_fp,
        'num_targets': total_tp + total_fn
    }


def evaluate_yolo_on_lmdb(model_path, lmdb_path, confidence_threshold=0.05, iou_threshold=0.5, device='cuda'):
    """
    Evaluate YOLO model on LMDB validation set

    Args:
        model_path: Path to YOLO model weights (.pt file)
        lmdb_path: Path to validation LMDB database
        confidence_threshold: Minimum confidence for predictions
        iou_threshold: IoU threshold for matching
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating YOLO model: {model_path}")
    print(f"Validation LMDB: {lmdb_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Device: {device}\n")

    # Load YOLO model
    model = YOLO(model_path)
    model.to(device)

    # Open LMDB
    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False
    )

    all_predictions = []
    all_targets = []

    with env.begin(write=False) as txn:
        # Get length
        length = int(txn.get(b'__len__').decode('ascii'))
        print(f"Processing {length} validation images...\n")

        # Process each sample
        for idx in tqdm(range(length), desc="Evaluating"):
            # Get serialized data
            key = f'{idx:08d}'.encode('ascii')
            data = txn.get(key)
            if data is None:
                continue

            sample = pickle.loads(data)

            # Decode image
            image_bytes = sample['image']
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                print(f"Warning: Could not decode image {idx}")
                continue

            # Convert BGR to RGB (YOLO expects RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            # Get ground truth boxes and labels
            gt_boxes = sample['boxes']  # (N, 4) [xmin, ymin, xmax, ymax]
            gt_labels = sample['labels']  # (N,)

            # Filter out background labels (0)
            mask = gt_labels > 0
            gt_boxes = gt_boxes[mask]
            gt_labels = gt_labels[mask]

            # Run YOLO inference
            results = model.predict(image_rgb, conf=confidence_threshold, iou=iou_threshold, verbose=False)

            # Extract predictions
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4) [xmin, ymin, xmax, ymax]
                scores = results[0].boxes.conf.cpu().numpy()  # (N,)
                labels = results[0].boxes.cls.cpu().numpy()  # (N,)

                # Filter by confidence
                conf_mask = scores >= confidence_threshold
                pred_boxes = boxes[conf_mask]
                pred_scores = scores[conf_mask]
                pred_labels = labels[conf_mask] + 1  # YOLO uses 0-indexed, we use 1-indexed
            else:
                pred_boxes = np.array([])
                pred_scores = np.array([])
                pred_labels = np.array([])

            # Store predictions and targets
            all_predictions.append({
                'boxes': pred_boxes,
                'scores': pred_scores,
                'labels': pred_labels
            })

            all_targets.append({
                'boxes': gt_boxes,
                'labels': gt_labels
            })

    env.close()

    # Compute metrics
    metrics = compute_detection_metrics(all_predictions, all_targets, iou_threshold=iou_threshold)

    return metrics, all_predictions, all_targets


def print_metrics(metrics):
    """Print metrics in a formatted way"""
    print("\n" + "=" * 60)
    print("YOLO Evaluation Results")
    print("=" * 60)
    print(f"Composite Score: {metrics['composite_score']:.4f}")
    print(f"F1 Score:        {metrics['f1']:.4f}")
    print(f"Mean IoU:        {metrics['mean_iou']:.4f}")
    print(f"Precision:       {metrics['precision']:.4f}")
    print(f"Recall:          {metrics['recall']:.4f}")
    print("\nDetection Statistics:")
    print(f"  True Positives:  {metrics['tp']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print(f"  Total Predictions: {metrics['num_predictions']}")
    print(f"  Total Targets:     {metrics['num_targets']}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO model with Faster R-CNN-compatible metrics')

    parser.add_argument('--model', type=str, required=True,
                        help='Path to YOLO model weights (.pt file)')
    parser.add_argument('--val-lmdb', type=str, default='../data/val.lmdb',
                        help='Path to validation LMDB database')
    parser.add_argument('--confidence-threshold', type=float, default=0.05,
                        help='Minimum confidence for predictions (default: 0.05 for training monitoring)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for matching predictions to ground truth (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}")
        exit(1)

    # Check if LMDB exists
    if not Path(args.val_lmdb).exists():
        print(f"ERROR: Validation LMDB not found: {args.val_lmdb}")
        print("Please run create_lmdb.py first to create LMDB databases")
        exit(1)

    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        device = 'cpu'

    # Evaluate model
    metrics, predictions, targets = evaluate_yolo_on_lmdb(
        model_path=args.model,
        lmdb_path=args.val_lmdb,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        device=device
    )

    # Print results
    print_metrics(metrics)

    # Print comparison guidance
    print("\n" + "=" * 60)
    print("Comparison with Faster R-CNN/RetinaNet")
    print("=" * 60)
    print("You can now compare these metrics directly with Faster R-CNN")
    print("and RetinaNet results from train_improved.py\n")
    print("Composite Score formula:")
    print("  0.40 × F1 + 0.25 × IoU + 0.20 × Recall + 0.15 × Precision")
    print("\nConfidence Thresholds:")
    print("  - Training monitoring: 0.05 (see learning progress)")
    print("  - Validation/benchmark: 0.3-0.5 (compare models)")
    print("  - Production deployment: 0.5-0.7 (high-confidence only)")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
