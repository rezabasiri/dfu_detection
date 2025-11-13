"""
Evaluate YOLO Model on Test Data

Loads the best YOLO model from train_all_models.py and evaluates on test set.
Computes metrics compatible with Faster R-CNN/RetinaNet for comparison.

Metrics computed:
- F1 Score
- Mean IoU
- Precision
- Recall
- Composite Score (0.40×F1 + 0.25×IoU + 0.20×Recall + 0.15×Precision)

Usage:
    # Evaluate best YOLO model on test data
    python yolo_test_data_evaluate.py

    # Use custom model path
    python yolo_test_data_evaluate.py --model ../checkpoints/yolo/weights/best.pt

    # Use custom confidence threshold for production evaluation
    python yolo_test_data_evaluate.py --confidence 0.5
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import lmdb
import pickle
import cv2
from tqdm import tqdm
import json
from datetime import datetime

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
    xmin_inter = max(box1[0], box2[0])
    ymin_inter = max(box1[1], box2[1])
    xmax_inter = min(box1[2], box2[2])
    ymax_inter = min(box1[3], box2[3])

    inter_width = max(0, xmax_inter - xmin_inter)
    inter_height = max(0, ymax_inter - ymin_inter)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

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
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0
    num_matched = 0

    for pred, target in zip(all_predictions, all_targets):
        pred_boxes = pred['boxes']
        target_boxes = target['boxes']

        if len(pred_boxes) == 0 and len(target_boxes) == 0:
            continue

        if len(pred_boxes) == 0:
            total_fn += len(target_boxes)
            continue

        if len(target_boxes) == 0:
            total_fp += len(pred_boxes)
            continue

        matched_targets = set()

        for pred_box in pred_boxes:
            best_iou = 0.0
            best_target_idx = -1

            for target_idx, target_box in enumerate(target_boxes):
                if target_idx in matched_targets:
                    continue

                iou = box_iou(pred_box, target_box)
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = target_idx

            if best_iou >= iou_threshold:
                total_tp += 1
                total_iou += best_iou
                num_matched += 1
                matched_targets.add(best_target_idx)
            else:
                total_fp += 1

        total_fn += len(target_boxes) - len(matched_targets)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = total_iou / num_matched if num_matched > 0 else 0.0

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


def evaluate_yolo_on_test_set(model_path, test_lmdb_path, confidence_threshold=0.5, iou_threshold=0.5, device='cuda'):
    """
    Evaluate YOLO model on test LMDB dataset

    Args:
        model_path: Path to YOLO model weights (.pt file)
        test_lmdb_path: Path to test LMDB database
        confidence_threshold: Minimum confidence for predictions
        iou_threshold: IoU threshold for matching
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*80}")
    print("YOLO TEST SET EVALUATION")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Test LMDB: {test_lmdb_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(model_path)
    model.to(device)
    print("✓ Model loaded\n")

    # Open LMDB
    if not Path(test_lmdb_path).exists():
        print(f"ERROR: Test LMDB not found: {test_lmdb_path}")
        print("Available LMDB files should include test.lmdb")
        print("If test.lmdb doesn't exist, you may need to create it with create_lmdb.py")
        exit(1)

    env = lmdb.open(
        str(test_lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False
    )

    all_predictions = []
    all_targets = []

    with env.begin(write=False) as txn:
        length = int(txn.get(b'__len__').decode('ascii'))
        print(f"Processing {length} test images...\n")

        for idx in tqdm(range(length), desc="Evaluating"):
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

            # Get ground truth boxes and labels
            gt_boxes = sample['boxes']
            gt_labels = sample['labels']

            # Filter out background labels (0)
            mask = gt_labels > 0
            gt_boxes = gt_boxes[mask]
            gt_labels = gt_labels[mask]

            # Run YOLO inference
            results = model.predict(image_rgb, conf=confidence_threshold, iou=iou_threshold, verbose=False)

            # Extract predictions
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                labels = results[0].boxes.cls.cpu().numpy()

                conf_mask = scores >= confidence_threshold
                pred_boxes = boxes[conf_mask]
                pred_scores = scores[conf_mask]
                pred_labels = labels[conf_mask] + 1
            else:
                pred_boxes = np.array([])
                pred_scores = np.array([])
                pred_labels = np.array([])

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
    print("\nComputing metrics...")
    metrics = compute_detection_metrics(all_predictions, all_targets, iou_threshold=iou_threshold)

    return metrics, all_predictions, all_targets


def print_metrics(metrics):
    """Print metrics in a formatted table"""
    print("\n" + "="*80)
    print("TEST SET RESULTS")
    print("="*80)
    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-"*80)
    print(f"{'Composite Score':<20} {metrics['composite_score']:>10.4f}  ⭐ (Medical-optimized)")
    print(f"{'F1 Score':<20} {metrics['f1']:>10.4f}")
    print(f"{'Mean IoU':<20} {metrics['mean_iou']:>10.4f}  (Localization quality)")
    print(f"{'Precision':<20} {metrics['precision']:>10.4f}  (False alarm rate)")
    print(f"{'Recall':<20} {metrics['recall']:>10.4f}  ⚕️ (Don't miss ulcers!)")
    print("\n" + "-"*80)
    print("Detection Statistics:")
    print("-"*80)
    print(f"{'True Positives':<20} {metrics['tp']:>10,}  ✓ Correctly detected ulcers")
    print(f"{'False Positives':<20} {metrics['fp']:>10,}  ⚠ False alarms")
    print(f"{'False Negatives':<20} {metrics['fn']:>10,}  ✗ Missed ulcers")
    print(f"{'Total Predictions':<20} {metrics['num_predictions']:>10,}")
    print(f"{'Total Ground Truth':<20} {metrics['num_targets']:>10,}")
    print("="*80)


def save_results(metrics, output_dir, model_name):
    """Save results to JSON file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"test_evaluation_{model_name}_{timestamp}.json"

    results = {
        'model': model_name,
        'timestamp': timestamp,
        'metrics': metrics,
        'composite_formula': '0.40×F1 + 0.25×IoU + 0.20×Recall + 0.15×Precision'
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate YOLO model on test data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate best YOLO model with default settings
    python yolo_test_data_evaluate.py

    # Use custom model and higher confidence for production eval
    python yolo_test_data_evaluate.py --model ../checkpoints/yolo/weights/best.pt --confidence 0.5

    # Evaluate on validation set instead of test
    python yolo_test_data_evaluate.py --test-lmdb ../data/val.lmdb --output ../results/yolo_val_eval

Confidence threshold guidance:
    - 0.3: Balanced evaluation (default for test set)
    - 0.5: Production deployment (high confidence only)
    - 0.7: Very conservative (minimal false positives)
        """
    )

    parser.add_argument('--model', type=str,
                        default='../checkpoints/yolo/weights/best.pt',
                        help='Path to YOLO model weights (.pt file)')
    parser.add_argument('--test-lmdb', type=str,
                        default='../data/test.lmdb',
                        help='Path to test LMDB database')
    parser.add_argument('--confidence', type=float,
                        default=0.3,
                        help='Confidence threshold for predictions (default: 0.3)')
    parser.add_argument('--iou-threshold', type=float,
                        default=0.5,
                        help='IoU threshold for matching predictions (default: 0.5)')
    parser.add_argument('--device', type=str,
                        default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--output', type=str,
                        default='../results/yolo_test_evaluation',
                        help='Output directory for results')

    args = parser.parse_args()

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {args.model}")
        print("\nExpected YOLO checkpoint locations:")
        print("  Production: ../checkpoints/yolo/weights/best.pt")
        print("  Test: ../checkpoints_test/yolo/weights/best.pt")
        exit(1)

    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        device = 'cpu'

    # Evaluate model
    metrics, predictions, targets = evaluate_yolo_on_test_set(
        model_path=args.model,
        test_lmdb_path=args.test_lmdb,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou_threshold,
        device=device
    )

    # Print results
    print_metrics(metrics)

    # Save results
    model_name = model_path.stem  # 'best' or similar
    save_results(metrics, args.output, model_name)

    # Interpretation guidance
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("\nComposite Score: Medical-optimized metric for DFU detection")
    print("  Formula: 0.40×F1 + 0.25×IoU + 0.20×Recall + 0.15×Precision")
    print("  Why? Prioritizes finding all ulcers (recall) while maintaining accuracy")
    print("\nBenchmark Targets (for test set):")
    print("  ⭐ Excellent:  Composite > 0.75, Recall > 0.85")
    print("  ✓ Good:        Composite > 0.70, Recall > 0.80")
    print("  ~ Acceptable:  Composite > 0.65, Recall > 0.75")
    print("\nKey Metrics:")
    print("  • Recall: Most important for medical diagnosis (don't miss ulcers!)")
    print("  • Precision: Minimize false alarms (reduce unnecessary interventions)")
    print("  • F1: Balance between recall and precision")
    print("  • IoU: Localization quality (how well boxes align)")
    print("\nComparison:")
    print("  Compare these results with Faster R-CNN and RetinaNet from")
    print("  train_improved.py to choose the best model for deployment.")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
