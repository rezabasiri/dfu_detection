"""
Evaluation script for DFU detection
Computes metrics like mAP, precision, recall
Supports multiple architectures: Faster R-CNN, RetinaNet, YOLO
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
import argparse
from typing import List, Dict

from dataset import DFUDataset, get_val_transforms, collate_fn
from models import ModelFactory, create_from_checkpoint

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two boxes

    Args:
        box1, box2: [xmin, ymin, xmax, ymax]

    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def compute_metrics(predictions: List[Dict], targets: List[Dict], iou_threshold: float = 0.5):
    """
    Compute detection metrics

    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        targets: List of target dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for considering a detection as correct

    Returns:
        Dictionary with metrics
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    all_scores = []
    all_ious = []

    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].numpy()
        pred_scores = pred['scores'].numpy()
        target_boxes = target['boxes'].numpy()

        matched_gt = set()

        sorted_indices = np.argsort(-pred_scores)

        for idx in sorted_indices:
            pred_box = pred_boxes[idx]
            pred_score = pred_scores[idx]

            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(target_boxes):
                if gt_idx in matched_gt:
                    continue

                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            all_scores.append(pred_score)
            all_ious.append(best_iou)

            if best_iou >= iou_threshold:
                true_positives += 1
                matched_gt.add(best_gt_idx)
            else:
                false_positives += 1

        false_negatives += len(target_boxes) - len(matched_gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'mean_iou': np.mean(all_ious) if all_ious else 0,
        'mean_confidence': np.mean(all_scores) if all_scores else 0
    }

@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    device,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5
):
    """
    Evaluate model on a dataset

    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        confidence_threshold: Minimum confidence for predictions
        iou_threshold: IoU threshold for matching predictions to ground truth

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_targets = []

    print(f"\nEvaluating on {len(data_loader)} batches...")

    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]

        predictions = model(images)

        for pred in predictions:
            # Filter by confidence threshold
            mask = pred['scores'] >= confidence_threshold

            # Filter predictions to only keep ulcers (not background)
            # For 2-class: ulcer = class 1, for 3-class: ulcer = class 2
            if 'labels' in pred and len(pred['labels']) > 0:
                # Keep only non-background predictions (label > 0)
                label_mask = pred['labels'] > 0
                mask = mask & label_mask

            filtered_pred = {
                'boxes': pred['boxes'][mask].cpu(),
                'scores': pred['scores'][mask].cpu(),
                'labels': pred['labels'][mask].cpu()
            }
            all_predictions.append(filtered_pred)

        # Filter targets to only include ulcer boxes (not background)
        for t in targets:
            if len(t['labels']) > 0:
                # Keep only non-background targets (label > 0)
                ulcer_mask = t['labels'] > 0
                filtered_target = {
                    'boxes': t['boxes'][ulcer_mask].cpu(),
                    'labels': t['labels'][ulcer_mask].cpu(),
                    'image_id': t['image_id'].cpu(),
                    'area': t['area'][ulcer_mask].cpu() if 'area' in t else torch.tensor([]),
                    'iscrowd': t['iscrowd'][ulcer_mask].cpu() if 'iscrowd' in t else torch.tensor([])
                }
            else:
                filtered_target = {k: v.cpu() for k, v in t.items()}
            all_targets.append(filtered_target)

    metrics = compute_metrics(all_predictions, all_targets, iou_threshold)

    return metrics, all_predictions, all_targets

def main():
    """Main evaluation function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate DFU detection model')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/faster_rcnn/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--test-csv', type=str, default='../data/test.csv',
                       help='Path to test CSV file')
    parser.add_argument('--image-folder', type=str,
                       default='/mnt/c/Users/90rez/OneDrive - University of Toronto/PhDUofT/SideProjects/DFU_Detection_Asem/DFUC2022_train_images',
                       help='Path to images folder')
    parser.add_argument('--conf-thresholds', type=float, nargs='+', default=[0.3, 0.5, 0.7],
                       help='Confidence thresholds to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for evaluation')

    args = parser.parse_args()

    print("="*60)
    print("DFU Detection - Model Evaluation")
    print("="*60)

    checkpoint_path = args.checkpoint
    test_csv = args.test_csv
    image_folder = args.image_folder

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train_improved.py")
        return

    if not os.path.exists(test_csv):
        print(f"Error: Test CSV not found at {test_csv}")
        print("Please run data_preprocessing.py first")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    print(f"\nLoading model from checkpoint...")
    print(f"Checkpoint: {checkpoint_path}")

    # Use ModelFactory to auto-detect and load model
    try:
        detector = create_from_checkpoint(checkpoint_path, device=device)
        model = detector.get_model()
        model.eval()

        # Get checkpoint metadata
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        img_size = checkpoint.get('img_size', 640)
        model_name = checkpoint.get('model_name', 'faster_rcnn')

        print(f"\nModel Info:")
        print(f"  Architecture: {model_name}")
        print(f"  Backbone: {detector.backbone_name}")
        print(f"  Image size: {img_size}")
        print(f"  Number of classes: {detector.num_classes}")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Attempting legacy loading method...")

        # Fallback for old checkpoints without model_name
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        from train_efficientdet import create_efficientdet_model

        backbone = checkpoint.get('backbone', 'efficientnet_b5')
        img_size = checkpoint.get('img_size', 640)
        num_classes = checkpoint.get('num_classes', 2)

        print(f"\nLegacy checkpoint detected:")
        print(f"  Backbone: {backbone}")
        print(f"  Image size: {img_size}")

        model = create_efficientdet_model(num_classes=num_classes, backbone=backbone, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

    # Print checkpoint info (handle different checkpoint formats)
    epoch_info = checkpoint.get('epoch', 'unknown')
    if 'loss' in checkpoint:
        print(f"Loaded model from epoch {epoch_info}, loss: {checkpoint['loss']:.4f}")
    elif 'val_loss' in checkpoint:
        print(f"Loaded model from epoch {epoch_info}, val_loss: {checkpoint['val_loss']:.4f}")
    elif 'train_loss' in checkpoint:
        print(f"Loaded model from epoch {epoch_info}, train_loss: {checkpoint['train_loss']:.4f}")
    else:
        print(f"Loaded model from epoch {epoch_info}")

    print(f"\nLoading test dataset...")
    test_dataset = DFUDataset(
        csv_file=test_csv,
        image_folder=image_folder,
        transforms=get_val_transforms(img_size),  # Use auto-detected img_size
        mode='test'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    confidence_thresholds = args.conf_thresholds

    results = {}

    for conf_thresh in confidence_thresholds:
        print(f"\n{'='*60}")
        print(f"Evaluating with confidence threshold: {conf_thresh}")
        print(f"{'='*60}")

        metrics, predictions, targets = evaluate_model(
            model=model,
            data_loader=test_loader,
            device=device,
            confidence_threshold=conf_thresh,
            iou_threshold=0.5
        )

        results[f"conf_{conf_thresh}"] = metrics

        print(f"\nMetrics (confidence >= {conf_thresh}):")
        print(f"  Precision:        {metrics['precision']:.4f}")
        print(f"  Recall:           {metrics['recall']:.4f}")
        print(f"  F1 Score:         {metrics['f1_score']:.4f}")
        print(f"  True Positives:   {metrics['true_positives']}")
        print(f"  False Positives:  {metrics['false_positives']}")
        print(f"  False Negatives:  {metrics['false_negatives']}")
        print(f"  Mean IoU:         {metrics['mean_iou']:.4f}")
        print(f"  Mean Confidence:  {metrics['mean_confidence']:.4f}")

    results_path = "../results/evaluation_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    results_serializable = convert_to_python_types(results)

    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()