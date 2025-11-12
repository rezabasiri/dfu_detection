#!/usr/bin/env python3
"""
Compare Test Model Results
Compares training results from all three models and generates a summary report.
"""

import json
import torch
from pathlib import Path
from datetime import datetime
import sys

def load_checkpoint_metrics(checkpoint_path):
    """Load metrics from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        return {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'train_loss': checkpoint.get('train_loss', 0.0),
            'val_loss': checkpoint.get('val_loss', 0.0),
            'f1_score': checkpoint.get('f1_score', 0.0),
            'mean_iou': checkpoint.get('mean_iou', 0.0),
            'precision': checkpoint.get('precision', 0.0),
            'recall': checkpoint.get('recall', 0.0),
            'composite_score': checkpoint.get('composite_score', 0.0),
            'learning_rate': checkpoint.get('learning_rate', 0.0),
            'model_name': checkpoint.get('model_name', 'unknown'),
            'backbone': checkpoint.get('backbone', 'unknown'),
        }
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None

def format_metric(value, decimals=4):
    """Format a metric value."""
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return str(value)

def print_model_comparison(models_data):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("  MODEL COMPARISON - Test Training Results")
    print("=" * 100)

    # Print header
    header = f"{'Model':<20} {'Composite':<12} {'F1':<10} {'IoU':<10} {'Recall':<10} {'Precision':<10}"
    print(f"\n{header}")
    print("-" * 100)

    # Print each model
    for model_name, data in models_data.items():
        if data:
            comp = format_metric(data['composite_score'])
            f1 = format_metric(data['f1_score'])
            iou = format_metric(data['mean_iou'])
            recall = format_metric(data['recall'])
            precision = format_metric(data['precision'])

            print(f"{model_name:<20} {comp:<12} {f1:<10} {iou:<10} {recall:<10} {precision:<10}")
        else:
            print(f"{model_name:<20} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

    print("\n" + "=" * 100)

def print_detailed_metrics(models_data):
    """Print detailed metrics for each model."""
    print("\n" + "=" * 100)
    print("  DETAILED METRICS")
    print("=" * 100)

    for model_name, data in models_data.items():
        print(f"\n{model_name.upper()}")
        print("-" * 50)

        if data:
            print(f"  Model Architecture:  {data['model_name']}")
            print(f"  Backbone:            {data['backbone']}")
            print(f"  Final Epoch:         {data['epoch']}")
            print(f"  Learning Rate:       {format_metric(data['learning_rate'], 6)}")
            print(f"  ")
            print(f"  Training Loss:       {format_metric(data['train_loss'])}")
            print(f"  Validation Loss:     {format_metric(data['val_loss'])}")
            print(f"  ")
            print(f"  Composite Score:     {format_metric(data['composite_score'])}")
            print(f"  F1 Score:            {format_metric(data['f1_score'])}")
            print(f"  Mean IoU:            {format_metric(data['mean_iou'])}")
            print(f"  Recall:              {format_metric(data['recall'])}")
            print(f"  Precision:           {format_metric(data['precision'])}")
        else:
            print("  No checkpoint found")

def print_recommendations(models_data):
    """Print recommendations based on results."""
    print("\n" + "=" * 100)
    print("  RECOMMENDATIONS")
    print("=" * 100)

    # Find best model for each metric
    valid_models = {k: v for k, v in models_data.items() if v is not None}

    if not valid_models:
        print("\nNo valid models found to compare.")
        return

    best_composite = max(valid_models.items(), key=lambda x: x[1]['composite_score'])
    best_f1 = max(valid_models.items(), key=lambda x: x[1]['f1_score'])
    best_recall = max(valid_models.items(), key=lambda x: x[1]['recall'])
    best_precision = max(valid_models.items(), key=lambda x: x[1]['precision'])

    print(f"\nBest Overall (Composite Score): {best_composite[0]}")
    print(f"  Score: {format_metric(best_composite[1]['composite_score'])}")

    print(f"\nBest F1 Score: {best_f1[0]}")
    print(f"  F1: {format_metric(best_f1[1]['f1_score'])}")

    print(f"\nBest Recall (Fewest False Negatives): {best_recall[0]}")
    print(f"  Recall: {format_metric(best_recall[1]['recall'])}")

    print(f"\nBest Precision (Fewest False Positives): {best_precision[0]}")
    print(f"  Precision: {format_metric(best_precision[1]['precision'])}")

    print("\n" + "-" * 100)
    print("\nGeneral Guidance:")
    print("  - For medical diagnosis (minimize missing ulcers): Use model with highest RECALL")
    print("  - For balanced performance: Use model with highest F1 SCORE")
    print("  - For reducing false alarms: Use model with highest PRECISION")
    print("  - For overall best performance: Use model with highest COMPOSITE SCORE")

def main():
    """Compare all test models."""
    print("\n" + "=" * 100)
    print("  DFU DETECTION - TEST MODEL COMPARISON")
    print("=" * 100)
    print(f"\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Find checkpoint directory - go up from test_small -> scripts -> dfu_detection
    script_dir = Path(__file__).parent        # test_small
    scripts_dir = script_dir.parent           # scripts
    project_dir = scripts_dir.parent          # dfu_detection
    checkpoint_base = project_dir / "checkpoints_test"

    if not checkpoint_base.exists():
        print(f"\n⚠ Error: Checkpoint directory not found: {checkpoint_base}")
        print("\nPlease run test training first:")
        print("  python run_test_training.py\n")
        return

    # Models to compare
    models = ['faster_rcnn', 'retinanet', 'yolo']

    # Load metrics for each model
    models_data = {}
    for model_name in models:
        checkpoint_path = checkpoint_base / model_name / "best_model.pth"
        if checkpoint_path.exists():
            print(f"Loading metrics for {model_name}...")
            models_data[model_name] = load_checkpoint_metrics(checkpoint_path)
        else:
            print(f"⚠ Checkpoint not found for {model_name}: {checkpoint_path}")
            models_data[model_name] = None

    # Print comparisons
    print_model_comparison(models_data)
    print_detailed_metrics(models_data)
    print_recommendations(models_data)

    # Print next steps
    print("\n" + "=" * 100)
    print("  NEXT STEPS")
    print("=" * 100)
    print("\n1. Run full training on complete dataset:")
    print("   python train_improved.py --model <model_name> --config configs/<config>.yaml")

    print("\n2. Evaluate models on test set:")
    print("   python evaluate.py --checkpoint checkpoints_test/<model>/best_model.pth")

    print("\n3. Run inference on new images:")
    print("   python inference_improved.py --checkpoint <path> --image <image_path>")

    print("\n4. View training logs:")
    print("   cat checkpoints_test/*/training_log_*.txt\n")

if __name__ == "__main__":
    main()
