"""
Train all detection models for comparison
Trains Faster R-CNN, RetinaNet, and YOLO with same data and configuration
"""

import subprocess
import sys
import argparse
from pathlib import Path


# Model configurations
MODELS = [
    {
        'name': 'faster_rcnn',
        'config': 'configs/faster_rcnn_b5.yaml',
        'description': 'Faster R-CNN with EfficientNet-B5 (current production model)'
    },
    {
        'name': 'retinanet',
        'config': 'configs/retinanet.yaml',
        'description': 'RetinaNet with EfficientNet-B3 (single-stage, focal loss)'
    },
    {
        'name': 'yolo',
        'config': 'configs/yolov8.yaml',
        'description': 'YOLOv8m (fastest inference, anchor-free)'
    }
]


def train_model(model_config: dict, args):
    """
    Train a single model

    Args:
        model_config: Model configuration dict
        args: Command-line arguments
    """
    model_name = model_config['name']
    config_path = model_config['config']
    description = model_config['description']

    print("\n" + "="*80)
    print(f"TRAINING: {model_name.upper()}")
    print("="*80)
    print(f"Description: {description}")
    print(f"Config: {config_path}")
    print("="*80 + "\n")

    # Build command
    cmd = [
        sys.executable,  # Use same Python interpreter
        'train_improved.py',
        '--model', model_name,
        '--config', config_path
    ]

    # Add optional arguments
    if args.epochs:
        cmd.extend(['--epochs', str(args.epochs)])
    if args.batch_size:
        cmd.extend(['--batch-size', str(args.batch_size)])
    if args.lr:
        cmd.extend(['--lr', str(args.lr)])
    if args.device:
        cmd.extend(['--device', args.device])

    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {model_name.upper()} training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {model_name.upper()} training failed with error code {e.returncode}")
        if not args.continue_on_error:
            print(f"Stopping due to error. Use --continue-on-error to train remaining models.")
            return False
        else:
            print(f"Continuing to next model...")
            return True
    except KeyboardInterrupt:
        print(f"\n\n⚠ Training interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Train all detection models for comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models that will be trained:
  1. Faster R-CNN (EfficientNet-B5): Two-stage detector, high accuracy
  2. RetinaNet (EfficientNet-B3): Single-stage with focal loss, good recall
  3. YOLOv8 (Medium): Fastest inference, good for deployment

Each model will be trained with its own configuration file and save
checkpoints to separate directories:
  - ../checkpoints/faster_rcnn/
  - ../checkpoints/retinanet/
  - ../checkpoints/yolo/

Usage examples:
  # Train all models with default configs
  python train_all_models.py

  # Train specific models
  python train_all_models.py --models faster_rcnn retinanet

  # Override epochs for quick testing
  python train_all_models.py --epochs 10

  # Continue on error
  python train_all_models.py --continue-on-error
        """
    )

    parser.add_argument('--models', type=str, nargs='+',
                       choices=['faster_rcnn', 'retinanet', 'yolo'],
                       default=None,
                       help='Specific models to train (default: all)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs from config')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size from config')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate from config')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue training remaining models if one fails')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print what would be done without actually training')

    args = parser.parse_args()

    # Filter models if specific ones requested
    if args.models:
        models_to_train = [m for m in MODELS if m['name'] in args.models]
    else:
        models_to_train = MODELS

    print("="*80)
    print("DFU DETECTION - TRAIN ALL MODELS FOR COMPARISON")
    print("="*80)
    print(f"\nModels to train: {len(models_to_train)}")
    for model in models_to_train:
        print(f"  - {model['name']}: {model['description']}")

    if args.epochs:
        print(f"\nOverride epochs: {args.epochs}")
    if args.batch_size:
        print(f"Override batch size: {args.batch_size}")
    if args.lr:
        print(f"Override learning rate: {args.lr}")

    print(f"\nDevice: {args.device}")
    print(f"Continue on error: {args.continue_on_error}")

    if args.dry_run:
        print("\n⚠ DRY RUN - No training will be performed")
        for model in models_to_train:
            cmd_parts = [
                'train_improved.py',
                f'--model {model["name"]}',
                f'--config {model["config"]}'
            ]
            if args.epochs:
                cmd_parts.append(f'--epochs {args.epochs}')
            if args.batch_size:
                cmd_parts.append(f'--batch-size {args.batch_size}')
            if args.lr:
                cmd_parts.append(f'--lr {args.lr}')
            print(f"\nWould run: python {' '.join(cmd_parts)}")
        return

    print("\n" + "="*80)
    input("Press Enter to start training, or Ctrl+C to cancel...")
    print("="*80)

    # Train each model
    results = {}
    for i, model in enumerate(models_to_train, 1):
        print(f"\n{'='*80}")
        print(f"MODEL {i}/{len(models_to_train)}")
        print(f"{'='*80}")

        success = train_model(model, args)
        results[model['name']] = success

        if not success and not args.continue_on_error:
            print(f"\n⚠ Stopping due to training failure")
            break

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]

    if successful:
        print(f"\n✓ Successfully trained ({len(successful)}):")
        for name in successful:
            print(f"  - {name}")

    if failed:
        print(f"\n✗ Failed ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Compare models with evaluate.py:")
    print("   python evaluate.py --checkpoint ../checkpoints/faster_rcnn/best_model.pth")
    print("   python evaluate.py --checkpoint ../checkpoints/retinanet/best_model.pth")
    print("   python evaluate.py --checkpoint ../checkpoints/yolo/best_model.pth")

    print("\n2. Run inference:")
    print("   python inference_improved.py --checkpoint <path> --image <image>")

    print("\n3. Check training logs:")
    for name in successful:
        print(f"   ../checkpoints/{name}/training_log_*.txt")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
