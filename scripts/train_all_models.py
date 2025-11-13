"""
Train all detection models for comparison
Trains Faster R-CNN, RetinaNet, and YOLO with same data and configuration

Note: YOLO uses train_yolo.py (native interface), others use train_improved.py
"""

import subprocess
import sys
import argparse
import yaml
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

    # Build command - YOLO uses dedicated script, others use unified script
    if model_name == 'yolo':
        # YOLO uses train_yolo.py with native interface
        # Load YOLO config to get training parameters
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                yolo_config = yaml.safe_load(f)

            # Extract parameters from YAML (with overrides from args)
            epochs = args.epochs if args.epochs else yolo_config.get('training', {}).get('num_epochs', 300)
            batch_size = args.batch_size if args.batch_size else yolo_config.get('training', {}).get('batch_size', 36)
            img_size = args.img_size if args.img_size else yolo_config.get('training', {}).get('img_size', 512)
            model_size = yolo_config.get('model', {}).get('model_size', 'yolov8m')
            save_period = yolo_config.get('checkpoint', {}).get('save_every_n_epochs', 10)

            # Resolve paths relative to config file location (not current directory)
            config_dir = config_file.parent.absolute()
            train_lmdb_rel = yolo_config.get('data', {}).get('train_lmdb', '../data/train.lmdb')
            val_lmdb_rel = yolo_config.get('data', {}).get('val_lmdb', '../data/val.lmdb')
            project_rel = yolo_config.get('checkpoint', {}).get('save_dir', '../checkpoints/yolo').rsplit('/', 1)[0]

            train_lmdb = str((config_dir / train_lmdb_rel).resolve())
            val_lmdb = str((config_dir / val_lmdb_rel).resolve())
            project = str((config_dir / project_rel).resolve())
        else:
            # Fallback defaults (resolve relative to current directory)
            epochs = args.epochs if args.epochs else 300
            batch_size = args.batch_size if args.batch_size else 36
            img_size = args.img_size if args.img_size else 512
            model_size = 'yolov8m'
            save_period = 10
            train_lmdb = str((Path('../data/train.lmdb').resolve()))
            val_lmdb = str((Path('../data/val.lmdb').resolve()))
            project = str((Path('../checkpoints').resolve()))

        # Check for existing checkpoint to resume
        checkpoint_path = Path(project) / 'yolo' / 'weights' / 'last.pt'

        cmd = [
            sys.executable,
            'train_yolo.py',
            '--model', model_size,
            '--train-lmdb', train_lmdb,
            '--val-lmdb', val_lmdb,
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
            '--img-size', str(img_size),
            '--device', args.device if args.device else 'cuda',
            '--project', project,
            '--name', 'yolo',
            '--save-period', str(save_period)
        ]

        # Add resume flag if checkpoint exists and resume requested
        if args.resume and checkpoint_path.exists():
            print(f"  Found checkpoint: {checkpoint_path}")
            print(f"  Resuming YOLO training from last checkpoint...")
            # For YOLO, we need to use the resume parameter differently
            # Will be handled by train_yolo.py when it detects existing training
    else:
        # Faster R-CNN and RetinaNet use train_improved.py
        # Check for existing checkpoint to resume (auto-detected by train_improved.py)
        checkpoint_dir = Path('..') / 'checkpoints' / model_name
        resume_checkpoint = checkpoint_dir / 'resume_training.pth'

        if args.resume and resume_checkpoint.exists():
            print(f"  Found checkpoint: {resume_checkpoint}")
            print(f"  {model_name.upper()} will auto-resume from checkpoint...")

        cmd = [
            sys.executable,
            'train_improved.py',
            '--model', model_name,
            '--config', config_path
        ]

        # Add optional arguments
        if args.epochs:
            cmd.extend(['--epochs', str(args.epochs)])
        if args.batch_size:
            cmd.extend(['--batch-size', str(args.batch_size)])
        # Note: img_size not supported by train_improved.py - must be set in config YAML
        if args.img_size:
            print(f"  Note: --img-size ignored for {model_name} (set in config YAML instead)")
        if args.lr:
            cmd.extend(['--lr', str(args.lr)])
        if args.device:
            cmd.extend(['--device', args.device])

    # Run training with real-time output
    print(f"\nCommand: {' '.join(cmd)}\n")
    print("="*80)
    print("Training output (streaming in real-time):")
    print("="*80 + "\n")

    try:
        # Run with real-time output (stdout/stderr not captured)
        # This allows continuous output like train_improved.py
        result = subprocess.run(cmd, check=True)
        print(f"\n{'='*80}")
        print(f"✓ {model_name.upper()} training completed successfully")
        print(f"{'='*80}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*80}")
        print(f"✗ {model_name.upper()} training failed with error code {e.returncode}")
        print(f"{'='*80}\n")
        if not args.continue_on_error:
            print(f"Stopping due to error. Use --continue-on-error to train remaining models.")
            return False
        else:
            print(f"Continuing to next model...")
            return True
    except KeyboardInterrupt:
        print(f"\n\n⚠ Training interrupted by user")
        print(f"Note: To resume later, use --resume flag")
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
  - ../checkpoints/faster_rcnn/best_model.pth
  - ../checkpoints/retinanet/best_model.pth
  - ../checkpoints/yolo/weights/best.pt  (note: YOLO uses weights/ subdirectory)

Note: YOLO uses train_yolo.py (native Ultralytics interface) for optimal
performance. Faster R-CNN and RetinaNet use train_improved.py.

Usage examples:
  # Train all models with default configs
  python train_all_models.py

  # Train specific models
  python train_all_models.py --models faster_rcnn retinanet

  # Override training parameters
  python train_all_models.py --epochs 10 --batch-size 12 --img-size 512

  # Quick test with small parameters
  python train_all_models.py --epochs 5 --batch-size 4 --img-size 256

  # Resume interrupted training
  python train_all_models.py --resume

  # Continue training other models if one fails
  python train_all_models.py --continue-on-error

  # See what would be run without actually training
  python train_all_models.py --dry-run --epochs 10 --img-size 640
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
    parser.add_argument('--img-size', type=int, default=None,
                       help='Override image size (YOLO only; Faster R-CNN/RetinaNet use config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate from config (Faster R-CNN/RetinaNet only)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue training remaining models if one fails')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint if available')
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

    print("\n" + "-"*80)
    print("Training Configuration")
    print("-"*80)
    if args.epochs:
        print(f"Epochs:        {args.epochs} (override)")
    else:
        print(f"Epochs:        From config files")
    if args.batch_size:
        print(f"Batch size:    {args.batch_size} (override)")
    else:
        print(f"Batch size:    From config files")
    if args.img_size:
        print(f"Image size:    {args.img_size} (override for YOLO only)")
    else:
        print(f"Image size:    From config files")
    if args.lr:
        print(f"Learning rate: {args.lr} (override, Faster R-CNN/RetinaNet only)")

    print(f"\nDevice:         {args.device}")
    print(f"Resume:         {args.resume}")
    print(f"Continue on error: {args.continue_on_error}")

    if args.dry_run:
        print("\n⚠ DRY RUN - No training will be performed")
        for model in models_to_train:
            if model['name'] == 'yolo':
                cmd_parts = [
                    'train_yolo.py',
                    f'--model yolov8m',
                ]
                if args.epochs:
                    cmd_parts.append(f'--epochs {args.epochs}')
                if args.batch_size:
                    cmd_parts.append(f'--batch-size {args.batch_size}')
                if args.img_size:
                    cmd_parts.append(f'--img-size {args.img_size}')
            else:
                cmd_parts = [
                    'train_improved.py',
                    f'--model {model["name"]}',
                    f'--config {model["config"]}'
                ]
                if args.epochs:
                    cmd_parts.append(f'--epochs {args.epochs}')
                if args.batch_size:
                    cmd_parts.append(f'--batch-size {args.batch_size}')
                # Note: img-size not shown for Faster R-CNN/RetinaNet (must be in config)
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
    print("\n1. Evaluate models:")
    print("   # Faster R-CNN / RetinaNet:")
    print("   python evaluate.py --checkpoint ../checkpoints/faster_rcnn/best_model.pth")
    print("   python evaluate.py --checkpoint ../checkpoints/retinanet/best_model.pth")
    print("   # YOLO:")
    print("   python evaluate_yolo.py --model ../checkpoints/yolo/weights/best.pt")

    print("\n2. Compare results:")
    print("   python compare_models.py  # Compares all trained models")

    print("\n3. Run inference:")
    print("   python inference_improved.py --checkpoint <path> --image <image>")

    print("\n4. Check training logs:")
    for name in successful:
        if name == 'yolo':
            print(f"   ../checkpoints/{name}/results.csv  # YOLO training history")
        else:
            print(f"   ../checkpoints/{name}/training_log_*.txt")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
