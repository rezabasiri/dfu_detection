#!/usr/bin/env python3
"""
Run Test Training for All Three Models
Trains Faster R-CNN, RetinaNet, and YOLOv8 on small test dataset sequentially.

Note: YOLO uses train_yolo.py (native interface), others use train_improved.py

Usage:
    # Train all models
    python run_test_training.py

    # Train specific models
    python run_test_training.py --models faster_rcnn retinanet
    python run_test_training.py --models yolo --gpu

    # Train on GPU
    python run_test_training.py --gpu
"""

import subprocess
import sys
import time
import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Model configurations
MODELS = [
    {
        'name': 'faster_rcnn',
        'config': 'test_small/test_faster_rcnn.yaml',
        'description': 'Faster R-CNN with EfficientNet-B0'
    },
    {
        'name': 'retinanet',
        'config': 'test_small/test_retinanet.yaml',
        'description': 'RetinaNet with EfficientNet-B0 (single-stage, focal loss)'
    },
    {
        'name': 'yolo',
        'config': 'test_small/test_yolov8.yaml',
        'description': 'YOLOv8-nano (fastest inference, anchor-free)'
    }
]

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

# Global device variable (set in main)
DEVICE = 'cpu'

def train_model(model_config):
    """Train a single model."""
    model_name = model_config['name']
    config_path = model_config['config']
    description = model_config['description']

    print_section(f"Training: {description}")

    # Build command - YOLO uses dedicated script, others use unified script
    if model_name == 'yolo':
        # Load YOLO config to get training parameters
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                yolo_config = yaml.safe_load(f)

            # Extract parameters from YAML
            epochs = yolo_config.get('training', {}).get('num_epochs', 2)
            batch_size = yolo_config.get('training', {}).get('batch_size', 8)
            img_size = yolo_config.get('training', {}).get('img_size', 128)
            model_size = yolo_config.get('model', {}).get('model_size', 'yolov8n')
            save_period = yolo_config.get('checkpoint', {}).get('save_every_n_epochs', 1)
        else:
            # Fallback defaults if config not found
            epochs = 2
            batch_size = 8
            img_size = 128
            model_size = 'yolov8n'
            save_period = 1

        # YOLO uses train_yolo.py with native interface
        # Note: YOLO saves checkpoints:
        #   - best.pt: when validation mAP improves (auto)
        #   - last.pt: every epoch (auto)
        #   - epochN.pt: every save_period epochs
        cmd = [
            sys.executable,
            'train_yolo.py',
            '--model', model_size,
            '--train-lmdb', '../data/test_train.lmdb',
            '--val-lmdb', '../data/test_val.lmdb',
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
            '--img-size', str(img_size),
            '--device', DEVICE,
            '--project', '../checkpoints_test',
            '--name', 'yolo',
            '--save-period', str(save_period)
        ]
    else:
        # Faster R-CNN and RetinaNet use train_improved.py
        cmd = [
            sys.executable,
            'train_improved.py',
            '--model', model_name,
            '--config', config_path,
            '--device', DEVICE
        ]

    print(f"Command: {' '.join(cmd)}\n")

    # Record start time
    start_time = time.time()

    try:
        # Run training
        result = subprocess.run(cmd, check=True)

        # Record end time
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        print(f"\n✓ {model_name} training completed successfully!")
        print(f"  Time taken: {minutes}m {seconds}s")

        return {
            'name': model_name,
            'success': True,
            'time': elapsed,
            'error': None
        }

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {model_name} training failed!")
        print(f"  Error code: {e.returncode}")

        return {
            'name': model_name,
            'success': False,
            'time': elapsed,
            'error': str(e)
        }

def print_summary(results):
    """Print training summary."""
    print_section("Training Summary")

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    if successful:
        print(f"✓ Successfully trained ({len(successful)}):")
        for r in successful:
            minutes = int(r['time'] // 60)
            seconds = int(r['time'] % 60)
            print(f"  - {r['name']:<15} Time: {minutes}m {seconds}s")

    if failed:
        print(f"\n✗ Failed ({len(failed)}):")
        for r in failed:
            print(f"  - {r['name']}")

    # Print next steps
    print("\n" + "=" * 70)
    print("  Next Steps")
    print("=" * 70)
    print("\n1. Compare models:")
    print("   python compare_test_models.py\n")

    print("2. View individual model results:")
    for model in MODELS:
        if model['name'] == 'yolo':
            checkpoint_path = f"../checkpoints_test/yolo/weights/best.pt"
        else:
            checkpoint_path = f"../checkpoints_test/{model['name']}/best_model.pth"
        print(f"   - {model['name']:<15} {checkpoint_path}")

    print("\n3. Check training logs:")
    print("   ls -lh ../checkpoints_test/*/training_log_*.txt\n")

def main():
    """Run all test trainings."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Run test training for DFU detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models on CPU
  python run_test_training.py

  # Train all models on GPU
  python run_test_training.py --gpu

  # Train specific models
  python run_test_training.py --models faster_rcnn retinanet
  python run_test_training.py --models yolo --gpu

  # Train only YOLO
  python run_test_training.py --models yolo
        """
    )
    parser.add_argument('--models', nargs='+',
                        choices=['faster_rcnn', 'retinanet', 'yolo', 'all'],
                        default=['all'],
                        help='Models to train (default: all)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training')

    args = parser.parse_args()

    # Filter models based on selection
    if 'all' in args.models:
        models_to_train = MODELS
    else:
        models_to_train = [m for m in MODELS if m['name'] in args.models]

    if not models_to_train:
        print("ERROR: No valid models selected!")
        print(f"Available models: {[m['name'] for m in MODELS]}")
        return

    # Change to scripts directory (parent of test_small)
    scripts_dir = Path(__file__).parent.parent
    os.chdir(scripts_dir)
    print(f"Working directory: {os.getcwd()}")

    print_section("DFU Detection - Test Training Suite")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models to train: {[m['name'] for m in models_to_train]}")
    print(f"Number of models: {len(models_to_train)}")

    # Check if test dataset exists
    data_dir = Path(scripts_dir).parent / "data"
    test_train_lmdb = data_dir / "test_train.lmdb"
    test_val_lmdb = data_dir / "test_val.lmdb"

    if not test_train_lmdb.exists() or not test_val_lmdb.exists():
        print("\n⚠ WARNING: Test dataset not found!")
        print("\nPlease create the test dataset first:")
        print("  python create_test_dataset.py\n")
        return

    # Print device info
    device = 'cuda' if args.gpu else 'cpu'
    print(f"Device: {device}")

    if device == 'cuda':
        try:
            import torch
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        except ImportError:
            print("⚠ PyTorch not installed yet")

    # Store device for train_model function
    global DEVICE
    DEVICE = device

    # Run training for each model
    results = []
    for i, model in enumerate(models_to_train, 1):
        print(f"\n[{i}/{len(models_to_train)}] Starting {model['name']}...")
        result = train_model(model)
        results.append(result)

        # Wait a bit between trainings
        if i < len(models_to_train):
            print("\nWaiting 5 seconds before next training...")
            time.sleep(5)

    # Print summary
    print_summary(results)

    # Final message
    print_section("All Test Trainings Complete!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    main()
