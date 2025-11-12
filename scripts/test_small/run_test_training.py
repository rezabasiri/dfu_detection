#!/usr/bin/env python3
"""
Run Test Training for All Three Models
Trains Faster R-CNN, RetinaNet, and YOLOv8 on small test dataset sequentially.
"""

import subprocess
import sys
import time
import os
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

def train_model(model_config):
    """Train a single model."""
    model_name = model_config['name']
    config_path = model_config['config']
    description = model_config['description']

    print_section(f"Training: {description}")

    # Build command
    cmd = [
        sys.executable,
        'train_improved.py',
        '--model', model_name,
        '--config', config_path,
        '--device', 'cuda' if sys.argv[1:] and sys.argv[1] == '--gpu' else 'cpu'
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
        checkpoint_path = f"../checkpoints_test/{model['name']}/best_model.pth"
        print(f"   - {model['name']:<15} {checkpoint_path}")

    print("\n3. Check training logs:")
    print("   ls -lh ../checkpoints_test/*/training_log_*.txt\n")

def main():
    """Run all test trainings."""
    # Change to scripts directory (parent of test_small)
    scripts_dir = Path(__file__).parent.parent
    os.chdir(scripts_dir)
    print(f"Working directory: {os.getcwd()}")

    print_section("DFU Detection - Test Training Suite")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of models: {len(MODELS)}")

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
    device = 'cuda' if sys.argv[1:] and sys.argv[1] == '--gpu' else 'cpu'
    print(f"Device: {device}")

    if device == 'cuda':
        try:
            import torch
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        except ImportError:
            print("⚠ PyTorch not installed yet")

    # Run training for each model
    results = []
    for i, model in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] Starting {model['name']}...")
        result = train_model(model)
        results.append(result)

        # Wait a bit between trainings
        if i < len(MODELS):
            print("\nWaiting 5 seconds before next training...")
            time.sleep(5)

    # Print summary
    print_summary(results)

    # Final message
    print_section("All Test Trainings Complete!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    main()
