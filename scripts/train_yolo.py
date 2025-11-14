"""
Dedicated YOLO Training Script for DFU Detection
Uses YOLO's native training interface for optimal performance and compatibility
"""

import os
import sys
import yaml
import argparse
import torch
from pathlib import Path
from datetime import datetime
import lmdb
import pickle
import cv2
import numpy as np
from tqdm import tqdm

# Import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("ERROR: ultralytics package not found. Install with: pip install ultralytics")
    sys.exit(1)


def export_lmdb_to_yolo_format(lmdb_path, output_dir, split_name='train'):
    """
    Export LMDB dataset to YOLO format (images + labels)

    YOLO format:
    - Images in: {output_dir}/images/{split_name}/
    - Labels in: {output_dir}/labels/{split_name}/
    - Each label file: one line per box with format: class_id x_center y_center width height (normalized)

    Args:
        lmdb_path: Path to LMDB database
        output_dir: Output directory for YOLO dataset
        split_name: 'train' or 'val'
    """
    print(f"\nExporting {split_name} LMDB to YOLO format...")
    print(f"  LMDB: {lmdb_path}")
    print(f"  Output: {output_dir}")

    # Create directories
    images_dir = Path(output_dir) / 'images' / split_name
    labels_dir = Path(output_dir) / 'labels' / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Open LMDB
    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False
    )

    with env.begin(write=False) as txn:
        # Get length
        length = int(txn.get(b'__len__').decode('ascii'))

        print(f"  Total samples: {length}")

        # Export each sample
        for idx in tqdm(range(length), desc=f"Exporting {split_name}"):
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

            h, w = image.shape[:2]

            # Get boxes and labels
            boxes = sample['boxes']  # (N, 4) [xmin, ymin, xmax, ymax]
            labels = sample['labels']  # (N,)

            # Save image
            image_filename = f"{idx:08d}.jpg"
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), image)

            # Convert boxes to YOLO format and save labels
            label_filename = f"{idx:08d}.txt"
            label_path = labels_dir / label_filename

            with open(label_path, 'w') as f:
                for box, label in zip(boxes, labels):
                    # Skip background class (label 0)
                    if label == 0:
                        continue

                    # Convert from [xmin, ymin, xmax, ymax] to [x_center, y_center, width, height]
                    xmin, ymin, xmax, ymax = box
                    x_center = ((xmin + xmax) / 2) / w
                    y_center = ((ymin + ymax) / 2) / h
                    width = (xmax - xmin) / w
                    height = (ymax - ymin) / h

                    # YOLO uses 0-indexed classes (no background)
                    yolo_class = int(label) - 1

                    # Write: class x_center y_center width height (normalized 0-1)
                    f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    env.close()
    print(f"✓ Exported {length} samples to {output_dir}")

    return length


def create_yolo_data_yaml(train_dir, val_dir, output_path, num_classes=1, class_names=None):
    """
    Create YOLO data.yaml configuration file

    Args:
        train_dir: Path to training images directory
        val_dir: Path to validation images directory
        output_path: Path to save data.yaml
        num_classes: Number of classes (excluding background)
        class_names: List of class names
    """
    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]

    # YOLO expects absolute paths or paths relative to data.yaml
    train_dir = Path(train_dir).resolve()
    val_dir = Path(val_dir).resolve()

    data_config = {
        'path': str(Path(train_dir).parent.parent),  # Root directory
        'train': str(train_dir.relative_to(Path(train_dir).parent.parent)),
        'val': str(val_dir.relative_to(Path(train_dir).parent.parent)),
        'nc': num_classes,
        'names': class_names
    }

    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    print(f"\nCreated YOLO data config: {output_path}")
    print(f"  Classes: {num_classes}")
    print(f"  Names: {class_names}")

    return output_path


def train_yolo(
    model_size='yolov8n',
    data_yaml='dfu_data.yaml',
    epochs=300,
    batch_size=36,
    img_size=512,
    device='cuda',
    project='runs/yolo',
    name='dfu_detection',
    pretrained=True,
    save_period=10,
    box_loss=7.5,
    cls_loss=0.5,
    dfl_loss=1.5,
    **kwargs
):
    """
    Train YOLO model using native training interface

    Args:
        model_size: YOLO model variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        data_yaml: Path to data.yaml configuration file
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        device: Device to use ('cuda' or 'cpu')
        project: Project directory for results
        name: Experiment name
        pretrained: Use COCO pretrained weights
        save_period: Save checkpoint every N epochs (best.pt and last.pt always saved)
        box_loss: Box regression loss weight (default: 7.5)
        cls_loss: Classification loss weight (default: 0.5)
        dfl_loss: Distribution Focal Loss weight (default: 1.5)
        **kwargs: Additional YOLO training arguments
    """

    print("\n" + "="*60)
    print("YOLO Training for DFU Detection")
    print("="*60)
    print(f"\nModel: {model_size}")
    print(f"Data config: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Device: {device}")
    print(f"Pretrained: {pretrained}")
    print(f"\nLoss Weights:")
    print(f"  Box loss: {box_loss}")
    print(f"  Cls loss: {cls_loss}")
    print(f"  DFL loss: {dfl_loss}")
    print("="*60 + "\n")

    # Initialize YOLO model
    if pretrained:
        model = YOLO(f"{model_size}.pt")
        print(f"✓ Loaded pretrained {model_size} model")
    else:
        model = YOLO(f"{model_size}.yaml")
        print(f"✓ Created {model_size} model from scratch")

    # Train the model
    # YOLO automatically handles:
    # - Class number adjustment (from 80 COCO classes to your custom classes)
    # - Loss computation
    # - Optimizer and scheduler
    # - Data augmentation
    # - Checkpointing
    # - Tensorboard logging
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=pretrained,
        optimizer='AdamW',  # Match your Faster R-CNN training
        lr0=0.001,  # Initial learning rate (matches your config)
        lrf=0.01,  # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=box_loss,  # Box loss weight (from config or args)
        cls=cls_loss,  # Classification loss weight (from config or args)
        dfl=dfl_loss,  # DFL loss weight (from config or args)
        save=True,  # Save checkpoints
        save_period=save_period,  # Save checkpoint every N epochs
        val=True,  # Validate during training
        plots=True,  # Generate plots
        verbose=True,
        **kwargs
    )

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Results saved to: {project}/{name}")
    print(f"Best model: {project}/{name}/weights/best.pt")
    print("="*60 + "\n")

    return model, results


def main():
    parser = argparse.ArgumentParser(description='Train YOLO for DFU Detection')

    # Data arguments
    parser.add_argument('--train-lmdb', type=str, default='../data/train.lmdb',
                        help='Path to training LMDB database')
    parser.add_argument('--val-lmdb', type=str, default='../data/val.lmdb',
                        help='Path to validation LMDB database')
    parser.add_argument('--yolo-data-dir', type=str, default='../data/yolo_format',
                        help='Directory for YOLO format data (will be created)')
    parser.add_argument('--export-only', action='store_true',
                        help='Only export data to YOLO format, do not train')

    # Model arguments
    parser.add_argument('--model', type=str, default='yolov8n',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                        help='YOLO model variant')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use COCO pretrained weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='Train from scratch')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=36,
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    # Output arguments
    parser.add_argument('--project', type=str, default='../runs/yolo',
                        help='Project directory for results')
    parser.add_argument('--name', type=str, default='dfu_detection',
                        help='Experiment name')
    parser.add_argument('--save-period', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10). Note: best.pt and last.pt are always saved.')

    # Loss weight arguments
    parser.add_argument('--box-loss', type=float, default=7.5,
                        help='Box regression loss weight (default: 7.5)')
    parser.add_argument('--cls-loss', type=float, default=0.5,
                        help='Classification loss weight (default: 0.5)')
    parser.add_argument('--dfl-loss', type=float, default=1.5,
                        help='Distribution Focal Loss weight (default: 1.5)')

    # Evaluation arguments
    parser.add_argument('--compute-metrics', action='store_true', default=True,
                        help='Compute comparable metrics after training (F1, IoU, etc.)')
    parser.add_argument('--no-metrics', dest='compute_metrics', action='store_false',
                        help='Skip metric computation after training')

    args = parser.parse_args()

    # Check if LMDB files exist
    if not os.path.exists(args.train_lmdb):
        print(f"ERROR: Training LMDB not found: {args.train_lmdb}")
        print("Please run create_lmdb.py first to create LMDB databases")
        sys.exit(1)

    if not os.path.exists(args.val_lmdb):
        print(f"ERROR: Validation LMDB not found: {args.val_lmdb}")
        print("Please run create_lmdb.py first to create LMDB databases")
        sys.exit(1)

    # Export LMDB to YOLO format (one-time setup)
    yolo_data_dir = Path(args.yolo_data_dir)
    data_yaml_path = yolo_data_dir / 'data.yaml'

    # Check if already exported
    if not data_yaml_path.exists():
        print("\n" + "="*60)
        print("First-time setup: Exporting LMDB to YOLO format")
        print("="*60)

        # Export training data
        train_count = export_lmdb_to_yolo_format(
            args.train_lmdb,
            yolo_data_dir,
            split_name='train'
        )

        # Export validation data
        val_count = export_lmdb_to_yolo_format(
            args.val_lmdb,
            yolo_data_dir,
            split_name='val'
        )

        # Create data.yaml
        create_yolo_data_yaml(
            train_dir=yolo_data_dir / 'images' / 'train',
            val_dir=yolo_data_dir / 'images' / 'val',
            output_path=data_yaml_path,
            num_classes=1,  # DFU ulcer (background is implicit)
            class_names=['ulcer']
        )

        print(f"\n✓ YOLO dataset created successfully!")
        print(f"  Train samples: {train_count}")
        print(f"  Val samples: {val_count}")
        print(f"  Data config: {data_yaml_path}")
    else:
        print(f"\n✓ Using existing YOLO dataset: {yolo_data_dir}")
        print(f"  Data config: {data_yaml_path}")

    # Exit if only exporting
    if args.export_only:
        print("\nExport complete. Exiting (--export-only flag set)")
        return

    # Train YOLO model
    model, results = train_yolo(
        model_size=args.model,
        data_yaml=str(data_yaml_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=args.pretrained,
        save_period=args.save_period,  # Pass save_period to YOLO
        box_loss=args.box_loss,        # Pass loss weights
        cls_loss=args.cls_loss,
        dfl_loss=args.dfl_loss
    )

    print("\n✓ Training complete! Results saved to:")
    print(f"  {args.project}/{args.name}")

    # Compute comparable metrics (F1, IoU, Composite Score)
    if args.compute_metrics:
        print("\n" + "="*60)
        print("Computing Comparable Metrics (F1, IoU, Precision, Recall)")
        print("="*60)

        # Import evaluation function
        try:
            from evaluate_yolo import evaluate_yolo_on_lmdb, print_metrics

            # Evaluate best model
            best_model_path = Path(args.project) / args.name / 'weights' / 'best.pt'

            if best_model_path.exists():
                print(f"\nEvaluating best model: {best_model_path}\n")

                metrics, _, _ = evaluate_yolo_on_lmdb(
                    model_path=str(best_model_path),
                    lmdb_path=args.val_lmdb,
                    confidence_threshold=0.05,  # Use low threshold for training monitoring
                    iou_threshold=0.5,
                    device=args.device
                )

                print_metrics(metrics)

                print("\nNote: These metrics are directly comparable with Faster R-CNN")
                print("and RetinaNet results from train_improved.py\n")
            else:
                print(f"\nWarning: Best model not found at {best_model_path}")
                print("Skipping metric computation\n")

        except ImportError:
            print("\nWarning: Could not import evaluate_yolo.py")
            print("Skipping metric computation\n")


if __name__ == '__main__':
    main()
