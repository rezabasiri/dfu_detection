"""
Improved Training script for DFU detection
Features: Early stopping, validation loss tracking, detailed logging
Supports multiple architectures: Faster R-CNN, RetinaNet, YOLO
"""

import os
import gc
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime
import shutil
import argparse
import yaml

from dataset import DFUDataset, DFUDatasetLMDB, get_train_transforms, get_val_transforms, collate_fn
from train_efficientdet import AverageMeter  # Keep utility class
from balanced_sampler import BalancedBatchSampler
from evaluate import compute_metrics
from models import ModelFactory, create_model

# ============================================================
# PRE-TRAINING MEMORY CLEANUP
# ============================================================
# Clear caches and memory to prevent segfaults from previous runs
# ============================================================

def cleanup_memory():
    """Clear Python, PyTorch, and CUDA caches before training"""
    print("\n" + "="*60)
    print("Pre-Training Memory Cleanup")
    print("="*60)

    # 1. Clear Python garbage collector
    gc.collect()
    print("✓ Python garbage collected")

    # 2. Clear PyTorch shared memory (common source of DataLoader segfaults)
    if os.path.exists('/dev/shm'):
        try:
            # Remove PyTorch shared memory files
            for item in os.listdir('/dev/shm'):
                if 'torch' in item.lower():
                    path = os.path.join('/dev/shm', item)
                    try:
                        if os.path.isfile(path):
                            os.remove(path)
                        elif os.path.isdir(path):
                            shutil.rmtree(path)
                    except:
                        pass
            print("✓ PyTorch shared memory cleared")
        except:
            print("⚠ Could not clear shared memory (may require permissions)")

    # 3. Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✓ CUDA cache cleared")

        # Show GPU memory status
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  GPU Memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")

    # 4. Clear Python import cache
    sys.path_importer_cache.clear()
    print("✓ Python import cache cleared")

    # 5. Force another garbage collection
    gc.collect()

    print("="*60 + "\n")

# Run cleanup immediately on import
cleanup_memory()

def cleanup_epoch():
    """Lightweight cleanup after each epoch to prevent memory accumulation"""
    # Clear Python garbage
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, max_grad_norm=1.0):
    """Train for one epoch with gradient clipping and NaN detection"""
    model.train()
    loss_meter = AverageMeter()
    skipped_batches = 0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")

    for images, targets in pbar:
        # NOTE: Balanced sampler ensures each batch has both DFU and healthy images
        # Empty boxes from healthy images act as hard negatives
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # Check for NaN/Inf in loss before backward pass
            if not torch.isfinite(losses):
                print(f"\nWarning: Non-finite loss detected ({losses.item()}). Skipping batch.")
                continue

            scaler.scale(losses).backward()

            # Unscale gradients before clipping
            scaler.unscale_(optimizer)

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Check for NaN/Inf in loss before backward pass
            if not torch.isfinite(losses):
                print(f"\nWarning: Non-finite loss detected ({losses.item()}). Skipping batch.")
                continue

            losses.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

        loss_meter.update(losses.item(), len(images))
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    return loss_meter.avg

@torch.no_grad()
def validate(model, data_loader, device, compute_detection_metrics=True, confidence_threshold=0.5):
    """
    Validate the model and return loss + detection metrics (F1, IoU)

    Args:
        model: The model to validate
        data_loader: Validation data loader
        device: Device to run on
        compute_detection_metrics: Whether to compute F1/IoU (slower but more informative)
        confidence_threshold: Confidence threshold for detections

    Returns:
        Tuple of (loss, metrics_dict) where metrics_dict contains f1_score and mean_iou
    """
    # First pass: compute loss (requires train mode)
    model.train()
    loss_meter = AverageMeter()

    pbar = tqdm(data_loader, desc="Computing loss")

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_meter.update(losses.item(), len(images))
        pbar.set_postfix({"val_loss": f"{loss_meter.avg:.4f}"})

    val_loss = loss_meter.avg

    # Second pass: compute detection metrics (F1, IoU)
    metrics = {'f1_score': 0.0, 'mean_iou': 0.0, 'precision': 0.0, 'recall': 0.0}

    if compute_detection_metrics:
        model.eval()  # Switch to eval mode for predictions

        all_predictions = []
        all_targets = []

        pbar = tqdm(data_loader, desc="Computing metrics")

        for images, targets in pbar:
            # Process images in smaller sub-batches to reduce memory pressure
            batch_size = len(images)
            sub_batch_size = min(4, batch_size)  # Process max 4 images at a time

            for i in range(0, batch_size, sub_batch_size):
                sub_images = images[i:i+sub_batch_size]
                sub_targets = targets[i:i+sub_batch_size]

                sub_images = [img.to(device) for img in sub_images]

                # Get predictions
                predictions = model(sub_images)

                # Filter predictions by confidence and label
                for pred in predictions:
                    mask = pred['scores'] >= confidence_threshold
                    if 'labels' in pred and len(pred['labels']) > 0:
                        label_mask = pred['labels'] > 0  # Keep only ulcer predictions (not background)
                        mask = mask & label_mask

                    filtered_pred = {
                        'boxes': pred['boxes'][mask].cpu(),
                        'scores': pred['scores'][mask].cpu(),
                        'labels': pred['labels'][mask].cpu()
                    }
                    all_predictions.append(filtered_pred)

                # Filter targets to only include ulcer boxes
                for t in sub_targets:
                    if len(t['labels']) > 0:
                        ulcer_mask = t['labels'] > 0
                        filtered_target = {
                            'boxes': t['boxes'][ulcer_mask].cpu(),
                            'labels': t['labels'][ulcer_mask].cpu()
                        }
                    else:
                        filtered_target = {
                            'boxes': t['boxes'].cpu(),
                            'labels': t['labels'].cpu()
                        }
                    all_targets.append(filtered_target)

                # Clear GPU cache after each sub-batch
                torch.cuda.empty_cache()

        # Compute metrics
        metrics = compute_metrics(all_predictions, all_targets, iou_threshold=0.5)

    return val_loss, metrics

def train_model(
    train_csv,
    val_csv,
    image_folder,
    num_epochs=50,
    batch_size=8,
    learning_rate=0.001,
    img_size=640,
    backbone="efficientnet_b0",
    device="cuda",
    checkpoint_dir="../checkpoints",
    log_file=None,
    early_stopping_patience=10,
    use_amp=True,
    train_image_list=None,
    val_image_list=None,
    healthy_folder=None,
    max_grad_norm=1.0,
    model_name="faster_rcnn",
    config_path=None,
    composite_weights=None
):
    """
    Main training function with early stopping and detailed logging

    Args:
        model_name: Model architecture ('faster_rcnn', 'retinanet', 'yolo')
        config_path: Path to YAML config file (optional)
        composite_weights: Dict with weights for composite score (optional)
            Keys: 'f1', 'iou', 'recall', 'precision'
    """
    # Load configuration from YAML if provided (do this BEFORE creating directories)
    config = {}
    if config_path:
        config = load_config(config_path)

        # Extract training parameters from config (override function defaults)
        if 'training' in config:
            train_config = config['training']
            img_size = train_config.get('img_size', img_size)
            batch_size = train_config.get('batch_size', batch_size)
            num_epochs = train_config.get('num_epochs', num_epochs)
            learning_rate = train_config.get('learning_rate', learning_rate)
            use_amp = train_config.get('use_amp', use_amp)
            max_grad_norm = train_config.get('max_grad_norm', max_grad_norm)
            early_stopping_patience = train_config.get('early_stopping_patience', early_stopping_patience)

            # Extract composite weights from config
            if 'composite_weights' in train_config and composite_weights is None:
                composite_weights = train_config['composite_weights']

        # Extract model config
        if 'model' in config:
            model_config = config['model']
            if 'backbone' in model_config:
                backbone = model_config['backbone']

        # Extract checkpoint directory from config
        if 'checkpoint' in config and 'save_dir' in config['checkpoint']:
            checkpoint_dir = config['checkpoint']['save_dir']

    # Setup logging (after checkpoint_dir is finalized)
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(checkpoint_dir, f"training_log_{timestamp}.txt")

    # Create checkpoint directory AFTER loading config
    os.makedirs(checkpoint_dir, exist_ok=True)

    def log_print(message):
        """Print and log to file"""
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')

    # Log config load
    if config_path:
        log_print(f"Loaded configuration from: {config_path}")

    # Set default composite weights if not provided
    if composite_weights is None:
        composite_weights = {
            'f1': 0.40,
            'iou': 0.25,
            'recall': 0.20,
            'precision': 0.15
        }

    log_print("="*60)
    log_print(f"Training {model_name.upper()} for DFU Detection")
    log_print("="*60)
    log_print(f"\nTraining started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Device setup
    if device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log_print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")
            log_print(f"CUDA Version: {torch.version.cuda}")
            log_print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            log_print("\nWARNING: CUDA requested but not available. Using CPU.")
            device = torch.device("cpu")
            use_amp = False
    else:
        device = torch.device("cpu")
        use_amp = False

    # Load datasets - check for LMDB first, fallback to raw images
    log_print(f"\nLoading datasets...")

    # Check if LMDB paths are specified in config, otherwise use defaults
    if 'data' in config and 'train_lmdb' in config['data']:
        train_lmdb = config['data']['train_lmdb']
        val_lmdb = config['data']['val_lmdb']

        # If paths are relative, resolve them relative to config file directory
        if config_path and not os.path.isabs(train_lmdb):
            config_dir = os.path.dirname(os.path.abspath(config_path))
            train_lmdb = os.path.abspath(os.path.join(config_dir, train_lmdb))
            val_lmdb = os.path.abspath(os.path.join(config_dir, val_lmdb))
            log_print(f"Resolved LMDB paths relative to config:")
            log_print(f"  train_lmdb: {train_lmdb}")
            log_print(f"  val_lmdb: {val_lmdb}")
    else:
        # Default LMDB paths
        data_dir = os.path.dirname(train_csv)
        train_lmdb = os.path.join(data_dir, "train.lmdb")
        val_lmdb = os.path.join(data_dir, "val.lmdb")

    use_lmdb = os.path.exists(train_lmdb) and os.path.exists(val_lmdb)

    if use_lmdb:
        log_print(f"LMDB databases found! Using LMDB for faster loading...")
        train_dataset = DFUDatasetLMDB(
            lmdb_path=train_lmdb,
            transforms=get_train_transforms(img_size),
            mode="train"
        )

        val_dataset = DFUDatasetLMDB(
            lmdb_path=val_lmdb,
            transforms=get_val_transforms(img_size),
            mode="val"
        )
    else:
        log_print(f"LMDB databases not found. Loading from raw images...")
        log_print(f"To create LMDB databases for faster loading, run: python create_lmdb.py")
        train_dataset = DFUDataset(
            csv_file=train_csv,
            image_folder=image_folder,
            transforms=get_train_transforms(img_size),
            mode="train",
            image_list_csv=train_image_list,
            healthy_folder=healthy_folder
        )

        val_dataset = DFUDataset(
            csv_file=val_csv,
            image_folder=image_folder,
            transforms=get_val_transforms(img_size),
            mode="val",
            image_list_csv=val_image_list,
            healthy_folder=healthy_folder
        )

    # Use balanced batch sampler for training to ensure each batch has DFU images
    train_sampler = BalancedBatchSampler(
        train_dataset,
        batch_size=batch_size,
        min_dfu_per_batch=max(2, batch_size // 2)  # At least 50% DFU images per batch
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=4,  # Worker-safe LMDB: each worker gets own connection
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True  # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Worker-safe LMDB: each worker gets own connection
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True  # Keep workers alive between epochs
    )

    log_print(f"\nDataset loaded:")
    log_print(f"  Training samples: {len(train_dataset)}")
    log_print(f"  Validation samples: {len(val_dataset)}")
    log_print(f"  Batch size: {batch_size}")
    log_print(f"  Image size: {img_size}x{img_size}")

    # Create model with 2 classes: background (0), ulcer (1)
    log_print(f"\nCreating {model_name} model...")
    if model_name == 'faster_rcnn':
        log_print(f"Backbone: {backbone}")
    log_print(f"Classes: 0=background, 1=ulcer (2-class detection)")

    # Create model using ModelFactory
    if config_path and 'model' in config:
        # Use model config from YAML
        model_config = config['model']
        model_config['backbone'] = backbone  # Ensure backbone is set
        detector = ModelFactory.create_model(
            model_name=model_name,
            num_classes=2,
            config=model_config
        )
    else:
        # Use default config with specified backbone
        model_config = {'backbone': backbone, 'pretrained': True}
        detector = ModelFactory.create_model(
            model_name=model_name,
            num_classes=2,
            config=model_config
        )

    # Print model information
    detector.print_model_info()

    # Get underlying PyTorch model and move to device
    model = detector.get_model()
    model.to(device)
    detector.to(device)

    # Check for existing checkpoints to resume training
    start_epoch = 1
    resume_checkpoint = None

    # Priority 1: Check for manual resume checkpoint
    resume_path = os.path.join(checkpoint_dir, "resume_training.pth")
    if os.path.exists(resume_path):
        resume_checkpoint = resume_path
        log_print(f"\n{'='*60}")
        log_print(f"Found resume checkpoint: resume_training.pth")

    # Priority 2: Check for best_model.pth
    elif os.path.exists(os.path.join(checkpoint_dir, "best_model.pth")):
        resume_checkpoint = os.path.join(checkpoint_dir, "best_model.pth")
        log_print(f"\n{'='*60}")
        log_print(f"Found existing checkpoint: best_model.pth")

    # Load checkpoint if found
    if resume_checkpoint:
        try:
            log_print(f"Loading checkpoint from: {resume_checkpoint}")
            # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy scalars
            checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)

            # Check if checkpoint model matches requested model
            checkpoint_model_name = checkpoint.get('model_name', 'faster_rcnn')  # Default to faster_rcnn for old checkpoints
            if checkpoint_model_name != model_name:
                log_print(f"⚠ WARNING: Checkpoint model ({checkpoint_model_name}) differs from requested model ({model_name})")
                log_print(f"  Attempting to load anyway... this may fail if architectures are incompatible")

            # Load model weights only (fresh optimizer as requested)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Get the epoch to continue from
            start_epoch = checkpoint.get('epoch', 0) + 1

            # CRITICAL: Restore best scores to prevent overwriting with worse models
            best_composite_from_checkpoint = checkpoint.get('composite_score', 0.0)
            best_f1_from_checkpoint = checkpoint.get('f1_score', 0.0)
            best_iou_from_checkpoint = checkpoint.get('mean_iou', 0.0)
            best_precision_from_checkpoint = checkpoint.get('precision', 0.0)
            best_recall_from_checkpoint = checkpoint.get('recall', 0.0)
            best_val_loss_from_checkpoint = checkpoint.get('val_loss', float('inf'))

            log_print(f"✓ Successfully loaded model weights")
            log_print(f"  Previous epoch: {checkpoint.get('epoch', 'unknown')}")
            log_print(f"  Previous train loss: {checkpoint.get('train_loss', 'unknown'):.4f}" if 'train_loss' in checkpoint else "")
            log_print(f"  Previous val loss: {checkpoint.get('val_loss', 'unknown'):.4f}" if 'val_loss' in checkpoint else "")
            log_print(f"  Previous composite score: {best_composite_from_checkpoint:.4f}")
            log_print(f"  Previous F1: {best_f1_from_checkpoint:.4f}")
            log_print(f"  Previous IoU: {best_iou_from_checkpoint:.4f}")
            log_print(f"  Previous Recall: {best_recall_from_checkpoint:.4f}")
            log_print(f"  Previous Precision: {best_precision_from_checkpoint:.4f}")
            if 'learning_rate' in checkpoint:
                log_print(f"  Learning rate when best model saved: {checkpoint['learning_rate']:.6f}")
            log_print(f"  Resuming from epoch: {start_epoch}")
            log_print(f"  Optimizer: Starting fresh (not restored)")
            log_print(f"{'='*60}")
        except Exception as e:
            log_print(f"✗ Error loading checkpoint: {e}")
            log_print(f"  Starting training from scratch with ImageNet weights")
            start_epoch = 1
    else:
        log_print(f"\nNo checkpoint found. Starting from ImageNet pretrained weights.")
        log_print(f"  To resume training, use 'resume_training.pth' or 'best_model.pth'")


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"Total parameters: {total_params:,}")
    log_print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0001)

    # ReduceLROnPlateau: Reduce LR when validation loss plateaus
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',          # Minimize validation loss
        factor=0.5,          # Reduce LR by 50% when plateau detected
        patience=4,          # Wait 4 epochs before reducing
        min_lr=learning_rate * 0.0001  # Minimum learning rate
    )

    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Training loop
    log_print(f"\nTraining Configuration:")
    log_print(f"  Epochs: {num_epochs}")
    log_print(f"  Learning rate: {learning_rate}")
    log_print(f"  Early stopping patience: {early_stopping_patience}")
    log_print(f"  Mixed precision: {use_amp}")
    log_print(f"  Gradient clipping: {max_grad_norm}")

    log_print(f"\nStarting training...")

    # Initialize best scores (will be overridden if resuming from checkpoint)
    best_val_loss = float('inf')
    best_composite_score = 0.0  # Composite metric for checkpoint saving
    best_f1_score = 0.0
    best_mean_iou = 0.0
    best_precision = 0.0
    best_recall = 0.0
    epochs_without_improvement = 0

    # If resuming, restore best scores from checkpoint
    if resume_checkpoint and 'best_composite_from_checkpoint' in locals():
        best_composite_score = best_composite_from_checkpoint
        best_f1_score = best_f1_from_checkpoint
        best_mean_iou = best_iou_from_checkpoint
        best_precision = best_precision_from_checkpoint
        best_recall = best_recall_from_checkpoint
        best_val_loss = best_val_loss_from_checkpoint
        log_print(f"\n✓ Restored best scores from checkpoint:")
        log_print(f"  Best composite: {best_composite_score:.4f}")
        log_print(f"  Best F1: {best_f1_score:.4f}")
        log_print(f"  Best IoU: {best_mean_iou:.4f}")

    history = {
        "train_loss": [],
        "val_loss": [],
        "f1_score": [],
        "mean_iou": [],
        "precision": [],
        "recall": [],
        "composite_score": [],
        "learning_rate": []
    }

    for epoch in range(start_epoch, num_epochs + 1):
        log_print(f"\n{'='*60}")
        log_print(f"Epoch {epoch}/{num_epochs}")
        log_print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler, max_grad_norm)

        # Validate with metrics (compute F1/IoU every epoch)
        val_loss, val_metrics = validate(model, val_loader, device, compute_detection_metrics=True, confidence_threshold=0.5)

        # Check for NaN values - immediate stop if detected
        if not (torch.isfinite(torch.tensor(train_loss)) and torch.isfinite(torch.tensor(val_loss))):
            log_print(f"\n{'='*60}")
            log_print("ERROR: NaN or Inf detected in loss values!")
            log_print(f"  Train Loss: {train_loss}")
            log_print(f"  Val Loss:   {val_loss}")
            log_print("Training stopped to prevent further corruption.")
            log_print(f"Last good checkpoint may be available at epoch {epoch - 1}")
            log_print(f"{'='*60}")
            break

        # Compute composite score (weighted combination of metrics)
        # Weights configurable via config file or defaults
        composite_score = (
            composite_weights['f1'] * val_metrics.get('f1_score', 0.0) +
            composite_weights['iou'] * val_metrics.get('mean_iou', 0.0) +
            composite_weights['recall'] * val_metrics.get('recall', 0.0) +
            composite_weights['precision'] * val_metrics.get('precision', 0.0)
        )

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["f1_score"].append(val_metrics.get('f1_score', 0.0))
        history["mean_iou"].append(val_metrics.get('mean_iou', 0.0))
        history["precision"].append(val_metrics.get('precision', 0.0))
        history["recall"].append(val_metrics.get('recall', 0.0))
        history["composite_score"].append(composite_score)
        history["learning_rate"].append(optimizer.param_groups[0]['lr'])

        # Track best metrics
        if val_metrics.get('f1_score', 0.0) > best_f1_score:
            best_f1_score = val_metrics['f1_score']
        if val_metrics.get('mean_iou', 0.0) > best_mean_iou:
            best_mean_iou = val_metrics['mean_iou']
        if val_metrics.get('precision', 0.0) > best_precision:
            best_precision = val_metrics['precision']
        if val_metrics.get('recall', 0.0) > best_recall:
            best_recall = val_metrics['recall']

        log_print(f"\nResults:")
        log_print(f"  Train Loss: {train_loss:.4f}")
        log_print(f"  Val Loss:   {val_loss:.4f}")
        log_print(f"  F1 Score:   {val_metrics.get('f1_score', 0.0):.4f} (best: {best_f1_score:.4f})")
        log_print(f"  Mean IoU:   {val_metrics.get('mean_iou', 0.0):.4f} (best: {best_mean_iou:.4f})")
        log_print(f"  Precision:  {val_metrics.get('precision', 0.0):.4f} (best: {best_precision:.4f})")
        log_print(f"  Recall:     {val_metrics.get('recall', 0.0):.4f} (best: {best_recall:.4f})")
        log_print(f"  Composite:  {composite_score:.4f} (best: {best_composite_score:.4f})")

        # Update learning rate based on validation loss (stability signal)
        lr_scheduler.step(val_loss)

        # Save best model based on composite score
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            epochs_without_improvement = 0

            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            current_lr = optimizer.param_groups[0]['lr']
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "f1_score": val_metrics.get('f1_score', 0.0),
                "mean_iou": val_metrics.get('mean_iou', 0.0),
                "precision": val_metrics.get('precision', 0.0),
                "recall": val_metrics.get('recall', 0.0),
                "composite_score": composite_score,
                "composite_weights": composite_weights,  # Save weights used
                "learning_rate": current_lr,  # Save LR at which best model was achieved
                "backbone": backbone,
                "img_size": img_size,
                "num_classes": 2,  # 2-class: background + ulcer
                "model_name": model_name,  # Save model architecture name
                "model_config": model_config  # Save model configuration
            }, checkpoint_path)
            log_print(f"  ✓ New best model! Saved to {checkpoint_path}")
            log_print(f"    Composite Score: {composite_score:.4f}")
            log_print(f"    Learning Rate: {current_lr:.6f}")
            log_print(f"    Breakdown: F1={val_metrics.get('f1_score', 0.0):.4f}, IoU={val_metrics.get('mean_iou', 0.0):.4f}, Recall={val_metrics.get('recall', 0.0):.4f}, Precision={val_metrics.get('precision', 0.0):.4f}")
        else:
            epochs_without_improvement += 1
            log_print(f"  No improvement in composite score for {epochs_without_improvement} epoch(s)")

        # Save periodic checkpoint
        if epoch % 25 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "backbone": backbone,
                "img_size": img_size,
                "num_classes": 2,  # 2-class: background + ulcer
                "model_name": model_name,
                "model_config": model_config
            }, checkpoint_path)
            log_print(f"  Checkpoint saved: epoch_{epoch}.pth")

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            log_print(f"\nEarly stopping triggered after {epoch} epochs")
            log_print(f"No improvement for {early_stopping_patience} consecutive epochs")
            break

        # Cleanup after each epoch to prevent memory accumulation
        cleanup_epoch()

    # Save training history
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    log_print("\n" + "="*60)
    log_print("Training complete!")
    log_print("="*60)
    log_print(f"\nBest Metrics Achieved:")
    log_print(f"  Composite Score: {best_composite_score:.4f} (0.4*F1 + 0.25*IoU + 0.2*Recall + 0.15*Precision)")
    log_print(f"  F1 Score:        {best_f1_score:.4f}")
    log_print(f"  Mean IoU:        {best_mean_iou:.4f}")
    log_print(f"  Recall:          {best_recall:.4f}")
    log_print(f"  Precision:       {best_precision:.4f}")
    log_print(f"  Val Loss:        {best_val_loss:.4f} (used for LR scheduling only)")
    log_print(f"\nBest model saved to: {os.path.join(checkpoint_dir, 'best_model.pth')}")
    log_print(f"Training history saved to: {history_path}")
    log_print(f"Training log saved to: {log_file}")
    log_print(f"\nTraining finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"\nResume Training Instructions:")
    log_print(f"  - To resume from best model: Re-run this script (auto-detects best_model.pth)")
    log_print(f"  - To resume from specific epoch: cp checkpoint_epoch_N.pth resume_training.pth")

    return model, history

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train DFU detection model')
    parser.add_argument('--model', type=str, default='faster_rcnn',
                       choices=['faster_rcnn', 'retinanet', 'yolo'],
                       help='Model architecture to train')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (optional)')
    parser.add_argument('--backbone', type=str, default='efficientnet_b5',
                       help='Backbone architecture (for Faster R-CNN/RetinaNet)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Checkpoint directory (overrides config)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')

    args = parser.parse_args()

    # Data paths
    data_dir = "../data"
    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "val.csv")

    # Check for healthy feet image lists
    train_image_list = os.path.join(data_dir, "train_images.csv")
    val_image_list = os.path.join(data_dir, "val_images.csv")

    # Use healthy feet if available, otherwise None
    train_images = train_image_list if os.path.exists(train_image_list) else None
    val_images = val_image_list if os.path.exists(val_image_list) else None

    data_root = "/mnt/c/Users/90rez/OneDrive - University of Toronto/PhDUofT/SideProjects/DFU_Detection_Asem"
    image_folder = os.path.join(data_root, "DFUC2022_train_images")
    healthy_folder = os.path.join(data_root, "HealthyFeet")

    if not all(os.path.exists(f) for f in [train_csv, val_csv]):
        print("Error: CSV files not found. Please run data_preprocessing.py first.")
        exit(1)

    if train_images:
        print("Using healthy feet images (negative samples) for training!")
        print(f"Healthy feet folder: {healthy_folder}")
    else:
        print("Training with DFU images only (no healthy feet).")
        print("To add healthy feet: run 'python add_healthy_feet.py'")

    # Determine checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    elif args.config:
        # Will be loaded from config
        checkpoint_dir = None
    else:
        # Default based on model name
        checkpoint_dir = f"../checkpoints/{args.model}"

    print("\n" + "="*60)
    print(f"TRAINING CONFIGURATION - {args.model.upper()}")
    print("="*60)
    print("SETUP:")
    print("  1. 2-class detection: 0=background, 1=ulcer")
    print("  2. Healthy images as hard negatives (reduces false positives)")
    print("  3. Balanced batch sampling (50% DFU images/batch for stability)")
    print("  4. ReduceLROnPlateau scheduler (adapts to val loss)")
    print(f"  5. Model: {args.model}")
    if args.config:
        print(f"  6. Config file: {args.config}")
    else:
        print(f"  6. Using default configuration")
    print("\nMODEL SELECTION:")
    print("  - Best model saved based on COMPOSITE SCORE")
    print("  - Composite weights configurable via YAML config")
    print("  - Val loss used only for LR scheduling")
    print("  - Early stopping based on composite score")
    print("="*60 + "\n")

    # Prepare training arguments
    train_args = {
        'train_csv': train_csv,
        'val_csv': val_csv,
        'image_folder': image_folder,
        'backbone': args.backbone,
        'device': args.device,
        'train_image_list': train_images,
        'val_image_list': val_images,
        'healthy_folder': healthy_folder if train_images else None,
        'model_name': args.model,
        'config_path': args.config
    }

    # Add checkpoint directory if specified
    if checkpoint_dir:
        train_args['checkpoint_dir'] = checkpoint_dir

    # Override config with command-line arguments if provided
    if args.epochs is not None:
        train_args['num_epochs'] = args.epochs
    if args.batch_size is not None:
        train_args['batch_size'] = args.batch_size
    if args.lr is not None:
        train_args['learning_rate'] = args.lr

    # Train model
    model, history = train_model(**train_args)

    print("\nTraining finished! Check the log file for details.")