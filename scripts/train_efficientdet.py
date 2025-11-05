"""
Training script for DFU detection using EfficientDet  
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

from dataset import DFUDataset, get_train_transforms, get_val_transforms, collate_fn

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2,
    efficientnet_b3, efficientnet_b4, efficientnet_b5,
    efficientnet_b6, efficientnet_b7
)

def create_efficientdet_model(num_classes=2, backbone="efficientnet_b0", pretrained=True):
    """
    Create EfficientDet model with configurable EfficientNet backbone

    Supported backbones and their specs:
    - efficientnet_b0: 1280 channels, ~5M params, 224x224 input
    - efficientnet_b1: 1280 channels, ~8M params, 240x240 input
    - efficientnet_b2: 1408 channels, ~9M params, 260x260 input
    - efficientnet_b3: 1536 channels, ~12M params, 300x300 input
    - efficientnet_b4: 1792 channels, ~19M params, 380x380 input (needs ~12GB VRAM)
    - efficientnet_b5: 2048 channels, ~30M params, 456x456 input (needs ~16GB VRAM)
    - efficientnet_b6: 2304 channels, ~43M params, 528x528 input (needs ~24GB VRAM)
    - efficientnet_b7: 2560 channels, ~66M params, 600x600 input (needs ~32GB VRAM)
    """

    backbone_configs = {
        "efficientnet_b0": (efficientnet_b0, 1280),
        "efficientnet_b1": (efficientnet_b1, 1280),
        "efficientnet_b2": (efficientnet_b2, 1408),
        "efficientnet_b3": (efficientnet_b3, 1536),
        "efficientnet_b4": (efficientnet_b4, 1792),
        "efficientnet_b5": (efficientnet_b5, 2048),
        "efficientnet_b6": (efficientnet_b6, 2304),
        "efficientnet_b7": (efficientnet_b7, 2560),
    }

    if backbone not in backbone_configs:
        raise ValueError(f"Unsupported backbone: {backbone}. Supported: {list(backbone_configs.keys())}")

    efficientnet_fn, out_channels = backbone_configs[backbone]
    efficientnet = efficientnet_fn(weights="IMAGENET1K_V1" if pretrained else None)

    backbone_model = nn.Sequential(*list(efficientnet.children())[:-2])
    backbone_model.out_channels = out_channels

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone=backbone_model,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None):
    model.train()
    loss_meter = AverageMeter()
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
        
        loss_meter.update(losses.item(), len(images))
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
    
    return loss_meter.avg

@torch.no_grad()
def validate(model, data_loader, device):
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(data_loader, desc="Validating")
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        
        predictions = model(images)
        
        all_predictions.extend([{k: v.cpu() for k, v in p.items()} for p in predictions])
        all_targets.extend([{k: v.cpu() for k, v in t.items()} for t in targets])
    
    total_gt_boxes = sum(len(t["boxes"]) for t in all_targets)
    total_pred_boxes = sum(len(p["boxes"]) for p in all_predictions)
    
    return {
        "total_gt_boxes": total_gt_boxes,
        "total_pred_boxes": total_pred_boxes
    }

def train_model(
    train_csv,
    val_csv,
    image_folder,
    num_epochs=50,
    batch_size=4,
    learning_rate=0.001,
    img_size=640,
    backbone="efficientnet_b0",
    device="cuda",
    checkpoint_dir="../checkpoints",
    use_amp=True
):
    print("="*60)
    print("Training EfficientDet for DFU Detection")
    print("="*60)
    
    if device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("\nWARNING: CUDA requested but not available. Using CPU.")
            device = torch.device("cpu")
            use_amp = False
    else:
        device = torch.device("cpu")
        use_amp = False
    
    print(f"\nLoading datasets...")
    train_dataset = DFUDataset(
        csv_file=train_csv,
        image_folder=image_folder,
        transforms=get_train_transforms(img_size),
        mode="train"
    )
    
    val_dataset = DFUDataset(
        csv_file=val_csv,
        image_folder=image_folder,
        transforms=get_val_transforms(img_size),
        mode="val"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"\nDataset loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}")
    
    print(f"\nCreating model with {backbone} backbone...")
    model = create_efficientdet_model(num_classes=2, backbone=backbone, pretrained=True)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0001)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=learning_rate * 0.01
    )
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    best_loss = float("inf")
    
    history = {
        "train_loss": [],
        "val_metrics": []
    }
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler)
        history["train_loss"].append(train_loss)
        
        val_metrics = validate(model, val_loader, device)
        history["val_metrics"].append(val_metrics)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val GT boxes: {val_metrics['total_gt_boxes']}, Pred boxes: {val_metrics['total_pred_boxes']}")
        
        lr_scheduler.step()
        
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
                "backbone": backbone
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
                "backbone": backbone
            }, checkpoint_path)
    
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_path, "w") as f:
        history_serializable = {
            "train_loss": history["train_loss"],
            "val_metrics": [{k: int(v) if isinstance(v, (np.integer, torch.Tensor)) else v 
                           for k, v in m.items()} for m in history["val_metrics"]]
        }
        json.dump(history_serializable, f, indent=2)
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"Best model saved to: {os.path.join(checkpoint_dir, 'best_model.pth')}")
    print(f"Training history saved to: {history_path}")
    
    return model, history

if __name__ == "__main__":
    data_dir = "../data"
    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "val.csv")
    
    image_folder = "/mnt/c/Users/90rez/OneDrive - University of Toronto/PhDUofT/SideProjects/DFU_Detection_Asem/DFUC2022_train_images"
    
    if not all(os.path.exists(f) for f in [train_csv, val_csv]):
        print("Error: CSV files not found. Please run data_preprocessing.py first.")
        exit(1)
    
    model, history = train_model(
        train_csv=train_csv,
        val_csv=val_csv,
        image_folder=image_folder,
        num_epochs=50,
        batch_size=8,
        learning_rate=0.001,
        img_size=640,
        backbone="efficientnet_b0",
        device="cuda",
        use_amp=True
    )
    
    print("\nTraining finished!")