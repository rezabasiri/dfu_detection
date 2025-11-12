# Faster R-CNN Training Guide for DFU Detection

Complete guide for training Faster R-CNN models for Diabetic Foot Ulcer detection.

---

## Table of Contents
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Backbone Options](#backbone-options)
- [Training Parameters](#training-parameters)
- [Usage Examples](#usage-examples)
- [Configuration Files](#configuration-files)
- [Output Structure](#output-structure)
- [Advanced Tuning](#advanced-tuning)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Basic Training (Recommended)
```bash
cd scripts
python train_improved.py \
    --model faster_rcnn \
    --backbone efficientnet_b5 \
    --epochs 300 \
    --batch-size 36 \
    --img-size 512 \
    --device cuda
```

### Test Training (Fast)
```bash
python train_improved.py \
    --model faster_rcnn \
    --config test_small/test_faster_rcnn.yaml \
    --device cuda
```

---

## Model Architecture

Faster R-CNN is a **two-stage object detector**:

1. **Stage 1 - Region Proposal Network (RPN)**
   - Generates candidate object regions
   - Filters background regions
   - Outputs region proposals

2. **Stage 2 - ROI Head**
   - Refines proposals
   - Classifies objects
   - Regresses bounding boxes

### Advantages
- ✅ High accuracy
- ✅ Excellent localization
- ✅ Handles small objects well
- ✅ Robust to variations

### Disadvantages
- ❌ Slower training than single-stage detectors
- ❌ Slower inference than YOLO
- ❌ Higher memory usage

---

## Backbone Options

The backbone extracts features from images. Choose based on your needs:

| Backbone | Parameters | Speed | Accuracy | Memory | Recommended For |
|----------|-----------|-------|----------|--------|-----------------|
| `efficientnet_b0` | 5.3M | ⚡⚡⚡⚡ | ⭐⭐⭐ | 4 GB | Testing, Fast training |
| `efficientnet_b1` | 7.8M | ⚡⚡⚡ | ⭐⭐⭐⭐ | 6 GB | Balanced |
| `efficientnet_b3` | 12M | ⚡⚡ | ⭐⭐⭐⭐ | 8 GB | Good balance |
| `efficientnet_b5` | 30M | ⚡ | ⭐⭐⭐⭐⭐ | 12 GB | **Production (recommended)** |
| `resnet50` | 25.6M | ⚡⚡ | ⭐⭐⭐⭐ | 8 GB | Classic choice |
| `resnet101` | 44.5M | ⚡ | ⭐⭐⭐⭐⭐ | 12 GB | High accuracy |

**Recommendation**: Use `efficientnet_b5` for best accuracy, or `efficientnet_b3` for faster training.

---

## Training Parameters

### Required Parameters
```bash
--model faster_rcnn      # Model type
```

### Common Parameters
```bash
--backbone efficientnet_b5    # Feature extraction backbone
--epochs 300                  # Number of training epochs
--batch-size 36               # Batch size (GPU memory dependent)
--img-size 512                # Input image size
--learning-rate 0.001         # Initial learning rate
--device cuda                 # Device: cuda or cpu
```

### Data Parameters
```bash
--train-csv ../data/train.csv       # Training annotations
--val-csv ../data/val.csv           # Validation annotations
--image-folder /path/to/images      # Image directory
--checkpoint-dir ../checkpoints_b5  # Checkpoint directory
```

### Configuration File
```bash
--config path/to/config.yaml   # Use YAML configuration (recommended)
```

---

## Usage Examples

### 1. Production Training (Recommended)
**Best accuracy with EfficientNet-B5:**
```bash
python train_improved.py \
    --model faster_rcnn \
    --backbone efficientnet_b5 \
    --epochs 300 \
    --batch-size 36 \
    --img-size 512 \
    --learning-rate 0.001 \
    --device cuda \
    --checkpoint-dir ../checkpoints_b5
```

**Expected time**: ~12-16 hours on NVIDIA TITAN Xp
**Expected results**: F1 > 0.80, mAP > 0.75, Composite Score > 0.74

---

### 2. Fast Training (Testing)
**Quick validation with EfficientNet-B0:**
```bash
python train_improved.py \
    --model faster_rcnn \
    --backbone efficientnet_b0 \
    --epochs 10 \
    --batch-size 16 \
    --img-size 256 \
    --device cuda \
    --checkpoint-dir ../checkpoints_test
```

**Expected time**: ~1-2 hours
**Use case**: Test setup, debug, quick iteration

---

### 3. High Accuracy Training
**Maximum accuracy with ResNet101:**
```bash
python train_improved.py \
    --model faster_rcnn \
    --backbone resnet101 \
    --epochs 500 \
    --batch-size 24 \
    --img-size 640 \
    --learning-rate 0.0005 \
    --device cuda \
    --checkpoint-dir ../checkpoints_resnet101
```

**Expected time**: ~20-24 hours
**Expected results**: F1 > 0.82, mAP > 0.78

---

### 4. Balanced Training
**Good speed/accuracy tradeoff with EfficientNet-B3:**
```bash
python train_improved.py \
    --model faster_rcnn \
    --backbone efficientnet_b3 \
    --epochs 300 \
    --batch-size 48 \
    --img-size 512 \
    --learning-rate 0.001 \
    --device cuda \
    --checkpoint-dir ../checkpoints_b3
```

**Expected time**: ~8-10 hours
**Use case**: Faster iteration while maintaining good accuracy

---

### 5. Resume Training
**Continue from checkpoint:**
```bash
python train_improved.py \
    --model faster_rcnn \
    --backbone efficientnet_b5 \
    --epochs 400 \
    --batch-size 36 \
    --img-size 512 \
    --device cuda \
    --checkpoint-dir ../checkpoints_b5
# Automatically resumes from best_model.pth if exists
```

---

### 6. Training with Configuration File
**Use YAML config (recommended for reproducibility):**
```bash
python train_improved.py \
    --model faster_rcnn \
    --config configs/faster_rcnn_b5.yaml \
    --device cuda
```

**Example config file** (`configs/faster_rcnn_b5.yaml`):
```yaml
model:
  backbone: efficientnet_b5
  pretrained: true
  anchor_sizes: [32, 64, 128, 256, 512]
  aspect_ratios: [0.5, 1.0, 2.0]
  box_score_thresh: 0.05
  box_nms_thresh: 0.5

training:
  num_epochs: 300
  batch_size: 36
  img_size: 512
  learning_rate: 0.001
  use_amp: true
  max_grad_norm: 1.0
  early_stopping_patience: 23

data:
  train_lmdb: ../data/train.lmdb
  val_lmdb: ../data/val.lmdb

checkpoint:
  save_dir: ../checkpoints_b5

composite_weights:
  f1: 0.40
  iou: 0.25
  recall: 0.20
  precision: 0.15
```

---

### 7. Training with Custom Data Paths
```bash
python train_improved.py \
    --model faster_rcnn \
    --backbone efficientnet_b5 \
    --train-csv /path/to/train.csv \
    --val-csv /path/to/val.csv \
    --image-folder /path/to/images \
    --epochs 300 \
    --batch-size 36 \
    --device cuda
```

---

### 8. CPU Training (No GPU)
```bash
python train_improved.py \
    --model faster_rcnn \
    --backbone efficientnet_b0 \
    --epochs 50 \
    --batch-size 4 \
    --img-size 256 \
    --device cpu \
    --checkpoint-dir ../checkpoints_cpu
```

**Expected time**: ~48-72 hours
**Note**: Very slow, not recommended

---

## Configuration Files

### Model Configuration

```yaml
model:
  backbone: efficientnet_b5
  pretrained: true  # Use ImageNet pretrained weights

  # Anchor configuration
  anchor_sizes: [32, 64, 128, 256, 512]
  aspect_ratios: [0.5, 1.0, 2.0]

  # RPN settings
  rpn_positive_iou: 0.5   # Lowered for better recall
  rpn_negative_iou: 0.3
  rpn_pre_nms_top_n_train: 2000
  rpn_post_nms_top_n_train: 2000

  # Detection head settings
  box_score_thresh: 0.05  # Low threshold for better recall
  box_nms_thresh: 0.5
```

### Training Configuration

```yaml
training:
  num_epochs: 300
  batch_size: 36
  img_size: 512
  learning_rate: 0.001
  use_amp: true           # Mixed precision training
  max_grad_norm: 1.0      # Gradient clipping
  early_stopping_patience: 23
```

### Composite Score Weights

The best model is saved based on a **composite score**, not validation loss:

```yaml
composite_weights:
  f1: 0.40          # Most important for detection
  iou: 0.25         # Localization quality
  recall: 0.20      # Don't miss ulcers
  precision: 0.15   # Avoid false positives
```

**Formula**:
```
Composite Score = 0.40×F1 + 0.25×IoU + 0.20×Recall + 0.15×Precision
```

---

## Output Structure

### Checkpoint Directory
```
checkpoints_b5/
├── best_model.pth              # Best model (highest composite score)
├── resume_training.pth         # Periodic checkpoint
├── training_log_<timestamp>.txt  # Detailed training log
├── training_history.json       # Metrics history (JSON)
└── plots/                      # Training curves (if enabled)
```

### Checkpoint Contents

```python
checkpoint = {
    'epoch': 156,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,

    # Metrics
    'train_loss': 0.245,
    'val_loss': 0.312,
    'f1_score': 0.7846,
    'mean_iou': 0.5627,
    'precision': 0.7153,
    'recall': 0.8688,
    'composite_score': 0.7356,

    # Model info
    'model_name': 'faster_rcnn',
    'backbone': 'efficientnet_b5',
    'num_classes': 2,
    'img_size': 512,

    # Training settings
    'learning_rate': 0.000250,
    'composite_weights': {...}
}
```

---

## Advanced Tuning

### Anchor Tuning

Adjust anchor sizes based on your ulcer sizes:

```yaml
# Default (works well for DFU)
anchor_sizes: [32, 64, 128, 256, 512]
aspect_ratios: [0.5, 1.0, 2.0]

# For smaller ulcers
anchor_sizes: [16, 32, 64, 128, 256]

# For larger ulcers
anchor_sizes: [64, 128, 256, 512, 1024]
```

### RPN Tuning

```yaml
# More lenient (better recall)
rpn_positive_iou: 0.5
rpn_negative_iou: 0.3

# Stricter (better precision)
rpn_positive_iou: 0.7
rpn_negative_iou: 0.5
```

### Detection Thresholds

```yaml
# Lower threshold (more detections, lower precision)
box_score_thresh: 0.03

# Higher threshold (fewer detections, higher precision)
box_score_thresh: 0.10
```

### Learning Rate Schedule

The training uses **ReduceLROnPlateau**:
- Reduces LR when validation loss plateaus
- Factor: 0.5 (halves the LR)
- Patience: 4 epochs
- Min LR: 0.0001 × initial_lr

```yaml
# More aggressive LR reduction
lr_patience: 2
lr_factor: 0.3

# More conservative
lr_patience: 6
lr_factor: 0.7
```

---

## Performance Benchmarks

### Training Time (300 epochs, 512×512)

| Backbone | GPU | Batch Size | Time |
|----------|-----|-----------|------|
| efficientnet_b0 | TITAN Xp | 48 | ~6h |
| efficientnet_b3 | TITAN Xp | 36 | ~10h |
| efficientnet_b5 | TITAN Xp | 36 | ~16h |
| resnet50 | TITAN Xp | 36 | ~12h |
| resnet101 | TITAN Xp | 24 | ~20h |

### Expected Results (Validation Set)

| Backbone | Composite | F1 | IoU | Recall | Precision |
|----------|-----------|-----|-----|--------|-----------|
| efficientnet_b0 | 0.68 | 0.72 | 0.51 | 0.80 | 0.65 |
| efficientnet_b3 | 0.71 | 0.76 | 0.54 | 0.83 | 0.70 |
| efficientnet_b5 | 0.74 | 0.78 | 0.56 | 0.87 | 0.72 |
| resnet50 | 0.72 | 0.77 | 0.54 | 0.84 | 0.71 |
| resnet101 | 0.76 | 0.80 | 0.58 | 0.88 | 0.74 |

---

## Monitoring Training

### Real-Time Log Monitoring
```bash
# Watch training progress
tail -f ../checkpoints_b5/training_log_*.txt

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Training Log Example
```
============================================================
Epoch 156/300
Learning rate: 0.000250
------------------------------------------------------------
Training Loss: 0.2447
Validation Loss: 0.3124

Detection Metrics:
  F1 Score:      0.7846  ⬆ (Previous: 0.7801)
  Mean IoU:      0.5627  ⬆ (Previous: 0.5589)
  Recall:        0.8688  ⬆ (Previous: 0.8645)
  Precision:     0.7153  ⬆ (Previous: 0.7098)

Composite Score: 0.7356  ⬆ NEW BEST! (Previous: 0.7298)

✓ Saved best model to: ../checkpoints_b5/best_model.pth
============================================================
```

---

## Troubleshooting

### Error: LMDB not found
```bash
# Create LMDB databases
cd scripts
python create_lmdb.py
```

### Error: Out of memory
```bash
# Reduce batch size
--batch-size 16  # or lower

# Use smaller backbone
--backbone efficientnet_b0

# Reduce image size
--img-size 416
```

### Training loss not decreasing
- Check learning rate (try 0.0001 or 0.00001)
- Check if data is loaded correctly
- Verify augmentations aren't too aggressive
- Try training longer (patience!)

### Validation loss increasing (overfitting)
- Reduce model size (use smaller backbone)
- Increase augmentation strength
- Reduce training epochs
- Add dropout/regularization

### Low recall (missing ulcers)
```yaml
# Adjust thresholds
box_score_thresh: 0.03  # Lower threshold
rpn_positive_iou: 0.4   # More lenient proposals
```

### Low precision (false positives)
```yaml
# Adjust thresholds
box_score_thresh: 0.10  # Higher threshold
rpn_positive_iou: 0.7   # Stricter proposals
```

---

## Comparison with Other Models

| Metric | Faster R-CNN | RetinaNet | YOLO |
|--------|--------------|-----------|------|
| Accuracy | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Training Speed | Slow | Medium | ⚡ Fast |
| Inference Speed | Slow (~5 FPS) | Medium (~15 FPS) | ⚡ Fast (~50 FPS) |
| Small Objects | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Memory Usage | High | Medium | Low |
| Tuning Complexity | High | Medium | Low |

**Use Faster R-CNN when:**
- Accuracy is the top priority
- You have good GPU resources
- Inference speed is not critical
- You need best small object detection

---

## Next Steps

### After Training
1. **Evaluate Model**:
   ```bash
   python evaluate.py \
       --checkpoint ../checkpoints_b5/best_model.pth \
       --device cuda
   ```

2. **Test on New Images**:
   ```python
   from models import ModelFactory
   import torch

   # Load model
   model = ModelFactory.create_from_checkpoint(
       '../checkpoints_b5/best_model.pth',
       device=torch.device('cuda')
   )

   # Inference
   predictions = model.forward(images)
   ```

3. **Compare with Other Models**:
   ```bash
   cd test_small
   python compare_test_models.py
   ```

### Further Optimization
- Try different backbones
- Tune anchor configurations
- Experiment with different augmentations
- Use ensemble with other models
- Fine-tune on specific failure cases

---

## Additional Resources

- **PyTorch Faster R-CNN**: https://pytorch.org/vision/stable/models.html#object-detection
- **Faster R-CNN Paper**: https://arxiv.org/abs/1506.01497
- **EfficientNet Paper**: https://arxiv.org/abs/1905.11946

---

**Last Updated**: 2025-11-12
**Maintainer**: DFU Detection Team
