# RetinaNet Training Guide for DFU Detection

Complete guide for training RetinaNet models for Diabetic Foot Ulcer detection.

---

## Quick Start

```bash
cd scripts
python train_improved.py \
    --model retinanet \
    --backbone efficientnet_b3 \
    --epochs 300 \
    --batch-size 48 \
    --img-size 512 \
    --device cuda
```

---

## What is RetinaNet?

RetinaNet is a **single-stage object detector** with **Focal Loss**:

### Key Features
- **Single-stage**: Direct prediction (faster than Faster R-CNN)
- **Focal Loss**: Addresses class imbalance
- **Feature Pyramid Network (FPN)**: Multi-scale detection
- **Anchor-based**: Uses predefined anchor boxes

### Advantages
- ✅ Faster than Faster R-CNN
- ✅ Good accuracy (comparable to two-stage)
- ✅ Handles class imbalance well
- ✅ Efficient training

### Disadvantages
- ❌ Slower than YOLO
- ❌ More complex than YOLO
- ❌ Requires careful tuning

---

## Backbone Options

| Backbone | Parameters | Speed | Accuracy | Memory | Recommended For |
|----------|-----------|-------|----------|--------|-----------------|
| `efficientnet_b0` | 5.3M | ⚡⚡⚡⚡ | ⭐⭐⭐ | 3 GB | Testing |
| `efficientnet_b1` | 7.8M | ⚡⚡⚡ | ⭐⭐⭐⭐ | 4 GB | Fast training |
| `efficientnet_b3` | 12M | ⚡⚡ | ⭐⭐⭐⭐⭐ | 6 GB | **Recommended** |
| `resnet50` | 25.6M | ⚡⚡ | ⭐⭐⭐⭐ | 6 GB | Classic choice |

---

## Usage Examples

### 1. Production Training (Recommended)
```bash
python train_improved.py \
    --model retinanet \
    --backbone efficientnet_b3 \
    --epochs 300 \
    --batch-size 48 \
    --img-size 512 \
    --learning-rate 0.001 \
    --device cuda \
    --checkpoint-dir ../checkpoints_retinanet_b3
```

**Expected time**: ~8-10 hours
**Expected results**: F1 > 0.76, Composite > 0.71

---

### 2. Fast Training
```bash
python train_improved.py \
    --model retinanet \
    --backbone efficientnet_b0 \
    --epochs 50 \
    --batch-size 24 \
    --img-size 256 \
    --device cuda
```

**Expected time**: ~2-3 hours

---

### 3. High Accuracy
```bash
python train_improved.py \
    --model retinanet \
    --backbone resnet50 \
    --epochs 400 \
    --batch-size 36 \
    --img-size 640 \
    --device cuda
```

**Expected time**: ~12-14 hours
**Expected results**: F1 > 0.78, Composite > 0.73

---

### 4. Configuration File (Recommended)
```bash
python train_improved.py \
    --model retinanet \
    --config configs/retinanet_b3.yaml \
    --device cuda
```

**Example config**:
```yaml
model:
  backbone: efficientnet_b3
  pretrained: true
  anchor_sizes: [32, 64, 128, 256, 512]
  aspect_ratios: [0.5, 1.0, 2.0]
  focal_loss_alpha: 0.25
  focal_loss_gamma: 2.0
  score_thresh: 0.05
  nms_thresh: 0.5

training:
  num_epochs: 300
  batch_size: 48
  img_size: 512
  learning_rate: 0.001
  use_amp: true
```

---

## RetinaNet-Specific Parameters

### Focal Loss Parameters

**Alpha (α)**: Weight for positive samples
```yaml
focal_loss_alpha: 0.25  # Default (balanced)
focal_loss_alpha: 0.30  # More weight on positives
focal_loss_alpha: 0.20  # Less weight on positives
```

**Gamma (γ)**: Focusing parameter
```yaml
focal_loss_gamma: 2.0   # Default (standard)
focal_loss_gamma: 3.0   # More focus on hard examples
focal_loss_gamma: 1.5   # Less aggressive
```

**Focal Loss Formula**:
```
FL(pt) = -α(1-pt)^γ log(pt)
```

### Detection Thresholds

```yaml
score_thresh: 0.05     # Detection confidence threshold
nms_thresh: 0.5        # Non-maximum suppression threshold
detections_per_img: 100  # Max detections per image
```

---

## Performance Benchmarks

### Training Time (300 epochs)

| Backbone | Batch Size | Time (TITAN Xp) |
|----------|-----------|-----------------|
| efficientnet_b0 | 64 | ~5h |
| efficientnet_b3 | 48 | ~10h |
| resnet50 | 36 | ~12h |

### Expected Results

| Backbone | Composite | F1 | IoU | Recall | Precision |
|----------|-----------|-----|-----|--------|-----------|
| efficientnet_b0 | 0.66 | 0.71 | 0.50 | 0.78 | 0.65 |
| efficientnet_b3 | 0.71 | 0.76 | 0.54 | 0.82 | 0.71 |
| resnet50 | 0.73 | 0.78 | 0.56 | 0.84 | 0.73 |

---

## Comparison: RetinaNet vs Others

| Feature | Faster R-CNN | RetinaNet | YOLO |
|---------|--------------|-----------|------|
| **Architecture** | Two-stage | Single-stage | Single-stage |
| **Speed** | Slow | ⚡ Medium | ⚡⚡ Fast |
| **Accuracy** | Very High | High | High |
| **Training Time** | Long | Medium | Short |
| **Memory** | High | Medium | Low |
| **Complexity** | High | Medium | Low |

### When to Use RetinaNet
- ✅ Need better speed than Faster R-CNN
- ✅ Need better accuracy than YOLO
- ✅ Class imbalance is an issue
- ✅ Want single-stage simplicity with good accuracy

### RetinaNet Sweet Spot
**Best balance between speed and accuracy** - faster than Faster R-CNN, more accurate than YOLO.

---

## Advanced Tuning

### Focal Loss Tuning

**For class imbalance:**
```yaml
# More imbalance (few ulcers, many background)
focal_loss_alpha: 0.35
focal_loss_gamma: 2.5

# Less imbalance
focal_loss_alpha: 0.25
focal_loss_gamma: 2.0
```

### Anchor Tuning

```yaml
# Default
anchor_sizes: [32, 64, 128, 256, 512]
aspect_ratios: [0.5, 1.0, 2.0]

# For smaller ulcers
anchor_sizes: [16, 32, 64, 128, 256]

# More aspect ratios
aspect_ratios: [0.5, 0.75, 1.0, 1.5, 2.0]
```

### Detection Threshold Tuning

```yaml
# Higher recall (detect more, some false positives)
score_thresh: 0.03

# Higher precision (fewer false positives)
score_thresh: 0.10

# Stricter NMS (fewer overlapping boxes)
nms_thresh: 0.3

# Looser NMS (allow more overlaps)
nms_thresh: 0.7
```

---

## Troubleshooting

### Loss not decreasing
- **Reduce learning rate**: Try 0.0001 or 0.00001
- **Adjust focal loss gamma**: Try γ=1.5 or γ=2.5
- **Check data loading**: Verify images and labels match
- **Increase warmup**: More gradual training start

### Low recall (missing ulcers)
```yaml
# Lower detection threshold
score_thresh: 0.03

# Adjust focal loss (favor recall)
focal_loss_alpha: 0.30
focal_loss_gamma: 1.5

# More aggressive NMS
nms_thresh: 0.6
```

### Low precision (false positives)
```yaml
# Higher detection threshold
score_thresh: 0.08

# Adjust focal loss (favor precision)
focal_loss_alpha: 0.20
focal_loss_gamma: 2.5

# Stricter NMS
nms_thresh: 0.4
```

### Out of memory
```bash
# Reduce batch size
--batch-size 24

# Use smaller backbone
--backbone efficientnet_b0

# Reduce image size
--img-size 416
```

---

## Monitoring Training

### Watch Training Progress
```bash
tail -f ../checkpoints_retinanet_b3/training_log_*.txt
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Training Log Example
```
============================================================
Epoch 145/300
Learning rate: 0.000125
------------------------------------------------------------
Training Loss: 0.2891
Validation Loss: 0.3456

Detection Metrics:
  F1 Score:      0.7612
  Mean IoU:      0.5389
  Recall:        0.8234
  Precision:     0.7089

Composite Score: 0.7145  ⬆ NEW BEST!
============================================================
```

---

## Next Steps

### After Training

1. **Evaluate Model:**
```bash
python evaluate.py --checkpoint ../checkpoints_retinanet_b3/best_model.pth
```

2. **Compare with Other Models:**
```bash
cd test_small
python compare_test_models.py
```

3. **Inference:**
```python
from models import ModelFactory
model = ModelFactory.create_from_checkpoint('path/to/best_model.pth')
predictions = model.forward(images)
```

---

## Key Takeaways

✅ **Faster** than Faster R-CNN, **more accurate** than YOLO
✅ Focal Loss handles class imbalance naturally
✅ Good for production when speed matters but accuracy is critical
✅ Easier to tune than Faster R-CNN
✅ Recommended backbone: **EfficientNet-B3**

---

## Additional Resources

- **RetinaNet Paper**: https://arxiv.org/abs/1708.02002
- **Focal Loss Paper**: https://arxiv.org/abs/1708.02002
- **PyTorch Implementation**: https://pytorch.org/vision/stable/models.html

---

**Last Updated**: 2025-11-12
**Maintainer**: DFU Detection Team
