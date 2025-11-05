# EfficientNet Backbone Selection Guide

Guide to choosing the right EfficientNet backbone for your training setup.

## Available Backbones

All EfficientNet variants (B0-B7) are now supported!

| Backbone | Parameters | Output Channels | Recommended Input Size | VRAM Required | Speed | Accuracy |
|----------|-----------|----------------|----------------------|---------------|-------|----------|
| **efficientnet_b0** | ~5M | 1280 | 224x224 | ~4-6 GB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ |
| **efficientnet_b1** | ~8M | 1280 | 240x240 | ~5-7 GB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ |
| **efficientnet_b2** | ~9M | 1408 | 260x260 | ~6-8 GB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ |
| **efficientnet_b3** | ~12M | 1536 | 300x300 | ~8-10 GB | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| **efficientnet_b4** | ~19M | 1792 | 380x380 | ~12-14 GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ |
| **efficientnet_b5** | ~30M | 2048 | 456x456 | ~16-20 GB | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| **efficientnet_b6** | ~43M | 2304 | 528x528 | ~24-28 GB | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| **efficientnet_b7** | ~66M | 2560 | 600x600 | ~32-40 GB | ⚡ | ⭐⭐⭐⭐⭐ |

*VRAM estimates include model + data + gradients with batch_size=4*

## Recommendations by GPU

### Local Machine: NVIDIA Titan XP (12GB VRAM)

**Recommended**: B0, B1, B2, B3

```python
# Conservative (safe for long training)
backbone="efficientnet_b0"
batch_size=8
img_size=512

# Balanced (good accuracy/speed tradeoff)
backbone="efficientnet_b3"
batch_size=4
img_size=640

# Maximum (uses most VRAM)
backbone="efficientnet_b4"
batch_size=2
img_size=512
```

### Vast.ai: RTX PRO 6000 WS (48GB VRAM)

**Recommended**: B4, B5, B6, B7

```python
# Good balance (RECOMMENDED for your cluster)
backbone="efficientnet_b4"
batch_size=16
img_size=640

# High accuracy
backbone="efficientnet_b5"
batch_size=8
img_size=640

# Maximum accuracy (slow but best results)
backbone="efficientnet_b6"
batch_size=4
img_size=640

# Extreme (research purposes)
backbone="efficientnet_b7"
batch_size=2
img_size=640
```

## Configuration Examples

### For Your Titan XP (12GB) - Local Testing

```python
# train_improved.py - Fast testing
model, history = train_model(
    train_csv=train_csv,
    val_csv=val_csv,
    image_folder=image_folder,
    num_epochs=20,
    batch_size=8,
    learning_rate=0.001,
    img_size=512,
    backbone="efficientnet_b0",  # Fast for testing
    device="cuda",
    use_amp=True
)

# train_improved.py - Production training
model, history = train_model(
    train_csv=train_csv,
    val_csv=val_csv,
    image_folder=image_folder,
    num_epochs=100,
    batch_size=4,
    learning_rate=0.001,
    img_size=640,
    backbone="efficientnet_b3",  # Good accuracy
    device="cuda",
    use_amp=True
)
```

### For Vast.ai RTX PRO 6000 WS (48GB) - Cluster Training

```python
# RECOMMENDED: Best accuracy for medical imaging
model, history = train_model(
    train_csv=train_csv,
    val_csv=val_csv,
    image_folder=image_folder,
    num_epochs=100,
    batch_size=16,              # Large batch! You have 48GB
    learning_rate=0.001,
    img_size=640,
    backbone="efficientnet_b4",  # Sweet spot for 48GB
    device="cuda",
    use_amp=True,
    early_stopping_patience=15
)

# For maximum accuracy (slower)
model, history = train_model(
    train_csv=train_csv,
    val_csv=val_csv,
    image_folder=image_folder,
    num_epochs=150,
    batch_size=8,
    learning_rate=0.0005,        # Lower LR for larger model
    img_size=640,
    backbone="efficientnet_b6",  # Very high accuracy
    device="cuda",
    use_amp=True,
    early_stopping_patience=20
)
```

## Performance Comparison

Estimated training time per epoch (4812 images with LMDB):

| Backbone | Titan XP (12GB) | RTX PRO 6000 (48GB) |
|----------|-----------------|---------------------|
| B0 | ~2 min | ~1 min |
| B1 | ~2.5 min | ~1.2 min |
| B2 | ~3 min | ~1.5 min |
| B3 | ~4 min | ~2 min |
| B4 | ~6 min* | ~3 min |
| B5 | N/A** | ~5 min |
| B6 | N/A** | ~7 min |
| B7 | N/A** | ~10 min |

*Requires batch_size=2, may be unstable
**Cannot fit in 12GB VRAM

## Accuracy vs Speed Tradeoff

### When to use each:

**B0/B1** - Quick experiments, testing code changes
- ✅ Fast iteration
- ✅ Low VRAM
- ❌ Lower accuracy

**B2/B3** - Production use on limited hardware
- ✅ Good accuracy
- ✅ Reasonable speed
- ✅ Fits on most GPUs

**B4/B5** - Production use on powerful hardware (YOUR CLUSTER!)
- ✅ High accuracy
- ✅ Better feature extraction
- ❌ Slower training
- ❌ Needs 12-20GB VRAM

**B6/B7** - Research, competitions, maximum accuracy
- ✅ State-of-the-art accuracy
- ✅ Best for medical imaging
- ❌ Very slow
- ❌ Needs 24-40GB VRAM

## My Recommendation for Your Setup

### Local Machine (Titan XP 12GB):
```python
backbone="efficientnet_b3"
batch_size=4
img_size=640
```
Good balance of accuracy and VRAM usage.

### Vast.ai Cluster (RTX PRO 6000 48GB):
```python
backbone="efficientnet_b4"  # or B5 for max accuracy
batch_size=16               # Take advantage of 48GB!
img_size=640
```
This uses your cluster's power effectively!

## Testing Different Backbones

You can easily test different backbones:

```bash
# Test B0 (fast baseline)
python train_improved.py --backbone efficientnet_b0 --epochs 10

# Test B3 (good for Titan XP)
python train_improved.py --backbone efficientnet_b3 --epochs 50

# Test B4 (good for cluster)
python train_improved.py --backbone efficientnet_b4 --batch-size 16 --epochs 100
```

*(Note: You'll need to add argparse support if you want command-line arguments)*

## Adjusting Batch Size for Larger Models

Rule of thumb for batch size:
- **B0-B2**: batch_size = 8-16 (12GB GPU) or 16-32 (48GB GPU)
- **B3**: batch_size = 4-8 (12GB GPU) or 12-24 (48GB GPU)
- **B4**: batch_size = 2-4 (12GB GPU) or 8-16 (48GB GPU)
- **B5**: batch_size = N/A (12GB GPU) or 4-8 (48GB GPU)
- **B6**: batch_size = N/A (12GB GPU) or 2-4 (48GB GPU)
- **B7**: batch_size = N/A (12GB GPU) or 2 (48GB GPU)

If you get "CUDA out of memory":
1. Reduce batch_size by half
2. Reduce img_size (640 → 512)
3. Use a smaller backbone

## Summary

**For your specific case**:
- **Local testing**: Use B3 (proven to work on your Titan XP)
- **Cluster training**: Use B4 or B5 (make use of that 48GB VRAM!)
- **Maximum accuracy**: Use B6 on cluster (medical imaging benefits from larger models)

The larger models (B4+) are specifically valuable for medical imaging because they can capture finer details in ulcer textures and boundaries.
