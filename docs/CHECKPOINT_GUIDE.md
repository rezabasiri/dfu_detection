# Checkpoint Saving Guide - DFU Detection Models

## Overview

All three models (Faster R-CNN, RetinaNet, YOLO) save checkpoints to `dfu_detection/checkpoints_test/` during test training:

```
checkpoints_test/
├── faster_rcnn/
│   ├── best_model.pth
│   ├── resume_training.pth
│   └── training_log_*.txt
├── retinanet/
│   ├── best_model.pth
│   ├── resume_training.pth
│   └── training_log_*.txt
└── yolo/
    ├── weights/
    │   ├── best.pt
    │   ├── last.pt
    │   ├── epoch1.pt
    │   ├── epoch2.pt
    │   └── ...
    ├── results.csv
    ├── results.png
    └── confusion_matrix.png
```

---

## Checkpoint Saving Behavior

### Faster R-CNN & RetinaNet

#### Checkpoint Files:
1. **`best_model.pth`** - Best model checkpoint
2. **`resume_training.pth`** - Latest epoch checkpoint (for resuming)

#### Saving Criteria:

**best_model.pth** is saved when **Composite Score improves**:
```
Composite Score = 0.40 × F1 + 0.25 × IoU + 0.20 × Recall + 0.15 × Precision
```

**Why Composite Score?**
- **Not just validation loss** - Loss measures prediction cost, not clinical value
- **F1 Score (40%)** - Most important: balance precision/recall for detection
- **IoU (25%)** - Ensures good bounding box localization
- **Recall (20%)** - Weighted higher (don't miss ulcers!)
- **Precision (15%)** - Prevents false alarms

#### Saving Frequency:
- **Every epoch**: `resume_training.pth` updated
- **When score improves**: `best_model.pth` updated

#### Configuration (YAML):
```yaml
checkpoint:
  save_dir: ../../checkpoints_test/faster_rcnn
  save_every_n_epochs: 1  # Save resume_training.pth every epoch
  keep_best_only: false   # Keep all checkpoints
```

---

### YOLO (YOLOv8)

#### Checkpoint Files:
1. **`best.pt`** - Best model by mAP
2. **`last.pt`** - Most recent epoch (auto-updated)
3. **`epochN.pt`** - Periodic checkpoints (every N epochs)

#### Saving Criteria:

**best.pt** is saved when **Fitness Score improves**:
```
Fitness = 0.1 × mAP@0.5 + 0.9 × mAP@0.5:0.95
```

This is YOLO's default metric (COCO evaluation standard).

**Why Fitness Score?**
- **mAP@0.5:0.95 (90%)** - Primary metric: average precision across IoU thresholds
- **mAP@0.5 (10%)** - Secondary metric: loose IoU threshold
- **Standard benchmark** - Allows comparison with other YOLO papers/implementations

#### Saving Frequency:
- **Every epoch**: `last.pt` updated automatically
- **When mAP improves**: `best.pt` updated automatically
- **Every N epochs**: `epochN.pt` saved (configurable)

#### Configuration (YAML):
```yaml
checkpoint:
  save_dir: ../../checkpoints_test/yolo
  save_every_n_epochs: 1  # Save epoch1.pt, epoch2.pt, etc.
  keep_best_only: false

# YOLO saves three types:
# 1. best.pt - Best model by mAP50-95 (auto)
# 2. last.pt - Most recent epoch (auto, every epoch)
# 3. epochN.pt - Periodic checkpoints (every save_every_n_epochs)
```

#### Command-Line Control:
```bash
# Save checkpoint every epoch (test training default)
python train_yolo.py --save-period 1

# Save checkpoint every 10 epochs (production default)
python train_yolo.py --save-period 10

# Save checkpoint every 5 epochs
python train_yolo.py --save-period 5
```

---

## Comparison Table

| Model | Best Checkpoint Criteria | Save Frequency | Checkpoint Files |
|-------|-------------------------|----------------|------------------|
| **Faster R-CNN** | Composite Score<br>(0.40×F1 + 0.25×IoU + 0.20×Recall + 0.15×Precision) | Every epoch | `best_model.pth`<br>`resume_training.pth` |
| **RetinaNet** | Composite Score<br>(same as above) | Every epoch | `best_model.pth`<br>`resume_training.pth` |
| **YOLO** | Fitness Score<br>(0.1×mAP@0.5 + 0.9×mAP@0.5:0.95) | Configurable<br>(default: every 10 epochs) | `best.pt`<br>`last.pt`<br>`epochN.pt` |

---

## Key Differences

### 1. Best Model Selection

**Faster R-CNN/RetinaNet**: Use **Composite Score** (custom for medical detection)
- Optimized for clinical use case
- Balances precision and recall
- Prioritizes not missing ulcers (high recall weight)

**YOLO**: Uses **mAP@0.5:0.95** (COCO standard)
- Standard computer vision metric
- Allows comparison with research papers
- Stricter localization requirements (IoU 0.5-0.95)

### 2. Checkpoint Frequency

**Faster R-CNN/RetinaNet**: Save **every epoch**
- More frequent checkpoints
- Easier to debug training
- Can resume from any epoch

**YOLO**: Save **every N epochs** (configurable)
- Less frequent by default (every 10 epochs)
- Saves disk space
- `last.pt` still saved every epoch for resuming

### 3. File Format

**Faster R-CNN/RetinaNet**: PyTorch `.pth` files
- Contains: model weights, optimizer state, epoch number, metrics

**YOLO**: Ultralytics `.pt` files
- `best.pt` and `epochN.pt`: model weights only
- `last.pt`: model + optimizer + training state (for resuming)

---

## Resume Training

### Faster R-CNN / RetinaNet
```bash
# Automatically resumes from resume_training.pth if found
python train_improved.py --model faster_rcnn --config configs/train_config_b5.yaml
```

### YOLO
```bash
# Resume from last.pt (most recent epoch)
python train_yolo.py --resume runs/yolo/dfu_detection/weights/last.pt

# Or specify in YOLO native way
yolo train resume model=runs/yolo/dfu_detection/weights/last.pt
```

---

## Which Checkpoint to Use?

### For Deployment (Production)
- **Faster R-CNN/RetinaNet**: Use `best_model.pth` (highest Composite Score)
- **YOLO**: Use `best.pt` (highest mAP)

### For Resuming Training
- **Faster R-CNN/RetinaNet**: Use `resume_training.pth` (contains optimizer state)
- **YOLO**: Use `last.pt` (contains optimizer state)

### For Debugging/Analysis
- **Faster R-CNN/RetinaNet**: Use any `resume_training.pth` (one per run)
- **YOLO**: Use specific `epochN.pt` to check performance at epoch N

---

## Adjusting Save Frequency

### Test Training (Quick Iterations)
Set `save_every_n_epochs: 1` in YAML:
```yaml
# test_yolov8.yaml
checkpoint:
  save_every_n_epochs: 1  # Save every epoch for debugging
```

### Production Training (Long Runs)
Set `save_every_n_epochs: 10` in YAML:
```yaml
# train_yolov8.yaml
checkpoint:
  save_every_n_epochs: 10  # Save every 10 epochs to save disk space
```

### Command-Line Override
```bash
# Override YAML setting
python train_yolo.py --save-period 5  # Save every 5 epochs
```

---

## Disk Space Considerations

### Small Test Dataset (100 images)
- **YOLO checkpoint size**: ~6 MB per checkpoint
- **Save every epoch**: OK (60 MB for 10 epochs)

### Full Dataset (5000+ images)
- **YOLO checkpoint size**: ~6 MB (same, model size doesn't change)
- **Save every epoch**: 1.8 GB for 300 epochs
- **Save every 10 epochs**: 180 MB for 300 epochs ✓ Better

**Recommendation**:
- Test training: `save_every_n_epochs: 1` (see every step)
- Production training: `save_every_n_epochs: 10` (save space)

---

## Checkpoint Contents

### Faster R-CNN / RetinaNet `.pth` file
```python
checkpoint = {
    'epoch': 17,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'lr_scheduler_state_dict': {...},
    'best_composite_score': 0.7356,
    'best_f1': 0.7846,
    'best_iou': 0.5627,
    'best_precision': 0.7153,
    'best_recall': 0.8688,
    'learning_rate': 0.00025
}
```

### YOLO `.pt` file (last.pt)
```python
checkpoint = {
    'model': model_state_dict,
    'optimizer': optimizer_state_dict,
    'epoch': 17,
    'best_fitness': 0.3245,
    'date': '2025-11-12',
    ...
}
```

### YOLO `.pt` file (best.pt, epochN.pt)
```python
checkpoint = {
    'model': model_state_dict,  # Weights only
    # No optimizer state (lighter file)
}
```

---

## Common Issues

### Issue: "Too many checkpoint files"
**Solution**: Increase `save_every_n_epochs` in YAML:
```yaml
checkpoint:
  save_every_n_epochs: 10  # Save less frequently
```

### Issue: "Want to keep only best checkpoint"
**Faster R-CNN/RetinaNet**: Set `keep_best_only: true` in YAML
```yaml
checkpoint:
  keep_best_only: true  # Only keep best_model.pth
```

**YOLO**: Use `--save-period -1` to disable periodic saves:
```bash
python train_yolo.py --save-period -1  # Only save best.pt and last.pt
```

### Issue: "Lost best checkpoint after crash"
- **Faster R-CNN/RetinaNet**: `best_model.pth` preserved until better score
- **YOLO**: `best.pt` preserved until better mAP

Both are safe! They only get overwritten when a **better** model is found.

---

## Best Practices

### ✓ DO:
- Use `best_model.pth` / `best.pt` for deployment
- Use `resume_training.pth` / `last.pt` for resuming
- Set `save_every_n_epochs: 1` for debugging short runs
- Set `save_every_n_epochs: 10` for long production runs

### ✗ DON'T:
- Don't use `last.pt` for deployment (may not be best)
- Don't use `best.pt` for resuming (no optimizer state)
- Don't save every epoch for 300-epoch runs (wastes space)
- Don't delete `resume_training.pth` / `last.pt` during training

---

## Summary

**All models save to `checkpoints_test/` ✓**

**Checkpoint saving is automatic:**
- Faster R-CNN/RetinaNet: Based on **Composite Score** (medical-optimized)
- YOLO: Based on **mAP@0.5:0.95** (COCO standard)

**Save frequency is configurable:**
- Test training: Every epoch (see all progress)
- Production training: Every 10 epochs (save space)

**Three checkpoint types:**
1. **Best checkpoint** - Highest score (use for deployment)
2. **Last checkpoint** - Most recent (use for resuming)
3. **Periodic checkpoints** - Every N epochs (use for analysis)

---

**Last Updated**: 2025-11-12
