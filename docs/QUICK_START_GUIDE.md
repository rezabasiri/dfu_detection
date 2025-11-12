# Quick Start Guide - DFU Detection Training

## Training Individual Models

### 1. Faster R-CNN (Two-stage, High Accuracy)
```bash
cd scripts
python train_improved.py --model faster_rcnn --config configs/train_config_b5.yaml
```

### 2. RetinaNet (Single-stage, Balanced)
```bash
cd scripts
python train_improved.py --model retinanet --config configs/train_config_retinanet.yaml
```

### 3. YOLO (Single-stage, Fast Inference)
```bash
cd scripts
python train_yolo.py --model yolov8n --epochs 300 --batch-size 36 --img-size 512
```

---

## Test Training (Small Dataset)

### Train All Models
```bash
cd scripts/test_small
python run_test_training.py --gpu
```

### Train Specific Models
```bash
# Train only Faster R-CNN and RetinaNet
python run_test_training.py --models faster_rcnn retinanet --gpu

# Train only YOLO
python run_test_training.py --models yolo --gpu

# Train on CPU (slower)
python run_test_training.py --models yolo
```

---

## Evaluating YOLO Models

YOLO now computes the same metrics as Faster R-CNN/RetinaNet for easy comparison!

### Automatic Evaluation (After Training)
By default, `train_yolo.py` automatically computes comparable metrics after training.

### Manual Evaluation
```bash
cd scripts
python evaluate_yolo.py --model ../checkpoints_test/yolo/weights/best.pt --val-lmdb ../data/val.lmdb
```

### Custom Thresholds
```bash
# Low confidence (training monitoring)
python evaluate_yolo.py --model best.pt --confidence-threshold 0.05

# Medium confidence (validation)
python evaluate_yolo.py --model best.pt --confidence-threshold 0.3

# High confidence (production)
python evaluate_yolo.py --model best.pt --confidence-threshold 0.5
```

---

## Understanding Metrics

All three models now report the same metrics:

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Composite Score** | 0.40×F1 + 0.25×IoU + 0.20×Recall + 0.15×Precision | Overall performance (used for best model selection) |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Balance between precision and recall |
| **Mean IoU** | Average IoU of matched boxes | Localization accuracy |
| **Precision** | TP / (TP + FP) | How many predictions are correct |
| **Recall** | TP / (TP + FN) | How many ground truth boxes are found |

### Example Output (All Models)
```
Composite Score: 0.7356
F1 Score:        0.7846
Mean IoU:        0.5627
Recall:          0.8688
Precision:       0.7153
```

This makes it easy to compare Faster R-CNN, RetinaNet, and YOLO directly!

---

## Confidence Thresholds

Different thresholds for different purposes:

| Purpose | Threshold | When to Use |
|---------|-----------|-------------|
| **Training Monitoring** | 0.05 | See learning progress during training |
| **Validation** | 0.3-0.5 | Compare different models fairly |
| **Production** | 0.5-0.7 | Only show high-confidence predictions |

**Why?** Early in training, models produce low-confidence predictions (0.05-0.3). Using threshold=0.5 would filter everything out and show 0.0000 metrics, making you think the model isn't learning. But it is! You just need a lower threshold to see it.

---

## Understanding Training Logs

### YOLO Training Log
```
train: Scanning ... 80 images, 40 backgrounds, 0 corrupt
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        1/2     0.178G      2.633       3.84      1.353          9        128: 100% ━━━━━━━━━━━━ 10/10
val: Scanning ... 20 images, 10 backgrounds, 0 corrupt
```

**Translation:**
- **Training set**: 80 images total (40 with ulcers, 40 healthy)
- **10/10 batches**: 10 batches × 8 images = 80 images (all processed ✓)
- **Validation set**: 20 images (shown separately)

All 80 training images were used! The "20 images" is validation, not training.

### Faster R-CNN/RetinaNet Training Log
```
Epoch 1/300
  Train Loss: 0.4532
  Val Loss:   0.3821
  F1:         0.7846
  IoU:        0.5627
  Composite:  0.7356
```

---

## Common Issues

### Issue: Metrics showing 0.0000
**Cause**: Confidence threshold too high (0.5) for early training
**Solution**: Use `--confidence-threshold 0.05` for training monitoring

### Issue: "Only 20 images processed"
**Explanation**: That's the validation set. Training used all 80 images (10 batches × 8)

### Issue: YOLO metrics different from Faster R-CNN
**Solution**: Use `evaluate_yolo.py` to get comparable metrics (F1, IoU, Composite)

---

## Model Selection Guide

| Model | Speed | Accuracy | GPU Memory | Best For |
|-------|-------|----------|------------|----------|
| **Faster R-CNN** | Slow | Highest | High | Research, high accuracy needed |
| **RetinaNet** | Medium | High | Medium | Balanced speed/accuracy |
| **YOLO** | Fast | Good | Low | Real-time, mobile, edge devices |

### Quick Decision Tree
- Need highest accuracy? → **Faster R-CNN**
- Need balanced performance? → **RetinaNet**
- Need fast inference? → **YOLO**
- Need mobile deployment? → **YOLOv8n** (nano)

---

## Further Reading

- [Faster R-CNN Training Guide](FASTER_RCNN_TRAINING_GUIDE.md) - 8 usage examples
- [RetinaNet Training Guide](RETINANET_TRAINING_GUIDE.md) - 4 usage examples
- [YOLO Training Guide](YOLO_TRAINING_GUIDE.md) - 7 usage examples
- [Training Overview](TRAINING_OVERVIEW.md) - Complete comparison

---

**Last Updated**: 2025-11-12
