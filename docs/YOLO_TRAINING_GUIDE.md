# YOLO Training Guide for DFU Detection

Complete guide for training YOLOv8 models for Diabetic Foot Ulcer detection.

---

## Table of Contents
- [Quick Start](#quick-start)
- [Model Variants](#model-variants)
- [Training Parameters](#training-parameters)
- [Usage Examples](#usage-examples)
- [Data Format](#data-format)
- [Output Structure](#output-structure)
- [Advanced Options](#advanced-options)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Basic Training (Recommended)
```bash
cd scripts
python train_yolo.py \
    --model yolov8m \
    --epochs 300 \
    --batch-size 36 \
    --img-size 512
```

### Test Training (Fast)
```bash
python train_yolo.py \
    --model yolov8n \
    --epochs 2 \
    --batch-size 8 \
    --img-size 128
```

---

## Model Variants

YOLO offers 5 variants with different speed/accuracy tradeoffs:

| Model | Parameters | Speed | Accuracy | Recommended For |
|-------|-----------|-------|----------|-----------------|
| `yolov8n` | 3.2M | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Testing, Fast inference |
| `yolov8s` | 11.2M | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Mobile deployment |
| `yolov8m` | 25.9M | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | **Production (best balance)** |
| `yolov8l` | 43.7M | ⚡⚡ | ⭐⭐⭐⭐⭐⭐ | High accuracy needs |
| `yolov8x` | 68.2M | ⚡ | ⭐⭐⭐⭐⭐⭐⭐ | Maximum accuracy |

**Recommendation**: Start with `yolov8m` for the best balance of speed and accuracy.

---

## Training Parameters

### Required Parameters
```bash
--model yolov8m          # Model variant (n/s/m/l/x)
```

### Common Parameters
```bash
--epochs 300             # Number of training epochs
--batch-size 36          # Batch size (adjust based on GPU memory)
--img-size 512           # Input image size (default: 512)
--device cuda            # Device: cuda or cpu (default: cuda)
--pretrained             # Use COCO pretrained weights (default: True)
```

### Data Parameters
```bash
--train-lmdb ../data/train.lmdb    # Training LMDB database
--val-lmdb ../data/val.lmdb        # Validation LMDB database
--yolo-data-dir ../data/yolo_format  # YOLO format data directory
```

### Output Parameters
```bash
--project ../runs/yolo            # Project directory
--name dfu_detection              # Experiment name
```

---

## Usage Examples

### 1. Production Training (Recommended)
**Best for final model with good balance:**
```bash
python train_yolo.py \
    --model yolov8m \
    --epochs 300 \
    --batch-size 36 \
    --img-size 512 \
    --device cuda \
    --pretrained \
    --project ../runs/yolo \
    --name dfu_production
```

**Expected time**: ~4-6 hours on NVIDIA TITAN Xp
**Expected results**: mAP@0.5 > 0.75, F1 > 0.78

---

### 2. Fast Training (For Testing)
**Quick validation of setup:**
```bash
python train_yolo.py \
    --model yolov8n \
    --epochs 10 \
    --batch-size 16 \
    --img-size 256 \
    --device cuda \
    --project ../runs/yolo \
    --name dfu_test
```

**Expected time**: ~10-15 minutes
**Use case**: Verify data, test setup, debug

---

### 3. High Accuracy Training
**Maximum accuracy for research:**
```bash
python train_yolo.py \
    --model yolov8x \
    --epochs 500 \
    --batch-size 24 \
    --img-size 640 \
    --device cuda \
    --pretrained \
    --project ../runs/yolo \
    --name dfu_high_accuracy
```

**Expected time**: ~12-16 hours
**Expected results**: mAP@0.5 > 0.80, F1 > 0.82

---

### 4. Mobile/Edge Deployment
**Optimized for fast inference:**
```bash
python train_yolo.py \
    --model yolov8s \
    --epochs 300 \
    --batch-size 48 \
    --img-size 416 \
    --device cuda \
    --pretrained \
    --project ../runs/yolo \
    --name dfu_mobile
```

**Expected time**: ~2-3 hours
**Use case**: Mobile apps, edge devices
**Inference speed**: ~50-100 FPS on GPU, ~10-20 FPS on CPU

---

### 5. CPU Training (No GPU)
**For machines without GPU:**
```bash
python train_yolo.py \
    --model yolov8n \
    --epochs 50 \
    --batch-size 8 \
    --img-size 256 \
    --device cpu \
    --project ../runs/yolo \
    --name dfu_cpu
```

**Expected time**: ~6-8 hours
**Note**: Training on CPU is slow but works

---

### 6. Resume Training
**Continue from checkpoint:**
```bash
python train_yolo.py \
    --model yolov8m \
    --resume ../runs/yolo/dfu_production/weights/last.pt \
    --epochs 400 \
    --batch-size 36 \
    --img-size 512
```

---

### 7. Custom Data Paths
**Use custom LMDB locations:**
```bash
python train_yolo.py \
    --model yolov8m \
    --train-lmdb /path/to/train.lmdb \
    --val-lmdb /path/to/val.lmdb \
    --yolo-data-dir /path/to/yolo_format \
    --epochs 300 \
    --batch-size 36 \
    --img-size 512
```

---

## Data Format

### First-Time Setup (Automatic)
The script automatically converts LMDB to YOLO format on first run:

```
data/yolo_format/
├── images/
│   ├── train/
│   │   ├── 00000000.jpg
│   │   ├── 00000001.jpg
│   │   └── ...
│   └── val/
│       ├── 00000000.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── 00000000.txt
│   │   ├── 00000001.txt
│   │   └── ...
│   └── val/
│       ├── 00000000.txt
│       └── ...
└── data.yaml
```

### Label Format
Each `.txt` file contains one line per bounding box:
```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized to [0, 1].

**Example:**
```
0 0.512 0.384 0.256 0.192
0 0.723 0.617 0.184 0.145
```

---

## Output Structure

### Training Results
```
runs/yolo/dfu_production/
├── weights/
│   ├── best.pt           # Best model (highest mAP)
│   ├── last.pt           # Latest checkpoint
│   ├── epoch10.pt        # Epoch checkpoints (every 10 epochs)
│   └── epoch20.pt
├── results.csv           # Training metrics
├── results.png           # Training curves
├── confusion_matrix.png  # Confusion matrix
├── F1_curve.png          # F1 score curve
├── PR_curve.png          # Precision-Recall curve
├── P_curve.png           # Precision curve
├── R_curve.png           # Recall curve
└── args.yaml             # Training arguments

```

### Key Files

**`weights/best.pt`** - Use this for inference and evaluation
**`results.csv`** - Detailed training metrics per epoch
**`results.png`** - Training/validation curves

---

## Advanced Options

### Batch Size Selection
Choose based on GPU memory:

| GPU Memory | Recommended Batch Size |
|-----------|----------------------|
| 12 GB | 24-36 (yolov8m) |
| 8 GB | 16-24 (yolov8m) |
| 6 GB | 8-16 (yolov8s) |
| 4 GB | 4-8 (yolov8n) |

### Image Size Selection
- **128-256**: Fast training, lower accuracy
- **416-512**: Good balance (recommended)
- **640-1024**: Maximum accuracy, slower training

### Learning Rate
YOLO automatically adjusts learning rate. Default: 0.001

### Export Only (No Training)
```bash
python train_yolo.py --export-only
```

This only exports LMDB to YOLO format without training.

---

## Performance Benchmarks

### Expected Training Time (300 epochs)
| Model | GPU | Time |
|-------|-----|------|
| yolov8n | TITAN Xp | ~2h |
| yolov8s | TITAN Xp | ~3h |
| yolov8m | TITAN Xp | ~5h |
| yolov8l | TITAN Xp | ~8h |
| yolov8x | TITAN Xp | ~12h |

### Expected Results (Validation Set)
| Model | mAP@0.5 | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| yolov8n | 0.72 | 0.74 | 0.76 | 0.72 |
| yolov8s | 0.75 | 0.77 | 0.79 | 0.75 |
| yolov8m | 0.78 | 0.80 | 0.82 | 0.78 |
| yolov8l | 0.80 | 0.82 | 0.84 | 0.80 |
| yolov8x | 0.82 | 0.84 | 0.86 | 0.82 |

---

## Monitoring Training

### Real-Time Monitoring
```bash
# Watch training progress
tail -f ../runs/yolo/dfu_production/results.csv

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### TensorBoard
```bash
tensorboard --logdir ../runs/yolo
```

Open browser: http://localhost:6006

---

## Troubleshooting

### Error: LMDB not found
```bash
# Create LMDB databases first
cd scripts
python create_lmdb.py
```

### Error: Out of memory
```bash
# Reduce batch size
--batch-size 16  # or lower

# Or reduce image size
--img-size 416  # or lower
```

### Error: CUDA out of memory during export
```bash
# Export runs on GPU, reduce batch processing
# Edit train_yolo.py and reduce batch processing in export_lmdb_to_yolo_format
```

### Training too slow
```bash
# Use smaller model
--model yolov8n

# Reduce image size
--img-size 416

# Reduce epochs for testing
--epochs 10
```

### Low accuracy
```bash
# Use larger model
--model yolov8l or yolov8x

# Train longer
--epochs 500

# Increase image size
--img-size 640
```

---

## Comparison with Other Models

| Metric | Faster R-CNN | RetinaNet | YOLO |
|--------|--------------|-----------|------|
| Training Speed | Slow | Medium | ⚡ Fast |
| Inference Speed | Slow | Medium | ⚡⚡ Very Fast |
| Accuracy | High | High | High |
| Memory Usage | High | Medium | Low |
| Easy to Use | Medium | Medium | ⚡ Very Easy |
| Deployment | Complex | Medium | ⚡ Easy |

**YOLO Advantages:**
- ✅ Fastest training and inference
- ✅ Easiest to deploy
- ✅ Native ultralytics support
- ✅ Built-in augmentation and optimization

**Use YOLO when:**
- You need real-time inference
- You're deploying to mobile/edge devices
- You want fast iteration during development

---

## Next Steps

### After Training
1. **Evaluate Model**:
   ```bash
   python evaluate.py --checkpoint ../runs/yolo/dfu_production/weights/best.pt
   ```

2. **Test Inference**:
   ```bash
   from ultralytics import YOLO

   model = YOLO('../runs/yolo/dfu_production/weights/best.pt')
   results = model.predict('test_image.jpg')
   ```

3. **Export for Deployment**:
   ```bash
   model = YOLO('../runs/yolo/dfu_production/weights/best.pt')
   model.export(format='onnx')  # or 'tflite', 'coreml', etc.
   ```

### Further Optimization
- Try different augmentations in YOLO config
- Experiment with different hyperparameters
- Use ensemble of multiple models
- Fine-tune on specific failure cases

---

## Additional Resources

- **Ultralytics Documentation**: https://docs.ultralytics.com
- **YOLO GitHub**: https://github.com/ultralytics/ultralytics
- **Model Hub**: https://hub.ultralytics.com
- **Community**: https://community.ultralytics.com

---

**Last Updated**: 2025-11-12
**Maintainer**: DFU Detection Team
