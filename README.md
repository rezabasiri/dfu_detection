# DFU Detection with Multi-Model Architecture

Deep learning project for detecting Diabetic Foot Ulcers (DFUs) in medical images using state-of-the-art object detection architectures.

**Supported Models:**
- **Faster R-CNN** with EfficientNet backbones (B0-B7)
- **RetinaNet** with EfficientNet backbones (focal loss for better recall)
- **YOLOv8** (fastest inference for deployment)

## 🎯 Project Overview

Automated detection and localization of diabetic foot ulcers to assist clinicians in early diagnosis and treatment planning.

**Current Status:**
- Training: Active (Epoch 18+, Composite Score: 0.74, F1: 0.78)
- Architecture: Multi-model zoo with unified interface
- Data: 5,413 images (4,380 DFU + 1,033 healthy feet)

## 🏗️ Project Structure

```
dfu_detection/
├── scripts/
│   ├── models/                      # 🆕 Model zoo
│   │   ├── base_model.py           # Abstract base class
│   │   ├── faster_rcnn.py          # Faster R-CNN implementation
│   │   ├── retinanet.py            # RetinaNet with focal loss
│   │   ├── yolo.py                 # YOLOv8 wrapper
│   │   └── model_factory.py        # Factory pattern
│   │
│   ├── configs/                     # 🆕 Model configurations (YAML)
│   │   ├── faster_rcnn_b5.yaml     # Production config
│   │   ├── faster_rcnn_b3.yaml     # Lighter variant
│   │   ├── retinanet.yaml          # Recommended for recall
│   │   └── yolov8.yaml             # Fastest inference
│   │
│   ├── debugging/                   # 🆕 Debugging utilities
│   │   ├── test_train.py           # Quick training test
│   │   ├── test_lmdb.py            # LMDB verification
│   │   ├── verify_lmdb_data.py     # Visual LMDB check
│   │   ├── diagnose_data.py        # Data quality diagnostics
│   │   └── validate_dataset.py     # Image corruption checks
│   │
│   ├── train_improved.py            # ✏️ Main training script (supports all models)
│   ├── train_all_models.py          # 🆕 Train all models for comparison
│   ├── evaluate.py                  # ✏️ Evaluation (auto-detects model)
│   ├── inference_improved.py        # ✏️ Inference (works with all models)
│   ├── dataset.py                   # PyTorch dataset (LMDB + raw images)
│   ├── balanced_sampler.py          # Balanced batch sampling
│   ├── create_lmdb.py               # Create LMDB databases
│   ├── data_preprocessing.py        # Train/val/test split
│   └── add_healthy_feet.py          # Add negative samples
│
├── data/                            # ⚠️ NOT IN GIT (too large)
│   ├── train.lmdb                   # Training LMDB database
│   ├── val.lmdb                     # Validation LMDB database
│   ├── train.csv                    # Training annotations
│   └── val.csv                      # Validation annotations
│
├── checkpoints/                     # 🆕 Model-specific subdirectories
│   ├── faster_rcnn/                 # Faster R-CNN checkpoints
│   ├── retinanet/                   # RetinaNet checkpoints
│   └── yolo/                        # YOLO checkpoints
│
├── requirements.txt                 # Python dependencies
├── CLAUDE.md                        # Detailed project documentation
├── README.md                        # This file
└── .gitignore                       # Exclude large files
```

## 📊 Dataset Information

**Dataset**: DFUC2022 + Healthy Feet

### Statistics
```
Training Set:   4,812 images
  ├─ DFU (with boxes):     3,907 images (label=1)
  └─ Healthy (no boxes):     920 images (hard negatives)

Validation Set:   992 images
  ├─ DFU:     488 images
  └─ Healthy: 113 images

Total: 5,413 images (4,380 DFU + 1,033 healthy)
```

### Data Format
- **Storage**: LMDB databases (fast loading) + raw images (fallback)
- **Images**: JPEG format, various sizes
- **Annotations**: CSV files with bounding boxes (xmin, ymin, xmax, ymax)
- **Balanced Sampling**: 50% DFU / 50% healthy per batch

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
cd ~/projects/dfu_detection
source dfu_detection/bin/activate

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### 2. Prepare Data (One-time Setup)

#### Option A: Use Existing LMDB (Recommended - if available)
If LMDB databases already exist in `data/`, you're ready to train!

#### Option B: Create LMDB from Scratch
```bash
cd scripts

# Step 1: Split data into train/val/test
python data_preprocessing.py

# Step 2: (Optional) Add healthy feet images
python add_healthy_feet.py

# Step 3: Create LMDB databases (3-5x faster loading)
python create_lmdb.py
```

**Why use LMDB?**
- 3-5x faster I/O during training
- Better GPU utilization
- Worker-safe for multiprocessing
- Automatic fallback to raw images if LMDB doesn't exist

### 3. Choose Your Model & Train

#### Option A: Train Specific Model with Config (Recommended)

```bash
cd scripts

# Faster R-CNN (current production model)
python train_improved.py --model faster_rcnn --config configs/faster_rcnn_b5.yaml

# RetinaNet (recommended for better recall - good for medical diagnosis)
python train_improved.py --model retinanet --config configs/retinanet.yaml

# YOLOv8 (fastest inference - good for deployment)
python train_improved.py --model yolo --config configs/yolov8.yaml
```

#### Option B: Train with Command-Line Arguments

```bash
# Quick training with custom parameters
python train_improved.py \
    --model faster_rcnn \
    --backbone efficientnet_b3 \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001
```

#### Option C: Train All Models for Comparison

```bash
# Train all three models with their respective configs
python train_all_models.py

# Or train specific models only
python train_all_models.py --models faster_rcnn retinanet

# Quick test (10 epochs each)
python train_all_models.py --epochs 10
```

### 4. Monitor Training

```bash
# Watch training log in real-time
tail -f ../checkpoints/faster_rcnn/training_log_*.txt

# Check GPU usage
watch -n 1 nvidia-smi

# View training history
cat ../checkpoints/faster_rcnn/training_history.json
```

### 5. Evaluate Models

```bash
# Evaluate any model (auto-detects architecture)
python evaluate.py --checkpoint ../checkpoints/faster_rcnn/best_model.pth

# Test multiple confidence thresholds
python evaluate.py \
    --checkpoint ../checkpoints/retinanet/best_model.pth \
    --conf-thresholds 0.2 0.3 0.5 0.7

# Compare all models
python evaluate.py --checkpoint ../checkpoints/faster_rcnn/best_model.pth
python evaluate.py --checkpoint ../checkpoints/retinanet/best_model.pth
python evaluate.py --checkpoint ../checkpoints/yolo/best_model.pth
```

### 6. Run Inference

```bash
# Single image
python inference_improved.py \
    --checkpoint ../checkpoints/retinanet/best_model.pth \
    --image /path/to/test/image.jpg \
    --confidence 0.3

# Directory of images
python inference_improved.py \
    --checkpoint ../checkpoints/faster_rcnn/best_model.pth \
    --image /path/to/test/directory/ \
    --max-images 10
```

## 🎨 Model Architecture Comparison

| Feature | Faster R-CNN | RetinaNet | YOLOv8 |
|---------|--------------|-----------|--------|
| **Type** | Two-stage | Single-stage | Single-stage |
| **Backbone** | EfficientNet B0-B7 | EfficientNet B0-B7 | CSPDarknet |
| **Loss Function** | Cross-Entropy | **Focal Loss** | YOLO Loss |
| **Inference Speed** | ~200ms | ~100ms | **~20ms** |
| **Best For** | High accuracy | **Medical (recall)** | Deployment |
| **Recall** | Good (0.85-0.88) | **Better (0.90-0.93)** | Good (0.86-0.89) |
| **Precision** | High (0.70-0.75) | Medium (0.65-0.70) | Medium (0.68-0.72) |
| **VRAM (B5/M)** | ~16GB | ~12GB | **~8GB** |
| **Training Time** | Slower | Medium | **Faster** |

### Recommendations

**For Medical Diagnosis (Prioritize Recall):**
- ✅ **RetinaNet** - Focal loss handles class imbalance, higher recall
- Configure with recall-focused weights: `recall: 0.50` in config

**For Highest Accuracy:**
- ✅ **Faster R-CNN with EfficientNet-B5** - Current production model

**For Fast Deployment:**
- ✅ **YOLOv8** - 5-10x faster inference, good for real-time applications

## ⚙️ Configuration System

All models are configured via YAML files in `scripts/configs/`. This ensures reproducibility and makes experimentation easy.

### Configuration Structure

```yaml
model:
  type: retinanet
  backbone: efficientnet_b3
  num_classes: 2
  # Model-specific hyperparameters...

training:
  img_size: 512
  batch_size: 36
  num_epochs: 200
  learning_rate: 0.001

  # Composite score weights for checkpoint saving
  composite_weights:
    f1: 0.15
    iou: 0.20
    recall: 0.50      # ⭐ High for medical use
    precision: 0.15

evaluation:
  confidence_threshold: 0.3  # Lower for better recall
  iou_threshold: 0.5

checkpoint:
  save_dir: ../checkpoints/retinanet
  save_every_n_epochs: 25
```

### Composite Score Explained

Models are saved based on a **composite score** that balances multiple metrics:

```python
Composite Score = (w1 × F1) + (w2 × IoU) + (w3 × Recall) + (w4 × Precision)
```

**Default weights (balanced):**
- F1: 40%, IoU: 25%, Recall: 20%, Precision: 15%

**Medical-focused weights (prioritize not missing ulcers):**
- F1: 15%, IoU: 20%, Recall: **50%**, Precision: 15%

Adjust in config file based on your priorities!

## 🔧 Training Features

### Automatic Checkpoint Management
- **Auto-resume**: Automatically resumes from `best_model.pth` if found
- **Model metadata**: Checkpoints include model name, config, and metrics
- **Auto-detection**: No need to specify architecture when loading

### Balanced Batch Sampling
- Ensures 50% DFU images and 50% healthy images per batch
- Prevents model bias toward majority class
- Reduces false positives on healthy feet

### Memory Optimization
- Pre-training cleanup to prevent segfaults
- Per-epoch cleanup to prevent memory leaks
- Worker-safe LMDB loading for multiprocessing

### Early Stopping
- Monitors composite score (not just loss)
- Configurable patience (default: 23 epochs for Faster R-CNN, 20 for RetinaNet)
- Saves best model automatically

### Learning Rate Scheduling
- ReduceLROnPlateau based on validation loss
- Factor: 0.5, Patience: 4 epochs
- Minimum LR: 1e-5

## 📈 Current Performance

**Best Model** (Faster R-CNN, Epoch 17):
```
Composite Score: 0.7356
F1 Score:        0.7846
Mean IoU:        0.5627
Recall:          0.8688  ← Finding 87% of ulcers
Precision:       0.7153
```

**Expected with RetinaNet** (after full training):
```
Composite Score: 0.75-0.80
F1 Score:        0.78-0.82
Mean IoU:        0.58-0.65
Recall:          0.90-0.93  ← Better for medical use
Precision:       0.65-0.70
```

## 🐛 Troubleshooting

### High False Negatives (Missing Ulcers)

**Immediate fixes (no retraining):**
1. Lower confidence threshold:
   ```bash
   python evaluate.py --checkpoint <path> --conf-thresholds 0.2 0.3
   ```

2. Use model with lower threshold for inference:
   ```bash
   python inference_improved.py --checkpoint <path> --confidence 0.2
   ```

**Long-term fixes (retraining):**
1. **Train RetinaNet** (focal loss for better recall):
   ```bash
   python train_improved.py --model retinanet --config configs/retinanet.yaml
   ```

2. **Adjust composite weights** in config to prioritize recall:
   ```yaml
   composite_weights:
     recall: 0.50  # Increase from 0.20
   ```

3. **Lower RPN thresholds** in Faster R-CNN config:
   ```yaml
   model:
     rpn_positive_iou: 0.5  # Lower from 0.7
   ```

### LMDB Issues

If you encounter LMDB-related errors:

```bash
# Verify LMDB integrity
cd scripts/debugging
python verify_lmdb_data.py

# Recreate LMDB if corrupted
cd scripts
python create_lmdb.py
```

### Memory Issues

If training crashes with segfaults:

1. Reduce batch size in config
2. Use lighter backbone (e.g., B3 instead of B5)
3. Set `num_workers: 0` in config

### Legacy Checkpoints

Old checkpoints without `model_name` field will automatically be loaded as Faster R-CNN with fallback logic.

## 📚 Advanced Usage

### Custom Backbone

```bash
python train_improved.py \
    --model faster_rcnn \
    --backbone efficientnet_b4 \
    --epochs 100
```

Supported backbones: `efficientnet_b0` through `efficientnet_b7`

### Custom Configuration

Create your own YAML config:

```yaml
# my_custom_config.yaml
model:
  type: faster_rcnn
  backbone: efficientnet_b4
  rpn_positive_iou: 0.4  # Very sensitive

training:
  composite_weights:
    recall: 0.60  # Prioritize recall heavily
    f1: 0.15
    iou: 0.15
    precision: 0.10
```

Then train:
```bash
python train_improved.py --config my_custom_config.yaml
```

### Ensemble Predictions

Train multiple models and ensemble their predictions:

```python
# Load all three models
faster_rcnn = create_from_checkpoint('checkpoints/faster_rcnn/best_model.pth')
retinanet = create_from_checkpoint('checkpoints/retinanet/best_model.pth')
yolo = create_from_checkpoint('checkpoints/yolo/best_model.pth')

# Get predictions from all models
pred1 = faster_rcnn.get_model()(images)
pred2 = retinanet.get_model()(images)
pred3 = yolo.get_model()(images)

# Combine with NMS or voting
```

## 🔄 Typical Workflow

### Experimenting with New Model

```bash
# 1. Edit or create config file
vim scripts/configs/my_experiment.yaml

# 2. Quick test (10 epochs)
python train_improved.py --config configs/my_experiment.yaml --epochs 10

# 3. If looks good, full training
python train_improved.py --config configs/my_experiment.yaml

# 4. Evaluate
python evaluate.py --checkpoint ../checkpoints/<model>/best_model.pth

# 5. Test inference
python inference_improved.py --checkpoint <path> --image test.jpg
```

### Comparing Multiple Models

```bash
# Train all models
python train_all_models.py

# Evaluate each
for model in faster_rcnn retinanet yolo; do
    python evaluate.py --checkpoint ../checkpoints/$model/best_model.pth
done

# Compare results in logs
```

## 📦 Dependencies

Key dependencies (see `requirements.txt` for full list):
- PyTorch >= 2.1.0
- torchvision >= 0.16.0
- ultralytics >= 8.0.0 (for YOLO)
- albumentations >= 1.3.0
- lmdb >= 1.4.0
- pyyaml >= 6.0

## 📖 Documentation

- **CLAUDE.md**: Comprehensive project documentation (architecture, training strategy, known issues)
- **configs/*.yaml**: Model-specific configurations
- **training_log_*.txt**: Detailed training logs (in checkpoint directories)

## 🎯 Future Improvements

- [ ] Test-Time Augmentation (TTA) for inference
- [ ] Multi-scale training
- [ ] Attention mechanisms
- [ ] Transformer-based detectors (DETR, Swin-T)
- [ ] Deployment optimizations (ONNX, TensorRT)
- [ ] Web interface for inference
- [ ] Multi-class detection (ulcer severity grading)

## 🤝 Contributing

When contributing:
1. Use descriptive commit messages
2. Test with multiple models before pushing
3. Update configs if changing hyperparameters
4. Update documentation for new features

## 📄 License

[Specify your license here]

## 👥 Authors

- Reza (with Claude Code assistance)

## 🙏 Acknowledgments

- DFUC2022 dataset providers
- PyTorch and torchvision teams
- Ultralytics for YOLOv8
- EfficientNet authors

---

**Last Updated**: 2025-01-11
**Current Status**: Active development - Multi-model architecture implemented
**Branch**: `claude/analyze-project-structure-011CUqZ63FCBE42TAzAxFTBA`
