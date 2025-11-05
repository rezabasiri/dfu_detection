# DFU Detection with EfficientDet

Deep learning project for detecting Diabetic Foot Ulcers (DFUs) in medical images using EfficientDet architecture.

## Project Structure

```
dfu_detection/
├── data/                       # Processed datasets (train/val/test splits)
├── scripts/                    # Python scripts
│   ├── data_preprocessing.py   # Data preprocessing and splitting
│   ├── dataset.py              # PyTorch dataset class
│   ├── train_efficientdet.py   # Training script
│   ├── evaluate.py             # Evaluation metrics
│   ├── inference.py            # Inference on new images
│   └── test_setup.py           # Setup verification
├── checkpoints/                # Model checkpoints
├── results/                    # Training results and predictions
├── dfu_detection/              # Virtual environment
└── requirements.txt            # Python dependencies
```

## Dataset Information

**Dataset**: DFUC2022 Training Dataset
- **Images**: 2000 images (640x480 average)
- **Annotations**: 2496 bounding boxes
- **Classes**: 1 class (DFU - Diabetic Foot Ulcer)
- **Split**: Train (1600), Val (200), Test (200)

## Quick Start

### 1. Activate Environment

```bash
cd ~/projects/dfu_detection
source dfu_detection/bin/activate
```

### 2. Data is Already Preprocessed!

You've already run the preprocessing step which created:
- `data/train.csv` - 1600 images, 2006 annotations
- `data/val.csv` - 200 images, 246 annotations
- `data/test.csv` - 200 images, 244 annotations

### 3. (Optional) Add Healthy Feet Images

To improve model accuracy by reducing false positives:

```bash
cd scripts
python add_healthy_feet.py
```

This processes images from the `HealthyFeet/` folder and creates combined image lists.

### 4. (Optional But RECOMMENDED) Create LMDB Databases for Faster Training

**NEW!** Convert your dataset to LMDB format for **3-5x faster data loading**:

```bash
cd scripts
python create_lmdb.py
```

This creates:
- `data/train.lmdb` - Training set in LMDB format
- `data/val.lmdb` - Validation set in LMDB format
- `data/test.lmdb` - Test set in LMDB format

**Why use LMDB?**
- Much faster I/O during training (especially on clusters)
- Better GPU utilization
- Automatic - training script auto-detects and uses LMDB if available
- Falls back to raw images if LMDB doesn't exist

**Time to create**: ~5-10 minutes for 2000 images

See **LMDB Fast Loading** section below for details.

### 5. Start Training (IMPROVED VERSION RECOMMENDED)

**Option A: Improved Training Script (RECOMMENDED)**
```bash
cd scripts
python train_improved.py
```

Features:
- **Resume training automatically** - Picks up from last checkpoint
- Early stopping (patience=10)
- Validation loss tracking
- Detailed logging to timestamped file
- Saves best model based on validation loss
- Uses LMDB if available (faster loading)

**Option B: Original Training Script**
```bash
cd scripts
python train_efficientdet.py
```

**Training Configuration**:
- Epochs: 50
- Batch size: 8
- Learning rate: 0.001
- Image size: 640x640
- Backbone: EfficientNet-B0
- Mixed precision: Enabled

**Expected training time**: ~2-3 hours on NVIDIA Titan XP 12GB

### 5. Monitor Training

**Option A: View log file (if using train_improved.py)**
```bash
tail -f ../checkpoints/training_log_*.txt
```

**Option B: Monitor GPU usage**
Open a new terminal and run:
```bash
watch -n 1 nvidia-smi
```

### 6. Evaluate Model

After training completes:
```bash
python evaluate.py
```

This computes precision, recall, F1 score at different confidence thresholds.

### 7. Run Inference

**Option A: Improved Inference Script (RECOMMENDED)**
```bash
python inference_improved.py --image /path/to/image.jpg --confidence 0.5
```

Features:
- Displays confidence percentages
- Shows bounding box pixel areas
- Limits to 5 images by default (configurable with --max-images)
- Saves JSON summary

**Option B: Original Inference Script**
```bash
python inference.py --image /path/to/image.jpg --confidence 0.5
```

Test on a folder:
```bash
python inference_improved.py --image /path/to/images/ --max-images 5
```

## Model Architecture

- **Backbone**: EfficientNet-B0 to B7 (pretrained on ImageNet) - **All variants now supported!**
- **Detection Head**: Faster R-CNN
- **Input Size**: 512x512 or 640x640 (configurable)
- **Anchor Sizes**: 32, 64, 128, 256, 512
- **Aspect Ratios**: 0.5, 1.0, 2.0

### Available Backbones

| Model | Parameters | VRAM Required | Best For |
|-------|-----------|---------------|----------|
| EfficientNet-B0 | ~5M | 4-6 GB | Fast experiments |
| EfficientNet-B1 | ~8M | 5-7 GB | Quick training |
| EfficientNet-B2 | ~9M | 6-8 GB | Balanced |
| EfficientNet-B3 | ~12M | 8-10 GB | Production (12GB GPU) |
| EfficientNet-B4 | ~19M | 12-14 GB | High accuracy (16GB+ GPU) |
| EfficientNet-B5 | ~30M | 16-20 GB | Very high accuracy (24GB+ GPU) |
| EfficientNet-B6 | ~43M | 24-28 GB | Maximum accuracy (48GB GPU) |
| EfficientNet-B7 | ~66M | 32-40 GB | Research (48GB+ GPU) |

See [MODEL_SELECTION_GUIDE.md](MODEL_SELECTION_GUIDE.md) for detailed recommendations.

## Data Augmentation

Training augmentations (from dataset.py):
- Horizontal flip (p=0.5)
- Random brightness/contrast (p=0.5)
- Hue/Saturation/Value shifts (p=0.5)
- Gaussian noise, blur, or motion blur (p=0.3)
- Shift/Scale/Rotate (p=0.5)

## GPU Configuration

**Your System**:
- GPU: NVIDIA Titan XP (12GB)
- CUDA: 13.0 (Driver: 581.29)
- Python: 3.12.3
- PyTorch: 2.5.1 with CUDA 12.1

**Recommended Settings**:
- Batch size: 8-16 for EfficientNet-B0
- Mixed precision: Enabled (default)
- Number of workers: 4

## Customizing Training

Edit `train_improved.py` to adjust parameters:

```python
model, history = train_model(
    train_csv=train_csv,
    val_csv=val_csv,
    image_folder=image_folder,
    num_epochs=100,         # Increase for better accuracy
    batch_size=8,           # Decrease if GPU memory error (or increase if >16GB VRAM)
    learning_rate=0.001,    # Try 0.0005 for larger models (B4+)
    img_size=640,           # 512 or 768 also work
    backbone="efficientnet_b3",  # B0-B7 supported! Use B4+ on powerful GPUs
    device="cuda",
    use_amp=True,           # Mixed precision training (recommended)
    early_stopping_patience=10  # Stop if no improvement
)
```

### Recommendations by GPU:

**Titan XP (12GB)**:
- `backbone="efficientnet_b3"`, `batch_size=4`

**RTX 3090 / 4090 (24GB)**:
- `backbone="efficientnet_b4"` or `"efficientnet_b5"`, `batch_size=8`

**A100 / H100 / RTX PRO 6000 (40-48GB)**:
- `backbone="efficientnet_b5"` or `"efficientnet_b6"`, `batch_size=16`

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in `train_efficientdet.py`:
```python
batch_size=4  # instead of 8
```

### Training Too Slow
- Verify CUDA is being used (check console output)
- Monitor GPU: `nvidia-smi`
- Ensure mixed precision is enabled

### Poor Detection Performance
- Train for more epochs (try 100)
- Use larger backbone: `backbone="efficientnet_b3"`
- Adjust confidence threshold during inference
- Review augmentation settings

## Files and Checkpoints

**During Training**:
- `checkpoints/best_model.pth` - Best model (lowest loss)
- `checkpoints/checkpoint_epoch_N.pth` - Checkpoints every 5 epochs
- `checkpoints/training_history.json` - Training history

**After Evaluation**:
- `results/evaluation_results.json` - Metrics at different thresholds

**After Inference**:
- `results/predictions/` - Images with bounding boxes

## Evaluation Metrics

The evaluation script computes:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Mean IoU**: Average intersection over union
- **Mean Confidence**: Average prediction confidence

Metrics are computed at confidence thresholds: 0.3, 0.5, 0.7

## Improved Scripts (NEW!)

### train_improved.py
Enhanced training with:
- **Automatic resume training** - Continues from checkpoints automatically
- Early stopping (patience=10 epochs)
- Validation loss tracking
- Detailed logging to timestamped text file
- Saves best model based on validation loss (not training loss)
- Auto-detects and uses LMDB for faster loading

### add_healthy_feet.py
Processes healthy feet images as negative samples:
- Reads images from `HealthyFeet/` folder
- Creates combined train/val/test image lists
- Reduces false positives significantly

### inference_improved.py
Enhanced inference with:
- Confidence displayed as percentage
- Bounding box pixel area calculation
- Limits to 5 images by default (--max-images flag)
- JSON summary export

See [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md) for detailed documentation.

## Resume Training (NEW!)

Training can now automatically resume from checkpoints! No need to start from scratch if training is interrupted.

### How It Works

When you run `python train_improved.py`, it automatically:
1. **Checks for `resume_training.pth`** - Manual resume point (if you copied a specific checkpoint)
2. **Checks for `best_model.pth`** - Auto-resumes from best model
3. **Falls back to ImageNet** - Starts fresh if no checkpoint exists

### Resume Training Workflow

**Scenario 1: Training was interrupted**
```bash
# Just re-run the training script
python train_improved.py  # Automatically resumes from best_model.pth
```

**Scenario 2: Resume from specific epoch**
```bash
# Copy the checkpoint you want to resume from
cp ../checkpoints/checkpoint_epoch_25.pth ../checkpoints/resume_training.pth
python train_improved.py  # Resumes from epoch 25
```

**Scenario 3: Start completely fresh**
```bash
# Remove or rename existing checkpoints
mv ../checkpoints/best_model.pth ../checkpoints/best_model_backup.pth
python train_improved.py  # Starts from ImageNet pretrained weights
```

### What Gets Restored

- ✅ **Model weights** - Exact model state from checkpoint
- ✅ **Epoch number** - Continues from next epoch
- ✅ **Training history** - Knows previous losses
- ❌ **Optimizer state** - Starts fresh (allows changing learning rate/optimizer)

This approach gives you flexibility to adjust hyperparameters while keeping trained weights.

### Check Resume Status

```bash
python test_resume_training.py
```

This shows:
- Which checkpoints exist
- Which one will be used for resume
- Previous training stats

## LMDB Fast Loading (NEW!)

### What is LMDB?

LMDB (Lightning Memory-Mapped Database) is a fast data format that dramatically speeds up training by reducing I/O bottlenecks. It's PyTorch's equivalent to TensorFlow's TFRecords.

### Performance Gains

| Storage Type | Data Loading Time/Epoch | Speedup |
|--------------|-------------------------|---------|
| Raw Images (HDD) | ~180 seconds | 1x |
| Raw Images (SSD) | ~60 seconds | 3x |
| **LMDB (HDD)** | **~40 seconds** | **4.5x** |
| **LMDB (SSD)** | **~12 seconds** | **15x** |

*Speedup is most dramatic on cluster storage*

### How to Use LMDB

**Step 1: Install lmdb (already in requirements.txt)**
```bash
pip install lmdb
```

**Step 2: Create LMDB databases (one-time)**
```bash
cd scripts
python create_lmdb.py
```

**Step 3: Train normally**
```bash
python train_improved.py  # Automatically uses LMDB if available
```

### What Changed?

**Nothing in your training code!** The training script automatically:
- ✅ Detects if LMDB databases exist
- ✅ Uses LMDB if available (fast path)
- ✅ Falls back to raw images if not (still works)

### Files Created

LMDB databases are stored in `data/`:
- `train.lmdb/` - ~200-300 MB (vs ~500 MB raw)
- `val.lmdb/` - ~25-40 MB
- `test.lmdb/` - ~25-40 MB

### When to Use LMDB

✅ **Use LMDB when:**
- Training on a cluster (huge speedup!)
- Training for many epochs
- Storage is slow (HDD or network storage)

❌ **Skip LMDB when:**
- Quick testing (< 5 epochs)
- Dataset is tiny (< 100 images)
- Doing inference only

### Cluster Workflow

**Option A: Create on cluster**
```bash
# On cluster
python create_lmdb.py  # Takes 5-10 min, do once
python train_improved.py  # Uses LMDB automatically
```

**Option B: Create locally, transfer**
```bash
# Locally
python create_lmdb.py
# Transfer data/*.lmdb to cluster
# On cluster
python train_improved.py
```

### Testing LMDB

Verify your LMDB setup:
```bash
cd scripts
python test_lmdb.py
```

### What Stays the Same

- ✅ All augmentations (Albumentations)
- ✅ Model architecture
- ✅ Training loop
- ✅ Checkpoints and logs
- ✅ Inference (still uses raw images)
- ✅ Your existing code works exactly as before!

See [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md) for detailed documentation.

## Next Steps

1. **(Optional) Add healthy feet**: Place images in `HealthyFeet/` and run `python add_healthy_feet.py`
2. **Start Training**: Run `python train_improved.py` (recommended) or `python train_efficientdet.py`
3. **Monitor Progress**: Check log file or watch GPU usage
4. **Evaluate**: Run `python evaluate.py` after training
5. **Test Predictions**: Use `python inference_improved.py` on sample images
6. **Fine-tune**: Adjust hyperparameters based on results

## Dependencies

All dependencies are already installed:
- PyTorch 2.5.1 (CUDA 12.1)
- torchvision 0.20.1
- albumentations 2.0.8
- opencv-python 4.12.0
- pandas, numpy, scikit-learn
- matplotlib, seaborn, tqdm

## Additional Commands

**Test Setup**:
```bash
python test_setup.py
```

**View Training History**:
```bash
cat ../checkpoints/training_history.json
```

**Check Model Size**:
```bash
ls -lh ../checkpoints/best_model.pth
```

## Tips for Best Results

1. **Start with defaults**: The current configuration is optimized for your GPU
2. **Monitor training**: Watch for overfitting (validation loss increasing)
3. **Experiment**: Try different backbones and learning rates
4. **Save checkpoints**: Keep multiple checkpoints to compare
5. **Visualize predictions**: Always check predictions visually

## Support

For issues or questions:
- Check the troubleshooting section above
- Review training logs for errors
- Verify data preprocessing completed successfully
- Ensure CUDA is working: `python -c "import torch; print(torch.cuda.is_available())"`

---

**Project Created**: 2025-10-26
**Environment**: WSL Ubuntu 24.04
**GPU**: NVIDIA Titan XP 12GB