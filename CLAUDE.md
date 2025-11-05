# DFU Detection Project

## 📋 Project Overview

**Diabetic Foot Ulcer (DFU) Detection** using deep learning object detection.

**Goal**: Automatically detect and localize diabetic foot ulcers in medical images to assist clinicians in early diagnosis and treatment planning.

**Model**: Faster R-CNN with EfficientNet-B5 backbone
**Framework**: PyTorch + torchvision
**Current Status**: Training at epoch 18+, Composite Score: 0.74, F1: 0.78

---

## 🏗️ Architecture

### Model Configuration
- **Base Architecture**: Faster R-CNN (two-stage object detector)
- **Backbone**: EfficientNet-B5 (pretrained on ImageNet)
- **Input Size**: 512×512 pixels
- **Classes**: 2-class system
  - Class 0: Background
  - Class 1: DFU ulcer
- **Anchor Sizes**: [32, 64, 128, 256, 512]
- **Aspect Ratios**: [0.5, 1.0, 2.0]

### Training Strategy
- **Optimizer**: AdamW (lr=0.001, weight_decay=0.0001)
- **LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=4)
- **Mixed Precision**: Enabled (AMP)
- **Gradient Clipping**: Max norm = 1.0
- **Early Stopping**: 23 epochs patience
- **Batch Size**: 36 images (18 DFU + 18 healthy via balanced sampling)

### Checkpoint Criteria (IMPORTANT!)
**Best model saved based on COMPOSITE SCORE**, not validation loss:
```
Composite Score = 0.40 × F1 + 0.25 × IoU + 0.20 × Recall + 0.15 × Precision
```

**Why?**
- Val loss measures prediction cost, not clinical value
- F1 balances precision/recall (most important for detection)
- IoU ensures good box localization
- Recall weighted higher (don't miss ulcers!)
- Precision prevents false alarms

---

## 📊 Dataset

### Data Location
⚠️ **Data files NOT in Git** (too large - stored locally/cluster)

**Local Machine**: `/mnt/c/Users/90rez/OneDrive - University of Toronto/PhDUofT/SideProjects/DFU_Detection_Asem/`
**Cluster**: `/workspace/dfu_detection/` (synced via scp)

### Dataset Statistics
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
- **Storage**: LMDB databases (fast loading)
  - `data/train.lmdb`
  - `data/val.lmdb`
- **Original Images**: JPEG format
- **Annotations**: CSV files with bounding boxes (xmin, ymin, xmax, ymax)
- **Bounding Boxes**: Pascal VOC format (pixel coordinates)

### Balanced Sampling
- **Strategy**: 50% DFU / 50% healthy per batch
- **Reason**: Prevent model bias toward majority class
- **Implementation**: Custom `BalancedBatchSampler` in `balanced_sampler.py`
- **Metadata**: Pre-computed DFU/healthy indices stored in LMDB

---

## 🔧 Key Components

### 1. Training Script: `train_improved.py`
**Main training loop with:**
- Worker-safe LMDB loading (multiprocessing compatible)
- Composite score checkpoint saving
- Memory cleanup (startup + per-epoch)
- F1, IoU, Precision, Recall tracking
- Learning rate saved with best model

**Recent Fixes**:
- ✅ LMDB thread-safety for `num_workers=2`
- ✅ Checkpoint based on composite score (not val_loss)
- ✅ Memory cleanup to prevent leaks
- ✅ Restored best scores when resuming training

### 2. Dataset: `dataset.py`
**Two implementations**:

**DFUDataset** (raw images):
- Loads JPEGs directly from disk
- Good for small datasets
- Works with any num_workers

**DFUDatasetLMDB** (database):
- Worker-safe LMDB implementation
- Much faster loading (~3x)
- Each worker gets own LMDB connection
- Includes `get_metadata()` for balanced sampler

**Important**: LMDB uses lazy worker initialization (`_init_lmdb()`) to avoid pickling errors.

### 3. Balanced Sampler: `balanced_sampler.py`
- Ensures 50% DFU + 50% healthy images per batch
- Loads pre-computed indices from LMDB metadata
- Fallback: categorizes images if metadata missing

### 4. Data Preparation: `create_lmdb.py`
**Creates LMDB databases from raw images**:
- Encodes images as JPEG bytes
- Stores bounding boxes and labels
- Pre-computes DFU/healthy indices
- Stores metadata for fast loading

**Run after any annotation changes**:
```bash
python create_lmdb.py
```

### 5. Evaluation: `evaluate.py`
- Computes F1, IoU, Precision, Recall
- Confidence threshold: 0.5
- IoU threshold: 0.5
- Generates per-image metrics

### 6. Model Creation: `train_efficientdet.py`
- Creates Faster R-CNN model
- Loads EfficientNet-B5 backbone
- Configures RPN and ROI heads
- Handles pretrained weights

---

## 🎨 Data Augmentation

### Training Augmentations (Applied)
**Geometric** (bbox-aware):
- HorizontalFlip (p=0.25)
- VerticalFlip (p=0.25)
- Perspective (p=0.2)
- Affine: rotate±40°, scale 0.95-1.05 (p=0.3)

**Color** (realistic medical variations):
- ColorJitter / RandomBrightnessContrast
- HueSaturationValue
- RandomGamma / CLAHE
- RandomShadow

**Quality** (camera/phone variations):
- GaussNoise / ISONoise
- GaussianBlur / MotionBlur
- Downscale / ImageCompression
- Sharpen

**Occlusions**:
- CoarseDropout (very low p=0.05)

**NO CROP AUGMENTATIONS** - Previously removed due to lost bounding boxes.

### Validation Augmentations
- Only resize + pad (no augmentation)

---

## 🚀 Training Workflow

### Local Development (Debugging/Testing)
```bash
cd /home/rezab/projects/dfu_detection/scripts
source /home/rezab/projects/dfu_detection/dfu_detection/bin/activate
python train_improved.py
```

### Cluster Training (Vast.ai)
```bash
# Transfer files
scp -P 45852 -r scripts/ root@93.91.156.87:/workspace/dfu_detection/

# SSH to cluster
ssh -p 45852 root@93.91.156.87
cd /workspace/dfu_detection/scripts

# Start training
python train_improved.py
```

**Cluster Specs**:
- GPU: 1x H200 NVL (140 GB VRAM)
- CPU: 24/192 cores (XEON® PLATINUM 8568Y+)
- RAM: 258 GB
- Storage: 56 GB NVMe (Dell Ent PM1733a)

### Monitoring Training
```bash
# Watch log in real-time
tail -f ../checkpoints_b5/training_log_*.txt

# Check GPU usage
watch -n 1 nvidia-smi

# Check training history
cat ../checkpoints_b5/training_history.json
```

---

## 📦 Project Structure

```
dfu_detection/
├── scripts/
│   ├── train_improved.py          # Main training (composite score, memory cleanup)
│   ├── dataset.py                 # Worker-safe LMDB + raw image dataset
│   ├── balanced_sampler.py        # 50/50 DFU/healthy sampling
│   ├── create_lmdb.py             # Create LMDB databases
│   ├── evaluate.py                # Compute F1, IoU metrics
│   ├── train_efficientdet.py      # Model creation
│   ├── data_preprocessing.py      # Train/val/test split
│   ├── add_healthy_feet.py        # Add healthy images to dataset
│   └── verify_lmdb_data.py        # Visual LMDB verification
│
├── data/                          # ⚠️ NOT IN GIT (too large)
│   ├── train.lmdb                 # Training LMDB database
│   ├── val.lmdb                   # Validation LMDB database
│   ├── train.csv                  # Training annotations
│   └── val.csv                    # Validation annotations
│
├── checkpoints_b5/                # ⚠️ NOT IN GIT (large files)
│   ├── best_model.pth             # Best composite score checkpoint
│   ├── resume_training.pth        # Periodic checkpoints
│   └── training_log_*.txt         # Training logs
│
├── requirements.txt               # Python dependencies
├── CLAUDE.md                      # This file
├── README.md                      # Project README
├── VASTAI_SETUP_GUIDE.md          # Cluster setup instructions
├── PIPELINE_AUDIT_SUMMARY.md      # Bug fixes and improvements
└── .gitignore                     # Exclude large files

⚠️ Large files stored locally/cluster:
  - DFUC2022_train_images/         # Original DFU images
  - HealthyFeet/                   # Healthy feet images
```

---

## 🐛 Known Issues & Solutions

### 1. LMDB Thread-Safety (SOLVED)
**Issue**: `TypeError: 'Transaction' object is not subscriptable` with `num_workers > 0`
**Cause**: LMDB environments cannot be pickled across processes
**Solution**: Worker-safe pattern - each worker initializes own LMDB connection

### 2. Segmentation Faults (SOLVED)
**Issue**: Crashes after 15-20 epochs
**Cause**: Shared memory accumulation + multi-worker DataLoader
**Solution**: Per-epoch memory cleanup + worker-safe LMDB

### 3. Checkpoint Overwriting (SOLVED)
**Issue**: Worse models overwriting better ones after resuming
**Cause**: Best scores not restored from checkpoint
**Solution**: Restore all metrics when loading checkpoint

### 4. Crop Augmentations (REMOVED)
**Issue**: Bounding boxes lost during training
**Cause**: RandomSizedBBoxSafeCrop sometimes removed all boxes
**Solution**: Removed all crop augmentations

---

## 📈 Current Performance

**Best Model** (Epoch 17):
```
Composite Score: 0.7356
F1 Score:        0.7846
Mean IoU:        0.5627
Recall:          0.8688  ← Finding 87% of ulcers!
Precision:       0.7153
Learning Rate:   0.000250
```

**Expected Final Performance** (after full training):
```
Composite: 0.75-0.80
F1:        0.80-0.85
IoU:       0.58-0.65
Recall:    0.85-0.90
Precision: 0.75-0.82
```

---

## 🔄 Typical Development Workflow

### Making Code Changes
1. **Local**: Edit code, test on small subset
2. **Transfer**: `scp` to cluster
3. **Train**: Run full training on cluster
4. **Monitor**: Check logs and metrics
5. **Evaluate**: Load best checkpoint, compute test metrics

### Adding New Features
1. Read relevant files in `scripts/`
2. Understand data flow: `dataset.py` → `balanced_sampler.py` → `train_improved.py`
3. Make changes locally
4. Test with small epoch count
5. Deploy to cluster for full training

### Debugging Training Issues
1. Check `training_log_*.txt` for errors
2. Verify LMDB integrity: `python verify_lmdb_data.py`
3. Check balanced sampling: Look for "DFU images with boxes: 3907"
4. Monitor GPU memory: `nvidia-smi`
5. Check shared memory: `du -sh /dev/shm`

---

## 🔧 Common Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create LMDB databases
python scripts/create_lmdb.py

# Verify LMDB data
python scripts/verify_lmdb_data.py
```

### Training
```bash
# Start training
python scripts/train_improved.py

# Resume from checkpoint (auto-detects best_model.pth)
python scripts/train_improved.py
```

### Evaluation
```bash
# Evaluate best model
python scripts/evaluate.py --checkpoint ../checkpoints_b5/best_model.pth
```

### Cluster Sync
```bash
# Transfer code to cluster
scp -P 45852 -r scripts/ root@93.91.156.87:/workspace/dfu_detection/

# Transfer specific file
scp -P 45852 scripts/train_improved.py root@93.91.156.87:/workspace/dfu_detection/scripts/
```

---

## 💡 Important Notes for Claude Code

### When Modifying Code:

1. **LMDB Changes**: If you modify `DFUDatasetLMDB`, remember:
   - Must be worker-safe (lazy initialization in `_init_lmdb()`)
   - Each worker needs own LMDB environment
   - Don't open LMDB in `__init__`

2. **Checkpoint Changes**: If you modify checkpoint saving:
   - Always save composite score
   - Save learning rate
   - Include all metrics (F1, IoU, Precision, Recall)

3. **Augmentation Changes**:
   - Avoid crop augmentations (cause box loss)
   - All transforms must be bbox-aware
   - Use `bbox_params=A.BboxParams(format='pascal_voc', ...)`

4. **Balanced Sampling**:
   - Requires metadata in LMDB (`__dfu_indices__`, `__healthy_indices__`)
   - Fallback to slow categorization if metadata missing
   - Recreate LMDB if sampling logic changes

5. **Memory Management**:
   - Keep startup cleanup (`cleanup_memory()`)
   - Keep per-epoch cleanup (`cleanup_epoch()`)
   - Critical for long training runs

### Testing Changes:

```bash
# Quick test (3 epochs)
python train_improved.py  # Edit num_epochs in code to 3

# Verify LMDB after changes
python verify_lmdb_data.py

# Check balanced sampling
# Look for: "DFU images with boxes: 3907, Healthy: 920"
```

---

## 📚 Additional Documentation

- [VASTAI_SETUP_GUIDE.md](VASTAI_SETUP_GUIDE.md) - Cluster setup instructions
- [PIPELINE_AUDIT_SUMMARY.md](PIPELINE_AUDIT_SUMMARY.md) - Complete bug fix history
- [FINAL_FIXES_SUMMARY.md](FINAL_FIXES_SUMMARY.md) - Latest session changes
- [WORKER_SAFE_LMDB_FIX.md](WORKER_SAFE_LMDB_FIX.md) - LMDB multiprocessing fix

---

## 🎯 Project Goals

**Short-term**:
- ✅ Fix training stability (LMDB, memory, checkpoints) - DONE
- 🔄 Complete current training run (300 epochs)
- ⏳ Achieve F1 > 0.80 on validation set

**Long-term**:
- Evaluate on test set
- Compare with other architectures (YOLO, RetinaNet)
- Deploy as clinical decision support tool
- Extend to multi-class (ulcer severity grading)

---

## ⚙️ Environment Setup

### Local Machine (WSL)
```bash
source /home/rezab/projects/dfu_detection/dfu_detection/bin/activate
```

### Cluster
```bash
# Python environment already configured in container
cd /workspace/dfu_detection/scripts
```

### Required Packages
See [requirements.txt](requirements.txt) for complete list.

**Key dependencies**:
- PyTorch >= 2.1.0
- torchvision >= 0.16.0
- albumentations >= 1.3.0
- lmdb >= 1.4.0
- opencv-python >= 4.8.0

---

**Last Updated**: 2025-10-28
**Training Status**: Active (Epoch 18+, Composite=0.74)
**Maintainer**: Reza (with Claude Code assistance)
