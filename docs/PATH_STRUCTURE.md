# Directory Structure and Checkpoint Paths

## Project Directory Structure

```
dfu_detection/                           (project root)
├── scripts/                             (working directory for training)
│   ├── test_small/
│   │   ├── run_test_training.py         (test runner script)
│   │   ├── compare_test_models.py       (comparison script)
│   │   ├── test_faster_rcnn.yaml
│   │   ├── test_retinanet.yaml
│   │   └── test_yolov8.yaml
│   │
│   ├── train_improved.py                (Faster R-CNN/RetinaNet training)
│   ├── train_yolo.py                    (YOLO training)
│   ├── evaluate_yolo.py                 (YOLO evaluation)
│   └── ... (other training scripts)
│
├── data/                                (dataset directory)
│   ├── test_train.lmdb
│   ├── test_val.lmdb
│   ├── train.lmdb
│   └── val.lmdb
│
├── checkpoints_test/                    (test checkpoints - created during training)
│   ├── faster_rcnn/
│   │   ├── best_model.pth               ← Best model (Composite Score)
│   │   ├── resume_training.pth          ← Latest epoch (for resuming)
│   │   └── training_log_*.txt
│   │
│   ├── retinanet/
│   │   ├── best_model.pth               ← Best model (Composite Score)
│   │   ├── resume_training.pth          ← Latest epoch (for resuming)
│   │   └── training_log_*.txt
│   │
│   └── yolo/
│       ├── weights/
│       │   ├── best.pt                  ← Best model (mAP@0.5:0.95)
│       │   ├── last.pt                  ← Latest epoch (for resuming)
│       │   ├── epoch1.pt                ← Periodic checkpoint
│       │   └── epoch2.pt                ← Periodic checkpoint
│       ├── results.csv
│       ├── results.png
│       └── confusion_matrix.png
│
└── docs/
    └── PATH_STRUCTURE.md                (this file)
```

---

## Path Resolution in Training Scripts

### run_test_training.py

**Location**: `scripts/test_small/run_test_training.py`

**Working Directory**: Changes to `scripts/` at startup
```python
scripts_dir = Path(__file__).parent.parent  # scripts/
os.chdir(scripts_dir)
```

**Paths Used**:
```python
# Data paths (relative to scripts/)
'--train-lmdb', '../data/test_train.lmdb'     → dfu_detection/data/test_train.lmdb ✓
'--val-lmdb', '../data/test_val.lmdb'         → dfu_detection/data/test_val.lmdb ✓

# Checkpoint paths (relative to scripts/)
'--project', '../checkpoints_test'            → dfu_detection/checkpoints_test ✓
```

**YAML Config Paths** (relative to YAML file location in `test_small/`):
```yaml
# In test_faster_rcnn.yaml, test_retinanet.yaml
checkpoint:
  save_dir: ../../checkpoints_test/faster_rcnn  → dfu_detection/checkpoints_test/faster_rcnn ✓

data:
  train_lmdb: ../../data/test_train.lmdb        → dfu_detection/data/test_train.lmdb ✓
  val_lmdb: ../../data/test_val.lmdb            → dfu_detection/data/test_val.lmdb ✓
```

### compare_test_models.py

**Location**: `scripts/test_small/compare_test_models.py`

**Path Resolution**:
```python
script_dir = Path(__file__).parent        # scripts/test_small/
scripts_dir = script_dir.parent           # scripts/
project_dir = scripts_dir.parent          # dfu_detection/
checkpoint_base = project_dir / "checkpoints_test"  → dfu_detection/checkpoints_test ✓
```

**Checkpoint Paths**:
```python
# Faster R-CNN / RetinaNet
checkpoint_base / "faster_rcnn" / "best_model.pth"
  → dfu_detection/checkpoints_test/faster_rcnn/best_model.pth ✓

# YOLO (different structure!)
checkpoint_base / "yolo" / "weights" / "best.pt"
  → dfu_detection/checkpoints_test/yolo/weights/best.pt ✓
```

---

## Key Differences: YOLO vs Faster R-CNN/RetinaNet

### Checkpoint Location

**Faster R-CNN / RetinaNet**:
```
checkpoints_test/<model>/best_model.pth        (root of model directory)
```

**YOLO**:
```
checkpoints_test/yolo/weights/best.pt          (in weights/ subdirectory)
```

### Checkpoint Format

**Faster R-CNN / RetinaNet** (`.pth` file):
- Contains: model, optimizer, metrics, epoch, learning rate
- Stores: F1, IoU, Precision, Recall, Composite Score
- Can resume training directly

**YOLO** (`.pt` file):
- `best.pt`: Model weights only (no optimizer, no custom metrics)
- `last.pt`: Model + optimizer + training state (for resuming)
- **Does NOT store**: F1, IoU, Precision, Recall, Composite Score
- Need to run `evaluate_yolo.py` separately for comparable metrics

---

## Common Operations

### Run Test Training

```bash
cd scripts/test_small

# Train all models
python run_test_training.py --gpu

# Train specific models
python run_test_training.py --models yolo --gpu
python run_test_training.py --models faster_rcnn retinanet
```

**Result**: Creates `dfu_detection/checkpoints_test/` with all model checkpoints ✓

### Compare Models

```bash
cd scripts/test_small
python compare_test_models.py
```

**What it does**:
1. Loads `../checkpoints_test/faster_rcnn/best_model.pth`
2. Loads `../checkpoints_test/retinanet/best_model.pth`
3. Loads `../checkpoints_test/yolo/weights/best.pt`
4. Compares metrics (YOLO shows 0.0000 - needs separate eval)

### Evaluate YOLO with Comparable Metrics

```bash
cd scripts
python evaluate_yolo.py --model ../checkpoints_test/yolo/weights/best.pt --val-lmdb ../data/test_val.lmdb
```

**Result**: Computes F1, IoU, Precision, Recall, Composite Score for YOLO ✓

---

## Troubleshooting

### Issue: "Checkpoint not found"

**Check**:
1. Did training complete? Look for `checkpoints_test/` directory
2. Are you in the correct directory?
   - For training: `cd scripts/test_small` then run
   - For comparison: `cd scripts/test_small` then run
   - For evaluation: `cd scripts` then run

**YOLO specific**: Remember YOLO saves to `checkpoints_test/yolo/weights/best.pt`, not `checkpoints_test/yolo/best.pt`!

### Issue: "YOLO shows 0.0000 for all metrics"

**This is expected!** YOLO checkpoints don't store custom metrics.

**Solution**: Run `evaluate_yolo.py` to compute comparable metrics:
```bash
cd scripts
python evaluate_yolo.py --model ../checkpoints_test/yolo/weights/best.pt
```

### Issue: "Relative path doesn't work"

**Always use paths relative to your current working directory:**

From `scripts/test_small/`:
- Checkpoints: `../checkpoints_test/`
- Data: `../data/`

From `scripts/`:
- Checkpoints: `../checkpoints_test/`
- Data: `../data/`

From project root (`dfu_detection/`):
- Checkpoints: `checkpoints_test/`
- Data: `data/`

---

## Summary

✅ **All checkpoint paths are correct!**

**Checkpoint locations**:
- Faster R-CNN: `dfu_detection/checkpoints_test/faster_rcnn/best_model.pth`
- RetinaNet: `dfu_detection/checkpoints_test/retinanet/best_model.pth`
- YOLO: `dfu_detection/checkpoints_test/yolo/weights/best.pt` ⚠️ Note: `weights/` subdirectory!

**Working directories**:
- `run_test_training.py`: Changes to `scripts/` at startup
- `compare_test_models.py`: Navigates from `test_small/` → `scripts/` → `dfu_detection/`
- Both resolve to same checkpoint directory: `dfu_detection/checkpoints_test/` ✓

**Special notes**:
- YOLO checkpoints don't store custom metrics (F1, IoU, etc.)
- Run `evaluate_yolo.py` to get comparable metrics
- `compare_test_models.py` will show 0.0000 for YOLO until you run evaluation

---

**Last Updated**: 2025-11-12
