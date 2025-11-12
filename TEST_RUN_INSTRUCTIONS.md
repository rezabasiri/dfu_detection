# Test Run Instructions

This guide will help you run a small test with 100 images, 2 epochs, batch size 8, and 128x128 images for all three models.

## Prerequisites

1. **Virtual Environment**: Created at `/home/rezab/projects/enviroments/dfu_detection_env`
2. **All dependencies installed** (PyTorch, ultralytics, albumentations, etc.)

## Quick Start

### Step 1: Activate Virtual Environment

```bash
source /home/rezab/projects/enviroments/dfu_detection_env/bin/activate
cd /home/rezab/projects/dfu_detection/scripts
```

### Step 2: Create Test Dataset (80 train + 20 val images)

```bash
python test_small/create_test_dataset.py
```

This will create:
- `../data/test_train.lmdb` (80 images, 40 DFU + 40 healthy)
- `../data/test_val.lmdb` (20 images, 10 DFU + 10 healthy)

### Step 3: Run Test Training (All Three Models)

```bash
# For CPU training
python test_small/run_test_training.py

# For GPU training (if CUDA is available)
python test_small/run_test_training.py --gpu
```

This will train:
1. **Faster R-CNN** with EfficientNet-B0 backbone
2. **RetinaNet** with EfficientNet-B0 backbone
3. **YOLOv8-nano** (smallest YOLO model)

Each model will train for:
- **2 epochs**
- **Batch size: 8**
- **Image size: 128x128**
- **Mixed precision enabled** (AMP)

Expected time per model:
- **CPU**: ~10-15 minutes per model
- **GPU**: ~2-3 minutes per model

### Step 4: Compare Results

```bash
python test_small/compare_test_models.py
```

This will generate a comprehensive comparison showing:
- Composite scores
- F1, IoU, Recall, Precision
- Best model for each metric
- Recommendations

## Manual Training (Individual Models)

If you want to train models individually:

### Faster R-CNN
```bash
python train_improved.py \
    --model faster_rcnn \
    --config configs/test_faster_rcnn.yaml \
    --device cuda  # or cpu
```

### RetinaNet
```bash
python train_improved.py \
    --model retinanet \
    --config configs/test_retinanet.yaml \
    --device cuda  # or cpu
```

### YOLOv8
```bash
python train_improved.py \
    --model yolo \
    --config configs/test_yolov8.yaml \
    --device cuda  # or cpu
```

## Configuration Files

Test configurations are located in `scripts/configs/`:
- `test_faster_rcnn.yaml` - Faster R-CNN config
- `test_retinanet.yaml` - RetinaNet config (recall-focused weights)
- `test_yolov8.yaml` - YOLOv8-nano config

### Key Settings:
```yaml
model:
  backbone: efficientnet_b0  # Lighter backbone for testing
  anchor_sizes: [16, 32, 64, 128]  # Adjusted for 128x128

training:
  img_size: 128
  batch_size: 8
  num_epochs: 2
  learning_rate: 0.001
  use_amp: true  # Mixed precision
```

## Output Locations

After training, you'll find:

```
checkpoints_test/
├── faster_rcnn/
│   ├── best_model.pth
│   ├── checkpoint_epoch_1.pth
│   ├── checkpoint_epoch_2.pth
│   ├── training_log_*.txt
│   └── training_history.json
├── retinanet/
│   └── ...
└── yolo/
    └── ...
```

## Troubleshooting

### Issue: "LMDB not found"
**Solution**: Run `python test_small/create_test_dataset.py` first

### Issue: "CUDA out of memory"
**Solution**: Use CPU mode: `python test_small/run_test_training.py` (without --gpu)

### Issue: "Model not found" or "Import error"
**Solution**: Make sure virtual environment is activated and all packages are installed:
```bash
source /home/rezab/projects/enviroments/dfu_detection_env/bin/activate
pip list | grep -E "(torch|ultralytics|albumentations)"
```

### Issue: Slow training on CPU
**Expected**: CPU training is 5-10x slower than GPU. This is normal.

## Next Steps After Test Run

Once test training completes successfully:

1. **Review test results**: `python test_small/compare_test_models.py`

2. **Run full training** on complete dataset:
   ```bash
   python train_improved.py --model retinanet --config configs/retinanet.yaml
   ```

3. **Evaluate on test set**:
   ```bash
   python evaluate.py --checkpoint checkpoints/retinanet/best_model.pth
   ```

4. **Run inference**:
   ```bash
   python inference_improved.py \
       --checkpoint checkpoints/retinanet/best_model.pth \
       --image /path/to/image.jpg
   ```

## Expected Test Results

With only 2 epochs and small dataset, don't expect high performance. Test results are just to verify everything works:

**Expected Metrics (approximate)**:
- Composite Score: 0.20 - 0.40
- F1 Score: 0.30 - 0.50
- Recall: 0.40 - 0.60
- Precision: 0.30 - 0.50

**Purpose**: Verify that:
- ✓ Data loading works
- ✓ Models initialize correctly
- ✓ Training runs without errors
- ✓ Checkpoints save properly
- ✓ All three architectures work

## Full Training Settings

For production training, use full settings:

```yaml
training:
  img_size: 512  # Higher resolution
  batch_size: 36  # Larger batch
  num_epochs: 200-300  # Many more epochs
```

See configs:
- `configs/faster_rcnn_b5.yaml` (production Faster R-CNN)
- `configs/retinanet.yaml` (recommended for recall)
- `configs/yolov8.yaml` (fastest inference)

## Resources

- **Project Documentation**: `CLAUDE.md`
- **Session Report**: `SESSION_REPORT.md`
- **Full README**: `README.md`
- **Model Selection**: `MODEL_SELECTION_GUIDE.md`

## Contact

If you encounter issues:
1. Check training logs: `cat checkpoints_test/*/training_log_*.txt`
2. Review error messages carefully
3. Verify virtual environment is activated
4. Ensure CUDA is available (for GPU training): `python -c "import torch; print(torch.cuda.is_available())"`

---

**Good luck with your test run!** 🚀
