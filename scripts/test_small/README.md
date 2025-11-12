# Test Small - Quick Model Testing

This folder contains scripts and configurations for running quick tests of all three detection models with a small dataset.

## Contents

### Python Scripts
- **`create_test_dataset.py`** - Creates small LMDB datasets (80 train + 20 val images)
- **`run_test_training.py`** - Runs all three models sequentially for testing
- **`compare_test_models.py`** - Compares results from all trained models

### Configuration Files
- **`test_faster_rcnn.yaml`** - Faster R-CNN test configuration (128x128, 2 epochs)
- **`test_retinanet.yaml`** - RetinaNet test configuration (recall-focused)
- **`test_yolov8.yaml`** - YOLOv8-nano test configuration

## Quick Start

From the `scripts` directory:

```bash
# 1. Create test dataset
python test_small/create_test_dataset.py

# 2. Run test training (all models)
python test_small/run_test_training.py         # CPU
python test_small/run_test_training.py --gpu   # GPU

# 3. Compare results
python test_small/compare_test_models.py
```

## Configuration Details

All test configurations use:
- **Image Size**: 128×128 (small for fast testing)
- **Batch Size**: 8
- **Epochs**: 2
- **Backbone**: EfficientNet-B0 (lightest)
- **Mixed Precision**: Enabled (AMP)

### Model-Specific Settings

#### Faster R-CNN
- Two-stage detector (RPN + ROI head)
- Anchor sizes: [16, 32, 64, 128]
- Balanced composite weights

#### RetinaNet
- Single-stage with focal loss
- **Recall-focused**: 50% recall weight
- Better for medical use (fewer missed detections)

#### YOLOv8-nano
- Smallest YOLO model (3.2M params)
- Anchor-free design
- Fastest inference

## Expected Results

With only 2 epochs and 80 images, expect low performance:
- Composite Score: 0.20 - 0.40
- F1 Score: 0.30 - 0.50
- Recall: 0.40 - 0.60

**Purpose**: Verify everything works, not achieve high accuracy.

## Training Times

**CPU Mode**:
- Faster R-CNN: ~15 min
- RetinaNet: ~12 min
- YOLO: ~8 min
- **Total**: ~35-45 min

**GPU Mode**:
- Faster R-CNN: ~3 min
- RetinaNet: ~2 min
- YOLO: ~1 min
- **Total**: ~6-9 min

## Output Locations

Checkpoints saved to:
```
../../checkpoints_test/
├── faster_rcnn/
│   ├── best_model.pth
│   ├── training_log_*.txt
│   └── training_history.json
├── retinanet/
│   └── ...
└── yolo/
    └── ...
```

## Troubleshooting

### "Test dataset not found"
Run: `python test_small/create_test_dataset.py`

### "CUDA out of memory"
Use CPU mode: `python test_small/run_test_training.py` (without --gpu)

### "Module not found"
Activate environment: `source /home/rezab/projects/enviroments/dfu_detection_env/bin/activate`

## After Testing

Once test training completes successfully, proceed to full training:

```bash
# Recommended: RetinaNet (best recall for medical use)
python train_improved.py --model retinanet --config configs/retinanet.yaml
```

See [TEST_RUN_INSTRUCTIONS.md](../../TEST_RUN_INSTRUCTIONS.md) for complete guide.
