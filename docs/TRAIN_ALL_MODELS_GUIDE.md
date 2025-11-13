# train_all_models.py - Complete Usage Guide

## ✅ NEW FEATURES (Just Added!)

### 1. Runtime Parameter Overrides
Override any training parameter at runtime without modifying config files!

### 2. Real-time Output Streaming
See training progress as it happens (logs stream continuously like train_improved.py)

### 3. Resume Interrupted Training
Pick up exactly where you left off if training is interrupted

---

## Quick Start

### Basic Usage (Default Configs)
```bash
cd scripts
python train_all_models.py
```

### With Parameter Overrides
```bash
# Override epochs, batch size, and image size
python train_all_models.py --epochs 50 --batch-size 16 --img-size 640

# Quick test with small parameters
python train_all_models.py --epochs 5 --batch-size 4 --img-size 256
```

### Train Specific Models
```bash
# Only Faster R-CNN
python train_all_models.py --models faster_rcnn

# Only YOLO
python train_all_models.py --models yolo --img-size 640

# Faster R-CNN and RetinaNet
python train_all_models.py --models faster_rcnn retinanet
```

---

## All Command-Line Options

### Model Selection
```bash
--models MODEL [MODEL ...]
```
**Choices**: `faster_rcnn`, `retinanet`, `yolo`
**Default**: All three models
**Examples**:
```bash
python train_all_models.py --models yolo
python train_all_models.py --models faster_rcnn retinanet
```

### Training Parameters

#### Epochs
```bash
--epochs N
```
**Applies to**: All models
**Default**: From config files (Faster R-CNN: 300, RetinaNet: 300, YOLO: 200)
**Example**:
```bash
python train_all_models.py --epochs 100
```

#### Batch Size
```bash
--batch-size N
```
**Applies to**: All models
**Default**: From config files (Faster R-CNN: 18, RetinaNet: 24, YOLO: 16)
**Example**:
```bash
python train_all_models.py --batch-size 12
```

#### Image Size
```bash
--img-size N
```
**Applies to**: YOLO only (Faster R-CNN/RetinaNet use config YAML)
**Default**: From config files (Faster R-CNN: 512, RetinaNet: 512, YOLO: 640)
**Examples**:
```bash
# YOLO: Uses 512x512 images
python train_all_models.py --models yolo --img-size 512

# Faster R-CNN: Shows warning, uses config value
python train_all_models.py --models faster_rcnn --img-size 512
# Output: "Note: --img-size ignored for faster_rcnn (set in config YAML instead)"
```

**Why different?**
- YOLO's native training interface supports runtime image size
- train_improved.py (Faster R-CNN/RetinaNet) requires image size in config YAML

#### Learning Rate
```bash
--lr RATE
```
**Applies to**: Faster R-CNN and RetinaNet only
**Default**: From config files (0.001)
**Example**:
```bash
python train_all_models.py --models faster_rcnn --lr 0.0005
```

### Device Selection
```bash
--device {cuda,cpu}
```
**Default**: cuda
**Example**:
```bash
python train_all_models.py --device cpu  # For testing without GPU
```

### Resume Training
```bash
--resume
```
**Behavior**:
- **Faster R-CNN/RetinaNet**: Automatically resumes from `resume_training.pth`
- **YOLO**: Detects `last.pt` and informs (YOLO handles resume internally)

**Example**:
```bash
# Start training
python train_all_models.py

# ... training gets interrupted (Ctrl+C or crash) ...

# Resume from checkpoint
python train_all_models.py --resume
```

**Output**:
```
Found checkpoint: ../checkpoints/faster_rcnn/resume_training.pth
FASTER_RCNN will auto-resume from checkpoint...
```

### Continue on Error
```bash
--continue-on-error
```
If one model fails, continue training the remaining models

**Example**:
```bash
python train_all_models.py --continue-on-error
```

**Use case**: If Faster R-CNN fails due to OOM, still train RetinaNet and YOLO

### Dry Run (Preview)
```bash
--dry-run
```
Show what would be executed without actually training

**Example**:
```bash
python train_all_models.py --dry-run --epochs 10 --batch-size 8 --img-size 512
```

**Output**:
```
⚠ DRY RUN - No training will be performed

Would run: python train_improved.py --model faster_rcnn --config configs/faster_rcnn_b5.yaml --epochs 10 --batch-size 8
Would run: python train_improved.py --model retinanet --config configs/retinanet.yaml --epochs 10 --batch-size 8
Would run: python train_yolo.py --model yolov8m --epochs 10 --batch-size 8 --img-size 512
```

---

## Real-World Usage Examples

### Example 1: Quick Test (5 Epochs)
```bash
python train_all_models.py --epochs 5 --batch-size 8
```
**Time**: ~2 hours for all three models
**Use**: Verify everything works before full training

### Example 2: Production Training with Custom Batch Sizes
```bash
python train_all_models.py --batch-size 12 --continue-on-error
```
**Time**: ~4 days
**Use**: Full production training with smaller batch size (if OOM occurs)

### Example 3: Train Only YOLO with Custom Image Size
```bash
python train_all_models.py --models yolo --img-size 512 --epochs 150
```
**Time**: ~10 hours
**Use**: YOLO-only training with different image size

### Example 4: Resume Interrupted Training
```bash
# Day 1: Start training
python train_all_models.py

# ... training interrupted at epoch 150 ...

# Day 2: Resume
python train_all_models.py --resume
```
**Result**: Continues from epoch 150, no progress lost

### Example 5: Faster R-CNN Only with Custom Learning Rate
```bash
python train_all_models.py --models faster_rcnn --lr 0.0005 --epochs 200
```
**Time**: ~30 hours
**Use**: Train only Faster R-CNN with lower learning rate

### Example 6: Test Configuration Before Starting
```bash
python train_all_models.py --dry-run --epochs 100 --batch-size 16 --img-size 640
```
**Time**: Instant
**Use**: Verify command-line arguments are correct before starting long training

---

## Output Display

### Before Training
```
================================================================================
DFU DETECTION - TRAIN ALL MODELS FOR COMPARISON
================================================================================

Models to train: 3
  - faster_rcnn: Faster R-CNN with EfficientNet-B5 (current production model)
  - retinanet: RetinaNet with EfficientNet-B3 (single-stage, focal loss)
  - yolo: YOLOv8m (fastest inference, anchor-free)

--------------------------------------------------------------------------------
Training Configuration
--------------------------------------------------------------------------------
Epochs:        50 (override)
Batch size:    16 (override)
Image size:    640 (override for YOLO only)

Device:         cuda
Resume:         True
Continue on error: False

================================================================================
Press Enter to start training, or Ctrl+C to cancel...
================================================================================
```

### During Training (Real-time Output)
```
================================================================================
  TRAINING: FASTER_RCNN
================================================================================
Description: Faster R-CNN with EfficientNet-B5 (current production model)
Config: configs/faster_rcnn_b5.yaml
================================================================================

Command: python train_improved.py --model faster_rcnn --config configs/faster_rcnn_b5.yaml --epochs 50

================================================================================
Training output (streaming in real-time):
================================================================================

Epoch 1/50
  Train Loss: 0.4532
  Val Loss:   0.3821
  F1:         0.7846
  IoU:        0.5627
  Composite:  0.7356
...
```

### After Completion
```
================================================================================
✓ FASTER_RCNN training completed successfully
================================================================================

================================================================================
  TRAINING: RETINANET
================================================================================
...
```

### Summary
```
================================================================================
TRAINING SUMMARY
================================================================================

✓ Successfully trained (3):
  - faster_rcnn    Time: 28h 34m
  - retinanet      Time: 22h 15m
  - yolo           Time: 12h 48m

================================================================================
NEXT STEPS
================================================================================

1. Evaluate models:
   # Faster R-CNN / RetinaNet:
   python evaluate.py --checkpoint ../checkpoints/faster_rcnn/best_model.pth
   python evaluate.py --checkpoint ../checkpoints/retinanet/best_model.pth
   # YOLO:
   python evaluate_yolo.py --model ../checkpoints/yolo/weights/best.pt

2. Compare results:
   python compare_models.py  # Compares all trained models

3. Run inference:
   python inference_improved.py --checkpoint <path> --image <image>

4. Check training logs:
   ../checkpoints/faster_rcnn/training_log_*.txt
   ../checkpoints/retinanet/training_log_*.txt
   ../checkpoints/yolo/results.csv  # YOLO training history

================================================================================
```

---

## Resume Behavior Details

### Faster R-CNN / RetinaNet
**Automatic Resume**: ✓
**Checkpoint**: `../checkpoints/<model>/resume_training.pth`
**How it works**:
1. train_improved.py automatically checks for `resume_training.pth`
2. If found, loads:
   - Model weights
   - Optimizer state
   - Learning rate scheduler
   - Current epoch number
   - Best metrics so far
3. Continues training from next epoch

**Example**:
```bash
# Training interrupted at epoch 47
python train_all_models.py --resume
# Resumes at epoch 48
```

### YOLO
**Automatic Resume**: Handled by YOLO internally
**Checkpoint**: `../checkpoints/yolo/weights/last.pt`
**How it works**:
1. train_yolo.py checks if training directory exists
2. YOLO native training automatically resumes if it finds previous training
3. train_all_models.py shows checkpoint detection message

**Example**:
```bash
# Training interrupted at epoch 73
python train_all_models.py --resume --models yolo
# YOLO automatically continues from epoch 74
```

---

## Parameter Override Priority

**Priority Order** (highest to lowest):
1. Command-line arguments (`--epochs 50`)
2. Config YAML files (`configs/*.yaml`)
3. Hardcoded defaults

**Examples**:

### Scenario 1: Override Epochs
```bash
# Config: epochs = 300
python train_all_models.py --epochs 100
# Result: Trains for 100 epochs (command-line wins)
```

### Scenario 2: Override Multiple Parameters
```bash
# Config: epochs=300, batch_size=18
python train_all_models.py --epochs 50 --batch-size 12
# Result: epochs=50, batch_size=12 (both overridden)
```

### Scenario 3: Partial Override
```bash
# Config: epochs=300, batch_size=18, lr=0.001
python train_all_models.py --epochs 100
# Result: epochs=100 (overridden), batch_size=18 (from config), lr=0.001 (from config)
```

---

## Troubleshooting

### Issue: "Training output not showing in real-time"
**Cause**: Buffering or terminal configuration
**Solution**: Output is configured for real-time streaming. If using `tee` or redirecting output, add `-u` flag:
```bash
python -u train_all_models.py 2>&1 | tee training.log
```

### Issue: "Resume doesn't work"
**Check**:
1. Checkpoint files exist:
   ```bash
   ls -lh ../checkpoints/*/resume_training.pth
   ls -lh ../checkpoints/yolo/weights/last.pt
   ```
2. Use `--resume` flag:
   ```bash
   python train_all_models.py --resume
   ```

### Issue: "--img-size ignored for Faster R-CNN"
**Expected behavior!** Faster R-CNN and RetinaNet read image size from config YAML.

**To change**:
1. Edit `configs/faster_rcnn_b5.yaml` or `configs/retinanet.yaml`
2. Find `img_size: 512` in training section
3. Change to desired value
4. Re-run training

### Issue: "Out of memory (OOM)"
**Solutions**:
```bash
# Reduce batch size
python train_all_models.py --batch-size 8

# Or for YOLO, also reduce image size
python train_all_models.py --models yolo --batch-size 8 --img-size 512
```

### Issue: "One model failed, others didn't run"
**Solution**: Use `--continue-on-error`
```bash
python train_all_models.py --continue-on-error
```

---

## Performance Tips

### Tip 1: Monitor During Training
Open separate terminal and monitor:
```bash
# Watch training logs
tail -f ../checkpoints/faster_rcnn/training_log_*.txt

# Watch GPU usage
watch -n 1 nvidia-smi
```

### Tip 2: Save Output to File
```bash
python train_all_models.py 2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log
```

### Tip 3: Run in Background (tmux/screen)
```bash
# Start tmux session
tmux new -s training

# Run training
python train_all_models.py

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

### Tip 4: Test First with Dry Run
```bash
# Always test your command first
python train_all_models.py --dry-run --epochs 100 --batch-size 16

# Then run for real
python train_all_models.py --epochs 100 --batch-size 16
```

---

## Summary of Features

| Feature | Command | Applies To | Notes |
|---------|---------|------------|-------|
| **Model Selection** | `--models` | All | Choose which models to train |
| **Override Epochs** | `--epochs N` | All | Works for all models |
| **Override Batch Size** | `--batch-size N` | All | Works for all models |
| **Override Image Size** | `--img-size N` | YOLO only | Faster R-CNN/RetinaNet use config |
| **Override Learning Rate** | `--lr RATE` | Faster R-CNN/RetinaNet | Not applicable to YOLO |
| **Device Selection** | `--device` | All | cuda or cpu |
| **Resume Training** | `--resume` | All | Auto-detects checkpoints |
| **Continue on Error** | `--continue-on-error` | All | Don't stop if one fails |
| **Dry Run** | `--dry-run` | All | Preview without training |
| **Real-time Output** | (automatic) | All | Output streams continuously |

---

## Next Steps

After training completes:
1. **Evaluate**: Run `evaluate.py` or `evaluate_yolo.py`
2. **Compare**: Use `compare_models.py` to see which performed best
3. **Inference**: Test on new images with `inference_improved.py`
4. **Deploy**: Use the best model for your application

---

**Last Updated**: 2025-11-12
**Version**: 2.0 (with runtime overrides and resume capability)
