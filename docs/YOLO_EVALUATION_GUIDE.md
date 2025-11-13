# YOLO Evaluation and Inference Guide

Quick guide for evaluating YOLO models and running inference on OwnHealth data.

---

## 🧪 Test Data Evaluation

Evaluate the best YOLO model on the test dataset to compute performance metrics.

### Script: `yolo_test_data_evaluate.py`

**Purpose**: Load best YOLO model from training and evaluate on test LMDB data.

**Metrics Computed**:
- F1 Score
- Mean IoU (Intersection over Union)
- Precision
- Recall
- Composite Score (0.40×F1 + 0.25×IoU + 0.20×Recall + 0.15×Precision)

**Basic Usage**:
```bash
cd scripts

# Evaluate with default settings (confidence 0.3)
python yolo_test_data_evaluate.py

# Use higher confidence for production evaluation
python yolo_test_data_evaluate.py --confidence 0.5

# Specify custom model path
python yolo_test_data_evaluate.py --model ../checkpoints/yolo/weights/best.pt

# Evaluate on validation set instead
python yolo_test_data_evaluate.py --test-lmdb ../data/val.lmdb
```

**Command-Line Arguments**:
```
--model           Path to YOLO weights (default: ../checkpoints/yolo/weights/best.pt)
--test-lmdb       Path to test LMDB (default: ../data/test.lmdb)
--confidence      Confidence threshold (default: 0.3)
--iou-threshold   IoU threshold for matching (default: 0.5)
--device          cuda or cpu (default: cuda)
--output          Output directory for results (default: ../results/yolo_test_evaluation)
```

**Confidence Threshold Guide**:
- **0.3**: Balanced evaluation (default for test set)
- **0.5**: Production deployment (high confidence only)
- **0.7**: Very conservative (minimal false positives)

**Output**:
- Console: Formatted metrics table
- JSON file: `test_evaluation_best_YYYYMMDD_HHMMSS.json` with complete results

**Example Output**:
```
================================================================================
TEST SET RESULTS
================================================================================

Metric               Value
--------------------------------------------------------------------------------
Composite Score       0.7542  ⭐ (Medical-optimized)
F1 Score              0.8123
Mean IoU              0.6234  (Localization quality)
Precision             0.7845  (False alarm rate)
Recall                0.8456  ⚕️ (Don't miss ulcers!)

--------------------------------------------------------------------------------
Detection Statistics:
--------------------------------------------------------------------------------
True Positives            245  ✓ Correctly detected ulcers
False Positives            67  ⚠ False alarms
False Negatives            45  ✗ Missed ulcers
Total Predictions         312
Total Ground Truth        290
================================================================================
```

---

## 🏥 OwnHealth Inference

Run YOLO inference on OwnHealth patient images and save annotated images with bounding boxes.

### Script: `yolo_inference_ownhealth.py`

**Purpose**: Process OwnHealth patient images, detect ulcers, and save visualizations.

**Features**:
- Randomly selects images from patient folders
- Draws bounding boxes with confidence scores
- Shows box area in pixels²
- Saves patient ID in output filenames
- Generates summary JSON with all predictions

**Basic Usage**:
```bash
cd scripts

# Process 50 random images with default settings
python yolo_inference_ownhealth.py

# Use higher confidence for clinical review
python yolo_inference_ownhealth.py --confidence 0.7 --num-images 100

# Specify custom model and output
python yolo_inference_ownhealth.py \
    --model ../checkpoints/yolo/weights/best.pt \
    --output ../results/ownhealth_yolo_high_conf \
    --confidence 0.8
```

**Command-Line Arguments**:
```
--ownhealth-folder   Path to OwnHealth folder (default: standard location)
--model              Path to YOLO weights (default: ../checkpoints/yolo/weights/best.pt)
--num-images         Number of random images (default: 50)
--confidence         Confidence threshold (default: 0.5)
--output             Output directory (default: ../results/ownhealth_yolo_predictions)
--seed               Random seed for reproducibility (default: 42)
--device             cuda or cpu (default: cuda)
```

**Confidence Threshold Guide**:
- **0.5**: Balanced detection (default)
- **0.7**: High confidence (fewer false positives)
- **0.8**: Very conservative (clinical screening)

**Output Structure**:
```
results/ownhealth_yolo_predictions/
├── patient_001_image1.jpg          # Annotated images
├── patient_001_image2.jpg
├── patient_002_image1.jpg
├── ...
└── inference_summary.json          # Complete results JSON
```

**Bounding Box Colors**:
- 🟢 **Green**: High confidence (≥ 0.8)
- 🟠 **Orange**: Medium confidence (0.5-0.8)
- 🔴 **Red**: Low confidence (< 0.5)

**Example Output**:
```
================================================================================
INFERENCE COMPLETE
================================================================================

Processed: 50 images
Images with detections: 23
Total detections: 31
Detection rate: 46.0%

Results saved to: ../results/ownhealth_yolo_predictions
Summary JSON: ../results/ownhealth_yolo_predictions/inference_summary.json
================================================================================

SAMPLE PREDICTIONS (Top 5 highest confidence)
================================================================================

1. Patient 027: foot_ulcer_3.jpg
   Confidence: 0.923
   Box: [234, 567, 789, 1023]
   Area: 253,020 pixels²

2. Patient 015: diabetic_foot_2.jpg
   Confidence: 0.891
   Box: [123, 345, 678, 901]
   Area: 308,580 pixels²
```

---

## 📊 Comparison with Other Models

### Test Set Evaluation

To compare YOLO with Faster R-CNN and RetinaNet:

```bash
# 1. Evaluate YOLO on test set
python yolo_test_data_evaluate.py --confidence 0.5

# 2. Evaluate Faster R-CNN on test set
python evaluate.py --checkpoint ../checkpoints/faster_rcnn/best_model.pth \
                   --val-lmdb ../data/test.lmdb \
                   --confidence 0.5

# 3. Evaluate RetinaNet on test set
python evaluate.py --checkpoint ../checkpoints/retinanet/best_model.pth \
                   --val-lmdb ../data/test.lmdb \
                   --confidence 0.5

# 4. Compare results
# Look at Composite Score, F1, Recall, and inference speed
```

### Key Metrics for Comparison

| Metric | What It Measures | Medical Priority |
|--------|------------------|------------------|
| **Composite Score** | Overall performance (medical-optimized) | ⭐⭐⭐ |
| **Recall** | Don't miss ulcers! | ⭐⭐⭐ |
| **F1 Score** | Balance of precision/recall | ⭐⭐ |
| **IoU** | Localization quality | ⭐⭐ |
| **Precision** | Minimize false alarms | ⭐ |

**Medical Context**:
- **Recall > 0.85**: Critical for clinical use (can't afford to miss ulcers)
- **Precision > 0.70**: Acceptable false positive rate
- **Composite > 0.75**: Excellent overall performance

---

## 🔄 Typical Workflow

### 1. Train Models
```bash
cd scripts
python train_all_models.py --epochs 200
```

### 2. Evaluate on Test Set
```bash
# YOLO
python yolo_test_data_evaluate.py --confidence 0.5

# Faster R-CNN
python evaluate.py --checkpoint ../checkpoints/faster_rcnn/best_model.pth \
                   --val-lmdb ../data/test.lmdb --confidence 0.5

# RetinaNet
python evaluate.py --checkpoint ../checkpoints/retinanet/best_model.pth \
                   --val-lmdb ../data/test.lmdb --confidence 0.5
```

### 3. Choose Best Model
Compare metrics and select model based on:
- Highest recall (don't miss ulcers)
- Acceptable precision (not too many false alarms)
- Inference speed (if deploying)

### 4. Run Inference on Real Data
```bash
# YOLO
python yolo_inference_ownhealth.py --confidence 0.7 --num-images 100

# Faster R-CNN/RetinaNet
python run_inference_ownhealth.py --checkpoint ../checkpoints/faster_rcnn/best_model.pth \
                                  --confidence 0.7 --num-images 100
```

### 5. Clinical Review
- Review annotated images in `results/ownhealth_*_predictions/`
- Check high-confidence detections (> 0.8) first
- Investigate low-confidence detections (< 0.6) for model improvement

---

## 📁 File Locations

### YOLO Model Checkpoints
```
checkpoints/yolo/
├── weights/
│   ├── best.pt              # Best model (use for evaluation/inference)
│   └── last.pt              # Last checkpoint (use for resuming training)
├── results.csv              # Training metrics
└── results.png              # Training curves
```

### Test Scripts
```
scripts/
├── yolo_test_data_evaluate.py      # YOLO test evaluation
├── yolo_inference_ownhealth.py     # YOLO OwnHealth inference
├── evaluate.py                     # Faster R-CNN/RetinaNet evaluation
└── run_inference_ownhealth.py      # Faster R-CNN/RetinaNet inference
```

### Data Files
```
data/
├── train.lmdb               # Training data
├── val.lmdb                 # Validation data
└── test.lmdb                # Test data (for final evaluation)
```

### Results
```
results/
├── yolo_test_evaluation/
│   └── test_evaluation_best_*.json
└── ownhealth_yolo_predictions/
    ├── patient_*_*.jpg      # Annotated images
    └── inference_summary.json
```

---

## 🐛 Troubleshooting

### Model Not Found
```
ERROR: Model not found: ../checkpoints/yolo/weights/best.pt
```
**Solution**: Train YOLO first with `python train_all_models.py --models yolo`

### Test LMDB Not Found
```
ERROR: Test LMDB not found: ../data/test.lmdb
```
**Solution**: Create test LMDB with `python create_lmdb.py` (ensure test split exists)

### OwnHealth Folder Not Found
```
ERROR: OwnHealth folder not found
```
**Solution**:
- Update `--ownhealth-folder` path
- Or create symlink: `ln -s /actual/path /mnt/c/.../OwnHealth`

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Use CPU instead: `--device cpu`

---

## 💡 Tips

### For Test Evaluation
1. **Use confidence 0.3-0.5** for balanced test evaluation
2. **Compare all three models** with same confidence threshold
3. **Focus on recall** for medical applications
4. **Save results** to JSON for later comparison

### For OwnHealth Inference
1. **Start with confidence 0.5** to see what the model detects
2. **Increase to 0.7-0.8** for clinical review (high confidence only)
3. **Check sample predictions** in console before reviewing all images
4. **Use consistent random seed** (`--seed 42`) for reproducibility

### For Model Selection
1. **YOLO**: Fastest inference, good for real-time applications
2. **Faster R-CNN**: Highest accuracy, slower inference
3. **RetinaNet**: Good balance, naturally high recall (focal loss)

---

**Last Updated**: 2025-11-13
**Author**: Reza (with Claude Code assistance)
