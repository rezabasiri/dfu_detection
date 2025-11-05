# Vast.ai Cluster Setup Guide

Complete guide to set up and run DFU detection training on your Vast.ai instance.

## Your Instance Details

- **GPU**: RTX PRO 6000 WS (48GB VRAM)
- **CPU**: AMD EPYC 96-core (24 cores allocated)
- **RAM**: 256GB (allocated: ~256GB)
- **Storage**: 16GB disk
- **IP**: 198.53.64.194
- **SSH Port**: 40535
- **Jupyter Port**: 40694
- **Instance ID**: 27340991
- **CUDA**: 12.9.1
- **Base Image**: vastai/pytorch

## Option 1: Quick Setup via Jupyter (RECOMMENDED)

### Step 1: Open Jupyter Terminal

1. Go to your Vast.ai console
2. Click "Open Jupyter" button
3. In Jupyter, click "New" → "Terminal"

### Step 2: Run Setup Commands

Copy and paste these commands in the Jupyter terminal:

```bash
# Check GPU
nvidia-smi

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Install dependencies
pip install albumentations opencv-python-headless pandas scikit-learn matplotlib seaborn tqdm lmdb Pillow

# Create project directory
mkdir -p dfu_detection
cd dfu_detection
mkdir -p scripts data checkpoints results

# Verify installation
python -c "import torch, albumentations, lmdb, cv2; print('✓ All packages ready!')"

### Step 3: Transfer Files from Your Local Machine

Open a **NEW terminal on your local WSL** (keep Jupyter terminal open):

```bash
# Go to your project directory
cd ~/projects/dfu_detection

cat ~/.ssh/id_ed25519.pub 

# Transfer scripts
scp -P 45852 -r scripts/ root@93.91.156.87:/workspace/dfu_detection/

# Transfer LMDB databases (RECOMMENDED - much faster than transferring raw images)
scp -P 45852 -r data/*.lmdb root@93.91.156.87:/workspace/dfu_detection/data/

# Transfer CSV files
scp -P 45852 data/*.csv root@93.91.156.87:/workspace/dfu_detection/data/

# Check transfer size first
du -sh data/*.lmdb
# Expected: ~1.9 GB total for all LMDB files
```

**Note**: You're transferring LMDB files (~1.9GB) instead of raw images (~5GB+), saving time!

### Step 4: Start Training

Back in the **Jupyter terminal on Vast.ai**:

```bash
cd /workspace/dfu_detection/scripts

# Run training
python train_improved.py


# reverse (download)
scp -P 45852 -r root@93.91.156.87:/workspace/dfu_detection/checkpoints ./cluster_checkpoints

# With compression
scp -C -P 45852 -r root@93.91.156.87:/workspace/dfu_detection/models ./models_downloaded

# Or use rsync (recommended for >1 GB transfers)
rsync -avz -e "ssh -P 45852" root@93.91.156.87:/workspace/dfu_detection/checkpoints_b5/ ./checkpoints_b5/

```

## Option 2: Setup via SSH (Alternative)

### Step 1: Add SSH Key (if not done)

1. Go to Vast.ai Account Settings
2. Add your SSH public key from: `cat ~/.ssh/id_rsa.pub`

### Step 2: Connect via SSH

```bash
# From your local WSL terminal
ssh -p 40535 root@198.53.64.194
```

### Step 3: Follow same setup commands as Option 1

## File Transfer Details

### What to Transfer:

**Required**:
- `scripts/` folder (~10 MB) - All Python scripts
- `data/*.lmdb` folders (~1.9 GB) - LMDB databases
- `data/*.csv` files (~1 MB) - Annotation files

**Optional** (if you don't use LMDB):
- Raw images (~5 GB) - Only if you want to regenerate LMDB on cluster

**NOT needed**:
- `dfu_detection/` - Virtual environment (recreate on cluster)
- `checkpoints/` - Will be created during training
- `results/` - Will be created during inference

### Transfer Commands:

```bash
# All-in-one transfer (from local WSL)
cd ~/projects/dfu_detection

# Transfer everything needed
scp -P 40535 -r scripts data/*.lmdb data/*.csv root@198.53.64.194:/workspace/dfu_detection/

# Or transfer separately
scp -P 40535 -r scripts root@198.53.64.194:/workspace/dfu_detection/
scp -P 40535 data/train.lmdb root@198.53.64.194:/workspace/dfu_detection/data/
scp -P 40535 data/val.lmdb root@198.53.64.194:/workspace/dfu_detection/data/
scp -P 40535 data/test.lmdb root@198.53.64.194:/workspace/dfu_detection/data/
scp -P 40535 data/*.csv root@198.53.64.194:/workspace/dfu_detection/data/
```

**Note**: SCP for directories uses `-r` flag. LMDB folders are directories, not single files.

## Training Configuration for Your GPU

Your RTX PRO 6000 WS has **48GB VRAM** - you can use much larger models!

### Supported Backbones (B0-B7)

All EfficientNet backbones are now supported:
- **B0-B3**: Work on any GPU (4-10GB VRAM)
- **B4**: ~12-14GB VRAM (RECOMMENDED for your 48GB GPU!)
- **B5**: ~16-20GB VRAM (High accuracy)
- **B6**: ~24-28GB VRAM (Maximum accuracy)
- **B7**: ~32-40GB VRAM (Research/competition use)

### Recommended Settings for RTX PRO 6000 WS

Edit the main section of `train_improved.py` before transferring:

```python
# RECOMMENDED: EfficientNet-B4 (sweet spot for 48GB GPU)
model, history = train_model(
    train_csv=train_csv,
    val_csv=val_csv,
    image_folder=image_folder,  # Not used if LMDB exists
    num_epochs=100,              # More epochs for better training
    batch_size=16,               # Large batch (take advantage of 48GB!)
    learning_rate=0.001,
    img_size=640,
    backbone="efficientnet_b4",  # Much better than B3!
    device="cuda",
    checkpoint_dir="../checkpoints",
    use_amp=True,
    early_stopping_patience=15
)

# For MAXIMUM accuracy (slower but best results)
model, history = train_model(
    train_csv=train_csv,
    val_csv=val_csv,
    image_folder=image_folder,
    num_epochs=150,
    batch_size=8,
    learning_rate=0.0005,        # Lower LR for larger model
    img_size=640,
    backbone="efficientnet_b5",  # or B6 for even better accuracy
    device="cuda",
    checkpoint_dir="../checkpoints",
    use_amp=True,
    early_stopping_patience=20
)
```

See [MODEL_SELECTION_GUIDE.md](MODEL_SELECTION_GUIDE.md) for detailed comparison.

## Monitoring Training

### Option 1: Via Jupyter Terminal

Training output appears directly in the terminal. Keep it open.

### Option 2: Via Training Log

In another Jupyter terminal or SSH session:

```bash
# Watch training log in real-time
tail -f /workspace/dfu_detection/checkpoints/training_log_*.txt
```

### Option 3: Monitor GPU Usage

In another terminal:

```bash
watch -n 1 nvidia-smi
```

## Training Time Estimates

With LMDB on RTX PRO 6000 WS:
- **Per epoch**: ~2-3 minutes (vs ~10-15 min without LMDB)
- **50 epochs**: ~2-3 hours
- **100 epochs**: ~4-6 hours

## Download Results After Training

### Download Checkpoints

```bash
# From your local WSL terminal
scp -P 40535 root@198.53.64.194:/workspace/dfu_detection/checkpoints/best_model.pth ~/projects/dfu_detection/checkpoints/

# Download all checkpoints
scp -P 40535 -r root@198.53.64.194:/workspace/dfu_detection/checkpoints/ ~/projects/dfu_detection/checkpoints_cluster/
```

### Download Training Logs

```bash
scp -P 40535 root@198.53.64.194:/workspace/dfu_detection/checkpoints/training_log_*.txt ~/projects/dfu_detection/
scp -P 40535 root@198.53.64.194:/workspace/dfu_detection/checkpoints/training_history.json ~/projects/dfu_detection/
```

## Storage Considerations

Your instance has **16GB disk**. LMDB files are ~1.9GB, so you have plenty of space.

Check disk usage:
```bash
df -h
du -sh /workspace/dfu_detection
```

If running low on space:
- Remove LMDB files after loading them into memory (not recommended)
- Don't save periodic checkpoints (only best_model.pth)

## Troubleshooting

### "CUDA out of memory"

Reduce batch size in training script:
```python
batch_size=8  # Instead of 16
```

### "No space left on device"

Check disk usage:
```bash
df -h
du -sh /workspace/dfu_detection/*
```

Clean up:
```bash
# Remove unnecessary files
rm -rf /workspace/dfu_detection/checkpoints/checkpoint_epoch_*.pth
```

### "LMDB not found"

Verify LMDB transferred correctly:
```bash
ls -lh /workspace/dfu_detection/data/
```

Should see:
- `train.lmdb/` (directory)
- `val.lmdb/` (directory)
- `test.lmdb/` (directory)

If missing, transfer again or the script will fall back to raw images (which you'd need to transfer).

### Connection Lost

Vast.ai instances can disconnect. Training continues in background if you used:
- Jupyter terminal (might stop)
- SSH with tmux (recommended)

**Use tmux for long training**:
```bash
# Start tmux
tmux

# Run training
python train_improved.py

# Detach: Ctrl+B, then D
# Re-attach later: tmux attach
```

### SSH Connection Issues

If SSH fails:
1. Verify instance is "Running" in Vast.ai console
2. Check you added SSH key in account settings
3. Use Jupyter terminal instead

## Cost Monitoring

Your instance costs **$0.006/hour**:
- 1 hour: $0.006
- 10 hours: $0.06
- 50 hours: $0.30

For 100 epochs (~5 hours): **~$0.03** total!

Monitor billing in Vast.ai dashboard.

## When Done Training

### Stop Instance (Save Money!)

Once training completes and you've downloaded checkpoints:

1. Go to Vast.ai console
2. Click "Destroy" on your instance
3. You'll stop paying

### Resume Training Later

If you stopped the instance mid-training:
1. Create new instance (may be different GPU)
2. Transfer scripts + LMDB + **checkpoints**
3. Run `python train_improved.py` - it auto-resumes!

## Quick Command Summary

```bash
# === ON LOCAL MACHINE ===
# Transfer files to cluster
scp -P 40535 -r scripts data/*.lmdb data/*.csv root@198.53.64.194:/workspace/dfu_detection/

# Download checkpoints from cluster
scp -P 40535 root@198.53.64.194:/workspace/dfu_detection/checkpoints/best_model.pth ~/projects/dfu_detection/checkpoints/

# === ON VAST.AI CLUSTER (Jupyter Terminal or SSH) ===
# Install dependencies
pip install albumentations opencv-python-headless pandas scikit-learn matplotlib seaborn tqdm lmdb Pillow

# Check GPU
nvidia-smi

# Start training
cd /workspace/dfu_detection/scripts
python train_improved.py

# Monitor training
tail -f ../checkpoints/training_log_*.txt

# Monitor GPU
watch -n 1 nvidia-smi
```

## Next Steps

1. ✅ Open Jupyter terminal
2. ✅ Install dependencies
3. ✅ Transfer files from local machine
4. ✅ Start training with `python train_improved.py`
5. ✅ Monitor progress
6. ✅ Download checkpoints when done
7. ✅ Destroy instance to stop billing

Good luck with your training! 🚀
