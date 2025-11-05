#!/bin/bash

# ============================================================
# DFU Detection - Dependency Installation Script
# ============================================================
# Installs Python packages required for the project
# Called automatically by Claude Web Code via SessionStart hook
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "DFU Detection - Installing Dependencies"
echo "============================================================"

# Check if running in Claude Web Code remote environment
if [ "$CLAUDE_CODE_REMOTE" = "true" ]; then
    echo "✓ Running in Claude Web Code remote environment"
else
    echo "✓ Running locally"
fi

# Navigate to project root
cd "$CLAUDE_PROJECT_DIR" || exit 1
echo "✓ Project directory: $CLAUDE_PROJECT_DIR"

# Check Python version
echo ""
echo "Checking Python version..."
python --version || python3 --version

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip || python3 -m pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt || pip3 install -r requirements.txt
    echo "✓ Requirements installed successfully"
else
    echo "⚠️  requirements.txt not found, skipping package installation"
fi

# Verify key packages
echo ""
echo "Verifying key packages..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || echo "✗ PyTorch not found"
python -c "import torchvision; print(f'✓ torchvision {torchvision.__version__}')" || echo "✗ torchvision not found"
python -c "import albumentations; print(f'✓ albumentations {albumentations.__version__}')" || echo "✗ albumentations not found"
python -c "import lmdb; print('✓ lmdb installed')" || echo "✗ lmdb not found"

echo ""
echo "============================================================"
echo "✓ Installation complete!"
echo "============================================================"

# Note: LMDB data files are NOT in the repo
echo ""
echo "⚠️  NOTE: This project requires large data files (LMDB databases)"
echo "    Data files are stored locally/cluster and NOT in Git."
echo "    See CLAUDE.md for data setup instructions."
echo ""

exit 0
