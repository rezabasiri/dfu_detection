#!/bin/bash
# Pipeline Integrity Check Script
# Verifies that all components are working correctly

set -e  # Exit on error

echo "======================================================================"
echo "DFU Detection Pipeline - Integrity Check"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Required files exist
echo "Step 1: Checking required files..."
files=(
    "data_preprocessing.py"
    "add_healthy_feet.py"
    "create_lmdb.py"
    "dataset.py"
    "balanced_sampler.py"
    "train_improved.py"
    "evaluate.py"
    "verify_lmdb_data.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file - MISSING!"
        exit 1
    fi
done

echo ""

# Check 2: CSV files exist
echo "Step 2: Checking CSV files..."
cd ../data
if [ -f "train.csv" ] && [ -f "val.csv" ] && [ -f "test.csv" ]; then
    echo -e "  ${GREEN}✓${NC} Annotation CSVs found"
    echo "    - train.csv: $(wc -l < train.csv) lines"
    echo "    - val.csv: $(wc -l < val.csv) lines"
    echo "    - test.csv: $(wc -l < test.csv) lines"
else
    echo -e "  ${YELLOW}⚠${NC} Annotation CSVs not found"
    echo "    Run: python data_preprocessing.py"
    cd ../scripts
    exit 1
fi

if [ -f "train_images.csv" ] && [ -f "val_images.csv" ]; then
    echo -e "  ${GREEN}✓${NC} Image list CSVs found"
    train_count=$(tail -n +2 train_images.csv | wc -l)
    val_count=$(tail -n +2 val_images.csv | wc -l)
    echo "    - train_images.csv: $train_count images"
    echo "    - val_images.csv: $val_count images"
else
    echo -e "  ${YELLOW}⚠${NC} Image list CSVs not found"
    echo "    Run: python add_healthy_feet.py"
fi

cd ../scripts
echo ""

# Check 3: LMDB databases
echo "Step 3: Checking LMDB databases..."
cd ../data
if [ -d "train.lmdb" ] && [ -d "val.lmdb" ]; then
    echo -e "  ${GREEN}✓${NC} LMDB databases found"
    echo "    - train.lmdb: $(du -sh train.lmdb | cut -f1)"
    echo "    - val.lmdb: $(du -sh val.lmdb | cut -f1)"
    echo ""
    echo -e "  ${YELLOW}⚠${NC} IMPORTANT: If you haven't recreated LMDB after fixes,"
    echo "    your databases contain CORRUPTED data!"
    echo "    Run: cd ../scripts && python create_lmdb.py"
else
    echo -e "  ${RED}✗${NC} LMDB databases not found"
    echo "    Run: python create_lmdb.py"
fi

cd ../scripts
echo ""

# Check 4: Python imports
echo "Step 4: Checking Python dependencies..."
python3 << 'PYTHON_CHECK'
import sys

required = [
    'torch',
    'torchvision',
    'numpy',
    'pandas',
    'PIL',
    'albumentations',
    'lmdb',
    'tqdm',
    'sklearn'
]

missing = []
for package in required:
    try:
        if package == 'PIL':
            __import__('PIL')
        elif package == 'sklearn':
            __import__('sklearn')
        else:
            __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} - MISSING!")
        missing.append(package)

if missing:
    print(f"\nInstall missing packages: pip install {' '.join(missing)}")
    sys.exit(1)
PYTHON_CHECK

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""

# Summary
echo "======================================================================"
echo "Pipeline Check Summary"
echo "======================================================================"
echo ""
echo -e "${GREEN}✓ All required files present${NC}"
echo -e "${GREEN}✓ Python dependencies satisfied${NC}"
echo ""
echo "Next steps:"
echo "  1. ${YELLOW}CRITICAL${NC}: Recreate LMDB with fixed code"
echo "     ${GREEN}cd /home/rezab/projects/dfu_detection/scripts${NC}"
echo "     ${GREEN}python create_lmdb.py${NC}"
echo ""
echo "  2. Verify LMDB integrity"
echo "     ${GREEN}python verify_lmdb_data.py${NC}"
echo ""
echo "  3. Start training with metrics"
echo "     ${GREEN}python train_improved.py${NC}"
echo ""
echo "======================================================================"

