"""
Test script to verify LMDB dataset implementation
"""

import os
import sys

# Test LMDB creation and loading
def test_lmdb():
    print("="*60)
    print("Testing LMDB Implementation")
    print("="*60)

    # Check if lmdb is installed
    try:
        import lmdb
        print("\n✓ lmdb package is installed")
    except ImportError:
        print("\n✗ lmdb package not found. Installing...")
        os.system("pip install lmdb")
        import lmdb
        print("✓ lmdb installed successfully")

    # Check if data files exist
    data_dir = "../data"
    train_csv = os.path.join(data_dir, "train.csv")

    if not os.path.exists(train_csv):
        print(f"\n✗ {train_csv} not found. Please run data_preprocessing.py first.")
        return False

    print(f"✓ CSV files found in {data_dir}")

    # Check if LMDB already exists
    train_lmdb = os.path.join(data_dir, "train.lmdb")
    if os.path.exists(train_lmdb):
        print(f"\n✓ LMDB already exists at {train_lmdb}")
        print(f"  Testing LMDB loading...")

        # Test loading from LMDB
        from dataset import DFUDatasetLMDB, get_train_transforms

        dataset = DFUDatasetLMDB(
            lmdb_path=train_lmdb,
            transforms=get_train_transforms(512),
            mode='train'
        )

        print(f"  Dataset length: {len(dataset)}")

        # Load first sample
        image, target = dataset[0]
        print(f"  Sample 0:")
        print(f"    Image shape: {image.shape}")
        print(f"    Boxes: {len(target['boxes'])}")
        print(f"    Labels: {target['labels']}")

        print("\n✓ LMDB loading test passed!")
        return True
    else:
        print(f"\n! LMDB not found. You can create it by running:")
        print(f"  cd scripts && python create_lmdb.py")
        return False

if __name__ == "__main__":
    test_lmdb()
