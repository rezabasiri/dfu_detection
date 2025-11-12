#!/usr/bin/env python3
"""
Create Small Test Dataset
Creates a small subset of the training and validation data for quick testing.
Extracts 80 training images and 20 validation images from LMDB databases.
"""

import lmdb
import pickle
import os
import random
from pathlib import Path
from tqdm import tqdm

def create_small_lmdb(source_lmdb, target_lmdb, num_samples, seed=42):
    """
    Create a smaller LMDB database from a larger one.

    Args:
        source_lmdb: Path to source LMDB database
        target_lmdb: Path to target LMDB database
        num_samples: Number of samples to extract
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Open source database
    source_env = lmdb.open(source_lmdb, readonly=True, lock=False)

    # Get all keys
    with source_env.begin() as txn:
        # Get metadata first
        dfu_indices_raw = txn.get(b'__dfu_indices__')
        healthy_indices_raw = txn.get(b'__healthy_indices__')

        if dfu_indices_raw and healthy_indices_raw:
            dfu_indices = pickle.loads(dfu_indices_raw)
            healthy_indices = pickle.loads(healthy_indices_raw)
            print(f"Source database has {len(dfu_indices)} DFU images and {len(healthy_indices)} healthy images")

            # Sample balanced subset (50% DFU, 50% healthy)
            num_dfu = min(num_samples // 2, len(dfu_indices))
            num_healthy = min(num_samples - num_dfu, len(healthy_indices))

            selected_dfu = random.sample(dfu_indices, num_dfu)
            selected_healthy = random.sample(healthy_indices, num_healthy)
            selected_indices = selected_dfu + selected_healthy

            print(f"Selected {len(selected_dfu)} DFU images and {len(selected_healthy)} healthy images")
        else:
            # No metadata, just sample randomly
            all_keys = [key for key in txn.cursor().iternext(keys=True, values=False)
                       if not key.startswith(b'__')]
            selected_indices = random.sample(range(len(all_keys)), min(num_samples, len(all_keys)))
            selected_dfu = []
            selected_healthy = []
            print(f"No metadata found, randomly selected {len(selected_indices)} images")

    # Create target database
    os.makedirs(os.path.dirname(target_lmdb) if os.path.dirname(target_lmdb) else '.', exist_ok=True)

    # Estimate map size (100MB should be enough for 100 images)
    map_size = 100 * 1024 * 1024  # 100MB
    target_env = lmdb.open(target_lmdb, map_size=map_size)

    # Copy selected samples
    with source_env.begin() as source_txn:
        with target_env.begin(write=True) as target_txn:
            print(f"Copying {len(selected_indices)} samples to {target_lmdb}...")

            for new_idx, old_idx in enumerate(tqdm(selected_indices)):
                old_key = str(old_idx).encode('utf-8')
                data = source_txn.get(old_key)

                if data is not None:
                    new_key = str(new_idx).encode('utf-8')
                    target_txn.put(new_key, data)

            # Save metadata
            new_dfu_indices = list(range(len(selected_dfu)))
            new_healthy_indices = list(range(len(selected_dfu), len(selected_indices)))

            target_txn.put(b'__dfu_indices__', pickle.dumps(new_dfu_indices))
            target_txn.put(b'__healthy_indices__', pickle.dumps(new_healthy_indices))
            target_txn.put(b'__length__', pickle.dumps(len(selected_indices)))

            print(f"Saved metadata: {len(new_dfu_indices)} DFU, {len(new_healthy_indices)} healthy")

    source_env.close()
    target_env.close()

    print(f"Created test database: {target_lmdb}")
    print(f"Total samples: {len(selected_indices)}")


def main():
    """Create test datasets from train and val LMDBs."""
    # Paths - go up from test_small -> scripts -> dfu_detection
    script_dir = Path(__file__).parent  # test_small
    scripts_dir = script_dir.parent      # scripts
    project_dir = scripts_dir.parent     # dfu_detection
    data_dir = project_dir / "data"

    source_train_lmdb = data_dir / "train.lmdb"
    source_val_lmdb = data_dir / "val.lmdb"

    target_train_lmdb = data_dir / "test_train.lmdb"
    target_val_lmdb = data_dir / "test_val.lmdb"

    # Check if source databases exist
    if not source_train_lmdb.exists():
        print(f"Error: Source training database not found: {source_train_lmdb}")
        print("Please ensure you have the full dataset LMDB files.")
        return

    if not source_val_lmdb.exists():
        print(f"Error: Source validation database not found: {source_val_lmdb}")
        print("Please ensure you have the full dataset LMDB files.")
        return

    # Create test datasets
    print("=" * 60)
    print("Creating Test Training Dataset (80 images)")
    print("=" * 60)
    create_small_lmdb(str(source_train_lmdb), str(target_train_lmdb), num_samples=80, seed=42)

    print("\n" + "=" * 60)
    print("Creating Test Validation Dataset (20 images)")
    print("=" * 60)
    create_small_lmdb(str(source_val_lmdb), str(target_val_lmdb), num_samples=20, seed=42)

    print("\n" + "=" * 60)
    print("Test Dataset Creation Complete!")
    print("=" * 60)
    print(f"Training LMDB: {target_train_lmdb}")
    print(f"Validation LMDB: {target_val_lmdb}")
    print("\nYou can now run the test training scripts with these datasets.")


if __name__ == "__main__":
    main()
