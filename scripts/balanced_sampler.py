"""
Balanced Batch Sampler for DFU Detection
Ensures each batch contains a mix of DFU images (with boxes) and healthy/hard negative images
This prevents training instability from batches with only empty boxes.
"""

import torch
from torch.utils.data.sampler import Sampler
import numpy as np
from typing import Iterator, List


class BalancedBatchSampler(Sampler):
    """
    Custom batch sampler that ensures each batch contains:
    - At least min_dfu_per_batch images with DFU boxes
    - Remaining slots filled with healthy/hard negative images

    This prevents batches with only empty boxes which can cause training instability.
    """

    def __init__(
        self,
        data_source,
        batch_size: int,
        min_dfu_per_batch: int = 2,
        drop_last: bool = False
    ):
        """
        Args:
            data_source: The dataset
            batch_size: Total batch size
            min_dfu_per_batch: Minimum number of DFU images (with boxes) per batch
            drop_last: Whether to drop the last incomplete batch
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.min_dfu_per_batch = min_dfu_per_batch
        self.drop_last = drop_last

        if min_dfu_per_batch >= batch_size:
            raise ValueError(f"min_dfu_per_batch ({min_dfu_per_batch}) must be < batch_size ({batch_size})")

        # Identify DFU vs healthy images
        self.dfu_indices = []
        self.healthy_indices = []

        print("\nAnalyzing dataset for balanced sampling...")

        # Check if dataset has 'image_names' and 'dfu_images' attributes
        if hasattr(data_source, 'image_names') and hasattr(data_source, 'dfu_images') and hasattr(data_source, 'df'):
            # Regular DFUDataset
            for idx in range(len(data_source)):
                img_name = data_source.image_names[idx]
                is_healthy = img_name not in data_source.dfu_images

                if is_healthy:
                    self.healthy_indices.append(idx)
                else:
                    # Check if this DFU image actually has boxes in the CSV
                    img_annotations = data_source.df[data_source.df['name'] == img_name]
                    if len(img_annotations) > 0:
                        self.dfu_indices.append(idx)
                    else:
                        # DFU image with no boxes - treat as hard negative
                        self.healthy_indices.append(idx)
        else:
            # LMDB dataset - check for pre-computed metadata
            print("  LMDB dataset detected - loading metadata...")

            # Try to load pre-computed indices from LMDB metadata
            try:
                import pickle

                # Use the safe get_metadata method (works with worker-safe LMDB)
                dfu_indices_bytes = data_source.get_metadata(b'__dfu_indices__')
                healthy_indices_bytes = data_source.get_metadata(b'__healthy_indices__')

                if dfu_indices_bytes and healthy_indices_bytes:
                    self.dfu_indices = pickle.loads(dfu_indices_bytes)
                    self.healthy_indices = pickle.loads(healthy_indices_bytes)
                    print("  ✓ Loaded pre-computed indices from LMDB metadata")
                else:
                    raise ValueError("Metadata not found")

            except Exception as e:
                # Fallback: categorize by checking targets
                print(f"  ⚠️  Metadata not found ({e}), categorizing images...")
                print("     This may take a moment. Consider recreating LMDB with: python create_lmdb.py")

                for idx in range(len(data_source)):
                    try:
                        _, target = data_source[idx]
                        labels = target['labels']
                        # If has class 1 (ulcer) boxes, it's DFU
                        if len(labels) > 0 and torch.any(labels == 1):
                            self.dfu_indices.append(idx)
                        else:
                            self.healthy_indices.append(idx)
                    except:
                        # If we can't load it, treat as healthy/negative
                        self.healthy_indices.append(idx)

                    # Progress indicator every 500 images
                    if (idx + 1) % 500 == 0:
                        print(f"      Processed {idx + 1}/{len(data_source)} images...")

        print(f"  DFU images with boxes: {len(self.dfu_indices)}")
        print(f"  Healthy/negative images: {len(self.healthy_indices)}")
        print(f"  Batch composition: {min_dfu_per_batch} DFU + {batch_size - min_dfu_per_batch} healthy per batch")

        if len(self.dfu_indices) == 0:
            raise ValueError("No DFU images with boxes found!")

        # Calculate number of batches
        self.num_batches = len(self.dfu_indices) // min_dfu_per_batch

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with balanced DFU/healthy mix"""

        # Shuffle indices
        dfu_perm = np.random.permutation(self.dfu_indices).tolist()
        healthy_perm = np.random.permutation(self.healthy_indices).tolist()

        # Generate batches
        dfu_idx = 0
        healthy_idx = 0

        for _ in range(self.num_batches):
            batch = []

            # Add min_dfu_per_batch DFU images
            for _ in range(self.min_dfu_per_batch):
                if dfu_idx < len(dfu_perm):
                    batch.append(dfu_perm[dfu_idx])
                    dfu_idx += 1

            # Fill remaining slots with healthy images
            num_healthy_needed = self.batch_size - len(batch)
            for _ in range(num_healthy_needed):
                if healthy_idx < len(healthy_perm):
                    batch.append(healthy_perm[healthy_idx])
                    healthy_idx += 1
                else:
                    # Wrap around if we run out of healthy images
                    healthy_idx = 0
                    if len(healthy_perm) > 0:
                        batch.append(healthy_perm[healthy_idx])
                        healthy_idx += 1

            # Shuffle within batch for variety
            np.random.shuffle(batch)

            # Yield the batch as a list (batch_sampler yields batches, not individual indices)
            yield batch

    def __len__(self) -> int:
        """Return number of batches"""
        return self.num_batches
