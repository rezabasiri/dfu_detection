"""
Custom PyTorch Dataset for DFU Detection
Supports both object detection and classification tasks
Includes LMDB-based dataset for faster loading
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image, ImageFile, PngImagePlugin
import os
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import lmdb
import pickle
import io

# Fix for corrupted PNG images with large iCCP chunks
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)  # 100MB

class DFUDataset(Dataset):
    """
    Dataset class for Diabetic Foot Ulcer detection
    
    Args:
        csv_file: Path to CSV file with annotations
        image_folder: Path to folder containing images
        transforms: Albumentations transforms
        mode: 'train', 'val', or 'test'
    """
    
    def __init__(
        self,
        csv_file: str,
        image_folder: str,
        transforms: Optional[A.Compose] = None,
        mode: str = 'train',
        image_list_csv: Optional[str] = None,
        healthy_folder: Optional[str] = None
    ):
        # Load annotations (bounding boxes)
        self.df = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.healthy_folder = healthy_folder
        self.transforms = transforms
        self.mode = mode

        # If image list is provided (includes healthy feet), use it
        # Otherwise, use only images from annotation CSV
        if image_list_csv and os.path.exists(image_list_csv):
            image_list_df = pd.read_csv(image_list_csv)
            self.image_names = image_list_df['name'].tolist()
            print(f"Loaded {len(self.image_names)} images for {mode} set (including healthy feet)")

            # Count how many have DFUs vs healthy
            dfu_images = set(self.df['name'].unique())
            num_dfu = len([img for img in self.image_names if img in dfu_images])
            num_healthy = len(self.image_names) - num_dfu
            print(f"  - {num_dfu} images with DFUs")
            print(f"  - {num_healthy} healthy feet images")

            # Store which images are DFU vs healthy for path lookup
            self.dfu_images = dfu_images
        else:
            # Original behavior: only images with annotations
            self.image_names = self.df['name'].unique().tolist()
            self.dfu_images = set(self.image_names)
            print(f"Loaded {len(self.image_names)} images for {mode} set")
    
    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_name = self.image_names[idx]
        is_healthy = img_name not in self.dfu_images

        # Check if this is a DFU image or healthy feet image
        if img_name in self.dfu_images:
            # DFU image - load from main image folder
            img_path = os.path.join(self.image_folder, img_name)
        else:
            # Healthy feet image - load from healthy folder
            if self.healthy_folder and os.path.exists(os.path.join(self.healthy_folder, img_name)):
                img_path = os.path.join(self.healthy_folder, img_name)
            else:
                # Fallback to main folder
                img_path = os.path.join(self.image_folder, img_name)

        # Load image with error handling for corrupted images
        try:
            # Some PNG files have corrupted iCCP chunks - ignore them
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True

            image = Image.open(img_path)
            # Verify the image can be loaded
            image.load()
            image = image.convert("RGB")
            image = np.array(image)
        except Exception as e:
            # If image is corrupted, print warning and create a black dummy image
            print(f"Warning: Could not load {img_name}: {e}. Using dummy image.")
            image = np.zeros((480, 640, 3), dtype=np.uint8)  # Black dummy image

        # Get all annotations for this image
        img_annotations = self.df[self.df['name'] == img_name]

        boxes = []
        labels = []

        if is_healthy:
            # Healthy feet image - no boxes (hard negative sample)
            # Empty boxes will be handled below - helps model learn to reject false positives
            pass
        else:
            # DFU image - load real ulcer boxes
            for _, row in img_annotations.iterrows():
                xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)  # Class 1: DFU ulcer (2-class: background=0, ulcer=1)

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)
        
        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        elif image.dtype == torch.uint8:
            # ToTensorV2 returns uint8, convert to float
            image = image.float() / 255.0
        
        # Handle case with no boxes after transforms
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dict (PyTorch format)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros(0),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }
        
        return image, target

class DFUDatasetLMDB(Dataset):
    """
    LMDB-based Dataset for Diabetic Foot Ulcer detection
    Much faster than loading images from disk

    This implementation is WORKER-SAFE for PyTorch DataLoader multiprocessing.
    Each worker process initializes its own LMDB environment and transactions.

    Args:
        lmdb_path: Path to LMDB database
        transforms: Albumentations transforms
        mode: 'train', 'val', or 'test'
    """

    def __init__(
        self,
        lmdb_path: str,
        transforms: Optional[A.Compose] = None,
        mode: str = 'train'
    ):
        self.lmdb_path = lmdb_path
        self.transforms = transforms
        self.mode = mode

        # CRITICAL: Don't open LMDB here - will be opened per worker in __getitem__
        # This prevents "Transaction object not subscriptable" errors
        self.env = None
        self._worker_id = None

        # Get dataset length by temporarily opening LMDB
        temp_env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        with temp_env.begin(write=False) as txn:
            self.length = int(txn.get(b'__len__').decode('ascii'))
        temp_env.close()

        print(f"Loaded LMDB dataset from {lmdb_path}")
        print(f"  Mode: {mode}")
        print(f"  Samples: {self.length}")

    def _init_lmdb(self):
        """
        Initialize LMDB environment for the current worker process.
        This is called lazily in __getitem__ to ensure each worker gets its own connection.
        """
        import torch.utils.data

        # Get current worker info
        worker_info = torch.utils.data.get_worker_info()
        current_worker_id = worker_info.id if worker_info is not None else None

        # Initialize LMDB if not done yet or if worker changed
        if self.env is None or self._worker_id != current_worker_id:
            # Close old environment if exists
            if self.env is not None:
                self.env.close()

            # Open new environment for this worker
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
            self._worker_id = current_worker_id

    def __len__(self) -> int:
        return self.length

    def get_metadata(self, key: bytes):
        """
        Get metadata from LMDB (for use in main process, e.g., by balanced sampler).
        Opens a temporary connection to read metadata.

        Args:
            key: Metadata key (e.g., b'__dfu_indices__')

        Returns:
            Metadata bytes or None if not found
        """
        temp_env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        with temp_env.begin(write=False) as txn:
            result = txn.get(key)
        temp_env.close()
        return result

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        # Ensure LMDB is initialized for this worker
        self._init_lmdb()

        # Open transaction
        with self.env.begin(write=False) as txn:
            # Get serialized data
            key = f"{idx:08d}".encode('ascii')
            data_bytes = txn.get(key)

            if data_bytes is None:
                raise ValueError(f"Key {idx} not found in LMDB")

            # Deserialize
            data = pickle.loads(data_bytes)

        # Decode image from JPEG bytes
        image_bytes = data['image']
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        image = np.array(image)

        # Get boxes and labels
        boxes = data['boxes'].copy()
        labels = data['labels'].copy()

        # Check if this is a healthy feet image (no boxes originally)
        was_healthy = len(boxes) == 0

        # Apply transforms
        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)

        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        elif image.dtype == torch.uint8:
            # ToTensorV2 returns uint8, convert to float
            image = image.float() / 255.0

        # Handle case with no boxes after transforms
        if len(boxes) == 0:
            # Empty boxes (healthy images or DFU images that lost boxes during augmentation)
            # Use as hard negative samples
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Labels already set correctly in LMDB (1 for ulcer in 2-class system)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create target dict (PyTorch format)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros(0),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }

        return image, target

    def __del__(self):
        """Close LMDB environment when dataset is deleted"""
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()

def get_train_transforms(img_size: int = 640) -> A.Compose:
    """
    Get training augmentation transforms with strong augmentations
    Designed for medical imaging with realistic variations

    NO CROP AUGMENTATIONS - to avoid losing bounding boxes

    Args:
        img_size: Target image size

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        # Resize and pad (always applied) - NO CROPPING
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),

        # Geometric transforms - FURTHER REDUCED to preserve boxes
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
        A.Perspective(scale=(0.05, 0.08), p=0.2),  # REDUCED probability
        A.Affine(
            scale=(0.95, 1.05),  # FURTHER REDUCED - minimal zoom
            translate_percent=(-0.05, 0.05),  # FURTHER REDUCED - minimal shift
            rotate=(-40, 40),
            p=0.3  # REDUCED from 0.5
        ),

        # Color augmentations (strong - simulates different cameras, lighting, skin tones)
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        ], p=0.4),

        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.3
        ),

        A.OneOf([
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),  # Exposure variation
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),  # Local contrast enhancement
        ], p=0.2),

        # Lighting and shadow effects (clinical vs home photos)
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 2),
            shadow_dimension=5,
            p=0.3
        ),

        # Blur and noise (camera quality, focus issues)
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.5),

        # Image quality degradation
        A.OneOf([
            A.Downscale(scale_range=(0.5, 0.75), p=1.0),
            A.ImageCompression(quality_range=(60, 90), p=1.0),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),  # Over-sharpened
        ], p=0.4),

        # Occlusions (shadows, dressings, partial visibility)
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(int(img_size * 0.05), int(img_size * 0.15)),
            hole_width_range=(int(img_size * 0.05), int(img_size * 0.15)),
            fill=0,
            p=0.05  # VERY LOW probability to avoid losing boxes
        ),
        
        # CROP AUGMENTATIONS - Conservative BBoxSafeRandomCrop only
        # BBoxSafeRandomCrop is the safest option - ensures ALL boxes remain after crop
        # Using very low probability (5%) and low erosion_rate (0.05) to minimize risk
        # If NaN losses return, disable this by setting p=0.0
        A.BBoxSafeRandomCrop(
            erosion_rate=0.05,  # Very conservative - minimal cropping
            p=0.05  # Only 5% of images (reduced from 20%)
        ),

        # Must resize and pad AFTER crop to ensure all images are same size
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),

        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3,  # Keep boxes if at least 30% visible (increased from 0.1 to prevent NaN)
        min_area=100.0  # Reject boxes smaller than 100 pixels (increased from 50 for stability)
    ))

def get_val_transforms(img_size: int = 640) -> A.Compose:
    """
    Get validation transforms (no augmentation)
    
    Args:
        img_size: Target image size
        
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def collate_fn(batch):
    """
    Custom collate function for DataLoader
    Handles batches with varying number of boxes
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    
    return images, targets

if __name__ == "__main__":
    # Test the dataset
    data_root = "/mnt/c/Users/90rez/OneDrive - University of Toronto/PhDUofT/SideProjects/DFU_Detection_Asem"
    image_folder = os.path.join(data_root, "DFUC2022_train_images")
    
    # Test with train.csv if it exists
    train_csv = "../data/train.csv"
    if os.path.exists(train_csv):
        print("Testing dataset...")
        dataset = DFUDataset(
            csv_file=train_csv,
            image_folder=image_folder,
            transforms=get_train_transforms(),
            mode='train'
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        # Get first sample
        image, target = dataset[0]
        print(f"\nSample 0:")
        print(f"  Image shape: {image.shape}")
        print(f"  Number of boxes: {len(target['boxes'])}")
        print(f"  Boxes: {target['boxes']}")
        print(f"  Labels: {target['labels']}")
        
        print("\nDataset test successful!")
    else:
        print(f"Please run data_preprocessing.py first to create {train_csv}")
