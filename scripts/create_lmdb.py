"""
Create LMDB databases from DFU dataset
Converts images and annotations to LMDB format for faster training

Usage:
    python create_lmdb.py

This will create:
    - ../data/train.lmdb
    - ../data/val.lmdb
    - ../data/test.lmdb
"""

import os
import lmdb
import pickle
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
import io

# Fix for corrupted PNG images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def serialize_sample(image_path: str, annotations: pd.DataFrame, img_name: str, is_healthy: bool = False):
    """
    Serialize a single image and its annotations

    Args:
        image_path: Path to the image file
        annotations: DataFrame with all annotations for this image
        img_name: Image filename
        is_healthy: Whether this is a healthy feet image (negative sample)

    Returns:
        Serialized bytes containing image and metadata
    """
    try:
        # Load and encode image as JPEG (compressed, fast to decode)
        with Image.open(image_path) as img:
            img.load()  # Verify image can be loaded
            img = img.convert('RGB')

            # Get original size
            width, height = img.size

            # Encode image to JPEG bytes for storage
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=95)
            img_bytes = img_byte_arr.getvalue()

        # Get all bounding boxes for this image
        boxes = []
        labels = []

        if is_healthy:
            # Healthy feet image - NO boxes (hard negative sample)
            # Empty boxes array will be used as-is
            # This teaches the model to reject false positives
            pass
        else:
            # DFU image - load real ulcer boxes
            for _, row in annotations.iterrows():
                boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                labels.append(1)  # Class 1: ulcer (2-class system: 0=background, 1=ulcer)

        # Create data dictionary
        data = {
            'image': img_bytes,
            'width': width,
            'height': height,
            'boxes': np.array(boxes, dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64),
            'filename': img_name
        }

        # Serialize with pickle
        return pickle.dumps(data)

    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        return None

def create_lmdb_dataset(csv_file: str, image_folder: str, lmdb_path: str,
                       image_list_csv: str = None, healthy_folder: str = None,
                       map_size: int = 1e12):
    """
    Create LMDB database from CSV and images

    Args:
        csv_file: Path to CSV with annotations
        image_folder: Path to folder with DFU images
        lmdb_path: Path where LMDB database will be created
        image_list_csv: Optional CSV with list of all images (including healthy feet)
        healthy_folder: Optional path to healthy feet images
        map_size: Maximum size of database (default 1TB)
    """
    # Load annotations
    df = pd.read_csv(csv_file)

    # Get list of images to process
    if image_list_csv and os.path.exists(image_list_csv):
        image_list_df = pd.read_csv(image_list_csv)
        image_names = image_list_df['name'].tolist()
        print(f"Processing {len(image_names)} images (including healthy feet)")

        # Identify which are DFU vs healthy
        dfu_images = set(df['name'].unique())
        num_dfu = len([img for img in image_names if img in dfu_images])
        num_healthy = len(image_names) - num_dfu
        print(f"  - {num_dfu} images with DFUs")
        print(f"  - {num_healthy} healthy feet images")
    else:
        image_names = df['name'].unique().tolist()
        dfu_images = set(image_names)
        print(f"Processing {len(image_names)} images (DFU only)")

    # Create LMDB environment
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=int(map_size))

    # Write samples to LMDB
    successful = 0
    failed = 0

    with env.begin(write=True) as txn:
        for idx, img_name in enumerate(tqdm(image_names, desc=f"Creating {os.path.basename(lmdb_path)}")):
            # Determine image path
            if img_name in dfu_images:
                img_path = os.path.join(image_folder, img_name)
            else:
                # Healthy feet image
                if healthy_folder and os.path.exists(os.path.join(healthy_folder, img_name)):
                    img_path = os.path.join(healthy_folder, img_name)
                else:
                    img_path = os.path.join(image_folder, img_name)

            # Check if image exists
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                failed += 1
                continue

            # Get annotations for this image (empty if healthy feet)
            img_annotations = df[df['name'] == img_name]
            is_healthy = img_name not in dfu_images

            # Serialize sample
            serialized = serialize_sample(img_path, img_annotations, img_name, is_healthy=is_healthy)

            if serialized is not None:
                # Store in LMDB with index as key
                key = f"{idx:08d}".encode('ascii')
                txn.put(key, serialized)
                successful += 1
            else:
                failed += 1

        # Store metadata: total number of samples and indices
        txn.put(b'__len__', str(successful).encode('ascii'))
        txn.put(b'__keys__', pickle.dumps(list(range(successful))))

        # Store indices for DFU vs healthy images (for balanced sampling)
        dfu_indices = []
        healthy_indices = []

        for idx, img_name in enumerate(image_names[:successful]):
            if img_name in dfu_images:
                dfu_indices.append(idx)
            else:
                healthy_indices.append(idx)

        txn.put(b'__dfu_indices__', pickle.dumps(dfu_indices))
        txn.put(b'__healthy_indices__', pickle.dumps(healthy_indices))

        print(f"\n  Metadata stored:")
        print(f"    DFU image indices: {len(dfu_indices)}")
        print(f"    Healthy image indices: {len(healthy_indices)}")

    env.close()

    print(f"\nLMDB created: {lmdb_path}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total size: {get_dir_size(lmdb_path) / 1024**2:.2f} MB")

def get_dir_size(path):
    """Get total size of directory"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total

if __name__ == "__main__":
    # Paths
    data_dir = "../data"
    data_root = "/mnt/c/Users/90rez/OneDrive - University of Toronto/PhDUofT/SideProjects/DFU_Detection_Asem"
    image_folder = os.path.join(data_root, "DFUC2022_train_images")
    healthy_folder = os.path.join(data_root, "HealthyFeet")

    # CSV files
    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "val.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    # Optional: healthy feet image lists
    train_image_list = os.path.join(data_dir, "train_images.csv")
    val_image_list = os.path.join(data_dir, "val_images.csv")
    test_image_list = os.path.join(data_dir, "test_images.csv")

    # LMDB output paths
    train_lmdb = os.path.join(data_dir, "train.lmdb")
    val_lmdb = os.path.join(data_dir, "val.lmdb")
    test_lmdb = os.path.join(data_dir, "test.lmdb")

    # Check if CSV files exist
    if not all(os.path.exists(f) for f in [train_csv, val_csv, test_csv]):
        print("Error: CSV files not found. Please run data_preprocessing.py first.")
        exit(1)

    print("="*60)
    print("Creating LMDB Databases for DFU Detection")
    print("="*60)

    # Check for healthy feet
    use_healthy = os.path.exists(train_image_list)
    if use_healthy:
        print("\nIncluding healthy feet images (negative samples)")
        print(f"Healthy feet folder: {healthy_folder}")
    else:
        print("\nUsing DFU images only")
        print("To include healthy feet, run: python add_healthy_feet.py")

    print("\nCreating LMDB databases...")

    # Create training LMDB
    create_lmdb_dataset(
        csv_file=train_csv,
        image_folder=image_folder,
        lmdb_path=train_lmdb,
        image_list_csv=train_image_list if use_healthy else None,
        healthy_folder=healthy_folder if use_healthy else None
    )

    # Create validation LMDB
    create_lmdb_dataset(
        csv_file=val_csv,
        image_folder=image_folder,
        lmdb_path=val_lmdb,
        image_list_csv=val_image_list if use_healthy else None,
        healthy_folder=healthy_folder if use_healthy else None
    )

    # Create test LMDB
    create_lmdb_dataset(
        csv_file=test_csv,
        image_folder=image_folder,
        lmdb_path=test_lmdb,
        image_list_csv=test_image_list if use_healthy else None,
        healthy_folder=healthy_folder if use_healthy else None
    )

    print("\n" + "="*60)
    print("LMDB Creation Complete!")
    print("="*60)
    print(f"\nCreated databases:")
    print(f"  - {train_lmdb}")
    print(f"  - {val_lmdb}")
    print(f"  - {test_lmdb}")
    print("\nYou can now use these LMDB databases for faster training!")
    print("The training script will automatically detect and use them.")