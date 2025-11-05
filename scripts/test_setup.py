"""
Test script to verify that the setup is working correctly
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")

    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'sklearn': 'Scikit-learn',
        'albumentations': 'Albumentations',
        'cv2': 'OpenCV',
        'tqdm': 'TQDM',
        'matplotlib': 'Matplotlib'
    }

    failed = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - {e}")
            failed.append(name)

    if failed:
        print(f"\nFailed to import: {', '.join(failed)}")
        return False
    else:
        print("\nAll packages imported successfully!")
        return True

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")

    import torch

    if torch.cuda.is_available():
        print(f"  ✓ CUDA is available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return True
    else:
        print("  ✗ CUDA is NOT available")
        print("  Training will be very slow on CPU!")
        return False

def test_data_paths():
    """Test if data paths exist"""
    print("\nTesting data paths...")

    data_root = "/mnt/c/Users/90rez/OneDrive - University of Toronto/PhDUofT/SideProjects/DFU_Detection_Asem"
    csv_path = os.path.join(data_root, "groundtruth.csv")
    image_folder = os.path.join(data_root, "DFUC2022_train_images")

    if os.path.exists(csv_path):
        print(f"  ✓ CSV file found: {csv_path}")
    else:
        print(f"  ✗ CSV file NOT found: {csv_path}")
        return False

    if os.path.exists(image_folder):
        # Count images
        images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  ✓ Image folder found: {image_folder}")
        print(f"  Found {len(images)} images")
    else:
        print(f"  ✗ Image folder NOT found: {image_folder}")
        return False

    return True

def test_preprocessed_data():
    """Test if preprocessed data exists"""
    print("\nTesting preprocessed data...")

    data_dir = "../data"
    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "val.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    all_exist = True
    for csv_file, name in [(train_csv, "train.csv"), (val_csv, "val.csv"), (test_csv, "test.csv")]:
        if os.path.exists(csv_file):
            import pandas as pd
            df = pd.read_csv(csv_file)
            print(f"  ✓ {name}: {df['name'].nunique()} images, {len(df)} annotations")
        else:
            print(f"  ✗ {name} not found")
            all_exist = False

    if not all_exist:
        print(f"  ℹ Run 'python data_preprocessing.py' to create train/val/test splits")

    # Check for healthy feet integration
    healthy_train_csv = os.path.join(data_dir, "train_images.csv")
    if os.path.exists(healthy_train_csv):
        import pandas as pd
        df = pd.read_csv(healthy_train_csv)
        print(f"  ✓ train_images.csv (with healthy feet): {len(df)} total images")
    else:
        print(f"  ℹ Healthy feet not yet added. Run 'python add_healthy_feet.py' to add them.")

    return all_exist

def test_model_creation():
    """Test if we can create the model"""
    print("\nTesting model creation...")

    try:
        from train_efficientdet import create_efficientdet_model

        model = create_efficientdet_model(num_classes=2, backbone="efficientnet_b0", pretrained=False)
        print(f"  ✓ Model created successfully")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model has {total_params:,} parameters")

        return True
    except Exception as e:
        print(f"  ✗ Failed to create model: {e}")
        return False

def test_dataset():
    """Test if dataset can be loaded"""
    print("\nTesting dataset loading...")

    # Check if preprocessed data exists
    train_csv = "../data/train.csv"

    if not os.path.exists(train_csv):
        print(f"  ℹ Preprocessed data not found")
        print(f"  Run 'python data_preprocessing.py' first to create train/val/test splits")
        return True  # Not a failure, just not done yet

    try:
        from dataset import DFUDataset, get_train_transforms

        image_folder = "/mnt/c/Users/90rez/OneDrive - University of Toronto/PhDUofT/SideProjects/DFU_Detection_Asem/DFUC2022_train_images"

        dataset = DFUDataset(
            csv_file=train_csv,
            image_folder=image_folder,
            transforms=get_train_transforms(640),
            mode='train'
        )

        print(f"  ✓ Dataset loaded successfully")
        print(f"  Dataset size: {len(dataset)} images")

        # Try loading one sample
        image, target = dataset[0]
        print(f"  ✓ Sample loaded: image shape {image.shape}, {len(target['boxes'])} boxes")

        return True
    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("DFU Detection - Setup Verification")
    print("="*60)

    results = {
        "Package Imports": test_imports(),
        "CUDA Support": test_cuda(),
        "Data Paths": test_data_paths(),
        "Preprocessed Data": test_preprocessed_data(),
        "Model Creation": test_model_creation(),
        "Dataset Loading": test_dataset()
    }

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! Setup is ready.")
        print("\nNext steps:")
        print("  1. If you haven't already: python data_preprocessing.py")
        print("  2. (Optional) Add healthy feet: python add_healthy_feet.py")
        print("  3. Start training: python train_improved.py (RECOMMENDED)")
        print("     Or use original: python train_efficientdet.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("="*60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())