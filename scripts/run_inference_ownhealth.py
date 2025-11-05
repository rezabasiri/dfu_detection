"""
Script to run inference on randomly selected images from OwnHealth folder
OwnHealth contains patient subfolders, each with multiple images
"""

import os
import random
from pathlib import Path
import argparse
import subprocess

def collect_all_images(ownhealth_folder):
    """
    Collect all image paths from OwnHealth folder (nested in patient subfolders)

    Args:
        ownhealth_folder: Path to OwnHealth folder containing patient subfolders

    Returns:
        List of image paths with patient IDs
    """
    all_images = []

    # Walk through all patient folders
    for patient_folder in Path(ownhealth_folder).iterdir():
        if patient_folder.is_dir():
            patient_id = patient_folder.name

            # Find all images in this patient's folder
            image_files = list(patient_folder.glob('*.jpg')) + \
                         list(patient_folder.glob('*.jpeg')) + \
                         list(patient_folder.glob('*.png'))

            for img_path in image_files:
                all_images.append({
                    'path': str(img_path),
                    'patient_id': patient_id,
                    'filename': img_path.name
                })

    return all_images

def main():
    parser = argparse.ArgumentParser(description="Run inference on random OwnHealth images")
    parser.add_argument('--ownhealth-folder', type=str,
                       default='/mnt/c/Users/90rez/OneDrive - University of Toronto/PhDUofT/SideProjects/DFU_Detection_Asem/OwnHealth',
                       help='Path to OwnHealth folder')
    parser.add_argument('--num-images', type=int, default=50,
                       help='Number of random images to select (default: 50)')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints_b5/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='../results/ownhealth_predictions_b5',
                       help='Output directory for predictions')
    parser.add_argument('--confidence', type=float, default=0.8,
                       help='Confidence threshold (default: 0.8)')
    parser.add_argument('--img-size', type=int, default=512,
                       help='Input image size (default: 512)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    print("="*60)
    print("OwnHealth Image Selection & Inference")
    print("="*60)

    # Check if OwnHealth folder exists
    if not os.path.exists(args.ownhealth_folder):
        print(f"Error: OwnHealth folder not found at {args.ownhealth_folder}")
        return

    # Collect all images
    print(f"\nScanning OwnHealth folder: {args.ownhealth_folder}")
    all_images = collect_all_images(args.ownhealth_folder)

    if len(all_images) == 0:
        print("Error: No images found in OwnHealth folder")
        return

    print(f"Found {len(all_images)} total images across all patient folders")

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Randomly select images
    num_to_select = min(args.num_images, len(all_images))
    selected_images = random.sample(all_images, num_to_select)

    print(f"\nRandomly selected {num_to_select} images:")
    print("-"*60)
    for i, img_info in enumerate(selected_images, 1):
        print(f"{i:2d}. Patient {img_info['patient_id']}: {img_info['filename']}")

    # Create a temporary directory with symlinks to selected images
    # This makes it easier to pass to inference_improved.py
    temp_dir = '/tmp/ownhealth_selected'
    os.makedirs(temp_dir, exist_ok=True)

    # Clear any existing files
    for f in Path(temp_dir).glob('*'):
        f.unlink()

    # Create symlinks with patient ID in filename for traceability
    print(f"\nCreating temporary folder with selected images...")
    for img_info in selected_images:
        # Create filename with patient ID: patientID_originalname.jpg
        new_name = f"patient_{img_info['patient_id']}_{img_info['filename']}"
        symlink_path = os.path.join(temp_dir, new_name)

        # Create symlink
        if os.path.exists(symlink_path):
            os.unlink(symlink_path)
        os.symlink(img_info['path'], symlink_path)

    print(f"Created {len(selected_images)} symlinks in {temp_dir}")

    # Run inference_improved.py on the temporary directory
    print(f"\n{'='*60}")
    print("Running inference on selected images...")
    print(f"{'='*60}\n")

    cmd = [
        'python3',
        'inference_improved.py',
        '--image', temp_dir,
        '--checkpoint', args.checkpoint,
        '--output', args.output,
        '--confidence', str(args.confidence),
        '--img-size', str(args.img_size),
        '--max-images', str(num_to_select)
    ]

    subprocess.run(cmd)

    print(f"\n{'='*60}")
    print("OwnHealth inference complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
