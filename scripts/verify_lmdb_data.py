"""
LMDB Data Verification Script
Verifies that LMDB databases contain correct data:
- Healthy images have NO boxes
- DFU images have boxes with label=1
- Images and boxes are correctly matched
- Visualizes samples for manual inspection
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset import DFUDatasetLMDB, get_val_transforms

def denormalize_image(image_tensor):
    """Convert normalized tensor back to displayable numpy array"""
    if image_tensor.dtype == torch.uint8:
        img = image_tensor.numpy()
    else:
        # Denormalize from [0, 1] to [0, 255]
        img = (image_tensor.clamp(0, 1) * 255).byte().numpy()
    
    # CHW -> HWC
    if len(img.shape) == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    
    return img

def visualize_sample(image, target, idx, title="Sample"):
    """Visualize image with bounding boxes"""
    img = denormalize_image(image)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img)
    
    boxes = target['boxes'].numpy() if torch.is_tensor(target['boxes']) else target['boxes']
    labels = target['labels'].numpy() if torch.is_tensor(target['labels']) else target['labels']
    
    # Title with box info
    if len(boxes) > 0:
        ax.set_title(f"{title} #{idx} - {len(boxes)} boxes (labels: {labels.tolist()})", fontsize=14, weight='bold')
        
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            
            # Red for ulcer (class 1), blue for others (shouldn't happen)
            color = 'red' if label == 1 else 'blue'
            
            rect = patches.Rectangle(
                (xmin, ymin), width, height,
                linewidth=3, edgecolor=color, facecolor='none', linestyle='-'
            )
            ax.add_patch(rect)
            
            # Label text
            ax.text(xmin, ymin - 10, f'Class {label}', 
                   color='white', fontsize=12, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.9))
    else:
        ax.set_title(f"{title} #{idx} - NO BOXES (Healthy Image)", fontsize=14, weight='bold', color='green')
    
    ax.axis('off')
    plt.tight_layout()
    return fig

def verify_lmdb(lmdb_path, mode='train', num_samples=20, save_dir='lmdb_verification'):
    """
    Verify LMDB database integrity
    
    Args:
        lmdb_path: Path to LMDB database
        mode: Dataset mode  (train/val/test)
        num_samples: Number of samples to check
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*70)
    print(f"LMDB DATA VERIFICATION - {mode.upper()}")
    print("="*70)
    print(f"Database: {lmdb_path}\n")
    
    # Load dataset WITHOUT augmentations
    try:
        dataset = DFUDatasetLMDB(
            lmdb_path=lmdb_path,
            transforms=get_val_transforms(640),  # No augmentation for verification
            mode=mode
        )
    except Exception as e:
        print(f"❌ Error loading LMDB: {e}")
        return False
    
    print(f"\nChecking {num_samples} samples from {len(dataset)} total...\n")
    
    # Statistics
    stats = {
        'total_checked': 0,
        'with_boxes': 0,
        'without_boxes': 0,
        'label_counts': {},
        'invalid_labels': [],
        'errors': []
    }
    
    # Check random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        try:
            image, target = dataset[idx]
            
            boxes = target['boxes']
            labels = target['labels']
            
            stats['total_checked'] += 1
            
            # Check boxes
            if len(boxes) > 0:
                stats['with_boxes'] += 1
                
                # Count labels
                for label in labels:
                    label_val = label.item() if torch.is_tensor(label) else int(label)
                    stats['label_counts'][label_val] = stats['label_counts'].get(label_val, 0) + 1
                    
                    # Check for invalid labels
                    if label_val != 1:
                        stats['invalid_labels'].append((idx, label_val))
                
                print(f"✓ Sample {idx}: {len(boxes)} boxes, labels={labels.tolist()}")
            else:
                stats['without_boxes'] += 1
                print(f"✓ Sample {idx}: NO boxes (healthy image)")
            
            # Visualize
            fig = visualize_sample(image, target, idx, title=f"{mode.upper()}")
            save_path = os.path.join(save_dir, f"{mode}_sample_{i:03d}_idx{idx}.png")
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            stats['errors'].append((idx, str(e)))
            print(f"❌ Sample {idx}: ERROR - {e}")
    
    # Print summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"Samples checked: {stats['total_checked']}")
    print(f"  With boxes (DFU): {stats['with_boxes']}")
    print(f"  Without boxes (healthy): {stats['without_boxes']}")
    
    if stats['label_counts']:
        print(f"\nLabel distribution:")
        for label, count in sorted(stats['label_counts'].items()):
            label_name = "ulcer" if label == 1 else f"INVALID_{label}"
            symbol = "✓" if label == 1 else "❌"
            print(f"  {symbol} Class {label} ({label_name}): {count} boxes")
    
    # Check for issues
    print("\n" + "="*70)
    print("INTEGRITY CHECKS")
    print("="*70)
    
    passed = True
    
    # Check 1: Labels should only be 1
    if stats['invalid_labels']:
        print(f"❌ FAILED: Found {len(stats['invalid_labels'])} invalid labels!")
        print(f"   Expected only label=1 for ulcer.")
        print(f"   Invalid labels found: {set(l for _, l in stats['invalid_labels'])}")
        passed = False
    else:
        if stats['with_boxes'] > 0:
            print("✅ PASSED: All DFU boxes have correct label (label=1)")
    
    # Check 2: Should have healthy images
    if stats['without_boxes'] == 0:
        print(f"⚠️  WARNING: No healthy images found in sample!")
        print(f"    This might be normal if dataset has few healthy images.")
    else:
        pct = 100 * stats['without_boxes'] / stats['total_checked']
        print(f"✅ PASSED: Found {stats['without_boxes']} healthy images ({pct:.1f}%)")
    
    # Check 3: Should have DFU images
    if stats['with_boxes'] == 0:
        print(f"❌ FAILED: No DFU images found!")
        passed = False
    else:
        pct = 100 * stats['with_boxes'] / stats['total_checked']
        print(f"✅ PASSED: Found {stats['with_boxes']} DFU images ({pct:.1f}%)")
    
    # Check 4: Errors
    if stats['errors']:
        print(f"❌ FAILED: Encountered {len(stats['errors'])} errors loading data")
        for idx, err in stats['errors'][:3]:
            print(f"   Sample {idx}: {err}")
        passed = False
    else:
        print(f"✅ PASSED: No errors loading samples")
    
    # Final verdict
    print("\n" + "="*70)
    if passed:
        print("✅ VERIFICATION PASSED!")
        print(f"   LMDB database appears correct.")
        print(f"   Visualizations saved to: {save_dir}")
    else:
        print("❌ VERIFICATION FAILED!")
        print(f"   LMDB database has issues.")
        print(f"   Please recreate with: python create_lmdb.py")
    print("="*70 + "\n")
    
    return passed

if __name__ == "__main__":
    data_dir = "../data"
    
    # Verify train set
    print("\n" + "="*70)
    print("VERIFYING TRAINING SET")
    print("="*70)
    train_lmdb = os.path.join(data_dir, "train.lmdb")
    if os.path.exists(train_lmdb):
        train_passed = verify_lmdb(
            lmdb_path=train_lmdb,
            mode='train',
            num_samples=20,
            save_dir='../lmdb_verification_train'
        )
    else:
        print(f"❌ Train LMDB not found: {train_lmdb}")
        print("   Create it with: python create_lmdb.py")
        train_passed = False
    
    # Verify val set
    print("\n" + "="*70)
    print("VERIFYING VALIDATION SET")
    print("="*70)
    val_lmdb = os.path.join(data_dir, "val.lmdb")
    if os.path.exists(val_lmdb):
        val_passed = verify_lmdb(
            lmdb_path=val_lmdb,
            mode='val',
            num_samples=20,
            save_dir='../lmdb_verification_val'
        )
    else:
        print(f"❌ Val LMDB not found: {val_lmdb}")
        print("   Create it with: python create_lmdb.py")
        val_passed = False
    
    # Overall result
    print("\n" + "="*70)
    print("OVERALL RESULT")
    print("="*70)
    if train_passed and val_passed:
        print("✅ ALL VERIFICATIONS PASSED!")
        print("   Your LMDB databases are ready for training.")
    else:
        print("❌ VERIFICATION FAILED!")
        print("   Please fix the issues and recreate LMDB databases.")
    print("="*70)
    
    sys.exit(0 if (train_passed and val_passed) else 1)
