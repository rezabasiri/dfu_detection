"""
Improved Inference script for DFU detection
Shows confidence scores and bounding box pixel areas
Saves top 5 predictions
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from train_efficientdet import create_efficientdet_model

def get_inference_transform(img_size: int = 640):
    """Get inference transforms"""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
        ToTensorV2()
    ])

@torch.no_grad()
def predict(
    model,
    image_path: str,
    device,
    confidence_threshold: float = 0.5,
    img_size: int = 480
):
    """
    Run inference on a single image

    Args:
        model: Trained model
        image_path: Path to image
        device: Device to run on
        confidence_threshold: Minimum confidence for detections
        img_size: Input image size

    Returns:
        Dictionary with predictions (boxes in original image coordinates)
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)
    original_width, original_height = original_size
    image_np = np.array(image)

    transform = get_inference_transform(img_size)
    transformed = transform(image=image_np)
    image_tensor = transformed['image']

    # Get the actual transformed image size (after padding)
    transformed_height, transformed_width = image_tensor.shape[1], image_tensor.shape[2]

    # Convert to float if uint8
    if image_tensor.dtype == torch.uint8:
        image_tensor = image_tensor.float() / 255.0

    image_tensor = image_tensor.unsqueeze(0).to(device)

    model.eval()
    predictions = model(image_tensor)[0]

    # Filter by confidence threshold
    mask = predictions['scores'] >= confidence_threshold

    # UPDATED: For 3-class models, filter out class 1 (healthy) predictions
    # Only keep class 2 (ulcer) predictions
    if 'labels' in predictions and len(predictions['labels']) > 0:
        # Check if this is a 3-class model by looking at max label
        max_label = predictions['labels'].max().item()
        if max_label >= 2:
            # 3-class model: only keep class 2 (ulcer) predictions
            label_mask = predictions['labels'] == 2
            mask = mask & label_mask

    boxes = predictions['boxes'][mask].cpu().numpy()
    scores = predictions['scores'][mask].cpu().numpy()
    labels = predictions['labels'][mask].cpu().numpy()

    # Calculate scale factors to map from transformed coords to original coords
    # LongestMaxSize keeps aspect ratio, so we need to figure out the actual scaling
    original_aspect = original_width / original_height

    if original_width > original_height:
        # Width is the longest side
        scale = original_width / img_size
        scaled_height = original_height / scale
        pad_top = (img_size - scaled_height) / 2
        pad_left = 0
    else:
        # Height is the longest side
        scale = original_height / img_size
        scaled_width = original_width / scale
        pad_left = (img_size - scaled_width) / 2
        pad_top = 0

    # Scale boxes back to original image coordinates
    boxes_original = []
    areas = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box

        # Remove padding offset
        xmin = xmin - pad_left
        ymin = ymin - pad_top
        xmax = xmax - pad_left
        ymax = ymax - pad_top

        # Scale back to original size
        xmin = xmin * scale
        ymin = ymin * scale
        xmax = xmax * scale
        ymax = ymax * scale

        # Clip to image boundaries
        xmin = max(0, min(xmin, original_width))
        ymin = max(0, min(ymin, original_height))
        xmax = max(0, min(xmax, original_width))
        ymax = max(0, min(ymax, original_height))

        boxes_original.append([xmin, ymin, xmax, ymax])

        # Calculate area in original image coordinates
        area = int((xmax - xmin) * (ymax - ymin))
        areas.append(area)

    return {
        'boxes': np.array(boxes_original) if len(boxes_original) > 0 else boxes,
        'scores': scores,
        'labels': labels,
        'areas': areas,
        'original_size': original_size
    }

def visualize_predictions(
    image_path: str,
    predictions: dict,
    output_path: str,
    class_names: dict = None
):
    """
    Visualize predictions on image with confidence and area

    Args:
        image_path: Path to original image
        predictions: Prediction dictionary
        output_path: Path to save visualization
        class_names: Dictionary mapping class IDs to names
    """
    if class_names is None:
        # Support both 2-class and 3-class models
        class_names = {1: 'DFU', 2: 'DFU'}  # Map both to DFU for backward compatibility

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        # font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()
        font_small = font

    for box, score, label, area in zip(predictions['boxes'], predictions['scores'],
                                       predictions['labels'], predictions['areas']):
        xmin, ymin, xmax, ymax = box

        # Draw rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=5)

        # Prepare text
        class_name = class_names.get(int(label), f"Class {label}")
        text_line1 = f"{class_name} confidence: {score:.1%}"  # Show as percentage
        text_line2 = f"Area: {area:,} px"

        # Draw background for text
        bbox1 = draw.textbbox((xmin, ymin - 105), text_line1, font=font)
        bbox2 = draw.textbbox((xmin, ymin - 60), text_line2, font=font_small)

        # Combine bboxes
        text_bbox = (
            min(bbox1[0], bbox2[0]),
            bbox1[1],
            max(bbox1[2], bbox2[2]),
            bbox2[3]
        )

        draw.rectangle(text_bbox, fill="red")
        draw.text((xmin, ymin - 100), text_line1, fill="white", font=font)
        draw.text((xmin, ymin - 60), text_line2, fill="white", font=font_small)

    image.save(output_path)
    print(f"Saved visualization to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="DFU Detection Inference (Improved)")
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='../results/predictions',
                       help='Output directory for predictions')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--img-size', type=int, default=480,
                       help='Input image size (default: 480)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--max-images', type=int, default=5,
                       help='Maximum number of images to process (default: 5)')

    args = parser.parse_args()

    print("="*60)
    print("DFU Detection - Inference (Improved)")
    print("="*60)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"\nUsing device: {device}")

    print(f"\nLoading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    backbone = checkpoint.get('backbone', 'efficientnet_b3')

    # Auto-detect img_size from checkpoint if available
    checkpoint_img_size = checkpoint.get('img_size', None)
    if checkpoint_img_size is not None:
        img_size = checkpoint_img_size
        print(f"Auto-detected img_size from checkpoint: {img_size}")
    else:
        img_size = args.img_size
        print(f"Using img_size from arguments: {img_size}")

    # UPDATED: Auto-detect num_classes from checkpoint
    num_classes = checkpoint.get('num_classes', 2)
    print(f"Model has {num_classes} classes")
    if num_classes == 3:
        print(f"  Classes: 0=background, 1=healthy, 2=ulcer")
        print(f"  Will filter out class 1 (healthy) predictions")

    model = create_efficientdet_model(num_classes=num_classes, backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"Validation loss: {checkpoint['val_loss']:.4f}")

    os.makedirs(args.output, exist_ok=True)

    image_path = Path(args.image)
    if image_path.is_dir():
        image_files = list(image_path.glob('*.jpg')) + list(image_path.glob('*.png')) + \
                      list(image_path.glob('*.jpeg'))
    else:
        image_files = [image_path]

    # Limit to max_images
    if len(image_files) > args.max_images:
        print(f"\nLimiting to first {args.max_images} images (found {len(image_files)} total)")
        image_files = image_files[:args.max_images]

    print(f"\nProcessing {len(image_files)} image(s)...")
    print(f"Confidence threshold: {args.confidence:.2%}")

    all_results = []

    for img_file in image_files:
        print(f"\n{'-'*60}")
        print(f"Processing: {img_file.name}")

        predictions = predict(
            model=model,
            image_path=str(img_file),
            device=device,
            confidence_threshold=args.confidence,
            img_size=img_size  # Use auto-detected or specified img_size
        )

        num_detections = len(predictions['boxes'])
        print(f"  Detected {num_detections} DFU(s)")

        if num_detections > 0:
            for i, (box, score, area) in enumerate(zip(predictions['boxes'],
                                                       predictions['scores'],
                                                       predictions['areas'])):
                print(f"    Detection {i+1}:")
                print(f"      Confidence: {score:.1%}")
                print(f"      Bounding box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
                print(f"      Area: {area:,} pixels")

        output_file = os.path.join(args.output, f"{img_file.stem}_prediction.jpg")
        visualize_predictions(
            image_path=str(img_file),
            predictions=predictions,
            output_path=output_file
        )

        all_results.append({
            'image': img_file.name,
            'num_detections': num_detections,
            'detections': [
                {
                    'confidence': float(score),
                    'bbox': box.tolist(),
                    'area_pixels': int(area)
                }
                for box, score, area in zip(predictions['boxes'],
                                            predictions['scores'],
                                            predictions['areas'])
            ]
        })

    # Save results summary
    summary_file = os.path.join(args.output, 'inference_summary.json')
    import json
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Inference complete!")
    print(f"{'='*60}")
    print(f"Processed {len(image_files)} images")
    print(f"Results saved to: {args.output}")
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()