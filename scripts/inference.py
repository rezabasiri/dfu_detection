"""
Inference script for DFU detection
Run trained model on new images
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
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=0),
        ToTensorV2()
    ])

@torch.no_grad()
def predict(
    model,
    image_path: str,
    device,
    confidence_threshold: float = 0.5,
    img_size: int = 640
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
        Dictionary with predictions
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image_np = np.array(image)

    transform = get_inference_transform(img_size)
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].float() / 255.0

    image_tensor = image_tensor.unsqueeze(0).to(device)

    model.eval()
    predictions = model(image_tensor)[0]

    # Filter by confidence threshold and non-background predictions
    mask = predictions['scores'] >= confidence_threshold

    # Only keep non-background predictions (works for both 2-class and 3-class)
    if len(predictions['labels']) > 0:
        label_mask = predictions['labels'] > 0
        mask = mask & label_mask

    boxes = predictions['boxes'][mask].cpu().numpy()
    scores = predictions['scores'][mask].cpu().numpy()
    labels = predictions['labels'][mask].cpu().numpy()

    return {
        'boxes': boxes,
        'scores': scores,
        'labels': labels,
        'original_size': original_size
    }

def visualize_predictions(
    image_path: str,
    predictions: dict,
    output_path: str,
    class_names: dict = None
):
    """
    Visualize predictions on image

    Args:
        image_path: Path to original image
        predictions: Prediction dictionary
        output_path: Path to save visualization
        class_names: Dictionary mapping class IDs to names
    """
    if class_names is None:
        # Support both 2-class and 3-class models
        class_names = {1: 'DFU', 2: 'DFU'}

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    for box, score, label in zip(predictions['boxes'], predictions['scores'], predictions['labels']):
        xmin, ymin, xmax, ymax = box

        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

        class_name = class_names.get(int(label), f"Class {label}")
        text = f"{class_name}: {score:.2f}"

        bbox = draw.textbbox((xmin, ymin - 25), text, font=font)
        draw.rectangle(bbox, fill="red")
        draw.text((xmin, ymin - 25), text, fill="white", font=font)

    image.save(output_path)
    print(f"Saved visualization to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="DFU Detection Inference")
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='../results/predictions',
                       help='Output directory for predictions')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size (default: 640)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    print("="*60)
    print("DFU Detection - Inference")
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
    backbone = checkpoint.get('backbone', 'efficientnet_b0')

    # Auto-detect num_classes from checkpoint (default to 2 for new models)
    num_classes = checkpoint.get('num_classes', 2)
    print(f"Model: {backbone} with {num_classes} classes")
    if num_classes == 2:
        print(f"  Classes: 0=background, 1=ulcer")
    elif num_classes == 3:
        print(f"  Classes: 0=background, 1=healthy, 2=ulcer (legacy)")

    model = create_efficientdet_model(num_classes=num_classes, backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")

    os.makedirs(args.output, exist_ok=True)

    image_path = Path(args.image)
    if image_path.is_dir():
        image_files = list(image_path.glob('*.jpg')) + list(image_path.glob('*.png')) + list(image_path.glob('*.jpeg'))
    else:
        image_files = [image_path]

    print(f"\nProcessing {len(image_files)} image(s)...")

    for img_file in image_files:
        print(f"\nProcessing: {img_file.name}")

        predictions = predict(
            model=model,
            image_path=str(img_file),
            device=device,
            confidence_threshold=args.confidence,
            img_size=args.img_size
        )

        num_detections = len(predictions['boxes'])
        print(f"  Detected {num_detections} DFU(s)")

        if num_detections > 0:
            for i, (box, score) in enumerate(zip(predictions['boxes'], predictions['scores'])):
                print(f"    Detection {i+1}: confidence={score:.3f}, box={box}")

        output_file = os.path.join(args.output, f"{img_file.stem}_prediction.jpg")
        visualize_predictions(
            image_path=str(img_file),
            predictions=predictions,
            output_path=output_file
        )

    print(f"\n{'='*60}")
    print(f"Inference complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()