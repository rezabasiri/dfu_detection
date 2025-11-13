"""
YOLO Inference on OwnHealth Data

Run YOLO inference on OwnHealth patient images and save visualizations with bounding boxes.
Similar to run_inference_ownhealth.py but specifically for YOLO models.

Features:
- Randomly selects images from OwnHealth folder
- Runs YOLO inference with configurable confidence threshold
- Draws bounding boxes with confidence scores
- Saves annotated images with patient ID tracking
- Generates summary JSON with all predictions

Usage:
    # Run inference on 50 random OwnHealth images
    python yolo_inference_ownhealth.py

    # Use custom model and confidence
    python yolo_inference_ownhealth.py --model ../checkpoints/yolo/weights/best.pt --confidence 0.7

    # Process specific number of images
    python yolo_inference_ownhealth.py --num-images 100 --confidence 0.5
"""

import os
import random
import argparse
import json
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("ERROR: ultralytics package not found. Install with: pip install ultralytics")
    exit(1)


def collect_all_images(ownhealth_folder):
    """
    Collect all image paths from OwnHealth folder (nested in patient subfolders)

    Args:
        ownhealth_folder: Path to OwnHealth folder containing patient subfolders

    Returns:
        List of image info dicts with patient IDs
    """
    all_images = []

    for patient_folder in Path(ownhealth_folder).iterdir():
        if patient_folder.is_dir():
            patient_id = patient_folder.name

            # Find all images
            image_files = (
                list(patient_folder.glob('*.jpg')) +
                list(patient_folder.glob('*.jpeg')) +
                list(patient_folder.glob('*.png')) +
                list(patient_folder.glob('*.JPG')) +
                list(patient_folder.glob('*.JPEG')) +
                list(patient_folder.glob('*.PNG'))
            )

            for img_path in image_files:
                all_images.append({
                    'path': str(img_path),
                    'patient_id': patient_id,
                    'filename': img_path.name
                })

    return all_images


def run_yolo_inference(model, image_path, confidence_threshold=0.5):
    """
    Run YOLO inference on a single image

    Args:
        model: YOLO model
        image_path: Path to image file
        confidence_threshold: Minimum confidence for detections

    Returns:
        Dict with boxes, scores, and image info
    """
    # Load image (YOLO handles this internally, but we need it for drawing)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # Run inference
    results = model.predict(image_rgb, conf=confidence_threshold, verbose=False)

    # Extract predictions
    predictions = {
        'boxes': [],
        'scores': [],
        'labels': [],
        'image_size': (width, height)
    }

    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [xmin, ymin, xmax, ymax]
        scores = results[0].boxes.conf.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            predictions['boxes'].append(box.tolist())
            predictions['scores'].append(float(score))
            predictions['labels'].append(int(label))

    return predictions, image_rgb


def draw_bounding_boxes(image_rgb, predictions, confidence_threshold=0.5):
    """
    Draw bounding boxes on image with confidence scores

    Args:
        image_rgb: RGB image (numpy array)
        predictions: Dict with boxes and scores
        confidence_threshold: Minimum confidence to display

    Returns:
        PIL Image with boxes drawn
    """
    image_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image_pil)

    # Try to load a nice font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    boxes = predictions['boxes']
    scores = predictions['scores']

    # Draw each box
    for box, score in zip(boxes, scores):
        if score < confidence_threshold:
            continue

        xmin, ymin, xmax, ymax = box

        # Choose color based on confidence
        if score >= 0.8:
            color = (0, 255, 0)  # Green - high confidence
        elif score >= 0.5:
            color = (255, 165, 0)  # Orange - medium confidence
        else:
            color = (255, 0, 0)  # Red - low confidence

        # Draw box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

        # Draw confidence label
        label = f"DFU: {score:.2f}"

        # Get text bounding box
        bbox = draw.textbbox((xmin, ymin - 25), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw background for text
        draw.rectangle(
            [xmin, ymin - text_height - 8, xmin + text_width + 8, ymin],
            fill=color
        )

        # Draw text
        draw.text((xmin + 4, ymin - text_height - 4), label, fill=(255, 255, 255), font=font)

        # Draw area info
        area = int((xmax - xmin) * (ymax - ymin))
        area_label = f"Area: {area:,} px²"
        draw.text((xmin, ymax + 5), area_label, fill=color, font=small_font)

    # Draw detection count
    num_detections = len([s for s in scores if s >= confidence_threshold])
    if num_detections > 0:
        header = f"Detected: {num_detections} ulcer(s)"
        header_bbox = draw.textbbox((10, 10), header, font=font)
        draw.rectangle(
            [10, 10, header_bbox[2] + 10, header_bbox[3] + 10],
            fill=(255, 0, 0)
        )
        draw.text((15, 15), header, fill=(255, 255, 255), font=font)
    else:
        header = "No ulcers detected"
        header_bbox = draw.textbbox((10, 10), header, font=font)
        draw.rectangle(
            [10, 10, header_bbox[2] + 10, header_bbox[3] + 10],
            fill=(0, 200, 0)
        )
        draw.text((15, 15), header, fill=(255, 255, 255), font=font)

    return image_pil


def main():
    parser = argparse.ArgumentParser(
        description='Run YOLO inference on OwnHealth patient images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process 50 random images with default confidence
    python yolo_inference_ownhealth.py

    # Use high confidence threshold for clinical review
    python yolo_inference_ownhealth.py --confidence 0.7 --num-images 100

    # Use specific YOLO model
    python yolo_inference_ownhealth.py --model ../checkpoints/yolo/weights/best.pt

Confidence threshold guidance:
    - 0.5: Balanced detection (default)
    - 0.7: High confidence (fewer false positives)
    - 0.8: Very conservative (clinical screening)
        """
    )

    parser.add_argument('--ownhealth-folder', type=str,
                        default='/mnt/c/Users/90rez/OneDrive - University of Toronto/PhDUofT/SideProjects/DFU_Detection_Asem/OwnHealth',
                        help='Path to OwnHealth folder with patient subfolders')
    parser.add_argument('--model', type=str,
                        default='../checkpoints/yolo/weights/best.pt',
                        help='Path to YOLO model weights (.pt file)')
    parser.add_argument('--num-images', type=int,
                        default=50,
                        help='Number of random images to select (default: 50)')
    parser.add_argument('--confidence', type=float,
                        default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--output', type=str,
                        default='../results/ownhealth_yolo_predictions',
                        help='Output directory for annotated images')
    parser.add_argument('--seed', type=int,
                        default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--device', type=str,
                        default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    print("="*80)
    print("YOLO INFERENCE - OWNHEALTH DATA")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Output: {args.output}")
    print("="*80 + "\n")

    # Check if OwnHealth folder exists
    if not os.path.exists(args.ownhealth_folder):
        print(f"ERROR: OwnHealth folder not found: {args.ownhealth_folder}")
        exit(1)

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {args.model}")
        print("\nExpected YOLO checkpoint locations:")
        print("  Production: ../checkpoints/yolo/weights/best.pt")
        print("  Test: ../checkpoints_test/yolo/weights/best.pt")
        exit(1)

    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(args.model)
    model.to(args.device)
    print("✓ Model loaded\n")

    # Collect all images
    print(f"Scanning OwnHealth folder: {args.ownhealth_folder}")
    all_images = collect_all_images(args.ownhealth_folder)

    if len(all_images) == 0:
        print("ERROR: No images found in OwnHealth folder")
        exit(1)

    print(f"Found {len(all_images)} total images across all patient folders")

    # Randomly select images
    random.seed(args.seed)
    num_to_select = min(args.num_images, len(all_images))
    selected_images = random.sample(all_images, num_to_select)

    print(f"\nRandomly selected {num_to_select} images")
    print("-"*80)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    results_summary = {
        'model': str(model_path),
        'confidence_threshold': args.confidence,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_images': num_to_select,
        'images_with_detections': 0,
        'total_detections': 0,
        'predictions': []
    }

    print("\nRunning inference...\n")

    for i, img_info in enumerate(tqdm(selected_images, desc="Processing images")):
        # Run inference
        predictions, image_rgb = run_yolo_inference(
            model,
            img_info['path'],
            confidence_threshold=args.confidence
        )

        if predictions is None:
            continue

        # Count detections
        num_detections = len([s for s in predictions['scores'] if s >= args.confidence])
        if num_detections > 0:
            results_summary['images_with_detections'] += 1
            results_summary['total_detections'] += num_detections

        # Draw boxes
        annotated_image = draw_bounding_boxes(image_rgb, predictions, args.confidence)

        # Save annotated image with patient ID in filename
        output_filename = f"patient_{img_info['patient_id']}_{img_info['filename']}"
        output_path = output_dir / output_filename
        annotated_image.save(output_path)

        # Store prediction info
        results_summary['predictions'].append({
            'patient_id': img_info['patient_id'],
            'filename': img_info['filename'],
            'original_path': img_info['path'],
            'output_path': str(output_path),
            'num_detections': num_detections,
            'detections': [
                {
                    'box': box,
                    'confidence': float(score),
                    'label': int(label)
                }
                for box, score, label in zip(predictions['boxes'], predictions['scores'], predictions['labels'])
                if score >= args.confidence
            ]
        })

    # Save summary JSON
    summary_path = output_dir / 'inference_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print(f"\nProcessed: {num_to_select} images")
    print(f"Images with detections: {results_summary['images_with_detections']}")
    print(f"Total detections: {results_summary['total_detections']}")
    print(f"Detection rate: {results_summary['images_with_detections']/num_to_select*100:.1f}%")
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary JSON: {summary_path}")
    print("="*80)

    # Show sample predictions
    if results_summary['total_detections'] > 0:
        print("\n" + "="*80)
        print("SAMPLE PREDICTIONS (Top 5 highest confidence)")
        print("="*80)

        # Get all detections with patient info
        all_detections = []
        for pred in results_summary['predictions']:
            for det in pred['detections']:
                all_detections.append({
                    'patient_id': pred['patient_id'],
                    'filename': pred['filename'],
                    'confidence': det['confidence'],
                    'box': det['box']
                })

        # Sort by confidence
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)

        # Show top 5
        for i, det in enumerate(all_detections[:5], 1):
            box = det['box']
            area = int((box[2] - box[0]) * (box[3] - box[1]))
            print(f"\n{i}. Patient {det['patient_id']}: {det['filename']}")
            print(f"   Confidence: {det['confidence']:.3f}")
            print(f"   Box: [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
            print(f"   Area: {area:,} pixels²")

        print("\n" + "="*80)

    print("\n✓ All annotated images saved with bounding boxes!")
    print(f"✓ View results in: {output_dir}\n")


if __name__ == "__main__":
    main()
