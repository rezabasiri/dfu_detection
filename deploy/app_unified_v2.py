"""
DFU Detection - Unified Demo V2
Multi-model automatic comparison with intelligent selection
Designed by Reza Basiri (90reza@gmail.com)
"""

import os
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from typing import List, Dict, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


# ==================== MODEL LOADING ====================
@st.cache_resource
def load_yolo_model(model_path: str):
    """Load YOLO model with caching and extract training config"""
    from ultralytics import YOLO
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = YOLO(model_path)

    # Extract training image size from model metadata
    training_img_size = None
    try:
        if hasattr(model, 'ckpt') and model.ckpt is not None:
            train_args = model.ckpt.get('train_args', {})
            training_img_size = train_args.get('imgsz', None)
            if isinstance(training_img_size, (list, tuple)):
                training_img_size = training_img_size[0]
    except:
        pass

    return model, "yolo", training_img_size


@st.cache_resource
def load_pytorch_model(model_path: str, device='cpu'):
    """Load Faster R-CNN or RetinaNet model from .pth checkpoint"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    from models import create_from_checkpoint

    device_obj = torch.device(device)
    detector = create_from_checkpoint(model_path, device=device_obj)
    detector.set_eval_mode()

    # Extract metadata from checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_name = checkpoint.get('model_name', 'faster_rcnn')
    img_size = checkpoint.get('img_size', 512)
    backbone = detector.backbone_name if hasattr(detector, 'backbone_name') else checkpoint.get('backbone', 'unknown')

    return detector, model_name, img_size, backbone


def get_model_metadata(model_path: str, device: str = 'cpu') -> Dict:
    """Extract model metadata without full loading"""
    ext = os.path.splitext(model_path)[1].lower()
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB

    metadata = {
        'path': model_path,
        'size_mb': file_size,
        'type': 'yolo' if ext == '.pt' else 'pytorch',
        'training_img_size': None,
        'num_params': 0
    }

    try:
        if metadata['type'] == 'yolo':
            model, _, training_size = load_yolo_model(model_path)
            metadata['training_img_size'] = training_size
            metadata['num_params'] = sum(p.numel() for p in model.model.parameters())
        else:
            model, _, training_size, backbone = load_pytorch_model(model_path, device)
            metadata['training_img_size'] = training_size
            metadata['num_params'] = sum(p.numel() for p in model.parameters())
            metadata['backbone'] = backbone
    except Exception as e:
        metadata['error'] = str(e)

    return metadata


# ==================== INFERENCE ====================
def run_yolo_inference(model, image_np, img_size, conf_threshold, device):
    """Run YOLO inference"""
    results = model(
        image_np,
        imgsz=img_size,
        device=device,
        conf=conf_threshold,
        verbose=False
    )

    result = results[0]
    boxes_obj = result.boxes

    if boxes_obj is None or len(boxes_obj) == 0:
        return []

    detections = []
    for box in boxes_obj:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        detections.append({
            'box': [float(x1), float(y1), float(x2), float(y2)],
            'confidence': conf,
            'label': 1
        })

    return detections


def run_pytorch_inference(detector, image, img_size, conf_threshold, device):
    """Run Faster R-CNN or RetinaNet inference"""
    import torchvision.transforms as T

    if hasattr(detector, 'get_model'):
        model = detector.get_model()
    else:
        model = detector

    model.eval()

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])

    image_tensor = transform(image).to(device)

    with torch.no_grad():
        predictions = model([image_tensor])

    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()

    mask = (scores >= conf_threshold) & (labels > 0)

    detections = []
    h_orig, w_orig = np.array(image).shape[:2]

    for box, score in zip(boxes[mask], scores[mask]):
        x1, y1, x2, y2 = box
        x1 = x1 * w_orig / img_size
        y1 = y1 * h_orig / img_size
        x2 = x2 * w_orig / img_size
        y2 = y2 * h_orig / img_size

        detections.append({
            'box': [float(x1), float(y1), float(x2), float(y2)],
            'confidence': float(score),
            'label': 1
        })

    return detections


def run_inference_on_model(model_info: Dict, image, img_size: int, conf_threshold: float, device: str) -> List[Dict]:
    """Run inference on a single model at a specific size"""
    model_type = model_info['type']
    model = model_info['model']

    image_np = np.array(image)

    if model_type == 'yolo':
        return run_yolo_inference(model, image_np, img_size, conf_threshold, device)
    else:
        return run_pytorch_inference(model, image, img_size, conf_threshold, device)


# ==================== AUTOMATIC MODEL COMPARISON ====================
def compare_all_models_and_sizes(
    loaded_models: List[Dict],
    image: Image.Image,
    img_sizes: List[int],
    conf_threshold: float,
    device: str
) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Test all models at all image sizes and select the best combination.

    Selection criteria:
    1. Highest average confidence
    2. Fewest boxes (tiebreaker)
    3. Smaller model (tiebreaker)

    Returns:
        (best_result, all_results)
    """
    all_results = []

    for model_idx, model_info in enumerate(loaded_models):
        if model_info is None:
            continue

        for img_size in img_sizes:
            try:
                detections = run_inference_on_model(
                    model_info, image, img_size, conf_threshold, device
                )

                # Calculate metrics for this combination
                if len(detections) > 0:
                    avg_confidence = np.mean([d['confidence'] for d in detections])
                else:
                    avg_confidence = 0.0

                result = {
                    'model_idx': model_idx,
                    'model_name': f"Model {model_idx + 1}",
                    'img_size': img_size,
                    'detections': detections,
                    'num_detections': len(detections),
                    'avg_confidence': avg_confidence,
                    'model_size_mb': model_info['metadata']['size_mb'],
                    'model_params': model_info['metadata']['num_params']
                }

                all_results.append(result)

            except Exception as e:
                # Skip failed combinations
                continue

    if not all_results:
        return None, []

    # Sort by: avg_confidence (desc), num_detections (asc), model_params (asc)
    best_result = max(
        all_results,
        key=lambda x: (
            x['avg_confidence'],           # Primary: highest avg confidence
            -x['num_detections'],           # Tiebreaker 1: fewest boxes (negative for ascending)
            -x['model_params']              # Tiebreaker 2: smaller model (negative for ascending)
        )
    )

    return best_result, all_results


# ==================== VISUALIZATION ====================
def get_font(size=20):
    """Get font that works on Mac"""
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except:
        try:
            return ImageFont.truetype("Arial.ttf", size)
        except:
            return ImageFont.load_default()


def get_text_width(font, text):
    """Get text width with fallback for older PIL versions"""
    try:
        return font.getlength(text)
    except AttributeError:
        return len(text) * 10


def draw_detections(image, detections, font_size_multiplier=4):
    """Draw bounding boxes on image with larger text"""
    drawn = image.copy()
    draw = ImageDraw.Draw(drawn)

    # 4x bigger font as requested
    base_font_size = 20
    bbox_font_size = base_font_size * font_size_multiplier
    font = get_font(bbox_font_size)

    detections_info = []

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['box']
        conf = det['confidence']

        # Calculate pixel area
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        pixel_area = int(width * height)

        # Color based on confidence
        if conf >= 0.8:
            color = "green"
        elif conf >= 0.5:
            color = "orange"
        else:
            color = "red"

        # Draw rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)

        # Label text with bigger font
        text = f"{conf*100:.1f}%"
        text_width = get_text_width(font, text)

        # Adjust text background size for bigger font
        text_height = bbox_font_size + 10
        text_pos = (x1, max(0, y1 - text_height - 5))
        draw.rectangle(
            [text_pos, (x1 + text_width + 15, y1 - 5)],
            fill=color
        )

        # Draw confidence text
        draw.text((x1 + 7, y1 - text_height), text, fill="white", font=font)

        # Draw area below box with bigger font
        area_text = f"{pixel_area:,} px²"
        draw.text((x1, y2 + 10), area_text, fill=color, font=font)

        # Store detection info
        detections_info.append({
            "ID": i + 1,
            "Confidence": f"{conf:.3f}",
            "Area (px²)": f"{pixel_area:,}",
            "Box": f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
        })

    # Add copyright in bottom-right corner with small font
    copyright_font = get_font(10)
    copyright_text = "Demo only - Designed by Reza Basiri (90reza@gmail.com)"
    img_width, img_height = drawn.size
    copyright_width = get_text_width(copyright_font, copyright_text)
    copyright_pos = (img_width - copyright_width - 10, img_height - 20)

    # Semi-transparent background for copyright
    draw.rectangle(
        [copyright_pos, (img_width - 5, img_height - 5)],
        fill=(0, 0, 0, 128)
    )
    draw.text(copyright_pos, copyright_text, fill="white", font=copyright_font)

    return drawn, detections_info


# ==================== STREAMLIT UI ====================
st.set_page_config(page_title="DFU Detection V2", layout="wide")

st.title("🩺 DFU Detection – Multi-Model Demo")
st.markdown("Upload a foot image to detect diabetic foot ulcers using multiple models")

# Sidebar - Model Selection
st.sidebar.header("📦 Model Selection")

# Mode selection
detection_mode = st.sidebar.radio(
    "Detection Mode",
    ["Automatic", "Manual"],
    help="Automatic: Tests all models at all sizes and picks best result. Manual: Select specific model and size."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Load Models (up to 5)")

# Model slots
loaded_models = []
model_paths = []

for i in range(5):
    with st.sidebar.expander(f"Model {i+1}", expanded=(i==0)):
        col1, col2 = st.columns([3, 1])

        with col1:
            model_path = st.text_input(
                "Path",
                key=f"model_path_{i}",
                placeholder="/path/to/model.pt or .pth"
            )

        with col2:
            # File browser would require additional library, using text input for now
            st.caption("📁")

        if model_path and os.path.exists(model_path):
            model_paths.append(model_path)

            # Load model
            try:
                device = 'cpu'  # Force CPU for Mac compatibility
                ext = os.path.splitext(model_path)[1].lower()

                if ext == '.pt':
                    model, model_type, training_size = load_yolo_model(model_path)
                else:
                    model, model_type, training_size, backbone = load_pytorch_model(model_path, device)

                metadata = get_model_metadata(model_path, device)

                loaded_models.append({
                    'index': i,
                    'model': model,
                    'type': 'yolo' if ext == '.pt' else 'pytorch',
                    'metadata': metadata
                })

                # Display model info
                st.success(f"✓ Loaded ({metadata['size_mb']:.1f} MB, {metadata['num_params']:,} params)")
                if metadata['training_img_size']:
                    st.caption(f"Training size: {metadata['training_img_size']}px")

            except Exception as e:
                st.error(f"Failed to load: {e}")
                loaded_models.append(None)
        elif model_path:
            st.warning("⚠️ File not found")
            loaded_models.append(None)
        else:
            loaded_models.append(None)

# Pad loaded_models to always have 5 entries
while len(loaded_models) < 5:
    loaded_models.append(None)

# Count valid models
num_loaded = sum(1 for m in loaded_models if m is not None)

st.sidebar.markdown("---")
st.sidebar.info(f"**{num_loaded} model(s) loaded**")

# Settings
st.sidebar.header("⚙️ Settings")

# Manual mode: model selection
selected_model_idx = None
if detection_mode == "Manual" and num_loaded > 0:
    st.sidebar.subheader("Select Model")

    # Create radio buttons for loaded models only
    model_options = []
    model_indices = []
    for i, model_info in enumerate(loaded_models):
        if model_info is not None:
            model_options.append(f"Model {i+1}")
            model_indices.append(i)

    if model_options:
        selected_label = st.sidebar.radio("Active Model", model_options)
        selected_model_idx = model_indices[model_options.index(selected_label)]

# Image size configuration
if detection_mode == "Automatic":
    st.sidebar.subheader("Image Sizes to Test")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        test_512 = st.checkbox("512px", value=True)
        test_640 = st.checkbox("640px", value=True)
    with col2:
        test_1024 = st.checkbox("1024px", value=True)
        test_1280 = st.checkbox("1280px", value=False)

    img_sizes_to_test = []
    if test_512: img_sizes_to_test.append(512)
    if test_640: img_sizes_to_test.append(640)
    if test_1024: img_sizes_to_test.append(1024)
    if test_1280: img_sizes_to_test.append(1280)

    if not img_sizes_to_test:
        st.sidebar.warning("⚠️ Select at least one size")
else:
    # Manual mode: single size selection
    manual_img_size = st.sidebar.selectbox(
        "Image Size",
        [512, 640, 1024, 1280],
        index=2
    )

# Confidence threshold
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_image = st.file_uploader("📷 Upload Foot Image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.markdown("### Detection Results")
    result_placeholder = st.empty()

# Detection button
if st.button("🔍 Find DFUs", type="primary", use_container_width=True):

    if num_loaded == 0:
        st.error("❌ Please load at least one model")
    elif uploaded_image is None:
        st.error("❌ Please upload an image first")
    elif detection_mode == "Automatic" and not img_sizes_to_test:
        st.error("❌ Please select at least one image size to test")
    elif detection_mode == "Manual" and selected_model_idx is None:
        st.error("❌ Please select a model")
    else:
        # Load image
        image = Image.open(uploaded_image).convert("RGB")
        device = 'cpu'

        # ========== AUTOMATIC MODE ==========
        if detection_mode == "Automatic":
            with st.spinner(f"Testing {num_loaded} model(s) at {len(img_sizes_to_test)} size(s)..."):

                best_result, all_results = compare_all_models_and_sizes(
                    loaded_models,
                    image,
                    img_sizes_to_test,
                    conf_threshold,
                    device
                )

                if best_result is None:
                    st.warning("⚠️ No DFUs detected by any model at any size")
                    st.info(f"Try lowering the confidence threshold (current: {conf_threshold})")
                else:
                    detections = best_result['detections']

                    # Draw results
                    drawn, detections_info = draw_detections(image, detections)

                    # Display winning combination
                    st.success(
                        f"✅ Best Result: **{best_result['model_name']}** @ **{best_result['img_size']}px** "
                        f"({best_result['num_detections']} DFU(s), Avg Conf: {best_result['avg_confidence']:.1%})"
                    )

                    with col2:
                        st.image(drawn, caption=f"Detected by {best_result['model_name']}", use_container_width=True)

                    # Detection details
                    st.markdown("### 📊 Detection Details")
                    st.dataframe(detections_info, use_container_width=True)

                    # Color legend
                    st.markdown("""
                    **Confidence Color Code:**
                    - 🟢 **Green**: High confidence (≥ 80%)
                    - 🟠 **Orange**: Medium confidence (50-80%)
                    - 🔴 **Red**: Low confidence (< 50%)
                    """)

        # ========== MANUAL MODE ==========
        else:
            model_info = loaded_models[selected_model_idx]

            with st.spinner(f"Running Model {selected_model_idx + 1} @ {manual_img_size}px..."):
                try:
                    detections = run_inference_on_model(
                        model_info,
                        image,
                        manual_img_size,
                        conf_threshold,
                        device
                    )

                    if len(detections) == 0:
                        st.warning("⚠️ No DFUs detected")
                        st.info(f"Try lowering the confidence threshold (current: {conf_threshold})")
                    else:
                        # Draw results
                        drawn, detections_info = draw_detections(image, detections)

                        # Display results
                        st.success(f"✅ Detected {len(detections)} DFU(s)")

                        with col2:
                            st.image(drawn, caption=f"Model {selected_model_idx + 1} @ {manual_img_size}px", use_container_width=True)

                        # Detection details
                        st.markdown("### 📊 Detection Details")
                        st.dataframe(detections_info, use_container_width=True)

                        # Color legend
                        st.markdown("""
                        **Confidence Color Code:**
                        - 🟢 **Green**: High confidence (≥ 80%)
                        - 🟠 **Orange**: Medium confidence (50-80%)
                        - 🔴 **Red**: Low confidence (< 50%)
                        """)

                except Exception as e:
                    st.error(f"❌ Inference failed: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Quick Tips:**
- **Automatic Mode**: Tests all model+size combos, picks best
- **Manual Mode**: Full control over model and size
- Lower confidence → more detections
- Larger images → better accuracy but slower

**Selection Criteria (Automatic):**
1. Highest average confidence
2. Fewest bounding boxes
3. Smallest model size
""")

st.sidebar.caption("Demo only - Designed by Reza Basiri (90reza@gmail.com)")
