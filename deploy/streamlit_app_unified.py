"""
DFU Detection - Demo
Optimized for Mac (CPU inference)
"""

import os
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


# --- Model Loading Functions ---
@st.cache_resource
def load_yolo_model(model_path: str):
    """Load YOLO model with caching and extract training config"""
    from ultralytics import YOLO
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = YOLO(model_path)

    # Try to extract training image size from model metadata
    training_img_size = None
    try:
        # YOLO stores training args in model.ckpt['train_args']
        if hasattr(model, 'ckpt') and model.ckpt is not None:
            train_args = model.ckpt.get('train_args', {})
            training_img_size = train_args.get('imgsz', None)
            if isinstance(training_img_size, (list, tuple)):
                training_img_size = training_img_size[0]  # Take first value if list
    except:
        pass

    return model, "yolo", training_img_size


@st.cache_resource
def load_pytorch_model(model_path: str, device='cpu'):
    """Load Faster R-CNN or RetinaNet model from .pth checkpoint"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Use the model factory to load from checkpoint (auto-detects model type)
    from models import create_from_checkpoint

    device_obj = torch.device(device)
    detector = create_from_checkpoint(model_path, device=device_obj)

    # Ensure model is in eval mode
    detector.set_eval_mode()

    # Extract metadata from checkpoint for display
    checkpoint = torch.load(model_path, map_location=device)
    model_name = checkpoint.get('model_name', 'faster_rcnn')
    img_size = checkpoint.get('img_size', 512)
    backbone = detector.backbone_name if hasattr(detector, 'backbone_name') else checkpoint.get('backbone', 'unknown')

    return detector, model_name, img_size, backbone


def detect_model_type(model_path: str):
    """Detect whether model is YOLO (.pt) or PyTorch (.pth)"""
    ext = os.path.splitext(model_path)[1].lower()
    if ext == '.pt':
        return 'yolo'
    elif ext == '.pth':
        return 'pytorch'
    else:
        raise ValueError(f"Unsupported model extension: {ext}")


# --- Inference Functions ---
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
            'label': 1  # DFU class
        })

    return detections


def run_pytorch_inference(detector, image, img_size, conf_threshold, device):
    """Run Faster R-CNN or RetinaNet inference"""
    import torchvision.transforms as T

    # Get the underlying PyTorch model from the detector wrapper
    if hasattr(detector, 'get_model'):
        model = detector.get_model()
    else:
        model = detector  # Already an nn.Module

    # Ensure model is in eval mode
    model.eval()

    # Prepare image
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])

    image_tensor = transform(image).to(device)

    # Run inference (model expects list of tensors)
    with torch.no_grad():
        predictions = model([image_tensor])

    # Parse predictions
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()

    # Filter by confidence and non-background class
    mask = (scores >= conf_threshold) & (labels > 0)

    detections = []
    h_orig, w_orig = np.array(image).shape[:2]

    for box, score in zip(boxes[mask], scores[mask]):
        # Scale boxes back to original image size
        x1, y1, x2, y2 = box
        x1 = x1 * w_orig / img_size
        y1 = y1 * h_orig / img_size
        x2 = x2 * w_orig / img_size
        y2 = y2 * h_orig / img_size

        detections.append({
            'box': [float(x1), float(y1), float(x2), float(y2)],
            'confidence': float(score),
            'label': 1  # DFU class
        })

    return detections


# --- Visualization Functions ---
def get_font(size=20):
    """Get font that works on Mac"""
    try:
        # Mac system font
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except:
        try:
            # Fallback for other systems
            return ImageFont.truetype("Arial.ttf", size)
        except:
            return ImageFont.load_default()


def get_text_width(font, text):
    """Get text width with fallback for older PIL versions"""
    try:
        return font.getlength(text)
    except AttributeError:
        # Fallback for PIL < 8.0
        return len(text) * 10


def draw_detections(image, detections):
    """Draw bounding boxes on image"""
    drawn = image.copy()
    draw = ImageDraw.Draw(drawn)
    font = get_font(20)

    detections_info = []

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['box']
        conf = det['confidence']

        # Calculate pixel area
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        pixel_area = int(width * height)

        # Choose color based on confidence
        if conf >= 0.8:
            color = "green"  # High confidence
        elif conf >= 0.5:
            color = "orange"  # Medium confidence
        else:
            color = "red"  # Low confidence

        # Draw rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)

        # Label text
        text = f"DFU {i+1}: {conf*100:.1f}%"
        text_width = get_text_width(font, text)

        # Draw text background
        text_pos = (x1, max(0, y1 - 25))
        draw.rectangle(
            [text_pos, (x1 + text_width + 8, y1 - 2)],
            fill=color
        )

        # Draw text
        draw.text((x1 + 4, y1 - 22), text, fill="white", font=font)

        # Draw area below box
        area_text = f"{pixel_area:,} px²"
        draw.text((x1, y2 + 5), area_text, fill=color, font=font)

        # Store detection info
        detections_info.append({
            "ID": i + 1,
            "Confidence": f"{conf:.3f}",
            "Area (px²)": f"{pixel_area:,}",
            "Box": f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
        })

    return drawn, detections_info


# --- Streamlit UI ---
st.title("🩺 DFU Detection – Demo")
st.markdown("Upload a foot image to detect diabetic foot ulcers")

st.sidebar.header("⚙️ Model & Settings")

model_path = st.sidebar.text_input(
    "Model path",
    value="",
    placeholder="Enter path to .pt or .pth file"
)

# CPU option (recommended for Mac)
run_on_cpu = st.sidebar.checkbox("Force CPU (recommended on Mac)", value=True)

# Image size - auto-detected from model training config
img_size = st.sidebar.selectbox(
    "Image size (for inference)",
    [512, 640, 1024, 1280, 1536],
    index=2,  # Default to 1024
    help="PyTorch models: Auto-set from checkpoint (this setting ignored)."
)

# Performance warning for large sizes
if img_size >= 1280:
    st.sidebar.warning(f"⏱️ {img_size}px may take 15-30+ sec on Mac CPU")

# Confidence threshold
conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.05)

# Image upload
uploaded_image = st.file_uploader("📷 Upload a foot image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Show uploaded image
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

if st.button("🔍 Find DFUs", type="primary"):

    if not model_path:
        st.error("Please specify a model path.")
    elif not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")
    elif uploaded_image is None:
        st.error("Please upload an image first.")
    else:
        # Detect model type
        try:
            model_type = detect_model_type(model_path)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        # Load model
        with st.spinner(f"Loading {model_type.upper()} model..."):
            try:
                device = 'cpu' if run_on_cpu else 'cuda'

                if model_type == 'yolo':
                    model, model_name, yolo_training_size = load_yolo_model(model_path)
                    model_name = "YOLO"

                    # Use training size if available, otherwise user-specified
                    if yolo_training_size is not None:
                        training_img_size = yolo_training_size
                        if img_size != yolo_training_size:
                            st.info(f"🤖 trained @ {yolo_training_size}px, but using {img_size}px (user override)")
                        else:
                            st.info(f"🤖 Using YOLO training size: {yolo_training_size}px")
                    else:
                        training_img_size = img_size
                        st.info(f"🤖 model (training size unknown), using {img_size}px")
                else:  # pytorch (Faster R-CNN or RetinaNet)
                    model, model_name, training_img_size, backbone = load_pytorch_model(model_path, device)
                    # st.info(f"🤖 Detected: {model_name} ({backbone}) @ {training_img_size}px")
                    st.info(f"🤖 Detected: ({backbone}) @ {training_img_size}px")
                    img_size = training_img_size  # Override with checkpoint value

            except Exception as e:
                st.error(f"Failed to load model: {e}")
                st.stop()

        # Load image
        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)

        # Run inference
        with st.spinner(f"Running {model_name} inference (img_size={img_size})..."):
            try:
                if model_type == 'yolo':
                    detections = run_yolo_inference(
                        model, image_np, img_size, conf_threshold, device
                    )
                else:
                    detections = run_pytorch_inference(
                        model, image, img_size, conf_threshold, device
                    )
            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.stop()

        # Display results
        if len(detections) == 0:
            st.warning("⚠️ No DFUs detected.")
            st.info(f"Try lowering the confidence threshold (current: {conf_threshold})")
        else:
            # Draw boxes
            drawn, detections_info = draw_detections(image, detections)

            # Show results
            st.success(f"✅ Detected {len(detections)} DFU(s)")
            st.image(drawn, use_container_width=True)

            # Show detection details table
            st.markdown("### 📊 Detection Details")
            st.dataframe(detections_info, use_container_width=True)

            # Color legend
            st.markdown("""
            **Confidence Color Code:**
            - 🟢 **Green**: High confidence (≥ 80%)
            - 🟠 **Orange**: Medium confidence (50-80%)
            - 🔴 **Red**: Low confidence (< 50%)
            """)

# Footer - Model Info
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
if model_path and os.path.exists(model_path):
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    st.sidebar.info(f"📦 Model size: {file_size:.1f} MB")

    try:
        model_type = detect_model_type(model_path)

        if model_type == 'yolo':
            temp_model, _, yolo_train_size = load_yolo_model(model_path)
            total_params = sum(p.numel() for p in temp_model.model.parameters())

            if total_params < 5_000_000:
                variant = "yolov8n (nano)"
            elif total_params < 15_000_000:
                variant = "yolov8s (small)"
            elif total_params < 30_000_000:
                variant = "yolov8m (medium)"
            elif total_params < 50_000_000:
                variant = "yolov8l (large)"
            else:
                variant = "yolov8x (extra-large)"

            # st.sidebar.info(f"🤖 Model: YOLO - {variant}")
            st.sidebar.caption(f"Parameters: {total_params:,}")
            if yolo_train_size is not None:
                st.sidebar.caption(f"Training size: {yolo_train_size}px")

        else:  # pytorch
            device = 'cpu' if run_on_cpu else 'cuda'
            temp_model, model_name, train_img_size, backbone = load_pytorch_model(model_path, device)
            total_params = sum(p.numel() for p in temp_model.parameters())

            # st.sidebar.info(f"🤖 Model: {model_name}")
            st.sidebar.caption(f"Backbone: {backbone}")
            st.sidebar.caption(f"Training size: {train_img_size}px")
            st.sidebar.caption(f"Parameters: {total_params:,}")

    except Exception as e:
        st.sidebar.error(f"Could not read model info: {e}")
else:
    st.sidebar.warning("⚠️ Model not found")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Quick Tips:**
- Lower confidence → more detections
- For Mac: Always use CPU mode

""")
