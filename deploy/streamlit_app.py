"""
DFU Detection - Streamlit Demo
Deploys YOLO model for diabetic foot ulcer detection
Optimized for Mac (CPU inference)
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
from ultralytics import YOLO


# --- Helper to cache model loading ---
@st.cache_resource
def load_model(model_path: str):
    """Load YOLO model with caching"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = YOLO(model_path)
    return model


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


# --- Streamlit UI ---
st.title("🩺 DFU Detection – Demo")
st.markdown("Upload a foot image to detect diabetic foot ulcers")

st.sidebar.header("⚙️ Model & Settings")

# Model path input
default_model_path = "/Users/rezabasiri/Documents/yolo/best.pt"
model_path = st.sidebar.text_input("Model path (.pt)", value=default_model_path)

# CPU option (recommended for Mac)
run_on_cpu = st.sidebar.checkbox("Force CPU (recommended on Mac)", value=True)

# Image size (IMPORTANT: Must match training!)
img_size = st.sidebar.selectbox(
    "Image size",
    [640, 1024, 1280, 1536],
    index=1,  # Default to 1024 (best balance for high-res images)
    help="Larger = better detail but slower. Use 1024 for most cases."
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
        with st.spinner("Loading model..."):
            try:
                model = load_model(model_path)
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                st.stop()

        # Load image
        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)

        # Run inference with correct image size
        device = 'cpu' if run_on_cpu else 'cuda'
        with st.spinner(f"Running inference (img_size={img_size})..."):
            try:
                # CRITICAL: imgsz must match training size!
                results = model(
                    image_np,
                    imgsz=img_size,  # Match training size (640 or 1024)
                    device=device,
                    conf=conf_threshold,
                    verbose=False
                )
            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.stop()

        # Get first result (single image)
        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            st.warning("⚠️ No DFUs detected.")
            st.info(f"Try lowering the confidence threshold (current: {conf_threshold})")
        else:
            # Draw boxes on image
            drawn = image.copy()
            draw = ImageDraw.Draw(drawn)
            font = get_font(20)

            detections_info = []

            for i, box in enumerate(boxes):
                # Get box coordinates [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

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

            # Show results
            st.success(f"✅ Detected {len(boxes)} DFU(s)")
            st.image(drawn, caption="Detected DFUs", use_container_width=True)

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

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    st.sidebar.info(f"📦 Model size: {file_size:.1f} MB")

    # Try to detect model type from architecture
    try:
        temp_model = load_model(model_path)
        # Count parameters (rough detection)
        total_params = sum(p.numel() for p in temp_model.model.parameters())

        if total_params < 5_000_000:
            model_type = "yolov8n (nano)"
        elif total_params < 15_000_000:
            model_type = "yolov8s (small)"
        elif total_params < 30_000_000:
            model_type = "yolov8m (medium) ⚠️"
        elif total_params < 50_000_000:
            model_type = "yolov8l (large)"
        else:
            model_type = "yolov8x (extra-large)"

        st.sidebar.info(f"🤖 Detected: {model_type}")
        st.sidebar.caption(f"Parameters: {total_params:,}")

        # Recommendation based on model type
        if "medium" in model_type:
            st.sidebar.info("💡 Trained @ 640px, but 1024px often works better for high-res images")
        elif "extra-large" in model_type:
            st.sidebar.success("💡 Use img_size=1024+ for best results")

    except Exception as e:
        st.sidebar.error(f"Could not read model: {e}")
else:
    st.sidebar.warning("⚠️ Model not found")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Quick Tips:**
- **1024px**: Best default for high-res images
- **1280/1536px**: Max detail (slower, 15-30+ sec)
- **640px**: Fast but may miss small ulcers
- Lower confidence → more detections
""")
