# streamlit_garbage_classifier.py
# Streamlit app to classify garbage images with three input sources:
# 1) Upload from local file
# 2) Image URL
# 3) Real-time camera (frame capture or streaming)

import io
import time
from typing import List, Tuple

import numpy as np
import PIL.Image as Image
import streamlit as st

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

# Optional: OpenCV only used for streaming; if not installed streaming will be disabled
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

# ---------------------- USER CONFIG ----------------------
# Path to the uploaded model file (provided by user). Edit this if needed.
MODEL_PATH = "model/garbage_classifier_model.pth"

# Default class names (edit to match your trained model order)
CLASS_NAMES = [
    'battery', 
    'biological', 
    'cardboard', 
    'clothes',
    'glass', 
    'metal', 
    'paper', 
    'plastic', 
    'shoes', 
    'trash'
]

# Image size expected by model (if unknown, 224 is a safe default for ResNet-like models)
IMG_SIZE = 224
# ---------------------------------------------------------

st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("🗑️ Garbage Classifier — Upload / URL / Camera")

# Sidebar: options and model load
st.sidebar.header("Model & Settings")
st.sidebar.write(f"Model path: `{MODEL_PATH}`")

use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=False)
show_raw = st.sidebar.checkbox("Show preprocessed image", value=False)

# Allow user to override class names if needed
if st.sidebar.checkbox("Edit class names (click to edit)"):
    txt = st.sidebar.text_area("Class names (comma separated)", ", ".join(CLASS_NAMES))
    try:
        CLASS_NAMES = [s.strip() for s in txt.split(",") if s.strip()]
    except Exception:
        st.sidebar.error("Couldn't parse class names — keep defaults.")

# Device
device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

@st.cache_resource
def load_checkpoint(path: str, device: torch.device):
    """Attempt to load a model checkpoint robustly.
    This function tries several heuristics:
    - If the file contains a full nn.Module, use it directly.
    - If it contains a state_dict, try to adapt a ResNet18 backbone and load weights.
    - If keys indicate another layout, attempt to map fc/classifier weights.
    If none works, raise an informative error.
    """
    checkpoint = torch.load(path, map_location=device)

    # If user saved the full model object
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        model.to(device)
        model.eval()
        return model

    # If it's a dict with 'state_dict'
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    else:
        # assume checkpoint itself is a state_dict
        state = checkpoint if isinstance(checkpoint, dict) else None

    if state is None:
        raise RuntimeError("Loaded checkpoint is not a state_dict nor an nn.Module. Please provide a compatible file.")

    # Try to detect final layer name and out_features from state_dict weight shapes
    # Common keys: 'fc.weight', 'classifier.0.weight', 'classifier.weight'
    final_weight_key = None
    for key in state.keys():
        if key.endswith('fc.weight') or key.endswith('classifier.weight') or key.endswith('head.weight'):
            final_weight_key = key
            break

    out_features = None
    if final_weight_key is not None:
        out_features = state[final_weight_key].shape[0]

    # Build a ResNet18 backbone by default and adapt
    model = models.resnet18(pretrained=False)
    if out_features is not None:
        model.fc = nn.Linear(model.fc.in_features, out_features)
    else:
        # fallback to default number of classes equal to length of CLASS_NAMES
        model.fc = nn.Linear(model.fc.in_features, max(1, len(CLASS_NAMES)))

    # Try to load state dict — be flexible about key name prefixes
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # attempt to strip 'module.' prefix (from DataParallel) and reload
        new_state = {}
        for k, v in state.items():
            new_key = k.replace('module.', '')
            new_state[new_key] = v
        try:
            model.load_state_dict(new_state)
        except Exception as e:
            # last resort: load strict=False
            model.load_state_dict(new_state, strict=False)

    model.to(device)
    model.eval()
    return model

# Load model with a spinner
with st.spinner("Loading model..."):
    try:
        model = load_checkpoint(MODEL_PATH, device)
        st.sidebar.success("Model loaded")
    except Exception as e:
        st.sidebar.error(f"Failed loading model: {e}")
        st.stop()

# Preprocessing transform
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Prediction function
@st.cache_data
def predict_image(img: Image.Image) -> Tuple[str, List[Tuple[str, float]]]:
    img_rgb = img.convert('RGB')
    inp = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        if isinstance(out, (tuple, list)):
            out = out[0]
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]

    # Map to class names; if sizes mismatch, trim or extend
    n = len(probs)
    names = CLASS_NAMES
    if len(names) != n:
        if len(names) < n:
            # generate generic names
            names = [f'class_{i}' for i in range(n)]
        else:
            names = names[:n]

    pairs = list(zip(names, [float(p) for p in probs]))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[0][0], pairs

# ---- UI: Input selection ----
st.header("Choose input source")
input_mode = st.radio("Input mode", 
                      ["Upload from device", 
                       "Image URL", 
                       "Camera (single capture)",
                       "Camera (real-time stream)"])

col1, col2 = st.columns([1, 3])
result_image = None

if input_mode == "Upload from device":
    uploaded = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    if uploaded is not None:
        image = Image.open(uploaded)
        result_image = image

elif input_mode == "Image URL":
    url = st.text_input("Image URL (http/https)")
    if url:
        try:
            from urllib.request import urlopen
            with urlopen(url) as response:
                data = response.read()
            image = Image.open(io.BytesIO(data))
            result_image = image
        except Exception as e:
            st.error(f"Failed to fetch image from URL: {e}")

elif input_mode == "Camera (single capture)":
    # st.camera_input returns a file-like object
    cam_file = st.camera_input("Take a picture")
    if cam_file is not None:
        image = Image.open(cam_file)
        result_image = image

elif input_mode == "Camera (real-time stream)":
    if not OPENCV_AVAILABLE:
        st.error("OpenCV not available in the environment: install opencv-python to use real-time streaming.")
    else:
        st.write("Real-time camera stream. Click 'Start stream' then press 'Stop' to end.")
        run_stream = st.button("Start stream")
        stop_stream = st.button("Stop")

        # We'll create a placeholder for streaming frames
        FRAME_PLACEHOLDER = st.empty()
        if run_stream:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open camera")
            else:
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Failed to read frame")
                            break
                        # Convert BGR -> RGB
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)

                        # Run prediction on every Nth frame to reduce load (optional)
                        top1, pairs = predict_image(pil_img)

                        # Overlay label on frame
                        label = f"{top1} ({pairs[0][1]*100:.1f}%)"
                        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        FRAME_PLACEHOLDER.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

                        # break if user pressed Stop button
                        if st.button("Stop"):
                            break
                finally:
                    cap.release()
                    FRAME_PLACEHOLDER.empty()

# If we got an image to classify (from upload, url, or single camera capture)
if result_image is not None:
    with col1:
        st.subheader("Input")
        st.image(result_image, use_column_width=True)
        if show_raw:
            st.subheader("Preprocessed (tensor)")
            t = transform(result_image)
            st.write(t.shape)

    with col2:
        st.subheader("Prediction")
        with st.spinner("Running inference..."):
            try:
                label, probs = predict_image(result_image)
                st.markdown(f"### Predicted: **{label}**")
                st.write("Confidence scores:")
                for name, p in probs:
                    st.write(f"- {name}: {p*100:.2f}%")
            except Exception as e:
                st.error(f"Failed to run inference: {e}")

st.markdown("---")
st.caption("Notes: If your model architecture is not ResNet-like or the saved file is only a state_dict compatible with a custom class, you may need to replace the `load_checkpoint` routine with a loader for your model class. Edit CLASS_NAMES and IMG_SIZE at the top to match training settings.")

st.write("Dependencies: torch, torchvision, pillow, streamlit. For real-time streaming install opencv-python.")

# Footer: quick run instructions
st.write("**How to run:** `pip install -r requirements.txt` then `streamlit run streamlit_garbage_classifier.py`\n\nMake sure the model file is at the path specified by MODEL_PATH.")
