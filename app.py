import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import gdown

# ----------------------------
# Google Drive Model Settings
# ----------------------------
MODEL_PATH = "rice_disease_model.h5"
DRIVE_FILE_ID = "1syroRsKo08V2-tqF-Amvu4PZ4zB8Vxgf"  # Replace with your ID

# Download the model if it doesn't exist locally
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="RiceGuard - AI Disease Detection",
    page_icon="🌾",
    layout="wide"
)

# ----------------------------
# Disease Info Dictionary
# ----------------------------
disease_info = {
    "Bacterial Leaf Blight": {
        "description": "Causes yellowing and wilting of leaves. It can significantly reduce yield.",
        "symptoms": "Water-soaked streaks, yellow lesions, milky dew drops.",
        "treatment": "Use copper-based fungicides and avoid excess nitrogen fertilizer.",
        "icon": "🦠"
    },
    "Brown Spot": {
        "description": "A fungal disease causing brown circular spots on leaves.",
        "symptoms": "Round to oval brown spots with yellow halo.",
        "treatment": "Improve soil nutrients and treat seeds with fungicides.",
        "icon": "🍂"
    },
    "Leaf Blast": {
        "description": "Causes diamond-shaped lesions on rice leaves.",
        "symptoms": "Spindle-shaped spots with gray centers.",
        "treatment": "Use systemic fungicides and resistant varieties.",
        "icon": "🔥"
    },
    "Leaf Smut": {
        "description": "Creates black powdery masses on leaves.",
        "symptoms": "Small, black, linear lesions on leaf blades.",
        "treatment": "Remove infected debris and apply fungicides if severe.",
        "icon": "🌑"
    },
    "Narrow Brown Leaf Spot": {
        "description": "Causes long narrow brown streaks, usually late in the season.",
        "symptoms": "Short, linear, narrow brown streaks.",
        "treatment": "Use resistant varieties and balanced fertilization.",
        "icon": "🌾"
    },
    "Healthy Rice Leaf": {
        "description": "Plant is in good condition with no visible disease.",
        "symptoms": "Green, vibrant leaves with no spots.",
        "treatment": "Continue regular monitoring and maintenance.",
        "icon": "✅"
    }
}

# ----------------------------
# Load Model Function
# ----------------------------
@st.cache_resource
def load_system():
    if not os.path.exists(MODEL_PATH):
        st.warning("Model not found!")
        return None, None

    model = tf.keras.models.load_model(MODEL_PATH)

    # Load class names
    if os.path.exists('class_names.json'):
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
    else:
        class_names = list(disease_info.keys())

    return model, class_names


# ----------------------------
# Image Preprocessing (No cv2)
# ----------------------------
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)


# ----------------------------
# Main App
# ----------------------------
def main():
    st.sidebar.title("🌾 Navigation")
    app_mode = st.sidebar.radio("Go to", ["Home", "Disease Detection", "Gallery & Info"])

    model, class_names = load_system()
    if model is None:
        return

    # ---------------- Home Page ----------------
    if app_mode == "Home":
        st.markdown("<h1 style='text-align:center;'>RiceGuard - AI Disease Detection</h1>", 
                    unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>Upload a rice leaf image to detect diseases.</p>",
                    unsafe_allow_html=True)

    # ---------------- Disease Detection ----------------
    elif app_mode == "Disease Detection":
        st.markdown("## 🔎 Detect Disease")
        uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption="Uploaded Leaf", use_container_width=True)

            if st.button("Analyze Now"):
                img_array = preprocess_image(uploaded_file)

                # Predict
                predictions = model.predict(img_array)
                predicted_idx = np.argmax(predictions[0])
                predicted_label = class_names[predicted_idx].strip()
                confidence = round(100 * np.max(predictions[0]), 2)

                # Disease info
                info = disease_info.get(predicted_label, {})
                icon = info.get("icon", "🌱")
                desc = info.get("description", "No description available.")
                treat = info.get("treatment", "Consult an agricultural expert.")

                if "Healthy" in predicted_label:
                    st.success(f"{icon} {predicted_label} (Confidence: {confidence}%)\n\n{desc}")
                else:
                    st.error(
                        f"{icon} {predicted_label} Detected (Confidence: {confidence}%)\n\n"
                        f"**Description:** {desc}\n\n"
                        f"**Treatment:** {treat}"
                    )

    # ---------------- Gallery & Info ----------------
    elif app_mode == "Gallery & Info":
        st.markdown("## 🌿 Disease Gallery")

        for disease, info in disease_info.items():
            with st.expander(f"{info['icon']} {disease}"):
                st.write(f"**About:** {info['description']}")
                st.write(f"**Symptoms:** {info['symptoms']}")
                st.write(f"**Treatment:** {info['treatment']}")


# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    main()
