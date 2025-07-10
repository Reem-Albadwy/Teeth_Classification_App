import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

st.set_page_config(
    page_title="Teeth Disease Classifier",
    page_icon="🦷",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown('<div style="text-align:center; font-size:36px; font-weight:bold; color:#0066cc;">🦷 Teeth Disease Classification</div>', unsafe_allow_html=True)
st.write("Upload a teeth image and get the predicted disease with confidence score.")

@st.cache_resource
def load_model():
    model_path = "SavedModel_format"
    if not os.path.exists(model_path):
        st.error("Model folder not found!")
        return None
    # تحميل الموديل كـ tf.saved_model.Loader
    loaded = tf.saved_model.load(model_path)
    return loaded

model = load_model()

class_names = ['OC', 'CaS', 'OT', 'CoS', 'Gum', 'MC', 'OLP']

uploaded_file = st.file_uploader("🖼️ Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    infer = model.signatures["serving_default"]
    input_tensor = tf.convert_to_tensor(img_array)
    predictions = infer(input_tensor)
    key = list(predictions.keys())[0]
    preds = predictions[key].numpy()

    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds)) * 100

    st.success(f"🩺 **Predicted Disease:** {pred_class}")
    st.info(f"📊 **Confidence:** {confidence:.2f}%")
