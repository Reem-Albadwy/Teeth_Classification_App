import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

st.set_page_config(
    page_title="Teeth Disease Classifier",
    page_icon="??",
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #0066cc;
        text-align: center;
        margin-bottom: 20px;
    }
    .footer {
        position: fixed;
        bottom: 10px;
        width: 100%;
        text-align: center;
        color: gray;
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">?? Teeth Disease Classification</div>', unsafe_allow_html=True)
st.write("Upload a teeth image and get the predicted disease with confidence score.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Final_teeth_model.h5")

model = load_model()

class_names = ['MC', 'OLP', 'Gum', 'CoS', 'OT', 'CaS', 'OC']

uploaded_file = st.file_uploader("?? Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img = img.resize((224, 224))  
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    pred_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100

    st.success(f"?? **Predicted Disease:** {pred_class}")
    st.info(f"?? **Confidence:** {confidence:.2f}%")



