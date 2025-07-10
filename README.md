# 🦷 Teeth Disease Classifier

This project is a deep learning application for classifying 7 different teeth-related diseases using a fine-tuned MobileNetV2 model. The model was trained on a custom dataset, with augmentation and fine-tuning for improved accuracy.

---

## 🚀 Live Demo

👉 [Try the Streamlit App Here](https://teethclassificationapp-mertpqfappgs2kcexyuavj2.streamlit.app/)

---

## 📂 Project Structure

- `app.py` — Streamlit app for image upload and prediction
- `SavedModel_format/` — The trained model in TensorFlow SavedModel format
- `requirements.txt` — dependencies to run the project

---

## 🧠 Model Details

- **Base model**: MobileNetV2 (pretrained on ImageNet)
- **Fine-tuning**: Last 30 layers unfrozen
- **Input size**: 224x224 RGB images
- **Number of classes**: 7  
  Classes: `CaS`, `CoS`, `Gum`, `MC`, `OC`, `OLP`, `OT`
- **Framework**: TensorFlow 2.10.1
