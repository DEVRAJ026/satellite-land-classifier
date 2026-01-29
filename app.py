import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("satellite_classifier_dropout.h5")

# ⚠️ NOTE:
# These labels must match the order used during training.
# If your model is actually 10-class (EuroSAT), we will fix this later.
labels = ["Forest", "Water", "Urban", "Agriculture"]

IMG_SIZE = 64

st.title("🛰️ Satellite Image Land Classification")
st.write("Upload a satellite image to classify land type.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    # --- Load image using PIL (NO OpenCV) ---
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # --- Preprocess image ---
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    if st.button("Predict"):
        prediction = model.predict(img_array)[0]

        st.subheader("Prediction Results")
        for label, prob in zip(labels, prediction):
            st.write(f"{label} → {prob * 100:.2f}%")


