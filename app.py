import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("satellite_classifier_dropout.h5")

labels = ["Forest", "Water", "Urban", "Agriculture"]
IMG_SIZE = 64

st.title("🛰️ Satellite Image Land Classification")
st.write("Upload a satellite image to classify land type.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_resized = img_resized / 255.0
    img_resized = np.reshape(img_resized, (1, IMG_SIZE, IMG_SIZE, 3))

    if st.button("Predict"):
        prediction = model.predict(img_resized)[0]

        st.subheader("Prediction Results")
        for label, prob in zip(labels, prediction):
            st.write(f"{label} → {prob*100:.2f}%")
