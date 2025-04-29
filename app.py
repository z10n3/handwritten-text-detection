import streamlit as st
import easyocr
from PIL import Image
import os

st.set_page_config(page_title="Text Detection", layout="centered")

st.title("📝 Text Detection with EasyOCR")

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image)

    detected_text = " ".join([text for _, text, _ in results])

    st.subheader("🧾 Detected Text")
    st.write(detected_text)
