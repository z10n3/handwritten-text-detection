import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Handwritten Text Reader", layout="centered")

st.title("✍️ Handwritten Text Detector")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(np.array(image), detail=0)

    full_text = " ".join(results)
    st.markdown("### 🧾 Detected Text")
    st.write(full_text)
