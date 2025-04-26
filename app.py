import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="Handwritten Text Detection", layout="centered")

st.title("📝 Handwritten Text Detection")
st.write("Upload a photo with handwritten text to extract its contents.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    st.subheader("Detected Text")
    st.code(text.strip() or "[No text detected]", language="text")
