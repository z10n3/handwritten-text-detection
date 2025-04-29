import os
import pandas as pd
import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image

# Set up Tesseract path (only for local running; not needed on Streamlit Cloud)
if os.name == 'nt': 
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else: 
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Set up Streamlit app configuration
st.set_page_config(page_title="Handwritten Text Detection", layout="centered")
st.title("📝 Handwritten Text Detection")
st.write("Upload a photo with handwritten text to extract its contents.")

# Folder paths
image_folder = 'images/extracted'
csv_file = 'data/english.csv'

# Display CSV data (if it exists)
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    st.subheader("CSV Data (Example Words):")
    st.write(df.head())

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Preprocess image for better OCR
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # OCR with Tesseract
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)

    # Display original image
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # Display extracted text
    st.subheader("Detected Text")
    st.code(text.strip() or "[No text detected]", language="text")
else:
    st.info("Please upload an image to start text detection.")
