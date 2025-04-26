import os
import py7zr
import pandas as pd
import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image

# Set up Tesseract path for Windows or Linux
if os.name == 'nt': 
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else: 
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Set up Streamlit app configuration
st.set_page_config(page_title="Handwritten Text Detection", layout="centered")
st.title("📝 Handwritten Text Detection")
st.write("Upload a photo with handwritten text to extract its contents.")

# Function to extract images from .7z file
def extract_images_from_7z(archive_path, output_folder):
    with py7zr.SevenZipFile(archive_path, mode='r') as z:
        z.extractall(path=output_folder)
    st.success("Images extracted successfully!")

# Folder paths
image_folder = 'images/extracted'
csv_file = 'data/english.csv'

# Extract images from the .7z archive (run this once to extract)
if not os.path.exists(image_folder):
    os.makedirs(image_folder)
    extract_images_from_7z('images/Img.7z', image_folder)

# Display CSV data (if any)
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    st.subheader("CSV Data:")
    st.write(df.head())

# File uploader to choose images for text detection
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Preprocess image for better text detection
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Extract text using pytesseract
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)

    # Display the original image
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # Display detected text
    st.subheader("Detected Text")
    st.code(text.strip() or "[No text detected]", language="text")
