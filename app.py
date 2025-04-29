import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr

# Streamlit page setup
st.set_page_config(page_title="Handwritten Text Detection", layout="centered")
st.title("📝 Handwritten Text Detection with EasyOCR")
st.write("Upload an image with handwritten text to extract its contents.")

# Optional: show CSV if it exists
csv_path = 'data/english.csv'
if csv_path and os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.subheader("📄 CSV Sample Data")
    st.write(df.head())

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Display original image
    st.subheader("🖼️ Uploaded Image")
    st.image(image, use_container_width=True)

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])  # Add other language codes if needed

    # Run OCR
    with st.spinner("🔍 Detecting text..."):
        results = reader.readtext(image_np)

    # Show extracted text
    st.subheader("📃 Detected Text")
    if results:
        for i, (bbox, text, confidence) in enumerate(results, 1):
            st.write(f"**{i}.** {text} (Confidence: {confidence:.2f})")
    else:
        st.write("No text detected.")
