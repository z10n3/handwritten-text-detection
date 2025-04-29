import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import pandas as pd

# Streamlit UI setup
st.set_page_config(page_title="EasyOCR Handwritten Text Detection", layout="centered")
st.title("✍️ Handwritten Text Detection with EasyOCR")
st.write("Upload a photo of handwritten text and see the extracted content.")

# Optional: show CSV if it exists
csv_path = 'data/english.csv'
if csv_path and os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.subheader("📄 CSV Sample Data")
    st.write(df.head())

# Image uploader
uploaded_file = st.file_uploader("📷 Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("🖼️ Uploaded Image")
    st.image(image, use_container_width=True)

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=False)  # Use 'gpu=True' if CUDA is available

    # Convert PIL to NumPy for EasyOCR
    image_np = np.array(image)

    # OCR process
    result = reader.readtext(image_np)

    # Extract and display text
    extracted_text = "\n".join([item[1] for item in result])

    st.subheader("🧾 Detected Text")
    st.code(extracted_text.strip() or "[No text detected]", language="text")
