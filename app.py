import os
import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import pandas as pd
import torch

# Load TrOCR model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Streamlit UI setup
st.set_page_config(page_title="TrOCR Handwritten Text Recognition", layout="centered")
st.title("✍️ Handwritten Text Detection with TrOCR")
st.write("Upload an image of handwritten text to extract it using a Transformer model.")

# Optional: show CSV if needed
csv_path = 'data/english.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.subheader("📄 Sample CSV Data")
    st.write(df.head())

# Upload image
uploaded_file = st.file_uploader("📷 Upload a handwritten image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Show image
    st.subheader("🖼️ Uploaded Image")
    st.image(image, use_container_width=True)

    # Preprocess and infer
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Display result
    st.subheader("🧾 Detected Text")
    st.code(generated_text.strip(), language="text")
