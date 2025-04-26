import streamlit as st
import cv2
import os
import pytesseract
import numpy as np
from PIL import Image

if os.name == 'nt': 
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else: 
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

st.set_page_config(page_title="Handwritten Text Detection", layout="centered")

st.title("📝 Handwritten Text Detection")
st.write("Upload a photo with handwritten text to extract its contents.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Предобработка изображения
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    # Можно использовать адаптивную бинаризацию для лучшего контраста
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Другие фильтры (например, эрозия) для очистки изображения
    kernel = np.ones((1, 1), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    # Конфигурация Tesseract для OCR
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)

    # Отображение оригинального изображения
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # Отображение распознанного текста
    st.subheader("Detected Text")
    if text.strip():
        st.code(text.strip(), language="text")
    else:
        st.warning("No text detected. Please try another image or improve the quality of the image.")
