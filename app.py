import streamlit as st
import easyocr
import numpy as np
from PIL import Image
from autocorrect import Speller
from spellchecker import SpellChecker

st.set_page_config(page_title="Handwritten Text Reader", layout="centered")

st.title("Handwritten Text Reader")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    reader = easyocr.Reader(['en'])
    results = reader.readtext(np.array(image), detail=0)

    spell = SpellChecker()
    corrected_text = []

    for line in results:
        words = line.split()
        corrected_line = ' '.join([spell.correction(word) if spell.correction(word) else word for word in words])
        corrected_text.append(corrected_line)

    final_text = ' '.join(corrected_text)

    st.subheader("Detected Text")
    st.write(final_text)
    

