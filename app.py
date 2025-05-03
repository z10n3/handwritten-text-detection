import streamlit as st
from PIL import Image
import easyocr
from spellchecker import SpellChecker

st.set_page_config(page_title="Handwritten Text Detection", layout="centered")
st.title("🧾 Detected Text")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    reader = easyocr.Reader(['en'])
    results = reader.readtext(np.array(image), detail=1)

    words = [text for (_, text, confidence) in results]
    spell = SpellChecker()
    corrected_words = [spell.correction(word) if word.isalpha() else word for word in words]

    detected_text = " ".join(corrected_words)
    st.markdown(f"**{detected_text}**")
