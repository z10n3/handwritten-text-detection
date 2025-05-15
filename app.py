import streamlit as st
import easyocr
import numpy as np
from PIL import Image
from spellchecker import SpellChecker

def load_wordlist(path):
    with open(path, 'r', encoding='utf-8') as f:
        return set(word.strip().lower() for word in f.readlines())

russian_words = load_wordlist("russian_words.txt")
uzbek_words = load_wordlist("uzbek_words.txt")
spell_en = SpellChecker()

def correct_word(word):
    lw = word.lower()
    if lw in spell_en:
        return word
    elif lw in russian_words or lw in uzbek_words:
        return word
    else:
        suggestion = spell_en.correction(word)
        return suggestion if suggestion else word

st.set_page_config(page_title="Handwritten Text Reader", layout="centered")
st.title("Handwritten Text Reader")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    reader = easyocr.Reader(['en', 'ru', 'uz'], gpu=False)
    results = reader.readtext(np.array(image), detail=0)

    corrected_text = []

    for line in results:
        words = line.split()
        corrected_line = ' '.join(correct_word(word) for word in words)
        corrected_text.append(corrected_line)

    final_text = ' '.join(corrected_text)

    st.subheader("Detected & Corrected Text")
    st.write(final_text)
