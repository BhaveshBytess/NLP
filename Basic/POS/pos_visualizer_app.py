# pos_visualizer_app.py

import spacy
import streamlit as st
from spacy import displacy

# Load spaCy's model
nlp = spacy.load("en_core_web_sm")

# Custom style mapping for POS tags
POS_COLOR_MAP = {
    "NOUN": "#fde68a", "VERB": "#fca5a5", "ADJ": "#6ee7b7", "ADV": "#93c5fd",
    "PRON": "#fcd34d", "PROPN": "#ddd6fe", "INTJ": "#fbcfe8", "ADP": "#bae6fd",
    "CCONJ": "#fecaca", "SCONJ": "#bbf7d0", "NUM": "#fdba74", "PART": "#c4b5fd",
    "SYM": "#f0abfc", "X": "#e0e7ff", "PUNCT": "#ffffff", "DET": "#fef08a"
}

def get_pos_html(doc):
    html = ""
    for sent in doc.sents:
        for token in sent:
            color = POS_COLOR_MAP.get(token.pos_, "#ffffff")
            html += (
                f'<span title="{token.pos_}" '
                f'style="background-color: {color}; padding:3px 6px; margin:2px; '
                f'border-radius:4px; display:inline-block; cursor:help;">'
                f'{token.text}</span> '
            )
        html += "<br><br>"
    return html

# Streamlit UI
st.title("🧠 POS Tag Visualizer")
text_input = st.text_area("Enter a sentence or paragraph:")

if st.button("Visualize POS Tags"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        doc = nlp(text_input)
        html = get_pos_html(doc)
        st.markdown(html, unsafe_allow_html=True)
