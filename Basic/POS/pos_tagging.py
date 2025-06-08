import spacy

# Load the English NLP pipeline
nlp = spacy.load("en_core_web_sm")

# POS tag color mapping
POS_COLOR_MAP = {
    "NOUN": "#f9c74f",
    "VERB": "#90be6d",
    "ADJ": "#f94144",
    "ADV": "#577590",
    "PRON": "#f9844a",
    "DET": "#43aa8b",
    "ADP": "#277da1",
    "CCONJ": "#9d4edd",
    "PART": "#e76f51",
    "NUM": "#adb5bd",
    "PUNCT": "#ced4da",
    "INTJ": "#d00000",
    "SYM": "#6a4c93",
    "X": "#adb5bd"
}

def pos_tag_sentence(text):
    """Return list of (token, POS) tuples for the input text."""
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

def get_pos_html(text):
    """Return HTML string for POS-tagged visualization of the input text."""
    doc = nlp(text)
    html = ""
    for sent in doc.sents:
        for token in sent:
            word = token.text
            pos = token.pos_
            color = POS_COLOR_MAP.get(pos, "#ffffff")
            html += (
                f'<span title="{pos}" '
                f'style="background-color: {color}; padding:2px 4px; margin:1px; '
                f'border-radius:4px; display:inline-block; cursor:help;">'
                f'{word}</span> '
            )
        html += "<br><br>"
    return html


