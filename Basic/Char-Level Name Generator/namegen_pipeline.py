# namegen_pipeline.py

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
PAD = "<PAD>"
START = "<S>"
END = "<E>"
GENDER_M = "<M>"
GENDER_F = "<F>"

# Sampling helper
def sample_from_probs(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

# Single-name generator
def generate_name(model, char2idx, idx2char, gender_label, temperature=1.0, max_len=20):
    gender_token = GENDER_M if gender_label in (0, 'M', 'm') else GENDER_F
    input_seq = [char2idx[gender_token], char2idx[START]]
    name = ''

    for _ in range(max_len-2):
        padded = pad_sequences([input_seq],
                               maxlen=max_len,
                               padding='post',
                               value=char2idx[PAD])
        preds = model.predict(padded, verbose=0)[0, len(input_seq)-1]
        next_idx = sample_from_probs(preds, temperature)
        next_char = idx2char[next_idx]

        if next_char == END:
            break
        name += next_char
        input_seq.append(next_idx)

    return name.capitalize()

# Batch generator
def generate_names(model, char2idx, idx2char, n=10, gender='F', temperature=1.0):
    label = 1 if gender in ('F', 'f', 1) else 0
    return [generate_name(model, char2idx, idx2char, label, temperature) for _ in range(n)]
