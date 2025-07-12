# data_utils.py

import numpy as np

def prepare_dataset(text, seq_length=100, step=1):
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}

    sequences = []
    next_chars = []

    for i in range(0, len(text) - seq_length, step):
        sequences.append(text[i:i+seq_length])
        next_chars.append(text[i+seq_length])

    X = np.zeros((len(sequences), seq_length, len(chars)), dtype=np.bool_)
    y = np.zeros((len(sequences), len(chars)), dtype=np.bool_)

    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            X[i, t, char_to_idx[char]] = 1
        y[i, char_to_idx[next_chars[i]]] = 1

    return X, y, char_to_idx, idx_to_char
