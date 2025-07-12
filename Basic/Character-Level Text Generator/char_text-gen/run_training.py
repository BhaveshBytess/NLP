# run_training.py

from tensorflow.keras.utils import get_file
from config import *
from data_utils import prepare_dataset
from model_utils import build_model, get_callbacks
from generate import generate_text

# Load data
path = get_file('shakespeare.txt', origin='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
with open(path, 'r', encoding='utf-8') as f:
    text = f.read().lower()

# Prepare data
X, y, char_to_idx, idx_to_char = prepare_dataset(text, SEQ_LENGTH, STEP)

# Build model
model = build_model((SEQ_LENGTH, len(char_to_idx)), len(char_to_idx), LSTM_UNITS)

# Train
history = model.fit(
    X, y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    callbacks=get_callbacks()
)

# Model saving
model.save("saved_models/lstm_char_model.keras")

# Generate sample
seed = text[:SEQ_LENGTH]
print("\n🔮 Generated Text:\n")
print(generate_text(model, seed, char_to_idx, idx_to_char))

# Save mappings and sequence length
import pickle

with open("saved_models/mappings.pkl", "wb") as f:
    pickle.dump({
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "SEQ_LENGTH": SEQ_LENGTH
    }, f)
    