import pickle
from tensorflow.keras.models import load_model
from generate import generate_text

# Load mappings and sequence length
with open("saved_models/mappings.pkl", "rb") as f:
    mappings = pickle.load(f)
char_to_idx = mappings["char_to_idx"]
idx_to_char = mappings["idx_to_char"]
SEQ_LENGTH = mappings["SEQ_LENGTH"]

# Load trained model
model = load_model("saved_models/lstm_char_model.keras")

# Generate text
seed = "ROMEO: "  # Or any seed string from your training data
print(generate_text(model, seed, char_to_idx, idx_to_char, SEQ_LENGTH))