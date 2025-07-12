# model_utils.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_model(input_shape, num_chars, lstm_units=256):
    model = Sequential([
        LSTM(lstm_units, input_shape=input_shape),
        Dense(num_chars, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

def get_callbacks():
    return [
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=2, verbose=1)
    ]
