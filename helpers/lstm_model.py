"""
LSTM model creation utility.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from helpers.metrics_and_plots import r2_keras


def build_lstm_model(input_shape, horizon, lr=0.001):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, activation="tanh"))
    model.add(Dense(horizon))
    model.compile(
        loss="mse",
        optimizer=Adam(learning_rate=lr),
        metrics=["mae", r2_keras],
    )
    return model
