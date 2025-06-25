"""Sequencing and scaling helpers."""
import numpy as np
import pandas as pd

def create_sequences(df: pd.DataFrame, window_size: int, horizon: int, target_col: str):
    """
    Create time-series sequences from a DataFrame:
    X.shape => (num_samples, window_size, num_features)
    y.shape => (num_samples, horizon)
    """
    data = df.values
    X, y = [], []
    for i in range(len(df) - window_size - horizon + 1):
        X_i = data[i : i + window_size, :]
        y_i = data[
            i + window_size : i + window_size + horizon, df.columns.get_loc(target_col)
        ]
        X.append(X_i)
        y.append(y_i)
    return np.array(X), np.array(y)

def inverse_transform_predictions(y_scaled, scaler, df_columns, target_col):
    """
    Given scaled predictions (y_scaled) of shape (num_samples, horizon),
    create a dummy array so we can apply inverse_transform,
    then return just the target column portion in original scale.
    """
    n_samples, horizon = y_scaled.shape
    n_features = len(df_columns)
    target_idx = df_columns.index(target_col)
    dummy = np.zeros((n_samples * horizon, n_features))
    for i in range(horizon):
        dummy[i * n_samples : (i + 1) * n_samples, target_idx] = y_scaled[:, i]
    dummy_inv = scaler.inverse_transform(dummy)
    target_inv = dummy_inv[:, target_idx]
    target_inv = target_inv.reshape(horizon, n_samples).T
    return target_inv
