"""Common helper functions for ET and rainfall training scripts."""
import math
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from config import (
    BASE_DIR,
    START_YEAR,
    END_YEAR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    WINDOW_SIZE,
    HORIZON,
)


# -------------------------------
# Custom R² Metric
# -------------------------------

def r2_keras(y_true, y_pred):
    """Custom R² metric for Keras."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    ss_res = K.sum(K.square(y_true_f - y_pred_f))
    ss_tot = K.sum(K.square(y_true_f - K.mean(y_true_f)))
    return 1 - ss_res / (ss_tot + K.epsilon())


# -------------------------------
# Data Helpers
# -------------------------------

def load_station_data(station_folder):
    """Load ``all_years_data.csv`` for a station and filter by year."""
    csv_path = os.path.join(BASE_DIR, station_folder, "all_years_data.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Station: {station_folder}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        if START_YEAR is not None:
            df = df[df["Date"].dt.year >= START_YEAR]
        if END_YEAR is not None:
            df = df[df["Date"].dt.year <= END_YEAR]
        df.reset_index(drop=True, inplace=True)
    return df


def split_by_percentages(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    """Chronologically split a DataFrame into train/val/test portions."""
    if df.empty or "Date" not in df.columns:
        return df, pd.DataFrame(), pd.DataFrame()

    df_sorted = df.sort_values("Date").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    df_train = df_sorted.iloc[:train_end].copy()
    df_val = df_sorted.iloc[train_end:val_end].copy()
    df_test = df_sorted.iloc[val_end:].copy()

    return df_train, df_val, df_test


def create_sequences(df, window_size, horizon, target_col):
    """Create rolling window sequences from a feature DataFrame."""
    data = df.values
    X, y = [], []
    for i in range(len(df) - window_size - horizon + 1):
        X_i = data[i : i + window_size, :]
        y_i = data[i + window_size : i + window_size + horizon, df.columns.get_loc(target_col)]
        X.append(X_i)
        y.append(y_i)
    return np.array(X), np.array(y)


def inverse_transform_predictions(y_scaled, scaler, df_columns, target_col):
    """Inverse-transform scaled predictions back to original units."""
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


def compute_metrics(y_true, y_pred):
    """Return MAE, MSE, RMSE, average R² and per-horizon R²."""
    mse = mean_squared_error(y_true, y_pred, multioutput="uniform_average")
    mae = mean_absolute_error(y_true, y_pred, multioutput="uniform_average")
    rmse = math.sqrt(mse)
    r2_avg = r2_score(y_true, y_pred, multioutput="uniform_average")
    r2_each = r2_score(y_true, y_pred, multioutput="raw_values")
    return mae, mse, rmse, r2_avg, r2_each


# -------------------------------
# Plotting Helpers
# -------------------------------

def plot_time_series_predictions(y_true, y_pred, horizon, station_folder, plots_dir, variable_name):
    """Plot actual vs. predicted values for each day in the horizon."""
    station_plot_dir = os.path.join(plots_dir, station_folder)
    os.makedirs(station_plot_dir, exist_ok=True)
    save_path = os.path.join(station_plot_dir, "test_predictions.png")

    plt.figure(figsize=(10, 8))
    for day_idx in range(horizon):
        ax = plt.subplot(horizon, 1, day_idx + 1)
        ax.plot(y_true[:, day_idx], label="Actual", color="blue")
        ax.plot(y_pred[:, day_idx], label="Predicted", color="red")
        ax.set_title(f"Day +{day_idx+1} {variable_name} Forecast")
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_scatter_day(y_true, y_pred, day_idx, station_folder, plots_dir, variable_name, dataset_type="test"):
    """Create a scatter plot for one forecast day."""
    station_plot_dir = os.path.join(plots_dir, station_folder)
    os.makedirs(station_plot_dir, exist_ok=True)

    if dataset_type == "test":
        filename = f"scatter_day{day_idx+1}.png"
        title_prefix = "Scatter Plot"
    else:
        filename = f"{dataset_type}_scatter_day{day_idx+1}.png"
        title_prefix = f"{dataset_type.capitalize()} Scatter"

    save_path = os.path.join(station_plot_dir, filename)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true[:, day_idx], y_pred[:, day_idx], alpha=0.6, color="tab:blue", label="Predictions")
    plt.title(f"{title_prefix}: Day +{day_idx+1} {variable_name}")
    plt.xlabel(f"Actual {variable_name}")
    plt.ylabel(f"Predicted {variable_name}")

    min_val = min(y_true[:, day_idx].min(), y_pred[:, day_idx].min())
    max_val = max(y_true[:, day_idx].max(), y_pred[:, day_idx].max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal")

    mae_d, mse_d, rmse_d, r2_d, _ = compute_metrics(
        y_true[:, day_idx : day_idx + 1], y_pred[:, day_idx : day_idx + 1]
    )
    metrics_text = f"MAE: {mae_d:.2f}\nMSE: {mse_d:.2f}\nRMSE: {rmse_d:.2f}\nR²: {r2_d:.2f}"
    plt.gca().text(
        0.05,
        0.95,
        metrics_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_history(history, save_path):
    """Plot training and validation loss/R² over epochs."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    if "r2_keras" in history.history:
        plt.plot(history.history["r2_keras"], label="Train R2")
    if "val_r2_keras" in history.history:
        plt.plot(history.history["val_r2_keras"], label="Val R2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / R²")
    plt.title("Training History (Loss and R²)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
