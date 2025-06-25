"""
Metrics, plotting, and custom Keras metric for ET LSTM models.
"""
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow.keras.backend as K

def r2_keras(y_true, y_pred):
    """Custom R² metric for Keras."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    ss_res = K.sum(K.square(y_true_f - y_pred_f))
    ss_tot = K.sum(K.square(y_true_f - K.mean(y_true_f)))
    return 1 - ss_res / (ss_tot + K.epsilon())

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred, multioutput="uniform_average")
    mae = mean_absolute_error(y_true, y_pred, multioutput="uniform_average")
    rmse = math.sqrt(mse)
    r2_avg = r2_score(y_true, y_pred, multioutput="uniform_average")
    r2_each = r2_score(y_true, y_pred, multioutput="raw_values")
    return mae, mse, rmse, r2_avg, r2_each

def plot_training_history(history, save_path):
    plt.figure(figsize=(8,5))
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

def plot_time_series_predictions(y_true, y_pred, horizon, station_folder, plots_dir):
    station_plot_dir = os.path.join(plots_dir, station_folder)
    os.makedirs(station_plot_dir, exist_ok=True)
    save_path = os.path.join(station_plot_dir, "test_predictions.png")
    plt.figure(figsize=(10, 8))
    for day_idx in range(horizon):
        ax = plt.subplot(horizon, 1, day_idx + 1)
        ax.plot(y_true[:, day_idx], label="Actual", color="blue")
        ax.plot(y_pred[:, day_idx], label="Predicted", color="red")
        ax.set_title(f"Day +{day_idx+1} Forecast")
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_scatter_day(y_true, y_pred, day_idx, station_folder, plots_dir, dataset_type="test"):
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
    plt.scatter(
        y_true[:, day_idx],
        y_pred[:, day_idx],
        alpha=0.6,
        color="tab:blue",
        label="Predictions",
    )
    plt.title(f"{title_prefix}: Day +{day_idx+1}")
    plt.xlabel("Actual ET")
    plt.ylabel("Predicted ET")
    min_val = min(y_true[:, day_idx].min(), y_pred[:, day_idx].min())
    max_val = max(y_true[:, day_idx].max(), y_pred[:, day_idx].max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linestyle="--",
        label="Ideal",
    )
    mae_d, mse_d, rmse_d, r2_d, _ = compute_metrics(
        y_true[:, day_idx : day_idx + 1], y_pred[:, day_idx : day_idx + 1]
    )
    metrics_text = (
        f"MAE: {mae_d:.2f}\nMSE: {mse_d:.2f}\nRMSE: {rmse_d:.2f}\nR²: {r2_d:.2f}"
    )
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
