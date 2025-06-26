"""
Multi-Station, Multi-Step LSTM Forecast for Rainfall (mm).

For each farm the records are split chronologically: 70% for training,
15% for validation and the remaining 15% for testing.  Models may be
trained individually per farm or using the combined data.
"""

import pickle
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime

# Concurrency
from concurrent.futures import ProcessPoolExecutor, as_completed

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Limit TensorFlow thread usage per process
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# Matplotlib
import matplotlib.pyplot as plt

# Apply a professional plotting style and consistent fonts
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12})

# -------------------------------
# 1. Configuration
# -------------------------------

STATION_FOLDERS = [
    "Kahuku_Farm",
    "Nozawa_Farms",
    "Kuilima_Farms",
    "Cabaero_Farms",  # Corresponds to Cabaero Farms (lat: 20.8425, lng: -156.3471)
    "Kupaa_Farms",  # Corresponds to Kupa'a Farms (lat: 20.7658, lng: -156.3513)
    "MAO_Organic_Farms_(new_site)",  # From MA'O Organic Farms (original site)
    "MAO_Organic_Farms_(original_site)",  # From MA'O Organic Farms (new site)
    "2K_Farm_LLC",  # From 2K Farm LLC
    "Wong_Hon_Hin_Inc",  # From Wong Hon Hin Inc
    "Hawaii_Taro_Farm_LLC",  # From Hawaii Taro Farm, LLC
    "Hawaii_Seed_Pro_LLC_Farm",  # From Hawaii Seed Pro LLC Farm
    "Cabaero_Farm",  # Corresponds to Cabaero Farm (lat: 20.791703, lng: -156.358194)
    "Kupaa_Farms2",  # Second instance of Kupaa Farms (lat: 20.765515, lng: -156.35185)
    "Hirako_Farm",  # First instance of Hirako Farm
    "Hirako_Farm1",  # Second instance of Hirako Farm
    "Anoano_Farms",  # From Anoano Farms
]

# Train models individually for each farm when True. When False, all data is
# combined as before.
TRAIN_PER_FARM = True

# Directory to store trained models and scalers when TRAIN_PER_FARM is enabled
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Base directory for station folders
BASE_DIR = "farm_data"

# Ratios for chronological split of each farm's data
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# -------------------------------
# Year Filtering
# -------------------------------
# Define the year range to use.  Set to ``None`` to omit the bound.
# The filter is applied prior to splitting the data.
START_YEAR = 2000  # e.g. 2018
END_YEAR = 2025  # e.g. 2022

# Input sequence length (days to look back)
WINDOW_SIZE = 24

# Forecast horizon (days to predict)
HORIZON = 3

MAX_WORKERS = 8  # Number of parallel processes for training
# Target column name for rainfall forecasting
TARGET_COL = "Rainfall (mm)"

# Random seed for reproducibility
RANDOM_SEED = 42

# Where to save the trained model
MODEL_PATH = "model_rain_lstm.h5"

# Where to save rainfall plots and results
PLOTS_DIR = (
    "plots_test_rainfall"
    if START_YEAR is None
    else f"plots_test_rainfall_{START_YEAR}-{END_YEAR}"
)
PLOTS_DIR = os.path.join("plots", PLOTS_DIR)
os.makedirs(PLOTS_DIR, exist_ok=True)


# -------------------------------
# 2. Custom R² Metric
# -------------------------------
def r2_keras(y_true, y_pred):
    """
    Custom R² metric for Keras.
    For multi-output (shape=[batch, horizon]), we flatten both
    y_true and y_pred to compute an overall R².
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    ss_res = K.sum(K.square(y_true_f - y_pred_f))
    ss_tot = K.sum(K.square(y_true_f - K.mean(y_true_f)))
    return 1 - ss_res / (ss_tot + K.epsilon())


# -------------------------------
# 3. Helper Functions
# -------------------------------
def load_station_data(station_folder):
    """
    Loads 'all_years_data.csv' for one station, sorts by Date.
    Returns the DataFrame or empty if not found.
    """
    csv_path = os.path.join(BASE_DIR, station_folder, "all_years_data.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Station: {station_folder}")
        return pd.DataFrame()  # empty

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


def split_by_percentages(
    df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO
):
    """Split a DataFrame chronologically into train/val/test portions."""
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


def feature_engineering(df):
    """
    Adds 'day', 'month' features, drops columns not needed (Date, etc.),
    and removes NaNs.
    """
    if df.empty:
        return df

    # Extract day & month
    if "Date" in df.columns:
        df["day"] = df["Date"].dt.day
        df["month"] = df["Date"].dt.month
        # Cyclical month features for seasonality
        df["month_sin"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)

    # Handle erroneous extremely low Tmin values
    if "Tmin (°C)" in df.columns:
        df.loc[df["Tmin (°C)"] < -10, "Tmin (°C)"] = np.nan

    # Log-transform rainfall to stabilize variance
    if "Rainfall (mm)" in df.columns:
        df = df[df["Rainfall (mm)"].notna()]
        df["Rainfall (mm)"] = np.log1p(df["Rainfall (mm)"])

    # Drop columns we don't want in the model
    drop_cols = [
        "Date",
        "Latitude",
        "Longitude",
        "Station",
        "month",
        "RH (%)",
        "Wind Speed (m/s)",
    ]
    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def create_sequences(df, window_size, horizon, target_col):
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
    then return just the target column portion in the original scale.
    """
    n_samples, horizon = y_scaled.shape
    n_features = len(df_columns)
    target_idx = df_columns.index(target_col)

    dummy = np.zeros((n_samples * horizon, n_features))
    # Place the scaled predictions in the target column
    for i in range(horizon):
        dummy[i * n_samples : (i + 1) * n_samples, target_idx] = y_scaled[:, i]

    dummy_inv = scaler.inverse_transform(dummy)
    target_inv = dummy_inv[:, target_idx]
    # Reshape to (n_samples, horizon)
    target_inv = target_inv.reshape(horizon, n_samples).T
    return target_inv


def compute_metrics(y_true, y_pred):
    """Return MAE, MSE, RMSE, R² average and per-horizon array."""
    mse = mean_squared_error(y_true, y_pred, multioutput="uniform_average")
    mae = mean_absolute_error(y_true, y_pred, multioutput="uniform_average")
    rmse = math.sqrt(mse)
    r2_avg = r2_score(y_true, y_pred, multioutput="uniform_average")
    r2_each = r2_score(y_true, y_pred, multioutput="raw_values")
    return mae, mse, rmse, r2_avg, r2_each


def plot_time_series_predictions(y_true, y_pred, horizon, station_folder):
    """
    Plots actual vs predicted Rainfall for each day in the forecast horizon (day+1, day+2, day+3),
    as a time series (blue line for actual vs. red line for predicted).
    """
    station_plot_dir = os.path.join(PLOTS_DIR, station_folder)
    os.makedirs(station_plot_dir, exist_ok=True)
    save_path = os.path.join(station_plot_dir, "test_predictions.png")

    plt.figure(figsize=(10, 8))
    for day_idx in range(horizon):
        ax = plt.subplot(horizon, 1, day_idx + 1)
        ax.plot(y_true[:, day_idx], label="Actual", color="blue")
        ax.plot(y_pred[:, day_idx], label="Predicted", color="red")
        ax.set_title(f"Day +{day_idx+1} Rainfall Forecast")
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_scatter_day(y_true, y_pred, day_idx, station_folder):
    """
    Creates a scatter plot: x=actual, y=predicted, for a specific day in the forecast horizon.
    Saves as scatter_day{day_idx+1}.png in the station's folder.
    """
    station_plot_dir = os.path.join(PLOTS_DIR, station_folder)
    os.makedirs(station_plot_dir, exist_ok=True)
    save_path = os.path.join(station_plot_dir, f"scatter_day{day_idx+1}.png")

    plt.figure(figsize=(6, 6))
    plt.scatter(
        y_true[:, day_idx],
        y_pred[:, day_idx],
        alpha=0.6,
        color="tab:blue",
        label="Predictions",
    )
    plt.title(f"Scatter Plot: Day +{day_idx+1} Rainfall")
    plt.xlabel("Actual Rainfall")
    plt.ylabel("Predicted Rainfall")

    # Add y=x reference line
    min_val = min(y_true[:, day_idx].min(), y_pred[:, day_idx].min())
    max_val = max(y_true[:, day_idx].max(), y_pred[:, day_idx].max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linestyle="--",
        label="Ideal",
    )

    # Compute metrics for this forecast day
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


def plot_training_history(history, save_path):
    """
    Plots training & validation loss over epochs, saving the plot to save_path.
    If R² is in the history, that is plotted as well.
    """
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


# -------------------------------
# 4. Per-Farm Training Function
# -------------------------------
def train_for_station(station_folder):
    """Train and evaluate a rainfall model for a single farm.

    Returns a dictionary with the final training, validation, and test metrics
    for that farm.
    """
    df_station = load_station_data(station_folder)
    if df_station.empty:
        print(f"No data for station {station_folder}. Skipping.")
        return None

    df_train, df_val, df_test = split_by_percentages(df_station)
    print(
        f"{station_folder} data sizes -> Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}"
    )

    df_train = feature_engineering(df_train)
    df_val = feature_engineering(df_val)
    df_test = feature_engineering(df_test)
    print(
        f"{station_folder} feature-engineered shapes -> Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}"
    )

    if df_train.empty:
        print(f"No training data for station {station_folder}. Skipping.")
        return None

    scaler = StandardScaler()
    scaler.fit(df_train.values)

    scaler_path = os.path.join(MODELS_DIR, f"{station_folder}_scaler_Rain.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    df_train_scaled = pd.DataFrame(
        scaler.transform(df_train.values), columns=df_train.columns
    )
    df_val_scaled = pd.DataFrame(
        scaler.transform(df_val.values), columns=df_val.columns
    )

    X_train, y_train = create_sequences(
        df_train_scaled,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        target_col=TARGET_COL,
    )

    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    X_val, y_val = create_sequences(
        df_val_scaled,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        target_col=TARGET_COL,
    )
    print(f"{station_folder} sequences -> Train: {len(X_train)}, Val: {len(X_val)}")

    num_features = df_train.shape[1]
    model = Sequential()
    model.add(Input(shape=(WINDOW_SIZE, num_features)))
    model.add(LSTM(64, activation="tanh"))
    model.add(Dense(HORIZON))
    model.compile(
        loss="mse",
        optimizer=Adam(learning_rate=0.001),
        metrics=["mae", r2_keras],
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    model_path = os.path.join(MODELS_DIR, f"{station_folder}_model_rain_lstm.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    station_plot_dir = os.path.join(PLOTS_DIR, station_folder)
    os.makedirs(station_plot_dir, exist_ok=True)

    history_csv_path = os.path.join(station_plot_dir, "training_history_values.csv")
    pd.DataFrame(history.history).to_csv(history_csv_path, index=False)

    plot_history_path = os.path.join(station_plot_dir, "training_history.png")
    plot_training_history(history, plot_history_path)

    # --- Metrics on training set ---
    y_train_pred_scaled = model.predict(X_train)
    df_cols = list(df_train.columns)

    y_train_inv = inverse_transform_predictions(y_train, scaler, df_cols, TARGET_COL)
    y_train_pred_inv = inverse_transform_predictions(
        y_train_pred_scaled, scaler, df_cols, TARGET_COL
    )
    y_train_inv = np.expm1(y_train_inv)
    y_train_pred_inv = np.expm1(y_train_pred_inv)

    train_mae, train_mse, train_rmse, train_r2_avg, train_r2_each = compute_metrics(
        y_train_inv, y_train_pred_inv
    )

    # --------- Calculate validation metrics over whole val set ----------
    y_val_pred_scaled = model.predict(X_val)
    df_cols = list(df_val.columns)

    y_val_inv = inverse_transform_predictions(y_val, scaler, df_cols, TARGET_COL)
    y_val_pred_inv = inverse_transform_predictions(
        y_val_pred_scaled, scaler, df_cols, TARGET_COL
    )
    y_val_inv = np.expm1(y_val_inv)
    y_val_pred_inv = np.expm1(y_val_pred_inv)

    val_mae, val_mse, val_rmse, val_r2_avg, val_r2_each = compute_metrics(
        y_val_inv, y_val_pred_inv
    )

    metrics_summary = {
        "farm": station_folder,
        "train_loss": history.history.get("loss", [None])[-1],
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_r2_avg": train_r2_avg,
        "train_r2_day1": train_r2_each[0],
        "train_r2_day2": train_r2_each[1],
        "train_r2_day3": train_r2_each[2],
        # Validation metrics
        "val_loss": val_mse,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_r2_avg": val_r2_avg,
        "val_r2_day1": val_r2_each[0],
        "val_r2_day2": val_r2_each[1],
        "val_r2_day3": val_r2_each[2],
        # Placeholders for test metrics (filled below if available)
        "test_mae": None,
        "test_mse": None,
        "test_rmse": None,
        "test_r2_avg": None,
        "test_r2_day1": None,
        "test_r2_day2": None,
        "test_r2_day3": None,
    }

    if not df_test.empty:
        df_test_scaled = pd.DataFrame(
            scaler.transform(df_test.values), columns=df_test.columns
        )
        X_test, y_test = create_sequences(
            df_test_scaled,
            window_size=WINDOW_SIZE,
            horizon=HORIZON,
            target_col=TARGET_COL,
        )
        print(f"{station_folder} sequences -> Test: {len(X_test)}")
        if len(X_test) > 0:
            y_test_pred_scaled = model.predict(X_test)

            df_cols = list(df_test.columns)

            y_test_inv = inverse_transform_predictions(
                y_test, scaler, df_cols, TARGET_COL
            )
            y_pred_inv = inverse_transform_predictions(
                y_test_pred_scaled, scaler, df_cols, TARGET_COL
            )
            y_test_inv = np.expm1(y_test_inv)
            y_pred_inv = np.expm1(y_pred_inv)

            mae, mse, rmse, r2_avg, r2_each = compute_metrics(y_test_inv, y_pred_inv)

            print(f"\n--- Test Metrics for station: {station_folder} ---")
            print(f"MAE:  {mae:.4f}")
            print(f"MSE:  {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R²:   {r2_avg:.4f}")

            plot_time_series_predictions(
                y_test_inv, y_pred_inv, HORIZON, station_folder
            )
            for day_idx in range(HORIZON):
                plot_scatter_day(y_test_inv, y_pred_inv, day_idx, station_folder)

            metrics_summary["test_mae"] = mae
            metrics_summary["test_mse"] = mse
            metrics_summary["test_rmse"] = rmse
            metrics_summary["test_r2_avg"] = r2_avg
            metrics_summary["test_r2_day1"] = r2_each[0]
            metrics_summary["test_r2_day2"] = r2_each[1]
            metrics_summary["test_r2_day3"] = r2_each[2]

        else:
            print(
                f"Not enough test data to form sequences for station: {station_folder}. Skipping evaluation."
            )

    return metrics_summary


# -------------------------------
# 4. Main Script
# -------------------------------
def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    if TRAIN_PER_FARM:
        metrics_list = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(train_for_station, station): station
                for station in STATION_FOLDERS
            }
            for future in as_completed(futures):
                station = futures[future]
                try:
                    result = future.result()
                    if result:
                        metrics_list.append(result)
                except Exception as exc:
                    print(f"Training failed for {station}: {exc}")

        summary_df = pd.DataFrame(metrics_list).round(4)
        train_val_cols = [
            "farm",
            "train_loss",
            "train_mae",
            "train_rmse",
            "train_r2_avg",
            "train_r2_day1",
            "train_r2_day2",
            "train_r2_day3",
            "val_loss",
            "val_mae",
            "val_rmse",
            "val_r2_avg",
            "val_r2_day1",
            "val_r2_day2",
            "val_r2_day3",
        ]
        test_cols = [
            "farm",
            "test_mae",
            "test_mse",
            "test_rmse",
            "test_r2_avg",
            "test_r2_day1",
            "test_r2_day2",
            "test_r2_day3",
        ]

        train_val_df = summary_df[train_val_cols]
        test_df = summary_df[test_cols]

        train_val_path = os.path.join(PLOTS_DIR, "training_validation_metrics.csv")
        test_path = os.path.join(PLOTS_DIR, "test_metrics.csv")
        train_val_df.to_csv(train_val_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"\nTraining/validation metrics saved to {train_val_path}")
        print(f"Test metrics saved to {test_path}")

        print("\nAll done. End of script.")
        return

    train_list = []
    val_list = []
    test_data_by_station = {}

    # A) For each station, load data and split into train/test by date range
    for station in STATION_FOLDERS:
        df_station = load_station_data(station)
        if df_station.empty:
            continue

        df_train_stn, df_val_stn, df_test_stn = split_by_percentages(df_station)
        print(
            f"{station} data sizes -> Train: {len(df_train_stn)}, Val: {len(df_val_stn)}, Test: {len(df_test_stn)}"
        )

        df_train_stn = feature_engineering(df_train_stn)
        df_val_stn = feature_engineering(df_val_stn)
        df_test_stn = feature_engineering(df_test_stn)
        print(
            f"{station} feature-engineered shapes -> Train: {df_train_stn.shape}, Val: {df_val_stn.shape}, Test: {df_test_stn.shape}"
        )

        if not df_train_stn.empty:
            train_list.append(df_train_stn)
        if not df_val_stn.empty:
            val_list.append(df_val_stn)
        test_data_by_station[station] = df_test_stn

    # B) Combine all training data
    if not train_list:
        print("No training data found.")
        return
    df_train_all = pd.concat(train_list, ignore_index=True)
    print("Combined training shape:", df_train_all.shape)

    # C) Scale the training data (using StandardScaler here, can be changed to MinMaxScaler if desired)
    scaler = StandardScaler()
    scaler.fit(df_train_all.values)

    with open("scaler_model_Rain.pkl", "wb") as f:
        pickle.dump(scaler, f)
        # return
    df_train_scaled = pd.DataFrame(
        scaler.transform(df_train_all.values), columns=df_train_all.columns
    )
    df_val_all = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
    if not df_val_all.empty:
        df_val_scaled = pd.DataFrame(
            scaler.transform(df_val_all.values), columns=df_val_all.columns
        )
    else:
        df_val_scaled = pd.DataFrame(columns=df_train_all.columns)
    print(
        f"Combined feature-engineered shapes -> Train: {df_train_all.shape}, Val: {df_val_all.shape}"
    )

    # D) Create sequences for training and validation sets
    X_train, y_train = create_sequences(
        df_train_scaled, window_size=WINDOW_SIZE, horizon=HORIZON, target_col=TARGET_COL
    )
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    X_val, y_val = create_sequences(
        df_val_scaled, window_size=WINDOW_SIZE, horizon=HORIZON, target_col=TARGET_COL
    )

    print(f"Combined sequences -> Train: {len(X_train)}, Val: {len(X_val)}")

    # F) Build & train LSTM model with custom R² metric
    num_features = df_train_all.shape[1]
    model = Sequential()
    model.add(LSTM(64, activation="tanh", input_shape=(WINDOW_SIZE, num_features)))
    model.add(Dense(HORIZON))
    model.compile(
        loss="mse",
        optimizer=Adam(learning_rate=0.001),
        metrics=["mae", r2_keras],  # add custom R² metric
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    # Save the trained model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Save training history as CSV
    history_df = pd.DataFrame(history.history).round(4)
    history_csv_path = os.path.join(PLOTS_DIR, "training_history_values.csv")
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history values saved to {history_csv_path}")

    # Plot training history (loss and R²)
    plot_history_path = os.path.join(PLOTS_DIR, "training_history.png")
    plot_training_history(history, plot_history_path)
    print(f"Training history plot saved to {plot_history_path}")

    # ----- Training and validation metrics -----
    y_train_pred_scaled = model.predict(X_train)
    df_cols = list(df_train_all.columns)

    y_train_inv = inverse_transform_predictions(y_train, scaler, df_cols, TARGET_COL)
    y_train_pred_inv = inverse_transform_predictions(
        y_train_pred_scaled, scaler, df_cols, TARGET_COL
    )
    y_train_inv = np.expm1(y_train_inv)
    y_train_pred_inv = np.expm1(y_train_pred_inv)
    train_mae, train_mse, train_rmse, train_r2_avg, train_r2_each = compute_metrics(
        y_train_inv, y_train_pred_inv
    )

    y_val_pred_scaled = model.predict(X_val)
    y_val_inv = inverse_transform_predictions(y_val, scaler, df_cols, TARGET_COL)
    y_val_pred_inv = inverse_transform_predictions(
        y_val_pred_scaled, scaler, df_cols, TARGET_COL
    )
    y_val_inv = np.expm1(y_val_inv)
    y_val_pred_inv = np.expm1(y_val_pred_inv)
    val_mae, val_mse, val_rmse, val_r2_avg, val_r2_each = compute_metrics(
        y_val_inv, y_val_pred_inv
    )

    train_val_metrics = {
        "farm": "combined",
        "train_loss": history.history.get("loss", [None])[-1],
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_r2_avg": train_r2_avg,
        "train_r2_day1": train_r2_each[0],
        "train_r2_day2": train_r2_each[1],
        "train_r2_day3": train_r2_each[2],
        "val_loss": val_mse,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_r2_avg": val_r2_avg,
        "val_r2_day1": val_r2_each[0],
        "val_r2_day2": val_r2_each[1],
        "val_r2_day3": val_r2_each[2],
    }

    test_metrics_list = []

    # ----- Evaluate on each station's test set -----
    for station in STATION_FOLDERS:
        df_test_stn = test_data_by_station.get(station, pd.DataFrame())
        if df_test_stn.empty:
            print(f"No test data for station: {station}. Skipping.")
            continue

        df_test_scaled = pd.DataFrame(
            scaler.transform(df_test_stn.values), columns=df_test_stn.columns
        )

        X_test, y_test = create_sequences(
            df_test_scaled,
            window_size=WINDOW_SIZE,
            horizon=HORIZON,
            target_col=TARGET_COL,
        )
        print(f"{station} sequences -> Test: {len(X_test)}")
        if len(X_test) == 0:
            print(
                f"Not enough test data to form sequences for station: {station}. Skipping."
            )
            continue

        y_test_pred_scaled = model.predict(X_test)

        df_cols = list(df_test_stn.columns)

        y_test_inv = inverse_transform_predictions(y_test, scaler, df_cols, TARGET_COL)
        y_pred_inv = inverse_transform_predictions(
            y_test_pred_scaled, scaler, df_cols, TARGET_COL
        )
        y_test_inv = np.expm1(y_test_inv)
        y_pred_inv = np.expm1(y_pred_inv)

        mae, mse, rmse, r2_avg, r2_each = compute_metrics(y_test_inv, y_pred_inv)

        print(f"\n--- Test Metrics for station: {station} ---")
        print(f"MAE:  {mae:.4f}")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2_avg:.4f}")

        plot_time_series_predictions(y_test_inv, y_pred_inv, HORIZON, station)

        for day_idx in range(HORIZON):
            plot_scatter_day(y_test_inv, y_pred_inv, day_idx, station)

        test_metrics_list.append(
            {
                "farm": station,
                "test_mae": mae,
                "test_mse": mse,
                "test_rmse": rmse,
                "test_r2_avg": r2_avg,
                "test_r2_day1": r2_each[0],
                "test_r2_day2": r2_each[1],
                "test_r2_day3": r2_each[2],
            }
        )

    train_val_df = pd.DataFrame([train_val_metrics]).round(4)
    test_df = pd.DataFrame(test_metrics_list).round(4)

    train_val_path = os.path.join(PLOTS_DIR, "training_validation_metrics.csv")
    test_path = os.path.join(PLOTS_DIR, "test_metrics.csv")
    train_val_df.to_csv(train_val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\nTraining/validation metrics saved to {train_val_path}")
    print(f"Test metrics saved to {test_path}")

    print("\nAll done. End of script.")


if __name__ == "__main__":
    main()
