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
from generate_scatter_report import generate_scatter_report
from config import (
    STATION_FOLDERS,
    TRAIN_PER_FARM,
    MODELS_DIR,
    TRAINING_ANALYSIS_DIR,
    BASE_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    START_YEAR,
    END_YEAR,
    WINDOW_SIZE,
    HORIZON,
    MAX_WORKERS,
    RANDOM_SEED,
)
from training_utils import (
    r2_keras,
    load_station_data,
    split_by_percentages,
    create_sequences,
    inverse_transform_predictions,
    compute_metrics,
    plot_time_series_predictions,
    plot_scatter_day,
    plot_training_history,
)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRAINING_ANALYSIS_DIR, exist_ok=True)

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
# Configuration constants are imported from `config`.

# Target column name for rainfall forecasting
TARGET_COL = "Rainfall (mm)"
MODEL_PATH = "model_rain_lstm.h5"
PLOTS_DIR = (
    "plots_test_rainfall" if START_YEAR is None else f"plots_test_rainfall_{START_YEAR}-{END_YEAR}"
)
PLOTS_DIR = os.path.join("plots", PLOTS_DIR)
os.makedirs(PLOTS_DIR, exist_ok=True)
# -------------------------------
# 2. Custom R² Metric
# -------------------------------


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
        df.loc[:, "Rainfall (mm)"] = np.log1p(df["Rainfall (mm)"])

    # --- Add the new feature column: relative_humidity ---
    if "relative_humidity" in df.columns:
        df["relative_humidity"] = df["relative_humidity"]

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
            df = df.drop(col, axis=1)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


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
        metrics=["mae"],
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
        verbose=0,
    )

    model_path = os.path.join(MODELS_DIR, f"{station_folder}_model_rain_lstm.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    station_plot_dir = os.path.join(PLOTS_DIR, station_folder)
    os.makedirs(station_plot_dir, exist_ok=True)

    history_csv_path = os.path.join(station_plot_dir, "training_history_values.csv")
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_csv_path, index=False)

    analysis_csv_path = os.path.join(TRAINING_ANALYSIS_DIR, f"{station_folder}_history.csv")
    history_df.to_csv(analysis_csv_path, index=False)

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
                y_test_inv,
                y_pred_inv,
                HORIZON,
                station_folder,
                PLOTS_DIR,
                "Rainfall",
            )
            for day_idx in range(HORIZON):
                plot_scatter_day(
                    y_test_inv,
                    y_pred_inv,
                    day_idx,
                    station_folder,
                    PLOTS_DIR,
                    "Rainfall",
                )

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
        metrics=["mae"],  # add custom R² metric
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
        verbose=0,
    )

    # Save the trained model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Save training history as CSV
    history_df = pd.DataFrame(history.history).round(4)
    history_csv_path = os.path.join(PLOTS_DIR, "training_history_values.csv")
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history values saved to {history_csv_path}")

    analysis_csv_path = os.path.join(TRAINING_ANALYSIS_DIR, "combined_history.csv")
    history_df.to_csv(analysis_csv_path, index=False)

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

        plot_time_series_predictions(
            y_test_inv,
            y_pred_inv,
            HORIZON,
            station,
            PLOTS_DIR,
            "Rainfall",
        )

        for day_idx in range(HORIZON):
            plot_scatter_day(
                y_test_inv,
                y_pred_inv,
                day_idx,
                station,
                PLOTS_DIR,
                "Rainfall",
            )

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

    summary_path = os.path.join(TRAINING_ANALYSIS_DIR, "summary.csv")
    summary_df = train_val_df.merge(test_df, on="farm", how="left")
    summary_df.to_csv(summary_path, index=False)

    print("\nAll done. End of script.")


if __name__ == "__main__":
    main()
