"""
Multi-Station, Multi-Step LSTM Forecast for ET (mm/day)
Test Data = 2 Specific Years for One Station

In this example:
- We pick one station (TEST_STATION) and a date range (TEST_START, TEST_END)
  as the test set.
- All data from other stations and all other dates of the test station go to training.
- We scale the training data, create sequences (24-day input -> next 3 days).
- We train LSTM (80%/20% train/val) and compute a custom R² metric (r2_keras).
- We evaluate on the 2-year test range of the chosen station, 
  plotting actual vs. predicted time-series and scatter plots.

Adjust station name, date range, or scaling method as needed.
"""
import pickle
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler , StandardScaler

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# Matplotlib
import matplotlib.pyplot as plt

# -------------------------------
# 1. Configuration
# -------------------------------

# # All station folders
# STATION_FOLDERS = [
#     "Kahuku_Farm",
#     "Nozawa_Farms",
#     "Kuilima_Farms",
#     "Cabaero_Farms",
#     "Kupaa_Farms",
#     # Add more if needed...
# ]

STATION_FOLDERS = [
    "Kahuku_Farm",
    "Nozawa_Farms",
    "Kuilima_Farms",
    "Cabaero_Farms",                 # Corresponds to Cabaero Farms (lat: 20.8425, lng: -156.3471)
    "Kupaa_Farms",                   # Corresponds to Kupa'a Farms (lat: 20.7658, lng: -156.3513)
    "MAO_Organic_Farms_Original",    # From MA'O Organic Farms (original site)
    "MAO_Organic_Farms_New",         # From MA'O Organic Farms (new site)
    "2K_Farm_LLC",                   # From 2K Farm LLC
    "Wong_Hon_Hin_Inc",              # From Wong Hon Hin Inc
    "Hawaii_Taro_Farm_LLC",          # From Hawaii Taro Farm, LLC
    "Hawaii_Seed_Pro_LLC_Farm",      # From Hawaii Seed Pro LLC Farm
    "Cabaero_Farm",                  # Corresponds to Cabaero Farm (lat: 20.791703, lng: -156.358194)
    "Kupaa_Farms2",                 # Second instance of Kupaa Farms (lat: 20.765515, lng: -156.35185)
    "Hirako_Farm",                   # First instance of Hirako Farm
    "Hirako_Farm1",                 # Second instance of Hirako Farm
    "Anoano_Farms"                   # From Anoano Farms
]

# Base directory for station folders
BASE_DIR = "farm_data"

# We designate one station + date range as test data
TEST_STATION = "Kupaa_Farms"
TEST_START   = "2021-01-01"
TEST_END     = "2022-12-31"

# Input sequence length (days to look back)
WINDOW_SIZE = 24

# Forecast horizon (days to predict)
HORIZON = 3

# Target column name
TARGET_COL = "ET (mm/day)"

# Train/Validation ratio for the training set
TRAIN_VAL_RATIO = 0.8  # 80% train, 20% validation

# Random seed for reproducibility
RANDOM_SEED = 42

# Where to save the trained model
MODEL_PATH = "model_lstm.h5"

# Where to save plots and results
PLOTS_DIR = "plots_test_2years"
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
    return df

def split_test_by_date(df, station_name, test_station, start_date, end_date):
    """
    If station_name != test_station:
      -> All data is training.
    Else (station_name == test_station):
      -> Rows in [start_date, end_date] = test data, rest = train data.
    Returns (df_train, df_test).
    """
    if df.empty or "Date" not in df.columns:
        return df, pd.DataFrame()

    if station_name != test_station:
        # Entire DataFrame is training data
        return df, pd.DataFrame()
    else:
        # For the test station, split by date range
        mask_test = (df["Date"] >= start_date) & (df["Date"] <= end_date)
        df_test = df[mask_test].copy()
        df_train = df[~mask_test].copy()
        return df_train, df_test

def feature_engineering(df):
    """
    Adds 'day', 'month' features, drops columns not needed (Date, etc.), 
    removes NaNs.
    """
    if df.empty:
        return df

    # Extract day & month
    if "Date" in df.columns:
        df["day"] = df["Date"].dt.day
        df["month"] = df["Date"].dt.month

    # Drop columns we don't want in the model
    drop_cols = ["Date", "Latitude", "Longitude", "Station"]
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
        y_i = data[i + window_size : i + window_size + horizon, df.columns.get_loc(target_col)]
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
    # place the scaled predictions in the target column
    for i in range(horizon):
        dummy[i*n_samples:(i+1)*n_samples, target_idx] = y_scaled[:, i]

    dummy_inv = scaler.inverse_transform(dummy)
    target_inv = dummy_inv[:, target_idx]
    # reshape to (n_samples, horizon)
    target_inv = target_inv.reshape(horizon, n_samples).T
    return target_inv

def plot_time_series_predictions(y_true, y_pred, horizon, station_folder):
    """
    Plots actual vs predicted ET for each day in horizon (day+1, day+2, day+3),
    as a time series (blue line vs. red line).
    """
    station_plot_dir = os.path.join(PLOTS_DIR, station_folder)
    os.makedirs(station_plot_dir, exist_ok=True)
    save_path = os.path.join(station_plot_dir, "test_predictions.png")

    plt.figure(figsize=(10, 8))
    for day_idx in range(horizon):
        ax = plt.subplot(horizon, 1, day_idx+1)
        ax.plot(y_true[:, day_idx], label="Actual", color='blue')
        ax.plot(y_pred[:, day_idx], label="Predicted", color='red')
        ax.set_title(f"Day +{day_idx+1} Forecast")
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_scatter_day(y_true, y_pred, day_idx, station_folder):
    """
    Creates a scatter plot: x=actual, y=predicted, for a specific day_idx in horizon.
    Saves as scatter_day{day_idx+1}.png in station's folder.
    """
    station_plot_dir = os.path.join(PLOTS_DIR, station_folder)
    os.makedirs(station_plot_dir, exist_ok=True)
    save_path = os.path.join(station_plot_dir, f"scatter_day{day_idx+1}.png")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true[:, day_idx], y_pred[:, day_idx], alpha=0.5, color='green')
    plt.title(f"Scatter Plot: Day +{day_idx+1}")
    plt.xlabel("Actual ET")
    plt.ylabel("Predicted ET")

    # Optionally, add a y=x line for reference
    min_val = min(y_true[:, day_idx].min(), y_pred[:, day_idx].min())
    max_val = max(y_true[:, day_idx].max(), y_pred[:, day_idx].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_history(history, save_path):
    """
    Plots training & validation loss over epochs, saving to save_path.
    If R² is in the history, we plot that too.
    """
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

# -------------------------------
# 4. Main Script
# -------------------------------
def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    train_list = []
    test_data_by_station = {}

    # A) For each station, load data and split into train/test by date range
    for station in STATION_FOLDERS:
        df_station = load_station_data(station)
        if df_station.empty:
            continue
        
        # If station != TEST_STATION, all data goes to training
        # If station == TEST_STATION, rows in [TEST_START, TEST_END] => test
        df_train_stn, df_test_stn = split_test_by_date(
            df_station, 
            station, 
            TEST_STATION, 
            TEST_START, 
            TEST_END
        )

        # Feature engineering
        df_train_stn = feature_engineering(df_train_stn)
        df_test_stn  = feature_engineering(df_test_stn)

        if not df_train_stn.empty:
            train_list.append(df_train_stn)
        test_data_by_station[station] = df_test_stn

    # B) Combine all training data
    if not train_list:
        print("No training data found.")
        return
    df_train_all = pd.concat(train_list, ignore_index=True)
    print("Combined training shape:", df_train_all.shape)

    # C) Scale the training data (MinMaxScaler or StandardScaler)
    scaler = StandardScaler()  # or StandardScaler()
    scaler.fit(df_train_all.values)


    # ذخیره اسکیلر
    with open('scaler_model_ET.pkl', 'wb') as f:
        pickle.dump(scaler, f)    
        # return
    df_train_scaled = pd.DataFrame(scaler.transform(df_train_all.values), columns=df_train_all.columns)

    # D) Create sequences for training
    X_train_full, y_train_full = create_sequences(
        df_train_scaled,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        target_col=TARGET_COL
    )
    print("Full training sequences:", len(X_train_full))

    # E) Shuffle & split into train/validation
    indices = np.arange(len(X_train_full))
    np.random.shuffle(indices)
    X_train_full = X_train_full[indices]
    y_train_full = y_train_full[indices]

    train_size = int(TRAIN_VAL_RATIO * len(X_train_full))
    X_val = X_train_full[train_size:]
    y_val = y_train_full[train_size:]
    X_train = X_train_full[:train_size]
    y_train = y_train_full[:train_size]

    print("Training sequences:", len(X_train))
    print("Validation sequences:", len(X_val))

    # F) Build & train LSTM model with custom R² metric
    num_features = df_train_all.shape[1]
    model = Sequential()
    model.add(LSTM(64, activation='tanh', input_shape=(WINDOW_SIZE, num_features)))
    model.add(Dense(HORIZON))
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=0.001),
        metrics=['mae', r2_keras]  # add custom R²
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Save training history as CSV
    history_df = pd.DataFrame(history.history)
    history_csv_path = os.path.join(PLOTS_DIR, "training_history_values.csv")
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history values saved to {history_csv_path}")

    # Plot training history (including R²)
    plot_history_path = os.path.join(PLOTS_DIR, "training_history.png")
    plot_training_history(history, plot_history_path)
    print(f"Training history plot saved to {plot_history_path}")

    # G) Evaluate on each station's test set separately
    for station in STATION_FOLDERS:
        df_test_stn = test_data_by_station.get(station, pd.DataFrame())
        if df_test_stn.empty:
            print(f"No test data for station: {station}. Skipping.")
            continue

        # Scale test data
        df_test_scaled = pd.DataFrame(
            scaler.transform(df_test_stn.values),
            columns=df_test_stn.columns
        )

        # Create sequences
        X_test, y_test = create_sequences(
            df_test_scaled,
            window_size=WINDOW_SIZE,
            horizon=HORIZON,
            target_col=TARGET_COL
        )
        if len(X_test) == 0:
            print(f"Not enough test data to form sequences for station: {station}. Skipping.")
            continue

        # Predict
        y_test_pred_scaled = model.predict(X_test)

        # Invert scaling
        df_cols = list(df_test_stn.columns)  # same order
        y_test_inv  = inverse_transform_predictions(y_test, scaler, df_cols, TARGET_COL)
        y_pred_inv = inverse_transform_predictions(y_test_pred_scaled, scaler, df_cols, TARGET_COL)

        # Metrics (averaged across horizon=3)
        mse = mean_squared_error(y_test_inv, y_pred_inv, multioutput='uniform_average')
        mae = mean_absolute_error(y_test_inv, y_pred_inv, multioutput='uniform_average')
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test_inv, y_pred_inv, multioutput='uniform_average')

        print(f"\n--- Test Metrics for station: {station} (range: {TEST_START} to {TEST_END}) ---")
        print(f"MAE:  {mae:.4f}")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")

        # Plot time-series predictions vs. actual
        plot_time_series_predictions(y_test_inv, y_pred_inv, HORIZON, station)

        # Plot scatter for each day in horizon
        for day_idx in range(HORIZON):
            plot_scatter_day(y_test_inv, y_pred_inv, day_idx, station)

    print("\nAll done. End of script.")

if __name__ == "__main__":
    main()
