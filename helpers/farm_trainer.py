"""
Class for per-farm ET LSTM training, evaluation, and artifacts.
"""

import os
import pickle
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from helpers.feature_engineering import feature_engineering
from helpers.sequencing import create_sequences, inverse_transform_predictions
from helpers.metrics_and_plots import (
    compute_metrics,
    plot_training_history,
    plot_time_series_predictions,
    plot_scatter_day,
)
from helpers.lstm_model import build_lstm_model


class FarmModelTrainer:
    def __init__(self, station_folder: str, config: dict):
        self.station_folder = station_folder
        self.cfg = config
        self.metrics_summary = None
        self.scaler = None
        self.model = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(station_folder)
        self._prepare_dirs()

    def _prepare_dirs(self):
        os.makedirs(self.cfg["MODELS_DIR"], exist_ok=True)
        self.plot_dir = os.path.join(self.cfg["PLOTS_DIR"], self.station_folder)
        os.makedirs(self.plot_dir, exist_ok=True)

    def load_data(self):
        csv_path = os.path.join(
            self.cfg["BASE_DIR"], self.station_folder, "all_years_data.csv"
        )
        if not os.path.exists(csv_path):
            self.logger.warning(f"{csv_path} not found.")
            return pd.DataFrame()
        df = pd.read_csv(csv_path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.sort_values("Date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            if self.cfg["START_YEAR"] is not None:
                df = df[df["Date"].dt.year >= self.cfg["START_YEAR"]]
            if self.cfg["END_YEAR"] is not None:
                df = df[df["Date"].dt.year <= self.cfg["END_YEAR"]]
            df.reset_index(drop=True, inplace=True)
        return df

    def split_by_percentages(self, df):
        if df.empty or "Date" not in df.columns:
            return df, pd.DataFrame(), pd.DataFrame()
        df_sorted = df.sort_values("Date").reset_index(drop=True)
        n = len(df_sorted)
        train_end = int(n * self.cfg["TRAIN_RATIO"])
        val_end = train_end + int(n * self.cfg["VAL_RATIO"])
        df_train = df_sorted.iloc[:train_end].copy()
        df_val = df_sorted.iloc[train_end:val_end].copy()
        df_test = df_sorted.iloc[val_end:].copy()
        return df_train, df_val, df_test

    def train(self):
        df_station = self.load_data()
        if df_station.empty:
            self.logger.warning("No data. Skipping.")
            return None
        df_train, df_val, df_test = self.split_by_percentages(df_station)
        self.logger.info(
            f"Split sizes: Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}"
        )

        df_train = feature_engineering(df_train)
        df_val = feature_engineering(df_val)
        df_test = feature_engineering(df_test)
        self.logger.info(
            f"Feature shapes: Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}"
        )

        if df_train.empty:
            self.logger.warning("No training data. Skipping.")
            return None
        self.scaler = StandardScaler().fit(df_train.values)
        scaler_path = os.path.join(
            self.cfg["MODELS_DIR"], f"{self.station_folder}_scaler_ET.pkl"
        )
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        df_train_scaled = pd.DataFrame(
            self.scaler.transform(df_train.values), columns=df_train.columns
        )
        df_val_scaled = pd.DataFrame(
            self.scaler.transform(df_val.values), columns=df_val.columns
        )

        X_train, y_train = create_sequences(
            df_train_scaled,
            self.cfg["WINDOW_SIZE"],
            self.cfg["HORIZON"],
            self.cfg["TARGET_COL"],
        )
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        X_val, y_val = create_sequences(
            df_val_scaled,
            self.cfg["WINDOW_SIZE"],
            self.cfg["HORIZON"],
            self.cfg["TARGET_COL"],
        )
        self.logger.info(f"Sequences: Train: {len(X_train)}, Val: {len(X_val)}")

        input_shape = (self.cfg["WINDOW_SIZE"], df_train.shape[1])
        self.model = build_lstm_model(input_shape, self.cfg["HORIZON"])
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0,
        )
        model_path = os.path.join(
            self.cfg["MODELS_DIR"], f"{self.station_folder}_model_lstm.keras"
        )
        self.model.save(model_path)
        self.logger.info(f"Model saved to {model_path}")
        history_csv = os.path.join(self.plot_dir, "training_history_values.csv")
        pd.DataFrame(history.history).to_csv(history_csv, index=False)
        plot_hist_path = os.path.join(self.plot_dir, "training_history.png")
        plot_training_history(history, plot_hist_path)
        self.logger.info("Artifacts and plots saved.")
        self.df_train, self.df_val, self.df_test = df_train, df_val, df_test
        self.X_train, self.y_train, self.X_val, self.y_val = (
            X_train,
            y_train,
            X_val,
            y_val,
        )
        return True

    def evaluate(self):
        # --- Metrics on training set ---
        results = {"farm": self.station_folder}
        df_cols = list(self.df_train.columns)
        y_train_pred_scaled = self.model.predict(self.X_train)
        y_train_inv = inverse_transform_predictions(
            self.y_train, self.scaler, df_cols, self.cfg["TARGET_COL"]
        )
        y_train_pred_inv = inverse_transform_predictions(
            y_train_pred_scaled, self.scaler, df_cols, self.cfg["TARGET_COL"]
        )
        train_mae, train_mse, train_rmse, train_r2_avg, train_r2_each = compute_metrics(
            y_train_inv, y_train_pred_inv
        )
        for day_idx in range(self.cfg["HORIZON"]):
            plot_scatter_day(
                y_train_inv,
                y_train_pred_inv,
                day_idx,
                self.station_folder,
                self.cfg["PLOTS_DIR"],
                dataset_type="train",
            )

        y_val_pred_scaled = self.model.predict(self.X_val)
        df_cols_val = list(self.df_val.columns)
        y_val_inv = inverse_transform_predictions(
            self.y_val, self.scaler, df_cols_val, self.cfg["TARGET_COL"]
        )
        y_val_pred_inv = inverse_transform_predictions(
            y_val_pred_scaled, self.scaler, df_cols_val, self.cfg["TARGET_COL"]
        )
        val_mae, val_mse, val_rmse, val_r2_avg, val_r2_each = compute_metrics(
            y_val_inv, y_val_pred_inv
        )
        for day_idx in range(self.cfg["HORIZON"]):
            plot_scatter_day(
                y_val_inv,
                y_val_pred_inv,
                day_idx,
                self.station_folder,
                self.cfg["PLOTS_DIR"],
                dataset_type="val",
            )

        results.update(
            {
                "train_loss": train_mse,
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
        )
        # Test, if available
        if not self.df_test.empty:
            df_test_scaled = pd.DataFrame(
                self.scaler.transform(self.df_test.values), columns=self.df_test.columns
            )
            X_test, y_test = create_sequences(
                df_test_scaled,
                self.cfg["WINDOW_SIZE"],
                self.cfg["HORIZON"],
                self.cfg["TARGET_COL"],
            )
            if len(X_test) > 0:
                y_test_pred_scaled = self.model.predict(X_test)
                df_cols_test = list(self.df_test.columns)
                y_test_inv = inverse_transform_predictions(
                    y_test, self.scaler, df_cols_test, self.cfg["TARGET_COL"]
                )
                y_pred_inv = inverse_transform_predictions(
                    y_test_pred_scaled,
                    self.scaler,
                    df_cols_test,
                    self.cfg["TARGET_COL"],
                )
                mae, mse, rmse, r2_avg, r2_each = compute_metrics(
                    y_test_inv, y_pred_inv
                )
                plot_time_series_predictions(
                    y_test_inv,
                    y_pred_inv,
                    self.cfg["HORIZON"],
                    self.station_folder,
                    self.cfg["PLOTS_DIR"],
                )
                for day_idx in range(self.cfg["HORIZON"]):
                    plot_scatter_day(
                        y_test_inv,
                        y_pred_inv,
                        day_idx,
                        self.station_folder,
                        self.cfg["PLOTS_DIR"],
                    )
                results.update(
                    {
                        "test_mae": mae,
                        "test_mse": mse,
                        "test_rmse": rmse,
                        "test_r2_avg": r2_avg,
                        "test_r2_day1": r2_each[0],
                        "test_r2_day2": r2_each[1],
                        "test_r2_day3": r2_each[2],
                    }
                )
        self.metrics_summary = results
        return results

    def run(self):
        """Run train -> evaluate as standard workflow."""
        success = self.train()
        if not success:
            return None
        return self.evaluate()
