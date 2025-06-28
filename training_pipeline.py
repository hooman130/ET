from dataclasses import dataclass, field
from typing import Callable, Dict, List
import os
import pickle
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from config import (
    STATION_FOLDERS,
    TRAIN_PER_FARM,
    MODELS_DIR,
    WINDOW_SIZE,
    HORIZON,
    MAX_WORKERS,
    RANDOM_SEED,
)
from training_utils import (
    load_station_data,
    split_by_percentages,
    create_sequences,
    inverse_transform_predictions,
    compute_metrics,
    plot_time_series_predictions,
    plot_scatter_day,
    plot_training_history,
)


def get_plots_base_dir(variable_name: str) -> str:
    """Return base directory for plots of a given variable."""
    return os.path.join("plots", variable_name.lower())


@dataclass
class ModelConfig:
    """Configuration for building and training an LSTM model."""

    units: int = 64
    dropout: float = 0.0
    dense_units: int | None = None
    loss: str = "mse"
    metrics: List[str] = field(default_factory=lambda: ["mae"])
    learning_rate: float = 0.001


def build_lstm_model(num_features: int, horizon: int, cfg: ModelConfig) -> Sequential:
    """Construct and compile a simple LSTM model."""
    model = Sequential()
    model.add(Input(shape=(WINDOW_SIZE, num_features)))
    model.add(LSTM(cfg.units, activation="tanh"))
    if cfg.dropout > 0:
        model.add(Dropout(cfg.dropout))
    if cfg.dense_units:
        model.add(Dense(cfg.dense_units, activation="relu"))
    model.add(Dense(horizon))
    model.compile(
        loss=cfg.loss,
        optimizer=Adam(learning_rate=cfg.learning_rate),
        metrics=cfg.metrics,
    )
    return model


def scale_datasets(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale datasets using a StandardScaler fitted on the training set."""
    scaler = StandardScaler()
    scaler.fit(df_train.values)

    df_train_scaled = pd.DataFrame(scaler.transform(df_train.values), columns=df_train.columns)
    df_val_scaled = pd.DataFrame(scaler.transform(df_val.values), columns=df_val.columns)
    df_test_scaled = pd.DataFrame(scaler.transform(df_test.values), columns=df_test.columns)
    return df_train_scaled, df_val_scaled, df_test_scaled, scaler


class LSTMTrainer:
    """Reusable training pipeline for per-farm and combined scenarios."""

    def __init__(
        self,
        target_col: str,
        feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
        variable_name: str,
        model_suffix: str,
        scaler_suffix: str,
        cfg: ModelConfig | None = None,
        postprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.target_col = target_col
        self.feature_fn = feature_fn
        self.variable_name = variable_name
        self.model_suffix = model_suffix
        self.scaler_suffix = scaler_suffix
        self.cfg = cfg or ModelConfig()
        self.postprocess_fn = postprocess_fn

    # ---------------------------------------
    # Per-farm training
    # ---------------------------------------
    def _train_station(self, station_folder: str) -> Dict:
        df_station = load_station_data(station_folder)
        if df_station.empty:
            print(f"No data for station {station_folder}. Skipping.")
            return {}

        df_train, df_val, df_test = split_by_percentages(df_station)
        df_train = self.feature_fn(df_train)
        df_val = self.feature_fn(df_val)
        df_test = self.feature_fn(df_test)

        if df_train.empty:
            print(f"No training data for station {station_folder}. Skipping.")
            return {}

        df_train_s, df_val_s, df_test_s, scaler = scale_datasets(df_train, df_val, df_test)

        scaler_path = os.path.join(MODELS_DIR, f"{station_folder}_{self.scaler_suffix}.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        X_train, y_train = create_sequences(
            df_train_s,
            window_size=WINDOW_SIZE,
            horizon=HORIZON,
            target_col=self.target_col,
        )
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train, y_train = X_train[indices], y_train[indices]

        X_val, y_val = create_sequences(
            df_val_s,
            window_size=WINDOW_SIZE,
            horizon=HORIZON,
            target_col=self.target_col,
        )

        model = build_lstm_model(df_train.shape[1], HORIZON, self.cfg)
        early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early],
            verbose=0,
        )

        model_path = os.path.join(MODELS_DIR, f"{station_folder}_{self.model_suffix}.keras")
        model.save(model_path)

        base_dir = get_plots_base_dir(self.variable_name)
        station_plot_dir = os.path.join(base_dir, station_folder)
        os.makedirs(station_plot_dir, exist_ok=True)

        history_csv = os.path.join(station_plot_dir, "training_history_values.csv")
        pd.DataFrame(history.history).to_csv(history_csv, index=False)
        plot_training_history(history, os.path.join(station_plot_dir, "training_history.png"))

        # Training metrics
        y_train_pred = model.predict(X_train)
        cols = list(df_train.columns)
        y_train_inv = inverse_transform_predictions(y_train, scaler, cols, self.target_col)
        y_train_pred_inv = inverse_transform_predictions(y_train_pred, scaler, cols, self.target_col)
        if self.postprocess_fn is not None:
            y_train_inv = self.postprocess_fn(y_train_inv)
            y_train_pred_inv = self.postprocess_fn(y_train_pred_inv)
        tr_mae, tr_mse, tr_rmse, tr_r2_avg, tr_r2_each = compute_metrics(y_train_inv, y_train_pred_inv)

        for day in range(HORIZON):
            plot_scatter_day(
                y_train_inv,
                y_train_pred_inv,
                day,
                station_folder,
                station_plot_dir,
                self.variable_name,
                dataset_type="train",
            )

        # Validation metrics
        y_val_pred = model.predict(X_val)
        y_val_inv = inverse_transform_predictions(y_val, scaler, cols, self.target_col)
        y_val_pred_inv = inverse_transform_predictions(y_val_pred, scaler, cols, self.target_col)
        if self.postprocess_fn is not None:
            y_val_inv = self.postprocess_fn(y_val_inv)
            y_val_pred_inv = self.postprocess_fn(y_val_pred_inv)
        v_mae, v_mse, v_rmse, v_r2_avg, v_r2_each = compute_metrics(y_val_inv, y_val_pred_inv)

        for day in range(HORIZON):
            plot_scatter_day(
                y_val_inv,
                y_val_pred_inv,
                day,
                station_folder,
                station_plot_dir,
                self.variable_name,
                dataset_type="val",
            )

        metrics = {
            "farm": station_folder,
            "train_loss": history.history.get("loss", [None])[-1],
            "train_mae": tr_mae,
            "train_rmse": tr_rmse,
            "train_r2_avg": tr_r2_avg,
            "train_r2_day1": tr_r2_each[0],
            "train_r2_day2": tr_r2_each[1],
            "train_r2_day3": tr_r2_each[2],
            "val_loss": v_mse,
            "val_mae": v_mae,
            "val_rmse": v_rmse,
            "val_r2_avg": v_r2_avg,
            "val_r2_day1": v_r2_each[0],
            "val_r2_day2": v_r2_each[1],
            "val_r2_day3": v_r2_each[2],
            "test_mae": None,
            "test_mse": None,
            "test_rmse": None,
            "test_r2_avg": None,
            "test_r2_day1": None,
            "test_r2_day2": None,
            "test_r2_day3": None,
        }

        if not df_test.empty:
            df_test_s = pd.DataFrame(scaler.transform(df_test.values), columns=df_test.columns)
            X_test, y_test = create_sequences(
                df_test_s,
                window_size=WINDOW_SIZE,
                horizon=HORIZON,
                target_col=self.target_col,
            )
            if len(X_test) > 0:
                y_test_pred = model.predict(X_test)
                test_cols = list(df_test.columns)
                y_test_inv = inverse_transform_predictions(y_test, scaler, test_cols, self.target_col)
                y_test_pred_inv = inverse_transform_predictions(y_test_pred, scaler, test_cols, self.target_col)
                if self.postprocess_fn is not None:
                    y_test_inv = self.postprocess_fn(y_test_inv)
                    y_test_pred_inv = self.postprocess_fn(y_test_pred_inv)
                mae, mse, rmse, r2_avg, r2_each = compute_metrics(y_test_inv, y_test_pred_inv)
                plot_time_series_predictions(
                    y_test_inv,
                    y_test_pred_inv,
                    HORIZON,
                    station_folder,
                    station_plot_dir,
                    self.variable_name,
                )
                for day in range(HORIZON):
                    plot_scatter_day(
                        y_test_inv,
                        y_test_pred_inv,
                        day,
                        station_folder,
                        station_plot_dir,
                        self.variable_name,
                    )
                metrics.update(
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
        pd.DataFrame([metrics]).round(4).to_csv(
            os.path.join(station_plot_dir, "metrics.csv"), index=False
        )
        return metrics

    # ---------------------------------------
    # Combined training
    # ---------------------------------------
    def _train_combined(self) -> None:
        train_list: List[pd.DataFrame] = []
        val_list: List[pd.DataFrame] = []
        test_data_by_station: Dict[str, pd.DataFrame] = {}

        for station in STATION_FOLDERS:
            df_station = load_station_data(station)
            if df_station.empty:
                continue
            df_train, df_val, df_test = split_by_percentages(df_station)
            df_train = self.feature_fn(df_train)
            df_val = self.feature_fn(df_val)
            df_test = self.feature_fn(df_test)
            if not df_train.empty:
                train_list.append(df_train)
            if not df_val.empty:
                val_list.append(df_val)
            test_data_by_station[station] = df_test

        if not train_list:
            print("No training data found.")
            return

        df_train_all = pd.concat(train_list, ignore_index=True)
        scaler = StandardScaler()
        scaler.fit(df_train_all.values)
        df_train_s = pd.DataFrame(scaler.transform(df_train_all.values), columns=df_train_all.columns)

        df_val_all = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame(columns=df_train_all.columns)
        df_val_s = pd.DataFrame(scaler.transform(df_val_all.values), columns=df_val_all.columns) if not df_val_all.empty else df_val_all

        with open(f"scaler_model_{self.variable_name}.pkl", "wb") as f:
            pickle.dump(scaler, f)

        X_train, y_train = create_sequences(
            df_train_s, window_size=WINDOW_SIZE, horizon=HORIZON, target_col=self.target_col
        )
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train, y_train = X_train[indices], y_train[indices]

        X_val, y_val = create_sequences(
            df_val_s, window_size=WINDOW_SIZE, horizon=HORIZON, target_col=self.target_col
        )

        model = build_lstm_model(df_train_all.shape[1], HORIZON, self.cfg)
        early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early],
            verbose=0,
        )
        model.save(f"model_{self.variable_name.lower()}.keras")

        base_dir = get_plots_base_dir(self.variable_name)
        plots_dir = os.path.join(base_dir, "combined")
        os.makedirs(plots_dir, exist_ok=True)

        pd.DataFrame(history.history).to_csv(os.path.join(plots_dir, "training_history_values.csv"), index=False)
        plot_training_history(history, os.path.join(plots_dir, "training_history.png"))

        y_train_pred = model.predict(X_train)
        cols = list(df_train_all.columns)
        y_train_inv = inverse_transform_predictions(y_train, scaler, cols, self.target_col)
        y_train_pred_inv = inverse_transform_predictions(y_train_pred, scaler, cols, self.target_col)
        tr_mae, tr_mse, tr_rmse, tr_r2_avg, tr_r2_each = compute_metrics(y_train_inv, y_train_pred_inv)
        for day in range(HORIZON):
            plot_scatter_day(
                y_train_inv,
                y_train_pred_inv,
                day,
                "combined",
                plots_dir,
                self.variable_name,
                dataset_type="train",
            )

        y_val_pred = model.predict(X_val)
        y_val_inv = inverse_transform_predictions(y_val, scaler, cols, self.target_col)
        y_val_pred_inv = inverse_transform_predictions(y_val_pred, scaler, cols, self.target_col)
        v_mae, v_mse, v_rmse, v_r2_avg, v_r2_each = compute_metrics(y_val_inv, y_val_pred_inv)
        for day in range(HORIZON):
            plot_scatter_day(
                y_val_inv,
                y_val_pred_inv,
                day,
                "combined",
                plots_dir,
                self.variable_name,
                dataset_type="val",
            )

        train_val_metrics = {
            "farm": "combined",
            "train_loss": history.history.get("loss", [None])[-1],
            "train_mae": tr_mae,
            "train_rmse": tr_rmse,
            "train_r2_avg": tr_r2_avg,
            "train_r2_day1": tr_r2_each[0],
            "train_r2_day2": tr_r2_each[1],
            "train_r2_day3": tr_r2_each[2],
            "val_loss": v_mse,
            "val_mae": v_mae,
            "val_rmse": v_rmse,
            "val_r2_avg": v_r2_avg,
            "val_r2_day1": v_r2_each[0],
            "val_r2_day2": v_r2_each[1],
            "val_r2_day3": v_r2_each[2],
        }

        test_metrics_list: List[Dict] = []
        for station in STATION_FOLDERS:
            df_test = test_data_by_station.get(station, pd.DataFrame())
            if df_test.empty:
                continue
            df_test_s = pd.DataFrame(scaler.transform(df_test.values), columns=df_test.columns)
            X_test, y_test = create_sequences(
                df_test_s, window_size=WINDOW_SIZE, horizon=HORIZON, target_col=self.target_col
            )
            if len(X_test) == 0:
                continue
            y_pred = model.predict(X_test)
            cols = list(df_test.columns)
            y_test_inv = inverse_transform_predictions(y_test, scaler, cols, self.target_col)
            y_pred_inv = inverse_transform_predictions(y_pred, scaler, cols, self.target_col)
            if self.postprocess_fn is not None:
                y_test_inv = self.postprocess_fn(y_test_inv)
                y_pred_inv = self.postprocess_fn(y_pred_inv)
            mae, mse, rmse, r2_avg, r2_each = compute_metrics(y_test_inv, y_pred_inv)
            plot_time_series_predictions(
                y_test_inv, y_pred_inv, HORIZON, station, plots_dir, self.variable_name
            )
            for day in range(HORIZON):
                plot_scatter_day(
                    y_test_inv, y_pred_inv, day, station, plots_dir, self.variable_name
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
        train_val_df.to_csv(os.path.join(base_dir, "training_validation_metrics.csv"), index=False)
        test_df.to_csv(os.path.join(base_dir, "test_metrics.csv"), index=False)

    # ---------------------------------------
    def run(self) -> None:
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)
        if TRAIN_PER_FARM:
            metrics_list: List[Dict] = []
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = {ex.submit(self._train_station, st): st for st in STATION_FOLDERS}
                for fut in as_completed(futures):
                    res = fut.result()
                    if res:
                        metrics_list.append(res)
            if metrics_list:
                df = pd.DataFrame(metrics_list).round(4)
                base_dir = get_plots_base_dir(self.variable_name)
                os.makedirs(base_dir, exist_ok=True)
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
                df[train_val_cols].to_csv(
                    os.path.join(base_dir, "training_validation_metrics.csv"),
                    index=False,
                )
                df[test_cols].to_csv(
                    os.path.join(base_dir, "test_metrics.csv"), index=False
                )
        else:
            self._train_combined()


