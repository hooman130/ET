"""LSTM training for rainfall forecasting."""

import numpy as np
from training_pipeline import LSTMTrainer, ModelConfig

TARGET_COL = "Rainfall (mm)"


def feature_engineering(df):
    """Feature engineering for rainfall data."""
    if df.empty:
        return df
    if "Date" in df.columns:
        df["day"] = df["Date"].dt.day
        df["month"] = df["Date"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)
    if "Tmin (°C)" in df.columns:
        df.loc[df["Tmin (°C)"] < -10, "Tmin (°C)"] = np.nan
    if "Rainfall (mm)" in df.columns:
        df = df[df["Rainfall (mm)"].notna()]
        df.loc[:, "Rainfall (mm)"] = np.log1p(df["Rainfall (mm)"])
    if "relative_humidity" in df.columns:
        df.loc[:, "relative_humidity"] = df["relative_humidity"]
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


def inverse_log(arr: np.ndarray) -> np.ndarray:
    return np.expm1(arr)


config = ModelConfig(units=16, dropout=0.2, dense_units=16, loss="mae", metrics=["mse"])
trainer = LSTMTrainer(
    target_col=TARGET_COL,
    feature_fn=feature_engineering,
    variable_name="Rainfall",
    model_suffix="model_rain_lstm",
    scaler_suffix="scaler_Rain",
    cfg=config,
    postprocess_fn=inverse_log,
)

if __name__ == "__main__":
    trainer.run()
