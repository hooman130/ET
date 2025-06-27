"""LSTM training for evapotranspiration forecasting."""

from training_pipeline import LSTMTrainer, ModelConfig

TARGET_COL = "ET (mm/day)"


def feature_engineering(df):
    """Add time based features and drop unused columns."""
    if df.empty:
        return df
    if "Date" in df.columns:
        df["day"] = df["Date"].dt.day
        df["month"] = df["Date"].dt.month
    for col in ["Date", "Latitude", "Longitude", "Station"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


config = ModelConfig(units=64, loss="mse", metrics=["mae"])
trainer = LSTMTrainer(
    target_col=TARGET_COL,
    feature_fn=feature_engineering,
    variable_name="ET",
    model_suffix="model_lstm",
    scaler_suffix="scaler_ET",
    cfg=config,
)

if __name__ == "__main__":
    trainer.run()
