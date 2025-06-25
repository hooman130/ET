"""Feature engineering utilities."""
import pandas as pd

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'day', 'month' features, drops columns not needed (Date, etc.),
    removes NaNs.
    """
    if df.empty:
        return df
    if "Date" in df.columns:
        df["day"] = df["Date"].dt.day
        df["month"] = df["Date"].dt.month

    drop_cols = ["Date", "Latitude", "Longitude", "Station"]
    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
