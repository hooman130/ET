import os
import pandas as pd


def analyze_histories(directory="."):
    records = []
    for fname in os.listdir(directory):
        if fname.endswith("_history.csv"):
            path = os.path.join(directory, fname)
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"Could not read {path}: {e}")
                continue
            session = fname.rsplit("_history.csv", 1)[0]
            epochs = len(df)
            train_loss_last = df.get("loss", pd.Series([None])).iloc[-1]
            val_loss_last = df.get("val_loss", pd.Series([None])).iloc[-1]
            overfit = (
                val_loss_last - train_loss_last
                if train_loss_last is not None and val_loss_last is not None
                else None
            )
            improvement = (
                df.get("val_loss", pd.Series([None])).iloc[0] - val_loss_last
                if val_loss_last is not None and "val_loss" in df
                else None
            )
            records.append(
                {
                    "session": session,
                    "epochs": epochs,
                    "train_loss_last": train_loss_last,
                    "val_loss_last": val_loss_last,
                    "val_minus_train": overfit,
                    "val_loss_reduction": improvement,
                }
            )
    if records:
        summary_df = pd.DataFrame(records)
        summary_df.to_csv(os.path.join(directory, "history_analysis.csv"), index=False)
        print(f"Summary saved to {os.path.join(directory, 'history_analysis.csv')}")
    else:
        print("No history files found for analysis.")


if __name__ == "__main__":
    analyze_histories(os.path.dirname(__file__))
