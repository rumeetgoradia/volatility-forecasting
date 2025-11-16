import pandas as pd
from typing import Tuple, Dict


class DataSplitter:
    def __init__(self, train_end: str, val_end: str):
        self.train_end = train_end
        self.val_end = val_end

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = df.sort_values("datetime").reset_index(drop=True)

        train_df = df[df["datetime"] <= self.train_end].copy()
        val_df = df[
            (df["datetime"] > self.train_end) & (df["datetime"] <= self.val_end)
        ].copy()
        test_df = df[df["datetime"] > self.val_end].copy()

        print(
            f"Train: {len(train_df)} records ({train_df['datetime'].min()} to {train_df['datetime'].max()})"
        )
        print(
            f"Val:   {len(val_df)} records ({val_df['datetime'].min()} to {val_df['datetime'].max()})"
        )
        print(
            f"Test:  {len(test_df)} records ({test_df['datetime'].min()} to {test_df['datetime'].max()})"
        )

        return train_df, val_df, test_df

    def get_split_info(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Dict:
        return {
            "train": {
                "n_records": len(train_df),
                "n_instruments": train_df["Future"].nunique(),
                "date_range": (train_df["datetime"].min(), train_df["datetime"].max()),
            },
            "val": {
                "n_records": len(val_df),
                "n_instruments": val_df["Future"].nunique(),
                "date_range": (val_df["datetime"].min(), val_df["datetime"].max()),
            },
            "test": {
                "n_records": len(test_df),
                "n_instruments": test_df["Future"].nunique(),
                "date_range": (test_df["datetime"].min(), test_df["datetime"].max()),
            },
        }
