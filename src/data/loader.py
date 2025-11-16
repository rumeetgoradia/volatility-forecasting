import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
import warnings

warnings.filterwarnings("ignore")


class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.df = None

    def load(self, instruments: Optional[List[str]] = None) -> pd.DataFrame:
        self.df = pd.read_parquet(self.file_path)

        if "datetime" not in self.df.columns and "timestamp" in self.df.columns:
            self.df["datetime"] = pd.to_datetime(self.df["timestamp"], unit="us")

        self.df = self.df.sort_values(["Future", "datetime"]).reset_index(drop=True)

        if instruments:
            self.df = self.df[self.df["Future"].isin(instruments)]

        return self.df

    def get_instrument(self, instrument: str) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self.df[self.df["Future"] == instrument].copy()

    def get_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        mask = (self.df["datetime"] >= start_date) & (self.df["datetime"] <= end_date)
        return self.df[mask].copy()

    def get_instruments_list(self) -> List[str]:
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self.df["Future"].unique().tolist()

    def info(self) -> dict:
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        return {
            "n_records": len(self.df),
            "n_instruments": self.df["Future"].nunique(),
            "instruments": self.df["Future"].unique().tolist(),
            "date_range": (self.df["datetime"].min(), self.df["datetime"].max()),
            "memory_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
        }
