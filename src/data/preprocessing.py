import pandas as pd
import numpy as np
from typing import Optional


class DataPreprocessor:
    def __init__(
        self, remove_zero_volume: bool = True, outlier_threshold: float = 0.05
    ):
        self.remove_zero_volume = remove_zero_volume
        self.outlier_threshold = outlier_threshold

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = self._remove_duplicates(df)

        if self.remove_zero_volume:
            df = self._filter_zero_volume(df)

        df = self._add_returns(df)

        df = self._handle_outliers(df)

        df = self._add_gaps_indicator(df)

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_len = len(df)
        df = df.drop_duplicates(subset=["Future", "datetime"], keep="first")
        removed = initial_len - len(df)
        if removed > 0:
            print(f"Removed {removed} duplicate records")
        return df

    def _filter_zero_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_len = len(df)
        df = df[df["volume"] > 0].copy()
        removed = initial_len - len(df)
        if removed > 0:
            print(
                f"Removed {removed} zero-volume records ({100*removed/initial_len:.2f}%)"
            )
        return df

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["returns"] = df.groupby("Future")["last"].pct_change()
        df["log_returns"] = np.log(df["last"] / df.groupby("Future")["last"].shift(1))
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.outlier_threshold is None:
            return df

        initial_len = len(df)

        mask = df["returns"].abs() > self.outlier_threshold
        outliers = mask.sum()

        if outliers > 0:
            print(f"Found {outliers} outliers (>{self.outlier_threshold*100}% moves)")
            print(f"Keeping outliers but flagging them")
            df["is_outlier"] = mask
        else:
            df["is_outlier"] = False

        return df

    def _add_gaps_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        df["time_diff"] = df.groupby("Future")["datetime"].diff()

        expected_interval = pd.Timedelta(minutes=5)
        tolerance = pd.Timedelta(seconds=30)

        df["is_gap"] = df["time_diff"] > (expected_interval + tolerance)

        n_gaps = df["is_gap"].sum()
        if n_gaps > 0:
            print(f"Identified {n_gaps} time gaps (>{expected_interval + tolerance})")

        return df

    def get_clean_trading_hours(
        self, df: pd.DataFrame, start_hour: int = 8, end_hour: int = 22
    ) -> pd.DataFrame:
        df = df.copy()
        df["hour"] = df["datetime"].dt.hour
        return df[(df["hour"] >= start_hour) & (df["hour"] <= end_hour)]

    def resample_to_regular_intervals(
        self, df: pd.DataFrame, freq: str = "5T"
    ) -> pd.DataFrame:
        df = df.set_index("datetime")

        resampled_dfs = []
        for instrument in df["Future"].unique():
            inst_df = df[df["Future"] == instrument].copy()

            inst_resampled = (
                inst_df.resample(freq).agg({"last": "last", "volume": "sum"}).dropna()
            )

            inst_resampled["Future"] = instrument
            resampled_dfs.append(inst_resampled)

        result = pd.concat(resampled_dfs).reset_index()
        return result
