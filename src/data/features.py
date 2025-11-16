import pandas as pd
import numpy as np
from typing import List, Dict


class FeatureEngineer:
    def __init__(self):
        pass

    def compute_realized_volatility(
        self, df: pd.DataFrame, freq: str = "1D", return_col: str = "log_returns"
    ) -> pd.DataFrame:
        df = df.copy()

        if return_col not in df.columns:
            df["log_returns"] = np.log(
                df["last"] / df.groupby("Future")["last"].shift(1)
            )

        df["date"] = df["datetime"].dt.date

        rv_list = []
        for instrument in df["Future"].unique():
            inst_df = df[df["Future"] == instrument].copy()

            if freq == "1D":
                grouped = inst_df.groupby("date")
                group_col = "date"
            elif freq == "1W":
                inst_df["week"] = inst_df["datetime"].dt.to_period("W")
                grouped = inst_df.groupby("week")
                group_col = "week"
            elif freq == "1M":
                inst_df["month"] = inst_df["datetime"].dt.to_period("M")
                grouped = inst_df.groupby("month")
                group_col = "month"
            else:
                raise ValueError(f"Unsupported frequency: {freq}")

            rv = grouped[return_col].apply(lambda x: np.sqrt(np.sum(x**2)))
            rv_df = rv.reset_index()
            rv_df.columns = [group_col, f"RV_{freq}"]
            rv_df["Future"] = instrument

            rv_list.append(rv_df)

        rv_combined = pd.concat(rv_list, ignore_index=True)
        return rv_combined

    def compute_realized_quarticity(
        self, df: pd.DataFrame, freq: str = "1D", return_col: str = "log_returns"
    ) -> pd.DataFrame:
        df = df.copy()

        if return_col not in df.columns:
            df["log_returns"] = np.log(
                df["last"] / df.groupby("Future")["last"].shift(1)
            )

        df["date"] = df["datetime"].dt.date

        rq_list = []
        for instrument in df["Future"].unique():
            inst_df = df[df["Future"] == instrument].copy()

            if freq == "1D":
                grouped = inst_df.groupby("date")
                group_col = "date"
            elif freq == "1W":
                inst_df["week"] = inst_df["datetime"].dt.to_period("W")
                grouped = inst_df.groupby("week")
                group_col = "week"
            elif freq == "1M":
                inst_df["month"] = inst_df["datetime"].dt.to_period("M")
                grouped = inst_df.groupby("month")
                group_col = "month"
            else:
                raise ValueError(f"Unsupported frequency: {freq}")

            rq = grouped[return_col].apply(lambda x: np.sum(x**4))
            rq_df = rq.reset_index()
            rq_df.columns = [group_col, f"RQ_{freq}"]
            rq_df["Future"] = instrument

            rq_list.append(rq_df)

        rq_combined = pd.concat(rq_list, ignore_index=True)
        return rq_combined

    def add_lagged_features(
        self, df: pd.DataFrame, feature_cols: List[str], lags: List[int]
    ) -> pd.DataFrame:
        df = df.copy()

        for col in feature_cols:
            for lag in lags:
                df[f"{col}_lag{lag}"] = df.groupby("Future")[col].shift(lag)

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["hour"] = df["datetime"].dt.hour
        df["day_of_week"] = df["datetime"].dt.dayofweek
        df["month"] = df["datetime"].dt.month
        df["day_of_month"] = df["datetime"].dt.day
        df["quarter"] = df["datetime"].dt.quarter

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        return df

    def add_volume_features(
        self, df: pd.DataFrame, windows: List[int] = [20, 60]
    ) -> pd.DataFrame:
        df = df.copy()

        for window in windows:
            df[f"volume_ma_{window}"] = df.groupby("Future")["volume"].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

            df[f"volume_std_{window}"] = df.groupby("Future")["volume"].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )

        df["volume_log"] = np.log1p(df["volume"])

        return df

    def create_har_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "RV_1D" not in df.columns:
            raise ValueError("Daily RV not found. Compute RV first.")

        df["RV_daily"] = df.groupby("Future")["RV_1D"].shift(1)

        df["RV_weekly"] = (
            df.groupby("Future")["RV_1D"]
            .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
            .shift(1)
        )

        df["RV_monthly"] = (
            df.groupby("Future")["RV_1D"]
            .transform(lambda x: x.rolling(window=22, min_periods=1).mean())
            .shift(1)
        )

        return df

    def detect_jumps(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        df = df.copy()

        if "RV_1D" not in df.columns or "RQ_1D" not in df.columns:
            raise ValueError("RV and RQ required for jump detection")

        df["bipower_var"] = df["RV_1D"] ** 2 - df["RQ_1D"]
        df["bipower_var"] = df["bipower_var"].clip(lower=0)

        df["jump_test"] = (df["RV_1D"] ** 2 - df["bipower_var"]) / df["RV_1D"]

        df["has_jump"] = df["jump_test"] > threshold

        return df

    def build_full_feature_set(
        self, df: pd.DataFrame, config: Dict = None
    ) -> pd.DataFrame:
        print("Building full feature set")

        df = df.copy()

        print("Computing returns")
        if "log_returns" not in df.columns:
            df["log_returns"] = np.log(
                df["last"] / df.groupby("Future")["last"].shift(1)
            )

        print("Computing daily RV")
        rv_daily = self.compute_realized_volatility(df, freq="1D")
        df = df.merge(rv_daily, on=["Future", "date"], how="left")

        print("Computing weekly RV")
        rv_weekly = self.compute_realized_volatility(df, freq="1W")
        df["week"] = df["datetime"].dt.to_period("W")
        df = df.merge(rv_weekly, on=["Future", "week"], how="left")

        print("Computing monthly RV")
        rv_monthly = self.compute_realized_volatility(df, freq="1M")
        df["month_rv"] = df["datetime"].dt.to_period("M")
        df = df.merge(
            rv_monthly,
            left_on=["Future", "month_rv"],
            right_on=["Future", "month"],
            how="left",
        )
        df = df.drop(columns=["month_rv", "month"])

        print("Adding HAR features")
        df = self.create_har_features(df)

        print("Adding time features")
        df = self.add_time_features(df)

        print("Adding volume features")
        df = self.add_volume_features(df)

        print("Dropping NaN rows")
        initial_len = len(df)
        df = df.dropna(subset=["RV_daily", "RV_weekly", "RV_monthly"])
        removed = initial_len - len(df)
        print(f"Removed {removed} rows with NaN values")

        print("Feature engineering complete")
        return df
