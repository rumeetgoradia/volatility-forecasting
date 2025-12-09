import yaml
import pandas as pd
from pathlib import Path
import sys

sys.path.append("src")
from data.loader import DataLoader
from data.preprocessing import DataPreprocessor
from data.features import FeatureEngineer
from data.splitter import DataSplitter
from data.horizons import add_intraday_horizon_columns, downsample_hourly_rows
from data.validation import assert_hourly_downsampled


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    print("DATA PREPARATION PIPELINE")

    config = load_config()
    target_cfg = config.get("target", {})

    print("Loading data")
    loader = DataLoader(config["data"]["raw_path"])
    df = loader.load(instruments=config["data"]["instruments"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    print(f"Loaded {len(df):,} records")
    print(f"Instruments: {loader.get_instruments_list()}")

    print("Preprocessing")
    preprocessor = DataPreprocessor(
        remove_zero_volume=config["preprocessing"]["remove_zero_volume"],
        outlier_threshold=config["preprocessing"]["outlier_threshold"],
    )
    df = preprocessor.preprocess(df)
    print(f"After preprocessing: {len(df):,} records")

    print("Feature engineering")
    engineer = FeatureEngineer()

    df["date"] = df["datetime"].dt.date

    df = engineer.build_full_feature_set(df, config=config["features"])
    print(f"After feature engineering: {len(df):,} records")

    if not config["features"].get("keep_monthly_rv", True):
        drop_cols = [c for c in ["RV_1M", "RV_monthly"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            print(f"Dropped monthly RV cols: {drop_cols}")

    intr = config["features"].get("intraday", {})
    if intr:
        print("Adding intraday short-horizon features")
        df = engineer.add_intraday_simple_features(
            df,
            rv_windows_minutes=intr.get("rv_windows_minutes", []),
            rq_windows_minutes=intr.get("rq_windows_minutes", []),
            vol_corr_windows_minutes=intr.get("vol_corr_windows_minutes", []),
            bar_minutes=target_cfg.get("bar_minutes", 5),
        )

    print("Adding 1H target/features")
    df = add_intraday_horizon_columns(
        df,
        target_col=target_cfg.get("target_col", "RV_1H"),
        horizon_minutes=target_cfg.get("horizon_minutes", 60),
        bar_minutes=target_cfg.get("bar_minutes", 5),
        return_col="log_returns",
        har_windows=target_cfg.get("har_windows", [1, 6, 24]),
    )

    minute_mark = target_cfg.get("hourly_minute")
    if minute_mark is None:
        raise ValueError("target.hourly_minute must be set to enforce hourly downsampling.")

    print(f"Downsampling to hourly rows at minute={minute_mark}")
    df = downsample_hourly_rows(df, minute=minute_mark)
    print(f"After downsampling: {len(df):,} records")
    assert_hourly_downsampled([("full", df)], minute_mark)

    print("Splitting data")
    splitter = DataSplitter(
        train_end=config["data"]["split"]["train_end"],
        val_end=config["data"]["split"]["val_end"],
    )
    train_df, val_df, test_df = splitter.split(df)

    print("Saving processed data")
    output_dir = Path(config["data"]["processed_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    print(f"Saved to {output_dir}/")
    print(f"train.parquet: {len(train_df):,} records")
    print(f"val.parquet: {len(val_df):,} records")
    print(f"test.parquet: {len(test_df):,} records")

    print("PIPELINE COMPLETE")

    feature_cols = [
        col
        for col in df.columns
        if col not in ["timestamp", "Future", "datetime", "date", "week"]
    ]
    print(f"Total features: {len(feature_cols)}")

    print(f"Missing values in train: {train_df.isnull().sum().sum()}")
    print(f"Missing values in val: {val_df.isnull().sum().sum()}")
    print(f"Missing values in test: {test_df.isnull().sum().sum()}")


if __name__ == "__main__":
    main()
