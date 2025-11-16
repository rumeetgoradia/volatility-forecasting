import yaml
import pandas as pd
from pathlib import Path
import sys

sys.path.append("src")
from data.loader import DataLoader
from data.preprocessing import DataPreprocessor
from data.features import FeatureEngineer
from data.splitter import DataSplitter


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    print("DATA PREPARATION PIPELINE")

    config = load_config()

    print("Loading data")
    loader = DataLoader(config["data"]["raw_path"])
    df = loader.load(instruments=config["data"]["instruments"])
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
