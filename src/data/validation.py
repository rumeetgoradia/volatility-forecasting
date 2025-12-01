import pandas as pd


def assert_hourly_downsampled(dfs, minute: int, datetime_col: str = "datetime"):
    """
    Ensure all provided DataFrames are downsampled to one row per hour at the given minute.
    dfs: iterable of (name, DataFrame) pairs for clearer error messages.
    """
    if minute is None:
        raise ValueError("Config target.hourly_minute must be set to enforce hourly downsampled data.")

    for name, df in dfs:
        if datetime_col not in df.columns:
            raise ValueError(f"{name} is missing '{datetime_col}' for frequency validation.")

        minutes = pd.to_datetime(df[datetime_col]).dt.minute
        unique_minutes = pd.Series(minutes.unique()).dropna().astype(int).tolist()

        if len(unique_minutes) != 1 or unique_minutes[0] != int(minute):
            raise ValueError(
                f"{name} is not hourly downsampled at minute={minute}. "
                f"Found minute values: {sorted(unique_minutes)}"
            )
