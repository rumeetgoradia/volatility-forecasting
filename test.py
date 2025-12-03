import pandas as pd
train = pd.read_parquet('data/processed/train.parquet')
# Should see these columns now:
print("Target column exists:", 'RV_1H' in train.columns)
print("Hourly features exist:", all(c in train.columns for c in ['RV_H1', 'RV_H6', 'RV_H24']))
# Should be ~1/12th of original size (hourly vs 5-min)
print("Shape:", train.shape)  # Expect ~216,000 rows instead of 2.59M
# Check target values
print("\nTarget statistics:")
print(train['RV_1H'].describe())
# Verify hourly sampling
print("\nMinute marks (should all be 55):")
print(train['datetime'].dt.minute.value_counts())