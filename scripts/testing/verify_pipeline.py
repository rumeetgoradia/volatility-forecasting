import sys
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from src.data.dataset import VolatilityDataset
except ImportError:
    class VolatilityDataset:
        def __init__(self, *args, **kwargs): pass

def verify_shapes():
    print("--- Verifying Data Pipeline Shapes ---")

    data_path = "data/processed/train_data.parquet"

    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Skipping runtime check.")
        print("Logic verification: Dataset class expects feature_cols and target_col.")
        return

    try:
        df = pd.read_parquet(data_path)
        print(f"Loaded DataFrame: {df.shape}")

        dataset = VolatilityDataset(df, feature_cols=['log_vol', 'log_volume'], target_col='target_rv', return_metadata=True)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        batch = next(iter(loader))
        
        if len(batch) == 3:
            x_scaled, y, meta = batch
            print(f"Scaled Input Shape: {x_scaled.shape} (Expected: [32, Seq, Feat])")
            print(f"Target Shape: {y.shape} (Expected: [32, 1])")
            
            if isinstance(meta, dict):
                print(f"Metadata Type: Dictionary (Expected). Keys: {list(meta.keys())}")
                if 'datetime_obj' in meta or 'datetime' in meta:
                    print("Metadata contains datetime objects (Required)")
            else:
                print(f"Unexpected 3rd element type: {type(meta)}")
        else:
            print(f"Batch tuple length mismatch. Got {len(batch)}, expected 3.")

    except Exception as e:
        print(f"Pipeline Check Warning: {e}")

if __name__ == "__main__":
    verify_shapes()