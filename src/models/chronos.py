import torch
import torch.nn as nn
from chronos import ChronosPipeline
import traceback
import numpy as np

class ChronosExpert(nn.Module):
    def __init__(self, model_name="amazon/chronos-t5-small", device="cuda"):
        super().__init__()
        self.device = device
        print(f"Loading Chronos Expert ({model_name})...")
        
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16, 
        )
    
    @torch.no_grad()
    def forward(self, x, prediction_length=1):
        try:
            # 1. Data Prep: Convert 3D Tensor to List of 1D Tensors
            context_list = []
            for ts in x:
                # FIX: Chronos is Univariate. Take index 0.
                if ts.dim() > 1:
                    ts_univariate = ts[:, 0] 
                else:
                    ts_univariate = ts
                context_list.append(ts_univariate.detach().cpu())
            
            # 2. Inference (Low samples for memory safety)
            forecast = self.pipeline.predict(
                context_list,
                prediction_length,
                5,   # Reduced samples (20 -> 5)
                None
            )
            
            # 3. Aggregation
            median_forecast = forecast.median(axis=1).values
            
            # 4. Safe Conversion to Tensor
            if isinstance(median_forecast, np.ndarray):
                forecast_tensor = torch.from_numpy(median_forecast).to(self.device).float()
            elif torch.is_tensor(median_forecast):
                forecast_tensor = median_forecast.clone().detach().to(self.device).float()
            else:
                forecast_tensor = torch.tensor(median_forecast).to(self.device).float()

            # 5. BRUTE FORCE RESHAPE (Fixes the broadcasting/fake loss issue)
            # We strictly enforce [Batch, 1] output
            current_batch_size = x.shape[0]
            flat = forecast_tensor.view(current_batch_size, -1)
            forecast_tensor = flat[:, 0:1]

            return forecast_tensor

        except Exception as e:
            print(f"\\n!!! CHRONOS CRASH !!!")
            traceback.print_exc()
            return torch.zeros((x.shape[0], 1), device=self.device)