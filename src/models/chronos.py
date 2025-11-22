import torch
import torch.nn as nn
from chronos import ChronosPipeline

class ChronosExpert(nn.Module):
    def __init__(self, model_name="amazon/chronos-bolt-small", device="cuda"):
        super().__init__()
        self.device = device
        print(f"Loading Chronos Expert ({model_name})...")

        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16, #adjust this according to compute available
        )
        
    def forward(self, x, prediction_length=1):
        context_list = [ts.detach().cpu() for ts in x]
        forecast = self.pipeline.predict(
            context=context_list,
            prediction_length=prediction_length,
            num_samples=20, 
            limit_prediction_length=False 
        )

        forecast_mean = torch.tensor(forecast.median(axis=1).values).to(self.device) #Taking median instead of mean
        return forecast_mean