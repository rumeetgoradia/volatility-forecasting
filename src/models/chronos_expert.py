import torch
import torch.nn as nn
from typing import Optional

try:
    from chronos import ChronosPipeline
except ImportError:
    print("Warning: Chronos not installed.")
    ChronosPipeline = None

class ChronosExpert(nn.Module):
    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        prediction_length: int = 1,
        num_samples: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        context_length: int = 20,
    ):
        super().__init__()
        self.device = device
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.context_length = context_length

        print(f"Loading Chronos Expert ({model_name}) on {device}...")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )

        # Freeze parameters to prevent gradients from exploding
        for param in self.pipeline.model.parameters():
            param.requires_grad = False

    def forward(self, x_raw: torch.Tensor):
        # FIX: The Chronos Tokenizer boundaries reside on the CPU.
        # Even if the model is on CUDA, we MUST pass CPU tensors to the predict method
        # or the tokenizer will crash with a device mismatch error.
        context_tensor = x_raw.cpu()

        with torch.no_grad():
            # Pass the CPU tensor (positional argument)
            forecast = self.pipeline.predict(
                context_tensor,
                prediction_length=self.prediction_length,
                num_samples=self.num_samples,
                limit_prediction_length=False
            )

        # forecast is returned as a tensor. Calculating the median here.
        # Shape: [Batch, Samples, Horizon] -> [Batch, Horizon]
        median_forecast = torch.quantile(forecast, 0.5, dim=1)

        # Select the first step
        point_forecast = median_forecast[:, 0]

        # Reshape to [Batch, 1] and move back to GPU to match the MoE pipeline
        output = point_forecast.unsqueeze(-1).to(self.device)

        return output.float()
