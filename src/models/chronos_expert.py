import torch
import torch.nn as nn
from typing import Optional

# accurate handling of the chronos dependency
try:
    from chronos import ChronosPipeline
except ImportError:
    print("Warning: Chronos not installed. Please install via: pip install git+https://github.com/amazon-science/chronos-forecasting.git")
    ChronosPipeline = None

class ChronosExpert(nn.Module):
    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        prediction_length: int = 1, # We only need the next step (RV_1D)
        num_samples: int = 5,      # Number of probabilistic samples to generate
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        context_length: int = 20,   # Matches your sequence_length
    ):
        """
        Adapter class for Amazon Chronos Foundation Model.
        Wraps the probabilistic forecaster to output a deterministic point forecast compatible with MoE.
        """
        super().__init__()
        
        self.device = device
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.context_length = context_length

        print(f"Loading Chronos Expert ({model_name}) on {device}...")
        
        # Load the pre-trained pipeline
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        
        # FREEZE THE MODEL: We cannot backpropagate through the LLM in this setup
        for param in self.pipeline.model.parameters():
            param.requires_grad = False

    def forward(self, x_raw: torch.Tensor):
        """
        Args:
            x_raw: Tensor of shape (Batch, Sequence_Length). 
                   Must be the RAW values (not scaled).
        Returns:
            Tensor of shape (Batch, 1) representing the point forecast.
        """
        # Ensure input is on the correct device
        # Chronos expects the context to be a tensor on the same device
        if x_raw.device.type != self.device:
            x_raw = x_raw.to(self.device)

        # Run Inference (No Gradients)
        with torch.no_grad():
            # forecast shape: [Batch, Num_Samples, Prediction_Length]
            # Example: [64, 20, 1] if prediction_length is 1
            forecast = self.pipeline.predict(
                context=x_raw,
                prediction_length=self.prediction_length,
                num_samples=self.num_samples,
                limit_prediction_length=False 
            )

        # Statistical Reduction
        # We collapse the probabilistic distribution (Num_Samples) into a single median value.
        # Shape changes: [Batch, Samples, Horizon] -> [Batch, Horizon]
        median_forecast = torch.quantile(forecast, 0.5, dim=1)

        # Select the specific horizon step
        # Since we predicted 1 step ahead, we take index 0.
        # Shape changes: [Batch, 1] -> [Batch]
        point_forecast = median_forecast[:, 0]
        
        # Reshape to match LSTM/Target output [Batch, 1]
        output = point_forecast.unsqueeze(-1)
        
        return output.float() # Ensure we return float32 for compatibility