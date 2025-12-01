# Wrapper utilities for Amazon Chronos-2 forecasting

from typing import Iterable
import numpy as np
import torch


def _import_chronos_pipeline():
    """
    Import ChronosPipeline. Some versions expose Chronos2Pipeline, but ChronosPipeline
    works across 1.x/2.x. Raises a helpful error if chronos is missing.
    """
    try:
        from chronos import ChronosPipeline  # type: ignore

        return ChronosPipeline
    except Exception:
        raise ImportError(
            "chronos-forecasting is required. Install with `pip install chronos-forecasting`."
        )


class Chronos2Forecaster:
    """
    Lightweight wrapper around Chronos2Pipeline to generate H-step forecasts.
    """

    def __init__(
        self,
        model_id: str,
        prediction_length: int,
        num_samples: int = 20,
        reduce: str = "mean",
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ):
        self.model_id = model_id
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.reduce = reduce
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.pipeline = None

    def load(self):
        ChronosPipeline = _import_chronos_pipeline()
        # `torch_dtype` is deprecated in newer HF; accept both keys for compatibility.
        dtype_kwargs = {"dtype": self.torch_dtype} if self.torch_dtype != "auto" else {}
        self.pipeline = ChronosPipeline.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            torch_dtype=None if not dtype_kwargs else dtype_kwargs.get("dtype"),
            **dtype_kwargs,
        )
        return self

    def predict(self, context: Iterable[float]) -> np.ndarray:
        """
        context: 1D iterable of log-RV history.
        Returns an array of length prediction_length with aggregated samples.
        """
        if self.pipeline is None:
            self.load()

        ctx = torch.tensor(np.asarray(context, dtype=np.float32)).unsqueeze(0)
        forecast = self.pipeline.predict(
            ctx,
            prediction_length=self.prediction_length,
            num_samples=self.num_samples,
        )

        forecast_arr = forecast.cpu().numpy()
        if forecast_arr.ndim == 1:
            agg = forecast_arr
        else:
            if self.reduce == "median":
                agg = np.median(forecast_arr, axis=0)
            else:
                agg = forecast_arr.mean(axis=0)

        return agg
