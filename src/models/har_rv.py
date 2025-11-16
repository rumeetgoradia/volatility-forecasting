import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Optional, List
import sys

sys.path.append("src")
from models.base import BaseModel


class HARRV(BaseModel):
    def __init__(self, name: str = "HAR-RV", feature_cols: Optional[List[str]] = None):
        super().__init__(name)

        if feature_cols is None:
            self.feature_cols = ["RV_daily", "RV_weekly", "RV_monthly"]
        else:
            self.feature_cols = feature_cols

        self.model = LinearRegression()
        self.coefficients = None
        self.intercept = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARRV":
        X_features = X[self.feature_cols].copy()

        mask = X_features.notna().all(axis=1) & y.notna()
        X_clean = X_features[mask]
        y_clean = y[mask]

        if len(X_clean) == 0:
            raise ValueError("No valid samples after removing NaN values")

        self.model.fit(X_clean, y_clean)

        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        self.feature_names = self.feature_cols
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_features = X[self.feature_cols].copy()
        predictions = self.model.predict(X_features)

        return predictions

    def get_params(self) -> dict:
        if not self.is_fitted:
            return {"fitted": False}

        params = {
            "intercept": self.intercept,
            "coefficients": dict(zip(self.feature_cols, self.coefficients)),
            "fitted": True,
        }
        return params

    def summary(self) -> str:
        if not self.is_fitted:
            return "Model not fitted"

        lines = [f"{self.name} Model Summary"]
        lines.append(f"Intercept: {self.intercept:.6f}")
        lines.append("Coefficients:")
        for name, coef in zip(self.feature_cols, self.coefficients):
            lines.append(f"  {name}: {coef:.6f}")

        return "\n".join(lines)
