import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
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

        self.model = Ridge(alpha=0.01, positive=True)
        self.coefficients = None
        self.intercept = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARRV":
        X_features = X[self.feature_cols].copy()
        # Assume caller has already cleaned data
        if X_features.isna().any().any() or y.isna().any():
            raise ValueError("Input contains NaN values.  Clean data before fitting.")
        self.model.fit(X_features, y)

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

    def summary(self, X_train=None, y_train=None) -> str:
        if not self.is_fitted:
            return "Model not fitted"
        lines = [f"{self.name} Model Summary"]
        lines.append(f"Intercept: {self.intercept:.6f}")
        lines.append("\nCoefficients:")

        # Sort by absolute value to show importance
        coef_dict = dict(zip(self.feature_cols, self.coefficients))
        sorted_coefs = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)

        for name, coef in sorted_coefs:
            lines.append(f"  {name}: {coef:.6f}")

        # Add R^2 if training data provided
        if X_train is not None and y_train is not None:
            train_score = self.model.score(X_train[self.feature_cols], y_train)
            lines.append(f"\nTraining R^2: {train_score:.4f}")

        return "\n".join(lines)
