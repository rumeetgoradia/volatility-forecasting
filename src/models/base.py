from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import pickle
from pathlib import Path


class BaseModel(ABC):
    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.is_fitted = False
        self.feature_names = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    def fit_predict(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        self.fit(X, y)
        return self.predict(X)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "BaseModel":
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_params(self) -> Dict[str, Any]:
        return {}

    def set_params(self, **params) -> "BaseModel":
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
