import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from typing import Tuple, Optional, List
import pickle
from pathlib import Path


class RegimeDetector:
    def __init__(self, method: str = "hmm", n_regimes: int = 3):
        self.method = method
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = StandardScaler()
        self.regime_names = {0: "Low", 1: "Medium", 2: "High"}
        self.is_fitted = False

    def fit(self, features: np.ndarray, **kwargs) -> "RegimeDetector":
        features_scaled = self.scaler.fit_transform(features)

        if self.method == "hmm":
            self.model = self._fit_hmm(features_scaled, **kwargs)
        elif self.method == "kmeans":
            self.model = self._fit_kmeans(features_scaled, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.is_fitted = True
        return self

    def _fit_hmm(self, features: np.ndarray, **kwargs) -> hmm.GaussianHMM:
        n_components = kwargs.get("n_components", self.n_regimes)
        covariance_type = kwargs.get("covariance_type", "full")
        n_iter = kwargs.get("n_iter", 100)
        random_state = kwargs.get("random_state", 42)

        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )

        model.fit(features)
        return model

    def _fit_kmeans(self, features: np.ndarray, **kwargs) -> KMeans:
        n_clusters = kwargs.get("n_clusters", self.n_regimes)
        n_init = kwargs.get("n_init", 10)
        random_state = kwargs.get("random_state", 42)

        model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
        model.fit(features)
        return model

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        features_scaled = self.scaler.transform(features)

        if self.method == "hmm":
            regimes = self.model.predict(features_scaled)
        elif self.method == "kmeans":
            regimes = self.model.predict(features_scaled)

        regimes = self._reorder_regimes(features, regimes)
        return regimes

    def _reorder_regimes(self, features: np.ndarray, regimes: np.ndarray) -> np.ndarray:
        regime_means = []
        for i in range(self.n_regimes):
            mask = regimes == i
            if mask.sum() > 0:
                regime_means.append(features[mask, 0].mean())
            else:
                regime_means.append(0)

        sorted_indices = np.argsort(regime_means)
        mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}

        reordered = np.array([mapping[r] for r in regimes])
        return reordered

    def fit_predict(self, features: np.ndarray, **kwargs) -> np.ndarray:
        self.fit(features, **kwargs)
        return self.predict(features)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "RegimeDetector":
        with open(path, 'rb') as f:
            return pickle.load(f)


def detect_regimes_per_instrument(
    df: pd.DataFrame,
    instruments: List[str],
    feature_cols: List[str],
    method: str = "hmm",
    n_regimes: int = 3,
    **kwargs,
) -> Tuple[pd.DataFrame, dict]:

    all_regimes = []
    detectors = {}

    for instrument in instruments:
        print(f"Fitting regime detector for {instrument}")
        inst_df = df[df["Future"] == instrument].copy()
        inst_df = inst_df.sort_values("datetime").reset_index(drop=True)

        features = inst_df[feature_cols].values
        mask = ~np.isnan(features).any(axis=1)

        detector = RegimeDetector(method=method, n_regimes=n_regimes)
        regimes_clean = detector.fit_predict(features[mask], **kwargs)

        regimes = np.full(len(inst_df), -1)
        regimes[mask] = regimes_clean

        inst_df["regime"] = regimes
        all_regimes.append(inst_df[["datetime", "Future", "regime"]])

        detectors[instrument] = detector

    combined = pd.concat(all_regimes, ignore_index=True)
    return combined, detectors


def apply_regime_detectors(
    df: pd.DataFrame,
    detectors: dict,
    feature_cols: List[str],
) -> pd.DataFrame:

    all_regimes = []

    for instrument, detector in detectors.items():
        inst_df = df[df["Future"] == instrument].copy()
        inst_df = inst_df.sort_values("datetime").reset_index(drop=True)

        features = inst_df[feature_cols].values
        mask = ~np.isnan(features).any(axis=1)

        regimes_clean = detector.predict(features[mask])

        regimes = np.full(len(inst_df), -1)
        regimes[mask] = regimes_clean

        inst_df["regime"] = regimes
        all_regimes.append(inst_df[["datetime", "Future", "regime"]])

    combined = pd.concat(all_regimes, ignore_index=True)
    return combined
