"""Gradient-boosting quantile forecaster with split-conformal (CQR) interval calibration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import feature_columns

QUANTILES = (0.1, 0.5, 0.9)


@dataclass
class QuantileForecaster:
    use_weather: bool = True
    use_calendar: bool = True
    max_iter: int = 200
    learning_rate: float = 0.08
    max_leaf_nodes: int = 31
    random_state: int = 0
    quantiles: tuple = QUANTILES
    models: dict = field(default_factory=dict)
    cqr_shift: float = 0.0  # fraction of capacity added to each side of the [q10, q90] interval

    @property
    def columns(self) -> list[str]:
        return feature_columns(self.use_weather, self.use_calendar)

    @staticmethod
    def base(df: pd.DataFrame) -> np.ndarray:
        """Anchor the forecast on the latest reading (else seasonal values); the model learns the change from it."""
        b = df["cur"].fillna(df["snaive_week"]).fillna(df["snaive_day"]).fillna(0.5)
        return b.to_numpy()

    def fit(self, df: pd.DataFrame) -> QuantileForecaster:
        d = df.dropna(subset=["y"])
        X = d[self.columns]
        cat = [c == "lot" for c in self.columns]
        target = d["y"].to_numpy() - self.base(d)
        for q in self.quantiles:
            m = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.max_leaf_nodes,
                categorical_features=cat,
                random_state=self.random_state,
            )
            m.fit(X, target)
            self.models[q] = m
        return self

    def predict_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Uncalibrated quantile forecasts as free fraction of capacity, with monotone ordering enforced."""
        X = df[self.columns]
        base = self.base(df)
        p = np.column_stack([self.models[q].predict(X) + base for q in self.quantiles])
        p = np.sort(p, axis=1)
        names = {0.1: "q10", 0.5: "q50", 0.9: "q90"}
        return pd.DataFrame(p, columns=[names[q] for q in sorted(self.quantiles)], index=df.index)

    def calibrate(self, df: pd.DataFrame, alpha: float = 0.2) -> float:
        """Conformalised quantile regression: widen/narrow [q10, q90] so held-out coverage is 1 - alpha."""
        d = df.dropna(subset=["y"])
        p = self.predict_raw(d)
        self.cqr_shift = cqr_shift(d["y"].to_numpy(), p["q10"].to_numpy(), p["q90"].to_numpy(), alpha)
        return self.cqr_shift

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.predict_raw(df)
        out = p.copy()
        out["q10"] = np.clip(p["q10"] - self.cqr_shift, 0.0, None)
        out["q90"] = p["q90"] + self.cqr_shift
        out["q50"] = np.clip(p["q50"], 0.0, None)
        out["q10"] = np.minimum(out["q10"], out["q50"])
        out["q90"] = np.maximum(out["q90"], out["q50"])
        return out


def cqr_shift(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.2) -> float:
    """Split-conformal correction (Romano et al. 2019): finite-sample quantile of max(lo - y, y - hi)."""
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level))
