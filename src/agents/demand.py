"""Demand Prediction Agent.

Forecasts station utilization (and expected load) per location-hour and exposes
a congestion probability. Trained per source (ACN, UrbanEV) because their scales
differ. Uses a strict temporal split to avoid leakage and reports against a
seasonal-naive baseline so the gain is interpretable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..config import HGB_PARAMS, LGBM_PARAMS, MODELS
from ..features import add_lag_features, add_time_features

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except Exception:  # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingRegressor
    _HAS_LGBM = False

FEATURES = [
    "hour", "dow", "is_weekend", "hour_sin", "hour_cos",
    "lag_1h", "lag_24h", "roll_3h", "roll_24h",
    "capacity", "is_cbd", "is_dynamic_pricing",
]
TARGET = "utilization"
CONGESTION_THRESHOLD = 0.8


def _new_model():
    if _HAS_LGBM:
        return LGBMRegressor(**LGBM_PARAMS)
    return HistGradientBoostingRegressor(**HGB_PARAMS)


def temporal_split(df: pd.DataFrame, train=0.7, val=0.15):
    """Split by time so all test timestamps are strictly after train/val."""
    cuts = df["timestamp"].quantile([train, train + val]).values
    tr = df[df["timestamp"] <= cuts[0]]
    va = df[(df["timestamp"] > cuts[0]) & (df["timestamp"] <= cuts[1])]
    te = df[df["timestamp"] > cuts[1]]
    return tr, va, te


def _metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Predict each (location, hour-of-day) mean utilization from train."""
    means = train.groupby(["location_id", "hour"])[TARGET].mean()
    glob = train[TARGET].mean()
    idx = list(zip(test["location_id"], test["hour"]))
    return np.array([means.get(k, glob) for k in idx])


@dataclass
class DemandAgent:
    """Per-source utilization forecaster."""
    models: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    residual_std: dict = field(default_factory=dict)

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_time_features(df)
        df = add_lag_features(df, TARGET)
        return df.dropna(subset=["lag_24h"])  # need history

    def fit_eval(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Fit one model per source; return tidy metrics + store predictions."""
        rows = []
        self.predictions = []
        for src, g in panel.groupby("source"):
            g = self._prep(g)
            tr, va, te = temporal_split(g)
            if len(te) < 50:
                continue
            assert tr["timestamp"].max() < te["timestamp"].min(), "temporal leak"
            m = _new_model()
            m.fit(pd.concat([tr, va])[FEATURES], pd.concat([tr, va])[TARGET])
            self.models[src] = m
            pred = np.clip(m.predict(te[FEATURES]), 0, 1)
            base = np.clip(seasonal_naive(pd.concat([tr, va]), te), 0, 1)
            self.metrics[src] = {"model": _metrics(te[TARGET], pred),
                                 "baseline": _metrics(te[TARGET], base)}
            self.residual_std[src] = float(np.std(te[TARGET].values - pred))
            for tag, mt in self.metrics[src].items():
                rows.append({"source": src, "estimator": tag, **mt, "n_test": len(te)})
            out = te[["timestamp", "location_id", "source", TARGET]].copy()
            out["utilization_pred"] = pred
            self.predictions.append(out)
        self.predictions = pd.concat(self.predictions, ignore_index=True)
        return pd.DataFrame(rows)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict utilization + expected load + congestion probability."""
        df = self._prep(df)
        out = df[["timestamp", "location_id", "source", "capacity"]].copy()
        util = np.zeros(len(df))
        for src, idx in df.groupby("source").groups.items():
            if src in self.models:
                rows = df.loc[idx]
                util[df.index.get_indexer(idx)] = np.clip(self.models[src].predict(rows[FEATURES]), 0, 1)
        out["utilization_pred"] = util
        out["expected_load"] = util * df["capacity"].values
        # Congestion prob via Gaussian residual model: P(util > threshold).
        from scipy.stats import norm
        sd = df["source"].map(self.residual_std).fillna(0.15).values
        out["congestion_probability"] = 1 - norm.cdf((CONGESTION_THRESHOLD - util) / np.maximum(sd, 1e-3))
        return out

    def save(self, path: Path = MODELS) -> Path:
        """Persist trained models + residual stats so inference needs no retrain."""
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "residual_std": self.residual_std,
                     "metrics": self.metrics}, path / "demand_agent.joblib")
        return path / "demand_agent.joblib"

    @classmethod
    def load(cls, path: Path = MODELS) -> "DemandAgent":
        state = joblib.load(path / "demand_agent.joblib")
        return cls(models=state["models"], metrics=state["metrics"],
                   residual_std=state["residual_std"])
