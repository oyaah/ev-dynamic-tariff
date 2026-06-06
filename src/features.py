"""Feature engineering shared across EDA, demand model, and agents."""
from __future__ import annotations

import numpy as np
import pandas as pd

def demand_period_map(df: pd.DataFrame, target: str = "utilization") -> dict[int, str]:
    """Data-driven peak/shoulder/off-peak labels.

    EDA finding: charging occupancy does NOT peak at commute hours — in Shenzhen
    it peaks overnight. So we label hours by the data itself: the 8 highest-demand
    hours are 'peak', the 8 lowest are 'offpeak', the rest 'shoulder'. This feeds
    EDA grouping and the elasticity peak/off-peak split, instead of a hardcoded
    commute-hour assumption."""
    by_hour = df.groupby(df["timestamp"].dt.hour)[target].mean().sort_values()
    offpeak = set(by_hour.index[:8])
    peak = set(by_hour.index[-8:])
    return {h: ("peak" if h in peak else "offpeak" if h in offpeak else "shoulder")
            for h in range(24)}


def add_time_features(df: pd.DataFrame, period_map: dict[int, str] | None = None) -> pd.DataFrame:
    df = df.copy()
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    if period_map is None:
        period_map = demand_period_map(df)
    df["period"] = df["hour"].map(period_map)
    return df


def add_lag_features(df: pd.DataFrame, target: str = "utilization") -> pd.DataFrame:
    """Per-location lag/rolling features. Assumes df sorted by location+time.
    Uses only past values (shifted) to avoid leakage."""
    df = df.sort_values(["location_id", "timestamp"]).copy()
    g = df.groupby("location_id")[target]
    df["lag_1h"] = g.shift(1)
    df["lag_24h"] = g.shift(24)
    df["roll_3h"] = g.shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["roll_24h"] = g.shift(1).rolling(24, min_periods=1).mean().reset_index(level=0, drop=True)
    return df
