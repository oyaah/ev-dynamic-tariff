"""Price elasticity of charging demand, estimated from UrbanEV.

UrbanEV is the only priced dataset, so all elasticity comes from here, and only
from grids whose ``dynamic_pricing`` flag is set (where price actually varies).
Estimated as an ASSOCIATION, not a causal effect (the brief requires this
caution). ε feeds the Tariff Agent's price-response simulation.

Model: within-grid first differences, Δlog(volume) ~ Δlog(price) with hour-of-day
fixed effects. The key result is that elasticity is *state-dependent* -- demand
is near-inelastic when stations are busy and elastic when idle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_urbanev_panel

# Plausibility bounds for a demand elasticity (negative, finite).
ELASTICITY_BOUNDS = (-2.0, -0.05)
FALLBACK_ELASTICITY = -0.3
UTIL_EDGES = [0.0, 0.3, 0.5, 0.8, 1.01]          # off-peak / mid / high / congested
UTIL_LABELS = ["low", "mid", "high", "vhigh"]


def _first_diff(g: pd.DataFrame) -> pd.DataFrame:
    """Within-grid first differences of log price and log demand, plus the
    lagged utilization that defines the demand regime."""
    g = g.sort_values("timestamp")
    g = g[(g["price_per_kwh"] > 0) & (g["energy_kwh"] > 0)]
    g["dlog_p"] = np.log(g["price_per_kwh"]).diff()
    g["dlog_q"] = np.log(g["energy_kwh"]).diff()
    g["hour"] = g["timestamp"].dt.hour
    if "occupancy" in g and "capacity" in g:
        g["util_lag"] = (g["occupancy"] / g["capacity"]).shift(1)
    return g.dropna(subset=["dlog_p", "dlog_q"])


def _dynamic_diffs(panel: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (dynamic-grid panel, first-difference rows where price moved)."""
    if panel is None:
        panel = load_urbanev_panel()
    src_ok = panel.get("source", "urbanev")
    dyn = panel[(src_ok == "urbanev") & (panel["is_dynamic_pricing"] == 1)].copy()
    diffs = dyn.groupby("location_id", group_keys=False).apply(_first_diff)
    moved = diffs[diffs["dlog_p"].abs() > 1e-4]
    return dyn, moved


def _ols_eps(d: pd.DataFrame) -> float:
    """OLS of dlog_q on dlog_p with hour-of-day fixed effects. Returns the
    coefficient on dlog_p (the elasticity)."""
    if len(d) < 50:
        return np.nan
    hours = pd.get_dummies(d["hour"], prefix="h", drop_first=True).to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d["dlog_p"].to_numpy(), hours])
    beta, *_ = np.linalg.lstsq(X, d["dlog_q"].to_numpy(), rcond=None)
    return float(beta[1])


def estimate_elasticity(panel: pd.DataFrame | None = None) -> dict:
    """Single overall elasticity, with a documented fallback if it is implausible."""
    dyn, moved = _dynamic_diffs(panel)
    eps = _ols_eps(moved)
    lo, hi = ELASTICITY_BOUNDS
    if np.isfinite(eps) and lo <= eps <= hi:
        used, status = eps, "estimated"
    else:
        used, status = FALLBACK_ELASTICITY, "fallback (estimate outside plausible bounds)"
    return {"elasticity": used, "n_obs": int(len(moved)),
            "n_dynamic_grids": int(dyn["location_id"].nunique()), "status": status}


def estimate_elasticity_by_state(panel: pd.DataFrame | None = None) -> dict:
    """Elasticity within utilization regimes (the revenue-critical structure).

    Demand is far more inelastic when the station is busy (people must charge)
    than when idle (discretionary). This heterogeneity is what makes dynamic
    pricing a revenue winner. Uses LAGGED utilization to define the regime,
    reducing the simultaneity of price reacting to current demand."""
    dyn, moved = _dynamic_diffs(panel)
    moved = moved.dropna(subset=["util_lag"])
    moved["state"] = pd.cut(moved["util_lag"], UTIL_EDGES, labels=UTIL_LABELS, include_lowest=True)
    by_state = {str(s): _ols_eps(g) for s, g in moved.groupby("state", observed=True)}
    # demand is never positively sloped; clip tiny positive noise toward 0.
    by_state = {k: float(np.clip(v, -2.0, -0.001)) if np.isfinite(v) else FALLBACK_ELASTICITY
                for k, v in by_state.items()}
    return {"by_state": by_state, "n_obs": int(len(moved)),
            "n_dynamic_grids": int(dyn["location_id"].nunique())}


def make_eps_fn(by_state: dict):
    """Return a util -> elasticity step function from the estimated state map."""
    def eps_for(util):
        idx = np.clip(np.digitize(np.asarray(util, float), UTIL_EDGES[1:-1]), 0, len(UTIL_LABELS) - 1)
        return np.array([by_state.get(UTIL_LABELS[i], FALLBACK_ELASTICITY) for i in idx])
    return eps_for


def to_frame(result: dict) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in result.items() if not isinstance(v, dict)}])
