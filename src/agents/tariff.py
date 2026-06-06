"""Tariff Pricing Agent.

Prices continuously with demand: a surcharge grows as forecast utilization rises
above the network reference (materially so past the 80% congestion line), and a
discount applies below 30% utilization. Magnitude is calibrated by the
state-dependent price elasticity.

Why this beats a flat tariff (and why a single-elasticity model could not):
demand is near-inelastic when stations are busy (ε≈0) and elastic when idle
(ε≈-0.45). Surging inelastic peaks captures scarcity revenue at almost no volume
loss; modest off-peak discounts lift utilization. Revenue is a scale-free
counterfactual SIMULATION vs a flat ₹15/kWh baseline:
    revenue_i / flat_i = multiplier_i ** (1 + ε(state_i))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

REF_UTIL = 0.30          # surge above the network reference; discount below it
SURGE_UTIL = 0.80        # brief's congestion line (reporting reference)
DISCOUNT_UTIL = 0.30
SURGE_CAP = 1.5
DISCOUNT_FLOOR = 0.7


def _const_eps(eps: float) -> Callable:
    return lambda util: np.full(np.shape(util), eps, dtype=float)


@dataclass
class TariffAgent:
    surge_sensitivity: float = 1.0          # tuned by MonitorAgent
    discount_sensitivity: float = 1.0       # tuned by MonitorAgent
    elasticity: float = -0.3                # fallback if no eps_fn given
    eps_fn: Callable | None = None          # util -> elasticity (state-dependent)
    revenue_neutral: bool = False           # hold avg price = flat (isolate shape gain)

    def _eps(self, util) -> np.ndarray:
        fn = self.eps_fn or _const_eps(self.elasticity)
        return fn(util)

    def decide(self, util: np.ndarray | pd.Series) -> np.ndarray:
        """Continuous demand-responsive multiplier on the flat baseline.
        Neutral (1.0) at REF_UTIL; ramps to surge as util->1, to discount as
        util->0. Sensitivities scale each arm; bounded to [0.7, 1.5]."""
        util = np.asarray(util, dtype=float)
        m = np.ones_like(util)
        up = util > REF_UTIL
        dn = util < REF_UTIL
        m[up] = 1 + self.surge_sensitivity * (util[up] - REF_UTIL) / (1 - REF_UTIL) * (SURGE_CAP - 1)
        m[dn] = 1 - self.discount_sensitivity * (REF_UTIL - util[dn]) / REF_UTIL * (1 - DISCOUNT_FLOOR)
        return np.clip(m, DISCOUNT_FLOOR, SURGE_CAP)

    def simulate(self, df: pd.DataFrame, util_col="utilization_pred",
                 base_demand="energy_kwh", actual_util="utilization") -> tuple[pd.DataFrame, dict]:
        """Apply tariffs to a forecast table and compute pricing KPIs. The price
        decision uses the forecast; the demand response uses the slot's actual
        utilization state to pick ε (the true regime)."""
        d = df.copy()
        eps = self._eps(d[actual_util].to_numpy())
        d["eps_slot"] = eps
        m = self.decide(d[util_col])
        if self.revenue_neutral:
            # rescale so the energy-weighted average price equals the flat
            # baseline -> the resulting gain is pure price discrimination,
            # not a blanket price increase.
            w = d[base_demand].to_numpy()
            wmean = np.average(m, weights=w) if w.sum() > 0 else m.mean()
            m = np.clip(m / wmean, DISCOUNT_FLOOR, SURGE_CAP)
        d["multiplier"] = m
        m = d["multiplier"].to_numpy()
        d["demand_factor"] = m ** eps                      # >1 for discounts, <=1 for surge
        q_base = d[base_demand].to_numpy()
        d["demand_new"] = q_base * d["demand_factor"]

        # Scale-free revenue ratio with per-slot elasticity.
        rev_flat = q_base.sum()
        rev_dyn = (q_base * m ** (1 + eps)).sum()
        revenue_gain_pct = 100 * (rev_dyn - rev_flat) / rev_flat if rev_flat else 0.0

        d["util_new"] = np.clip(d[actual_util] * d["demand_factor"], 0, 1)
        peak = d[d[actual_util] >= SURGE_UTIL]
        off = d[d[actual_util] <= DISCOUNT_UTIL]
        offpeak_uplift = 100 * (off["demand_new"].sum() - off[base_demand].sum()) / max(off[base_demand].sum(), 1e-9)

        kpis = {
            "revenue_gain_pct": float(revenue_gain_pct),
            "peak_util_before": float(peak[actual_util].mean()) if len(peak) else np.nan,
            "peak_util_after": float(peak["util_new"].mean()) if len(peak) else np.nan,
            "offpeak_util_before": float(off[actual_util].mean()) if len(off) else np.nan,
            "offpeak_util_after": float(off["util_new"].mean()) if len(off) else np.nan,
            "offpeak_uplift_pct": float(offpeak_uplift),
            "pct_surged": float((m > 1).mean() * 100),
            "pct_discounted": float((m < 1).mean() * 100),
            "avg_price_multiplier": float(np.average(m, weights=q_base)) if q_base.sum() > 0 else float(m.mean()),
            "surge_sensitivity": self.surge_sensitivity,
            "discount_sensitivity": self.discount_sensitivity,
        }
        return d, kpis
