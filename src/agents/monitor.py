"""Monitoring & Learning Agent + feedback loop.

Scores each pricing decision against operational outcomes and feeds the result
back to the Tariff Agent, tuning its surge/discount sensitivity over episodes so
the system demonstrably improves. The update is a transparent coordinate-ascent
hill-climb on a composite objective -- inspectable, not a black box.

Composite = revenue_gain% + W_OFFPEAK * offpeak_uplift% + W_WAIT * wait_reduction%
(weights make explicit that we trade some revenue for grid balancing + shorter
waits, exactly the multi-objective the brief asks for).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .tariff import TariffAgent

W_OFFPEAK = 0.3
W_WAIT = 0.4


def wait_reduction_pct(decisions: pd.DataFrame, actual_util="utilization") -> float:
    """Reduction in peak overload (util above 80%) = queue/wait proxy."""
    before = (decisions[actual_util] - 0.8).clip(lower=0).sum()
    after = (decisions["util_new"] - 0.8).clip(lower=0).sum()
    return float(100 * (before - after) / before) if before > 0 else 0.0


def pricing_efficiency(decisions: pd.DataFrame, eps: float = -0.3, base_demand="energy_kwh") -> float:
    """Revenue per kWh delivered under dynamic pricing, relative to flat (=1.0).
    Scale-free in the baseline price. Uses per-slot elasticity when available."""
    q = decisions[base_demand].to_numpy()
    m = decisions["multiplier"].to_numpy()
    e = decisions["eps_slot"].to_numpy() if "eps_slot" in decisions else eps
    rev = (q * m ** (1 + e)).sum()
    kwh = (q * m ** e).sum()
    rev_per_kwh = rev / kwh if kwh else np.nan
    return float(rev_per_kwh)  # flat baseline = 1.0; >1 means more revenue per kWh


@dataclass
class MonitorAgent:
    elasticity: float = -0.3

    def evaluate(self, decisions: pd.DataFrame, kpis: dict) -> dict:
        wait = wait_reduction_pct(decisions)
        eff = pricing_efficiency(decisions, self.elasticity)
        response = float(decisions["demand_new"].sum() / decisions["energy_kwh"].sum() - 1) * 100
        composite = kpis["revenue_gain_pct"] + W_OFFPEAK * kpis["offpeak_uplift_pct"] + W_WAIT * wait
        return {
            "revenue_gain_pct": kpis["revenue_gain_pct"],
            "offpeak_uplift_pct": kpis["offpeak_uplift_pct"],
            "wait_reduction_pct": wait,
            "customer_response_pct": response,
            "pricing_efficiency": eff,
            "composite": composite,
            "surge_sensitivity": kpis["surge_sensitivity"],
            "discount_sensitivity": kpis["discount_sensitivity"],
        }


# candidate moves for coordinate ascent (surge up helps revenue; discount down
# protects revenue under inelastic demand; some discount still buys grid balance)
_CANDIDATES = [
    (0.0, 0.0), (+0.3, 0.0), (-0.3, 0.0), (0.0, +0.2), (0.0, -0.2),
    (+0.3, -0.2), (+0.3, +0.2), (-0.3, +0.2),
]
SURGE_BOUNDS = (0.2, 3.0)
DISCOUNT_BOUNDS = (0.0, 1.5)


def _composite_for(agent: TariffAgent, ep: pd.DataFrame, monitor: MonitorAgent) -> tuple[float, dict, pd.DataFrame]:
    dec, kpis = agent.simulate(ep)
    rec = monitor.evaluate(dec, kpis)
    return rec["composite"], rec, dec


def run_feedback_loop(pred_table: pd.DataFrame, elasticity: float, eps_fn=None,
                      surge0=0.5, discount0=1.2) -> tuple[pd.DataFrame, dict, TariffAgent]:
    """Run the Demand->Tariff->Monitor loop episode-by-episode (one per day),
    learning surge/discount sensitivity from outcomes.

    Starts from a deliberately naive over-discounting policy. Each episode the
    agent learns on that day's data (coordinate ascent), then the Monitor scores
    the *current* agent on a fixed evaluation horizon -- giving a clean learning
    curve. Returns the per-episode KPI log, final KPIs, and the tuned agent.
    """
    monitor = MonitorAgent(elasticity=elasticity)
    agent = TariffAgent(surge_sensitivity=surge0, discount_sensitivity=discount0,
                        elasticity=elasticity, eps_fn=eps_fn)

    pred_table = pred_table.copy()
    pred_table["day"] = pred_table["timestamp"].dt.date
    days = [ep for _, ep in pred_table.groupby("day") if len(ep) >= 50]

    episodes = []
    for i, ep in enumerate(days):
        # 1) score the current agent on a fixed horizon (the whole table) -> learning curve
        _, rec, _ = _composite_for(agent, pred_table, monitor)
        rec["episode_idx"] = i
        episodes.append(rec)
        # 2) learn from this episode's outcomes: keep the best candidate nudge
        comp_ep, _, _ = _composite_for(agent, ep, monitor)
        best = (comp_ep, agent.surge_sensitivity, agent.discount_sensitivity)
        for ds, dd in _CANDIDATES:
            cand = TariffAgent(
                surge_sensitivity=float(np.clip(agent.surge_sensitivity + ds, *SURGE_BOUNDS)),
                discount_sensitivity=float(np.clip(agent.discount_sensitivity + dd, *DISCOUNT_BOUNDS)),
                elasticity=elasticity, eps_fn=eps_fn)
            c, _, _ = _composite_for(cand, ep, monitor)
            if c > best[0]:
                best = (c, cand.surge_sensitivity, cand.discount_sensitivity)
        agent.surge_sensitivity, agent.discount_sensitivity = best[1], best[2]

    log = pd.DataFrame(episodes)
    # final KPIs = the tuned agent over the whole horizon
    dec, kpis = agent.simulate(pred_table)
    final = monitor.evaluate(dec, kpis)
    final["surge_sensitivity"] = agent.surge_sensitivity
    final["discount_sensitivity"] = agent.discount_sensitivity
    return log, final, agent
