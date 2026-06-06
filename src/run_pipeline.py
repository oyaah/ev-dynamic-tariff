"""End-to-end pipeline: builds the unified base, runs the three agents + the
feedback loop, and writes every CSV deliverable to outputs/.

Run:  python -m src.run_pipeline
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import OUT, build_unified, load_unified
from .agents.demand import DemandAgent
from .agents.monitor import run_feedback_loop
from .agents.tariff import TariffAgent
from .elasticity import (estimate_elasticity, estimate_elasticity_by_state,
                         make_eps_fn, to_frame)


def build_pred_table(panel: pd.DataFrame, agent: DemandAgent, source="urbanev") -> pd.DataFrame:
    preds = agent.predictions[agent.predictions.source == source].copy()
    energy = panel[["timestamp", "location_id", "energy_kwh", "is_dynamic_pricing", "is_cbd"]]
    return preds.merge(energy, on=["timestamp", "location_id"], how="left").dropna(subset=["energy_kwh"])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    panel = load_unified() if (OUT / "unified_panel.csv").exists() else build_unified()[0]

    # --- Demand Prediction Agent ---
    demand = DemandAgent()
    metrics = demand.fit_eval(panel)
    metrics.to_csv(OUT / "demand_metrics.csv", index=False)
    demand.predictions.to_csv(OUT / "demand_predictions.csv", index=False)

    # --- Elasticity (UrbanEV): overall + state-dependent (revenue-critical) ---
    elas = estimate_elasticity()
    state = estimate_elasticity_by_state()
    elas["by_state"] = state["by_state"]
    to_frame({**{k: v for k, v in elas.items() if k != "by_state"},
              **{f"eps_{k}": v for k, v in state["by_state"].items()}}
             ).to_csv(OUT / "elasticity.csv", index=False)
    eps = elas["elasticity"]
    eps_fn = make_eps_fn(state["by_state"])

    # --- Tariff Pricing Agent: pricing on forecast demand ---
    pred = build_pred_table(panel, demand, "urbanev")

    kcols = ["revenue_gain_pct", "offpeak_uplift_pct", "avg_price_multiplier",
             "peak_util_before", "peak_util_after"]
    rows = []
    rows.append({"scenario": "flat_baseline_Rs15", "revenue_gain_pct": 0.0,
                 "avg_price_multiplier": 1.0, "note": "fixed Rs15/kWh reference"})
    # ablation: single (averaged) elasticity ignores that peak demand is inelastic
    _, kB = TariffAgent(surge_sensitivity=1.0, discount_sensitivity=1.0, elasticity=eps).simulate(pred)
    rows.append({"scenario": "ablation_single_elasticity", **{k: kB[k] for k in kcols},
                 "note": "single avg elasticity (no state structure)"})

    # --- Monitoring & Learning Agent: feedback loop tunes the agent ---
    log, final, tuned = run_feedback_loop(pred, eps, eps_fn=eps_fn)
    log.to_csv(OUT / "monitor_episodes.csv", index=False)

    # CORE result: revenue-neutral mean price -> pure price-discrimination gain
    neutral = TariffAgent(surge_sensitivity=tuned.surge_sensitivity,
                          discount_sensitivity=tuned.discount_sensitivity,
                          elasticity=eps, eps_fn=eps_fn, revenue_neutral=True)
    decN, kN = neutral.simulate(pred)
    rows.append({"scenario": "dynamic_revenue_neutral", **{k: kN[k] for k in kcols},
                 "note": "same avg price as flat; gain = pure discrimination"})

    # UPPER BOUND: unconstrained learned policy (also corrects the underpriced level)
    dec, kC = tuned.simulate(pred)
    dec_out = dec[["timestamp", "location_id", "utilization_pred", "utilization",
                   "eps_slot", "multiplier", "demand_factor", "demand_new", "util_new"]]
    dec_out.to_csv(OUT / "tariff_decisions.csv", index=False)
    rows.append({"scenario": "dynamic_full_learned", **{k: kC[k] for k in kcols},
                 "note": "also corrects underpriced flat level (avg price up)"})
    pd.DataFrame(rows).to_csv(OUT / "revenue_comparison.csv", index=False)

    # --- consolidated headline KPIs for the deck ---
    final_kpis = {
        "elasticity_overall": eps,
        **{f"eps_{k}": v for k, v in state["by_state"].items()},
        "demand_r2_urbanev": float(metrics.query("source=='urbanev' and estimator=='model'")["r2"].iloc[0]),
        "demand_rmse_urbanev": float(metrics.query("source=='urbanev' and estimator=='model'")["rmse"].iloc[0]),
        "revenue_gain_neutral_pct": kN["revenue_gain_pct"],     # core: equal avg price
        "revenue_gain_full_pct": kC["revenue_gain_pct"],        # upper bound: + level correction
        "avg_price_multiplier_full": kC["avg_price_multiplier"],
        "offpeak_uplift_pct": kN["offpeak_uplift_pct"],
        "wait_reduction_pct": final["wait_reduction_pct"],       # brief: avg waiting-time reduction
        "customer_response_pct": final["customer_response_pct"], # brief: customer response rate
        "pricing_efficiency": final["pricing_efficiency"],       # brief: pricing efficiency score
        "episodes": len(log),
        "composite_first": float(log["composite"].iloc[0]),
        "composite_last": float(log["composite"].iloc[-1]),
        "surge_sensitivity_final": tuned.surge_sensitivity,
        "discount_sensitivity_final": tuned.discount_sensitivity,
    }
    pd.DataFrame([final_kpis]).to_csv(OUT / "final_kpis.csv", index=False)

    print("=== PIPELINE COMPLETE ===")
    print("Demand R2 (UrbanEV):", round(final_kpis["demand_r2_urbanev"], 3))
    print("Elasticity by state:", {k: round(v, 3) for k, v in state["by_state"].items()})
    print("Revenue gain % [neutral, equal avg price]:", round(kN["revenue_gain_pct"], 2))
    print("Revenue gain % [full learned, avg price x", round(kC["avg_price_multiplier"], 2), "]:",
          round(kC["revenue_gain_pct"], 2))
    print("Off-peak uplift % [neutral]:", round(kN["offpeak_uplift_pct"], 2))
    print("Composite first -> last:",
          round(final_kpis["composite_first"], 2), "->", round(final_kpis["composite_last"], 2))
    print("Outputs in:", OUT)


if __name__ == "__main__":
    main()
