"""End-to-end pipeline: builds the unified base, runs the three agents + the
feedback loop, and writes every CSV deliverable to outputs/.

Run:  python -m src.run_pipeline
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone

import pandas as pd

from .config import MODELS, OUTPUTS, SEED, set_seed
from .data import OUT, build_unified, load_unified
from .validate import validate_panel
from .agents.demand import DemandAgent
from .agents.monitor import run_feedback_loop
from .agents.tariff import TariffAgent
from .elasticity import (estimate_elasticity, estimate_elasticity_by_state,
                         make_eps_fn, to_frame)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline")


def build_pred_table(panel: pd.DataFrame, agent: DemandAgent, source="urbanev") -> pd.DataFrame:
    preds = agent.predictions[agent.predictions.source == source].copy()
    energy = panel[["timestamp", "location_id", "energy_kwh", "is_dynamic_pricing", "is_cbd"]]
    return preds.merge(energy, on=["timestamp", "location_id"], how="left").dropna(subset=["energy_kwh"])


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=OUTPUTS.parent, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    set_seed()
    OUT.mkdir(exist_ok=True)
    panel = load_unified() if (OUT / "unified_panel.csv").exists() else build_unified()[0]

    # --- Data validation gate: fail loudly on bad/drifted input ---
    data_summary = validate_panel(panel)
    log.info("data validated: %s", data_summary)

    # --- Demand Prediction Agent ---
    demand = DemandAgent()
    metrics = demand.fit_eval(panel)
    metrics.to_csv(OUT / "demand_metrics.csv", index=False)
    demand.predictions.to_csv(OUT / "demand_predictions.csv", index=False)
    demand.save(MODELS)  # persist trained models for inference without retraining
    log.info("demand models saved to %s", MODELS)

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
    ep_log, final, tuned = run_feedback_loop(pred, eps, eps_fn=eps_fn)
    ep_log.to_csv(OUT / "monitor_episodes.csv", index=False)

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
        "episodes": len(ep_log),
        "composite_first": float(ep_log["composite"].iloc[0]),
        "composite_last": float(ep_log["composite"].iloc[-1]),
        "surge_sensitivity_final": tuned.surge_sensitivity,
        "discount_sensitivity_final": tuned.discount_sensitivity,
    }
    pd.DataFrame([final_kpis]).to_csv(OUT / "final_kpis.csv", index=False)

    # --- run metadata: lineage + experiment tracking (lightweight, no MLflow) ---
    run_meta = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "seed": SEED,
        "data": data_summary,
        "elasticity_by_state": state["by_state"],
        "headline_kpis": {k: round(float(v), 4) for k, v in final_kpis.items()
                          if isinstance(v, (int, float))},
    }
    (OUT / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

    log.info("PIPELINE COMPLETE | demand R2 (UrbanEV)=%.3f", final_kpis["demand_r2_urbanev"])
    log.info("elasticity by state: %s", {k: round(v, 3) for k, v in state["by_state"].items()})
    log.info("revenue gain %% [neutral / full x%.2f]: %.2f / %.2f",
             kC["avg_price_multiplier"], kN["revenue_gain_pct"], kC["revenue_gain_pct"])
    log.info("off-peak uplift %% [neutral]: %.2f | composite %.2f -> %.2f",
             kN["offpeak_uplift_pct"], final_kpis["composite_first"], final_kpis["composite_last"])
    log.info("outputs -> %s | models -> %s", OUT, MODELS)


if __name__ == "__main__":
    main()
