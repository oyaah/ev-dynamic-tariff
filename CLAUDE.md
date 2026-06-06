# OP'26 Analytics — Agentic Dynamic Tariff for EV Charging

**Deadline: 7 June 2026 (today). Ship the simple, effective version. 20% effort → 80% result. Do only what's needed.**

Full plan: `docs/plans/2026-06-07-001-feat-ev-dynamic-tariff-plan.md`. Brief: `OP'26 Analytics.md`.

## The winning approach (locked — don't over-engineer)

Three Python agent classes in a feedback loop. No RL, no LLM agents, no deep learning, no live API.

1. **DemandAgent** — LightGBM (fallback `HistGradientBoostingRegressor`) forecasts utilization/load + congestion prob. Beat a seasonal-naive baseline. Metrics: RMSE/MAE/R².
2. **TariffAgent** — continuous demand-responsive multiplier (neutral at 0.3 util, ramps to 1.5× surge / 0.7× discount); revenue sim uses **state-dependent elasticity**. `revenue_neutral=True` holds avg price = flat to isolate the discrimination gain. Counterfactual vs ₹15/kWh flat. Metrics: Revenue Gain %, util before/after, off-peak uplift.
3. **MonitorAgent** — scores each decision (revenue, wait-proxy, response rate, pricing efficiency = rev/kWh), feeds back a bounded `surge_sensitivity` update across episodes. Show KPIs trending up = the "learning."

## Data truths (verified)

- **UrbanEV is the gold mine**: 247 Shenzhen grids, 5-min × 30 days, with *real time-varying prices* + `dynamic_pricing` flag (57 dynamic / 190 static). All pricing/elasticity claims come from here.
- **ACN**: 16,304 session logs (Caltech/JPL, 2018), GMT strings → convert to LA tz. No prices. Use for session behavior.
- Unified base = **hourly long-format panel**, `source` column kept separable (merge is schema-level, never numeric cross-network averaging — say so honestly).
- `utilization = occupancy / count`, clip [0,1]. Surge/discount thresholds key off this.

## Key result (honest)

- Elasticity is **state-dependent**: idle ε≈−0.51 (elastic), busy ε≈0 (inelastic). A single avg ε makes dynamic pricing *lose* to flat (Jensen) — this was the bug. Surge inelastic peaks, discount elastic off-peak.
- Revenue: **+2.8% at equal avg price** (pure discrimination, bulletproof) → **+20%** if operator also corrects the underpriced flat level (avg price ×1.2, disclose it). Off-peak uplift +15% at neutral.
- Honest limit: surge does NOT shed peak occupancy (inelastic demand won't move). Grid lever = off-peak attraction, not peak shaving.

## Non-negotiables

- **No causal claims** — elasticity is associational. State every assumption on-slide.
- **No time leakage** — strict temporal train/val/test split; lags from past only.
- **Reproducible** — notebooks 01→05 run top-to-bottom; all results to `outputs/*.csv`; `requirements.txt`.
- Revenue is *simulated* (demand × (newprice/baseprice)^ε); show ε-sensitivity in appendix.

## Build order

`01_preprocess` → (`02_eda` ∥ `03_demand_model`) → `04_tariff_agent` (incl. elasticity) → `05_monitor_eval` → `DECK.md` (5–7 slides, last).

## User style

Python + type hints, EAFP, module-level helpers, no docstrings on unchanged code. No READMEs unless asked.
