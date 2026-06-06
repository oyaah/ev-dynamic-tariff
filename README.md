# Agentic Dynamic Tariff Optimization for EV Charging

A self-improving, three-agent pricing engine for EV charging networks. It forecasts demand, prices dynamically using a **state-dependent** price elasticity, and learns the policy from outcomes — built on real charging-session data.

> Society of Business — Open Project 2026 (OP'26 Analytics).

## Key results

| Metric | Result |
|---|---|
| Demand forecast R² (UrbanEV) | **0.95** (seasonal-naive baseline 0.74) |
| Price elasticity | idle **−0.51** → congested **≈0** (state-dependent) |
| Revenue vs ₹15 flat — *equal average price* | **+2.8%** (pure price discrimination) |
| Revenue vs ₹15 flat — *with level correction* | **+20%** (avg price ×1.20, disclosed) |
| Off-peak utilization uplift | **+15%** |
| Self-improvement (composite objective) | 6.5 → 20.2 over learning episodes |

**Core insight:** elasticity is state-dependent — demand is near-inelastic when stations are busy and elastic when idle. A single averaged elasticity makes dynamic pricing *lose* to flat (Jensen's inequality); modelling it by demand state is what unlocks the gain.

## Architecture

Three agents in a feedback loop:

```
DemandAgent ── forecast utilization, load, congestion prob (LightGBM)
     │
TariffAgent ── demand-responsive tariff; surge inelastic peaks, discount elastic off-peak
     │
MonitorAgent ─ score revenue / off-peak uplift / wait / efficiency → tune the policy
     └──────── feedback loop over episodes
```

## Repo layout

```
src/                 engine (single-purpose modules)
  config.py          paths, seed, model hyperparameters
  data.py            load both datasets → unified hourly panel
  validate.py        data-quality gate (schema + range checks)
  features.py        time + lag features, data-driven demand periods
  elasticity.py      state-dependent price elasticity
  agents/            demand.py · tariff.py · monitor.py
  run_pipeline.py    runs everything, writes outputs/ + models/ + run_metadata.json
notebooks/           01_preprocess → 05_monitor_eval (narrative, executed)
tests/               pytest sanity suite (13 tests)
outputs/             result CSVs + figures + run_metadata.json (large/raw files gitignored)
models/              persisted trained models (gitignored, regenerable)
DECK.md              5–7 slide presentation content
docs/plans/          implementation plan
Dockerfile, Makefile, .github/workflows/ci.yml
```

## MLOps

Kept lightweight and fit-for-purpose (no MLflow/DVC/serving overhead this project doesn't need):

- **Reproducibility** — single `SEED` + centralized hyperparameters in `src/config.py`.
- **Data validation gate** — `src/validate.py` checks schema and value ranges before training; bad/drifted input fails loudly.
- **Model persistence** — trained demand models saved to `models/` (`DemandAgent.save()/load()`), so inference needs no retraining.
- **Run lineage / experiment tracking** — every run writes `outputs/run_metadata.json` (UTC timestamp, git SHA, seed, data summary, headline metrics).
- **Logging** — structured `logging` throughout the pipeline.
- **CI** — GitHub Actions runs the test suite on every push.
- **Reproducible env** — `Dockerfile` (`make docker`).

## Quickstart

```bash
pip install -r requirements.txt

# 1. Place the two datasets under "Datasets OP_26 Analytics/" (see Data below)
# 2. Run the full pipeline (builds the unified base + all results)
python -m src.run_pipeline

# 3. Tests
python -m pytest tests/ -q

# 4. Notebooks (narrative walk-through)
jupyter lab notebooks/
```

## Data

Not committed (externally sourced). Download and place under `Datasets OP_26 Analytics/`:

- **ACN-Data** (Caltech/JPL charging sessions) — https://ev.caltech.edu/dataset.html
- **UrbanEV / ST-EVCDP** (Shenzhen grid panel) — https://github.com/IntelligentSystemsLab/ST-EVCDP

`python -m src.run_pipeline` regenerates every artifact from these.

## Method notes & honest limitations

- Revenue is a counterfactual **simulation** (q = q₀·multiplier^ε(state)); elasticity is **associational, not causal** (endogeneity mitigated via lagged-state binning + hour fixed effects, not eliminated).
- The two datasets differ in geography/year/usage, so they share one schema but are sliced by `source` — never numerically merged across networks; all pricing analysis is UrbanEV-sourced.
- Surge does **not** reduce peak occupancy here — inelastic peak demand won't shift on price. Price's grid lever is off-peak attraction, not peak shedding.

## License

MIT — see [LICENSE](LICENSE).
