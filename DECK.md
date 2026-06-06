# Agentic AI Dynamic Tariff Optimization for EV Charging Networks
### OP'26 Analytics — Society of Business · Open Project 2026

> Slide-by-slide content (5–7 core slides, excluding cover / executive summary / appendix).
> Every number is reproducible: `python -m src.run_pipeline` (or notebooks `01→05`), outputs in `outputs/`.

---

## Executive Summary (not counted)

A self-improving, three-agent pricing engine for EV charging. It forecasts demand (**R² 0.95**), discovers that **price elasticity is state-dependent** — demand is *inelastic when busy, elastic when idle* — and uses that to price dynamically. **At the same average tariff as the ₹15 flat rate it earns +2.8% more revenue and lifts off-peak charging +15%, purely by aligning price with demand**; correcting the underpriced flat level takes revenue to +20%. A monitoring agent learns the policy from outcomes (composite objective 6.5 → 20.2 across episodes).

---

## Slide 1 — Data Landscape & Preprocessing

- **Two official datasets → one unified hourly dataframe** (`outputs/unified_panel.csv`, 279k rows), tagged by `source`, sliced per analysis:
  - **ACN** (Caltech/JPL): 16,304 sessions → 14,999 clean. GMT→LA timezone; session logs → hourly station occupancy + energy.
  - **UrbanEV** (Shenzhen): 247 grids × 720 h. Wide matrices → long, 5-min → hourly. **Carries real time-varying prices + a `dynamic_pricing` flag** (57 dynamic / 190 static) — the basis of every pricing result.
- **Economic features:** utilization, revenue/session, energy cost/kWh, queue & overload proxies, occupancy density, calendar + lag/rolling features.
- **Transparent missing-value handling**, logged at each step; prices never imputed.
- *ML-flow note:* EDA findings fed back into preprocessing — peak windows and elasticity regimes are **data-driven**, not assumed. Trees are scale-invariant → no normalization step needed.

---

## Slide 2 — EDA: How Demand Actually Behaves

*(figures: `01_intraday_utilization`, `03_dynamic_vs_static`, `04_price_vs_demand`, `05_volatility_by_period`)*

- **Counter-intuitive finding #1:** charging **occupancy peaks overnight** (hours 0–1, 6), *not* at commute hours (peak-state util 0.33 vs daytime 0.24). → peak windows must come from charging data; the agent prices on *realized* utilization.
- **Network is mostly under-utilized** (mean util 28%). The dominant opportunity is filling **idle off-peak capacity**, not just taming peaks.
- **Dynamic-pricing grids** show higher price dispersion (std 0.22 vs 0.16) — a real, usable pricing signal.
- **ACN workplace signature**: long connection times, low idle (median 0.74 h), daytime arrival spike — different usage from urban public charging → source-specific treatment.

---

## Slide 3 — Demand Prediction Agent

*(table: `outputs/demand_metrics.csv`)*

- Per-source gradient-boosted model (LightGBM), **strict temporal split** (no leakage), benchmarked vs a seasonal-naive baseline.

| Source | Model R² | Baseline R² | Model RMSE |
|---|---|---|---|
| **UrbanEV** | **0.95** | 0.74 | 0.040 |
| ACN | 0.28 | 0.10 | 0.219 |

- UrbanEV forecasts are excellent; ACN is harder (workplace stations near-saturated → low variance) but still **3× the baseline R²**.
- Outputs per location-hour: **predicted utilization, expected load, congestion probability** — the Tariff Agent's inputs.

---

## Slide 4 — The Elasticity Insight (why dynamic pricing wins)

*(file: `outputs/elasticity.csv`; figure `04_price_vs_demand`)*

- Elasticity estimated from UrbanEV dynamic grids (Δlog-demand on Δlog-price, hour fixed effects, lagged-utilization regimes; **association, not causal**).
- **Price elasticity is state-dependent** — the decisive result:

| Utilization state | ε | Behaviour |
|---|---|---|
| Off-peak (idle) | **−0.51** | elastic → discounts grow volume + fill capacity |
| Mid | −0.13 | |
| Busy / congested | **≈ 0.00** | inelastic → surge captures revenue at ~zero volume loss |

- **Why a single elasticity fails:** with one averaged ε and prices varying around the mean, Jensen's inequality makes dynamic pricing *lose* to flat. Modelling elasticity *by demand state* is what unlocks the gain — surge the inelastic, discount the elastic.

---

## Slide 5 — Tariff Pricing Agent: Results (honest decomposition)

*(file: `outputs/revenue_comparison.csv`)*

Continuous demand-responsive tariff (surcharge rises with utilization, discount below 30%), bounded 0.7×–1.5×. Revenue is a scale-free counterfactual simulation vs the ₹15/kWh flat baseline (q = q₀·m^ε(state)).

| Operating point | Avg price | Revenue vs flat | Off-peak uplift |
|---|---|---|---|
| Flat baseline | ×1.00 | 0% | — |
| **Revenue-neutral (core)** | **×1.00** | **+2.8%** | **+15.3%** |
| Full learned (+ level correction) | ×1.20 | +20.5% | +6.1% |

- **Core, bulletproof claim:** at the **same average tariff**, dynamic pricing earns **+2.8%** more *and* lifts off-peak charging **+15%** — pure price discrimination, not a price hike.
- **Upper bound:** because peak demand is inelastic, the flat ₹15 is underpriced; letting the operator also correct the *level* yields **+20%** (disclosed: average price ×1.20 — a policy choice, not a free lunch).
- **Honest limitation:** surge does **not** reduce peak *occupancy* — inelastic peak demand won't shift on price (peak util 0.87 → 0.87). Price's grid lever here is **off-peak attraction**, not peak shedding.

---

## Slide 6 — Monitoring & Learning Agent + Implications

*(file: `outputs/monitor_episodes.csv`, `outputs/final_kpis.csv`)*

- Monitor scores every decision (revenue, off-peak uplift, wait-proxy, pricing efficiency) and tunes the Tariff Agent's surge/discount sensitivity via transparent coordinate ascent.
- **Self-improvement:** composite objective **6.5 → 20.2** across 5 episodes; the loop learns to surge harder (sensitivity 0.5 → 2.0) as it confirms peak demand is inelastic.
- **Business/operational/policy:**
  - Dynamic pricing is a **revenue + access** tool here, not a peak-congestion tool: +2.8% revenue at equal price, +15% off-peak utilization (filling the 72% idle capacity).
  - **The real lever is surge, priced to inelasticity** — discounts are for grid balancing, not revenue.
  - **Policy:** set time-of-day tariffs from *charging* demand curves (overnight peaks here), and price *scarcity by demand state*, not by clock — a transferable, data-first principle.

---

## Appendix (not counted)

- **Assumptions & limitations:** revenue is a counterfactual *simulation* (constant-elasticity within state); elasticity is associational (endogeneity mitigated via lagged-state binning + hour FE, not eliminated); ACN/UrbanEV differ in geography/year/usage → unified panel is sliced by source, never numerically merged across networks; 30 days of UrbanEV → 5 learning episodes.
- **Ablation (`revenue_comparison.csv`):** a single averaged elasticity gives a misleading +7.4% with *spurious* peak shaving — because it wrongly assumes peak demand is elastic. The state-dependent model corrects this.
- **Robustness:** elasticity by 4 utilization states with hour FE; revenue decomposed into discrimination (+2.8%) vs level (+17.7%); demand model per-source; data-driven (not assumed) peak windows.
- **Future work:** spatial GNN demand model (`adj.csv`/`distance.csv`); cross-time substitution model to capture peak→off-peak shifting; live pricing API; RL policy; causal elasticity via instruments.
- **Reproducibility:** `pip install -r requirements.txt`, then `python -m src.run_pipeline` or notebooks `01→05`. Tests: `python -m pytest tests/ -q` (11 passing).
