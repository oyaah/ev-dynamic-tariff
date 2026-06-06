"""Insight-driven EDA: every figure is labelled and tied to a pricing
implication. Saves PNGs to outputs/figures/ and a tidy outputs/eda_summary.csv.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import OUT, load_unified
from .features import add_time_features

FIG = OUT / "figures"


def _save(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


def run_eda() -> pd.DataFrame:
    panel = load_unified()
    # Period labels are source-specific: ACN (workplace) peaks daytime, UrbanEV
    # (urban Shenzhen) peaks overnight. A shared map would mislabel one of them.
    urban = add_time_features(panel[panel.source == "urbanev"])
    acn = add_time_features(panel[panel.source == "acn"])
    panel = add_time_features(panel)
    summary: list[dict] = []

    # 1. Intraday utilization by source -> identifies peak windows for surge
    fig, ax = plt.subplots(figsize=(7, 4))
    for src, g in panel.groupby("source"):
        h = g.groupby("hour")["utilization"].mean()
        ax.plot(h.index, h.values, marker="o", label=src)
    ax.set(title="Intraday utilization by hour (peaks => surge windows)",
           xlabel="hour of day", ylabel="mean utilization")
    ax.legend(); ax.grid(alpha=.3)
    _save(fig, "01_intraday_utilization.png")

    # 2. Weekday vs weekend (UrbanEV)
    fig, ax = plt.subplots(figsize=(7, 4))
    for lab, sub in [("weekday", urban[urban.is_weekend == 0]), ("weekend", urban[urban.is_weekend == 1])]:
        h = sub.groupby("hour")["utilization"].mean()
        ax.plot(h.index, h.values, marker="o", label=lab)
    ax.set(title="UrbanEV: weekday vs weekend demand shape",
           xlabel="hour", ylabel="mean utilization")
    ax.legend(); ax.grid(alpha=.3)
    _save(fig, "02_weekday_weekend.png")

    # 3. Dynamic vs static grids (UrbanEV) -> motivates dynamic pricing
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for flag, lab in [(0, "static"), (1, "dynamic")]:
        sub = urban[urban.is_dynamic_pricing == flag]
        h = sub.groupby("hour")["utilization"].mean()
        axs[0].plot(h.index, h.values, marker="o", label=lab)
        axs[1].plot(h.index, sub.groupby("hour")["price_per_kwh"].std().values, marker="s", label=lab)
    axs[0].set(title="Utilization: dynamic vs static grids", xlabel="hour", ylabel="mean util")
    axs[1].set(title="Intra-hour price dispersion (std)", xlabel="hour", ylabel="price std")
    axs[0].legend(); axs[1].legend(); axs[0].grid(alpha=.3); axs[1].grid(alpha=.3)
    _save(fig, "03_dynamic_vs_static.png")

    # 4. Price vs demand (binned) -> elasticity precursor
    u = urban[(urban.price_per_kwh > 0) & (urban.energy_kwh > 0)].copy()
    u["pbin"] = pd.qcut(u["price_per_kwh"], 10, duplicates="drop")
    b = u.groupby("pbin", observed=True).agg(price=("price_per_kwh", "mean"),
                                             demand=("energy_kwh", "mean"))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(b["price"], b["demand"], marker="o")
    ax.set(title="Observed price vs mean demand (downward => price-responsive)",
           xlabel="price (¥/kWh)", ylabel="mean energy/hour (kWh)")
    ax.grid(alpha=.3)
    _save(fig, "04_price_vs_demand.png")

    # 5. Volatility by period
    pv = urban.groupby("period")["utilization"].agg(["mean", "std"])
    pv["cv"] = pv["std"] / pv["mean"]
    fig, ax = plt.subplots(figsize=(6, 4))
    pv["cv"].reindex(["offpeak", "shoulder", "peak"]).plot.bar(ax=ax, color="#4477aa")
    ax.set(title="Demand volatility (CV) by period", ylabel="coef. of variation")
    _save(fig, "05_volatility_by_period.png")

    # 6. ACN session behavior
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    sess = pd.read_csv(OUT / "sessions_clean.csv")
    axs[0].hist(sess["connection_hr"].clip(0, 24), bins=40, color="#66ccee")
    axs[0].set(title="ACN session connection hours", xlabel="hours", ylabel="sessions")
    arr = pd.to_datetime(sess["connectionTime"], utc=True).dt.tz_convert("America/Los_Angeles").dt.hour
    axs[1].hist(arr, bins=24, color="#ee6677")
    axs[1].set(title="ACN arrivals by hour (workplace signature)", xlabel="hour", ylabel="sessions")
    _save(fig, "06_acn_sessions.png")

    # --- summary table ---
    peak_h = urban.groupby("hour")["utilization"].mean().nlargest(3).index.tolist()
    summary.append({"metric": "urbanev_peak_hours", "value": str(peak_h)})
    summary.append({"metric": "urbanev_mean_util", "value": round(urban.utilization.mean(), 3)})
    summary.append({"metric": "acn_mean_util", "value": round(acn.utilization.mean(), 3)})
    for p in ["peak", "shoulder", "offpeak"]:
        summary.append({"metric": f"urbanev_util_{p}", "value": round(urban[urban.period == p].utilization.mean(), 3)})
        summary.append({"metric": f"urbanev_cv_{p}", "value": round(pv.loc[p, "cv"], 3)})
    dyn = urban[urban.is_dynamic_pricing == 1]; sta = urban[urban.is_dynamic_pricing == 0]
    summary.append({"metric": "dynamic_grid_price_std", "value": round(dyn.price_per_kwh.std(), 3)})
    summary.append({"metric": "static_grid_price_std", "value": round(sta.price_per_kwh.std(), 3)})
    summary.append({"metric": "cbd_mean_util", "value": round(urban[urban.is_cbd == 1].utilization.mean(), 3)})
    summary.append({"metric": "noncbd_mean_util", "value": round(urban[urban.is_cbd == 0].utilization.mean(), 3)})
    summary.append({"metric": "acn_median_idle_hr", "value": round(sess.idle_hr.median(), 2)})

    out = pd.DataFrame(summary)
    out.to_csv(OUT / "eda_summary.csv", index=False)
    return out


if __name__ == "__main__":
    s = run_eda()
    print(s.to_string(index=False))
    print("figures ->", FIG)
