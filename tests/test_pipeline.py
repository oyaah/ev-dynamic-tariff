"""Lightweight sanity tests for the EV tariff pipeline.

Run: python -m pytest tests/ -q   (from project root)
"""
import numpy as np
import pandas as pd
import pytest

from src.agents.tariff import TariffAgent
from src.agents.monitor import wait_reduction_pct, pricing_efficiency
from src.elasticity import _first_diff, ELASTICITY_BOUNDS
from src.validate import validate_panel, DataValidationError


# --- TariffAgent.decide -----------------------------------------------------
def test_surge_discount_neutral():
    ta = TariffAgent(surge_sensitivity=1.0, discount_sensitivity=1.0)
    assert ta.decide([0.9])[0] > 1.0      # surge at high utilization
    assert ta.decide([0.1])[0] < 1.0      # discount at low utilization
    assert ta.decide([0.30])[0] == pytest.approx(1.0)  # neutral at reference util


def test_multiplier_bounds():
    ta = TariffAgent(surge_sensitivity=10, discount_sensitivity=10)
    assert ta.decide([1.0])[0] <= 1.5     # surge cap
    assert ta.decide([0.0])[0] >= 0.7     # discount floor


# --- revenue simulation -----------------------------------------------------
def _toy(util, energy):
    return pd.DataFrame({"utilization_pred": util, "utilization": util, "energy_kwh": energy})


def test_revenue_gain_handcheck():
    # one surged slot, eps=0 -> demand unchanged, revenue = price change only
    ta = TariffAgent(surge_sensitivity=1.0, discount_sensitivity=1.0, elasticity=0.0)
    _, k = ta.simulate(_toy([0.9], [100.0]))
    m = ta.decide([0.9])[0]
    assert k["revenue_gain_pct"] == pytest.approx((m - 1) * 100, rel=1e-6)


def test_inelastic_discount_loses_revenue():
    # |eps|<1 and a discount -> revenue per slot falls (the key economic finding)
    ta = TariffAgent(surge_sensitivity=0.0, discount_sensitivity=1.0, elasticity=-0.3)
    _, k = ta.simulate(_toy([0.1, 0.1], [100.0, 100.0]))
    assert k["revenue_gain_pct"] < 0


def test_offpeak_uplift_nonneg_on_discount():
    ta = TariffAgent(surge_sensitivity=0.0, discount_sensitivity=1.0, elasticity=-0.5)
    _, k = ta.simulate(_toy([0.1], [100.0]))
    assert k["offpeak_uplift_pct"] >= 0


def test_revenue_neutral_holds_avg_price():
    # energy-weighted average multiplier is ~1.0 under revenue-neutral mode
    ta = TariffAgent(surge_sensitivity=1.5, discount_sensitivity=1.5,
                     elasticity=-0.3, revenue_neutral=True)
    _, k = ta.simulate(_toy([0.95, 0.5, 0.1], [200.0, 100.0, 50.0]))
    assert k["avg_price_multiplier"] == pytest.approx(1.0, abs=0.05)


def test_discrimination_gain_positive_when_peak_inelastic():
    # surge inelastic peak (eps~0) + discount elastic off-peak at equal avg price
    eps_fn = lambda u: __import__("numpy").where(__import__("numpy").asarray(u) > 0.5, -0.02, -0.6)
    ta = TariffAgent(surge_sensitivity=1.5, discount_sensitivity=1.5,
                     eps_fn=eps_fn, revenue_neutral=True)
    _, k = ta.simulate(_toy([0.95, 0.9, 0.1, 0.1], [200.0, 200.0, 50.0, 50.0]))
    assert k["revenue_gain_pct"] > 0


# --- monitor KPIs -----------------------------------------------------------
def test_wait_reduction_positive_when_smoothed():
    d = pd.DataFrame({"utilization": [0.95], "util_new": [0.85]})
    assert wait_reduction_pct(d) > 0


def test_pricing_efficiency_flat_is_one():
    d = pd.DataFrame({"energy_kwh": [100.0], "multiplier": [1.0]})
    assert pricing_efficiency(d, eps=-0.3) == pytest.approx(1.0)


# --- elasticity -------------------------------------------------------------
def test_elasticity_bounds_sane():
    lo, hi = ELASTICITY_BOUNDS
    assert lo < hi < 0


def test_first_diff_drops_nonpositive():
    g = pd.DataFrame({
        "timestamp": pd.date_range("2022-06-19", periods=4, freq="h"),
        "price_per_kwh": [1.0, 0.0, 1.1, 1.2],
        "energy_kwh": [50, 60, 70, 80],
    })
    out = _first_diff(g)
    assert (out["price_per_kwh"] > 0).all()


# --- data validation gate ---------------------------------------------------
def _good_panel():
    return pd.DataFrame({
        "timestamp": pd.date_range("2022-06-19", periods=2, freq="h").tolist() * 1,
        "source": ["acn", "urbanev"],
        "location_id": ["a", "1"],
        "energy_kwh": [1.0, 5.0], "occupancy": [1.0, 3.0], "capacity": [1.0, 10.0],
        "utilization": [0.5, 0.3], "price_per_kwh": [float("nan"), 1.0],
        "is_dynamic_pricing": [0, 1],
    })


def test_validate_accepts_good_panel():
    summary = validate_panel(_good_panel())
    assert summary["rows"] == 2 and summary["checks_passed"] >= 6


def test_validate_rejects_out_of_range_utilization():
    bad = _good_panel()
    bad.loc[0, "utilization"] = 1.5
    with pytest.raises(DataValidationError):
        validate_panel(bad)
