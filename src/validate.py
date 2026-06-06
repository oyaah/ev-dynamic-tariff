"""Data validation gate. Runs at the start of the pipeline so bad or drifted
input fails loudly with a clear message rather than silently corrupting results.
"""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "timestamp", "source", "location_id", "energy_kwh", "occupancy",
    "capacity", "utilization", "price_per_kwh", "is_dynamic_pricing",
}


class DataValidationError(ValueError):
    pass


def validate_panel(panel: pd.DataFrame) -> dict:
    """Assert the unified panel meets the schema and value contracts the
    pipeline relies on. Returns a summary dict; raises on violation."""
    missing = REQUIRED_COLUMNS - set(panel.columns)
    if missing:
        raise DataValidationError(f"missing columns: {sorted(missing)}")

    checks = {
        "utilization in [0,1]": panel["utilization"].between(0, 1).all(),
        "no null utilization": panel["utilization"].notna().all(),
        "capacity > 0": (panel["capacity"] > 0).all(),
        "urbanev prices > 0": (panel.loc[panel.source == "urbanev", "price_per_kwh"] > 0).all(),
        "both sources present": {"acn", "urbanev"} <= set(panel["source"].unique()),
        "non-empty": len(panel) > 0,
    }
    failed = [name for name, ok in checks.items() if not bool(ok)]
    if failed:
        raise DataValidationError(f"validation failed: {failed}")

    return {"rows": int(len(panel)),
            "rows_by_source": panel.groupby("source").size().to_dict(),
            "checks_passed": len(checks)}
