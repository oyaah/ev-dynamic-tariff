"""Data loading and unified-base construction for the EV tariff project.

Two source datasets are stacked into ONE long-format dataframe (the unified
panel) sharing a common hourly schema, tagged by ``source``. They are used
differently downstream (prices/elasticity come only from UrbanEV) but live in a
single dataset, sliced by ``source``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Datasets OP_26 Analytics"
ACN_XLSX = DATA / "ACN Data_ 25 April 2018 to 16 Dec 2018" / "acndata_sessions.json.xlsx"
URBAN = DATA / "UrbanEV_ SZ_districts"
OUT = ROOT / "outputs"

# Records of every missing-value / cleaning decision, surfaced in notebooks.
missing_value_log: dict[str, str] = {}


# --- ACN -------------------------------------------------------------------
def load_acn_sessions() -> pd.DataFrame:
    """Load + clean ACN session logs. Times parsed as GMT then converted to
    Los Angeles local (column present) so hour-of-day is meaningful."""
    raw = pd.read_excel(ACN_XLSX, engine="openpyxl")
    df = raw[[
        "stationID", "spaceID", "siteID", "userID", "kWhDelivered",
        "connectionTime", "disconnectTime", "doneChargingTime",
    ]].copy()

    for c in ("connectionTime", "disconnectTime", "doneChargingTime"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce").dt.tz_convert("America/Los_Angeles")

    df["kWhDelivered"] = pd.to_numeric(df["kWhDelivered"], errors="coerce")

    n0 = len(df)
    df = df.dropna(subset=["connectionTime", "disconnectTime"])
    df = df[df["disconnectTime"] > df["connectionTime"]]
    df = df[df["kWhDelivered"].fillna(0) >= 0]
    # doneChargingTime sometimes missing -> assume charging filled the session.
    df["doneChargingTime"] = df["doneChargingTime"].fillna(df["disconnectTime"])
    df["kWhDelivered"] = df["kWhDelivered"].fillna(0.0)
    missing_value_log["acn_dropped_rows"] = f"{n0 - len(df)} of {n0} sessions dropped (bad/zero-length/neg energy)"
    missing_value_log["acn_doneCharging_filled"] = "missing doneChargingTime -> disconnectTime"

    df["connection_hr"] = (df["disconnectTime"] - df["connectionTime"]).dt.total_seconds() / 3600.0
    df["charging_hr"] = (df["doneChargingTime"] - df["connectionTime"]).dt.total_seconds().clip(lower=0) / 3600.0
    df["idle_hr"] = (df["connection_hr"] - df["charging_hr"]).clip(lower=0)
    df["region"] = "us"
    df["site"] = df["siteID"].astype(str)
    return df.reset_index(drop=True)


def acn_hourly_panel(sessions: pd.DataFrame) -> pd.DataFrame:
    """Expand sessions into an hourly station panel.

    location = stationID, capacity = 1 connector. utilization = fraction of the
    hour the station was occupied (connection window). energy attributed to the
    charging window, split proportionally across overlapping hours.
    """
    occ_rows: list[tuple] = []
    eng_rows: list[tuple] = []
    for r in sessions.itertuples(index=False):
        start, end = r.connectionTime, r.disconnectTime
        hours = pd.date_range(start.floor("h"), end.floor("h"), freq="h")
        for h in hours:
            he = h + pd.Timedelta(hours=1)
            overlap = (min(end, he) - max(start, h)).total_seconds() / 3600.0
            if overlap > 0:
                occ_rows.append((r.stationID, h, overlap))
        # energy across charging window
        ce = r.doneChargingTime
        chg_hours = pd.date_range(start.floor("h"), ce.floor("h"), freq="h")
        total = max((ce - start).total_seconds(), 1.0)
        for h in chg_hours:
            he = h + pd.Timedelta(hours=1)
            ov = (min(ce, he) - max(start, h)).total_seconds()
            if ov > 0:
                eng_rows.append((r.stationID, h, r.kWhDelivered * ov / total))

    occ = pd.DataFrame(occ_rows, columns=["location_id", "timestamp", "occ"])
    eng = pd.DataFrame(eng_rows, columns=["location_id", "timestamp", "energy_kwh"])
    # Drop tz so ACN (LA-local) and UrbanEV (naive local) share one clock type.
    for frame in (occ, eng):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.tz_localize(None)
    occ = occ.groupby(["location_id", "timestamp"], as_index=False)["occ"].sum()
    occ["occ"] = occ["occ"].clip(upper=1.0)  # one connector
    eng = eng.groupby(["location_id", "timestamp"], as_index=False)["energy_kwh"].sum()

    panel = occ.merge(eng, on=["location_id", "timestamp"], how="left")
    panel["energy_kwh"] = panel["energy_kwh"].fillna(0.0)
    panel["capacity"] = 1.0
    panel["occupancy"] = panel["occ"]
    panel["utilization"] = panel["occ"].clip(0, 1)
    panel["price_per_kwh"] = np.nan
    panel["is_cbd"] = 0
    panel["is_dynamic_pricing"] = 0
    panel["source"] = "acn"
    panel["region"] = "us"
    return panel.drop(columns=["occ"])


# --- UrbanEV ---------------------------------------------------------------
def _urban_long(name: str, value: str) -> pd.DataFrame:
    df = pd.read_csv(URBAN / f"{name}.csv")
    idcol = df.columns[0]
    long = df.melt(id_vars=idcol, var_name="location_id", value_name=value)
    long = long.rename(columns={idcol: "row"})
    long["row"] = long["row"].astype(int)
    return long


def load_urbanev_panel() -> pd.DataFrame:
    """Reshape UrbanEV wide matrices -> long, attach real timestamps + grid
    metadata, aggregate 5-min -> hourly."""
    time = pd.read_csv(URBAN / "time.csv")
    time.columns = [c.strip().lstrip("﻿") for c in time.columns]
    time["timestamp"] = pd.to_datetime(time[["year", "month", "day", "hour", "minute", "second"]])
    time["row"] = np.arange(1, len(time) + 1)
    t = time[["row", "timestamp"]]

    occ = _urban_long("occupancy", "occupancy")
    vol = _urban_long("volume", "volume")
    price = _urban_long("price", "price")
    df = occ.merge(vol, on=["row", "location_id"]).merge(price, on=["row", "location_id"])
    df = df.merge(t, on="row", how="left")

    info = pd.read_csv(URBAN / "information.csv")
    info["location_id"] = info["grid"].astype(str)
    meta = info[["location_id", "count", "CBD", "dynamic_pricing"]].rename(
        columns={"count": "capacity", "CBD": "is_cbd", "dynamic_pricing": "is_dynamic_pricing"})
    df = df.merge(meta, on="location_id", how="left")

    df["hour_ts"] = df["timestamp"].dt.floor("h")
    agg = df.groupby(["location_id", "hour_ts"]).agg(
        occupancy=("occupancy", "mean"),
        energy_kwh=("volume", "sum"),
        price_per_kwh=("price", "mean"),
        capacity=("capacity", "first"),
        is_cbd=("is_cbd", "first"),
        is_dynamic_pricing=("is_dynamic_pricing", "first"),
    ).reset_index().rename(columns={"hour_ts": "timestamp"})

    agg["utilization"] = (agg["occupancy"] / agg["capacity"]).clip(0, 1)
    n_price_na = agg["price_per_kwh"].isna().sum()
    missing_value_log["urbanev_price_na"] = f"{n_price_na} hourly price cells NaN (left as-is, flagged)"
    agg["source"] = "urbanev"
    agg["region"] = "shenzhen"
    return agg


# --- unified ----------------------------------------------------------------
COLS = [
    "timestamp", "source", "region", "location_id", "energy_kwh", "occupancy",
    "capacity", "utilization", "price_per_kwh", "is_cbd", "is_dynamic_pricing",
]


def build_unified(write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = load_acn_sessions()
    acn = acn_hourly_panel(sessions)
    urban = load_urbanev_panel()

    panel = pd.concat([acn[COLS], urban[COLS]], ignore_index=True)
    panel = panel.sort_values(["source", "location_id", "timestamp"]).reset_index(drop=True)
    panel["queue_proxy"] = (panel["occupancy"] - panel["capacity"]).clip(lower=0)
    panel["overload"] = (panel["utilization"] - 0.8).clip(lower=0)

    if write:
        OUT.mkdir(exist_ok=True)
        panel.to_csv(OUT / "unified_panel.csv", index=False)
        sessions.to_csv(OUT / "sessions_clean.csv", index=False)
    return panel, sessions


def load_unified() -> pd.DataFrame:
    p = OUT / "unified_panel.csv"
    df = pd.read_csv(p, parse_dates=["timestamp"], dtype={"location_id": str})
    return df


if __name__ == "__main__":
    panel, sessions = build_unified()
    print("unified panel:", panel.shape)
    print(panel.groupby("source").size())
    print("missing-value log:")
    for k, v in missing_value_log.items():
        print(" ", k, "->", v)
