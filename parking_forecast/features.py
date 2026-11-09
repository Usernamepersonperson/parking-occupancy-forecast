"""Feature table: one row per (origin time, car park, horizon). Target = free fraction at origin + horizon."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import calendar_frame

HORIZONS = (1, 2, 3, 4, 6, 8)  # in 15-minute steps: 15, 30, 45, 60, 90, 120 min
STEP_MIN = 15
WEEK = 672
DAY = 96

CAL_COLS = ["tod_sin", "tod_cos", "dow", "is_holiday", "holiday_eve", "is_saturday", "is_sunday", "month"]
WEATHER_COLS = ["temp_c", "precip_mm", "precip_3h"]
LOT_COLS = ["cur", "lag1", "lag2", "lag4", "lag8", "lag_day", "mean4", "std4", "trend", "snaive_week", "snaive_day"]


def capacity_proxy(free: pd.DataFrame, q: float = 0.995) -> pd.Series:
    """Per-lot capacity proxy: a high quantile of observed free spaces (the feed carries no capacity)."""
    cap = free.quantile(q).clip(lower=1.0)
    return cap.rename("cap")


def full_threshold(cap: pd.Series | np.ndarray | float, frac: float = 0.03, floor: float = 3.0):
    """A car park counts as 'full' at or below max(floor, frac*capacity) free spaces."""
    return np.maximum(floor, frac * np.asarray(cap, dtype=float))


def _take(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Row-take with NaN for out-of-range indices. arr is (T, L); idx is (n,)."""
    out = np.full((len(idx), arr.shape[1]), np.nan)
    ok = (idx >= 0) & (idx < arr.shape[0])
    out[ok] = arr[idx[ok]]
    return out


def prepare(free: pd.DataFrame, weather: pd.DataFrame | None, cap: pd.Series) -> dict:
    """Pre-compute aligned arrays used by build_features. `free` is (T, L) on a regular 15-min UTC grid."""
    frac = (free / cap).to_numpy(dtype=float)
    fr = pd.DataFrame(frac, index=free.index)
    roll = fr.rolling(4, min_periods=3)
    cal = calendar_frame(free.index)
    if weather is None:
        w = pd.DataFrame(np.nan, index=free.index, columns=WEATHER_COLS)
    else:
        w = weather.reindex(free.index).copy()
        w["precip_3h"] = w["precip_mm"].rolling(12, min_periods=4).sum()
        w = w[WEATHER_COLS]
    return {
        "index": free.index,
        "lots": list(free.columns),
        "frac": frac,
        "mean4": roll.mean().to_numpy(),
        "std4": roll.std().to_numpy(),
        "cal": cal[CAL_COLS].to_numpy(dtype=float),
        "weather": w.to_numpy(dtype=float),
        "cap": cap.to_numpy(dtype=float),
    }


def build_features(prep: dict, origins: np.ndarray, h: int, use_weather: bool = True, use_calendar: bool = True) -> pd.DataFrame:
    """Rows for all lots at the given origin indices and horizon h (steps). Columns include y (NaN if unknown)."""
    frac = prep["frac"]
    n_lots = frac.shape[1]
    j = origins + h
    cur = _take(frac, origins)
    cols = {
        "cur": cur,
        "lag1": _take(frac, origins - 1),
        "lag2": _take(frac, origins - 2),
        "lag4": _take(frac, origins - 4),
        "lag8": _take(frac, origins - 8),
        "lag_day": _take(frac, origins - DAY),
        "mean4": _take(prep["mean4"], origins),
        "std4": _take(prep["std4"], origins),
        "snaive_week": _take(frac, j - WEEK),
        "snaive_day": _take(frac, j - DAY),
    }
    cols["trend"] = cur - cols["lag4"]
    n = len(origins)
    data = {k: v.reshape(-1) for k, v in cols.items()}
    data["lot"] = np.tile(np.arange(n_lots), n)
    data["cap"] = np.tile(prep["cap"], n)
    data["h"] = np.full(n * n_lots, h)
    if use_calendar:
        cal = prep["cal"][np.clip(j, 0, len(prep["cal"]) - 1)]
        for k, name in enumerate(CAL_COLS):
            data[name] = np.repeat(cal[:, k], n_lots)
    if use_weather:
        w = prep["weather"][np.clip(origins, 0, len(prep["weather"]) - 1)]
        for k, name in enumerate(WEATHER_COLS):
            data[name] = np.repeat(w[:, k], n_lots)
    y = _take(frac, j).reshape(-1)
    data["y"] = y
    df = pd.DataFrame(data)
    df["origin"] = np.repeat(origins, n_lots)
    return df


def feature_columns(use_weather: bool = True, use_calendar: bool = True) -> list[str]:
    cols = [*LOT_COLS, "lot", "cap", "h"]
    if use_calendar:
        cols += CAL_COLS
    if use_weather:
        cols += WEATHER_COLS
    return cols


def seasonal_naive(df: pd.DataFrame) -> np.ndarray:
    """Seasonal-naive forecast of free fraction: same time one week earlier, else same time one day earlier."""
    return df["snaive_week"].fillna(df["snaive_day"]).to_numpy()


def persistence(df: pd.DataFrame) -> np.ndarray:
    return df["cur"].to_numpy()
