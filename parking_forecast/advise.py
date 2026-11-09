"""Turn forecasts into driver-facing advice: a 'full by' warning and a re-routing suggestion."""

from __future__ import annotations

import numpy as np

from .evaluate import recommend
from .features import HORIZONS, STEP_MIN


def full_by(q50_by_step: dict[int, float], threshold: float) -> int | None:
    """Earliest horizon (minutes) at which the median forecast is at or below the 'full' threshold, else None."""
    for h in sorted(q50_by_step):
        v = q50_by_step[h]
        if not np.isnan(v) and v <= threshold:
            return h * STEP_MIN
    return None


def advisory(grid: dict, k: int, lot: int, thr: np.ndarray, names: list[str], near: dict[str, list[str]], arrive_h: int = 2) -> dict:
    """Advice for one car park at origin row k of the dense prediction grid.

    Returns the median/interval forecast per horizon, the 'full by' time, and, if the car park is forecast to be
    full on arrival, the nearby car park with the most free spaces at its lower (q10) forecast.
    """
    fc = {h * STEP_MIN: [float(grid[h][q][k, lot]) for q in ("q10", "q50", "q90")] for h in HORIZONS}
    fb = full_by({h: grid[h]["q50"][k, lot] for h in HORIZONS}, float(thr[lot]))
    out = {"lot": names[lot], "forecast_q10_q50_q90": fc, "full_by_min": fb, "reroute_to": None}
    if grid[arrive_h]["q50"][k, lot] <= thr[lot]:
        pos = {n: i for i, n in enumerate(names)}
        cands = near.get(names[lot], [])
        out["reroute_to"] = recommend(cands, {c: float(grid[arrive_h]["q10"][k, pos[c]]) for c in cands})
    return out
