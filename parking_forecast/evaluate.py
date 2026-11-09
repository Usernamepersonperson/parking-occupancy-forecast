"""Metrics: MAE in spaces, interval coverage, 'full' event precision/recall with lead time, guidance hit rate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import STEP_MIN


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def full_onsets(free: np.ndarray, thr: float, quiet_steps: int = 8) -> np.ndarray:
    """Indices where a car park becomes full after at least `quiet_steps` consecutive not-full readings."""
    free = np.asarray(free, dtype=float)
    out = []
    for i in range(quiet_steps, len(free)):
        if np.isnan(free[i]) or free[i] > thr:
            continue
        window = free[i - quiet_steps : i]
        if np.all(~np.isnan(window)) and np.all(window > thr):
            out.append(i)
    return np.array(out, dtype=int)


def warning_metrics(warn: np.ndarray, free: np.ndarray, thr: float, horizon: int, onsets: np.ndarray | None = None) -> dict:
    """Score a boolean per-step 'full within `horizon` steps' warning for one car park.

    precision: share of warnings (issued while not full) followed by a full reading within `horizon` steps.
    recall: share of full onsets preceded by at least one warning in the `horizon` steps before them.
    lead_min: list of minutes between the earliest such warning and the onset.
    """
    free = np.asarray(free, dtype=float)
    warn = np.asarray(warn, dtype=bool)
    issued = tp = 0
    for t in np.flatnonzero(warn):
        if np.isnan(free[t]) or free[t] <= thr:
            continue
        issued += 1
        fut = free[t + 1 : t + 1 + horizon]
        if len(fut) and np.any(fut <= thr):
            tp += 1
    if onsets is None:
        onsets = full_onsets(free, thr)
    hit = 0
    leads = []
    for tau in onsets:
        lo = max(0, tau - horizon)
        idx = np.flatnonzero(warn[lo:tau])
        if len(idx):
            hit += 1
            leads.append((tau - (lo + idx[0])) * STEP_MIN)
    return {"issued": issued, "tp": tp, "events": int(len(onsets)), "events_hit": hit, "lead_min": leads}


def combine_warning(parts: list[dict]) -> dict:
    issued = sum(p["issued"] for p in parts)
    tp = sum(p["tp"] for p in parts)
    events = sum(p["events"] for p in parts)
    hit = sum(p["events_hit"] for p in parts)
    leads = [x for p in parts for x in p["lead_min"]]
    return {
        "warnings": issued,
        "precision": tp / issued if issued else float("nan"),
        "events": events,
        "recall": hit / events if events else float("nan"),
        "lead_median_min": float(np.median(leads)) if leads else float("nan"),
        "lead_mean_min": float(np.mean(leads)) if leads else float("nan"),
    }


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lng2 - lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def nearest_lots(coords: pd.DataFrame, k: int = 3) -> dict[str, list[str]]:
    """For each lot with coordinates, its k nearest other lots with coordinates (by great-circle distance)."""
    c = coords.dropna(subset=["lat", "lng"])
    out = {}
    for lid, r in c.iterrows():
        d = [(haversine_km(r.lat, r.lng, o.lat, o.lng), oid) for oid, o in c.iterrows() if oid != lid]
        out[lid] = [oid for _, oid in sorted(d)[:k]]
    return out


def recommend(candidates: list[str], q10_free: dict[str, float]) -> str | None:
    """Pick the candidate with the most free spaces at the lower (q10) forecast; ties go to the first listed."""
    best, best_v = None, -np.inf
    for c in candidates:
        v = q10_free.get(c, np.nan)
        if not np.isnan(v) and v > best_v:
            best, best_v = c, v
    return best
