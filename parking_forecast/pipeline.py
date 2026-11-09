"""End-to-end experiment: chronological split, baselines, quantile GBM, full-by warning, guidance."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import read_dump, weather_to_grid
from .evaluate import combine_warning, coverage, full_onsets, mae, nearest_lots, recommend, warning_metrics
from .features import HORIZONS, STEP_MIN, build_features, capacity_proxy, full_threshold, persistence, prepare, seasonal_naive
from .models import QuantileForecaster

REPORT_H = (1, 4, 8)  # 15, 60, 120 min
ALPHA = 0.2  # 80 % nominal interval
QUIET = 8  # a full onset needs 2 h of not-full readings before it


@dataclass
class Splits:
    start: pd.Timestamp
    train_end: pd.Timestamp
    calib_end: pd.Timestamp
    end: pd.Timestamp

    @property
    def test_start(self) -> pd.Timestamp:
        return self.calib_end + pd.Timedelta(minutes=STEP_MIN)


def default_splits(index: pd.DatetimeIndex) -> Splits:
    """Chronological: first ~67 % train, next ~12 % calibration, the remainder (~21 %) test."""
    n = len(index)
    return Splits(index[0], index[int(n * 0.667)], index[int(n * 0.79)], index[-1])


def _utc(idx: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def load_committed(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    free = pd.read_csv(data_dir / "parking_2022_15min.csv.gz", index_col=0)
    free.index = _utc(pd.to_datetime(free.index, utc=True))
    w = pd.read_csv(data_dir / "weather_2022.csv", index_col=0)
    w.index = _utc(pd.to_datetime(w.index, utc=True))
    lots = pd.read_csv(data_dir / "lots.csv", index_col="lot_id")
    return free, w, lots


def build_from_cache(cache_dir: Path, year: int = 2022) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the wide 15-minute table and hourly weather from files fetched into cache/."""
    cols = {}
    for p in sorted((cache_dir / "archive").glob(f"lot_*-{year}.csv")):
        cols[p.name.split("-")[0]] = read_dump(p)
    free = pd.DataFrame(cols)
    free = free.reindex(pd.date_range(f"{year}-01-01", f"{year}-12-31 23:45", freq=f"{STEP_MIN}min", tz="UTC"))
    w = pd.read_csv(cache_dir / f"weather_{year}.csv", index_col=0)
    w.index = _utc(pd.to_datetime(w.index, utc=True))
    return free, w


def origin_indices(index: pd.DatetimeIndex, lo: pd.Timestamp, hi: pd.Timestamp, h: int, stride: int = 1) -> np.ndarray:
    """Origins i with lo <= t_i and t_{i+h} <= hi."""
    i0 = int(index.searchsorted(lo))
    i1 = int(index.searchsorted(hi, side="right")) - 1 - h
    return np.arange(i0, i1 + 1, stride)


def make_table(prep: dict, origins_by_h: dict[int, np.ndarray], **kw) -> pd.DataFrame:
    return pd.concat([build_features(prep, o, h, **kw) for h, o in origins_by_h.items()], ignore_index=True)


def _by_h_mae(y, p, hh):
    return {str(h * STEP_MIN): mae(y[hh == h], p[hh == h]) for h in REPORT_H}


def run_experiment(
    free: pd.DataFrame,
    weather: pd.DataFrame | None,
    lots: pd.DataFrame | None = None,
    splits: Splits | None = None,
    max_iter: int = 200,
    train_stride: int = 2,
    ablations: bool = True,
    verbose: bool = False,
) -> tuple[dict, dict]:
    """Returns (metrics, artifacts). The capacity proxy is fitted on the training period only."""
    index = free.index
    sp = splits or default_splits(index)
    wgrid = weather_to_grid(weather, index) if weather is not None else None
    cap = capacity_proxy(free.loc[: sp.train_end])
    prep = prepare(free, wgrid, cap)

    def tables(lo, hi, stride=1):
        return make_table(prep, {h: origin_indices(index, lo, hi, h, stride) for h in HORIZONS})

    train = tables(sp.start, sp.train_end, train_stride)
    calib = tables(sp.train_end + pd.Timedelta(minutes=STEP_MIN), sp.calib_end)
    test = tables(sp.test_start, sp.end)
    if verbose:
        print(f"rows train={len(train):,} calib={len(calib):,} test={len(test):,}")

    t0 = time.time()
    model = QuantileForecaster(max_iter=max_iter).fit(train)
    if verbose:
        print(f"fit {time.time() - t0:.0f}s", flush=True)
    shift = model.calibrate(calib, ALPHA)

    sn_all = seasonal_naive(test)
    ok = ~(np.isnan(test["y"].to_numpy()) | np.isnan(sn_all) | np.isnan(test["cur"].to_numpy()))
    t = test[ok].reset_index(drop=True)
    raw = model.predict_raw(t)
    pr = model.predict(t)
    cap_t = t["cap"].to_numpy()
    hh = t["h"].to_numpy()
    y_sp = t["y"].to_numpy() * cap_t
    sn_sp = seasonal_naive(t) * cap_t
    pers_sp = persistence(t) * cap_t
    gb_sp = pr["q50"].to_numpy() * cap_t
    lo_raw, hi_raw = raw["q10"].to_numpy() * cap_t, raw["q90"].to_numpy() * cap_t
    lo_c, hi_c = pr["q10"].to_numpy() * cap_t, pr["q90"].to_numpy() * cap_t

    by_h = {}
    for h in REPORT_H:
        m = hh == h
        v = {
            "n": int(m.sum()),
            "mae_seasonal_naive": mae(y_sp[m], sn_sp[m]),
            "mae_persistence": mae(y_sp[m], pers_sp[m]),
            "mae_gbm": mae(y_sp[m], gb_sp[m]),
            "coverage_raw": coverage(y_sp[m], lo_raw[m], hi_raw[m]),
            "coverage_calibrated": coverage(y_sp[m], lo_c[m], hi_c[m]),
            "mean_width_spaces": float(np.mean(hi_c[m] - lo_c[m])),
        }
        v["gbm_vs_naive_pct"] = 100.0 * (1 - v["mae_gbm"] / v["mae_seasonal_naive"])
        by_h[str(h * STEP_MIN)] = v
    metrics = {
        "split": {k: str(getattr(sp, k)) for k in ("start", "train_end", "calib_end", "end", "test_start")},
        "interval": {"nominal": 1 - ALPHA, "cqr_shift_frac_of_capacity": shift},
        "by_horizon_min": by_h,
        "overall": {
            "n": int(len(t)),
            "mae_seasonal_naive": mae(y_sp, sn_sp),
            "mae_persistence": mae(y_sp, pers_sp),
            "mae_gbm": mae(y_sp, gb_sp),
            "coverage_calibrated": coverage(y_sp, lo_c, hi_c),
        },
    }

    if ablations:
        abl = {"full_features": _by_h_mae(y_sp, raw["q50"].to_numpy() * cap_t, hh)}
        for name, kw in {"no_weather": {"use_weather": False}, "no_calendar": {"use_calendar": False}}.items():
            tr = make_table(prep, {h: origin_indices(index, sp.start, sp.train_end, h, train_stride) for h in HORIZONS}, **kw)
            mm = QuantileForecaster(max_iter=max_iter, quantiles=(0.5,), **kw).fit(tr)
            abl[name] = _by_h_mae(y_sp, mm.predict_raw(t)["q50"].to_numpy() * cap_t, hh)
        metrics["ablation_mae_q50_uncalibrated"] = abl

    # ---- dense test-grid predictions (every car park, every origin) for the warning and guidance scores ----
    common = origin_indices(index, sp.test_start, sp.end, max(HORIZONS))
    n_lots = free.shape[1]
    capv = cap.to_numpy()
    grid = {}
    for h in HORIZONS:
        tb = build_features(prep, common, h)
        p = model.predict(tb)
        grid[h] = {q: p[q].to_numpy().reshape(len(common), n_lots) * capv for q in ("q10", "q50", "q90")}
        grid[h]["snaive"] = seasonal_naive(tb).reshape(len(common), n_lots) * capv
    free_arr = free.to_numpy(dtype=float)
    thr = full_threshold(capv)
    metrics["full_definition"] = "free <= max(3, 3% of capacity proxy)"
    horizon = max(HORIZONS)
    metrics["full_warning_window_min"] = horizon * STEP_MIN

    parts = {"gbm": [], "seasonal_naive": []}
    for li in range(n_lots):
        w_g = np.zeros(len(common), dtype=bool)
        w_s = np.zeros(len(common), dtype=bool)
        for h in HORIZONS:
            w_g |= grid[h]["q50"][:, li] <= thr[li]
            w_s |= grid[h]["snaive"][:, li] <= thr[li]
        lo_i = common[0] - QUIET
        seg = free_arr[lo_i : common[-1] + 1 + horizon, li]
        onsets = full_onsets(seg, thr[li], QUIET)
        for name, w in (("gbm", w_g), ("seasonal_naive", w_s)):
            full_w = np.zeros(len(seg), dtype=bool)
            full_w[QUIET : QUIET + len(common)] = w
            parts[name].append(warning_metrics(full_w, seg, thr[li], horizon, onsets))
    metrics["full_warning"] = {k: combine_warning(v) for k, v in parts.items()}

    guidance = {}
    if lots is not None and {"lat", "lng"} <= set(lots.columns):
        near = nearest_lots(lots.reindex(free.columns), k=3)
        lot_pos = {lid: i for i, lid in enumerate(free.columns)}
        for h in (2, 4):
            guidance[str(h * STEP_MIN)] = guidance_eval(h, grid[h], free_arr, thr, near, lot_pos, common)
    metrics["guidance"] = guidance
    metrics["test_origins"] = int(len(common))
    art = {"model": model, "prep": prep, "cap": cap, "test": t, "pred": pr, "grid": grid, "common": common, "splits": sp, "thr": thr}
    return metrics, art


def guidance_eval(h: int, g: dict, free_arr: np.ndarray, thr: np.ndarray, near: dict, lot_pos: dict, common: np.ndarray) -> dict:
    """When the intended car park is forecast full on arrival, does the recommended alternative still have space?

    Cases: origins where the intended park still shows space now but its median forecast at arrival is 'full'.
    Hit = the car park a method sends the driver to has more than the full threshold free on arrival.
    """
    q10, q50 = g["q10"], g["q50"]
    counts = {"stay": 0, "nearest": 0, "most_free_now": 0, "forecast": 0}
    cases = 0
    for lid, cands in near.items():
        p = lot_pos[lid]
        pos = {c: lot_pos[c] for c in cands}
        for k, i in enumerate(common):
            if not q50[k, p] <= thr[p]:
                continue
            cur = free_arr[i, p]
            act_p = free_arr[i + h, p]
            if np.isnan(cur) or cur <= thr[p] or np.isnan(act_p):
                continue
            act = {c: free_arr[i + h, pos[c]] for c in cands}
            now = {c: free_arr[i, pos[c]] for c in cands}
            if any(np.isnan(v) for v in act.values()) or any(np.isnan(v) for v in now.values()):
                continue
            rec = recommend(cands, {c: q10[k, pos[c]] for c in cands})
            rec_now = recommend(cands, now)
            if rec is None or rec_now is None:
                continue
            cases += 1
            counts["stay"] += act_p > thr[p]
            counts["nearest"] += act[cands[0]] > thr[pos[cands[0]]]
            counts["most_free_now"] += act[rec_now] > thr[pos[rec_now]]
            counts["forecast"] += act[rec] > thr[pos[rec]]
    out = {"cases": int(cases)}
    for k, v in counts.items():
        out[f"hit_{k}"] = float(v / cases) if cases else float("nan")
    return out


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf8")
