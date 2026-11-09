import json

import numpy as np
import pandas as pd

from parking_forecast import pipeline as P


def test_default_splits_are_chronological(panel):
    sp = P.default_splits(panel.index)
    assert sp.start < sp.train_end < sp.calib_end < sp.end
    assert sp.test_start > sp.calib_end


def test_origin_indices_keep_target_inside_window(panel):
    o = P.origin_indices(panel.index, panel.index[100], panel.index[200], 8)
    assert o[0] == 100 and o[-1] + 8 == 200


def test_origin_indices_stride(panel):
    o = P.origin_indices(panel.index, panel.index[0], panel.index[100], 4, stride=2)
    assert np.all(np.diff(o) == 2)


def test_run_experiment_end_to_end_small(panel, weather):
    lots = pd.DataFrame({"lat": [47.0, 47.01, 47.02], "lng": [8.0, 8.0, 8.0]}, index=panel.columns)
    m, _ = P.run_experiment(panel, weather, lots, max_iter=15, train_stride=4, ablations=False)
    assert set(m["by_horizon_min"]) == {"15", "60", "120"}
    for v in m["by_horizon_min"].values():
        assert v["mae_gbm"] >= 0 and 0 <= v["coverage_calibrated"] <= 1
    assert pd.Timestamp(m["split"]["train_end"]) < pd.Timestamp(m["split"]["test_start"])
    assert "gbm" in m["full_warning"] and "seasonal_naive" in m["full_warning"]
    assert m["guidance"]["30"]["cases"] >= 0


def test_gbm_beats_persistence_on_periodic_signal(panel, weather):
    m, _ = P.run_experiment(panel, weather, None, max_iter=30, train_stride=4, ablations=False)
    assert m["overall"]["mae_gbm"] < m["overall"]["mae_persistence"]


def test_test_period_is_after_training_period(panel, weather):
    _, art = P.run_experiment(panel, weather, None, max_iter=10, train_stride=8, ablations=False)
    assert panel.index[art["common"][0]] >= art["splits"].test_start


def test_save_metrics_roundtrip(tmp_path):
    P.save_metrics({"a": np.float64(1.5)}, tmp_path / "x" / "m.json")
    assert json.loads((tmp_path / "x" / "m.json").read_text())["a"] == 1.5


def test_guidance_eval_counts_hit_for_good_alternative():
    free = np.full((40, 3), 50.0)
    free[10:, 0] = 2.0  # intended car park fills up
    thr = np.array([3.0, 3.0, 3.0])
    g = {"q10": np.full((20, 3), 40.0), "q50": np.full((20, 3), 50.0)}
    g["q50"][:, 0] = 1.0  # forecast: intended car park will be full
    out = P.guidance_eval(2, g, free, thr, {"a": ["b", "c"]}, {"a": 0, "b": 1, "c": 2}, np.arange(20))
    assert out["cases"] == 10  # origins 0..9 still show space now
    assert out["hit_forecast"] == 1.0
    assert out["hit_stay"] == 0.8  # origins 8, 9 reach full at index 10 or later... origins 0..7 stay fine


def test_guidance_eval_skips_cases_with_missing_data():
    free = np.full((40, 3), 50.0)
    free[:, 1] = np.nan
    free[:, 2] = np.nan
    thr = np.array([3.0, 3.0, 3.0])
    g = {"q10": np.full((20, 3), 40.0), "q50": np.full((20, 3), 1.0)}
    out = P.guidance_eval(2, g, free, thr, {"a": ["b", "c"]}, {"a": 0, "b": 1, "c": 2}, np.arange(20))
    assert out["cases"] == 0 and np.isnan(out["hit_forecast"])
