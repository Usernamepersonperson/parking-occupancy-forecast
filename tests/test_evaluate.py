import numpy as np
import pandas as pd

from parking_forecast import evaluate as E


def test_mae_known_value():
    assert E.mae([1, 2, 3], [2, 2, 5]) == 1.0


def test_coverage_inclusive_bounds():
    assert E.coverage([1, 2, 3, 10], [1, 1, 1, 1], [2, 2, 2, 2]) == 0.5


def test_full_onsets_requires_quiet_period():
    free = np.array([20, 20, 20, 20, 2, 2, 20, 20, 20, 20, 1], dtype=float)
    assert E.full_onsets(free, thr=3, quiet_steps=3).tolist() == [4, 10]


def test_full_onsets_ignores_ongoing_full_and_nan():
    free = np.array([20, 20, 2, 2, 2, np.nan, 2, 20, 20, 20, 2], dtype=float)
    assert E.full_onsets(free, thr=3, quiet_steps=2).tolist() == [2, 10]


def test_full_onsets_nan_in_window_blocks_event():
    free = np.array([20, np.nan, 20, 2], dtype=float)
    assert E.full_onsets(free, thr=3, quiet_steps=3).tolist() == []


def test_warning_metrics_exact_case():
    free = np.array([20] * 10 + [2, 2] + [20] * 6, dtype=float)
    warn = np.zeros(len(free), dtype=bool)
    warn[8] = True  # 2 steps before the onset at index 10
    warn[17] = True  # false alarm: nothing follows
    m = E.warning_metrics(warn, free, thr=3, horizon=4, onsets=np.array([10]))
    assert (m["issued"], m["tp"]) == (2, 1)
    assert (m["events"], m["events_hit"]) == (1, 1)
    assert m["lead_min"] == [30]


def test_warning_metrics_ignores_warnings_while_already_full():
    free = np.array([20, 20, 2, 2, 2, 2], dtype=float)
    warn = np.array([False, False, True, True, True, True])
    m = E.warning_metrics(warn, free, thr=3, horizon=2, onsets=np.array([], dtype=int))
    assert m["issued"] == 0


def test_warning_metrics_event_without_warning_is_missed():
    free = np.array([20] * 10 + [2], dtype=float)
    m = E.warning_metrics(np.zeros(len(free), dtype=bool), free, thr=3, horizon=4, onsets=np.array([10]))
    assert m["events"] == 1 and m["events_hit"] == 0 and m["lead_min"] == []


def test_warning_too_early_does_not_count_for_recall():
    free = np.array([20] * 12 + [2], dtype=float)
    warn = np.zeros(len(free), dtype=bool)
    warn[2] = True  # 10 steps before the onset, outside a 4-step window
    m = E.warning_metrics(warn, free, thr=3, horizon=4, onsets=np.array([12]))
    assert m["events_hit"] == 0 and m["tp"] == 0


def test_combine_warning_pools_counts():
    a = {"issued": 4, "tp": 2, "events": 2, "events_hit": 1, "lead_min": [15]}
    b = {"issued": 6, "tp": 6, "events": 2, "events_hit": 2, "lead_min": [45, 60]}
    c = E.combine_warning([a, b])
    assert c["precision"] == 0.8 and c["recall"] == 0.75 and c["lead_median_min"] == 45.0


def test_combine_warning_no_events_gives_nan():
    c = E.combine_warning([{"issued": 0, "tp": 0, "events": 0, "events_hit": 0, "lead_min": []}])
    assert np.isnan(c["precision"]) and np.isnan(c["recall"])


def test_haversine_known_distance():
    # Zurich main station to Bern main station is roughly 95 km as the crow flies
    assert 90 < E.haversine_km(47.3779, 8.5403, 46.9490, 7.4390) < 100
    assert E.haversine_km(47.0, 8.0, 47.0, 8.0) == 0.0


def test_nearest_lots_orders_by_distance_and_skips_missing_coords():
    coords = pd.DataFrame({"lat": [47.00, 47.01, 47.05, np.nan], "lng": [8.0, 8.0, 8.0, np.nan]}, index=["a", "b", "c", "d"])
    near = E.nearest_lots(coords, k=2)
    assert near["a"] == ["b", "c"]
    assert near["c"] == ["b", "a"]
    assert "d" not in near and all("d" not in v for v in near.values())


def test_recommend_picks_highest_and_handles_nan():
    assert E.recommend(["a", "b", "c"], {"a": 5.0, "b": 9.0, "c": np.nan}) == "b"
    assert E.recommend(["a"], {"a": np.nan}) is None
    assert E.recommend(["a", "b"], {"a": 3.0, "b": 3.0}) == "a"
