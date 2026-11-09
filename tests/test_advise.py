import numpy as np

from parking_forecast.advise import advisory, full_by
from parking_forecast.features import HORIZONS


def test_full_by_returns_first_horizon_in_minutes():
    assert full_by({1: 30.0, 2: 10.0, 4: 2.0, 8: 1.0}, 3.0) == 60


def test_full_by_none_when_never_full():
    assert full_by({1: 30.0, 8: 20.0}, 3.0) is None


def test_full_by_ignores_nan():
    assert full_by({1: np.nan, 2: 1.0}, 3.0) == 30


def _grid(q50_lot0, n_lots=3):
    g = {}
    for h in HORIZONS:
        g[h] = {q: np.full((1, n_lots), v) for q, v in (("q10", 20.0), ("q50", 30.0), ("q90", 40.0))}
        g[h]["q50"][0, 0] = q50_lot0[h]
    g[2]["q10"][0, 2] = 90.0  # lot c has the most free spaces at the lower forecast
    return g


def test_advisory_reroutes_when_full_on_arrival():
    g = _grid({h: 1.0 for h in HORIZONS})
    out = advisory(g, 0, 0, np.array([3.0, 3.0, 3.0]), ["a", "b", "c"], {"a": ["b", "c"]})
    assert out["full_by_min"] == 15
    assert out["reroute_to"] == "c"
    assert set(out["forecast_q10_q50_q90"]) == {15, 30, 45, 60, 90, 120}


def test_advisory_no_reroute_when_space_expected():
    g = _grid({h: 50.0 for h in HORIZONS})
    out = advisory(g, 0, 0, np.array([3.0, 3.0, 3.0]), ["a", "b", "c"], {"a": ["b", "c"]})
    assert out["full_by_min"] is None and out["reroute_to"] is None


def test_advisory_without_neighbours_has_no_recommendation():
    g = _grid({h: 1.0 for h in HORIZONS})
    out = advisory(g, 0, 0, np.array([3.0, 3.0, 3.0]), ["a", "b", "c"], {})
    assert out["full_by_min"] == 15 and out["reroute_to"] is None
