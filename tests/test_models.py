import numpy as np
import pandas as pd

from parking_forecast import features as F
from parking_forecast.models import QuantileForecaster, cqr_shift


def test_cqr_shift_negative_when_interval_wider_than_needed():
    y = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    assert cqr_shift(y, y - 1, y + 1, alpha=0.2) == -1.0  # interval can shrink by 1 on each side


def test_cqr_shift_positive_when_interval_too_narrow():
    y = np.arange(10, dtype=float)
    assert cqr_shift(y, y + 1, y + 2, alpha=0.2) == 1.0  # every point lies 1 below the interval


def test_cqr_shift_empty_is_zero():
    assert cqr_shift(np.array([]), np.array([]), np.array([]), 0.2) == 0.0


def test_cqr_shift_finite_sample_level():
    y = np.zeros(4)
    lo = np.array([-1.0, -2.0, -3.0, -4.0])
    hi = np.array([1.0, 2.0, 3.0, 4.0])
    # scores = [-1, -2, -3, -4]; level = min(1, ceil(5 * 0.8) / 4) = 1 -> the maximum score
    assert cqr_shift(y, lo, hi, alpha=0.2) == -1.0


def _tables(panel, weather):
    cap = F.capacity_proxy(panel)
    prep = F.prepare(panel, weather.reindex(panel.index).interpolate(), cap)
    n = len(panel)

    def tb(lo, hi):
        return pd.concat([F.build_features(prep, np.arange(lo, hi - h), h) for h in (1, 4)], ignore_index=True)

    return tb(700, int(n * 0.6)), tb(int(n * 0.6), int(n * 0.8)), tb(int(n * 0.8), n - 10)


def test_forecaster_beats_persistence_on_periodic_signal(panel, weather):
    train, _, test = _tables(panel, weather)
    m = QuantileForecaster(max_iter=60).fit(train)
    t = test.dropna(subset=["y", "cur"])
    p = m.predict_raw(t)
    err_gbm = np.abs(p["q50"].to_numpy() - t["y"].to_numpy()).mean()
    err_pers = np.abs(t["cur"].to_numpy() - t["y"].to_numpy()).mean()
    assert err_gbm < err_pers


def test_quantiles_are_ordered(panel, weather):
    train, calib, test = _tables(panel, weather)
    m = QuantileForecaster(max_iter=30).fit(train)
    m.calibrate(calib)
    p = m.predict(test)
    assert (p["q10"] <= p["q50"]).all() and (p["q50"] <= p["q90"]).all()
    assert (p["q10"] >= 0).all()


def test_calibration_brings_coverage_close_to_nominal(panel, weather):
    train, calib, test = _tables(panel, weather)
    m = QuantileForecaster(max_iter=30).fit(train)
    m.calibrate(calib, alpha=0.2)
    t = test.dropna(subset=["y"])
    p = m.predict(t)
    cov = float(((t["y"] >= p["q10"]) & (t["y"] <= p["q90"])).mean())
    assert 0.7 < cov < 0.95


def test_ablation_models_use_fewer_columns():
    assert len(QuantileForecaster(use_weather=False).columns) < len(QuantileForecaster().columns)
    assert len(QuantileForecaster(use_calendar=False).columns) < len(QuantileForecaster().columns)


def test_fit_ignores_rows_without_target(panel, weather):
    train, _, _ = _tables(panel, weather)
    train.loc[train.index[:50], "y"] = np.nan
    m = QuantileForecaster(max_iter=10).fit(train)
    assert set(m.models) == {0.1, 0.5, 0.9}
