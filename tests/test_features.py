import numpy as np
import pandas as pd

from parking_forecast import features as F


def test_capacity_proxy_is_high_quantile_and_floored():
    df = pd.DataFrame({"a": np.arange(1001, dtype=float), "b": np.zeros(1001)})
    cap = F.capacity_proxy(df, 0.995)
    assert cap["a"] == 995.0
    assert cap["b"] == 1.0  # never zero, avoids division by zero


def test_full_threshold_floor_and_fraction():
    assert F.full_threshold(50.0) == 3.0
    assert F.full_threshold(1000.0) == 30.0
    assert list(F.full_threshold(np.array([10.0, 200.0]))) == [3.0, 6.0]


def test_take_out_of_range_is_nan():
    arr = np.arange(6, dtype=float).reshape(3, 2)
    out = F._take(arr, np.array([-1, 0, 2, 3]))
    assert np.isnan(out[0]).all() and np.isnan(out[3]).all()
    assert out[1].tolist() == [0.0, 1.0] and out[2].tolist() == [4.0, 5.0]


def test_build_features_shape_target_and_horizon(panel, weather):
    cap = F.capacity_proxy(panel)
    prep = F.prepare(panel, weather.reindex(panel.index).interpolate(), cap)
    origins = np.arange(700, 720)
    df = F.build_features(prep, origins, 4)
    assert len(df) == 20 * panel.shape[1]
    assert (df["h"] == 4).all()
    expected = panel.iloc[704, 0] / cap.iloc[0]
    row = df[(df["origin"] == 700) & (df["lot"] == 0)].iloc[0]
    assert abs(row["y"] - expected) < 1e-12


def test_seasonal_naive_is_value_one_week_before_target(panel):
    cap = F.capacity_proxy(panel)
    prep = F.prepare(panel, None, cap)
    df = F.build_features(prep, np.array([1000]), 2, use_weather=False)
    row = df[df["lot"] == 1].iloc[0]
    assert abs(row["snaive_week"] - panel.iloc[1002 - 672, 1] / cap.iloc[1]) < 1e-12
    assert abs(row["snaive_day"] - panel.iloc[1002 - 96, 1] / cap.iloc[1]) < 1e-12


def test_seasonal_naive_falls_back_to_day_when_week_missing():
    df = pd.DataFrame({"snaive_week": [np.nan, 0.5], "snaive_day": [0.2, 0.9]})
    assert F.seasonal_naive(df).tolist() == [0.2, 0.5]


def test_origin_features_ignore_the_future(panel):
    cap = F.capacity_proxy(panel)
    origin = np.array([1000])
    a = F.build_features(F.prepare(panel, None, cap), origin, 4, use_weather=False)
    tampered = panel.copy()
    tampered.iloc[1001:, :] = 12345.0  # change everything after the origin
    b = F.build_features(F.prepare(tampered, None, cap), origin, 4, use_weather=False)
    cols = [c for c in F.LOT_COLS if c not in ("snaive_week", "snaive_day")]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_seasonal_features_reference_only_the_past(panel):
    cap = F.capacity_proxy(panel)
    origin = np.array([2000])
    a = F.build_features(F.prepare(panel, None, cap), origin, 8, use_weather=False)
    tampered = panel.copy()
    tampered.iloc[2000 + 1 :, :] = 999.0  # after the origin
    b = F.build_features(F.prepare(tampered, None, cap), origin, 8, use_weather=False)
    assert a["snaive_week"].equals(b["snaive_week"])
    assert a["snaive_day"].equals(b["snaive_day"])


def test_feature_columns_toggle():
    base = F.feature_columns(False, False)
    assert "temp_c" not in base and "dow" not in base
    assert "temp_c" in F.feature_columns(True, False)
    assert "dow" in F.feature_columns(False, True)


def test_persistence_returns_current():
    df = pd.DataFrame({"cur": [0.1, 0.2]})
    assert F.persistence(df).tolist() == [0.1, 0.2]


def test_missing_weather_yields_nan_columns(panel):
    prep = F.prepare(panel, None, F.capacity_proxy(panel))
    df = F.build_features(prep, np.array([800]), 1)
    assert df["temp_c"].isna().all()
