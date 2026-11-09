import numpy as np
import pandas as pd

from parking_forecast import data

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0"><channel>
<item><title>Parkhaus A / Strasse 1</title><description>open /  163</description><dc:date>2026-11-08T22:37:21Z</dc:date></item>
<item><title>Parkhaus B</title><description>closed /  0</description><dc:date>2026-11-08T22:40:00Z</dc:date></item>
<item><title>Broken</title><description>no data</description><dc:date>2026-11-08T22:40:00Z</dc:date></item>
<item><title>No date</title><description>open / 5</description></item>
</channel></rss>"""


def test_parse_feed_extracts_valid_items_only():
    rows = data.parse_feed(FEED.encode())
    assert [r["free"] for r in rows] == [163, 0]
    assert rows[0]["state"] == "open"
    assert rows[1]["state"] == "closed"
    assert rows[0]["timestamp"].startswith("2026-11-08T22:37")


def test_parse_feed_empty_channel():
    assert data.parse_feed(b"<rss><channel></channel></rss>") == []


def _series(vals, start="2022-01-01", freq="15min"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq=freq, tz="UTC"), dtype=float)


def test_clean_series_drops_negative():
    s = data.clean_series(_series([5, -1, 7]))
    assert np.isnan(s.iloc[1]) and s.iloc[0] == 5


def test_clean_series_long_zero_run_is_nodata():
    s = _series([10] + [0] * 40 + [10])  # 10 h of zeros
    out = data.clean_series(s)
    assert out.iloc[1:-1].isna().all()
    assert out.iloc[0] == 10 and out.iloc[-1] == 10


def test_clean_series_short_zero_run_is_kept_as_full():
    s = _series([10] + [0] * 8 + [10])  # 2 h of zeros
    out = data.clean_series(s)
    assert (out.iloc[1:-1] == 0).all()


def test_clean_series_empty():
    assert data.clean_series(pd.Series([], dtype=float, index=pd.DatetimeIndex([], tz="UTC"))).empty


def test_to_grid_last_in_bin_and_ffill_limit():
    idx = pd.DatetimeIndex(["2022-01-01 00:01", "2022-01-01 00:09", "2022-01-01 00:16"], tz="UTC")
    g = data.to_grid(pd.Series([10.0, 12.0, 20.0], index=idx))
    assert g.iloc[0] == 12.0 and g.iloc[1] == 20.0  # last reading per 15-min bin


def test_to_grid_gap_longer_than_limit_stays_missing():
    idx = pd.DatetimeIndex(["2022-01-01 00:00", "2022-01-01 06:00"], tz="UTC")
    g = data.to_grid(pd.Series([10.0, 20.0], index=idx))
    assert g.iloc[4] == 10.0  # one hour forward fill
    assert np.isnan(g.iloc[5])
    assert g.iloc[-1] == 20.0


def test_to_grid_duplicate_timestamps_keep_last():
    idx = pd.DatetimeIndex(["2022-01-01 00:00", "2022-01-01 00:00"], tz="UTC")
    g = data.to_grid(pd.Series([1.0, 2.0], index=idx))
    assert g.iloc[0] == 2.0


def test_read_dump_skips_garbage_lines(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("2022-01-01T00:00:03,100\nnot a line\n2022-01-01T00:05:03,abc\n2022-01-01T00:20:00,90\n,\n")
    s = data.read_dump(p)
    assert s.iloc[0] == 100 and s.iloc[1] == 90
    assert s.index.tz is not None


def test_list_archive_files_filters_year_and_sorts():
    html = '<a href="zuerichb-2022.csv">x</a><a href="zuericha-2022.csv">x</a><a href="zuericha-2021.csv">x</a><a href="bernx-2022.csv">'
    assert data.list_archive_files(html, 2022) == ["zuericha-2022.csv", "zuerichb-2022.csv"]


def test_lot_id_is_zero_padded():
    assert data.lot_id_from_position(0) == "lot_01"
    assert data.lot_id_from_position(35) == "lot_36"


def test_weather_to_grid_interpolates_temperature():
    h = pd.date_range("2022-01-01", periods=3, freq="1h", tz="UTC")
    w = pd.DataFrame({"temp_c": [0.0, 4.0, 8.0], "precip_mm": [0.0, 1.0, 0.0]}, index=h)
    g = data.weather_to_grid(w, pd.date_range("2022-01-01", periods=9, freq="15min", tz="UTC"))
    assert g["temp_c"].iloc[2] == 2.0
    assert g["precip_mm"].iloc[5] == 1.0


def test_collect_once_appends_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_get", lambda url, timeout=60: FEED.encode())
    out = tmp_path / "live.csv"
    assert data.collect_once(out) == 2
    assert data.collect_once(out) == 2
    df = pd.read_csv(out)
    assert len(df) == 4 and "polled_at" in df.columns
