from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from parking_forecast import cli, data, figures
from parking_forecast import pipeline as P

ROOT = Path(__file__).resolve().parent.parent


def test_cli_requires_a_subcommand():
    with pytest.raises(SystemExit):
        cli.main([])


def test_cli_collect_writes_rows(tmp_path, monkeypatch):
    feed = (
        b'<rss xmlns:dc="http://purl.org/dc/elements/1.1/"><channel><item><title>X</title>'
        b"<description>open / 7</description><dc:date>2026-01-01T00:00:00Z</dc:date></item></channel></rss>"
    )
    monkeypatch.setattr(data, "_get", lambda url, timeout=60: feed)
    out = tmp_path / "live.csv"
    assert cli.main(["collect", "--out", str(out), "--max-polls", "1"]) == 0
    assert pd.read_csv(out)["free"].tolist() == [7]


def test_collect_loop_survives_a_failed_poll(tmp_path, monkeypatch):
    calls = {"n": 0}

    def flaky(url, timeout=60):
        calls["n"] += 1
        raise OSError("network down")

    monkeypatch.setattr(data, "_get", flaky)
    data.collect_loop(tmp_path / "x.csv", interval_s=0, max_polls=2)
    assert calls["n"] == 2


def test_committed_dataset_is_complete_and_regular():
    free, w, lots = P.load_committed(ROOT / "data")
    assert free.shape == (35040, 36)
    assert free.index[0] == pd.Timestamp("2022-01-01", tz="UTC")
    assert free.index[-1] == pd.Timestamp("2022-12-31 23:45", tz="UTC")
    assert free.index.is_monotonic_increasing and free.index.is_unique
    assert (free.min() >= 0).all()
    assert len(w) == 8760
    assert list(lots.index) == list(free.columns)


def test_committed_lots_have_anonymous_ids():
    lots = pd.read_csv(ROOT / "data" / "lots.csv")
    assert lots["lot_id"].str.fullmatch(r"lot_\d\d").all()
    assert lots["lat"].between(47.3, 47.5).sum() == lots["lat"].notna().sum()


def test_post_figure_is_exactly_960_by_700(tmp_path):
    m = {
        "by_horizon_min": {h: {"mae_seasonal_naive": 19.7, "mae_gbm": 5.0 + i} for i, h in enumerate(["15", "60", "120"])},
        "guidance": {"30": {"cases": 100, "hit_stay": 0.3, "hit_nearest": 0.7, "hit_most_free_now": 0.9, "hit_forecast": 0.9}},
    }
    p = tmp_path / "post.png"
    figures.make_post_figure(m, p)
    assert Image.open(p).size == (960, 700)


def test_main_figure_is_1600_by_900(panel, weather, tmp_path):
    lots = pd.DataFrame({"lat": [47.0, 47.01, 47.02], "lng": [8.0, 8.0, 8.0]}, index=panel.columns)
    m, art = P.run_experiment(panel, weather, lots, max_iter=10, train_stride=8, ablations=False)
    figures.make_figure(m, art, panel, tmp_path / "f.png")
    assert Image.open(tmp_path / "f.png").size == (1600, 900)
