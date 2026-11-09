import numpy as np
import pandas as pd
import pytest


def make_panel(n_days=60, n_lots=3, seed=0):
    """Synthetic free-space panel: weekly pattern, a weekend dip, noise. Clearly synthetic, only used by tests."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_days * 96, freq="15min", tz="UTC")
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
    dow = idx.day_of_week.to_numpy()
    cols = {}
    for k in range(n_lots):
        cap = 100 + 50 * k
        occ = 0.45 + 0.4 * np.exp(-((hour - 13) ** 2) / 18) * np.where(dow >= 5, 0.5, 1.0)
        free = cap * (1 - occ) + rng.normal(0, 2, len(idx))
        cols[f"lot_{k + 1:02d}"] = np.clip(np.round(free), 0, None)
    return pd.DataFrame(cols, index=idx)


@pytest.fixture(scope="session")
def panel():
    return make_panel()


@pytest.fixture(scope="session")
def weather(panel):
    h = pd.date_range(panel.index[0], panel.index[-1], freq="1h")
    return pd.DataFrame({"temp_c": 5 + np.sin(np.arange(len(h)) / 24), "precip_mm": 0.0}, index=h)
