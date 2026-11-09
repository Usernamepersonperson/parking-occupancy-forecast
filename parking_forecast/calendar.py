"""Calendar features: Swiss (canton Zurich) public holidays, weekday flags, time-of-day encodings."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Zurich"


def easter_sunday(year: int) -> dt.date:
    """Gregorian Easter (anonymous algorithm, Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def zurich_holidays(year: int) -> dict[dt.date, str]:
    """Public holidays observed in the canton of Zurich (fixed and Easter-based)."""
    e = easter_sunday(year)
    td = dt.timedelta
    return {
        dt.date(year, 1, 1): "new_year",
        dt.date(year, 1, 2): "berchtold",
        e - td(days=2): "good_friday",
        e + td(days=1): "easter_monday",
        dt.date(year, 5, 1): "labour_day",
        e + td(days=39): "ascension",
        e + td(days=50): "whit_monday",
        dt.date(year, 8, 1): "national_day",
        dt.date(year, 12, 25): "christmas",
        dt.date(year, 12, 26): "st_stephen",
    }


def holiday_dates(years: range | list[int]) -> set[dt.date]:
    out: set[dt.date] = set()
    for y in years:
        out |= set(zurich_holidays(y))
    return out


def calendar_frame(index_utc: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features per UTC timestamp, computed on local (Europe/Zurich) wall-clock time."""
    if index_utc.tz is None:
        index_utc = index_utc.tz_localize("UTC")
    local = index_utc.tz_convert(LOCAL_TZ)
    years = range(int(local.year.min()) - 1, int(local.year.max()) + 2)
    hol = holiday_dates(years)
    dates = pd.Series(local.date, index=index_utc)
    is_hol = dates.isin(hol).to_numpy()
    next_hol = (dates + pd.Timedelta(days=1)).isin(hol).to_numpy()
    dow = local.day_of_week.to_numpy()
    tod = (local.hour.to_numpy() * 60 + local.minute.to_numpy()) / 1440.0
    return pd.DataFrame(
        {
            "tod_sin": np.sin(2 * np.pi * tod),
            "tod_cos": np.cos(2 * np.pi * tod),
            "dow": dow,
            "is_holiday": is_hol.astype(float),
            "holiday_eve": next_hol.astype(float),
            "is_saturday": (dow == 5).astype(float),
            "is_sunday": (dow == 6).astype(float),
            "month": local.month.to_numpy(),
        },
        index=index_utc,
    )
