import datetime as dt

import pandas as pd

from parking_forecast.calendar import calendar_frame, easter_sunday, zurich_holidays


def test_easter_known_dates():
    assert easter_sunday(2022) == dt.date(2022, 4, 17)
    assert easter_sunday(2024) == dt.date(2024, 3, 31)
    assert easter_sunday(2025) == dt.date(2025, 4, 20)
    assert easter_sunday(2000) == dt.date(2000, 4, 23)


def test_holidays_2022_easter_based():
    h = zurich_holidays(2022)
    assert h[dt.date(2022, 4, 15)] == "good_friday"
    assert h[dt.date(2022, 4, 18)] == "easter_monday"
    assert h[dt.date(2022, 5, 26)] == "ascension"
    assert h[dt.date(2022, 6, 6)] == "whit_monday"


def test_holidays_fixed_dates_present():
    h = zurich_holidays(2023)
    for d in [(1, 1), (1, 2), (5, 1), (8, 1), (12, 25), (12, 26)]:
        assert dt.date(2023, *d) in h
    assert len(h) == 10


def test_calendar_flags_holiday_and_weekend():
    idx = pd.DatetimeIndex(["2022-12-25 12:00", "2022-12-24 12:00", "2022-12-31 12:00", "2022-12-21 12:00"], tz="UTC")
    c = calendar_frame(idx)
    assert c["is_holiday"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert c["holiday_eve"].tolist() == [1.0, 1.0, 1.0, 0.0]  # 25th precedes the 26th; 31st precedes Jan 1
    assert c["is_saturday"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert c["is_sunday"].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_calendar_uses_local_time_across_dst():
    winter = calendar_frame(pd.DatetimeIndex(["2022-01-10 11:00"], tz="UTC"))  # 12:00 local (UTC+1)
    summer = calendar_frame(pd.DatetimeIndex(["2022-07-11 10:00"], tz="UTC"))  # 12:00 local (UTC+2)
    assert abs(winter["tod_sin"].iloc[0] - summer["tod_sin"].iloc[0]) < 1e-9


def test_calendar_midnight_local_late_utc_evening_belongs_to_next_day():
    c = calendar_frame(pd.DatetimeIndex(["2022-12-24 23:30"], tz="UTC"))  # 00:30 on Dec 25 local
    assert c["is_holiday"].iloc[0] == 1.0


def test_naive_index_is_treated_as_utc():
    a = calendar_frame(pd.DatetimeIndex(["2022-06-01 08:00"]))
    b = calendar_frame(pd.DatetimeIndex(["2022-06-01 08:00"], tz="UTC"))
    assert a["tod_cos"].iloc[0] == b["tod_cos"].iloc[0]
