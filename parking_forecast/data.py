"""Loading, cleaning and gridding of car-park free-space series, plus the live-feed parser and fetchers."""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

FEED_URL = "http://www.pls-zh.ch/plsFeed/rss"
ARCHIVE_INDEX = "https://parkendd.de/dumps/"
WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive?latitude=47.37&longitude=8.54"
    "&start_date={start}&end_date={end}&hourly=temperature_2m,precipitation&timezone=UTC"
)
STEP_MIN = 15
MAX_FFILL_BINS = 4  # carry a reading forward at most one hour
NODATA_ZERO_HOURS = 6  # a run of exact zeros longer than this is "no data", not "full"


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "parking-occupancy-forecast/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (fixed https/http open-data URLs)
        return r.read()


def parse_feed(xml_bytes: bytes) -> list[dict]:
    """Parse the real-time RSS feed into rows: name, status, free, timestamp (UTC, ISO).

    The item description reads ``"<state> /  <free>"``. Items without a numeric free count are skipped.
    """
    root = ET.fromstring(xml_bytes)
    ns = {"dc": "http://purl.org/dc/elements/1.1/"}
    rows = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        desc = (item.findtext("description") or "").strip()
        stamp = item.findtext("dc:date", namespaces=ns)
        m = re.match(r"^\s*([A-Za-z]+)\s*/\s*(-?\d+)\s*$", desc)
        if not m or not stamp:
            continue
        rows.append({"name": title, "state": m.group(1).lower(), "free": int(m.group(2)), "timestamp": stamp})
    return rows


def clean_series(s: pd.Series, nodata_zero_hours: float = NODATA_ZERO_HOURS) -> pd.Series:
    """Drop negative values and mark long runs of exact zeros as missing (closed / no data)."""
    s = s.astype(float).copy()
    s[s < 0] = np.nan
    if s.empty:
        return s
    is_zero = (s == 0).to_numpy()
    idx = s.index
    i = 0
    n = len(s)
    while i < n:
        if is_zero[i]:
            j = i
            while j + 1 < n and is_zero[j + 1]:
                j += 1
            if (idx[j] - idx[i]) > pd.Timedelta(hours=nodata_zero_hours):
                s.iloc[i : j + 1] = np.nan
            i = j + 1
        else:
            i += 1
    return s


def to_grid(raw: pd.Series, step_min: int = STEP_MIN, max_ffill: int = MAX_FFILL_BINS) -> pd.Series:
    """Resample irregular timestamps to a regular grid: last reading in each bin, forward-filled a short while."""
    if raw.empty:
        return raw
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    grid = raw.resample(f"{step_min}min").last()
    return grid.ffill(limit=max_ffill)


def read_dump(path: Path) -> pd.Series:
    """Read a ``timestamp,free`` dump file (UTC) into a cleaned, regularly gridded series."""
    df = pd.read_csv(path, header=None, names=["t", "free"], on_bad_lines="skip")
    df["t"] = pd.to_datetime(df["t"], errors="coerce", utc=True)
    df["free"] = pd.to_numeric(df["free"], errors="coerce")
    df = df.dropna()
    s = pd.Series(df["free"].to_numpy(), index=pd.DatetimeIndex(df["t"]), name="free")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return to_grid(clean_series(s))


def list_archive_files(index_html: str, year: int) -> list[str]:
    """File names of the per-car-park yearly dumps for one city and year, sorted (stable lot numbering)."""
    names = re.findall(rf'href="(zuerich[^"]*-{year}\.csv)"', index_html)
    return sorted(set(names))


def lot_id_from_position(i: int) -> str:
    return f"lot_{i + 1:02d}"


def fetch_archive(year: int, cache_dir: Path) -> dict[str, Path]:
    """Download the yearly dumps into cache (skips files already present). Returns lot_id -> path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    index = _get(ARCHIVE_INDEX).decode("utf8", "replace")
    out = {}
    for i, name in enumerate(list_archive_files(index, year)):
        dest = cache_dir / f"{lot_id_from_position(i)}-{year}.csv"
        if not dest.exists() or dest.stat().st_size == 0:
            dest.write_bytes(_get(ARCHIVE_INDEX + urllib.parse.quote(name)))
        out[lot_id_from_position(i)] = dest
    return out


def fetch_weather(start: str, end: str) -> pd.DataFrame:
    """Hourly 2 m temperature (deg C) and precipitation (mm) from the Open-Meteo archive, UTC index."""
    import json

    d = json.loads(_get(WEATHER_URL.format(start=start, end=end)))["hourly"]
    df = pd.DataFrame({"temp_c": d["temperature_2m"], "precip_mm": d["precipitation"]})
    df.index = pd.DatetimeIndex(pd.to_datetime(d["time"], utc=True), name="t")
    return df


def weather_to_grid(w: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly weather onto the 15-minute grid (temperature interpolated, precipitation held)."""
    w = w.reindex(w.index.union(index)).sort_index()
    out = pd.DataFrame(
        {"temp_c": w["temp_c"].interpolate(limit=8), "precip_mm": w["precip_mm"].ffill(limit=3)},
    )
    return out.reindex(index)


def collect_once(out_csv: Path) -> int:
    """Poll the live feed once and append rows to a CSV. Returns the number of rows written."""
    rows = parse_feed(_get(FEED_URL, timeout=30))
    polled = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    df = pd.DataFrame(rows)
    if df.empty:
        return 0
    df["polled_at"] = polled
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
    return len(df)


def collect_loop(out_csv: Path, interval_s: int = 300, max_polls: int | None = None) -> None:
    n = 0
    while max_polls is None or n < max_polls:
        try:
            collect_once(out_csv)
        except Exception as e:  # keep polling through transient network errors
            print(f"poll failed: {e}")
        n += 1
        if max_polls is None or n < max_polls:
            time.sleep(interval_s)
