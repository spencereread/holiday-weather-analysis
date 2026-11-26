"""Fetch hourly ERA5 data from the Open-Meteo historical archive, with a disk cache.

Open-Meteo serves ECMWF ERA5 reanalysis (the same dataset as the CDS API used
in the original notebooks) through a fast JSON API, so any point on Earth can
be fetched on demand without CDS queue times. Data are licensed CC BY 4.0.

Downloads are cached as Parquet in 5-year blocks per ERA5 grid cell, so a
location is only fetched once and changing the year range reuses blocks.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "sunshine_duration",
]

FIRST_YEAR = 1940
LAST_COMPLETE_YEAR = 2025  # ERA5 lags real time by ~5 days; use whole years only
BLOCK_YEARS = 5
GRID = 0.25  # ERA5 native resolution in degrees

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"


def snap(value: float) -> float:
    """Snap a coordinate to the ERA5 0.25° grid (used as the cache key)."""
    return round(round(value / GRID) * GRID, 2)


def _blocks(start_year: int, end_year: int) -> list[tuple[int, int]]:
    first = start_year - start_year % BLOCK_YEARS
    out = []
    for b in range(first, end_year + 1, BLOCK_YEARS):
        out.append((max(b, FIRST_YEAR), min(b + BLOCK_YEARS - 1, LAST_COMPLETE_YEAR)))
    return out


def _request(params: dict, retries: int = 4) -> dict:
    delay = 5.0
    for attempt in range(retries):
        resp = requests.get(ARCHIVE_URL, params=params, timeout=180)
        if resp.status_code == 429 and attempt < retries - 1:
            time.sleep(delay)  # rate limited: back off and retry
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("Open-Meteo request failed after retries")


def _fetch_block(lat: float, lon: float, y0: int, y1: int) -> pd.DataFrame:
    payload = _request(
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{y0}-01-01",
            "end_date": f"{y1}-12-31",
            "hourly": ",".join(HOURLY_VARS),
            "models": "era5",
            "timezone": "auto",  # local time, so "afternoon" means local afternoon
        }
    )
    hourly = payload["hourly"]
    df = pd.DataFrame({v: hourly[v] for v in HOURLY_VARS})
    df.index = pd.to_datetime(hourly["time"])
    df.index.name = "time"
    return df.astype("float32")


def load_hourly(
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    cache_dir: Path = DEFAULT_CACHE,
    progress: Callable[[float], None] | None = None,
) -> pd.DataFrame:
    """Hourly ERA5 data (local time index) for the grid cell containing lat/lon."""
    start_year = max(start_year, FIRST_YEAR)
    end_year = min(end_year, LAST_COMPLETE_YEAR)
    lat, lon = snap(lat), snap(lon)
    cache_dir.mkdir(parents=True, exist_ok=True)

    blocks = _blocks(start_year, end_year)
    frames = []
    for i, (y0, y1) in enumerate(blocks):
        path = cache_dir / f"era5_{lat:.2f}_{lon:.2f}_{y0}_{y1}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
        else:
            block = _fetch_block(lat, lon, y0, y1)
            block.to_parquet(path)
            frames.append(block)
        if progress:
            progress((i + 1) / len(blocks))

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated()]
    return df[(df.index.year >= start_year) & (df.index.year <= end_year)]


def geocode(name: str, count: int = 8) -> list[dict]:
    """Search for a place name. Returns dicts with name/country/lat/lon."""
    if not name.strip():
        return []
    resp = requests.get(GEOCODE_URL, params={"name": name, "count": count}, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("results", []) or []
    out = []
    for r in results:
        parts = [r.get("name"), r.get("admin1"), r.get("country")]
        out.append(
            {
                "name": r["name"],
                "label": ", ".join(p for p in parts if p),
                "lat": r["latitude"],
                "lon": r["longitude"],
            }
        )
    return out
