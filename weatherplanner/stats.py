"""Climatological probabilities from hourly ERA5 data.

Core idea: for a calendar day D, every year in the record contributes the days
within +/- `half_window` of D. With 35 years and a +/-7 day window each
estimate uses ~525 days, which is far more stable than the 35 samples of one
exact date, while still resolving the seasonal cycle.

Days are indexed on a 365-day calendar (0 = 1 Jan); 29 February is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_DAYS = 365
_REF_YEAR = 2001  # any non-leap year, for converting day index <-> date


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

def doy365(index: pd.DatetimeIndex) -> np.ndarray:
    """0-based day of year on a 365-day calendar; 29 Feb maps to -1."""
    idx = pd.DatetimeIndex(index)
    doy = idx.day_of_year.to_numpy() - 1
    leap_after_feb = idx.is_leap_year & (idx.month > 2)
    doy = np.where(leap_after_feb, doy - 1, doy)
    feb29 = (idx.month == 2) & (idx.day == 29)
    return np.where(feb29, -1, doy)


def doy_to_label(doy: int | np.ndarray) -> list[str] | str:
    dates = pd.Timestamp(f"{_REF_YEAR}-01-01") + pd.to_timedelta(np.atleast_1d(doy), unit="D")
    labels = [d.strftime("%d %b").lstrip("0") for d in dates]
    return labels[0] if np.isscalar(doy) else labels


def doy_to_ref_date(doy) -> pd.DatetimeIndex:
    """Map day index to a date in the reference year (handy for chart x-axes)."""
    return pd.Timestamp(f"{_REF_YEAR}-01-01") + pd.to_timedelta(np.atleast_1d(doy), unit="D")


def date_to_doy(month: int, day: int) -> int:
    if month == 2 and day == 29:
        day = 28
    return int(pd.Timestamp(year=_REF_YEAR, month=month, day=day).day_of_year - 1)


def doy_range(start: int, end: int) -> np.ndarray:
    """Inclusive day range that may wrap the year end (e.g. 20 Dec - 5 Jan)."""
    if end >= start:
        return np.arange(start, end + 1)
    return np.concatenate([np.arange(start, N_DAYS), np.arange(0, end + 1)])


# ---------------------------------------------------------------------------
# Aggregation from hourly data
# ---------------------------------------------------------------------------

def daily_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    """Whole-day statistics per local calendar date."""
    g = hourly.groupby(hourly.index.normalize())
    out = pd.DataFrame(
        {
            "tmax": g["temperature_2m"].max(),
            "tmin": g["temperature_2m"].min(),
            "precip": g["precipitation"].sum(),
            "sun_hours": g["sunshine_duration"].sum() / 3600.0,
            "gust_max": g["wind_gusts_10m"].max(),
            "cloud_mean": g["cloud_cover"].mean(),
        }
    )
    counts = g.size()
    return out[counts >= 20]  # drop partial days at record edges


def window_summary(hourly: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
    """Statistics for a local time window [start_hour, end_hour) on each date.

    Precipitation, gusts and sunshine are values for the hour *ending* at the
    timestamp, so for those the window uses timestamps start_hour+1 .. end_hour
    (shift back one hour, then select start_hour <= h < end_hour). Temperature
    and cloud are instantaneous, so they use timestamps start_hour .. end_hour.
    """
    if not 0 <= start_hour < end_hour <= 24:
        raise ValueError("Need 0 <= start_hour < end_hour <= 24")

    acc = hourly[["precipitation", "wind_gusts_10m", "sunshine_duration"]].copy()
    acc.index = acc.index - pd.Timedelta(hours=1)
    h = acc.index.hour
    acc = acc[(h >= start_hour) & (h < end_hour)]
    ga = acc.groupby(acc.index.normalize())

    inst = hourly[["temperature_2m", "cloud_cover"]]
    hi = inst.index.hour
    inst = inst[(hi >= start_hour) & (hi <= min(end_hour, 23))]
    gi = inst.groupby(inst.index.normalize())

    out = pd.DataFrame(
        {
            "rain": ga["precipitation"].sum(),
            "wet_hours": (acc["precipitation"] >= 0.1).groupby(acc.index.normalize()).sum(),
            "gust_max": ga["wind_gusts_10m"].max(),
            "sun_hours": ga["sunshine_duration"].sum() / 3600.0,
            "tmax": gi["temperature_2m"].max(),
            "tmean": gi["temperature_2m"].mean(),
            "cloud_mean": gi["cloud_cover"].mean(),
        }
    )
    expected = end_hour - start_hour
    complete = ga.size().reindex(out.index, fill_value=0) >= expected - 1  # DST days
    return out[complete].dropna()


# ---------------------------------------------------------------------------
# Probability estimation
# ---------------------------------------------------------------------------

def _circular_sum(a: np.ndarray, half_window: int) -> np.ndarray:
    if half_window <= 0:
        return a.astype(float)
    ext = np.concatenate([a[-half_window:], a, a[:half_window]])
    return np.convolve(ext, np.ones(2 * half_window + 1), mode="valid")


@dataclass
class Climatology:
    """Probability of an event for each day of the (365-day) year."""

    prob: np.ndarray   # shape (365,), in [0, 1]
    n: np.ndarray      # samples used for each day

    def at(self, doy: int) -> float:
        return float(self.prob[doy])

    def mean_over(self, days: np.ndarray) -> float:
        return float(np.nanmean(self.prob[days]))


def climatology(flags: pd.Series, half_window: int = 7) -> Climatology:
    """Smoothed day-of-year probability that `flags` (bool per date) is True."""
    flags = flags.dropna()
    d = doy365(flags.index)
    keep = d >= 0
    d = d[keep]
    v = flags.to_numpy()[keep].astype(float)
    hits = np.bincount(d, weights=v, minlength=N_DAYS)
    n = np.bincount(d, minlength=N_DAYS).astype(float)
    hs, ns = _circular_sum(hits, half_window), _circular_sum(n, half_window)
    with np.errstate(invalid="ignore", divide="ignore"):
        prob = np.where(ns > 0, hs / ns, np.nan)
    return Climatology(prob=prob, n=ns)


def climatology_mean(values: pd.Series, half_window: int = 7) -> np.ndarray:
    """Smoothed day-of-year mean of a numeric series."""
    values = values.dropna()
    d = doy365(values.index)
    keep = d >= 0
    s = np.bincount(d[keep], weights=values.to_numpy()[keep], minlength=N_DAYS)
    n = np.bincount(d[keep], minlength=N_DAYS).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _circular_sum(s, half_window) / _circular_sum(n, half_window)


def select_days(df: pd.DataFrame, days: np.ndarray) -> pd.DataFrame:
    """Rows of a date-indexed frame whose calendar day falls in `days`."""
    return df[np.isin(doy365(df.index), days)]


def per_year(flags: pd.Series, days: np.ndarray) -> pd.Series:
    """Fraction of days in `days` that meet the condition, for each year."""
    sub = select_days(flags.to_frame("f"), days)["f"]
    return sub.groupby(sub.index.year).mean()


def period_split(flags: pd.Series, days: np.ndarray) -> dict:
    """Compare the earlier and later halves of the record for a set of days."""
    sub = select_days(flags.to_frame("f"), days)["f"]
    years = np.sort(sub.index.year.unique())
    if len(years) < 6:
        return {}
    mid = years[len(years) // 2]
    early = sub[sub.index.year < mid]
    late = sub[sub.index.year >= mid]
    return {
        "early_label": f"{years[0]}–{mid - 1}",
        "late_label": f"{mid}–{years[-1]}",
        "early": float(early.mean()),
        "late": float(late.mean()),
    }


def around(doy: int, half_window: int) -> np.ndarray:
    """Calendar days within +/- half_window of doy (wrapping the year end)."""
    if half_window <= 0:
        return np.array([doy])
    return doy_range((doy - half_window) % N_DAYS, (doy + half_window) % N_DAYS)


def yearly_rates(flags: pd.Series, doy: int, half_window: int) -> pd.Series:
    """For each year, the fraction of days around `doy` meeting the condition."""
    return per_year(flags, around(doy, half_window))


def bootstrap_interval(rates: pd.Series, n_boot: int = 2000, level: float = 0.9,
                       seed: int = 0) -> tuple[float, float]:
    """Interval for the mean of per-year rates, resampling whole years.

    Resampling years (not days) respects the fact that days in the same
    year share weather regimes, so it gives honest uncertainty.
    """
    v = rates.dropna().to_numpy()
    if len(v) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def paired_difference(a: pd.Series, b: pd.Series, n_boot: int = 2000, level: float = 0.9,
                      seed: int = 0) -> dict:
    """Difference in odds between two options, paired by year, with an interval.

    Pairing by year removes weather shared by both options (e.g. a wet summer
    hits every June date), so the comparison is sharper than two separate
    intervals. If the interval spans zero the options are statistically tied.
    """
    years = a.dropna().index.intersection(b.dropna().index)
    d = (a[years] - b[years]).to_numpy()
    if len(d) < 3:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "tied": True}
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    alpha = (1 - level) / 2
    lo, hi = float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))
    return {"diff": float(d.mean()), "lo": lo, "hi": hi, "tied": lo <= 0 <= hi}


# ---------------------------------------------------------------------------
# Event (e.g. outdoor wedding) conditions
# ---------------------------------------------------------------------------

@dataclass
class EventCriteria:
    max_rain_mm: float = 0.2      # total rain allowed in the window
    max_gust_kmh: float = 50.0    # ~30 mph: decor / gazebo threshold
    min_temp_c: float = 15.0      # window maximum temperature at least this
    max_temp_c: float = 30.0      # ...and no hotter than this


def event_flags(window: pd.DataFrame, c: EventCriteria) -> pd.DataFrame:
    flags = pd.DataFrame(index=window.index)
    flags["dry"] = window["rain"] <= c.max_rain_mm + 1e-6
    flags["calm"] = window["gust_max"] <= c.max_gust_kmh
    flags["comfortable"] = (window["tmax"] >= c.min_temp_c) & (window["tmax"] <= c.max_temp_c)
    flags["all"] = flags["dry"] & flags["calm"] & flags["comfortable"]
    return flags


# ---------------------------------------------------------------------------
# Holiday "good day" conditions
# ---------------------------------------------------------------------------

@dataclass
class GoodDay:
    min_tmax: float = 22.0
    max_tmax: float = 32.0
    max_rain_mm: float = 1.0
    min_sun_hours: float = 6.0
    max_gust_kmh: float = 50.0


def good_day_flags(daily: pd.DataFrame, g: GoodDay) -> pd.DataFrame:
    flags = pd.DataFrame(index=daily.index)
    flags["warm_enough"] = daily["tmax"] >= g.min_tmax
    flags["not_too_hot"] = daily["tmax"] <= g.max_tmax
    flags["dry"] = daily["precip"] < g.max_rain_mm
    flags["sunny"] = daily["sun_hours"] >= g.min_sun_hours
    flags["calm"] = daily["gust_max"] <= g.max_gust_kmh
    flags["good"] = flags.all(axis=1)
    return flags


def holiday_summary(daily: pd.DataFrame, flags: pd.DataFrame, days: np.ndarray) -> dict:
    """Headline numbers for one location over a set of calendar days."""
    d = select_days(daily, days)
    f = select_days(flags, days)
    return {
        "p_good": float(f["good"].mean()),
        "tmax_median": float(d["tmax"].median()),
        "tmax_p10": float(d["tmax"].quantile(0.1)),
        "tmax_p90": float(d["tmax"].quantile(0.9)),
        "tmin_median": float(d["tmin"].median()),
        "p_wet": float((d["precip"] >= 1.0).mean()),
        "sun_mean": float(d["sun_hours"].mean()),
        "p_warm": float(f["warm_enough"].mean()),
        "p_too_hot": float((~f["not_too_hot"]).mean()),
        "n_days": int(len(d)),
    }
