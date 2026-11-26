import numpy as np
import pandas as pd
import pytest

from weatherplanner import stats


def synthetic_hourly(years=range(2001, 2011)):
    idx = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31 23:00", freq="h")
    df = pd.DataFrame(index=idx)
    df["temperature_2m"] = 15.0
    df["precipitation"] = 0.0
    df["cloud_cover"] = 50.0
    df["wind_speed_10m"] = 10.0
    df["wind_gusts_10m"] = 20.0
    df["sunshine_duration"] = np.where((idx.hour >= 8) & (idx.hour <= 18), 3600.0, 0.0)
    return df


def test_doy365_handles_leap_years():
    idx = pd.to_datetime(["2001-03-01", "2004-02-29", "2004-03-01", "2004-12-31", "2001-12-31"])
    assert list(stats.doy365(idx)) == [59, -1, 59, 364, 364]


def test_doy_range_wraps_year():
    r = stats.doy_range(362, 2)
    assert list(r) == [362, 363, 364, 0, 1, 2]


def test_window_uses_hour_ending_precip():
    df = synthetic_hourly()
    # rain in the hour 13:00-14:00 is stamped 14:00 -> inside a 12-19 window
    df.loc["2005-06-10 14:00", "precipitation"] = 2.0
    # rain stamped 12:00 fell 11:00-12:00 -> outside the window
    df.loc["2005-06-11 12:00", "precipitation"] = 3.0
    # rain stamped 19:00 fell 18:00-19:00 -> inside
    df.loc["2005-06-12 19:00", "precipitation"] = 1.0
    w = stats.window_summary(df, 12, 19)
    assert w.loc["2005-06-10", "rain"] == pytest.approx(2.0)
    assert w.loc["2005-06-11", "rain"] == pytest.approx(0.0)
    assert w.loc["2005-06-12", "rain"] == pytest.approx(1.0)
    assert w.loc["2005-06-10", "wet_hours"] == 1


def test_climatology_matches_known_frequency():
    # every other year is wet on all days -> probability 0.5 everywhere
    dates = pd.date_range("2001-01-01", "2010-12-31", freq="D")
    flags = pd.Series(dates.year % 2 == 0, index=dates)
    clim = stats.climatology(flags, half_window=7)
    assert np.allclose(clim.prob, 0.5)
    assert clim.n.min() >= 15 * 10 - 1  # ~15 days x 10 years


def test_climatology_wraps_new_year():
    dates = pd.date_range("2001-01-01", "2010-12-31", freq="D")
    # only 31 Dec is "true"
    flags = pd.Series((dates.month == 12) & (dates.day == 31), index=dates)
    clim = stats.climatology(flags, half_window=1)
    assert clim.prob[0] == pytest.approx(1 / 3)  # 1 Jan window includes 31 Dec
    assert clim.prob[364] == pytest.approx(1 / 3)
    assert clim.prob[100] == 0


def test_event_flags_and_good_day():
    df = synthetic_hourly()
    df.loc["2003-07-04 15:00", "wind_gusts_10m"] = 60.0
    w = stats.window_summary(df, 12, 19)
    f = stats.event_flags(w, stats.EventCriteria(min_temp_c=10))
    assert f["all"].mean() > 0.99
    assert not f.loc["2003-07-04", "calm"]

    daily = stats.daily_summary(df)
    assert daily["sun_hours"].iloc[0] == pytest.approx(11)
    g = stats.good_day_flags(daily, stats.GoodDay(min_tmax=10, min_sun_hours=10))
    summary = stats.holiday_summary(daily, g, stats.doy_range(180, 190))
    # 4 Jul 2003 (day 184) has a 60 km/h gust > 50 limit: 1 of 110 days fails
    assert summary["p_good"] == pytest.approx(109 / 110)
    assert summary["tmax_median"] == pytest.approx(15)


def test_paired_difference_detects_real_and_tied():
    years = pd.Index(range(1991, 2026))
    rng = np.random.default_rng(1)
    base = pd.Series(rng.uniform(0.3, 0.7, len(years)), index=years)
    clearly_better = base + 0.2
    real = stats.paired_difference(clearly_better, base)
    assert not real["tied"] and real["diff"] == pytest.approx(0.2)
    # same odds on average, independent year-to-year noise -> tied
    other = pd.Series(rng.uniform(0.3, 0.7, len(years)), index=years)
    other = other - other.mean() + base.mean()
    assert stats.paired_difference(other, base)["tied"]


def test_yearly_rates_and_interval():
    dates = pd.date_range("2001-01-01", "2010-12-31", freq="D")
    flags = pd.Series(dates.year % 2 == 0, index=dates)
    rates = stats.yearly_rates(flags, stats.date_to_doy(6, 15), 7)
    assert len(rates) == 10 and rates.mean() == pytest.approx(0.5)
    lo, hi = stats.bootstrap_interval(rates)
    assert lo < 0.5 < hi
    assert len(stats.around(0, 3)) == 7  # wraps the year end


def test_period_split():
    dates = pd.date_range("2001-01-01", "2010-12-31", freq="D")
    flags = pd.Series(dates.year >= 2006, index=dates)
    out = stats.period_split(flags, np.arange(365))
    assert out["early"] == 0 and out["late"] == 1
