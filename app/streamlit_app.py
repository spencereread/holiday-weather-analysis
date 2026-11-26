"""Weather Odds: plan holidays and outdoor events from 30+ years of ERA5 history.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weatherplanner import data, stats  # noqa: E402
from weatherplanner.locations import PRESETS, Place  # noqa: E402

st.set_page_config(page_title="Weather Odds", page_icon="☀️", layout="wide")

# Categorical slots (fixed order, validated for CVD separation on adjacent pairs)
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
BLUE_RAMP = [
    [0.0, "#cce3fd"], [0.2, "#9dc8f7"], [0.4, "#6aabf1"],
    [0.6, "#338ced"], [0.8, "#2370cd"], [1.0, "#0e3d7e"],
]
MAX_COMPARE = len(SERIES)
MONTH_TICKS = dict(tickformat="%b", dtick="M1", ticklabelmode="period")


# ---------------------------------------------------------------------------
# Data access (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=24)
def load_hourly(lat: float, lon: float, y0: int, y1: int) -> pd.DataFrame:
    return data.load_hourly(lat, lon, y0, y1)


@st.cache_data(show_spinner=False, max_entries=48)
def _window_flags(lat, lon, y0, y1, start_h, end_h, crit_tuple):
    w = stats.window_summary(data.load_hourly(lat, lon, y0, y1), start_h, end_h)
    return w, stats.event_flags(w, stats.EventCriteria(*crit_tuple))


def event_window_flags(place: Place, y0, y1, start_h, end_h, crit: stats.EventCriteria):
    """Window summary + pass/fail flags for a venue (cached per settings)."""
    with st.spinner(f"Loading ERA5 history for {place.name}…"):
        return _window_flags(place.lat, place.lon, y0, y1, start_h, end_h,
                             (crit.max_rain_mm, crit.max_gust_kmh, crit.min_temp_c, crit.max_temp_c))


def load_with_progress(place: Place, y0: int, y1: int) -> pd.DataFrame:
    """Load data, showing a progress bar when the disk cache is cold."""
    try:
        return load_hourly(place.lat, place.lon, y0, y1)
    except Exception:  # pragma: no cover - surfaced in UI
        st.error(f"Could not download data for {place.name}. Try again in a minute.")
        st.stop()


@st.cache_data(show_spinner=False)
def geocode(q: str) -> list[dict]:
    return data.geocode(q)


def all_places() -> list[Place]:
    return PRESETS + st.session_state.get("custom_places", [])


def place_by_label(label: str) -> Place:
    return next(p for p in all_places() if p.label == label)


def pct(x: float) -> str:
    return "–" if x is None or np.isnan(x) else f"{100 * x:.0f}%"


def style_fig(fig: go.Figure, height: int = 380, yaxis_pct: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    if yaxis_pct:
        fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
    return fig


# ---------------------------------------------------------------------------
# Sidebar: settings + add your own place
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    y0, y1 = st.slider(
        "Years of history", data.FIRST_YEAR + 20, data.LAST_COMPLETE_YEAR,
        (1991, data.LAST_COMPLETE_YEAR),
        help="1991–2025 matches the WMO climate normal plus recent years. "
             "Each new location takes ~30–60 s to download the first time, then it is cached.",
    )
    half_window = st.slider(
        "Smoothing (± days)", 0, 15, 7,
        help="Each date's odds use the same date ± this many days from every year. "
             "±7 days × 35 years ≈ 525 samples per estimate.",
    )

    st.divider()
    st.subheader("Add a place")
    q = st.text_input("Search a town, venue area or region", placeholder="e.g. Chipping Norton")
    if q:
        try:
            hits = geocode(q)
        except Exception:
            hits = []
            st.warning("Place search is unavailable right now.")
        if hits:
            choice = st.selectbox("Matches", hits, format_func=lambda h: h["label"])
            if st.button("Add to my places", width="stretch"):
                new = Place(choice["label"], choice["lat"], choice["lon"], "Custom")
                custom = st.session_state.setdefault("custom_places", [])
                if new.label not in [p.label for p in custom]:
                    custom.append(new)
                st.success(f"Added {choice['name']}. Pick it in either tab.")
        elif q:
            st.caption("No matches.")

    st.divider()
    st.caption(
        "Data: ECMWF ERA5 reanalysis via the Open-Meteo archive API (CC BY 4.0). "
        "These are historical odds, not forecasts."
    )


st.title("☀️ Weather Odds")
st.caption(
    "What actually happened on these dates, year after year: so you can choose "
    "*when* and *where* gives you the best chance of good weather."
)

tab_event, tab_holiday, tab_about = st.tabs(
    ["💍 Outdoor event planner", "🏖️ Holiday comparison", "ℹ️ About the data"]
)


# ---------------------------------------------------------------------------
# Tab 1: outdoor event / wedding planner
# ---------------------------------------------------------------------------

with tab_event:
    labels = [p.label for p in all_places()]
    c1, c2, c3 = st.columns([2, 1.2, 1.2])
    with c1:
        default = labels.index("Cotswolds (UK)")
        ev_place = place_by_label(st.selectbox("Venue location", labels, index=default, key="ev_place"))
    with c2:
        ev_date = st.date_input("Date you're considering", date(2027, 6, 19), format="DD/MM/YYYY")
    with c3:
        start_h, end_h = st.slider("Outdoor hours", 6, 24, (13, 19), format="%d:00",
                                   help="Local time. E.g. ceremony + drinks reception + photos.")

    with st.expander("What counts as a good day? (adjust thresholds)", expanded=False):
        k1, k2, k3 = st.columns(3)
        max_rain = k1.number_input(
            "Max rain in the window (mm)", 0.0, 5.0, 0.2, 0.1,
            help="0.2 mm allows a few spots. ERA5 tends to over-report light drizzle, "
                 "so 0.0 is stricter than reality.",
        )
        max_gust = k2.number_input(
            "Max wind gust (km/h)", 10, 100, 50, 5,
            help="50 km/h (~30 mph) is where gazebos, decor and hats start to struggle; "
                 "marquees are typically rated to ~60–80 km/h.",
        )
        ev_t_lo, ev_t_hi = k3.slider("Comfortable afternoon high (°C)", 0, 40, (15, 28))
    crit = stats.EventCriteria(max_rain, max_gust, ev_t_lo, ev_t_hi)

    with st.spinner(f"Loading {y1 - y0 + 1} years of hourly ERA5 data for {ev_place.name}…"):
        hourly = load_with_progress(ev_place, y0, y1)
    del hourly
    window, flags = event_window_flags(ev_place, y0, y1, start_h, end_h, crit)

    clim = {k: stats.climatology(flags[k], half_window) for k in ["dry", "calm", "comfortable", "all"]}
    sel_doy = stats.date_to_doy(ev_date.month, ev_date.day)
    n_years = window.index.year.nunique()

    st.subheader(f"{ev_date.day} {ev_date:%B}, {start_h}:00–{end_h}:00 near {ev_place.name}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dry during your hours", pct(clim["dry"].at(sel_doy)))
    m2.metric(f"Gusts under {crit.max_gust_kmh:.0f} km/h", pct(clim["calm"].at(sel_doy)))
    m3.metric(f"{crit.min_temp_c:.0f}–{crit.max_temp_c:.0f}°C", pct(clim["comfortable"].at(sel_doy)))
    m4.metric("All three ✔", pct(clim["all"].at(sel_doy)))

    exact = window[(window.index.month == ev_date.month) & (window.index.day == ev_date.day)]
    dry_years = int((exact["rain"] <= crit.max_rain_mm + 1e-6).sum())
    st.caption(
        f"On exactly this date, {start_h}:00–{end_h}:00 was dry in **{dry_years} of {len(exact)}** years. "
        f"Odds above use ±{half_window} days around it across {n_years} years "
        f"(≈{int(clim['dry'].n[sel_doy])} days of history)."
    )

    # --- Compare a shortlist of dates (and venues) -------------------------
    st.markdown("#### Compare your shortlist")
    st.caption("Add the dates you're choosing between. You can also change the venue per row, "
               "e.g. to weigh *Cotswolds on 12 June* against *Kent on 19 June*. "
               "All rows use the hours and thresholds above.")
    if "shortlist" not in st.session_state:
        st.session_state.shortlist = pd.DataFrame({
            "Date": [ev_date + timedelta(days=7 * i) for i in range(3)],
            "Venue": [ev_place.label] * 3,
        })
    edited = st.data_editor(
        st.session_state.shortlist, num_rows="dynamic", hide_index=True, key="shortlist_editor",
        column_config={
            "Date": st.column_config.DateColumn("Date", format="ddd D MMM YYYY", required=True),
            "Venue": st.column_config.SelectboxColumn("Venue", options=labels, required=True,
                                                      default=ev_place.label),
        },
    )
    cands = edited.dropna().drop_duplicates().head(MAX_COMPARE)

    if len(cands) < 2:
        st.info("Add at least two dates to compare.")
    else:
        rows, rates, records = [], {}, {}
        for _, r in cands.iterrows():
            d = pd.Timestamp(r["Date"])
            p = place_by_label(r["Venue"])
            w, f = event_window_flags(p, y0, y1, start_h, end_h, crit)
            doy = stats.date_to_doy(d.month, d.day)
            near = stats.around(doy, half_window)
            near_w = stats.select_days(w, near)
            name = f"{d.day} {d:%b %Y}" + ("" if p == ev_place else f" · {p.name}")
            rates[name] = stats.yearly_rates(f["all"], doy, half_window)
            lo, hi = stats.bootstrap_interval(rates[name])
            ex = w[(w.index.month == d.month) & (w.index.day == d.day)]
            records[name] = (ex["rain"] <= crit.max_rain_mm + 1e-6).groupby(ex.index.year).first()
            rows.append({
                "Option": name, "Weekday": f"{d:%A}",
                "All three ✔": float(stats.select_days(f, near)["all"].mean()),
                "lo": lo, "hi": hi,
                "Dry": float(stats.select_days(f, near)["dry"].mean()),
                "Calm": float(stats.select_days(f, near)["calm"].mean()),
                "Comfortable": float(stats.select_days(f, near)["comfortable"].mean()),
                "Washout (≥5 mm)": float((near_w["rain"] >= 5).mean()),
                "Typical high °C": float(near_w["tmax"].median()),
                "Dry on exact date": f"{int(records[name].sum())} of {len(records[name])} yrs",
            })
        res = pd.DataFrame(rows).sort_values("All three ✔", ascending=False).reset_index(drop=True)

        # Verdict: is the best option genuinely better than the runner-up?
        best, second = res.iloc[0], res.iloc[1]
        cmp = stats.paired_difference(rates[best["Option"]], rates[second["Option"]])
        if cmp["tied"]:
            st.info(
                f"**{best['Option']}** edges it ({pct(best['All three ✔'])} vs "
                f"{pct(second['All three ✔'])}), but that gap is within normal year-to-year "
                f"variation. **Weather-wise these are a tie**: choose on venue, cost or guests."
            )
        else:
            st.success(
                f"**{best['Option']}** is the better bet: about **{100 * cmp['diff']:.0f} percentage "
                f"points** more likely to be dry, calm and comfortable than {second['Option']} "
                f"(90% range {100 * cmp['lo']:.0f}–{100 * cmp['hi']:.0f} pts)."
            )

        left, right = st.columns([1, 1.25])
        with left:
            dots = go.Figure(go.Scatter(
                x=res["All three ✔"], y=res["Option"], mode="markers",
                marker=dict(color=SERIES[1], size=12, line=dict(color="white", width=2)),
                error_x=dict(type="data", symmetric=False, color="rgba(128,128,128,0.7)", thickness=2,
                             array=res["hi"] - res["All three ✔"],
                             arrayminus=res["All three ✔"] - res["lo"]),
                customdata=np.stack([res["lo"], res["hi"]], axis=1),
                hovertemplate="%{y}: %{x:.0%} (90% range %{customdata[0]:.0%}–%{customdata[1]:.0%})"
                              "<extra></extra>",
            ))
            dots.update_yaxes(autorange="reversed")
            dots.update_xaxes(tickformat=".0%", range=[0, 1], title="Chance of all three ✔")
            dots.update_layout(hovermode="closest")
            st.plotly_chart(style_fig(dots, height=80 + 50 * len(res), yaxis_pct=False), width="stretch")
            st.caption("Bars show the 90% range from resampling years. Overlapping bars "
                       "mean the history can't separate the options.")
        with right:
            st.dataframe(
                res.drop(columns=["lo", "hi"]), hide_index=True, width="stretch",
                column_config={
                    **{c: st.column_config.ProgressColumn(c, format="percent", min_value=0, max_value=1)
                       for c in ["All three ✔", "Dry", "Calm", "Comfortable"]},
                    "Washout (≥5 mm)": st.column_config.NumberColumn(format="percent",
                        help="Chance of 5 mm+ during your hours: a properly wet event."),
                    "Typical high °C": st.column_config.NumberColumn(format="%.0f"),
                },
            )

        # Head-to-head history: was each exact date dry, year by year?
        with st.expander("Year-by-year: was each exact date dry during your hours?"):
            hist = pd.DataFrame(records).T.reindex(res["Option"])
            z = hist.astype(float).to_numpy()
            text = np.where(np.isnan(z), "", np.where(z == 1, "dry", "wet"))
            hm = go.Figure(go.Heatmap(
                z=z, x=hist.columns, y=hist.index, zmin=0, zmax=1, xgap=2, ygap=2,
                colorscale=[[0, "#d9d9d9"], [1, SERIES[0]]], showscale=False,
                text=text, hovertemplate="%{y}, %{x}: %{text}<extra></extra>",
            ))
            hm.update_yaxes(autorange="reversed")
            hm.update_layout(hovermode="closest")
            st.plotly_chart(style_fig(hm, height=80 + 40 * len(hist), yaxis_pct=False), width="stretch")
            st.caption("Blue = dry, grey = wet. Columns where every option was wet are "
                       "the summers no date would have saved.")

    # --- Through-the-year chart -------------------------------------------
    st.markdown("#### Odds through the year")
    x = stats.doy_to_ref_date(np.arange(stats.N_DAYS))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=clim["dry"].prob, name="Dry during your hours",
                             line=dict(color=SERIES[0], width=2),
                             hovertemplate="%{y:.0%}"))
    fig.add_trace(go.Scatter(x=x, y=clim["all"].prob, name="Dry + calm + comfortable",
                             line=dict(color=SERIES[1], width=2),
                             hovertemplate="%{y:.0%}"))
    sel_x = stats.doy_to_ref_date(sel_doy)[0]
    fig.add_vline(x=sel_x, line=dict(color="rgba(128,128,128,0.6)", width=1, dash="dot"))
    fig.add_annotation(x=sel_x, y=1.0, text="your date", showarrow=False, yanchor="bottom",
                       font=dict(size=11, color="gray"))
    fig.update_xaxes(**MONTH_TICKS, hoverformat="%d %b")
    st.plotly_chart(style_fig(fig), width="stretch")

    # --- Best dates within a season --------------------------------------
    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("#### Best dates in a season")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        m_from, m_to = st.select_slider("Months", options=months, value=("May", "Sep"))
        d0 = stats.date_to_doy(months.index(m_from) + 1, 1)
        last_day = pd.Timestamp(2001, months.index(m_to) + 1, 1) + pd.offsets.MonthEnd(0)
        d1 = stats.date_to_doy(last_day.month, last_day.day)
        days = stats.doy_range(d0, d1)
        best = pd.DataFrame({
            "Date": stats.doy_to_label(days),
            "All three ✔": clim["all"].prob[days],
            "Dry": clim["dry"].prob[days],
            "Calm": clim["calm"].prob[days],
            "Comfortable": clim["comfortable"].prob[days],
        }).sort_values("All three ✔", ascending=False).head(12)
        st.dataframe(
            best, hide_index=True, width="stretch",
            column_config={c: st.column_config.ProgressColumn(c, format="percent", min_value=0, max_value=1)
                           for c in ["All three ✔", "Dry", "Calm", "Comfortable"]},
        )
        st.caption("Neighbouring dates share most of their history, so treat a cluster of "
                   "high dates as a *good period* rather than one magic day.")

    with right:
        st.markdown("#### Calendar: chance of all three")
        grid = np.full((12, 31), np.nan)
        hover = np.full((12, 31), "", dtype=object)
        for d in range(stats.N_DAYS):
            ts = stats.doy_to_ref_date(d)[0]
            grid[ts.month - 1, ts.day - 1] = clim["all"].prob[d]
        heat = go.Figure(go.Heatmap(
            z=grid, x=list(range(1, 32)), y=months, colorscale=BLUE_RAMP, zmin=0, zmax=1,
            xgap=2, ygap=2, colorbar=dict(tickformat=".0%", thickness=10),
            hovertemplate="%{y} %{x}: %{z:.0%}<extra></extra>",
        ))
        heat.update_yaxes(autorange="reversed")
        heat.update_layout(hovermode="closest")
        st.plotly_chart(style_fig(heat, height=420, yaxis_pct=False), width="stretch")

    # --- Year by year + trend --------------------------------------------
    left, right = st.columns([1.6, 1])
    with left:
        st.markdown(f"#### Every year on {stats.doy_to_label(sel_doy)}: rain during your hours")
        bars = go.Figure(go.Bar(
            x=exact.index.year, y=exact["rain"], marker=dict(color=SERIES[0], cornerradius=4),
            hovertemplate="%{x}: %{y:.1f} mm<extra></extra>",
        ))
        bars.add_hline(y=crit.max_rain_mm, line=dict(color="gray", dash="dot", width=1),
                       annotation_text="your limit", annotation_position="top left")
        bars.update_layout(bargap=0.25, hovermode="closest")
        bars.update_yaxes(title="mm")
        st.plotly_chart(style_fig(bars, height=300, yaxis_pct=False), width="stretch")
    with right:
        st.markdown("#### Is it changing?")
        near = stats.doy_range((sel_doy - 15) % 365, (sel_doy + 15) % 365)
        split = stats.period_split(flags["dry"], near)
        if split:
            delta = split["late"] - split["early"]
            st.metric(f"Dry odds ±15 days, {split['late_label']}", pct(split["late"]),
                      delta=f"{100 * delta:+.0f} pts vs {split['early_label']}")
            split_all = stats.period_split(flags["all"], near)
            st.metric(f"All three, {split_all['late_label']}", pct(split_all["late"]),
                      delta=f"{100 * (split_all['late'] - split_all['early']):+.0f} pts vs "
                            f"{split_all['early_label']}")
            st.caption("Differences of a few points are within year-to-year noise.")


# ---------------------------------------------------------------------------
# Tab 2: holiday comparison
# ---------------------------------------------------------------------------

def color_for(label: str) -> str:
    """Stable colour per location, so removing one doesn't repaint the rest."""
    slots: dict = st.session_state.setdefault("color_slots", {})
    if label not in slots:
        used = set(slots.values())
        free = [i for i in range(len(SERIES)) if i not in used]
        slots[label] = free[0] if free else len(slots) % len(SERIES)
    return SERIES[slots[label]]


with tab_holiday:
    labels = [p.label for p in all_places()]
    holiday_defaults = [p.label for p in PRESETS if p.group == "Holiday"]
    c1, c2 = st.columns([2.2, 1])
    with c1:
        chosen = st.multiselect(f"Destinations to compare (up to {MAX_COMPARE})", labels,
                                default=holiday_defaults, max_selections=MAX_COMPARE)
    with c2:
        trip = st.date_input("Travel dates", (date(2027, 4, 3), date(2027, 4, 10)),
                             format="DD/MM/YYYY")
    # release colour slots of deselected places
    slots = st.session_state.setdefault("color_slots", {})
    for lab in list(slots):
        if lab not in chosen:
            del slots[lab]

    with st.expander("Your ideal holiday day", expanded=True):
        k1, k2, k3, k4 = st.columns(4)
        t_lo, t_hi = k1.slider("Afternoon high (°C)", 5, 45, (20, 32))
        good = stats.GoodDay(
            min_tmax=t_lo, max_tmax=t_hi,
            max_rain_mm=k2.number_input("Rain under (mm/day)", 0.1, 20.0, 1.0, 0.5),
            min_sun_hours=k3.slider("Sunshine at least (hours)", 0, 14, 6),
            max_gust_kmh=k4.slider("Gusts under (km/h)", 20, 100, 50, 5),
        )

    if not chosen:
        st.info("Pick at least one destination.")
    elif not isinstance(trip, tuple) or len(trip) != 2:
        st.info("Pick an end date for your trip.")
    else:
        trip_days = stats.doy_range(stats.date_to_doy(trip[0].month, trip[0].day),
                                    stats.date_to_doy(trip[1].month, trip[1].day))
        rows, curves = [], {}
        prog = st.progress(0.0, text="Loading ERA5 history…")
        for i, lab in enumerate(chosen):
            p = place_by_label(lab)
            prog.progress(i / len(chosen), text=f"Loading {p.name} ({i + 1}/{len(chosen)})…")
            daily = stats.daily_summary(load_with_progress(p, y0, y1))
            gflags = stats.good_day_flags(daily, good)
            s = stats.holiday_summary(daily, gflags, trip_days)
            s["Destination"] = p.name
            s["label"] = lab
            rows.append(s)
            curves[lab] = (p.name, stats.climatology(gflags["good"], half_window).prob)
        prog.empty()

        res = pd.DataFrame(rows).sort_values("p_good", ascending=False)
        n_trip = len(trip_days)
        res["expected_good"] = res["p_good"] * n_trip

        win = res.iloc[0]
        st.success(
            f"**{win['Destination']}** gives the best odds: on average **{win['expected_good']:.1f} of "
            f"{n_trip} days** meet your ideal ({pct(win['p_good'])} of days, {y0}–{y1})."
        )

        left, right = st.columns([1, 1.4])
        with left:
            st.markdown("#### Chance any given day is ideal")
            bar = go.Figure(go.Bar(
                y=res["Destination"], x=res["p_good"], orientation="h",
                marker=dict(color=[color_for(l) for l in res["label"]], cornerradius=4),
                text=[pct(v) for v in res["p_good"]], textposition="outside",
                hovertemplate="%{y}: %{x:.0%}<extra></extra>",
            ))
            bar.update_yaxes(autorange="reversed")
            bar.update_xaxes(tickformat=".0%", range=[0, min(1.0, res["p_good"].max() * 1.25 + 0.05)])
            bar.update_layout(hovermode="closest", bargap=0.35)
            st.plotly_chart(style_fig(bar, height=60 + 48 * len(res), yaxis_pct=False),
                            width="stretch")
        with right:
            st.markdown("#### The numbers")
            table = res[["Destination", "p_good", "expected_good", "tmax_median", "tmin_median",
                         "p_wet", "sun_mean", "p_too_hot"]].rename(columns={
                "p_good": "Ideal days", "expected_good": f"Ideal days (of {n_trip})",
                "tmax_median": "Typical high °C", "tmin_median": "Typical low °C",
                "p_wet": "Wet day (≥1 mm)", "sun_mean": "Sun hrs/day",
                "p_too_hot": f"Over {t_hi}°C",
            })
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                "Ideal days": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                f"Ideal days (of {n_trip})": st.column_config.NumberColumn(format="%.1f"),
                "Typical high °C": st.column_config.NumberColumn(format="%.0f"),
                "Typical low °C": st.column_config.NumberColumn(format="%.0f"),
                "Wet day (≥1 mm)": st.column_config.NumberColumn(format="percent"),
                "Sun hrs/day": st.column_config.NumberColumn(format="%.1f"),
                f"Over {t_hi}°C": st.column_config.NumberColumn(format="percent"),
            })
            st.caption("Typical = median across every trip day in the record. "
                       "Daily highs/lows come from hourly ERA5 values for the whole grid cell.")

        st.markdown("#### When is each place at its best?")
        x = stats.doy_to_ref_date(np.arange(stats.N_DAYS))
        line = go.Figure()
        for lab in res["label"]:
            name, prob = curves[lab]
            line.add_trace(go.Scatter(x=x, y=prob, name=name, line=dict(color=color_for(lab), width=2),
                                      hovertemplate="%{y:.0%}"))
        t0, t1 = stats.doy_to_ref_date([trip_days[0], trip_days[-1]])
        if t1 >= t0:
            line.add_vrect(x0=t0, x1=t1, fillcolor="gray", opacity=0.12, line_width=0,
                           annotation_text="your trip", annotation_position="top left")
        line.update_xaxes(**MONTH_TICKS, hoverformat="%d %b")
        st.plotly_chart(style_fig(line, height=420), width="stretch")
        st.caption("Chance that a day meets *your* ideal, through the year. If another month "
                   "is much higher, shifting your dates may matter more than the destination.")


# ---------------------------------------------------------------------------
# Tab 3: about
# ---------------------------------------------------------------------------

with tab_about:
    st.markdown(
        f"""
### How it works
1. **Data**: hourly [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)
   reanalysis from ECMWF (the same dataset as this repo's CDS notebooks), fetched through the
   [Open-Meteo historical API](https://open-meteo.com/en/docs/historical-weather-api) in local time.
2. **Odds**: for each calendar day, every year's weather on that day ± {half_window} days is
   checked against your criteria. The share that pass is the probability.
3. **This is climatology, not a forecast.** Weather forecasts lose most of their skill beyond
   ~10–15 days, so for planning months ahead, the historical record *is* the best guide.
   Check a real forecast in the final week.

### Known limitations
- **Resolution**: ERA5 grid cells are ~31 km. Coasts, valleys and hills inside a cell can differ.
  A coastal cell may be partly sea, which flattens temperature extremes.
- **Rain**: ERA5 slightly over-reports light drizzle and under-represents short, intense
  showers. A small rain allowance (0.2 mm) gives more realistic "dry" odds than exactly zero.
- **Sunshine** is derived from modelled radiation and **runs high**. In the UK it is often
  ~1.5–2× the Met Office station figures. Use it to compare places, not as an absolute number.
- **Gusts** are modelled for the whole grid cell; sheltered venues will see less, exposed hills more.
- **Neighbouring dates** share most of their data. Differences of a few percent between
  adjacent days are noise.

Data: ERA5 © ECMWF / Copernicus Climate Change Service; delivered by Open-Meteo.com (CC BY 4.0).
"""
    )
