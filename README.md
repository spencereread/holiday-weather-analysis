# Holiday Weather Analysis → Weather Odds

A Python project using ERA5 reanalysis data from Copernicus to compare weather conditions across
potential holiday destinations. It now includes **Weather Odds**, an interactive app that turns
30+ years of hourly history into plain-English probabilities for planning.

## Weather Odds app

**💍 Outdoor event planner** (e.g. UK weddings)
- Pick a venue area (presets or search any town) and the hours you'll be outside
- Chance that window is **dry**, **calm** (gusts under your limit) and **comfortable**
- Odds through the whole year, best dates in a season, and a month × day calendar heatmap
- Year-by-year rainfall on your exact date, and whether the odds have shifted recently

**🏖️ Holiday comparison**
- Compare up to 8 destinations (defaults are the six regions from the original analysis)
- Define *your* ideal day: temperature range, rain, sunshine, wind
- Ranking by chance of an ideal day, expected ideal days in your trip, typical highs/lows,
  wet-day chance, and when each place is at its best through the year

### Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The first time you use a location it downloads ~35 years of hourly data (≈20–60 s). After that it
is cached in `data/cache/` and loads instantly.

### Deploy (Streamlit Community Cloud, free)

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub and click **Create app**.
2. Repository `spencereread/holiday-weather-analysis`, branch `main`,
   main file `app/streamlit_app.py`.
3. Under **Advanced settings**, choose Python **3.12**. No secrets are needed.

Dependencies come from `app/requirements.txt`. The preset locations load from `data/seed/`
(compact ERA5 copies committed to the repo, built with `python scripts/build_seed.py`), so they
need no API calls. Searched places are downloaded on demand and cached until the app restarts.

### How the odds are calculated

For each calendar day, every year's weather on that day **± 7 days** is checked against your
criteria; the probability is the share that pass. With 35 years that's ~525 days per estimate.
This is **climatology, not a forecast**: beyond ~2 weeks forecasts have little skill, so history is
the best guide for planning months ahead.

### Project layout

```
app/streamlit_app.py      UI
weatherplanner/data.py    ERA5 download (Open-Meteo archive API) + Parquet cache
weatherplanner/stats.py   window/daily summaries, smoothed day-of-year probabilities
weatherplanner/locations.py  preset places
tests/                    unit tests (pytest)
notebooks/, scripts/      original April villa analysis (CDS API)
```

### Limitations
- ERA5 grid cells are ~31 km. Local terrain and coastlines inside a cell are smoothed out.
- ERA5 over-reports light drizzle and under-represents short intense showers.
- Modelled sunshine duration runs high versus station records. Use it to compare places.

## Original analysis: April villa comparison

Locations: Algarve, South France, Croatia, Bosnia, Barcelona Coast, Romania. This part covers
temperature, rainfall, sunshine, wind, outdoor comfort and a holiday suitability ranking. See
`notebooks/` and `docs/country_weather_summary.md`.

## Tools
Python, CDS API, Open-Meteo API, Xarray, Pandas, Streamlit, Plotly, Matplotlib, ERA5 Reanalysis.

Data: ERA5 © ECMWF / Copernicus Climate Change Service; app data via Open-Meteo.com (CC BY 4.0).
