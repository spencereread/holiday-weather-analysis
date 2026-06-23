import cdsapi

client = cdsapi.Client()

years = [str(y) for y in range(1991, 2026)]

client.retrieve(
    "derived-era5-single-levels-daily-statistics",
    {
        "product_type": "reanalysis",
        "variable": "total_precipitation",
        "daily_statistic": "daily_sum",
        "time_zone": "UTC+00:00",
        "frequency": "1_hourly",
        "year": years,
        "month": "04",
        "day": ["03","04","05","06","07","08","09","10"],
        "area": [46, -10, 35, 30],
        "data_format": "netcdf",
    },
    r"F:\HolidayWeather\data\raw\april_precip_1991_2025.nc"
)

print("Download complete")