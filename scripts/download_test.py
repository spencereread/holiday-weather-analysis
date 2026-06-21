import cdsapi

client = cdsapi.Client()

client.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": "2m_temperature",
        "year": "2024",
        "month": "04",
        "day": "01",
        "time": "12:00",
        "data_format": "netcdf",
        "area": [46, -10, 35, 30]
    },
    r"F:\HolidayWeather\data\era5_test.nc"
)

print("Download complete")