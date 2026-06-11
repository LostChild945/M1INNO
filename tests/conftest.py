import pytest
import pandas as pd


class MockRow:
    """SQLAlchemy Row stub — expose ._mapping and direct attribute access."""
    def __init__(self, **kwargs):
        self._mapping = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def sample_sensor_df():
    """DataFrame matching load_dataset() output — used by xgboost feature tests."""
    return pd.DataFrame({
        "crop_type":        ["wheat", "corn", "rice", "soybean", "sunflower", "wheat"],
        "country":          ["France"] * 6,
        "year":             [2010, 2011, 2012, 2013, 2014, 2015],
        "soil_moisture":    [0.42, 0.50, 0.38, 0.45, 0.41, 0.44],
        "soil_ph":          [6.5, 6.8, 6.2, 6.4, 6.6, 6.5],
        "nitrogen_ppm":     [110.0, 130.0, 90.0, 120.0, 100.0, 115.0],
        "air_temp_c":       [14.0, 22.0, 27.0, 23.0, 20.0, 15.0],
        "humidity_pct":     [65.0, 70.0, 75.0, 68.0, 60.0, 62.0],
        "rainfall_mm":      [480.0, 620.0, 1150.0, 680.0, 430.0, 510.0],
        "solar_rad_wm2":    [180.0, 200.0, 190.0, 175.0, 185.0, 178.0],
        "value_tonnes":     [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0],
        "yoy_growth_pct":   [0.0, 10.0, 9.1, 8.3, 7.7, 7.1],
        "ma5_tonnes":       [1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1280.0],
        "value_normalized": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "yield_t_per_ha":   [3.5, 5.2, 4.1, 2.9, 2.1, 3.6],
    })
