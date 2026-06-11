"""
Tests unitaires pour src/ml/train_xgboost.py.

Couvre build_features() — transformation DataFrame pure sans accès DB ni MLflow.
"""
import numpy as np
import pandas as pd
import pytest

from ml.train_xgboost import build_features, PARAMS

EXPECTED_FEATURE_COLS = [
    "crop_encoded", "year",
    "soil_moisture", "soil_ph", "nitrogen_ppm",
    "air_temp_c", "humidity_pct", "rainfall_mm", "solar_rad_wm2",
    "value_tonnes", "yoy_growth_pct", "ma5_tonnes", "value_normalized",
]


class TestBuildFeaturesShape:
    def test_output_length_matches_input(self, sample_sensor_df):
        X, y, le, feature_cols = build_features(sample_sensor_df)
        assert len(X) == len(sample_sensor_df)
        assert len(y) == len(sample_sensor_df)

    def test_feature_columns_exact_order(self, sample_sensor_df):
        X, y, le, feature_cols = build_features(sample_sensor_df)
        assert feature_cols == EXPECTED_FEATURE_COLS
        assert list(X.columns) == EXPECTED_FEATURE_COLS

    def test_returns_four_elements(self, sample_sensor_df):
        result = build_features(sample_sensor_df)
        assert len(result) == 4


class TestBuildFeaturesEncoding:
    def test_crop_encoded_is_integer(self, sample_sensor_df):
        X, _, _, _ = build_features(sample_sensor_df)
        assert pd.api.types.is_integer_dtype(X["crop_encoded"])

    def test_label_encoder_alphabetical_order(self, sample_sensor_df):
        """LabelEncoder trie alphabétiquement : corn=0, rice=1, soybean=2, sunflower=3, wheat=4."""
        X, _, le, _ = build_features(sample_sensor_df)
        expected = {"corn": 0, "rice": 1, "soybean": 2, "sunflower": 3, "wheat": 4}
        for crop, code in expected.items():
            mask = sample_sensor_df["crop_type"] == crop
            if mask.any():
                actual_code = X.loc[mask, "crop_encoded"].iloc[0]
                assert actual_code == code, f"{crop} devrait être encodé {code}, obtenu {actual_code}"

    def test_label_encoder_fitted_classes(self, sample_sensor_df):
        """Le LabelEncoder retourné doit connaître les 5 cultures."""
        _, _, le, _ = build_features(sample_sensor_df)
        assert list(le.classes_) == ["corn", "rice", "soybean", "sunflower", "wheat"]

    def test_inverse_transform_roundtrip(self, sample_sensor_df):
        """inverse_transform(transform(x)) == x pour chaque culture présente."""
        X, _, le, _ = build_features(sample_sensor_df)
        for crop in sample_sensor_df["crop_type"].unique():
            encoded = le.transform([crop])
            decoded = le.inverse_transform(encoded)
            assert decoded[0] == crop


class TestBuildFeaturesTarget:
    def test_target_values_preserved(self, sample_sensor_df):
        _, y, _, _ = build_features(sample_sensor_df)
        pd.testing.assert_series_equal(
            y.reset_index(drop=True),
            sample_sensor_df["yield_t_per_ha"].reset_index(drop=True),
        )

    def test_target_not_in_features(self, sample_sensor_df):
        X, _, _, _ = build_features(sample_sensor_df)
        assert "yield_t_per_ha" not in X.columns


class TestBuildFeaturesNaN:
    def test_nan_values_filled_with_zero(self):
        """Les NaN dans les colonnes de features sont remplacés par 0."""
        df = pd.DataFrame({
            "crop_type":        ["wheat"],
            "year":             [2010],
            "soil_moisture":    [np.nan],
            "soil_ph":          [6.5],
            "nitrogen_ppm":     [110.0],
            "air_temp_c":       [14.0],
            "humidity_pct":     [65.0],
            "rainfall_mm":      [480.0],
            "solar_rad_wm2":    [180.0],
            "value_tonnes":     [np.nan],
            "yoy_growth_pct":   [np.nan],
            "ma5_tonnes":       [np.nan],
            "value_normalized": [np.nan],
            "yield_t_per_ha":   [3.5],
        })
        X, _, _, _ = build_features(df)
        assert not X.isnull().any().any()
        assert X.loc[0, "soil_moisture"] == 0.0
        assert X.loc[0, "value_tonnes"] == 0.0

    def test_no_nan_in_output_with_clean_input(self, sample_sensor_df):
        X, _, _, _ = build_features(sample_sensor_df)
        assert not X.isnull().any().any()


class TestBuildFeaturesImmutability:
    def test_original_df_not_modified(self, sample_sensor_df):
        """build_features ne doit pas modifier le DataFrame original."""
        original_cols = list(sample_sensor_df.columns)
        build_features(sample_sensor_df)
        assert list(sample_sensor_df.columns) == original_cols
        assert "crop_encoded" not in sample_sensor_df.columns


class TestParams:
    def test_required_keys_present(self):
        required = {
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "min_child_weight", "reg_alpha",
            "reg_lambda", "random_state",
        }
        assert required.issubset(set(PARAMS.keys()))

    def test_learning_rate_in_range(self):
        assert 0 < PARAMS["learning_rate"] < 1

    def test_n_estimators_positive(self):
        assert PARAMS["n_estimators"] > 0

    def test_regularization_non_negative(self):
        assert PARAMS["reg_alpha"] >= 0
        assert PARAMS["reg_lambda"] >= 0
