"""
Tests unitaires pour src/ml/simulate_data.py.

Couvre simulate_yield() — fonction pure sans dépendances externes.
Le bruit gaussien est patché pour rendre les assertions déterministes.
"""
import numpy as np
import pytest
from unittest.mock import patch

from ml.simulate_data import simulate_yield, CROPS


# ── Helpers ──────────────────────────────────────────────────────────────────

def _yield_no_noise(crop, pesticide_norm, soil_ph, soil_moisture,
                    soil_nitrogen, air_temp, rainfall):
    """Appelle simulate_yield avec bruit forcé à 0."""
    with patch("numpy.random.normal", return_value=0.0):
        return simulate_yield(crop, pesticide_norm, soil_ph,
                              soil_moisture, soil_nitrogen, air_temp, rainfall)


def _expected_no_noise(crop, pesticide_norm, soil_ph, soil_moisture,
                       soil_nitrogen, air_temp, rainfall):
    """Calcule la valeur attendue en appliquant directement la formule."""
    cfg  = CROPS[crop]
    base = cfg["base"]
    pest_effect    = base * 0.20 * np.log1p(pesticide_norm * 4)
    ph_dev         = soil_ph - 6.5
    soil_effect    = base * (-0.06 * ph_dev**2 + 0.04 * soil_moisture + 0.0008 * soil_nitrogen)
    temp_dev       = air_temp - cfg["opt_temp"]
    rain_dev       = rainfall - cfg["opt_rain"]
    climate_effect = base * (-0.008 * temp_dev**2 - 0.000008 * rain_dev**2 + 0.05)
    return float(round(max(0.1, base + pest_effect + soil_effect + climate_effect), 3))


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSimulateYieldReturnType:
    def test_returns_float(self):
        result = simulate_yield("wheat", 0.5, 6.5, 0.45, 120.0, 15.0, 500.0)
        assert isinstance(result, float)

    def test_rounded_to_3_decimals(self):
        np.random.seed(0)
        result = simulate_yield("wheat", 0.5, 6.5, 0.45, 120.0, 15.0, 500.0)
        assert result == round(result, 3)


class TestSimulateYieldClamp:
    def test_minimum_is_0_1_with_extreme_negative_noise(self):
        """Noise très négatif → rendement toujours ≥ 0.1 (clamp)."""
        with patch("numpy.random.normal", return_value=-100.0):
            result = simulate_yield("wheat", 0.0, 10.0, 0.0, 0.0, 50.0, 5000.0)
        assert result == pytest.approx(0.1, abs=1e-9)

    def test_always_positive_across_crops(self):
        """Toutes les cultures produisent un rendement ≥ 0.1 pour 50 tirages."""
        rng = np.random.RandomState(42)
        for crop in CROPS:
            for _ in range(50):
                result = simulate_yield(
                    crop,
                    pesticide_norm=rng.uniform(0, 1),
                    soil_ph=rng.uniform(5.0, 8.5),
                    soil_moisture=rng.uniform(0.1, 0.8),
                    soil_nitrogen=rng.uniform(30, 250),
                    air_temp=rng.uniform(0, 40),
                    rainfall=rng.uniform(100, 3000),
                )
                assert result >= 0.1, f"{crop}: rendement négatif inattendu"


class TestSimulateYieldFormula:
    def test_formula_wheat_optimal(self):
        """Vérifie la formule exacte pour wheat dans des conditions optimales."""
        result   = _yield_no_noise("wheat", 0.5, 6.5, 0.45, 120.0, 15.0, 500.0)
        expected = _expected_no_noise("wheat", 0.5, 6.5, 0.45, 120.0, 15.0, 500.0)
        assert result == pytest.approx(expected, abs=1e-3)

    def test_formula_corn_optimal(self):
        result   = _yield_no_noise("corn", 0.5, 6.5, 0.45, 120.0, 22.0, 650.0)
        expected = _expected_no_noise("corn", 0.5, 6.5, 0.45, 120.0, 22.0, 650.0)
        assert result == pytest.approx(expected, abs=1e-3)

    def test_ph_deviation_reduces_yield(self):
        """pH loin de 6.5 doit diminuer le rendement."""
        yield_optimal = _yield_no_noise("wheat", 0.5, 6.5, 0.45, 120.0, 15.0, 500.0)
        yield_bad_ph  = _yield_no_noise("wheat", 0.5, 9.5, 0.45, 120.0, 15.0, 500.0)
        assert yield_optimal > yield_bad_ph

    def test_pesticide_increases_yield(self):
        """Un indice pesticide plus élevé améliore le rendement (effet protecteur)."""
        yield_no_pest   = _yield_no_noise("wheat", 0.0, 6.5, 0.45, 120.0, 15.0, 500.0)
        yield_with_pest = _yield_no_noise("wheat", 0.8, 6.5, 0.45, 120.0, 15.0, 500.0)
        assert yield_with_pest > yield_no_pest

    def test_temperature_deviation_reduces_yield(self):
        """Température très éloignée de l'optimum diminue le rendement."""
        yield_opt  = _yield_no_noise("wheat", 0.5, 6.5, 0.45, 120.0, 15.0, 500.0)
        yield_hot  = _yield_no_noise("wheat", 0.5, 6.5, 0.45, 120.0, 40.0, 500.0)
        assert yield_opt > yield_hot

    def test_deterministic_with_zero_noise(self):
        """Avec bruit = 0, deux appels identiques donnent le même résultat."""
        r1 = _yield_no_noise("rice", 0.3, 6.5, 0.50, 100.0, 28.0, 1200.0)
        r2 = _yield_no_noise("rice", 0.3, 6.5, 0.50, 100.0, 28.0, 1200.0)
        assert r1 == r2


class TestSimulateYieldEdgeCases:
    def test_all_crops_accepted(self):
        """Chaque culture dans CROPS doit être acceptée sans erreur."""
        for crop in CROPS:
            cfg = CROPS[crop]
            result = _yield_no_noise(
                crop, 0.5, 6.5, 0.45, 120.0, cfg["opt_temp"], cfg["opt_rain"]
            )
            assert result >= 0.1

    def test_unknown_crop_raises_key_error(self):
        with pytest.raises(KeyError):
            simulate_yield("banana", 0.5, 6.5, 0.45, 120.0, 20.0, 500.0)

    def test_pesticide_norm_zero(self):
        """pesticide_norm=0 → log1p(0)=0, pas d'effet pesticide."""
        result = _yield_no_noise("wheat", 0.0, 6.5, 0.45, 120.0, 15.0, 500.0)
        expected = _expected_no_noise("wheat", 0.0, 6.5, 0.45, 120.0, 15.0, 500.0)
        assert result == pytest.approx(expected, abs=1e-3)
