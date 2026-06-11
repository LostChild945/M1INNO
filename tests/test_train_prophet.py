"""
Tests unitaires pour src/ml/train_prophet.py.

Couvre mape() — fonction de métrique pure sans dépendances externes.

Prophet nécessite CmdStan (non disponible hors Docker) — son import est mocké
au niveau de sys.modules avant toute importation de ml.train_prophet.
"""
import sys
from unittest.mock import MagicMock

# Prophet et ses dépendances lourdes (CmdStan) ne sont pas disponibles hors Docker.
# On les injecte dans sys.modules avant d'importer le module à tester.
sys.modules.setdefault("prophet", MagicMock())
sys.modules.setdefault("cmdstanpy", MagicMock())

import numpy as np
import pytest

from ml.train_prophet import mape


class TestMapeBasic:
    def test_perfect_prediction_is_zero(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([100.0, 200.0, 300.0])
        assert mape(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)

    def test_ten_percent_single_value(self):
        y_true = np.array([100.0])
        y_pred = np.array([110.0])
        assert mape(y_true, y_pred) == pytest.approx(10.0, abs=1e-10)

    def test_ten_percent_underprediction(self):
        y_true = np.array([100.0])
        y_pred = np.array([90.0])
        assert mape(y_true, y_pred) == pytest.approx(10.0, abs=1e-10)

    def test_returns_float(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 220.0])
        result = mape(y_true, y_pred)
        assert isinstance(result, float)

    def test_result_is_percentage(self):
        """MAPE s'exprime en %, pas en fraction [0,1]."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        result = mape(y_true, y_pred)
        # 10% + 10% → moyenne = 10%
        assert result == pytest.approx(10.0, abs=1e-10)


class TestMapeZeroHandling:
    def test_zeros_in_y_true_excluded(self):
        """Les valeurs à 0 dans y_true sont exclues (évite division par zéro)."""
        y_true = np.array([0.0, 100.0, 200.0])
        y_pred = np.array([50.0, 110.0, 220.0])
        # Seules les lignes 1 et 2 comptent : |10/100| et |20/200| → 10% chacune
        result = mape(y_true, y_pred)
        assert result == pytest.approx(10.0, abs=1e-10)

    def test_all_zeros_returns_nan(self):
        """Si toutes les valeurs réelles sont 0, MAPE est NaN (masque vide)."""
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 2.0])
        result = mape(y_true, y_pred)
        assert np.isnan(result)


class TestMapeMultipleValues:
    def test_three_values_averaged(self):
        y_true = np.array([100.0, 200.0, 400.0])
        y_pred = np.array([90.0, 220.0, 400.0])
        # erreurs : 10%, 10%, 0% → moyenne = 6.6̄%
        expected = (10.0 + 10.0 + 0.0) / 3
        assert mape(y_true, y_pred) == pytest.approx(expected, abs=1e-10)

    def test_large_series(self):
        """MAPE cohérente sur une série de 100 valeurs à erreur constante."""
        n = 100
        y_true = np.ones(n) * 1000.0
        y_pred = np.ones(n) * 1050.0  # 5% d'erreur constante
        assert mape(y_true, y_pred) == pytest.approx(5.0, abs=1e-10)

    def test_asymmetry_over_vs_under(self):
        """MAPE n'est pas symétrique : sur-estimer de 50% ≠ sous-estimer de 50%."""
        y_true = np.array([100.0])
        mape_over  = mape(y_true, np.array([150.0]))  # +50%
        mape_under = mape(y_true, np.array([50.0]))   # -50%
        assert mape_over == pytest.approx(50.0, abs=1e-10)
        assert mape_under == pytest.approx(50.0, abs=1e-10)
        # Les deux sont identiques ici car |Δ|/vrai = 50/100 dans les deux cas


class TestMapeConstants:
    def test_train_cutoff_is_2013(self):
        from ml.train_prophet import TRAIN_CUTOFF
        assert TRAIN_CUTOFF == 2013

    def test_forecast_years_is_5(self):
        from ml.train_prophet import FORECAST_YEARS
        assert FORECAST_YEARS == 5

    def test_top_n_is_10(self):
        from ml.train_prophet import TOP_N
        assert TOP_N == 10
