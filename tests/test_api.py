"""
Tests unitaires pour src/api/main.py.

Deux niveaux :
- Logique pure (LabelEncoder, irrigation, constantes) — pas de réseau.
- Routes FastAPI via TestClient avec DB et MLflow mockés.

Le lifespan tente de charger le modèle au démarrage ; l'exception est attrapée
en interne, donc le TestClient démarre même sans MLflow disponible.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from api.main import app, _label_encoder, CROP_CLASSES, CROP_WATER_NEEDS


# ── Helpers ──────────────────────────────────────────────────────────────────

class MockRow:
    def __init__(self, **kwargs):
        self._mapping = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


_UNSET = object()


def _mock_engine(fetchall=_UNSET, fetchone=_UNSET):
    """Construit un moteur SQLAlchemy mocké pour connect() et begin().

    Passe explicitement fetchall=[] ou fetchone=None pour simuler
    des résultats vides — le sentinel _UNSET laisse les defaults MagicMock.
    """
    conn = MagicMock()
    if fetchall is not _UNSET:
        conn.execute.return_value.fetchall.return_value = fetchall
    if fetchone is not _UNSET:
        conn.execute.return_value.fetchone.return_value = fetchone

    engine = MagicMock()
    for method in ("connect", "begin"):
        ctx = getattr(engine, method).return_value
        ctx.__enter__.return_value = conn
        ctx.__exit__.return_value = False
    return engine, conn


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── Logique pure ──────────────────────────────────────────────────────────────

class TestLabelEncoder:
    def test_all_crop_classes_encoded(self):
        encoded = _label_encoder.transform(CROP_CLASSES)
        assert list(encoded) == [0, 1, 2, 3, 4]

    def test_alphabetical_order(self):
        """corn=0, rice=1, soybean=2, sunflower=3, wheat=4."""
        assert _label_encoder.transform(["corn"])[0] == 0
        assert _label_encoder.transform(["rice"])[0] == 1
        assert _label_encoder.transform(["soybean"])[0] == 2
        assert _label_encoder.transform(["sunflower"])[0] == 3
        assert _label_encoder.transform(["wheat"])[0] == 4

    def test_inverse_transform_roundtrip(self):
        for crop in CROP_CLASSES:
            code = _label_encoder.transform([crop])
            assert _label_encoder.inverse_transform(code)[0] == crop

    def test_unknown_crop_raises(self):
        with pytest.raises(ValueError):
            _label_encoder.transform(["banana"])


class TestCropWaterNeeds:
    def test_all_crops_have_water_need(self):
        for crop in CROP_CLASSES:
            assert crop in CROP_WATER_NEEDS
            assert CROP_WATER_NEEDS[crop] > 0

    def test_rice_highest_need(self):
        assert CROP_WATER_NEEDS["rice"] == max(CROP_WATER_NEEDS.values())

    def test_sunflower_lowest_need(self):
        assert CROP_WATER_NEEDS["sunflower"] == min(CROP_WATER_NEEDS.values())


class TestIrrigationFormula:
    def test_deficit_computed_correctly(self):
        water_target = CROP_WATER_NEEDS["wheat"]  # 500
        rainfall_mm  = 320.0
        irrigation   = round(max(0.0, water_target - rainfall_mm), 1)
        assert irrigation == pytest.approx(180.0)

    def test_no_irrigation_when_rainfall_exceeds_need(self):
        water_target = CROP_WATER_NEEDS["corn"]  # 650
        rainfall_mm  = 700.0
        irrigation   = round(max(0.0, water_target - rainfall_mm), 1)
        assert irrigation == 0.0

    def test_irrigation_is_non_negative(self):
        for crop, need in CROP_WATER_NEEDS.items():
            for rainfall in [0.0, need / 2, need, need * 2]:
                irr = max(0.0, need - rainfall)
                assert irr >= 0.0, f"{crop} rainfall={rainfall}"


# ── Routes FastAPI ────────────────────────────────────────────────────────────

VALID_PAYLOAD = {
    "parcel_id":    1,
    "crop_type":    "wheat",
    "year":         2015,
    "soil_moisture": 0.42,
    "soil_ph":      6.5,
    "nitrogen_ppm": 110.0,
    "air_temp_c":   14.0,
    "humidity_pct": 65.0,
    "rainfall_mm":  320.0,
    "solar_rad_wm2": 180.0,
    "country":      "France",
}


class TestHealth:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_response_keys(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data

    def test_status_is_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_model_not_loaded_without_mlflow(self, client):
        with patch("api.main._model", None):
            data = client.get("/health").json()
        assert data["model_loaded"] is False


class TestPredictYield:
    def test_invalid_crop_returns_400(self, client):
        payload = {**VALID_PAYLOAD, "crop_type": "banana"}
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([3.5])
        with patch("api.main._load_model", return_value=mock_model), \
             patch("api.main._model", mock_model):
            response = client.post("/predict/yield", json=payload)
        assert response.status_code == 400
        assert "banana" in response.json()["detail"]

    def test_missing_required_field_returns_422(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "crop_type"}
        assert client.post("/predict/yield", json=payload).status_code == 422

    def test_model_unavailable_returns_503(self, client):
        with patch("api.main._model", None), \
             patch("api.main._load_model",
                   side_effect=HTTPException(status_code=503, detail="Model not available")):
            response = client.post("/predict/yield", json=VALID_PAYLOAD)
        assert response.status_code == 503

    def test_successful_prediction_structure(self, client):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([3.5])
        mock_engine, _ = _mock_engine(fetchone=None)

        with patch("api.main._load_model", return_value=mock_model), \
             patch("api.main._model", mock_model), \
             patch("api.main._model_version", "1"), \
             patch("api.main.engine", mock_engine):
            response = client.post("/predict/yield", json=VALID_PAYLOAD)

        assert response.status_code == 200
        data = response.json()
        assert data["parcel_id"] == VALID_PAYLOAD["parcel_id"]
        assert data["predicted_yield"] >= 0.0
        assert data["irrigation_rec_mm"] >= 0.0
        assert data["model_name"] == "yield-xgboost"
        assert data["model_version"] == "1"

    def test_irrigation_correct_for_wheat(self, client):
        """wheat: besoin=500mm, pluie=320mm → irrigation=180mm."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([3.5])
        mock_engine, _ = _mock_engine(fetchone=None)

        with patch("api.main._load_model", return_value=mock_model), \
             patch("api.main._model", mock_model), \
             patch("api.main._model_version", "1"), \
             patch("api.main.engine", mock_engine):
            response = client.post("/predict/yield", json=VALID_PAYLOAD)

        assert response.status_code == 200
        assert response.json()["irrigation_rec_mm"] == pytest.approx(180.0)

    def test_no_irrigation_when_rainfall_sufficient(self, client):
        """corn: besoin=650mm, pluie=700mm → irrigation=0."""
        payload = {**VALID_PAYLOAD, "crop_type": "corn", "rainfall_mm": 700.0}
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([5.5])
        mock_engine, _ = _mock_engine(fetchone=None)

        with patch("api.main._load_model", return_value=mock_model), \
             patch("api.main._model", mock_model), \
             patch("api.main._model_version", "1"), \
             patch("api.main.engine", mock_engine):
            response = client.post("/predict/yield", json=payload)

        assert response.status_code == 200
        assert response.json()["irrigation_rec_mm"] == 0.0


class TestParcels:
    def test_list_parcels_returns_list(self, client):
        row = MockRow(id=1, name="Parcelle 1", crop_type="wheat",
                      area_ha=100.0, soil_type="loam",
                      country="France", latitude=46.2, longitude=2.2)
        mock_engine, _ = _mock_engine(fetchall=[row])

        with patch("api.main.engine", mock_engine):
            response = client.get("/parcels")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["crop_type"] == "wheat"

    def test_list_parcels_empty(self, client):
        mock_engine, _ = _mock_engine(fetchall=[])
        with patch("api.main.engine", mock_engine):
            response = client.get("/parcels")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_parcel_not_found(self, client):
        mock_engine, _ = _mock_engine(fetchone=None)
        with patch("api.main.engine", mock_engine):
            response = client.get("/parcels/9999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_parcel_found(self, client):
        row = MockRow(id=1, name="Parcelle 1", crop_type="wheat",
                      area_ha=100.0, soil_type="loam",
                      country="France", latitude=46.2, longitude=2.2)
        mock_engine, _ = _mock_engine(fetchone=row)
        with patch("api.main.engine", mock_engine):
            response = client.get("/parcels/1")
        assert response.status_code == 200
        assert response.json()["id"] == 1


class TestPesticide:
    def test_countries_returns_list(self, client):
        rows = [MockRow(area="France"), MockRow(area="Germany")]
        mock_engine, _ = _mock_engine(fetchall=rows)
        with patch("api.main.engine", mock_engine):
            response = client.get("/pesticide/countries")
        assert response.status_code == 200
        assert response.json() == ["France", "Germany"]

    def test_history_not_found_returns_404(self, client):
        mock_engine, _ = _mock_engine(fetchall=[])
        with patch("api.main.engine", mock_engine):
            response = client.get("/pesticide/history/UnknownCountry")
        assert response.status_code == 404

    def test_history_found(self, client):
        row = MockRow(year=2010, value_tonnes=1000.0, yoy_growth_pct=5.0,
                      ma5_tonnes=950.0, cagr_5y_pct=3.0,
                      value_normalized=0.5, pct_vs_global_avg=10.0)
        mock_engine, _ = _mock_engine(fetchall=[row])
        with patch("api.main.engine", mock_engine):
            response = client.get("/pesticide/history/France")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["year"] == 2010
        assert data[0]["value_tonnes"] == 1000.0
