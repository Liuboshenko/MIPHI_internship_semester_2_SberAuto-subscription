"""Тесты фазы 06 -- FastAPI-сервис.

Контракт /health, /predict, /predict_batch; валидация (422); согласованность
API == pipeline.joblib; latency ≤ 3с; детерминированная сборка фич.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from api import model as model_api
from src import config

VALID_VISIT = {
    "utm_source": "ZpYIoDJMcFzVoPFsHGJL",
    "utm_medium": "banner",
    "device_category": "mobile",
    "device_os": "Android",
    "device_brand": "Samsung",
    "device_screen_resolution": "360x800",
    "device_browser": "Chrome",
    "geo_country": "Russia",
    "geo_city": "Moscow",
    "utm_campaign": "LEoPHuyFvzoNfnzGgfcd",
    "visit_number": 2,
    "visit_date": "2021-06-15",
    "visit_time": "14:30:00",
}


def _require_model() -> None:
    if not config.PIPELINE_JOBLIB.exists():
        pytest.skip("Нет models/pipeline.joblib. Сначала: python3 -m src.models.train")


@pytest.fixture(scope="module")
def client():
    _require_model()
    from api.main import app

    with TestClient(app) as test_client:  # context -> срабатывает lifespan (загрузка модели)
        yield test_client


@pytest.fixture(scope="module")
def artifacts():
    _require_model()
    return model_api.load_artifacts()


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert isinstance(body["model_version"], str) and body["model_version"]


def test_predict_contract(client) -> None:
    r = client.post("/predict", json=VALID_VISIT)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"prediction", "probability", "model_version"}
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["model_version"], str)


def test_missing_required_field_422(client) -> None:
    payload = {k: v for k, v in VALID_VISIT.items() if k != "utm_source"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_batch(client) -> None:
    r = client.post("/predict_batch", json={"items": [VALID_VISIT, VALID_VISIT]})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert all(set(item) == {"prediction", "probability", "model_version"} for item in body)


def test_api_matches_pipeline(client, artifacts) -> None:
    """Предсказание API == прямое предсказание pipeline.joblib на тех же данных."""
    pipeline, metadata = artifacts
    direct = model_api.predict(pipeline, metadata, [VALID_VISIT])[0]
    api_body = client.post("/predict", json=VALID_VISIT).json()
    assert abs(api_body["probability"] - direct["probability"]) < 1e-6
    assert api_body["prediction"] == direct["prediction"]


def test_latency_under_3s(client) -> None:
    client.post("/predict", json=VALID_VISIT)  # прогрев
    t = time.perf_counter()
    r = client.post("/predict", json=VALID_VISIT)
    elapsed = time.perf_counter() - t
    assert r.status_code == 200
    assert elapsed < 3.0, f"latency {elapsed:.3f}s > 3s"


def test_feature_assembly_deterministic(artifacts) -> None:
    """Предобработка/сборка фич детерминирована и даёт стабильный контракт."""
    df = model_api.to_dataframe([VALID_VISIT])
    assert list(df.columns) == list(config.MODEL_INPUT_COLUMNS)
    assert len(df) == 1
    pipeline, metadata = artifacts
    p1 = model_api.predict(pipeline, metadata, [VALID_VISIT])[0]
    p2 = model_api.predict(pipeline, metadata, [VALID_VISIT])[0]
    assert p1 == p2  # детерминированность


def test_optional_fields_default(client) -> None:
    """Опциональные поля можно опустить -- сервис не падает (-> 'unknown')."""
    minimal = {"utm_source": "x", "utm_medium": "organic", "device_category": "desktop"}
    r = client.post("/predict", json=minimal)
    assert r.status_code == 200
    assert 0.0 <= r.json()["probability"] <= 1.0
