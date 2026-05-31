"""Тесты фазы 04 -- инжиниринг признаков.

Юнит-тесты (на выборке из processed/dataset.parquet и на мок-визите):
  - feature pipeline трансформирует одиночный визит без ошибок;
  - стабильная размерность выхода (1 строка == батч);
  - неизвестные категории обрабатываются (handle_unknown='ignore');
  - среди признаков нет полей hits / session-исхода (анти-лик);
  - train=serve парность (сырой JSON-визит -> ожидаемые признаки).
Артефактные: VIF-отчёт построен, мультиколлинеарность устранена.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features.build import make_feature_pipeline
from src.features.transformers import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SessionFeatureBuilder,
)

INPUT = list(config.MODEL_INPUT_COLUMNS)


def _require(path) -> None:
    if not path.exists():
        pytest.skip(f"Нет артефакта {path}. Сначала: python3 -m src.features.build")


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    if not config.DATASET_PARQUET.exists():
        pytest.skip("Нет dataset.parquet. Сначала: python3 -m src.data.clean")
    df = pd.read_parquet(config.DATASET_PARQUET, columns=INPUT + ["target"])
    return df.sample(8000, random_state=config.SEED).reset_index(drop=True)


@pytest.fixture(scope="module")
def fitted_pipeline(sample: pd.DataFrame):
    return make_feature_pipeline().fit(sample[INPUT], sample["target"].to_numpy())


def test_builder_output_columns_and_no_leak(sample: pd.DataFrame) -> None:
    out = SessionFeatureBuilder().fit_transform(sample[INPUT])
    assert list(out.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert not any(c.startswith(("hit_", "event_")) for c in out.columns)
    for forbidden in ("target", "session_id", "client_id", "hit_page_path", "event_action"):
        assert forbidden not in out.columns


def test_single_row_transforms(fitted_pipeline, sample: pd.DataFrame) -> None:
    one = fitted_pipeline.transform(sample[INPUT].iloc[[0]])
    assert one.shape[0] == 1
    assert np.isfinite(one).all()


def test_stable_output_dimension(fitted_pipeline, sample: pd.DataFrame) -> None:
    full = fitted_pipeline.transform(sample[INPUT])
    one = fitted_pipeline.transform(sample[INPUT].iloc[[0]])
    assert one.shape[1] == full.shape[1]


def test_unknown_categories_ignored(fitted_pipeline, sample: pd.DataFrame) -> None:
    row = sample[INPUT].iloc[[0]].copy()
    row["utm_source"] = "NEVER_SEEN_SOURCE"
    row["geo_city"] = "Atlantis"
    row["device_category"] = "hologram"
    out = fitted_pipeline.transform(row)
    assert out.shape[1] == fitted_pipeline.transform(sample[INPUT].iloc[[0]]).shape[1]
    assert np.isfinite(out).all()


def test_train_serve_parity_raw_visit() -> None:
    raw = pd.DataFrame([{
        "visit_date": "2021-06-15", "visit_time": "14:30:00", "visit_number": 1,
        "utm_source": "abc", "utm_medium": "(none)", "utm_campaign": "(not set)",
        "utm_adcontent": None, "utm_keyword": None, "device_category": "mobile",
        "device_os": "Android", "device_brand": "Samsung",
        "device_screen_resolution": "360x800", "device_browser": "Chrome",
        "geo_country": "Russia", "geo_city": "Moscow",
    }])
    r = SessionFeatureBuilder().fit_transform(raw).iloc[0]
    assert r["is_organic"] == 1          # (none) ∈ органика
    assert r["is_russia"] == 1 and r["is_moscow"] == 1
    assert r["visit_hour"] == 14
    assert r["is_first_visit"] == 1
    assert r["has_utm_campaign"] == 0    # (not set) -> unknown
    assert r["screen_w"] == 360 and r["screen_h"] == 800
    assert r["screen_is_valid"] == 1


def test_vif_artifact_no_multicollinearity() -> None:
    _require(config.VIF_JSON)
    data = json.loads(config.VIF_JSON.read_text(encoding="utf-8"))
    assert data["flagged"] == [], f"Остались VIF>{data['threshold']}: {data['flagged']}"
    assert all(v <= data["threshold"] for v in data["vif"].values())


def test_feature_dictionary_exists() -> None:
    _require(config.FEATURE_DICTIONARY_MD)
    assert config.FEATURE_DICTIONARY_MD.stat().st_size > 0
