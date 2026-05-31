"""Тесты фазы 05 -- модель и метрики.

Проверяют артефакты обучения (`python3 -m src.models.train`):
  - pipeline.joblib загружается; predict_proba ∈ [0,1];
  - независимая перепроверка ROC-AUC на воспроизводимом hold-out > 0.65;
  - metadata.json содержит THRESHOLD, порядок фич, TARGET_ACTIONS, версии библиотек;
  - metrics.json: сравнение ≥3 моделей + бейзлайнов, acceptance passed.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src import config
from src.features.transformers import CATEGORICAL_FEATURES, NUMERIC_FEATURES

INPUT = list(config.MODEL_INPUT_COLUMNS)


def _require(path) -> None:
    if not path.exists():
        pytest.skip(f"Нет артефакта {path}. Сначала: python3 -m src.models.train")


@pytest.fixture(scope="module")
def pipeline():
    _require(config.PIPELINE_JOBLIB)
    return joblib.load(config.PIPELINE_JOBLIB)


@pytest.fixture(scope="module")
def metadata() -> dict:
    _require(config.METADATA_JSON)
    return json.loads(config.METADATA_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def metrics() -> dict:
    _require(config.METRICS_JSON)
    return json.loads(config.METRICS_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def holdout():
    """Воспроизвести тот же hold-out (тот же SEED/test_size, что в train.load_split)."""
    _require(config.DATASET_PARQUET)
    df = pd.read_parquet(config.DATASET_PARQUET, columns=INPUT + ["target"])
    X_all, y_all = df[INPUT], df["target"].to_numpy()
    _, X_test, _, y_test = train_test_split(
        X_all, y_all, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y_all)
    return X_test.reset_index(drop=True), y_test


def test_pipeline_loads(pipeline) -> None:
    assert hasattr(pipeline, "predict_proba")


def test_predict_proba_range(pipeline, holdout) -> None:
    X_test, _ = holdout
    proba = pipeline.predict_proba(X_test.iloc[:2000])[:, 1]
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    assert proba.std() > 0  # предсказания не константные


def test_holdout_roc_auc_above_065(pipeline, holdout) -> None:
    X_test, y_test = holdout
    n = min(80_000, len(X_test))
    idx = np.random.RandomState(config.SEED).choice(len(X_test), n, replace=False)
    proba = pipeline.predict_proba(X_test.iloc[idx])[:, 1]
    auc = roc_auc_score(y_test[idx], proba)
    assert auc > 0.65, f"hold-out ROC-AUC={auc:.4f} ≤ 0.65"


def test_metadata_fields(metadata) -> None:
    assert "THRESHOLD" in metadata and 0.0 <= metadata["THRESHOLD"] <= 1.0
    assert metadata["feature_order"] == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert sorted(metadata["target_actions"]) == sorted(config.TARGET_ACTIONS)
    assert len(metadata["target_actions"]) == 8
    for lib in ("scikit_learn", "catboost", "pandas", "numpy"):
        assert lib in metadata["library_versions"]


def test_metadata_auc_above_065(metadata) -> None:
    assert metadata["test_metrics"]["roc_auc"] > 0.65


def test_metrics_comparison(metrics) -> None:
    comp = metrics["comparison"]
    for model in ("dummy_prior", "logreg_base", "logreg_full", "random_forest", "catboost"):
        assert model in comp, f"нет модели {model} в сравнении"
    assert metrics["acceptance"]["passed"] is True
    # лучшая модель должна заметно превосходить dummy
    assert comp["catboost"]["roc_auc"] > comp["dummy_prior"]["roc_auc"] + 0.1
