"""Загрузка модели и сборка фич из запроса (фаза 06).

Чистые, тестируемые функции инференса -- общие для FastAPI и Streamlit-фоллбэка.
Контракт входа = `config.MODEL_INPUT_COLUMNS` (session-уровень, анти-лик). Тот же
`pipeline.joblib`, что обучен в фазе 05 -> нет расхождения train/serve.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src import config


def load_artifacts(pipeline_path: Path | None = None, metadata_path: Path | None = None):
    """Загрузить обученный Pipeline и metadata (один раз -- на старте сервиса)."""
    pipeline = joblib.load(pipeline_path or config.PIPELINE_JOBLIB)
    meta_path = Path(metadata_path or config.METADATA_JSON)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return pipeline, metadata


def to_dataframe(payloads: list[dict]) -> pd.DataFrame:
    """Список визитов (dict) -> DataFrame с колонками MODEL_INPUT_COLUMNS.

    Недостающие поля -> None (feature pipeline сам приводит их к 'unknown'/NaN).
    """
    cols = list(config.MODEL_INPUT_COLUMNS)
    rows = [{col: payload.get(col) for col in cols} for payload in payloads]
    return pd.DataFrame(rows, columns=cols)


def predict(pipeline, metadata: dict, payloads: list[dict]) -> list[dict]:
    """Предсказать для пачки визитов: [{prediction, probability, model_version}]."""
    df = to_dataframe(payloads)
    proba = pipeline.predict_proba(df)[:, 1]
    threshold = float(metadata["THRESHOLD"])
    version = str(metadata.get("generated_at", "unknown"))
    return [
        {
            "prediction": int(p >= threshold),
            "probability": round(float(p), 6),
            "model_version": version,
        }
        for p in proba
    ]
