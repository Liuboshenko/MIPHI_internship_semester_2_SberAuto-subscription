"""Тесты фазы 02 -- очистка/предобработка.

Две группы:
  - артефактные: читают processed/dataset.parquet + cleaning_stats.json;
  - юнит-тесты на мок-данных: проверяют логику нормализации детерминированно
    (в т.ч. сохранение '(none)' в utm_medium и схлопывание мусора в 'unknown').
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import config
from src.data import clean as clean_mod

EXPECTED_ROWS = 1_860_042
CATEGORY_PREFIXES = ("utm_", "device_", "geo_")


def _require(path) -> None:
    if not path.exists():
        pytest.skip(f"Нет артефакта {path}. Сначала: python3 -m src.data.clean")


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    _require(config.DATASET_PARQUET)
    return pd.read_parquet(config.DATASET_PARQUET)


@pytest.fixture(scope="module")
def stats() -> dict:
    _require(config.CLEANING_STATS_JSON)
    return json.loads(config.CLEANING_STATS_JSON.read_text(encoding="utf-8"))


# --- Артефактные проверки ---------------------------------------------------

def test_rows_and_session_unique(dataset: pd.DataFrame) -> None:
    assert len(dataset) == EXPECTED_ROWS
    assert dataset["session_id"].is_unique


def test_target_no_nan(dataset: pd.DataFrame) -> None:
    assert dataset["target"].notna().all()
    assert set(dataset["target"].unique()) <= {0, 1}


def test_device_model_dropped(dataset: pd.DataFrame) -> None:
    assert "device_model" not in dataset.columns


def test_no_hits_columns(dataset: pd.DataFrame) -> None:
    leaked = [c for c in dataset.columns if c.startswith(("hit_", "event_"))]
    assert leaked == [], f"Поля hits протекли: {leaked}"


def test_date_and_category_dtypes(dataset: pd.DataFrame) -> None:
    assert pd.api.types.is_datetime64_any_dtype(dataset["visit_date"])
    cat_cols = [c for c in dataset.columns if c.startswith(CATEGORY_PREFIXES)]
    for col in cat_cols:
        assert isinstance(dataset[col].dtype, pd.CategoricalDtype), f"{col} не category"


def test_no_nan_in_categories(dataset: pd.DataFrame) -> None:
    """Все NaN категорий должны быть заменены на 'unknown'."""
    cat_cols = [c for c in dataset.columns if c.startswith(CATEGORY_PREFIXES)]
    for col in cat_cols:
        assert dataset[col].isna().sum() == 0, f"{col} содержит NaN"


def test_utm_medium_preserves_none(dataset: pd.DataFrame) -> None:
    """Анти-регресс: '(none)' в utm_medium сохранён (органика)."""
    values = set(dataset["utm_medium"].astype("string").unique())
    assert "(none)" in values
    assert {"organic", "referral"} <= values


def test_unknown_present_in_high_nan_columns(dataset: pd.DataFrame) -> None:
    for col in ("device_os", "utm_keyword"):
        assert "unknown" in set(dataset[col].astype("string").unique())


def test_nan_table_before_after(stats: dict) -> None:
    assert stats["nan_before"] and stats["nan_after"]
    # device_model был в "до" и исчез в "после"
    assert "device_model" in stats["nan_before"]
    assert "device_model" not in stats["nan_after"]


# --- Юнит-тесты логики на мок-данных ---------------------------------------

def test_normalize_category_junk_to_unknown() -> None:
    s = pd.Series(["Chrome", "(not set)", "", None, "nan", "(not provided)"])
    out = clean_mod.normalize_category(s, lowercase=False, preserve=frozenset())
    assert list(out.astype("string")) == [
        "Chrome", "unknown", "unknown", "unknown", "unknown", "unknown",
    ]


def test_normalize_utm_medium_keeps_none_lowercases() -> None:
    s = pd.Series(["CPC", "Organic", "(none)", "(not set)", None])
    out = clean_mod.normalize_category(s, lowercase=True, preserve=config.ORGANIC_MEDIUMS)
    assert list(out.astype("string")) == ["cpc", "organic", "(none)", "unknown", "unknown"]


def test_clean_on_mock() -> None:
    sessions = pd.DataFrame({
        "session_id": ["s1", "s2", "s3"],
        "client_id": ["c1", "c2", "c3"],
        "visit_date": pd.Categorical(["2021-06-01", "2021-06-02", "2021-06-03"]),
        "visit_time": pd.Categorical(["10:00:00", "23:59:59", "00:00:30"]),
        "visit_number": [1, 5, 2],
        "utm_source": pd.Categorical(["AbC", "(none)", "XyZ"]),
        "utm_medium": pd.Categorical(["organic", "(none)", "CPC"]),
        "utm_campaign": pd.Categorical(["k1", "(not set)", "k2"]),
        "utm_adcontent": pd.Categorical(["a1", None, "a2"]),
        "utm_keyword": pd.Categorical([None, None, "kw"]),
        "device_category": pd.Categorical(["mobile", "desktop", "mobile"]),
        "device_os": pd.Categorical([None, "Android", None]),
        "device_brand": pd.Categorical(["Apple", None, "Samsung"]),
        "device_model": pd.Categorical([None, None, None]),
        "device_screen_resolution": pd.Categorical(["360x800", "1920x1080", "broken"]),
        "device_browser": pd.Categorical(["Safari", "Chrome", "Chrome"]),
        "geo_country": pd.Categorical(["Russia", "Russia", "Russia"]),
        "geo_city": pd.Categorical(["Moscow", "(not set)", "Kazan"]),
    })
    target = pd.DataFrame({"session_id": ["s1", "s3"], "target": [1, 1]})

    df, stats = clean_mod.clean(sessions, target)

    assert "device_model" not in df.columns
    assert df["target"].tolist() == [1, 0, 1]  # s2 без таргета -> 0
    assert pd.api.types.is_datetime64_any_dtype(df["visit_date"])
    assert df["visit_time"].tolist() == [36000, 86399, 30]  # секунды от полуночи
    # utm_medium: '(none)' сохранён, регистр нижний
    assert df["utm_medium"].astype("string").tolist() == ["organic", "(none)", "cpc"]
    # utm_source: '(none)' -> 'unknown' (мусор для не-medium)
    assert df["utm_source"].astype("string").tolist() == ["AbC", "unknown", "XyZ"]
    assert stats["screen_resolution_invalid"] == 1
    assert stats["device_model_dropped"] is True
