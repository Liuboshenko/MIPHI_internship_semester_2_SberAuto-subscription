"""Центральная конфигурация проекта "СберАвтоподписка".

Здесь живут все пути (относительные от корня репозитория), random seed и
определение таргета. Модуль намеренно "лёгкий": при импорте он только вычисляет
пути и константы, не делая I/O, -- чтобы его можно было дёшево импортировать из
тестов и любого модуля пайплайна.

Это черновик конфигурации фазы 01 (00 §4 "Финальное определение таргета",
00 §7 "Целевая структура репозитория"). По мере прохождения фаз сюда будут
добавляться новые константы (например, `SOCIAL_SOURCES` в фазе 03).
"""

from __future__ import annotations

from pathlib import Path

# --- Пути -------------------------------------------------------------------
# config.py лежит в src/, поэтому корень проекта -- на один уровень выше.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Сырые данные (в корне проекта; НЕ модифицируются -- read-only).
GA_SESSIONS_CSV: Path = PROJECT_ROOT / "ga_sessions.csv"
GA_HITS_CSV: Path = PROJECT_ROOT / "ga_hits.csv"

# Производные данные.
DATA_DIR: Path = PROJECT_ROOT / "data"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# Артефакты модели и отчётов.
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Конкретные файлы-артефакты фазы 01.
SESSIONS_PARQUET: Path = INTERIM_DIR / "sessions.parquet"
TARGET_BY_SESSION_PARQUET: Path = INTERIM_DIR / "target_by_session.parquet"
INGESTION_REPORT_MD: Path = REPORTS_DIR / "ingestion_report.md"
INGESTION_STATS_JSON: Path = REPORTS_DIR / "ingestion_stats.json"

# Конкретные файлы-артефакты фазы 02.
DATASET_PARQUET: Path = PROCESSED_DIR / "dataset.parquet"
CLEANING_REPORT_MD: Path = REPORTS_DIR / "cleaning_report.md"
CLEANING_STATS_JSON: Path = REPORTS_DIR / "cleaning_stats.json"

# Конкретные файлы-артефакты фазы 03.
EDA_FINDINGS_MD: Path = REPORTS_DIR / "eda_findings.md"
EDA_STATS_JSON: Path = REPORTS_DIR / "eda_stats.json"

# Конкретные файлы-артефакты фазы 04.
FEATURE_DICTIONARY_MD: Path = REPORTS_DIR / "feature_dictionary.md"
VIF_JSON: Path = REPORTS_DIR / "vif.json"
FEATURE_IMPORTANCE_JSON: Path = REPORTS_DIR / "feature_importance_prelim.json"

# Конкретные файлы-артефакты фазы 05.
PIPELINE_JOBLIB: Path = MODELS_DIR / "pipeline.joblib"
METADATA_JSON: Path = MODELS_DIR / "metadata.json"
METRICS_JSON: Path = MODELS_DIR / "metrics.json"

# Схема валидации (фаза 05). Отклонение от time-split задокументировано в
# todo/improvements_validation_split_random_vs_time.md: time-split недостижим под
# порог ROC-AUC>0.65 из-за сдвига распределения в декабре. Используем stratified
# random split (стандарт для propensity-модели); анти-лик сохранён (fit только на train).
TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.10

# --- Воспроизводимость ------------------------------------------------------
SEED: int = 42

# Колонки-идентификаторы: читаются как строки (у client_id смешанные типы).
ID_COLUMNS: tuple[str, ...] = ("session_id", "client_id")

# --- Определение таргета (00 §4) -------------------------------------------
# Ровно 8 фактически присутствующих в данных действий-конверсий.
# `sub_open_chat` из ТЗ в данных ОТСУТСТВУЕТ (0 событий) -> НЕ включаем.
# Чат-вовлечённость (`start_chat` и т. п.) -- не конверсия -> НЕ включаем.
TARGET_ACTIONS: frozenset[str] = frozenset({
    "sub_car_claim_click",
    "sub_car_claim_submit_click",
    "sub_open_dialog_click",
    "sub_custom_question_submit_click",
    "sub_call_number_click",
    "sub_callback_submit_click",
    "sub_submit_success",
    "sub_car_request_submit_click",
})


# --- Очистка/нормализация (00 §6, фаза 02) ---------------------------------
# Колонка удаляется при очистке (99.12% NaN -- нет сигнала).
DROP_COLUMNS: tuple[str, ...] = ("device_model",)

# Единый маркер пропуска/мусора в категориальных колонках.
UNKNOWN: str = "unknown"

# Маркеры пропуска/мусора, схлопываемые в `UNKNOWN`.
JUNK_TOKENS: frozenset[str] = frozenset({
    "(not set)", "(none)", "(not provided)", "", "nan",
})

# ⚠️ Исключение: в `utm_medium` значение `'(none)'` -- это direct/none-трафик,
# относимый к ОРГАНИКЕ (00 §6 / 04 §2), поэтому в `utm_medium` оно НЕ схлопывается.
ORGANIC_MEDIUMS: frozenset[str] = frozenset({"organic", "referral", "(none)"})


# --- EDA-производные определения (фаза 03) ---------------------------------
# ID соцсетей среди зашифрованных `utm_source` (00 §6 "Соцсети"). Набор выявлен в
# EDA: все 6 присутствуют в данных с заметным объёмом и социальными `utm_medium`
# (cpc/cpm/cpa/blogger_*/smm/post/referral).
SOCIAL_SOURCES: frozenset[str] = frozenset({
    "QxAxdyPLuQMEcrdZWdWb",
    "MvfHsxITijuriZxsqZqt",
    "ISrKoXQCxqqYvAZICvjs",
    "IZEXUFLARCUMynmHNBGo",
    "PlbkrSYoHuZBWfYjYnfw",
    "gVRrcxiDQubJiljoTbGm",
})

# Пороги "топ-N + other" для группировки редких категорий (вход в фазу 04).
# Обоснование порогов -- в reports/eda_findings.md (покрытие объёма топ-N).
TOP_N_BY_COLUMN: dict[str, int] = {
    "utm_medium": 15,
    "utm_source": 20,
    "utm_campaign": 20,
    "utm_adcontent": 20,
    "utm_keyword": 25,
    "geo_city": 20,
    "device_brand": 15,
    "device_browser": 15,
    "device_os": 12,
}


# --- Контракт входа модели/API (фаза 04/06) --------------------------------
# Сырые поля визита (как в ga_sessions), которые принимает feature pipeline и API.
# Это session-уровень: НЕТ session_id/client_id/target и НЕТ полей hits (анти-лик).
# `device_model` исключён (удалён в фазе 02).
MODEL_INPUT_COLUMNS: tuple[str, ...] = (
    "visit_date", "visit_time", "visit_number",
    "utm_source", "utm_medium", "utm_campaign", "utm_adcontent", "utm_keyword",
    "device_category", "device_os", "device_brand",
    "device_screen_resolution", "device_browser",
    "geo_country", "geo_city",
)

# Верхний клип visit_number перед log1p (гашение тяжёлого хвоста, nunique=537).
VISIT_NUMBER_CLIP: int = 50


def ensure_dirs() -> None:
    """Создать производные директории, если их ещё нет (идемпотентно)."""
    for directory in (INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
