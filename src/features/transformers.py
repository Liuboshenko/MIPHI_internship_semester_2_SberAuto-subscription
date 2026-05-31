"""sklearn-трансформеры инжиниринга признаков (фаза 04).

Все преобразования -- детерминированы и переносимы в прод: `SessionFeatureBuilder`
принимает сырые поля визита (как в `ga_sessions` / в JSON-запросе API) и выдаёт
инженерные признаки. Нормализация категорий повторяет фазу 02 (идемпотентно для уже
очищенного датасета), чем гарантируется парность train=serve.

⚠️ Анти-лик: используются ТОЛЬКО session-поля (`MODEL_INPUT_COLUMNS`); ни одно поле
hits не читается и не порождается.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src import config
from src.data.clean import normalize_category

# Итоговые наборы признаков (после анализа мультиколлинеарности, фаза 04).
# Исключены по построению/VIF (см. reports/vif.json, reports/feature_dictionary.md):
#   screen_area  = screen_w·screen_h (определённая коллинеарность);
#   screen_aspect = screen_w/screen_h (VIF>10 -> инфляция screen_w; устранено).
NUMERIC_FEATURES: list[str] = [
    "visit_hour", "visit_day", "visit_number_log",
    "is_weekend", "is_organic", "is_social", "is_first_visit",
    "has_utm_campaign", "has_utm_keyword", "has_utm_adcontent",
    "is_russia", "is_moscow", "is_spb",
    "screen_w", "screen_h", "screen_is_valid",
]
CATEGORICAL_FEATURES: list[str] = [
    "utm_medium", "utm_source", "utm_campaign", "utm_adcontent", "utm_keyword",
    "device_category", "device_os", "device_brand", "device_browser",
    "geo_city", "visit_dow", "visit_month", "visit_daypart",
]


def _normalized(series: pd.Series, col: str) -> pd.Series:
    """Нормализовать категориальную колонку как в фазе 02 (-> string)."""
    is_medium = col == "utm_medium"
    preserve = config.ORGANIC_MEDIUMS if is_medium else frozenset()
    return normalize_category(series, lowercase=is_medium, preserve=preserve).astype("string")


def _to_seconds(series: pd.Series) -> pd.Series:
    """visit_time -> секунды от полуночи (вход: Int64 секунды ИЛИ строка 'HH:MM:SS')."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_timedelta(series.astype("string"), errors="coerce").dt.total_seconds()


def _to_datetime(series: pd.Series) -> pd.Series:
    """visit_date -> datetime (вход: datetime ИЛИ строка 'YYYY-MM-DD')."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series.astype("string"), format="%Y-%m-%d", errors="coerce")


def _daypart(hour: float) -> str:
    if pd.isna(hour):
        return config.UNKNOWN
    h = int(hour)
    if h < 6:
        return "night"
    if h < 12:
        return "morning"
    if h < 18:
        return "day"
    return "evening"


class SessionFeatureBuilder(BaseEstimator, TransformerMixin):
    """Сырые поля визита -> инженерные признаки (без обучения на таргете)."""

    def fit(self, X: pd.DataFrame, y=None) -> "SessionFeatureBuilder":  # noqa: D401, N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = pd.DataFrame(X).reset_index(drop=True)
        out = pd.DataFrame(index=X.index)

        # --- Время из даты ---
        dt = _to_datetime(X["visit_date"])
        dow = dt.dt.dayofweek
        out["visit_day"] = dt.dt.day.astype("float64")
        out["is_weekend"] = (dow >= 5).fillna(False).astype("int8")

        # --- Время суток ---
        seconds = _to_seconds(X["visit_time"])
        hour = (seconds // 3600).clip(lower=0, upper=23)
        out["visit_hour"] = hour.astype("float64")

        # --- Трафик ---
        medium = _normalized(X["utm_medium"], "utm_medium")
        source = _normalized(X["utm_source"], "utm_source")
        out["is_organic"] = medium.isin(config.ORGANIC_MEDIUMS).astype("int8")
        out["is_social"] = source.isin(config.SOCIAL_SOURCES).astype("int8")

        campaign = _normalized(X["utm_campaign"], "utm_campaign")
        keyword = _normalized(X["utm_keyword"], "utm_keyword")
        adcontent = _normalized(X["utm_adcontent"], "utm_adcontent")
        out["has_utm_campaign"] = (campaign != config.UNKNOWN).astype("int8")
        out["has_utm_keyword"] = (keyword != config.UNKNOWN).astype("int8")
        out["has_utm_adcontent"] = (adcontent != config.UNKNOWN).astype("int8")

        # --- Устройство: разбор разрешения экрана WxH ---
        resolution = _normalized(X["device_screen_resolution"], "device_screen_resolution")
        parts = resolution.str.extract(r"^(\d+)x(\d+)$")
        screen_w = pd.to_numeric(parts[0], errors="coerce")
        screen_h = pd.to_numeric(parts[1], errors="coerce")
        out["screen_w"] = screen_w.astype("float64")
        out["screen_h"] = screen_h.astype("float64")
        out["screen_is_valid"] = screen_w.notna().astype("int8")

        # --- Гео ---
        country = _normalized(X["geo_country"], "geo_country")
        city = _normalized(X["geo_city"], "geo_city")
        out["is_russia"] = (country == "Russia").astype("int8")
        out["is_moscow"] = (city == "Moscow").astype("int8")
        out["is_spb"] = (city == "Saint Petersburg").astype("int8")

        # --- Поведение ---
        visit_number = pd.to_numeric(X["visit_number"], errors="coerce")
        out["is_first_visit"] = (visit_number == 1).fillna(False).astype("int8")
        out["visit_number_log"] = np.log1p(visit_number.clip(upper=config.VISIT_NUMBER_CLIP))

        # --- Категориальные (нормализованные) ---
        out["utm_medium"] = medium
        out["utm_source"] = source
        out["utm_campaign"] = campaign
        out["utm_adcontent"] = adcontent
        out["utm_keyword"] = keyword
        out["device_category"] = _normalized(X["device_category"], "device_category")
        out["device_os"] = _normalized(X["device_os"], "device_os")
        out["device_brand"] = _normalized(X["device_brand"], "device_brand")
        out["device_browser"] = _normalized(X["device_browser"], "device_browser")
        out["geo_city"] = city
        out["visit_dow"] = dow.astype("Int64").astype("string")
        out["visit_month"] = dt.dt.month.astype("Int64").astype("string")
        out["visit_daypart"] = hour.map(_daypart).astype("string")

        # Категориальные NaN -> 'unknown' (устойчивость на проде).
        for col in CATEGORICAL_FEATURES:
            out[col] = out[col].fillna(config.UNKNOWN).astype("object")

        return out[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    def get_feature_names_out(self, input_features=None):  # noqa: D401
        return np.asarray(NUMERIC_FEATURES + CATEGORICAL_FEATURES, dtype=object)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Топ-N значений категории + 'other' (учится на train; снижает размерность OHE)."""

    def __init__(self, top_n_by_column: dict[str, int] | None = None, other_token: str = "other"):
        self.top_n_by_column = top_n_by_column
        self.other_token = other_token

    def fit(self, X: pd.DataFrame, y=None) -> "RareCategoryGrouper":  # noqa: N803
        mapping = self.top_n_by_column or {}
        self.top_values_: dict[str, set] = {}
        for col, n in mapping.items():
            if col in X.columns:
                vc = X[col].astype("string").value_counts().head(n)
                self.top_values_[col] = set(vc.index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = pd.DataFrame(X).copy()
        for col, keep in self.top_values_.items():
            s = X[col].astype("string")
            X[col] = s.where(s.isin(keep), other=self.other_token).astype("object")
        return X
