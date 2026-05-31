"""Инжиниринг признаков уровня сессии (фаза 04).

Публичный интерфейс:
  - `SessionFeatureBuilder` -- сырые поля визита -> инженерные признаки;
  - `RareCategoryGrouper` -- топ-N + 'other' для высоко-кардинальных категорий;
  - `make_feature_pipeline` -- sklearn-Pipeline (builder -> grouper -> ColumnTransformer);
  - списки признаков `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`.
"""

from src.features.transformers import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RareCategoryGrouper,
    SessionFeatureBuilder,
)

# `make_feature_pipeline` импортируется напрямую из `src.features.build`
# (не реэкспортируется здесь, чтобы `python -m src.features.build` не ловил
# RuntimeWarning о частичной инициализации пакета).

__all__ = [
    "SessionFeatureBuilder",
    "RareCategoryGrouper",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
]
