"""Фаза 04 -- сборка feature pipeline и анализ признаков.

Содержит:
  - `make_feature_pipeline()` -- sklearn-Pipeline (builder -> grouper -> OHE+scale),
    переносимый в прод и встраиваемый в `pipeline.joblib` (фаза 05);
  - анализ мультиколлинеарности (VIF) и предварительной значимости (MI/χ^2);
  - генерацию `reports/feature_dictionary.md`, `reports/vif.json`,
    `reports/feature_importance_prelim.json`.

Запуск (анализ + артефакты):  `python3 -m src.features.build`

⚠️ Анти-лик: pipeline принимает только `config.MODEL_INPUT_COLUMNS` (session-уровень);
поля hits ни на одном шаге не используются.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src import config
from src.features.transformers import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RareCategoryGrouper,
    SessionFeatureBuilder,
)

# Размер выборки для анализа фичей (VIF/MI/χ^2) -- для скорости; ранжирование стабильно.
ANALYSIS_SAMPLE = 120_000
VIF_THRESHOLD = 10.0


def make_feature_pipeline() -> Pipeline:
    """Собрать переносимый в прод feature pipeline (без модели)."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encode = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("features", SessionFeatureBuilder()),
        ("group", RareCategoryGrouper(top_n_by_column=config.TOP_N_BY_COLUMN)),
        ("encode", encode),
    ])


# --------------------------------------------------------------------------- #
# Анализ
# --------------------------------------------------------------------------- #
def compute_vif(numeric_df: pd.DataFrame) -> dict[str, float]:
    """VIF по числовым признакам (после медианного импьютинга)."""
    imputed = SimpleImputer(strategy="median").fit_transform(numeric_df)
    mat = np.asarray(imputed, dtype="float64")
    mat = np.column_stack([np.ones(mat.shape[0]), mat])  # константа для корректного VIF
    vif = {}
    for i, col in enumerate(numeric_df.columns, start=1):
        try:
            vif[col] = round(float(variance_inflation_factor(mat, i)), 4)
        except Exception:
            vif[col] = float("nan")
    return vif


def compute_importance(pipeline: Pipeline, sample: pd.DataFrame, y: np.ndarray) -> dict:
    """MI (все фичи) + χ^2 (категориальные OHE), с агрегацией по исходным фичам."""
    pipeline.fit(sample, y)
    matrix = pipeline.transform(sample)
    names = pipeline.named_steps["encode"].get_feature_names_out()

    mi = mutual_info_classif(matrix, y, discrete_features=False, random_state=config.SEED)
    per_column = {n: round(float(m), 6) for n, m in zip(names, mi)}

    # χ^2 по неотрицательному блоку категориальных OHE (имена начинаются с 'cat__').
    cat_idx = [i for i, n in enumerate(names) if n.startswith("cat__")]
    chi_scores = {}
    if cat_idx:
        chi_stat, _ = chi2(matrix[:, cat_idx], y)
        chi_scores = {names[i]: round(float(s), 4) for i, s in zip(cat_idx, chi_stat)}

    # Агрегация MI по исходной фиче (сумма MI её OHE-столбцов).
    agg: dict[str, float] = {}
    for n, m in per_column.items():
        base = n.split("__", 1)[1] if "__" in n else n
        if base not in NUMERIC_FEATURES:  # категориальная: 'utm_medium_organic' -> 'utm_medium'
            for c in CATEGORICAL_FEATURES:
                if base.startswith(c + "_") or base == c:
                    base = c
                    break
        agg[base] = round(agg.get(base, 0.0) + m, 6)
    agg_sorted = dict(sorted(agg.items(), key=lambda x: -x[1]))

    return {
        "n_output_features": int(matrix.shape[1]),
        "mi_by_original_feature": agg_sorted,
        "mi_top_columns": dict(sorted(per_column.items(), key=lambda x: -x[1])[:30]),
        "chi2_top_columns": dict(sorted(chi_scores.items(), key=lambda x: -x[1])[:30]),
    }


# --------------------------------------------------------------------------- #
# Словарь фич
# --------------------------------------------------------------------------- #
FEATURE_SPEC: list[dict] = [
    {"name": "visit_hour", "type": "числовой", "formula": "visit_time // 3600 (0–23)", "source": "visit_time", "encoding": "median+scale"},
    {"name": "visit_day", "type": "числовой", "formula": "день месяца", "source": "visit_date", "encoding": "median+scale"},
    {"name": "visit_number_log", "type": "числовой", "formula": "log1p(clip(visit_number, ≤50))", "source": "visit_number", "encoding": "median+scale"},
    {"name": "is_weekend", "type": "бинарный", "formula": "day_of_week ≥ 5", "source": "visit_date", "encoding": "scale"},
    {"name": "is_organic", "type": "бинарный", "formula": "utm_medium ∈ {organic, referral, (none)}", "source": "utm_medium", "encoding": "scale"},
    {"name": "is_social", "type": "бинарный", "formula": "utm_source ∈ SOCIAL_SOURCES", "source": "utm_source", "encoding": "scale"},
    {"name": "is_first_visit", "type": "бинарный", "formula": "visit_number == 1", "source": "visit_number", "encoding": "scale"},
    {"name": "has_utm_campaign", "type": "бинарный", "formula": "utm_campaign известен", "source": "utm_campaign", "encoding": "scale"},
    {"name": "has_utm_keyword", "type": "бинарный", "formula": "utm_keyword известен", "source": "utm_keyword", "encoding": "scale"},
    {"name": "has_utm_adcontent", "type": "бинарный", "formula": "utm_adcontent известен", "source": "utm_adcontent", "encoding": "scale"},
    {"name": "is_russia", "type": "бинарный", "formula": "geo_country == Russia", "source": "geo_country", "encoding": "scale"},
    {"name": "is_moscow", "type": "бинарный", "formula": "geo_city == Moscow", "source": "geo_city", "encoding": "scale"},
    {"name": "is_spb", "type": "бинарный", "formula": "geo_city == Saint Petersburg", "source": "geo_city", "encoding": "scale"},
    {"name": "screen_w", "type": "числовой", "formula": "ширина из WxH", "source": "device_screen_resolution", "encoding": "median+scale"},
    {"name": "screen_h", "type": "числовой", "formula": "высота из WxH", "source": "device_screen_resolution", "encoding": "median+scale"},
    {"name": "screen_is_valid", "type": "бинарный", "formula": "разрешение распознано", "source": "device_screen_resolution", "encoding": "scale"},
    {"name": "utm_medium", "type": "категориальный", "formula": "топ-15 + other", "source": "utm_medium", "encoding": "OHE(ignore)"},
    {"name": "utm_source", "type": "категориальный", "formula": "топ-20 + other", "source": "utm_source", "encoding": "OHE(ignore)"},
    {"name": "utm_campaign", "type": "категориальный", "formula": "топ-20 + other", "source": "utm_campaign", "encoding": "OHE(ignore)"},
    {"name": "utm_adcontent", "type": "категориальный", "formula": "топ-20 + other", "source": "utm_adcontent", "encoding": "OHE(ignore)"},
    {"name": "utm_keyword", "type": "категориальный", "formula": "топ-25 + other", "source": "utm_keyword", "encoding": "OHE(ignore)"},
    {"name": "device_category", "type": "категориальный", "formula": "mobile/desktop/tablet", "source": "device_category", "encoding": "OHE(ignore)"},
    {"name": "device_os", "type": "категориальный", "formula": "топ-12 + other", "source": "device_os", "encoding": "OHE(ignore)"},
    {"name": "device_brand", "type": "категориальный", "formula": "топ-15 + other", "source": "device_brand", "encoding": "OHE(ignore)"},
    {"name": "device_browser", "type": "категориальный", "formula": "топ-15 + other", "source": "device_browser", "encoding": "OHE(ignore)"},
    {"name": "geo_city", "type": "категориальный", "formula": "топ-20 + other", "source": "geo_city", "encoding": "OHE(ignore)"},
    {"name": "visit_dow", "type": "категориальный", "formula": "день недели 0–6", "source": "visit_date", "encoding": "OHE(ignore)"},
    {"name": "visit_month", "type": "категориальный", "formula": "месяц 5–12", "source": "visit_date", "encoding": "OHE(ignore)"},
    {"name": "visit_daypart", "type": "категориальный", "formula": "night/morning/day/evening", "source": "visit_time", "encoding": "OHE(ignore)"},
]

# Признаки, исключённые из-за мультиколлинеарности/избыточности (документируется).
DROPPED_FEATURES: list[dict] = [
    {"name": "is_paid", "reason": "= 1 − is_organic (идеальная коллинеарность)"},
    {"name": "is_mobile", "reason": "поглощается OHE(device_category)"},
    {"name": "screen_area", "reason": "= screen_w · screen_h (определённая коллинеарность)"},
    {"name": "screen_aspect", "reason": "= screen_w / screen_h: VIF>10 (инфляция screen_w) -> исключён"},
    {"name": "geo_country", "reason": "кардинальность 166; заменён бинарным is_russia"},
    {"name": "device_model", "reason": "удалён в фазе 02 (99.12% NaN)"},
]


def write_feature_dictionary(vif: dict, importance: dict, vif_flags: list[str]) -> None:
    L: list[str] = []
    L.append("# Словарь признаков (фаза 04)\n")
    L.append("Все признаки -- **только session-уровень** (контракт API, анти-лик).\n")
    L.append("## Признаки\n")
    L.append("| Признак | Тип | Формула | Источник | Кодирование | MI |")
    L.append("|---|---|---|---|---|---:|")
    mi_orig = importance["mi_by_original_feature"]
    for spec in FEATURE_SPEC:
        mi = mi_orig.get(spec["name"], 0.0)
        L.append(f"| `{spec['name']}` | {spec['type']} | {spec['formula']} | "
                 f"`{spec['source']}` | {spec['encoding']} | {mi:.4f} |")
    L.append("")
    L.append("## Исключённые признаки (мультиколлинеарность/избыточность)\n")
    L.append("| Признак | Причина |")
    L.append("|---|---|")
    for d in DROPPED_FEATURES:
        L.append(f"| `{d['name']}` | {d['reason']} |")
    L.append("")
    L.append(f"## Мультиколлинеарность (VIF, порог {VIF_THRESHOLD})\n")
    L.append("| Числовой признак | VIF |")
    L.append("|---|---:|")
    for col, v in sorted(vif.items(), key=lambda x: -x[1]):
        flag = " ⚠️" if col in vif_flags else ""
        L.append(f"| `{col}` | {v}{flag} |")
    if vif_flags:
        L.append(f"\n> ⚠️ VIF > {VIF_THRESHOLD}: {', '.join(vif_flags)} -- "
                 "оставлены осознанно (бинарные индикаторы; модели регуляризованы/древесные); "
                 "определённо коллинеарный `screen_area` исключён по построению.")
    else:
        L.append(f"\n> Все VIF ≤ {VIF_THRESHOLD} после исключения `screen_area`.")
    L.append("")
    L.append("## Топ-признаки по Mutual Information (агрегировано)\n")
    for name, mi in list(mi_orig.items())[:15]:
        L.append(f"- `{name}`: MI={mi:.4f}")
    L.append("")
    config.FEATURE_DICTIONARY_MD.write_text("\n".join(L), encoding="utf-8")


def main() -> dict:
    config.ensure_dirs()
    print("[04] Загрузка датасета (выборка для анализа) …")
    df = pd.read_parquet(config.DATASET_PARQUET, columns=list(config.MODEL_INPUT_COLUMNS) + ["target"])
    sample = df.sample(n=min(ANALYSIS_SAMPLE, len(df)), random_state=config.SEED).reset_index(drop=True)
    y = sample["target"].to_numpy()
    X = sample[list(config.MODEL_INPUT_COLUMNS)]

    print("[04] Построение инженерных признаков …")
    engineered = SessionFeatureBuilder().fit_transform(X)

    print("[04] VIF (мультиколлинеарность) …")
    vif = compute_vif(engineered[NUMERIC_FEATURES])
    vif_flags = [c for c, v in vif.items() if v == v and v > VIF_THRESHOLD]

    print("[04] Значимость (MI/χ^2) …")
    pipeline = make_feature_pipeline()
    importance = compute_importance(pipeline, X, y)

    config.VIF_JSON.write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "threshold": VIF_THRESHOLD, "vif": vif, "flagged": vif_flags,
         "dropped_by_design": DROPPED_FEATURES},
        ensure_ascii=False, indent=2), encoding="utf-8")
    config.FEATURE_IMPORTANCE_JSON.write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "n_analysis_rows": int(len(sample)), **importance},
        ensure_ascii=False, indent=2), encoding="utf-8")
    write_feature_dictionary(vif, importance, vif_flags)

    print(f"[04] выход pipeline: {importance['n_output_features']} фич; "
          f"VIF>{VIF_THRESHOLD}: {vif_flags or 'нет'}")
    print(f"[04] -> {config.FEATURE_DICTIONARY_MD}, {config.VIF_JSON}, {config.FEATURE_IMPORTANCE_JSON}")
    return {"vif": vif, "vif_flags": vif_flags, "importance": importance}


if __name__ == "__main__":
    main()
