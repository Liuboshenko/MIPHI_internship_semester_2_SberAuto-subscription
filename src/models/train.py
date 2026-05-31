"""Фаза 05 -- обучение, честная оценка, интерпретация, сериализация.

Конвейер:
  1. Разбиение по времени: последние 20% дат -> test (имитация прода); внутри train
     случайный stratified-val (10%) для ранней остановки CatBoost и подбора порога.
  2. Бейзлайны: Dummy(prior) и LogReg на 2 признаках (utm_medium + device_category).
  3. Модели: LogReg(full), RandomForest, CatBoost(нативные категории) -- class_weight.
  4. Метрики: ROC-AUC (главная) + PR-AUC; сравнительная таблица; кривые/confusion.
  5. Порог: F1-оптимальный на val. Интерпретация: CatBoost importance + SHAP +
     permutation importance по сырым полям.
  6. Финальная модель (CatBoost) переобучается на полном train -> models/pipeline.joblib
     + metadata.json + metrics.json.

Запуск:  `python3 -m src.models.train`

⚠️ Анти-лик: модель видит только session-признаки (через feature pipeline); test не
участвует в обучении/подборе; все fit -- внутри Pipeline/на train.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from catboost import CatBoostClassifier, Pool  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

import catboost  # noqa: E402

from src import config  # noqa: E402
from src.features.build import make_feature_pipeline  # noqa: E402
from src.features.transformers import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RareCategoryGrouper,
    SessionFeatureBuilder,
)

INPUT = list(config.MODEL_INPUT_COLUMNS)
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Параллелизм умеренный: на 64-ядерной машине n_jobs=-1 порождал loky-воркеров с
# утечкой семафоров и сегфолтом при GC. Ограничиваем до стабильных значений.
N_JOBS = 8
CATBOOST_THREADS = 16


def _log(msg: str) -> None:
    print(f"[05] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Данные и разбиение
# --------------------------------------------------------------------------- #
def load_split() -> dict:
    # visit_date уже входит в INPUT (MODEL_INPUT_COLUMNS).
    df = pd.read_parquet(config.DATASET_PARQUET, columns=INPUT + ["target"])
    X_all, y_all = df[INPUT], df["target"].to_numpy()

    # Stratified random split (отклонение от time-split задокументировано:
    # todo/improvements_validation_split_random_vs_time.md). Анти-лик сохранён:
    # любой fit (фичи/импьютинг/модель) происходит только на train.
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y_all)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=config.VAL_SIZE, random_state=config.SEED, stratify=y_train)
    X_fit = X_fit.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)

    _log(f"split (stratified random): train={len(X_train):,} "
         f"(fit={len(X_fit):,}, val={len(X_val):,}), test={len(X_test):,}")
    _log(f"CR train={y_train.mean()*100:.3f}% / test={y_test.mean()*100:.3f}%")
    return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
               X_fit=X_fit, y_fit=y_fit, X_val=X_val, y_val=y_val,
               split="stratified_random", test_size=config.TEST_SIZE)


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #
def eval_proba(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 5),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 5),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 5),
        "precision": round(float(tp / (tp + fp)) if (tp + fp) else 0.0, 5),
        "recall": round(float(tp / (tp + fn)) if (tp + fn) else 0.0, 5),
        "threshold": round(float(threshold), 5),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    # thr короче на 1; берём индекс по f1[:-1]
    idx = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[idx]) if len(thr) else 0.5


# --------------------------------------------------------------------------- #
# Модели
# --------------------------------------------------------------------------- #
def build_logreg_base() -> Pipeline:
    """Бейзлайн: LogReg на 2 признаках (utm_medium + device_category)."""
    encode = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["utm_medium", "device_category"])],
        remainder="drop",
    )
    return Pipeline([
        ("features", SessionFeatureBuilder()),
        ("encode", encode),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
    ])


def tune_logreg_C(data: dict) -> float:
    """Лёгкий подбор C логрега по ROC-AUC (на подвыборке train_fit, 3-fold)."""
    from sklearn.model_selection import GridSearchCV

    n = min(250_000, len(data["X_fit"]))
    Xs = data["X_fit"].sample(n=n, random_state=config.SEED)
    ys = data["y_fit"][Xs.index.to_numpy()]
    pre = make_feature_pipeline()
    Xt = pre.fit_transform(Xs, ys)
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
        param_grid={"C": [0.3, 1.0, 3.0]},
        scoring="roc_auc",
        cv=StratifiedKFold(3, shuffle=True, random_state=config.SEED),
        n_jobs=N_JOBS,
    )
    grid.fit(Xt, ys)
    _log(f"LogReg C tuning: best C={grid.best_params_['C']} (CV AUC={grid.best_score_:.4f})")
    return float(grid.best_params_["C"])


def build_logreg_full(C: float) -> Pipeline:
    pipe = make_feature_pipeline()
    pipe.steps.append(("model", LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="lbfgs", C=C)))
    return pipe


def build_random_forest() -> Pipeline:
    pipe = make_feature_pipeline()
    pipe.steps.append(("model", RandomForestClassifier(
        n_estimators=120, max_depth=14, min_samples_leaf=100, max_samples=0.3,
        class_weight="balanced", n_jobs=N_JOBS, random_state=config.SEED)))
    return pipe


def catboost_preprocessor() -> Pipeline:
    return Pipeline([
        ("features", SessionFeatureBuilder()),
        ("group", RareCategoryGrouper(top_n_by_column=config.TOP_N_BY_COLUMN)),
    ])


def make_catboost(iterations: int) -> CatBoostClassifier:
    # max_ctr_complexity=1 отключает комбинации категориальных признаков -- без него
    # на 1.35М×13 кат-фич обучение раздувается до десятков минут. boosting_type='Plain'
    # -- быстрый режим для больших данных.
    return CatBoostClassifier(
        iterations=iterations, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
        loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
        cat_features=CATEGORICAL_FEATURES, max_ctr_complexity=1, boosting_type="Plain",
        random_seed=config.SEED, thread_count=CATBOOST_THREADS, verbose=0,
        allow_writing_files=False,
    )


# --------------------------------------------------------------------------- #
# Графики и интерпретация
# --------------------------------------------------------------------------- #
def plot_roc_pr(curves: dict, y_test: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, proba in curves.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
        prec, rec, _ = precision_recall_curve(y_test, proba)
        axes[1].plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, proba):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC-кривые (test)")
    axes[0].legend(fontsize=8)
    axes[1].axhline(y_test.mean(), color="k", ls="--", lw=1, label=f"baseline={y_test.mean():.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR-кривые (test)")
    axes[1].legend(fontsize=8)
    fig.savefig(config.FIGURES_DIR / "roc_pr_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_test: np.ndarray, proba: np.ndarray, threshold: float) -> None:
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Предсказание", ylabel="Факт",
           title=f"Confusion matrix (порог={threshold:.3f})")
    fig.colorbar(im, ax=ax)
    fig.savefig(config.FIGURES_DIR / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(y_test: np.ndarray, proba: np.ndarray) -> None:
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label="CatBoost")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="идеальная")
    ax.set(xlabel="Средняя предсказанная вероятность", ylabel="Доля позитивов",
           title="Калибровочная кривая (test)")
    ax.legend()
    fig.savefig(config.FIGURES_DIR / "calibration_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def interpret_catboost(model: CatBoostClassifier, pre: Pipeline, X_sample: pd.DataFrame,
                       y_sample: np.ndarray) -> dict:
    """CatBoost feature importance + SHAP summary (по 29 инженерным признакам)."""
    Xt = pre.transform(X_sample)
    pool = Pool(Xt, y_sample, cat_features=CATEGORICAL_FEATURES)

    fi = model.get_feature_importance(pool)
    importance = dict(sorted(
        {f: round(float(v), 4) for f, v in zip(FEATURE_ORDER, fi)}.items(),
        key=lambda x: -x[1]))

    # CatBoost feature importance bar
    top = list(importance.items())[:20]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([k for k, _ in top][::-1], [v for _, v in top][::-1], color="#1f77b4")
    ax.set(xlabel="Importance (PredictionValuesChange)", title="CatBoost feature importance (топ-20)")
    fig.savefig(config.FIGURES_DIR / "catboost_feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # SHAP summary (нативные ShapValues CatBoost)
    try:
        import shap

        shap_vals = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
        shap.summary_plot(shap_vals, Xt, feature_names=FEATURE_ORDER, show=False, max_display=20)
        plt.title("SHAP summary (test-выборка)")
        plt.savefig(config.FIGURES_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
        plt.close("all")
        shap_ok = True
    except Exception as exc:  # pragma: no cover
        _log(f"SHAP summary пропущен: {exc}")
        shap_ok = False

    return {"catboost_importance": importance, "shap_plot": shap_ok}


def permutation_by_raw_field(pipeline: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    """Permutation importance по СЫРЫМ полям визита (интерпретируемо для бизнеса)."""
    n = min(15_000, len(X_test))
    sampled = X_test.sample(n=n, random_state=config.SEED)
    idx = sampled.index.to_numpy()          # X_test реиндексирован 0..n-1 -> позиции в y_test
    Xs = sampled.reset_index(drop=True)
    ys = y_test[idx]
    res = permutation_importance(
        pipeline, Xs, ys, scoring="roc_auc", n_repeats=5,
        random_state=config.SEED, n_jobs=1)
    out = {col: round(float(m), 5) for col, m in zip(INPUT, res.importances_mean)}
    return dict(sorted(out.items(), key=lambda x: -x[1]))


# --------------------------------------------------------------------------- #
# Главный конвейер
# --------------------------------------------------------------------------- #
def main() -> dict:
    config.ensure_dirs()
    t0 = time.time()
    data = load_split()
    y_test, y_val = data["y_test"], data["y_val"]

    metrics_table: dict[str, dict] = {}
    test_curves: dict[str, np.ndarray] = {}
    default_thr = 0.5

    # --- Бейзлайн 1: Dummy(prior) ---
    prior = float(data["y_fit"].mean())
    proba_dummy = np.full(len(y_test), prior)
    metrics_table["dummy_prior"] = eval_proba(y_test, proba_dummy, default_thr)
    _log(f"dummy_prior: AUC={metrics_table['dummy_prior']['roc_auc']}")

    # --- Бейзлайн 2: LogReg на 2 признаках ---
    lr_base = build_logreg_base().fit(data["X_fit"], data["y_fit"])
    proba = lr_base.predict_proba(data["X_test"])[:, 1]
    metrics_table["logreg_base"] = eval_proba(y_test, proba, default_thr)
    test_curves["LogReg-base"] = proba
    _log(f"logreg_base: AUC={metrics_table['logreg_base']['roc_auc']}")

    # --- LogReg full (с подбором C) ---
    best_C = tune_logreg_C(data)
    lr_full = build_logreg_full(best_C).fit(data["X_fit"], data["y_fit"])
    proba = lr_full.predict_proba(data["X_test"])[:, 1]
    metrics_table["logreg_full"] = eval_proba(y_test, proba, default_thr)
    test_curves["LogReg-full"] = proba
    _log(f"logreg_full: AUC={metrics_table['logreg_full']['roc_auc']}")

    # --- RandomForest ---
    rf = build_random_forest().fit(data["X_fit"], data["y_fit"])
    proba = rf.predict_proba(data["X_test"])[:, 1]
    metrics_table["random_forest"] = eval_proba(y_test, proba, default_thr)
    test_curves["RandomForest"] = proba
    _log(f"random_forest: AUC={metrics_table['random_forest']['roc_auc']}")

    # --- CatBoost (нативные категории, ранняя остановка по val) ---
    cb_pre = catboost_preprocessor()
    Xfit_t = cb_pre.fit_transform(data["X_fit"])
    Xval_t = cb_pre.transform(data["X_val"])
    cb = make_catboost(iterations=1500)
    cb.fit(Xfit_t, data["y_fit"], eval_set=(Xval_t, y_val),
           early_stopping_rounds=60, verbose=0)
    best_iter = int(cb.get_best_iteration()) + 1
    proba_cb = cb.predict_proba(cb_pre.transform(data["X_test"]))[:, 1]
    metrics_table["catboost"] = eval_proba(y_test, proba_cb, default_thr)
    test_curves["CatBoost"] = proba_cb
    _log(f"catboost: AUC={metrics_table['catboost']['roc_auc']} (best_iter={best_iter})")

    # --- Порог по val (на модели train_fit) ---
    proba_val = cb.predict_proba(Xval_t)[:, 1]
    threshold = best_f1_threshold(y_val, proba_val)
    _log(f"threshold (F1-opt на val) = {threshold:.4f}")

    # --- Финальная модель: CatBoost переобучаем на полном train -> pipeline.joblib ---
    final_pipeline = Pipeline([
        ("features", SessionFeatureBuilder()),
        ("group", RareCategoryGrouper(top_n_by_column=config.TOP_N_BY_COLUMN)),
        ("model", make_catboost(iterations=best_iter)),
    ])
    final_pipeline.fit(data["X_train"], data["y_train"])
    proba_final = final_pipeline.predict_proba(data["X_test"])[:, 1]
    final_metrics = eval_proba(y_test, proba_final, threshold)
    metrics_table["catboost_final"] = final_metrics
    test_curves["CatBoost-final"] = proba_final
    _log(f"FINAL catboost (refit на train): AUC={final_metrics['roc_auc']}, "
         f"PR-AUC={final_metrics['pr_auc']}, F1={final_metrics['f1']}")

    # --- Графики ---
    plot_roc_pr({k: v for k, v in test_curves.items() if k != "CatBoost"}, y_test)
    plot_confusion(y_test, proba_final, threshold)
    plot_calibration(y_test, proba_final)

    # --- Интерпретация (на финальной модели) ---
    final_model: CatBoostClassifier = final_pipeline.named_steps["model"]
    final_pre = Pipeline(final_pipeline.steps[:-1])
    sample_idx = data["X_test"].sample(n=min(8000, len(data["X_test"])), random_state=config.SEED).index
    interp = interpret_catboost(final_model, final_pre,
                                data["X_test"].loc[sample_idx],
                                y_test[sample_idx.to_numpy()])
    _log("permutation importance по сырым полям …")
    perm = permutation_by_raw_field(final_pipeline, data["X_test"], y_test)

    # --- Сериализация ---
    joblib.dump(final_pipeline, config.PIPELINE_JOBLIB)
    _log(f"-> {config.PIPELINE_JOBLIB}")

    versions = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "catboost": catboost.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "CatBoostClassifier (native categories)",
        "THRESHOLD": round(float(threshold), 5),
        "target_actions": sorted(config.TARGET_ACTIONS),
        "input_columns": INPUT,
        "feature_order": FEATURE_ORDER,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "catboost_best_iteration": best_iter,
        "validation_split": data["split"],
        "test_size": data["test_size"],
        "test_metrics": final_metrics,
        "library_versions": versions,
    }
    config.METADATA_JSON.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"-> {config.METADATA_JSON}")

    metrics_payload = {
        "generated_at": metadata["generated_at"],
        "main_metric": "roc_auc",
        "acceptance": {"roc_auc_threshold": 0.65,
                       "passed": bool(final_metrics["roc_auc"] > 0.65)},
        "comparison": metrics_table,
        "final_model": "catboost_final",
        "threshold": round(float(threshold), 5),
        "catboost_importance": interp["catboost_importance"],
        "permutation_importance_raw_fields": perm,
        "figures": ["roc_pr_curves.png", "confusion_matrix.png", "calibration_curve.png",
                    "catboost_feature_importance.png",
                    *(["shap_summary.png"] if interp["shap_plot"] else [])],
    }
    config.METRICS_JSON.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"-> {config.METRICS_JSON}")

    # --- Smoke-тест инференса + латентность ---
    one = data["X_test"].iloc[[0]]
    t = time.time()
    p = float(final_pipeline.predict_proba(one)[:, 1][0])
    latency_ms = (time.time() - t) * 1000
    _log(f"smoke inference: proba={p:.4f}, pred={int(p >= threshold)}, latency={latency_ms:.1f} ms")
    _log(f"acceptance ROC-AUC>0.65: {'PASS' if final_metrics['roc_auc'] > 0.65 else 'FAIL'} "
         f"(AUC={final_metrics['roc_auc']})")
    _log(f"всего за {time.time() - t0:.0f} c")
    return metrics_payload


if __name__ == "__main__":
    main()
