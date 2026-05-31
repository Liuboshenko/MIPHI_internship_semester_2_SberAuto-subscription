"""Фаза 03 -- EDA и статистические гипотезы.

Из `processed/dataset.parquet` строит:
  - набор PNG-графиков в `reports/figures/` (баланс классов, распределения,
    CR-разрезы по utm/device/geo/visit_number/времени, Cramér's V heatmap);
  - формально проверяет ≥3 статистические гипотезы (χ^2/z-тест долей);
  - `reports/eda_findings.md` (инсайты + гипотезы фичей + пороги группировки);
  - `reports/eda_stats.json` (машиночитаемая сводка: CR-разрезы, гипотезы, фигуры).

Запуск:  `python3 -m src.eda.run_eda`

Анти-лик: используются только session-поля + target; ни одно поле hits не читается.
Производные (visit_hour, is_organic, is_social…) -- тоже из session-полей.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")  # headless: без дисплея
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.proportion import proportions_ztest  # noqa: E402

from src import config  # noqa: E402

ALPHA = 0.05
DPI = 120

# Колонки, читаемые для EDA (только session-уровень + target).
_EDA_COLUMNS = [
    "visit_date", "visit_time", "visit_number",
    "utm_source", "utm_medium",
    "device_category", "device_os", "device_browser", "device_brand",
    "geo_country", "geo_city", "target",
]


# --------------------------------------------------------------------------- #
# Подготовка данных
# --------------------------------------------------------------------------- #
def _daypart(hour: int) -> str:
    if hour < 6:
        return "ночь"
    if hour < 12:
        return "утро"
    if hour < 18:
        return "день"
    return "вечер"


def build_eda_frame(path=None) -> pd.DataFrame:
    """Загрузить датасет и добавить EDA-производные (всё из session-полей)."""
    df = pd.read_parquet(path or config.DATASET_PARQUET, columns=_EDA_COLUMNS)
    df["visit_hour"] = (df["visit_time"].astype("float") // 3600).astype("Int64").astype("int16")
    df["visit_dow"] = df["visit_date"].dt.dayofweek.astype("int8")  # 0=Пн
    df["is_weekend"] = df["visit_dow"] >= 5
    df["visit_daypart"] = df["visit_hour"].map(_daypart).astype("category")
    df["is_organic"] = df["utm_medium"].astype("string").isin(config.ORGANIC_MEDIUMS)
    df["is_paid"] = ~df["is_organic"]
    df["is_social"] = df["utm_source"].astype("string").isin(config.SOCIAL_SOURCES)
    df["is_first_visit"] = df["visit_number"] == 1
    return df


# --------------------------------------------------------------------------- #
# Утилиты
# --------------------------------------------------------------------------- #
def cr_by(df: pd.DataFrame, col: str, min_count: int = 0) -> pd.DataFrame:
    """CR (mean target) и объём по категориям колонки `col`."""
    g = df.groupby(col, observed=True)["target"].agg(count="count", cr="mean")
    g = g[g["count"] >= min_count].sort_values("cr", ascending=False)
    return g


def cramers_v(confusion: np.ndarray) -> float:
    """Cramér's V с поправкой на смещение (Bergsma)."""
    chi2 = stats.chi2_contingency(confusion, correction=False)[0]
    n = confusion.sum()
    if n == 0:
        return float("nan")
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return math.sqrt(phi2corr / denom) if denom > 0 else float("nan")


def _group_top_n(series: pd.Series, n: int) -> pd.Series:
    """Топ-N значений + 'other' (для управляемых таблиц Cramér's V)."""
    s = series.astype("string")
    top = s.value_counts().head(n).index
    return s.where(s.isin(top), other="other")


def _save(fig, name: str, generated: list[str]) -> None:
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    generated.append(name)


# --------------------------------------------------------------------------- #
# Графики
# --------------------------------------------------------------------------- #
def _cr_barplot(table: pd.DataFrame, title: str, name: str, generated: list[str],
                overall_cr: float | None = None, annotate_count: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(table) + 1.5)))
    ax.barh(table.index.astype(str), table["cr"] * 100, color="#1f77b4")
    ax.invert_yaxis()
    ax.set_xlabel("CR, %")
    ax.set_title(title)
    if overall_cr is not None:
        ax.axvline(overall_cr * 100, color="red", ls="--", lw=1, label=f"общий CR {overall_cr*100:.2f}%")
        ax.legend(loc="lower right")
    if annotate_count:
        for y, (_, row) in enumerate(table.iterrows()):
            ax.text(row["cr"] * 100, y, f"  n={int(row['count']):,}", va="center", fontsize=8)
    _save(fig, name, generated)


def make_figures(df: pd.DataFrame, overall_cr: float, generated: list[str]) -> None:
    # 1. Баланс классов
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["target"].value_counts().sort_index()
    ax.bar(["0 (нет конверсии)", "1 (конверсия)"], counts.values, color=["#999999", "#d62728"])
    ax.set_yscale("log")
    ax.set_ylabel("Число визитов (log)")
    ax.set_title(f"Баланс классов: CR={overall_cr*100:.3f}% (дисбаланс)")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    _save(fig, "class_balance.png", generated)

    # 2. CR по utm_medium (топ-15 по объёму)
    med = cr_by(df, "utm_medium", min_count=1000)
    _cr_barplot(med.head(15), "CR по utm_medium (n≥1000)", "cr_by_utm_medium.png",
                generated, overall_cr)

    # 3. Органика vs платный
    org = df.groupby("is_organic", observed=True)["target"].agg(count="count", cr="mean")
    org_tbl = pd.DataFrame({
        "count": [int(org.loc[False, "count"]), int(org.loc[True, "count"])],
        "cr": [float(org.loc[False, "cr"]), float(org.loc[True, "cr"])],
    }, index=["платный", "органика"])
    _cr_barplot(org_tbl, "CR: органика vs платный трафик", "cr_organic_vs_paid.png",
                generated, overall_cr)

    # 4. CR по device_category
    dev = cr_by(df, "device_category")
    _cr_barplot(dev, "CR по device_category", "cr_by_device_category.png", generated, overall_cr)

    # 5. Соцсети vs прочие
    soc = df.groupby("is_social", observed=True)["target"].agg(count="count", cr="mean")
    soc_tbl = pd.DataFrame({
        "count": [int(soc.loc[False, "count"]), int(soc.loc[True, "count"])],
        "cr": [float(soc.loc[False, "cr"]), float(soc.loc[True, "cr"])],
    }, index=["прочие", "соцсети"])
    _cr_barplot(soc_tbl, "CR: соцсети vs прочие источники", "cr_social_vs_other.png",
                generated, overall_cr)

    # 6. Новые vs возвратные
    nv = df.groupby("is_first_visit", observed=True)["target"].agg(count="count", cr="mean")
    nv_tbl = pd.DataFrame({
        "count": [int(nv.loc[False, "count"]), int(nv.loc[True, "count"])],
        "cr": [float(nv.loc[False, "cr"]), float(nv.loc[True, "cr"])],
    }, index=["возвратные (>1)", "новые (=1)"])
    _cr_barplot(nv_tbl, "CR: новые vs возвратные визиты", "cr_new_vs_returning.png",
                generated, overall_cr)

    # 7. CR по бакетам visit_number
    bins = [0, 1, 2, 5, 10, np.inf]
    labels = ["1", "2", "3-5", "6-10", "11+"]
    vb = df.assign(vn_bucket=pd.cut(df["visit_number"], bins=bins, labels=labels))
    vbt = vb.groupby("vn_bucket", observed=True)["target"].agg(count="count", cr="mean")
    _cr_barplot(vbt, "CR по числу визитов (visit_number)", "cr_by_visit_number_bucket.png",
                generated, overall_cr, annotate_count=True)

    # 8. CR по часу
    hr = df.groupby("visit_hour", observed=True)["target"].agg(count="count", cr="mean")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hr.index, hr["cr"] * 100, marker="o", color="#2ca02c")
    ax.axhline(overall_cr * 100, color="red", ls="--", lw=1, label=f"общий CR {overall_cr*100:.2f}%")
    ax.set_xlabel("Час визита")
    ax.set_ylabel("CR, %")
    ax.set_title("CR по часу визита")
    ax.set_xticks(range(0, 24))
    ax.legend()
    _save(fig, "cr_by_hour.png", generated)

    # 9. CR по дню недели
    dow_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    dw = df.groupby("visit_dow", observed=True)["target"].agg(count="count", cr="mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([dow_names[i] for i in dw.index], dw["cr"] * 100, color="#9467bd")
    ax.axhline(overall_cr * 100, color="red", ls="--", lw=1)
    ax.set_ylabel("CR, %")
    ax.set_title("CR по дню недели")
    _save(fig, "cr_by_dow.png", generated)

    # 10. CR по гео-региону (векторно: Москва / СПб / Россия-прочее / не Россия)
    region = np.where(df["geo_country"].astype("string") != "Russia", "не Россия",
              np.where(df["geo_city"].astype("string") == "Moscow", "Москва",
              np.where(df["geo_city"].astype("string") == "Saint Petersburg", "Санкт-Петербург",
                       "Россия (прочее)")))
    gtbl = df.assign(region=region).groupby("region", observed=True)["target"].agg(count="count", cr="mean")
    _cr_barplot(gtbl, "CR по гео-региону", "cr_by_geo_region.png", generated, overall_cr)

    # 11. Распределение visit_number (исходное и log1p)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df["visit_number"].clip(upper=30), bins=30, color="#8c564b")
    axes[0].set_title("visit_number (клип 30)")
    axes[0].set_xlabel("visit_number")
    axes[1].hist(np.log1p(df["visit_number"]), bins=40, color="#8c564b")
    axes[1].set_title("log1p(visit_number)")
    axes[1].set_xlabel("log1p(visit_number)")
    _save(fig, "visit_number_distribution.png", generated)

    # 12. Топ-20 utm_source по объёму
    top_src = df["utm_source"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_src.index.astype(str), top_src.values, color="#17becf")
    ax.invert_yaxis()
    ax.set_xlabel("Число визитов")
    ax.set_title("Топ-20 utm_source по объёму")
    _save(fig, "top_utm_source.png", generated)


def make_cramers_heatmap(df: pd.DataFrame, generated: list[str]) -> dict:
    """Cramér's V между ключевыми категориальными фичами и таргетом."""
    feats = pd.DataFrame({
        "utm_medium": _group_top_n(df["utm_medium"], 15),
        "device_category": df["device_category"].astype("string"),
        "device_os": _group_top_n(df["device_os"], 10),
        "device_browser": _group_top_n(df["device_browser"], 10),
        "geo_region": np.where(df["geo_country"].astype("string") != "Russia", "other",
                       np.where(df["geo_city"].astype("string") == "Moscow", "Moscow",
                       np.where(df["geo_city"].astype("string") == "Saint Petersburg", "SPb", "RU_other"))),
        "is_organic": df["is_organic"].astype("string"),
        "is_social": df["is_social"].astype("string"),
        "is_first_visit": df["is_first_visit"].astype("string"),
        "visit_daypart": df["visit_daypart"].astype("string"),
        "target": df["target"].astype("string"),
    })
    cols = list(feats.columns)
    mat = np.zeros((len(cols), len(cols)))
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i == j:
                mat[i, j] = 1.0
            elif i < j:
                conf = pd.crosstab(feats[a], feats[b]).to_numpy()
                v = cramers_v(conf)
                mat[i, j] = mat[j, i] = v
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="Cramers V")
    ax.set_title("Cramers V (категориальные фичи <-> target)")
    _save(fig, "cramers_v_heatmap.png", generated)
    # Связь каждой фичи с таргетом
    target_assoc = {cols[i]: round(float(mat[i, cols.index("target")]), 4)
                    for i in range(len(cols)) if cols[i] != "target"}
    return target_assoc


# --------------------------------------------------------------------------- #
# Гипотезы
# --------------------------------------------------------------------------- #
def _two_group_proportion(df: pd.DataFrame, mask: pd.Series) -> dict:
    """Доли и z-тест для бинарного разбиения (group=mask)."""
    a, b = df.loc[mask, "target"], df.loc[~mask, "target"]
    count = np.array([int(a.sum()), int(b.sum())])
    nobs = np.array([int(len(a)), int(len(b))])
    z, p = proportions_ztest(count, nobs)
    return {
        "cr_group": float(count[0] / nobs[0]),
        "cr_rest": float(count[1] / nobs[1]),
        "n_group": int(nobs[0]),
        "n_rest": int(nobs[1]),
        "z_stat": float(z),
        "pvalue": float(p),
    }


def _chi2_test(df: pd.DataFrame, col: str) -> dict:
    conf = pd.crosstab(df[col], df["target"])
    chi2, p, dof, _ = stats.chi2_contingency(conf)
    return {"chi2": float(chi2), "pvalue": float(p), "dof": int(dof),
            "cramers_v": round(cramers_v(conf.to_numpy()), 4)}


def run_hypotheses(df: pd.DataFrame) -> list[dict]:
    """≥3 формальные гипотезы (α=0.05, поправка Бонферрони отмечена)."""
    n_tests = 4
    bonf = ALPHA / n_tests
    results: list[dict] = []

    # H1: тип трафика <-> конверсия (органика vs платный), z-тест долей
    r = _two_group_proportion(df, df["is_organic"])
    results.append({
        "id": "H1", "name": "Органика vs платный трафик <-> конверсия",
        "H0": "CR(органика) = CR(платный)", "H1": "CR различаются",
        "test": "z-тест долей", "alpha": ALPHA, "bonferroni_alpha": bonf,
        "statistic": r["z_stat"], "pvalue": r["pvalue"],
        "reject_h0": bool(r["pvalue"] < bonf),
        "detail": {"cr_organic": r["cr_group"], "cr_paid": r["cr_rest"],
                   "n_organic": r["n_group"], "n_paid": r["n_rest"]},
    })

    # H2: тип устройства <-> конверсия, χ^2
    r2 = _chi2_test(df, "device_category")
    cr_dev = cr_by(df, "device_category")["cr"].to_dict()
    results.append({
        "id": "H2", "name": "Тип устройства <-> конверсия",
        "H0": "CR не зависит от device_category", "H1": "CR зависит от device_category",
        "test": "χ^2 независимости", "alpha": ALPHA, "bonferroni_alpha": bonf,
        "statistic": r2["chi2"], "pvalue": r2["pvalue"], "dof": r2["dof"],
        "reject_h0": bool(r2["pvalue"] < bonf),
        "detail": {"cramers_v": r2["cramers_v"], "cr_by_device": {k: round(float(v), 5) for k, v in cr_dev.items()}},
    })

    # H3: новые vs возвратные, z-тест долей
    r3 = _two_group_proportion(df, df["is_first_visit"])
    results.append({
        "id": "H3", "name": "Новые vs возвратные визиты <-> конверсия",
        "H0": "CR(visit_number=1) = CR(visit_number>1)", "H1": "CR различаются",
        "test": "z-тест долей", "alpha": ALPHA, "bonferroni_alpha": bonf,
        "statistic": r3["z_stat"], "pvalue": r3["pvalue"],
        "reject_h0": bool(r3["pvalue"] < bonf),
        "detail": {"cr_new": r3["cr_group"], "cr_returning": r3["cr_rest"],
                   "n_new": r3["n_group"], "n_returning": r3["n_rest"]},
    })

    # H4: соцсети vs прочие, z-тест долей
    r4 = _two_group_proportion(df, df["is_social"])
    results.append({
        "id": "H4", "name": "Соцсети vs прочие источники <-> конверсия",
        "H0": "CR(соцсети) = CR(прочие)", "H1": "CR различаются",
        "test": "z-тест долей", "alpha": ALPHA, "bonferroni_alpha": bonf,
        "statistic": r4["z_stat"], "pvalue": r4["pvalue"],
        "reject_h0": bool(r4["pvalue"] < bonf),
        "detail": {"cr_social": r4["cr_group"], "cr_other": r4["cr_rest"],
                   "n_social": r4["n_group"], "n_other": r4["n_rest"]},
    })
    return results


# --------------------------------------------------------------------------- #
# Покрытие топ-N (обоснование порогов группировки)
# --------------------------------------------------------------------------- #
def topn_coverage(df: pd.DataFrame) -> dict:
    cov = {}
    for col, n in config.TOP_N_BY_COLUMN.items():
        if col not in df.columns:
            continue
        vc = df[col].value_counts()
        cov[col] = {
            "cardinality": int(vc.shape[0]),
            "top_n": int(n),
            "coverage_pct": round(float(vc.head(n).sum() / vc.sum() * 100), 2),
        }
    return cov


# --------------------------------------------------------------------------- #
# Отчёты
# --------------------------------------------------------------------------- #
def write_findings(stats_obj: dict) -> None:
    config.EDA_STATS_JSON.write_text(
        json.dumps(stats_obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    L: list[str] = []
    L.append("# EDA: инсайты и проверенные гипотезы (фаза 03)\n")
    L.append(f"_Сгенерировано: {stats_obj['generated_at']}_\n")
    L.append(f"Общий CR = **{stats_obj['overall_cr']*100:.3f}%** "
             f"({stats_obj['n_target']:,} из {stats_obj['n']:,}) -- **сильный дисбаланс** "
             "(-> class_weight/scale_pos_weight, PR-AUC, подбор порога в фазе 05).\n")

    L.append("## Бизнес-разрезы\n")
    biz = stats_obj["business"]
    L.append(f"- **Органика** ({biz['n_organic']:,}, {biz['n_organic']/stats_obj['n']*100:.1f}%): "
             f"CR={biz['cr_organic']*100:.3f}%; **платный** ({biz['n_paid']:,}): CR={biz['cr_paid']*100:.3f}%.")
    L.append(f"- **Соцсети** ({biz['n_social']:,}): CR={biz['cr_social']*100:.3f}%; "
             f"прочие: CR={biz['cr_other']*100:.3f}%.")
    L.append(f"- **Новые** визиты: CR={biz['cr_new']*100:.3f}%; **возвратные**: CR={biz['cr_returning']*100:.3f}%.\n")

    L.append("## Связь категориальных фичей с таргетом (Cramers V)\n")
    for k, v in sorted(stats_obj["target_association"].items(), key=lambda x: -x[1]):
        L.append(f"- `{k}`: V={v:.3f}")
    L.append("")

    L.append("## Статистические гипотезы (α=0.05, поправка Бонферрони α'=0.0125)\n")
    for h in stats_obj["hypotheses"]:
        verdict = "✅ H0 отвергается" if h["reject_h0"] else "❌ H0 не отвергается"
        L.append(f"### {h['id']}. {h['name']}")
        L.append(f"- H0: {h['H0']}; H1: {h['H1']}")
        L.append(f"- Тест: {h['test']}; статистика={h['statistic']:.3f}; "
                 f"p-value={h['pvalue']:.3e}; **{verdict}**")
        L.append(f"- Детали: {json.dumps(h['detail'], ensure_ascii=False)}")
        L.append("")

    L.append("## Пороги группировки редких категорий (вход в фазу 04)\n")
    L.append("| Колонка | Кардинальность | top-N | Покрытие топ-N, % |")
    L.append("|---|---:|---:|---:|")
    for col, c in stats_obj["topn_coverage"].items():
        L.append(f"| `{col}` | {c['cardinality']} | {c['top_n']} | {c['coverage_pct']} |")
    L.append("")

    L.append("## Гипотезы полезных фичей (для блока 04)\n")
    L.append("- `is_organic`/`is_paid`, `is_social` -- сильные различия CR (см. гипотезы).")
    L.append("- `is_first_visit`/`visit_number_log` -- новизна визита связана с CR.")
    L.append("- `visit_hour`/`visit_daypart`, `visit_dow`/`is_weekend` -- временные паттерны CR.")
    L.append("- `device_category`/`is_mobile`, группировки device_os/brand/browser.")
    L.append("- Гео: `is_russia`, `is_moscow`, `is_spb`; группировка geo_city top-N.")
    L.append("- Индикаторы наличия: `has_utm_campaign/keyword/adcontent` (пропуск информативен).")
    L.append("- Группировка высоко-кардинальных utm_* и geo_city: топ-N + 'other'.")
    L.append("")
    config.EDA_FINDINGS_MD.write_text("\n".join(L), encoding="utf-8")


def main() -> dict:
    config.ensure_dirs()
    print("[03] Загрузка датасета + EDA-производные …")
    df = build_eda_frame()
    overall_cr = float(df["target"].mean())
    generated: list[str] = []

    print("[03] Генерация графиков …")
    make_figures(df, overall_cr, generated)
    target_assoc = make_cramers_heatmap(df, generated)

    print("[03] Проверка гипотез …")
    hypotheses = run_hypotheses(df)

    business = {
        "n_organic": int(df["is_organic"].sum()),
        "cr_organic": float(df.loc[df["is_organic"], "target"].mean()),
        "n_paid": int((~df["is_organic"]).sum()),
        "cr_paid": float(df.loc[~df["is_organic"], "target"].mean()),
        "n_social": int(df["is_social"].sum()),
        "cr_social": float(df.loc[df["is_social"], "target"].mean()),
        "cr_other": float(df.loc[~df["is_social"], "target"].mean()),
        "cr_new": float(df.loc[df["is_first_visit"], "target"].mean()),
        "cr_returning": float(df.loc[~df["is_first_visit"], "target"].mean()),
    }

    stats_obj = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": int(len(df)),
        "n_target": int(df["target"].sum()),
        "overall_cr": overall_cr,
        "social_sources": sorted(config.SOCIAL_SOURCES),
        "business": business,
        "target_association": target_assoc,
        "hypotheses": hypotheses,
        "topn_coverage": topn_coverage(df),
        "figures": generated,
    }
    write_findings(stats_obj)
    print(f"[03] Фигур: {len(generated)}; гипотез: {len(hypotheses)} "
          f"(отвергнуто H0: {sum(h['reject_h0'] for h in hypotheses)})")
    print(f"[03] -> {config.EDA_FINDINGS_MD}, {config.EDA_STATS_JSON}, reports/figures/*.png")
    return stats_obj


if __name__ == "__main__":
    main()
