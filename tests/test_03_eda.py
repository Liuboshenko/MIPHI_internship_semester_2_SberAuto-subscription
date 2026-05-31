"""Тесты фазы 03 -- EDA и гипотезы.

Артефактные проверки (фигуры, eda_findings.md, eda_stats.json с ≥3 гипотезами,
SOCIAL_SOURCES в config) + юнит-тест Cramér's V на синтетике.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src import config
from src.eda import run_eda

KEY_FIGURES = [
    "class_balance.png",
    "cr_by_utm_medium.png",
    "cr_organic_vs_paid.png",
    "cr_by_device_category.png",
    "cr_social_vs_other.png",
    "cr_new_vs_returning.png",
    "cr_by_hour.png",
    "cramers_v_heatmap.png",
]


def _require(path) -> None:
    if not path.exists():
        pytest.skip(f"Нет артефакта {path}. Сначала: python3 -m src.eda.run_eda")


@pytest.fixture(scope="module")
def stats() -> dict:
    _require(config.EDA_STATS_JSON)
    return json.loads(config.EDA_STATS_JSON.read_text(encoding="utf-8"))


def test_key_figures_exist() -> None:
    _require(config.EDA_STATS_JSON)
    missing = [f for f in KEY_FIGURES if not (config.FIGURES_DIR / f).exists()]
    assert missing == [], f"Не сгенерированы фигуры: {missing}"


def test_eda_findings_md_exists() -> None:
    _require(config.EDA_FINDINGS_MD)
    assert config.EDA_FINDINGS_MD.stat().st_size > 0


def test_at_least_three_hypotheses_with_pvalue(stats: dict) -> None:
    hyps = stats["hypotheses"]
    assert len(hyps) >= 3
    for h in hyps:
        assert "pvalue" in h
        p = float(h["pvalue"])
        assert 0.0 <= p <= 1.0
        assert "H0" in h and "H1" in h and "test" in h


def test_recommended_hypotheses_significant(stats: dict) -> None:
    """Рекомендованные гипотезы H1–H3 должны отвергать H0 (CR различается)."""
    by_id = {h["id"]: h for h in stats["hypotheses"]}
    for hid in ("H1", "H2", "H3"):
        assert by_id[hid]["reject_h0"] is True, f"{hid} не отвергла H0"


def test_social_sources_in_config() -> None:
    assert hasattr(config, "SOCIAL_SOURCES")
    assert len(config.SOCIAL_SOURCES) == 6
    assert all(isinstance(s, str) for s in config.SOCIAL_SOURCES)


def test_cramers_v_unit() -> None:
    """Идеально зависимая таблица -> V~1; независимая -> V~0."""
    dependent = np.array([[100, 0], [0, 100]])
    assert run_eda.cramers_v(dependent) > 0.95
    independent = np.array([[50, 50], [50, 50]])
    assert run_eda.cramers_v(independent) < 0.05
