"""Тесты фазы 01 -- загрузка и построение таргета.

Проверяют артефакты, созданные `python3 -m src.data.load_and_target`:
читают parquet/JSON (а не пересканируют 4 ГБ hits), поэтому быстрые.
Если артефакты ещё не сгенерированы -- тесты пропускаются с инструкцией.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import config

EXPECTED_ROWS = 1_860_042
EXPECTED_COLS = 18


def _require(path) -> None:
    if not path.exists():
        pytest.skip(
            f"Нет артефакта {path}. Сначала: python3 -m src.data.load_and_target"
        )


@pytest.fixture(scope="module")
def sessions() -> pd.DataFrame:
    _require(config.SESSIONS_PARQUET)
    return pd.read_parquet(config.SESSIONS_PARQUET)


@pytest.fixture(scope="module")
def target() -> pd.DataFrame:
    _require(config.TARGET_BY_SESSION_PARQUET)
    return pd.read_parquet(config.TARGET_BY_SESSION_PARQUET)


@pytest.fixture(scope="module")
def stats() -> dict:
    _require(config.INGESTION_STATS_JSON)
    return json.loads(config.INGESTION_STATS_JSON.read_text(encoding="utf-8"))


def test_sessions_shape(sessions: pd.DataFrame) -> None:
    assert sessions.shape == (EXPECTED_ROWS, EXPECTED_COLS)


def test_session_id_unique(sessions: pd.DataFrame) -> None:
    assert sessions["session_id"].is_unique


def test_no_hits_columns_in_sessions(sessions: pd.DataFrame) -> None:
    """Анти-лик: в таблице сессий не должно быть полей из ga_hits."""
    forbidden_prefixes = ("hit_", "event_")
    leaked = [c for c in sessions.columns if c.startswith(forbidden_prefixes)]
    assert leaked == [], f"Поля hits протекли в sessions: {leaked}"


def test_target_frame_valid(target: pd.DataFrame) -> None:
    assert set(target.columns) == {"session_id", "target"}
    assert target["target"].notna().all()
    assert set(target["target"].unique()) <= {0, 1}
    assert len(target) == EXPECTED_ROWS


def test_conversion_rate_in_range(target: pd.DataFrame) -> None:
    cr = float(target["target"].mean())
    assert 0.025 <= cr <= 0.030, f"CR={cr:.4%} вне диапазона 2.5–3.0%"


def test_n_target_sessions(target: pd.DataFrame) -> None:
    n_target = int(target["target"].sum())
    assert 45_000 <= n_target <= 55_000, f"целевых сессий {n_target}"


def test_pct_sessions_without_hits(stats: dict) -> None:
    pct = float(stats["pct_sessions_without_hits"])
    assert 6.5 <= pct <= 7.2, f"доля сессий без хитов {pct:.2f}% (ожидалось ~6.87%)"


def test_target_actions_config() -> None:
    assert len(config.TARGET_ACTIONS) == 8
    assert "sub_open_chat" not in config.TARGET_ACTIONS
