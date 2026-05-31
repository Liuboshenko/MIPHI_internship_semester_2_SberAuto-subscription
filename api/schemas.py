"""Pydantic-схемы контракта API (фаза 06).

Вход -- атрибуты визита из `ga_sessions` (session-уровень). Обязательны минимально
необходимые поля; остальные опциональны (по умолчанию -> 'unknown' в препроцессинге).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VisitRequest(BaseModel):
    """Атрибуты одного визита (вход модели). `device_model` не нужен (удалён в фазе 02)."""

    # Обязательные (минимальный осмысленный контракт).
    utm_source: str = Field(..., examples=["ZpYIoDJMcFzVoPFsHGJL"])
    utm_medium: str = Field(..., examples=["banner"])
    device_category: str = Field(..., examples=["mobile"])

    # Опциональные -- отсутствие информативно (-> 'unknown').
    utm_campaign: str | None = None
    utm_adcontent: str | None = None
    utm_keyword: str | None = None
    device_os: str | None = None
    device_brand: str | None = None
    device_screen_resolution: str | None = Field(default=None, examples=["360x800"])
    device_browser: str | None = None
    geo_country: str | None = "Russia"
    geo_city: str | None = Field(default=None, examples=["Moscow"])
    visit_number: int | None = Field(default=1, ge=1)
    visit_date: str | None = Field(default=None, examples=["2021-06-15"])
    visit_time: str | None = Field(default=None, examples=["14:30:00"])
    session_id: str | None = None

    model_config = ConfigDict(extra="ignore")


class BatchRequest(BaseModel):
    items: list[VisitRequest]


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int = Field(..., description="0/1 при пороге THRESHOLD")
    probability: float = Field(..., description="Вероятность конверсии ∈ [0,1]")
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: str
