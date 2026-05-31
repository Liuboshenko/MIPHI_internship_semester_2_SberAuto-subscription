"""FastAPI-приложение: предсказание вероятности конверсии визита (фаза 06).

Модель грузится ОДИН раз на старте (lifespan) -> быстрый инференс (миллисекунды).
Эндпоинты: GET /health, POST /predict, POST /predict_batch.

Запуск:  uvicorn api.main:app --host 0.0.0.0 --port 8000   (Swagger: /docs)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api import model as model_api
from api.schemas import BatchRequest, HealthResponse, PredictResponse, VisitRequest

# Состояние приложения: загруженные артефакты (заполняется в lifespan).
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline, metadata = model_api.load_artifacts()
    _state["pipeline"] = pipeline
    _state["metadata"] = metadata
    yield
    _state.clear()


app = FastAPI(
    title="СберАвтоподписка -- предсказание конверсии визита",
    version="1.0.0",
    description="Вход: атрибуты визита (session-уровень). Выход: 0/1 + вероятность.",
    lifespan=lifespan,
)


def _model_version() -> str:
    return str(_state.get("metadata", {}).get("generated_at", "unknown"))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded="pipeline" in _state,
        model_version=_model_version(),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: VisitRequest) -> PredictResponse:
    if "pipeline" not in _state:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    try:
        result = model_api.predict(_state["pipeline"], _state["metadata"], [request.model_dump()])
    except Exception as exc:  # pragma: no cover - защита от непредвиденного входа
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {exc}") from exc
    return PredictResponse(**result[0])


@app.post("/predict_batch", response_model=list[PredictResponse])
def predict_batch(request: BatchRequest) -> list[PredictResponse]:
    if "pipeline" not in _state:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    if not request.items:
        raise HTTPException(status_code=422, detail="Пустой список items")
    try:
        results = model_api.predict(
            _state["pipeline"], _state["metadata"], [i.model_dump() for i in request.items]
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {exc}") from exc
    return [PredictResponse(**r) for r in results]
