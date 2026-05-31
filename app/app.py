"""Streamlit-интерфейс поверх модели конверсии (фаза 06).

Основной режим: вызывает FastAPI `POST /predict`. Если API недоступен -- фоллбэк на
прямую загрузку `pipeline.joblib` (единая логика инференса из `api.model`).

Запуск:  streamlit run app/app.py     # http://localhost:8501
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
import streamlit as st

# Корень проекта в sys.path (streamlit запускается из произвольной CWD).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import model as model_api  # noqa: E402
from src import config  # noqa: E402

API_URL = "http://localhost:8000"

st.set_page_config(page_title="СберАвтоподписка -- конверсия визита", page_icon="🚗")


@st.cache_resource
def load_local_model():
    return model_api.load_artifacts()


@st.cache_data
def load_metadata() -> dict:
    return json.loads(config.METADATA_JSON.read_text(encoding="utf-8"))


def api_available() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=1.0)
        return r.status_code == 200 and r.json().get("model_loaded", False)
    except requests.RequestException:
        return False


def predict_visit(payload: dict) -> dict:
    """Через API, если доступен; иначе локально."""
    if api_available():
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=5.0)
        r.raise_for_status()
        return {**r.json(), "_source": "FastAPI"}
    pipeline, metadata = load_local_model()
    res = model_api.predict(pipeline, metadata, [payload])[0]
    return {**res, "_source": "локальная модель"}


# --- Интерфейс --------------------------------------------------------------
st.title("🚗 СберАвтоподписка -- вероятность целевого действия")
st.caption("Предсказание конверсии визита по его атрибутам (session-уровень).")

MEDIUMS = ["banner", "cpc", "(none)", "cpm", "referral", "organic", "email", "push",
           "stories", "smm", "unknown"]
CATEGORIES = ["mobile", "desktop", "tablet"]
OSES = ["unknown", "Android", "iOS", "Windows", "Macintosh", "Linux"]
BROWSERS = ["Chrome", "Safari", "YaBrowser", "Firefox", "Opera", "Edge", "unknown"]
CITIES = ["Moscow", "Saint Petersburg", "unknown"]

with st.form("visit"):
    col1, col2 = st.columns(2)
    with col1:
        utm_source = st.text_input("utm_source", value="ZpYIoDJMcFzVoPFsHGJL")
        utm_medium = st.selectbox("utm_medium", MEDIUMS, index=0)
        utm_campaign = st.text_input("utm_campaign", value="")
        utm_adcontent = st.text_input("utm_adcontent", value="")
        device_category = st.selectbox("device_category", CATEGORIES, index=0)
        device_os = st.selectbox("device_os", OSES, index=1)
    with col2:
        device_browser = st.selectbox("device_browser", BROWSERS, index=0)
        device_screen_resolution = st.text_input("device_screen_resolution", value="360x800")
        geo_city = st.selectbox("geo_city", CITIES, index=0)
        visit_number = st.number_input("visit_number", min_value=1, value=1, step=1)
        visit_date = st.text_input("visit_date (YYYY-MM-DD)", value="2021-06-15")
        visit_time = st.text_input("visit_time (HH:MM:SS)", value="14:30:00")
    submitted = st.form_submit_button("Предсказать")

if submitted:
    payload = {
        "utm_source": utm_source or None, "utm_medium": utm_medium,
        "utm_campaign": utm_campaign or None, "utm_adcontent": utm_adcontent or None,
        "device_category": device_category, "device_os": device_os,
        "device_browser": device_browser,
        "device_screen_resolution": device_screen_resolution or None,
        "geo_country": "Russia", "geo_city": geo_city,
        "visit_number": int(visit_number), "visit_date": visit_date or None,
        "visit_time": visit_time or None,
    }
    try:
        result = predict_visit(payload)
        prob = result["probability"]
        st.metric("Вероятность конверсии", f"{prob*100:.2f}%")
        if result["prediction"] == 1:
            st.success("Прогноз: ЦЕЛЕВОЕ действие вероятно (1)")
        else:
            st.info("Прогноз: целевое действие маловероятно (0)")
        st.caption(f"Источник инференса: {result['_source']} · версия модели: {result['model_version']}")
    except Exception as exc:
        st.error(f"Ошибка предсказания: {exc}")

# --- О модели ---------------------------------------------------------------
with st.expander("О модели"):
    try:
        md = load_metadata()
        tm = md["test_metrics"]
        st.write(f"**Модель:** {md['model']}")
        st.write(f"**Порог (THRESHOLD):** {md['THRESHOLD']}")
        st.write(f"**ROC-AUC (test):** {tm['roc_auc']} · **PR-AUC:** {tm['pr_auc']}")
        st.write(f"**Разбиение:** {md['validation_split']} (test={md['test_size']})")
        metrics_path = config.METRICS_JSON
        if metrics_path.exists():
            mm = json.loads(metrics_path.read_text(encoding="utf-8"))
            top = list(mm.get("catboost_importance", {}).items())[:8]
            st.write("**Топ-признаки (CatBoost importance):**")
            st.table({"признак": [k for k, _ in top], "importance": [v for _, v in top]})
    except Exception as exc:
        st.write(f"Метаданные недоступны: {exc}")
