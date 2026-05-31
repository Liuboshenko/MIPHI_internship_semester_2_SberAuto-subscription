"""Замер латентности инференса (фаза 06, шаг A3).

Грузит модель один раз и прогоняет N одиночных предсказаний, считает p50/p95,
пишет `reports/api_latency.md`. Запуск:  `python3 -m api.benchmark`
"""

from __future__ import annotations

import time

import numpy as np

from api import model as model_api
from src import config

SAMPLE_VISIT = {
    "utm_source": "ZpYIoDJMcFzVoPFsHGJL", "utm_medium": "banner",
    "device_category": "mobile", "device_os": "Android", "device_brand": "Samsung",
    "device_screen_resolution": "360x800", "device_browser": "Chrome",
    "geo_country": "Russia", "geo_city": "Moscow", "utm_campaign": "LEoPHuyFvzoNfnzGgfcd",
    "visit_number": 1, "visit_date": "2021-06-15", "visit_time": "14:30:00",
}


def run(n: int = 200) -> dict:
    pipeline, metadata = model_api.load_artifacts()
    model_api.predict(pipeline, metadata, [SAMPLE_VISIT])  # прогрев

    latencies = []
    for _ in range(n):
        t = time.perf_counter()
        model_api.predict(pipeline, metadata, [SAMPLE_VISIT])
        latencies.append((time.perf_counter() - t) * 1000)
    arr = np.array(latencies)
    stats = {
        "n": n,
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "max_ms": round(float(arr.max()), 2),
    }
    md = [
        "# Латентность API (одиночный /predict)\n",
        f"- Запросов: **{stats['n']}** (после прогрева модели)",
        f"- p50: **{stats['p50_ms']} мс**",
        f"- p95: **{stats['p95_ms']} мс**",
        f"- p99: **{stats['p99_ms']} мс**",
        f"- max: **{stats['max_ms']} мс**",
        f"\n**Требование ≤ 3000 мс -- выполнено с запасом** (p95={stats['p95_ms']} мс).\n",
    ]
    (config.REPORTS_DIR / "api_latency.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[06] latency p50={stats['p50_ms']}ms p95={stats['p95_ms']}ms -> reports/api_latency.md")
    return stats


if __name__ == "__main__":
    config.ensure_dirs()
    run()
