# Отчёт об очистке данных (фаза 02)

_Сгенерировано: 2026-05-31T12:56:14+00:00_

## Сводка

- Строк (визитов): **1,860,042**
- Колонок: **18** (`device_model` удалён)
- `session_id` уникален: **True**
- `target` NaN: **0**; распределение: {0: 1809728, 1: 50314}
- Полных дубликатов строк удалено: **0**
- `visit_date` NaT: **0**; диапазон 2021-05-19 … 2021-12-31
- `visit_time` NaN (секунды от полуночи): **0**
- `visit_number`: min **1**, max **564**
- Некорректный формат `device_screen_resolution` (не WxH): **8** (разбор в фазе 04)

## Стратегия пропусков и эффект (NaN, % -- ДО -> ПОСЛЕ)

| Колонка | NaN ДО, % | NaN ПОСЛЕ, % | dtype после |
|---|---:|---:|---|
| `session_id` | 0.00 | 0.00 | string |
| `client_id` | 0.00 | 0.00 | string |
| `visit_date` | 0.00 | 0.00 | datetime64[us] |
| `visit_time` | 0.00 | 0.00 | Int64 |
| `visit_number` | 0.00 | 0.00 | int32 |
| `utm_source` | 0.01 | 0.00 | category |
| `utm_medium` | 0.00 | 0.00 | category |
| `utm_campaign` | 11.81 | 0.00 | category |
| `utm_adcontent` | 18.04 | 0.00 | category |
| `utm_keyword` | 58.17 | 0.00 | category |
| `device_category` | 0.00 | 0.00 | category |
| `device_os` | 57.53 | 0.00 | category |
| `device_brand` | 19.74 | 0.00 | category |
| `device_model` | 99.12 | -- (удалена) | -- |
| `device_screen_resolution` | 0.00 | 0.00 | category |
| `device_browser` | 0.00 | 0.00 | category |
| `geo_country` | 0.00 | 0.00 | category |
| `geo_city` | 0.00 | 0.00 | category |
| `target` | 0.00 | 0.00 | int8 |

## Решения

- `device_model` удалён (99.12% NaN -- нет сигнала).
- Пропуски категорий -> `unknown` (информативная категория, строки не выбрасываем).
- Маркеры мусора (`(not set)`, `(not provided)`, `''`, `nan`) -> `unknown`.
- `utm_medium` приведён к нижнему регистру; значения органики `organic`/`referral`/`(none)` сохранены (см. `todo/improvements_utm_medium_none_normalization.md`).
- `visit_date` -> datetime64; `visit_time` -> секунды от полуночи; `visit_number` -> int32.
- Зашифрованные ID `utm_source`/`utm_campaign`/`utm_adcontent` не склеивались.
