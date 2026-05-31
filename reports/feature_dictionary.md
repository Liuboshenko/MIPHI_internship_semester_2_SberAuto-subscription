# Словарь признаков (фаза 04)

Все признаки -- **только session-уровень** (контракт API, анти-лик).

## Признаки

| Признак | Тип | Формула | Источник | Кодирование | MI |
|---|---|---|---|---|---:|
| `visit_hour` | числовой | visit_time // 3600 (0–23) | `visit_time` | median+scale | 0.0008 |
| `visit_day` | числовой | день месяца | `visit_date` | median+scale | 0.0013 |
| `visit_number_log` | числовой | log1p(clip(visit_number, ≤50)) | `visit_number` | median+scale | 0.0066 |
| `is_weekend` | бинарный | day_of_week ≥ 5 | `visit_date` | scale | 0.0115 |
| `is_organic` | бинарный | utm_medium ∈ {organic, referral, (none)} | `utm_medium` | scale | 0.0137 |
| `is_social` | бинарный | utm_source ∈ SOCIAL_SOURCES | `utm_source` | scale | 0.0090 |
| `is_first_visit` | бинарный | visit_number == 1 | `visit_number` | scale | 0.0122 |
| `has_utm_campaign` | бинарный | utm_campaign известен | `utm_campaign` | scale | 0.0079 |
| `has_utm_keyword` | бинарный | utm_keyword известен | `utm_keyword` | scale | 0.0125 |
| `has_utm_adcontent` | бинарный | utm_adcontent известен | `utm_adcontent` | scale | 0.0096 |
| `is_russia` | бинарный | geo_country == Russia | `geo_country` | scale | 0.0038 |
| `is_moscow` | бинарный | geo_city == Moscow | `geo_city` | scale | 0.0130 |
| `is_spb` | бинарный | geo_city == Saint Petersburg | `geo_city` | scale | 0.0074 |
| `screen_w` | числовой | ширина из WxH | `device_screen_resolution` | median+scale | 0.0034 |
| `screen_h` | числовой | высота из WxH | `device_screen_resolution` | median+scale | 0.0012 |
| `screen_is_valid` | бинарный | разрешение распознано | `device_screen_resolution` | scale | 0.0001 |
| `utm_medium` | категориальный | топ-15 + other | `utm_medium` | OHE(ignore) | 0.0180 |
| `utm_source` | категориальный | топ-20 + other | `utm_source` | OHE(ignore) | 0.0207 |
| `utm_campaign` | категориальный | топ-20 + other | `utm_campaign` | OHE(ignore) | 0.0177 |
| `utm_adcontent` | категориальный | топ-20 + other | `utm_adcontent` | OHE(ignore) | 0.0309 |
| `utm_keyword` | категориальный | топ-25 + other | `utm_keyword` | OHE(ignore) | 0.0306 |
| `device_category` | категориальный | mobile/desktop/tablet | `device_category` | OHE(ignore) | 0.0275 |
| `device_os` | категориальный | топ-12 + other | `device_os` | OHE(ignore) | 0.0284 |
| `device_brand` | категориальный | топ-15 + other | `device_brand` | OHE(ignore) | 0.0168 |
| `device_browser` | категориальный | топ-15 + other | `device_browser` | OHE(ignore) | 0.0297 |
| `geo_city` | категориальный | топ-20 + other | `geo_city` | OHE(ignore) | 0.0210 |
| `visit_dow` | категориальный | день недели 0–6 | `visit_date` | OHE(ignore) | 0.0097 |
| `visit_month` | категориальный | месяц 5–12 | `visit_date` | OHE(ignore) | 0.0113 |
| `visit_daypart` | категориальный | night/morning/day/evening | `visit_time` | OHE(ignore) | 0.0227 |

## Исключённые признаки (мультиколлинеарность/избыточность)

| Признак | Причина |
|---|---|
| `is_paid` | = 1 − is_organic (идеальная коллинеарность) |
| `is_mobile` | поглощается OHE(device_category) |
| `screen_area` | = screen_w · screen_h (определённая коллинеарность) |
| `screen_aspect` | = screen_w / screen_h: VIF>10 (инфляция screen_w) -> исключён |
| `geo_country` | кардинальность 166; заменён бинарным is_russia |
| `device_model` | удалён в фазе 02 (99.12% NaN) |

## Мультиколлинеарность (VIF, порог 10.0)

| Числовой признак | VIF |
|---|---:|
| `is_first_visit` | 2.2438 |
| `visit_number_log` | 2.2174 |
| `has_utm_adcontent` | 2.1954 |
| `has_utm_campaign` | 2.0885 |
| `screen_h` | 1.6597 |
| `screen_w` | 1.6441 |
| `is_organic` | 1.3672 |
| `is_moscow` | 1.2914 |
| `is_spb` | 1.2442 |
| `is_social` | 1.1694 |
| `is_russia` | 1.1404 |
| `has_utm_keyword` | 1.0823 |
| `visit_day` | 1.0132 |
| `is_weekend` | 1.0132 |
| `visit_hour` | 1.0037 |
| `screen_is_valid` | 1.0002 |

> Все VIF ≤ 10.0 после исключения `screen_area`.

## Топ-признаки по Mutual Information (агрегировано)

- `utm_adcontent`: MI=0.0309
- `utm_keyword`: MI=0.0306
- `device_browser`: MI=0.0297
- `device_os`: MI=0.0284
- `device_category`: MI=0.0275
- `visit_daypart`: MI=0.0227
- `geo_city`: MI=0.0210
- `utm_source`: MI=0.0207
- `utm_medium`: MI=0.0180
- `utm_campaign`: MI=0.0177
- `device_brand`: MI=0.0168
- `is_organic`: MI=0.0137
- `is_moscow`: MI=0.0130
- `has_utm_keyword`: MI=0.0125
- `is_first_visit`: MI=0.0122
