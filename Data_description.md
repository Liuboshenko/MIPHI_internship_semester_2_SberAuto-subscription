# GA Sessions (ga_sessions.pkl)

Одна строка = один визит на сайт.

## Описание атрибутов:

- **session_id** -- ID визита
- **client_id** -- ID посетителя
- **visit_date** -- дата визита
- **visit_time** -- время визита
- **visit_number** -- порядковый номер визита клиента
- **utm_source** -- канал привлечения
- **utm_medium** -- тип привлечения
- **utm_campaign** -- рекламная кампания
- **utm_keyword** -- ключевое слово
- **device_category** -- тип устройства
- **device_os** -- ОС устройства
- **device_brand** -- марка устройства
- **device_model** -- модель устройства
- **device_screen_resolution** -- разрешение экрана
- **device_browser** -- браузер
- **geo_country** -- страна
- **geo_city** -- город

---

# GA Hits (ga_hits.pkl)

Одна строка = одно событие в рамках одного визита на сайт.

## Описание атрибутов:

- **session_id** -- ID визита
- **hit_date** -- дата события
- **hit_time** -- время события
- **hit_number** -- порядковый номер события в рамках сессии
- **hit_type** -- тип события
- **hit_referer** -- источник события
- **hit_page_path** -- страница события
- **event_category** -- тип действия
- **event_action** -- действие
- **event_label** -- тег действия
- **event_value** -- значение результата действия
