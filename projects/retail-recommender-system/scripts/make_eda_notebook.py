"""Генерация notebooks/01_eda.ipynb из кода (чтобы notebook не расходился с src/).

Запуск: python scripts/make_eda_notebook.py
Затем:  jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

Первичный анализ данных: постановка задачи, метрики, описание и проверка
качества данных, статистический анализ, анализ пользователей / товаров /
транзакций, временной анализ.
"""
from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

CELLS: list[tuple[str, str]] = [
    ("md",
     "# Рекомендательная система для магазина — исследование данных\n\n"
     "Данные: [RetailRocket ecommerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) "
     "(открытый). Журнал событий интернет-магазина за ~4.5 месяца 2015 года.\n\n"
     "Notebook воспроизводимый: все числа считаются кодом из `data/`. Тяжёлая логика — "
     "в `src/retail_recommender/`, здесь только вызовы и графики. Перед запуском нужен "
     "`python scripts/build_dataset.py`."),

    ("code",
     "import sys\n"
     "from pathlib import Path\n"
     "sys.path.insert(0, str(Path.cwd().parent / 'src'))\n"
     "\n"
     "%matplotlib inline\n"
     "import matplotlib.pyplot as plt\n"
     "import pandas as pd\n"
     "\n"
     "from retail_recommender.config import load_config\n"
     "from retail_recommender import eda\n"
     "from retail_recommender.data.load import load_events\n"
     "from retail_recommender.preprocessing.interactions import load_interactions\n"
     "from retail_recommender.preprocessing.properties import load_metadata_snapshots\n"
     "from retail_recommender.preprocessing.categories import load_category_paths\n"
     "from retail_recommender.split.temporal import describe_windows\n"
     "\n"
     "pd.set_option('display.width', 140)\n"
     "cfg = load_config()\n"
     "events_raw = load_events(cfg)          # сырой events.csv, как есть\n"
     "itx = load_interactions(cfg)           # processed: +ts, +weight, +is_bot, +split, -460 дублей\n"
     "meta = load_metadata_snapshots(cfg)\n"
     "paths = load_category_paths(cfg)\n"
     "len(events_raw), len(itx)"),

    # ---------------------------------------------------------------- постановка
    ("md",
     "## 1. Постановка задачи\n\n"
     "**Объект рекомендации** — товар (`itemid`).\n"
     "**Пользовательская сущность** — визитор (`visitorid`); это идентификатор "
     "браузера/сессии, не аккаунт.\n"
     "**Взаимодействие** — событие одного из трёх типов: `view`, `addtocart`, "
     "`transaction`. Явных оценок нет — обратная связь неявная (implicit feedback).\n"
     "**Целевое действие** — сильное взаимодействие: `addtocart` или `transaction` "
     "(покупка — основной интерес бизнеса, `addtocart` — близкий по смыслу сигнал "
     "намерения). `view` — слабый сигнал и история пользователя.\n\n"
     "**Что выдаём** — Top-N персональных рекомендаций по `visitorid`: "
     "`recommend(visitorid) -> [itemid, ...]`.\n\n"
     "**Что доступно на момент рекомендации t** — все события с `ts < t` и срез "
     "свойств товаров (`categoryid`, `available`) на дату `<= t`. Цены, выручки и "
     "количества в датасете нет — денежные бизнес-метрики на этих данных недоступны.\n\n"
     "### Offline ML-метрики\n\n"
     "Постановка — Top-N recommendation с бинарной релевантностью (было сильное "
     "взаимодействие или нет), поэтому:\n\n"
     "- **Recall@K** — какую долю товаров, с которыми пользователь реально "
     "провзаимодействует, мы показали в топ-K. Основная метрика: у пользователя "
     "обычно 1–2 релевантных товара, важно их «поймать».\n"
     "- **NDCG@K** — то же, но с учётом позиции попадания в списке; список выдаётся "
     "упорядоченным, позиция важна.\n"
     "- **HitRate@K** — доля пользователей, у которых хотя бы одно попадание; "
     "простая и понятная для отчёта величина.\n"
     "- **MAP@K** — средняя точность; диагностическая, когда релевантных товаров "
     "несколько.\n"
     "- **Coverage** — какую долю каталога модель вообще рекомендует; ловит "
     "вырождение в «топ популярного всем».\n\n"
     "Precision@K для этого датасета малоинформативен (при 1–2 релевантных товарах "
     "его потолок ~ 1/K), берём его только как второстепенный.\n\n"
     "Offline-метрика ≠ рост выручки: она измеряет воспроизведение уже наблюдавшегося "
     "поведения под старой выдачей магазина. Коммерческий эффект проверяется онлайн "
     "A/B-тестом."),

    # ---------------------------------------------------------------- структура
    ("md", "## 2. Описание и проверка качества данных\n\n### 2.1. Сырой events.csv"),
    ("code",
     "eda.raw_overview(events_raw)"),
    ("md",
     "Поля: `timestamp` (Unix ms), `visitorid`, `event`, `itemid`, `transactionid` "
     "(заполнен только у `transaction`). Пропусков в ключевых полях нет. "
     "460 полностью повторяющихся строк — убираем на processed-стадии. "
     "`transactionid` есть ровно у строк с `event == transaction`."),

    ("code",
     "print('типы событий (сырые):')\n"
     "print(events_raw['event'].value_counts())\n"
     "print()\n"
     "print('processed: строк', len(itx), '| убрано полных дублей:', len(events_raw) - len(itx))\n"
     "print('боты (визиторы с > {} событий):'.format(cfg.bot_event_threshold))\n"
     "display(eda.bot_summary(itx, cfg))"),

    ("md", "### 2.2. Свойства товаров и дерево категорий"),
    ("code",
     "print('item_properties: строк', len(meta), '| товаров', meta['itemid'].nunique())\n"
     "print('недельных срезов:', meta['snapshot_ts'].nunique(),\n"
     "      '(', meta['snapshot_ts'].min().date(), '..', meta['snapshot_ts'].max().date(), ')')\n"
     "print('дерево категорий:', len(paths), 'категорий, корней', int((paths.depth == 0).sum()),\n"
     "      ', макс. глубина', int(paths.depth.max()))\n"
     "display(eda.property_coverage(itx, meta))"),
    ("md",
     "Категория известна не у всех просмотренных товаров (~79%), но почти у всех, что "
     "кладут в корзину и покупают (~98%). Для контентных признаков это приемлемо."),

    # ---------------------------------------------------------------- статистика
    ("md", "## 3. Статистический анализ\n\n### 3.1. Активность во времени"),
    ("code",
     "per_day = eda.events_per_day(itx)\n"
     "fig, ax = plt.subplots(figsize=(11, 3))\n"
     "ax.plot(per_day.index, per_day.values)\n"
     "ax.axvline(pd.Timestamp(cfg.feature_engineering['cutoff']), color='r', ls='--', label='cutoff 1 июля')\n"
     "ax.set_title('События по дням'); ax.legend()\n"
     "plt.tight_layout(); plt.show(); plt.close(fig)\n"
     "per_day.describe()"),
    ("md",
     "Поток ровный, ~20 тыс. событий в день, больших провалов нет; крайние дни "
     "неполные. Аномальных всплесков не видно."),

    ("code",
     "etc = eda.event_type_counts(itx)\n"
     "fig, ax = plt.subplots(figsize=(5, 3))\n"
     "ax.bar(etc.index.astype(str), etc.values)\n"
     "ax.set_yscale('log'); ax.set_title('Типы событий (лог-шкала)')\n"
     "plt.tight_layout(); plt.show(); plt.close(fig)\n"
     "etc"),

    # ---------------------------------------------------------------- пользователи
    ("md", "## 4. Анализ пользователей"),
    ("code",
     "epv = eda.events_per_visitor(itx)\n"
     "ipv = eda.unique_items_per_visitor(itx)\n"
     "print('событий на визитора: медиана', epv.median(), '| среднее', round(epv.mean(), 2),\n"
     "      '| максимум', epv.max())\n"
     "print('доля визиторов с 1 событием:', round((epv == 1).mean(), 3))\n"
     "print('доля визиторов с >= 2 товарами:', round((ipv >= 2).mean(), 3))\n"
     "display(eda.user_activity_distribution(itx))"),
    ("code",
     "fig, ax = plt.subplots(1, 2, figsize=(12, 3))\n"
     "ax[0].hist(epv.clip(upper=15), bins=15); ax[0].set_title('Событий на визитора (clip 15)')\n"
     "ax[1].hist(ipv.clip(upper=15), bins=15); ax[1].set_title('Уникальных товаров на визитора (clip 15)')\n"
     "plt.tight_layout(); plt.show(); plt.close(fig)"),
    ("md",
     "Явный long-tail: ~71% визиторов имеют одно событие. Персонализация по истории "
     "уверенно оценивается только на меньшинстве (warm-визиторы), остальным нужен "
     "fallback. Наиболее активные визиторы:"),
    ("code", "eda.top_users(itx, k=10)"),

    # ---------------------------------------------------------------- товары
    ("md", "## 5. Анализ товаров"),
    ("code",
     "pop = eda.item_popularity(itx)\n"
     "print('товаров:', len(pop), '| доля товаров с 1 событием:', round((pop == 1).mean(), 3))\n"
     "print('доля всех событий у топ-100 товаров:', round(pop.head(100).sum() / len(itx), 3))\n"
     "display(eda.item_popularity_buckets(itx))\n"
     "fig, ax = plt.subplots(figsize=(9, 3))\n"
     "ax.plot(range(500), pop.head(500).values)\n"
     "ax.set_title('Топ-500 товаров по числу событий'); ax.set_xlabel('ранг')\n"
     "plt.tight_layout(); plt.show(); plt.close(fig)"),
    ("md",
     "Популярность тоже длиннохвостая, но без сильной концентрации (топ-100 товаров — "
     "лишь ~4% событий). ~31% товаров встречаются один раз — для них персональные "
     "сигналы почти отсутствуют (item cold-start)."),

    # ---------------------------------------------------------------- транзакции
    ("md", "## 6. Анализ транзакций и воронки"),
    ("code",
     "display(eda.basket_stats(itx))\n"
     "tx_week = eda.transactions_over_time(itx, freq='W')\n"
     "fig, ax = plt.subplots(figsize=(11, 3))\n"
     "ax.plot(tx_week.index, tx_week.values, marker='o', ms=3)\n"
     "ax.axvline(pd.Timestamp(cfg.feature_engineering['cutoff']), color='r', ls='--')\n"
     "ax.set_title('Транзакции по неделям (красное — cutoff 1 июля)')\n"
     "plt.tight_layout(); plt.show(); plt.close(fig)"),
    ("code",
     "fn = eda.funnel_pair_level(itx)\n"
     "print(fn)\n"
     "print('view -> addtocart:', round(fn.loc[1, 'pairs'] / fn.loc[0, 'pairs'], 4))\n"
     "print('addtocart -> transaction:', round(fn.loc[2, 'pairs'] / fn.loc[1, 'pairs'], 4))"),
    ("md",
     "Воронка считается по парам (визитор, товар): доля пар, где было каждое "
     "действие. Это наличие действия, а не строгая последовательность во времени — "
     "трактуем осторожно. Чеки маленькие (в среднем ~1.3 товара)."),

    # ---------------------------------------------------------------- temporal split
    ("md",
     "## 7. Временной анализ и split\n\n"
     "Для оценки моделей используется скользящий вперёд split "
     "train → validation → test. Для построения признаков — отдельная "
     "точка отсечения **1 июля 2015**: признаки строятся "
     "из событий до неё, целевое окно — после."),
    ("code",
     "display(describe_windows(itx, cfg))\n"
     "display(eda.warm_cold_by_window(itx, cfg))"),
    ("code",
     "cut = pd.Timestamp(cfg.feature_engineering['cutoff'], tz='UTC')\n"
     "clean = itx[~itx.is_bot]\n"
     "before = clean[clean.ts < cut]; after = clean[clean.ts >= cut]\n"
     "print('cutoff 1 июля:')\n"
     "print('  до :', len(before), 'событий,', before.visitorid.nunique(), 'визиторов,',\n"
     "      before.itemid.nunique(), 'товаров')\n"
     "print('  после:', len(after), 'событий,', after.visitorid.nunique(), 'визиторов')\n"
     "warm_after = len(set(after.visitorid) & set(before.visitorid))\n"
     "print('  визиторов после cutoff с историей до него (warm):', warm_after,\n"
     "      '/', after.visitorid.nunique())"),
    ("md",
     "Принцип: признаки на момент t считаются только из событий с `ts < t`. "
     "Значит, нельзя брать популярность и агрегаты по данным после cutoff, "
     "использовать целевое событие в признаках или делать preprocessing на "
     "train и test вместе."),

    # ---------------------------------------------------------------- выводы
    ("md",
     "## Выводы\n\n"
     "1. Данные чистые: пропусков в ключах нет, 460 полных дублей убираются, "
     "`transactionid` согласован с типом события.\n"
     "2. Обратная связь неявная, события сильно несбалансированы "
     "(view : addtocart : transaction ≈ 96.7 : 2.5 : 0.8 %).\n"
     "3. Пользователи крайне разреженные — ~71% с одним событием; персонализация "
     "работает на warm-подвыборке, для остальных нужен fallback (популярность).\n"
     "4. Товары длиннохвостые, ~31% встречаются один раз (item cold-start).\n"
     "5. Активность во времени ровная — корректны и temporal split, и точка "
     "отсечения 1 июля.\n"
     "6. Категория покрывает ~98% покупаемых товаров — контентные признаки "
     "осмысленны.\n"
     "7. Цены/выручки в данных нет — бизнес-эффект только через прокси (сильные "
     "события) и онлайн-эксперимент."),
]


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in CELLS
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


if __name__ == "__main__":
    out = ROOT / "notebooks" / "01_eda.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(build(), out)
    print(f"написан {out} ({len(CELLS)} ячеек)")
