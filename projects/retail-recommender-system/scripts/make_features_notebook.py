"""Генерация notebooks/02_features.ipynb из кода (чтобы notebook не расходился с src/).

Запуск: python scripts/make_features_notebook.py
Затем:  jupyter nbconvert --to notebook --execute --inplace notebooks/02_features.ipynb

Неделя 2: граница train/target, признаки товаров, пользователей и пар
пользователь–товар, целевая переменная, проверки на утечки.
"""
from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

CELLS: list[tuple[str, str]] = [
    ("md",
     "# Рекомендательная система для магазина — подготовка признаков\n\n"
     "Задача проекта — по прошлому поведению визитора предсказывать, с какими "
     "товарами он будет взаимодействовать дальше, и выдавать Top-N рекомендаций "
     "(постановка и выбор метрик — в `01_eda.ipynb`).\n\n"
     "На этом этапе я готовлю данные для моделей:\n\n"
     "- разделяю историю по времени на train и целевое окно;\n"
     "- считаю признаки товаров, пользователей и пар пользователь–товар;\n"
     "- формирую целевую переменную;\n"
     "- проверяю, что признаки не заглядывают в будущее.\n\n"
     "Точка отсечения `cutoff = 2015-07-01` — момент, в который модель делает "
     "предсказание. Все признаки считаются только из событий до неё."),

    ("code",
     "import sys\n"
     "from pathlib import Path\n"
     "sys.path.insert(0, str(Path.cwd().parent / 'src'))\n"
     "\n"
     "%matplotlib inline\n"
     "import matplotlib.pyplot as plt\n"
     "import numpy as np\n"
     "import pandas as pd\n"
     "\n"
     "from retail_recommender.config import load_config\n"
     "from retail_recommender import features as F\n"
     "from retail_recommender.preprocessing.interactions import load_interactions\n"
     "from retail_recommender.preprocessing.properties import load_metadata_snapshots\n"
     "from retail_recommender.preprocessing.categories import load_category_paths\n"
     "\n"
     "pd.set_option('display.width', 140)\n"
     "pd.set_option('display.max_columns', 40)\n"
     "\n"
     "cfg = load_config()\n"
     "itx = load_interactions(cfg)      # processed: +ts, +is_bot, дедуп; см. build_dataset.py\n"
     "meta = load_metadata_snapshots(cfg)\n"
     "paths = load_category_paths(cfg)\n"
     "\n"
     "CUTOFF = F.cutoff_ts(cfg)\n"
     "WINDOW_END = F.label_window_end_ts(cfg)\n"
     "STRONG = set(cfg.strong_events)   # {'addtocart', 'transaction'}\n"
     "len(itx), CUTOFF.date(), STRONG"),

    ("md",
     "## 1. Короткая проверка данных\n\n"
     "Полная проверка качества — в `01_eda.ipynb`. Здесь только то, что важно для "
     "признаков: типы событий, отсутствие пропусков в ключах, аномально активные "
     "визиторы.\n\n"
     "Аномально активных визиторов (больше 500 событий за весь период) исключаю "
     "как парсеров — порог выбран по результатам EDA недели 1: таких визиторов "
     "единицы, а событий у них непропорционально много (у самого активного 7757 "
     "событий). Дальше они не используются."),
    ("code",
     "print('события:      ', len(itx))\n"
     "print('период:       ', itx.ts.min().date(), '—', itx.ts.max().date())\n"
     "print('типы событий: ', itx.event.value_counts().to_dict())\n"
     "print('пропуски в visitorid/itemid/event:',\n"
     "      int(itx[['visitorid', 'itemid', 'event']].isna().sum().sum()))\n"
     "print('аномально активных визиторов:', int(itx.loc[itx.is_bot, 'visitorid'].nunique()),\n"
     "      '| их доля событий:', round(itx.is_bot.mean(), 4))"),

    ("md",
     "## 2. Граница train / целевое окно\n\n"
     "- **train** — всё, что произошло до `cutoff`; только из этих событий "
     "считаются все признаки;\n"
     "- **целевое окно** — события после `cutoff` (до конца данных, 18 сентября); "
     "по ним видно, что визитор реально сделал дальше.\n\n"
     "`F.pre_cutoff` отбрасывает и будущее, и аномально активных визиторов."),
    ("code",
     "before = F.pre_cutoff(itx, CUTOFF)                       # события до cutoff, без ботов\n"
     "after = itx[(~itx.is_bot) & (itx.ts >= CUTOFF) & (itx.ts < WINDOW_END)]\n"
     "\n"
     "print(f'cutoff:            {CUTOFF.date()}')\n"
     "print(f'train  (до):       {before.ts.min().date()} — {before.ts.max().date()}'\n"
     "      f'   {len(before):>8} событий')\n"
     "print(f'target (после):    {after.ts.min().date()} — {after.ts.max().date()}'\n"
     "      f'   {len(after):>8} событий')\n"
     "print()\n"
     "print('до cutoff: визиторов', before.visitorid.nunique(),\n"
     "      '| товаров', before.itemid.nunique(),\n"
     "      '| пар (визитор, товар)', before.groupby(['visitorid', 'itemid']).ngroups)"),
    ("code",
     "daily = itx[~itx.is_bot].set_index('ts').resample('D').size()\n"
     "fig, ax = plt.subplots(figsize=(11, 3))\n"
     "ax.plot(daily.index, daily.values)\n"
     "ax.axvline(CUTOFF, color='red', ls='--', label='cutoff 2015-07-01')\n"
     "ax.set_title('Число событий по дням'); ax.set_ylim(0); ax.legend()\n"
     "plt.tight_layout(); plt.show(); plt.close(fig)"),

    ("md",
     "## 3. Признаки товаров\n\n"
     "Всё считается только по `before`. Цен, выручки и количества в датасете нет, "
     "поэтому денежных признаков тоже нет.\n\n"
     "Категорию беру из `item_properties` по последнему недельному срезу с датой до "
     "cutoff (`F.category_as_of`) — без заглядывания в будущее. Корневую категорию — "
     "из дерева категорий."),
    ("code",
     "it = F.counts_by_event(before, 'itemid').add_prefix('item_n_')\n"
     "it['item_n_events'] = it[['item_n_view', 'item_n_addtocart', 'item_n_transaction']].sum(axis=1)\n"
     "it['item_n_users'] = before.groupby('itemid')['visitorid'].nunique()\n"
     "\n"
     "views = it['item_n_view'].replace(0, np.nan)\n"
     "it['item_cart_rate'] = (it['item_n_addtocart'] / views).fillna(0.0)\n"
     "it['item_buy_rate'] = (it['item_n_transaction'] / views).fillna(0.0)\n"
     "it['item_pop_share'] = it['item_n_events'] / len(before)\n"
     "\n"
     "it['item_recency_days'] = F.days_before(CUTOFF, before.groupby('itemid')['ts'].max())\n"
     "it['item_age_days'] = F.days_before(CUTOFF, before.groupby('itemid')['ts'].min())\n"
     "\n"
     "last30 = before[before.ts >= CUTOFF - pd.Timedelta(days=30)]\n"
     "prev30 = before[(before.ts < CUTOFF - pd.Timedelta(days=30))\n"
     "                & (before.ts >= CUTOFF - pd.Timedelta(days=60))]\n"
     "it['item_events_last_30d'] = last30.groupby('itemid').size().reindex(it.index).fillna(0)\n"
     "it['item_trend_30d'] = (\n"
     "    it['item_events_last_30d'] / prev30.groupby('itemid').size().reindex(it.index)\n"
     ").replace([np.inf, np.nan], 0.0)\n"
     "\n"
     "item_category = F.category_as_of(meta, CUTOFF)\n"
     "root = paths.set_index('categoryid')['root']\n"
     "item_features = it.reset_index()\n"
     "item_features['item_categoryid'] = item_features['itemid'].map(item_category).astype('Int64')\n"
     "item_features['item_root_category'] = item_features['item_categoryid'].map(root).astype('Int64')\n"
     "\n"
     "int_cols = [c for c in item_features if c.startswith('item_n_')] + ['item_events_last_30d']\n"
     "item_features[int_cols] = item_features[int_cols].astype(int)\n"
     "print(item_features.shape)\n"
     "item_features.head()"),
    ("code",
     "print(item_features[['item_n_events', 'item_n_users', 'item_cart_rate', 'item_buy_rate',\n"
     "                     'item_recency_days', 'item_age_days', 'item_trend_30d']].describe().round(3))\n"
     "print()\n"
     "print('категория известна у',\n"
     "      round(item_features.item_categoryid.notna().mean(), 3), 'товаров')\n"
     "print('NaN по столбцам:', item_features.isna().sum()[lambda s: s > 0].to_dict())"),
    ("code",
     "fig, ax = plt.subplots(1, 2, figsize=(12, 3))\n"
     "ax[0].hist(np.log10(item_features.item_n_events), bins=40)\n"
     "ax[0].set_title('log10(число событий на товар)')\n"
     "ax[1].hist(item_features.item_recency_days.clip(upper=60), bins=40)\n"
     "ax[1].set_title('давность последнего события товара, дней (clip 60)')\n"
     "plt.tight_layout(); plt.show(); plt.close(fig)"),

    ("md",
     "## 4. Признаки пользователей\n\n"
     "Активность, давность, разнообразие интересов. Среднего чека нет — в данных "
     "нет цен."),
    ("code",
     "us = F.counts_by_event(before, 'visitorid').add_prefix('user_n_')\n"
     "us['user_n_events'] = us[['user_n_view', 'user_n_addtocart', 'user_n_transaction']].sum(axis=1)\n"
     "us['user_n_items'] = before.groupby('visitorid')['itemid'].nunique()\n"
     "us['user_is_buyer'] = (us['user_n_transaction'] > 0).astype(int)\n"
     "\n"
     "us['user_recency_days'] = F.days_before(CUTOFF, before.groupby('visitorid')['ts'].max())\n"
     "us['user_tenure_days'] = F.days_before(CUTOFF, before.groupby('visitorid')['ts'].min())\n"
     "us['user_active_days'] = (before.assign(d=before.ts.dt.floor('D'))\n"
     "                          .groupby('visitorid')['d'].nunique())\n"
     "us['user_events_last_30d'] = (last30.groupby('visitorid').size()\n"
     "                              .reindex(us.index).fillna(0).astype(int))\n"
     "\n"
     "us['user_cart_rate'] = us['user_n_addtocart'] / us['user_n_events']\n"
     "us['user_buy_rate'] = us['user_n_transaction'] / us['user_n_events']\n"
     "\n"
     "# разнообразие интересов: сколько разных категорий у пользователя\n"
     "before_cat = before[['visitorid', 'itemid']].assign(\n"
     "    categoryid=before['itemid'].map(item_category))\n"
     "us['user_n_categories'] = (before_cat.dropna().groupby('visitorid')['categoryid'].nunique()\n"
     "                           .reindex(us.index).fillna(0).astype(int))\n"
     "\n"
     "user_features = us.reset_index()\n"
     "print(user_features.shape)\n"
     "user_features.head()"),
    ("code",
     "print(user_features[['user_n_events', 'user_n_items', 'user_recency_days',\n"
     "                     'user_tenure_days', 'user_active_days', 'user_n_categories']]\n"
     "      .describe().round(3))\n"
     "print()\n"
     "print('доля покупателей:', round(user_features.user_is_buyer.mean(), 4))\n"
     "print('NaN по столбцам:', user_features.isna().sum()[lambda s: s > 0].to_dict() or 'нет')"),

    ("md",
     "## 5. Признаки пар пользователь–товар\n\n"
     "Кандидаты — пары (визитор, товар), которые встречались до cutoff. Для каждой "
     "пары: сколько и как взаимодействовали, давность, доля товара в истории "
     "пользователя, близость пользователя к категории товара."),
    ("code",
     "ui = F.counts_by_event(before, ['visitorid', 'itemid']).add_prefix('ui_n_')\n"
     "ui['ui_n_events'] = ui[['ui_n_view', 'ui_n_addtocart', 'ui_n_transaction']].sum(axis=1)\n"
     "ui['ui_has_carted'] = (ui['ui_n_addtocart'] > 0).astype(int)\n"
     "ui['ui_has_bought'] = (ui['ui_n_transaction'] > 0).astype(int)\n"
     "ui['ui_recency_days'] = F.days_before(CUTOFF, before.groupby(['visitorid', 'itemid'])['ts'].max())\n"
     "ui = ui.reset_index()\n"
     "\n"
     "ui['ui_share_of_user'] = ui['ui_n_events'] / ui['visitorid'].map(us['user_n_events'])\n"
     "ui['ui_item_pop_share'] = ui['itemid'].map(item_features.set_index('itemid')['item_pop_share'])\n"
     "\n"
     "# близость пользователя к категории товара: доля его событий в этой категории\n"
     "user_cat = (before_cat.dropna().groupby(['visitorid', 'categoryid']).size()\n"
     "            .rename('n').reset_index())\n"
     "ui['categoryid'] = ui['itemid'].map(item_category)\n"
     "ui = ui.merge(user_cat, on=['visitorid', 'categoryid'], how='left')\n"
     "ui['ui_user_cat_affinity'] = (ui['n'] / ui['visitorid'].map(us['user_n_events'])).fillna(0.0)\n"
     "user_item_features = ui.drop(columns=['categoryid', 'n'])\n"
     "\n"
     "print(user_item_features.shape,\n"
     "      '| ключ уникален:', not user_item_features.duplicated(['visitorid', 'itemid']).any())\n"
     "user_item_features.head()"),
    ("code",
     "print(user_item_features[['ui_n_events', 'ui_recency_days', 'ui_share_of_user',\n"
     "                          'ui_user_cat_affinity', 'ui_item_pop_share']].describe().round(4))\n"
     "print()\n"
     "print('пар с повторным взаимодействием (>1 события):',\n"
     "      int((user_item_features.ui_n_events > 1).sum()),\n"
     "      f'({(user_item_features.ui_n_events > 1).mean():.4f})')"),

    ("md",
     "## 6. Целевая переменная\n\n"
     "`target(u, i) = 1`, если пара `(u, i)` встречалась до cutoff и в целевом окне "
     "`[2015-07-01, 2015-09-18]` у этой же пары есть сильное событие "
     "(`addtocart` или `transaction`); иначе `0`.\n\n"
     "Считаю target только для исторических пар — тех, что уже есть в "
     "`user_item_features`. Положительных пар мало: повторное сильное "
     "взаимодействие с тем же товаром в этих данных редкое. Это ещё не готовый "
     "обучающий набор — в нём нет отрицательных примеров по парам, которых до "
     "cutoff не было; кандидатов для ранжирования будем генерировать на этапе "
     "моделей."),
    ("code",
     "strong_pairs = F.strong_events_after(itx, CUTOFF, WINDOW_END, STRONG)\n"
     "target = F.build_target(user_item_features[['visitorid', 'itemid']], strong_pairs)\n"
     "\n"
     "print('целевое окно:      2015-07-01 —', after.ts.max().date())\n"
     "print('исторических пар:  ', len(target))\n"
     "print('из них target = 1: ', int(target.target.sum()),\n"
     "      f'({target.target.mean():.4f})')\n"
     "target[target.target == 1].head()"),

    ("md",
     "## 7. Итоговая таблица\n\n"
     "Собираю всё в одну таблицу на уровне пары (визитор, товар): признаки пары + "
     "признаки пользователя + признаки товара + target. Её и будем подавать в "
     "модель классификации на следующем этапе."),
    ("code",
     "dataset = (user_item_features\n"
     "           .merge(user_features, on='visitorid', how='left')\n"
     "           .merge(item_features, on='itemid', how='left')\n"
     "           .merge(target, on=['visitorid', 'itemid'], how='left'))\n"
     "\n"
     "print('итоговая таблица:', dataset.shape)\n"
     "print('доля target = 1: ', round(dataset.target.mean(), 4))\n"
     "print('признаков:        ', dataset.shape[1] - 3, '(без visitorid, itemid, target)')\n"
     "dataset.head()"),
    ("code",
     "out = cfg.paths.processed_dir / 'features'\n"
     "out.mkdir(parents=True, exist_ok=True)\n"
     "item_features.to_parquet(out / 'item_features.parquet', index=False)\n"
     "user_features.to_parquet(out / 'user_features.parquet', index=False)\n"
     "user_item_features.to_parquet(out / 'user_item_features.parquet', index=False)\n"
     "dataset.to_parquet(out / 'dataset.parquet', index=False)\n"
     "for p in sorted(out.glob('*.parquet')):\n"
     "    print(f'{p.name:26} {round(p.stat().st_size / 1e6, 1):>5} МБ')"),

    ("md",
     "## 8. Проверки на утечки и корректность\n\n"
     "Признаки не должны зависеть от событий на/после cutoff, а агрегаты — "
     "совпадать с ручным пересчётом."),
    ("code",
     "# 1. в таблицы попали только сущности, встречавшиеся до cutoff\n"
     "assert set(item_features.itemid) <= set(before.itemid)\n"
     "assert set(user_features.visitorid) <= set(before.visitorid)\n"
     "\n"
     "# 2. давность считается от события до cutoff -> не может быть отрицательной\n"
     "assert (item_features.item_recency_days >= 0).all()\n"
     "assert (user_features.user_recency_days >= 0).all()\n"
     "\n"
     "# 3. ручной пересчёт на реальном визиторе\n"
     "v = int(user_features.sort_values('user_n_events', ascending=False).iloc[100].visitorid)\n"
     "h = before[before.visitorid == v]\n"
     "row = user_features[user_features.visitorid == v].iloc[0]\n"
     "assert row.user_n_events == len(h)\n"
     "assert row.user_n_items == h.itemid.nunique()\n"
     "assert row.user_n_transaction == int((h.event == 'transaction').sum())\n"
     "print(f'визитор {v}: {len(h)} событий до cutoff, {h.itemid.nunique()} товаров — совпало')\n"
     "\n"
     "# 4. портим все события после cutoff -> признаки товаров не меняются\n"
     "corrupt = itx.copy()\n"
     "corrupt.loc[corrupt.ts >= CUTOFF, 'event'] = 'transaction'\n"
     "corrupt.loc[corrupt.ts >= CUTOFF, 'itemid'] = -1\n"
     "it2 = F.counts_by_event(F.pre_cutoff(corrupt, CUTOFF), 'itemid')\n"
     "it0 = F.counts_by_event(before, 'itemid')\n"
     "pd.testing.assert_frame_equal(it0, it2)\n"
     "print('порча событий после cutoff не влияет на признаки до cutoff: OK')"),

    ("md",
     "## Выводы\n\n"
     "- История разделена по времени: `cutoff = 2015-07-01`, признаки — только из "
     "событий до этой даты, целевое окно — после.\n"
     "- Готовы три таблицы признаков: товары (`itemid`), пользователи "
     "(`visitorid`), пары (`visitorid`, `itemid`) — популярность, конверсии, "
     "свежесть, динамика, разнообразие интересов, история пары.\n"
     "- Целевая переменная — повторное сильное взаимодействие пары в целевом окне; "
     "положительных примеров мало, это надо учитывать при обучении.\n"
     "- Проверки показывают, что признаки не используют будущие события.\n"
     "- Следующий этап — рекомендательные модели: от популярности к "
     "коллаборативной фильтрации и классификации пар пользователь–товар."),
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
    out = ROOT / "notebooks" / "02_features.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(build(), out)
    print(f"написан {out} ({len(CELLS)} ячеек)")
