"""Генерация notebooks/03_models.ipynb из кода.

Запуск: python scripts/make_models_notebook.py
Затем:  jupyter nbconvert --to notebook --execute --inplace notebooks/03_models.ipynb

Переход от данных и факторов к моделям. Notebook читает готовые
результаты экспериментов из reports/week3_metrics.json (их считает
scripts/run_model_experiments.py) и пересчитывает baseline вживую для контроля
протокола. Тяжёлое обучение LightFM/ALS в notebook не повторяется.
"""
from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

CELLS: list[tuple[str, str]] = [
    ("md",
     "# Рекомендательная система — модели\n\n"
     "Данные и факторы готовы (`01_eda.ipynb`, `02_features.ipynb`). Здесь — цикл "
     "экспериментов с моделями и выбор лучшей по согласованной метрике.\n\n"
     "**Протокол без утечки из test:**\n"
     "- TRAIN — обучение (только train-окно);\n"
     "- VALIDATION — выбор модели, loss, factors, regularization, epochs;\n"
     "- TEST — один финальный замер выбранных конфигураций.\n\n"
     "Тяжёлые модели считает `scripts/run_model_experiments.py` -> "
     "`reports/week3_metrics.json`; notebook читает результаты и пересчитывает "
     "baseline вживую."),

    ("code",
     "import sys, json\n"
     "from pathlib import Path\n"
     "sys.path.insert(0, str(Path.cwd().parent / 'src'))\n"
     "import pandas as pd\n"
     "pd.set_option('display.width', 160); pd.set_option('display.max_columns', 30)\n"
     "\n"
     "from retail_recommender.config import load_config\n"
     "cfg = load_config()\n"
     "M = json.loads((cfg.paths.reports_dir / 'week3_metrics.json').read_text('utf-8'))\n"
     "P = M['protocol']\n"
     "P"),

    ("md",
     "## 1. Постановка\n\n"
     "Top-N персональных рекомендаций товаров визитору. Implicit feedback "
     "(view / addtocart / transaction), целевое действие — сильное "
     "взаимодействие (**addtocart ∪ transaction**). Основная метрика — "
     "**Recall@10 / NDCG@10**, диагностические — Coverage@10, HitRate@10, MAP@10. "
     "Уже виденные пользователем train-товары из выдачи исключаются "
     "(политика единая для всех моделей)."),

    ("md",
     "## 2. Train / validation / test\n\n"
     "Скользящий вперёд временной split (зафиксирован на этапе исследования данных). Признаки товара "
     "для гибридной модели берутся as-of начала val-окна — без информации из "
     "будущего относительно момента оценки."),
    ("code",
     "from retail_recommender.preprocessing.interactions import load_interactions\n"
     "from retail_recommender.split.temporal import describe_windows\n"
     "itx = load_interactions(cfg)\n"
     "win = describe_windows(itx, cfg)[['window','start','end','events','users','items',\n"
     "                                  'strong_events','strong_users']]\n"
     "win"),
    ("code",
     "print('target-пользователи (strong):', P['n_target_users'])\n"
     "print('из них warm (есть история до окна):', P['n_warm_users'])\n"
     "print('каталог train:', P['catalog_size'], 'товаров')\n"
     "cold_val = P['n_target_users']['val'] - P['n_warm_users']['val']\n"
     "print(f\"доля холодных на val: {cold_val / P['n_target_users']['val']:.1%} \"\n"
     "      f\"(для них любая модель = fallback на популярность)\")"),

    ("md",
     "## 3. Baseline — контрольная точка\n\n"
     "Три baseline, посчитанные ранее, пересчитаны в этом же протоколе (веса событий "
     "заморожены 1-3-5). Пересчёт вживую — проверка, что протокол honest."),
    ("code",
     "from retail_recommender.evaluation.protocol import build_eval_task, evaluate_model\n"
     "from retail_recommender.models import GlobalPopularity, CategoryPopularity, ItemItemCooccurrence\n"
     "\n"
     "val_task = build_eval_task(itx, cfg, 'val', 'strong')\n"
     "rows = []\n"
     "for cls in (GlobalPopularity, CategoryPopularity, ItemItemCooccurrence):\n"
     "    mdl = cls().fit(val_task, cfg)\n"
     "    m = evaluate_model(mdl, val_task, cfg)\n"
     "    rows.append({'model': mdl.name,\n"
     "                 'R@10 ALL': m['ALL']['recall@10'], 'NDCG@10 ALL': m['ALL']['ndcg@10'],\n"
     "                 'R@10 WARM': m['WARM']['recall@10'], 'NDCG@10 WARM': m['WARM']['ndcg@10'],\n"
     "                 'cov@10': m['ALL']['coverage@10']})\n"
     "baseline_live = pd.DataFrame(rows).set_index('model').round(4)\n"
     "baseline_live"),
    ("code",
     "# те же числа из сохранённого прогона экспериментов — должны совпасть\n"
     "pd.DataFrame({k: v['val']['ALL'] for k, v in M['baselines'].items()}).T[\n"
     "    ['recall@10','ndcg@10','coverage@10']].round(4)"),

    ("md",
     "## 4. Какие модели сравниваем и почему\n\n"
     "| модель | зачем |\n"
     "|---|---|\n"
     "| **global / category / item-item** | нижняя планка, контрольная точка |\n"
     "| **ALS** (implicit) | матричная факторизация implicit feedback — должна "
     "аккуратнее ранжировать warm-пользователей, чем co-occurrence |\n"
     "| **BPR** (implicit) | попарное ранжирование — доп. эксперимент |\n\n"
     "Все модели: обучение только на train, оценка одним протоколом, **на полном "
     "train-каталоге** (candidate pool не сужаем — implicit скорит весь каталог "
     "за миллисекунды на пользователя).\n\n"
     "> LightFM рассматривался как гибридный кандидат, но в текущей Windows-среде "
     "скомпилированное C-расширение нестабильно при обучении, поэтому в итоговое "
     "сравнение вошли ALS и BPR.\n\n"
     "TOP-20 property-кодов товара всё равно посчитаны "
     "(`data/interim/top_property_codes.json`) — это отдельная часть анализа; "
     "они предназначались для side-features гибрида."),

    ("md", "## 5. Эксперименты (validation)\n\n"
     "Сетка по factors / regularization / iterations. Метрика отбора — "
     "Recall@10 (ALL / strong / val), tie-break NDCG@10."),
    ("code",
     "def exp_table(family):\n"
     "    d = M['val_experiments'][family]\n"
     "    r = []\n"
     "    for name, e in d.items():\n"
     "        r.append({'config': name,\n"
     "                  'R@10 ALL': e['ALL']['recall@10'], 'NDCG@10 ALL': e['ALL']['ndcg@10'],\n"
     "                  'R@10 WARM': e['WARM']['recall@10'], 'NDCG@10 WARM': e['WARM']['ndcg@10'],\n"
     "                  'cov@10': e['ALL']['coverage@10'], 'fit_sec': e.get('fit_sec')})\n"
     "    return pd.DataFrame(r).set_index('config').round(4)\n"
     "exp_table('als')"),
    ("code", "exp_table('bpr')"),

    ("md", "## 6. Сводная таблица качества\n\nЛучшая конфигурация каждого семейства "
     "(по validation) против лучшего baseline, на val и test."),
    ("code",
     "sel = M['selected']\n"
     "def summ(entry):\n"
     "    return {'R@10 ALL': entry['ALL']['recall@10'], 'NDCG@10 ALL': entry['ALL']['ndcg@10'],\n"
     "            'R@10 WARM': entry['WARM']['recall@10'], 'R@10 COLD': entry['COLD']['recall@10'],\n"
     "            'MAP@10': entry['ALL']['map@10'], 'cov@10': entry['ALL']['coverage@10']}\n"
     "\n"
     "val_rows, test_rows = {}, {}\n"
     "bb = sel['best_baseline']\n"
     "val_rows[f'baseline:{bb}'] = summ(M['baselines'][bb]['val'])\n"
     "test_rows[f'baseline:{bb}'] = summ(M['baselines'][bb]['test'])\n"
     "for fam in ('als','bpr'):\n"
     "    ckey = sel[fam]\n"
     "    val_rows[f'{fam}:{ckey}'] = summ(M['val_experiments'][fam][ckey])\n"
     "for label, s in M['test'].items():\n"
     "    test_rows[label] = summ(s)\n"
     "print('VALIDATION'); display(pd.DataFrame(val_rows).T.round(4))\n"
     "print('TEST'); display(pd.DataFrame(test_rows).T.round(4))"),

    ("md", "## 7. Лучшая конфигурация\n"),
    ("code",
     "for k in ('best_baseline','als','bpr','overall_by_val'):\n"
     "    print(f'{k:16}: {sel[k]}')\n"
     "print('\\nкритерий отбора:', sel['criterion'])"),

    ("md", "## 8. Финальная оценка на test\n\n"
     "Один замер выбранных по validation конфигураций. Test для подбора "
     "гиперпараметров не использовался."),
    ("code",
     "test_df = pd.DataFrame({k: {'R@10 ALL': v['ALL']['recall@10'],\n"
     "                            'NDCG@10 ALL': v['ALL']['ndcg@10'],\n"
     "                            'R@10 WARM': v['WARM']['recall@10'],\n"
     "                            'NDCG@10 WARM': v['WARM']['ndcg@10'],\n"
     "                            'R@10 COLD': v['COLD']['recall@10'],\n"
     "                            'cov@10': v['ALL']['coverage@10']}\n"
     "                        for k, v in M['test'].items()}).T.round(4)\n"
     "test_df"),
    ("code",
     "base_test = M['baselines'][bb]['test']['ALL']['recall@10']\n"
     "print(f'лучший baseline (test, R@10 ALL) = {base_test:.4f}\\n')\n"
     "for label, s in M['test'].items():\n"
     "    r = s['ALL']['recall@10']; d = r - base_test\n"
     "    rel = d / base_test * 100 if base_test else float('nan')\n"
     "    print(f'{label:44} R@10={r:.4f}  Δ={d:+.4f} ({rel:+.1f}%)')"),

    ("md", "## 8b. TOP-20 property-кодов товара\n\n"
     "Метод — частота строк в `item_properties` "
     "среди срезов `snapshot_ts <= cutoff` (не хардкод, `scripts/build_top_properties.py`). "
     "Два списка:\n"
     "- **A** — cutoff `2015-07-01` (feature cutoff проекта) — основной для генератора факторов;\n"
     "- **B** — cutoff `2015-08-22` (начало validation) — train-window диагностика для этого этапа.\n\n"
     "ALS/BPR TOP-20 не использовали, поэтому списки только фиксируются."),
    ("code",
     "tp = M['top_properties']\n"
     "A, B = tp['primary'], tp['val_window_diagnostic']\n"
     "print(f\"A @ {A['cutoff']}: {A['top_codes']}\")\n"
     "print(f\"B @ {B['cutoff']}: {B['top_codes']}\")\n"
     "print()\n"
     "print('списки идентичны:', tp['lists_identical'],\n"
     "      '| только в A:', tp['only_in_primary'], '| только в B:', tp['only_in_val_window'])\n"
     "assert len(A['top_codes']) == 20 and len(B['top_codes']) == 20\n"
     "pd.DataFrame({'A (2015-07-01)': A['top_codes'], 'B (2015-08-22)': B['top_codes']})"),

    ("md", "## 9. Выводы\n\n"
     "**Новые модели не побили baseline на test.** По validation ALS и BPR слегка "
     "опережали co-occurrence на warm-сегменте (R@10 ~0.019–0.020 против ~0.015), "
     "и по этому критерию были выбраны. На test картина развернулась: "
     "co-occurrence — R@10 ALL **0.0116**, WARM **0.0221**; ALS — 0.0101 / 0.0122; "
     "BPR — 0.0101 / 0.0118. То есть на test ALS/BPR примерно на 12–13% хуже "
     "baseline по ALL и на ~45% хуже по WARM.\n\n"
     "**Почему так:**\n"
     "- warm-сегмент крошечный (val 472, test 495 пользователей), у каждого обычно "
     "1–2 релевантных товара из каталога 214k — метрика на нём шумная, и зазор "
     "val→test у ALS/BPR (0.019→0.012) укладывается в этот шум;\n"
     "- 86% target-пользователей холодные — для них все модели дают один и тот же "
     "popularity-fallback, поэтому разница по ALL мала и определяется меньшинством;\n"
     "- ALS даёт почти одинаковый результат при любых factors/regularization "
     "(0.0094 на val) — implicit-сигнала в истории ~470 warm-пользователей мало, "
     "чтобы факторизация извлекла устойчивое ранжирование;\n"
     "- BPR при learning_rate=0.05 разваливается (WARM R@10 ~0.006);\n"
     "- co-occurrence выигрывает именно там, где есть короткая история "
     "(сегмент hist=2–4 на test: R@10 0.057 против 0.023 у ALS) — прямая "
     "«вместе смотрят / вместе покупают» связь на этих данных сильнее латентных "
     "факторов.\n\n"
     "**Вывод:** на текущем объёме warm-взаимодействий матричная "
     "факторизация не даёт выигрыша над explainable-baseline; честный лучший "
     "результат на test — **item-item co-occurrence, Recall@10 = 0.0116 (WARM "
     "0.0221)**. Резерв — контентные/сессионные подходы для холодных (LightFM "
     "здесь и задумывался, но не запустился в среде).\n\n"
     "TOP-20 property-кодов зафиксированы в `data/interim/top_property_codes.json` "
     "— два списка (основной при feature cutoff 2015-07-01, диагностический при "
     "2015-08-22)."),
]


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s)
        for k, s in CELLS
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


if __name__ == "__main__":
    out = ROOT / "notebooks" / "03_models.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(build(), out)
    print(f"написан {out} ({len(CELLS)} ячеек)")
