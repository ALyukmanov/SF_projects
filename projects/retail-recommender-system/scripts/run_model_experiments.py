"""Цикл экспериментов с моделями и выбор лучшей.

Запуск: python scripts/run_model_experiments.py   (нужен venv с numpy<2 и implicit)

Протокол (без утечки из test):
  TRAIN       — обучение (только train-окно, ts <= 2015-08-21);
  VALIDATION  — выбор модели / factors / regularization / iterations;
  TEST        — один финальный замер выбранных конфигураций.

Все модели сравниваются ОДНИМ протоколом (``evaluation/protocol.py``), режим
strong (addtocart ∪ transaction), основная метрика recall@10 / ndcg@10, политика
исключения уже виденных train-товаров не меняется между моделями. Оценка идёт по
ПОЛНОМУ train-каталогу — candidate pool не сужается.

LightFM рассматривался как гибридный кандидат, но в текущей Windows-среде его
скомпилированное C-расширение нестабильно при обучении (access violation в
fit_warp/fit_bpr на numpy 1.x и 2.x). Поэтому в сравнение вошли ALS и BPR.
Обёртка LightFM в проект не включена как необязательная, нерабочая в этой среде
зависимость.

Пишет: reports/week3_metrics.json, reports/week3_model_results.md
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import _bootstrap  # noqa: F401,E402
import numpy as np  # noqa: E402

from retail_recommender.config import load_config  # noqa: E402
from retail_recommender.evaluation.protocol import build_eval_task, evaluate_model  # noqa: E402
from retail_recommender.models import (  # noqa: E402
    CategoryPopularity,
    GlobalPopularity,
    ItemItemCooccurrence,
)
from retail_recommender.models.als import ALSRecommender, BPRRecommender  # noqa: E402
from retail_recommender.preprocessing.interactions import load_interactions  # noqa: E402

PRIMARY = "recall@10"
SECONDARY = "ndcg@10"

ALS_GRID = [
    {"factors": f, "regularization": r, "iterations": 20}
    for f in (32, 64)
    for r in (0.01, 0.1, 1.0)
] + [
    {"factors": 64, "regularization": 0.1, "iterations": 40},
]

BPR_GRID = [
    {"factors": f, "learning_rate": lr, "regularization": 0.01, "iterations": 150}
    for f in (32, 64)
    for lr in (0.01, 0.05)
]

SEGMENTS = ("ALL", "WARM", "hist=0", "hist=1", "hist=2-4", "hist>=5")
SEG_LABEL = {"hist=0": "COLD"}


def _summary(metrics: dict) -> dict:
    out = {}
    for seg in SEGMENTS:
        m = metrics.get(seg, {})
        out[SEG_LABEL.get(seg, seg)] = {
            "recall@10": round(m.get("recall@10", 0.0), 5),
            "ndcg@10": round(m.get("ndcg@10", 0.0), 5),
            "recall@20": round(m.get("recall@20", 0.0), 5),
            "ndcg@20": round(m.get("ndcg@20", 0.0), 5),
            "hit_rate@10": round(m.get("hit_rate@10", 0.0), 5),
            "map@10": round(m.get("map@10", 0.0), 5),
            "coverage@10": round(m.get("coverage@10", 0.0), 6),
            "n_users": int(m.get("n_users_evaluated", 0)),
        }
    return out


def _val_score(entry: dict) -> tuple[float, float]:
    a = entry["ALL"]
    return (a["recall@10"], a["ndcg@10"])


def run(quick: bool = False) -> int:
    cfg = load_config()
    np.random.seed(cfg.random_seed)
    itx = load_interactions(cfg)

    val_task = build_eval_task(itx, cfg, "val", "strong")
    test_task = build_eval_task(itx, cfg, "test", "strong")

    metrics_out: dict = {
        "protocol": {
            "mode": "strong",
            "primary": PRIMARY,
            "secondary": SECONDARY,
            "candidate_pool": "full train catalog (не сужается)",
            "filter_seen": bool(cfg.evaluation["filter_seen"]),
            "event_weights": cfg.event_weights,
            "confidence": "1 + Σ(веса событий пары) — одинаково для ALS и BPR",
            "split": {
                "train_end": cfg.split.train_end,
                "val": [cfg.split.val_start, cfg.split.val_end],
                "test": [cfg.split.test_start, cfg.split.test_end],
            },
            "n_target_users": {
                "val": len(val_task.target_users),
                "test": len(test_task.target_users),
            },
            "n_warm_users": {
                "val": len(val_task.warm_users),
                "test": len(test_task.warm_users),
            },
            "catalog_size": len(val_task.catalog),
            "lightfm_note": (
                "LightFM рассматривался как гибридный кандидат, но в текущей "
                "Windows-среде скомпилированное C-расширение нестабильно при "
                "обучении, поэтому в итоговое сравнение вошли ALS и BPR."
            ),
        },
        "baselines": {},
        "val_experiments": {"als": {}, "bpr": {}},
        "selected": {},
        "test": {},
        "timings_sec": {},
    }

    # ---------------------------------------------------------------- baselines
    print("== baselines (тот же протокол, замороженные веса 1-3-5, полный каталог) ==")
    baseline_classes = {
        "global_popularity": GlobalPopularity,
        "category_popularity": CategoryPopularity,
        "item_item_cooccurrence": ItemItemCooccurrence,
    }
    fitted_baselines = {}
    for name, cls in baseline_classes.items():
        t0 = time.time()
        m = cls().fit(val_task, cfg)
        metrics_out["timings_sec"][f"baseline:{name}:fit"] = round(time.time() - t0, 2)
        fitted_baselines[name] = m
        vm = evaluate_model(m, val_task, cfg)
        tm = evaluate_model(m, test_task, cfg)
        metrics_out["baselines"][name] = {"val": _summary(vm), "test": _summary(tm)}
        print(f"  {name:24} val R@10 ALL={vm['ALL']['recall@10']:.4f} "
              f"WARM={vm['WARM']['recall@10']:.4f}  cov@10={vm['ALL']['coverage@10']:.4f}")

    best_baseline = max(
        metrics_out["baselines"],
        key=lambda k: _val_score(metrics_out["baselines"][k]["val"]),
    )

    # ---------------------------------------------------------------- ALS / BPR
    families = {
        "als": (ALSRecommender, ALS_GRID[:2] if quick else ALS_GRID),
        "bpr": (BPRRecommender, BPR_GRID[:2] if quick else BPR_GRID),
    }
    fitted: dict[str, dict] = {"als": {}, "bpr": {}}
    for fam, (cls, grid) in families.items():
        print(f"\n== {fam.upper()} (validation, полный каталог) ==")
        for params in grid:
            m = cls(random_state=cfg.random_seed, **params)
            t0 = time.time()
            m.fit(val_task, cfg)
            dt = round(time.time() - t0, 2)
            t0 = time.time()
            vm = evaluate_model(m, val_task, cfg)
            ev = round(time.time() - t0, 2)
            fitted[fam][m.config_str] = m
            metrics_out["val_experiments"][fam][m.config_str] = {
                "params": params, "fit_sec": dt, "eval_sec": ev, **_summary(vm),
            }
            print(f"  {m.config_str:44} R@10 ALL={vm['ALL']['recall@10']:.4f} "
                  f"WARM={vm['WARM']['recall@10']:.4f} NDCG@10 ALL={vm['ALL']['ndcg@10']:.4f} "
                  f"cov={vm['ALL']['coverage@10']:.4f} ({dt}s fit)")

    best = {
        fam: max(
            metrics_out["val_experiments"][fam],
            key=lambda k: _val_score(metrics_out["val_experiments"][fam][k]),
        )
        for fam in ("als", "bpr")
    }
    metrics_out["selected"] = {
        "best_baseline": best_baseline,
        "als": best["als"],
        "bpr": best["bpr"],
        "criterion": f"max {PRIMARY} на ALL/strong/val, tie-break {SECONDARY}",
    }

    # ---------------------------------------------------------------- TEST
    print("\n== TEST (один замер выбранных по validation конфигураций) ==")
    test_targets = {
        best_baseline: fitted_baselines[best_baseline],
        f"als[{best['als']}]": fitted["als"][best["als"]],
        f"bpr[{best['bpr']}]": fitted["bpr"][best["bpr"]],
    }
    for label, m in test_targets.items():
        tm = evaluate_model(m, test_task, cfg)
        metrics_out["test"][label] = _summary(tm)
        print(f"  {label:40} R@10 ALL={tm['ALL']['recall@10']:.4f} "
              f"WARM={tm['WARM']['recall@10']:.4f} NDCG@10 ALL={tm['ALL']['ndcg@10']:.4f}")

    overall = max(
        [
            ("baseline:" + best_baseline, metrics_out["baselines"][best_baseline]["val"]),
            ("als:" + best["als"], metrics_out["val_experiments"]["als"][best["als"]]),
            ("bpr:" + best["bpr"], metrics_out["val_experiments"]["bpr"][best["bpr"]]),
        ],
        key=lambda kv: _val_score(kv[1]),
    )
    metrics_out["selected"]["overall_by_val"] = overall[0]

    tp_path = cfg.paths.interim_dir / "top_property_codes.json"
    if tp_path.exists():
        metrics_out["top_properties"] = json.loads(tp_path.read_text("utf-8"))

    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.paths.reports_dir / "week3_metrics.json").write_text(
        json.dumps(metrics_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(cfg, metrics_out)
    print("\nOK -> reports/week3_metrics.json, reports/week3_model_results.md")
    print("overall (по val):", overall[0])
    return 0


def _write_markdown(cfg, mo: dict) -> None:
    p = mo["protocol"]
    L = ["# Эксперименты с моделями\n"]
    L.append(
        f"Режим **strong** (addtocart ∪ transaction). Основная метрика "
        f"**{p['primary']} / {p['secondary']}**, K=10 (+20 диагностически), "
        f"дополнительно Coverage@10, HitRate@10, MAP@10. Политика исключения уже "
        f"виденных train-товаров `filter_seen={p['filter_seen']}` — единая для всех "
        f"моделей. Веса событий {p['event_weights']} (зафиксированы ранее).\n"
    )
    L.append(
        f"Split: train ≤ {p['split']['train_end']}, val {p['split']['val'][0]}…"
        f"{p['split']['val'][1]}, test {p['split']['test'][0]}…{p['split']['test'][1]}. "
        f"Каталог (train) {p['catalog_size']} товаров. Target-пользователи: "
        f"val {p['n_target_users']['val']} (warm {p['n_warm_users']['val']}), "
        f"test {p['n_target_users']['test']} (warm {p['n_warm_users']['test']}).\n"
    )
    L.append(
        f"**Candidate pool:** {p['candidate_pool']}. Модели оцениваются на всём "
        f"допустимом train-каталоге — это вычислительно посильно (implicit скорит "
        f"весь каталог за миллисекунды на пользователя), поэтому пул не сужаем.\n"
    )
    L.append(
        f"Отбор конфигураций — **по validation** ({mo['selected']['criterion']}). "
        f"TEST замерян один раз для выбранных конфигураций, для подбора не "
        f"использовался. Confidence-матрица: {p['confidence']}.\n"
    )
    L.append(f"> {p['lightfm_note']}\n")

    def row(name, s):
        a, w, c = s["ALL"], s["WARM"], s["COLD"]
        return (
            f"| {name} | {a['recall@10']:.4f} | {a['ndcg@10']:.4f} | {a['coverage@10']:.5f} "
            f"| {a['map@10']:.4f} | {a['hit_rate@10']:.4f} | {w['recall@10']:.4f} "
            f"| {w['ndcg@10']:.4f} | {c['recall@10']:.4f} |"
        )

    hdr = (
        "| модель | R@10 ALL | NDCG@10 ALL | cov@10 | MAP@10 | Hit@10 "
        "| R@10 WARM | NDCG@10 WARM | R@10 COLD |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )

    L.append("## Validation — baseline\n")
    L.append(hdr)
    for name, d in mo["baselines"].items():
        L.append(row(name, d["val"]))
    L.append("")

    for fam, title in (("als", "ALS"), ("bpr", "BPR")):
        L.append(f"## Validation — {title}\n")
        L.append(hdr)
        for cname, d in mo["val_experiments"][fam].items():
            L.append(row(cname, d))
        L.append("")

    L.append("## Выбрано по validation\n")
    for k in ("best_baseline", "als", "bpr", "overall_by_val"):
        L.append(f"- **{k}**: `{mo['selected'][k]}`")
    L.append("")

    L.append("## TEST — финальный замер\n")
    L.append(hdr)
    for name, s in mo["test"].items():
        L.append(row(name, s))
    L.append("")

    bb = mo["baselines"][mo["selected"]["best_baseline"]]["test"]["ALL"]
    bw = mo["baselines"][mo["selected"]["best_baseline"]]["test"]["WARM"]
    L.append("## Прирост относительно лучшего baseline (TEST)\n")
    L.append(
        f"Лучший baseline по val — **{mo['selected']['best_baseline']}** "
        f"(test recall@10: ALL {bb['recall@10']:.4f}, WARM {bw['recall@10']:.4f}).\n"
    )
    L.append("| модель | R@10 ALL | Δ абс ALL | Δ отн ALL | R@10 WARM | Δ отн WARM |\n|---|---|---|---|---|---|")
    for name, s in mo["test"].items():
        ra, rw = s["ALL"]["recall@10"], s["WARM"]["recall@10"]
        da = ra - bb["recall@10"]
        rela = (da / bb["recall@10"] * 100) if bb["recall@10"] else float("nan")
        relw = ((rw - bw["recall@10"]) / bw["recall@10"] * 100) if bw["recall@10"] else float("nan")
        L.append(f"| {name} | {ra:.4f} | {da:+.4f} | {rela:+.1f}% | {rw:.4f} | {relw:+.1f}% |")
    L.append("")

    tp_path = cfg.paths.interim_dir / "top_property_codes.json"
    if tp_path.exists():
        tp = json.loads(tp_path.read_text("utf-8"))
        pr, di = tp["primary"], tp["val_window_diagnostic"]
        L.append("## TOP-20 property-кодов товара\n")
        L.append(
            "Метод одинаков для обоих списков — частота строк в `item_properties` "
            "среди срезов с `snapshot_ts <= cutoff`. Коды в RetailRocket захешированы, "
            "кроме литералов `categoryid` и `available`. Списки получены из "
            "данных скриптом `scripts/build_top_properties.py`, не хардкодом.\n"
        )
        L.append(
            f"**A. Основной TOP-20 — cutoff = {pr['cutoff']}** (зафиксированная точка "
            f"отсечения генератора факторов проекта). Различных кодов: "
            f"{pr['n_distinct_codes']}.\n"
        )
        L.append("| # | код | строк <= cutoff |\n|---|---|---|")
        for i, code in enumerate(pr["top_codes"], 1):
            L.append(f"| {i} | `{code}` | {pr['top_counts'][code]} |")
        L.append("")
        L.append(
            f"**B. Диагностический TOP-20 — cutoff = {di['cutoff']}** (начало "
            f"validation; train-window для экспериментов с моделями, ALS/BPR его не "
            f"использовали). Различных кодов: {di['n_distinct_codes']}.\n"
        )
        L.append("| # | код | строк <= cutoff |\n|---|---|---|")
        for i, code in enumerate(di["top_codes"], 1):
            L.append(f"| {i} | `{code}` | {di['top_counts'][code]} |")
        L.append("")
        L.append(
            f"Списки идентичны: **{tp['lists_identical']}**. Только в A: "
            f"`{tp['only_in_primary']}`; только в B: `{tp['only_in_val_window']}` "
            f"(различие в хвосте — коды 19–20 и порядок 10/14; первые 9 совпадают). "
            f"Один список другим не подменяется.\n"
        )

    L.append("## Тайминги\n")
    L.append("baseline fit (сек): " + ", ".join(
        f"{k.split(':')[1]} {v}" for k, v in mo["timings_sec"].items()))
    for fam in ("als", "bpr"):
        fits = [d.get("fit_sec", 0) for d in mo["val_experiments"][fam].values()]
        if fits:
            L.append(f"{fam.upper()} fit (сек): min {min(fits)}, max {max(fits)}, "
                     f"суммарно {round(sum(fits))} по {len(fits)} конфигурациям")
    L.append("")
    L.append("_Полные числа (val и test, все сегменты, @10 и @20) — "
             "`reports/week3_metrics.json`._")

    (cfg.paths.reports_dir / "week3_model_results.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="усечённая сетка для дымового прогона")
    ap.add_argument("--md-only", action="store_true",
                    help="только пересобрать week3_model_results.md из существующего JSON")
    args = ap.parse_args()
    if args.md_only:
        _cfg = load_config()
        _mo = json.loads((_cfg.paths.reports_dir / "week3_metrics.json").read_text("utf-8"))
        _write_markdown(_cfg, _mo)
        print("week3_model_results.md пересобран")
        raise SystemExit(0)
    raise SystemExit(run(quick=args.quick))
