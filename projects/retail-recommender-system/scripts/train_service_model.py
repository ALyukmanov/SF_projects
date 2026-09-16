"""Обучение и экспорт модели для сервиса рекомендаций.

Запуск: python scripts/train_service_model.py [--out artifacts/model]

Строит выбранную модель (item-item co-occurrence) на train-окне и сохраняет всё
необходимое для инференса:

  artifacts/model/model.npz   — соседи товара (sparse, top-N на товар),
                                популярность (fallback), истории визиторов;
  artifacts/model/meta.json   — тип модели, версия, время сборки, метрики на test.

Инференс в сервисе использует ту же логику, что и обучение —
``retail_recommender.models.cooccurrence.ItemItemCooccurrence.recommend``. Отдельной
реализации алгоритма нет.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
import numpy as np

from retail_recommender.config import load_config
from retail_recommender.evaluation.protocol import build_eval_task
from retail_recommender.models.cooccurrence import ItemItemCooccurrence
from retail_recommender.preprocessing.interactions import load_interactions
from retail_recommender.service.artifact import neighbours_to_arrays

MODEL_TYPE = "item_item_cooccurrence"
GLOBAL_FALLBACK_TOPK = 3000   # хватает для n<=100 даже после исключения истории
DEFAULT_N = 10
MAX_N = 100


def _user_histories(history_events) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Истории всех не-бот визиторов до начала test-окна, в CSR-подобном виде."""
    ev = history_events.sort_values(["visitorid", "timestamp"])
    user_ids = np.sort(ev["visitorid"].unique()).astype(np.int64)
    sizes = ev.groupby("visitorid", sort=True).size().to_numpy()
    offsets = np.zeros(len(user_ids) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes)
    hist_items = ev["itemid"].to_numpy().astype(np.int64)
    hist_weights = ev["weight"].to_numpy().astype(np.int16)
    return user_ids, offsets, hist_items, hist_weights


def _test_metrics(cfg) -> dict:
    """Метрики выбранной модели на test из отчёта экспериментов (если он есть)."""
    path = cfg.paths.reports_dir / "week3_metrics.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text("utf-8"))
    t = data.get("test", {}).get(MODEL_TYPE, {})
    if not t:
        return {}
    return {
        "recall@10": t["ALL"]["recall@10"],
        "ndcg@10": t["ALL"]["ndcg@10"],
        "recall@10_warm": t["WARM"]["recall@10"],
        "coverage@10": t["ALL"]["coverage@10"],
        "n_test_users": t["ALL"]["n_users"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Экспорт модели рекомендаций для сервиса.")
    ap.add_argument("--out", default="artifacts/model", help="каталог для артефакта")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = (cfg.paths.root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] загрузка processed-данных ...")
    itx = load_interactions(cfg)
    task = build_eval_task(itx, cfg, "test", "strong")

    print(f"[2/4] обучение {MODEL_TYPE} на train-окне ...")
    model = ItemItemCooccurrence().fit(task, cfg)

    print("[3/4] сериализация ...")
    nb_arrays = neighbours_to_arrays(model.neighbours)
    u_ids, u_offsets, u_items, u_weights = _user_histories(task.history_events)
    global_ranked = np.array(model.global_ranked[:GLOBAL_FALLBACK_TOPK], dtype=np.int64)
    catalog = np.array(sorted(task.catalog), dtype=np.int64)

    npz_path = out_dir / "model.npz"
    np.savez_compressed(
        npz_path,
        **nb_arrays,
        global_ranked=global_ranked,
        catalog=catalog,
        user_ids=u_ids,
        user_offsets=u_offsets,
        user_hist_items=u_items,
        user_hist_weights=u_weights,
    )

    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    meta = {
        "model_type": MODEL_TYPE,
        "version": version,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query_items": int(model._query_items),
        "filter_seen": bool(cfg.evaluation["filter_seen"]),
        "default_n": DEFAULT_N,
        "max_n": MAX_N,
        "event_weights": cfg.event_weights,
        "split": {
            "train_end": cfg.split.train_end,
            "history_before": cfg.split.test_start,
            "test": [cfg.split.test_start, cfg.split.test_end],
        },
        "catalog_size": len(task.catalog),
        "n_items_with_neighbours": int(len(nb_arrays["neighbour_item_ids"])),
        "n_neighbour_pairs": int(len(nb_arrays["neighbour_ids"])),
        "n_users_with_history": int(len(u_ids)),
        "n_history_events": int(len(u_items)),
        "global_fallback_size": int(len(global_ranked)),
        "test_metrics": _test_metrics(cfg),
        "artifact_files": ["model.npz", "meta.json"],
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    size_mb = npz_path.stat().st_size / 1024 / 1024
    print(f"[4/4] готово: {npz_path}  ({size_mb:.1f} МБ)")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
