"""Формат артефакта модели: упаковка/распаковка sparse-структур.

Используется и при экспорте (``scripts/train_service_model.py``), и при загрузке
в сервисе — чтобы формат был описан в одном месте.
"""
from __future__ import annotations

import numpy as np


def neighbours_to_arrays(
    neighbours: dict[int, list[tuple[int, float]]],
) -> dict[str, np.ndarray]:
    """dict{item: [(nb, sim), ...]} -> CSR-подобные массивы."""
    item_ids = np.array(sorted(neighbours), dtype=np.int64)
    offsets = np.zeros(len(item_ids) + 1, dtype=np.int64)
    nbr_ids: list[int] = []
    nbr_sims: list[float] = []
    for k, it in enumerate(item_ids):
        pairs = neighbours[int(it)]
        offsets[k + 1] = offsets[k] + len(pairs)
        for nb, sim in pairs:
            nbr_ids.append(int(nb))
            nbr_sims.append(float(sim))
    return {
        "neighbour_item_ids": item_ids,
        "neighbour_offsets": offsets,
        "neighbour_ids": np.array(nbr_ids, dtype=np.int64),
        "neighbour_sims": np.array(nbr_sims, dtype=np.float32),
    }


class ArrayNeighbours:
    """dict-подобный доступ к соседям товара поверх плоских массивов.

    Реализует ровно то, что использует ``ItemItemCooccurrence.recommend``:
    ``.get(item, ())`` и проверку истинности. В python-dict на миллионы пар не
    разворачивается — в памяти только массивы.
    """

    def __init__(
        self,
        item_ids: np.ndarray,
        offsets: np.ndarray,
        nbr_ids: np.ndarray,
        nbr_sims: np.ndarray,
    ) -> None:
        self._item_ids = item_ids
        self._offsets = offsets
        self._nbr_ids = nbr_ids
        self._nbr_sims = nbr_sims

    def get(self, item: int, default=()):  # noqa: ANN001 - drop-in для dict.get
        idx = int(np.searchsorted(self._item_ids, item))
        if idx >= len(self._item_ids) or int(self._item_ids[idx]) != int(item):
            return default
        s, e = int(self._offsets[idx]), int(self._offsets[idx + 1])
        return list(zip(self._nbr_ids[s:e].tolist(), self._nbr_sims[s:e].tolist()))

    def __len__(self) -> int:
        return int(len(self._item_ids))

    def __bool__(self) -> bool:
        return len(self._item_ids) > 0
