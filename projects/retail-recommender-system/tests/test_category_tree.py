"""Проверки дерева категорий: цикл ловится, путь до корня строится."""
import pandas as pd

from retail_recommender.data import validate as V
from retail_recommender.preprocessing import categories as C


def test_cycle_detection(cfg, monkeypatch, tmp_path):
    cyclic = pd.DataFrame({"categoryid": [1, 2, 3], "parentid": pd.array([2, 1, pd.NA], dtype="Int64")})
    monkeypatch.setattr(V, "load_category_tree", lambda _cfg: cyclic)
    report = V.ValidationReport()
    V.validate_category_tree(cfg, report)
    names = {r.name: r.passed for r in report.results}
    assert names["category_tree: отсутствие циклов"] is False


def test_paths_to_root(cfg, monkeypatch):
    tree = pd.DataFrame({
        "categoryid": [10, 20, 30],
        "parentid": pd.array([20, 30, pd.NA], dtype="Int64"),
    })
    monkeypatch.setattr(C, "load_category_tree", lambda _cfg: tree)
    paths = C.build_category_paths(cfg).set_index("categoryid")
    assert paths.loc[10, "root"] == 30
    assert paths.loc[10, "depth"] == 2
    assert list(paths.loc[10, "path"]) == [10, 20, 30]
    assert paths.loc[30, "depth"] == 0
