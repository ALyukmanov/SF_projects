"""
Promotion gate for the geo model — PROMOTION_READY / PROMOTION_NOT_RECOMMENDED.

Usage: python scripts/geo_promotion_gate.py [--full-tests]

A RECOMMENDATION only. Never writes models/current_model.json. Reads the
evidence produced by the other scripts and applies an explicit gate:

  1. Location holdout MAE(geo winner) < MAE(production pre-geo).
  2. Location holdout MAPE(geo winner) not worse than production pre-geo
     (<= +0.3 pp tolerance).
  3. No city regression — per-city holdout MAE(geo) <= MAE(pre-geo) * 1.02
     for BOTH Moscow and Saint Petersburg.
  4. has_coordinates=0 rows not materially hurt — robust error (median AE)
     and MAPE for that subset within +10% / +3 pp of the pre-geo model.
     (Mean MAE on this ~8% subset is tail-dominated on ~160 rows and is
     reported but not gated.)
  5. No leakage — the feature matrix carries no identifier / target-derived
     column, AND verify_split_leakage.py reports 0 groups crossing the
     location split.
  6. Training -> artefact -> Predictor feature schema agree — the live
     pipeline's expected feature list (promote_model._expected_feature_names,
     with geo) equals the location split's feature_names and contains every
     GEO_FEATURE_COLUMNS entry.
  7. Inference parity + geo Predictor tests pass.
  8. (with --full-tests) the whole test suite passes.
  9. The candidate trains reproducibly from the repo (seed fixed, cleaned
     CSV regenerable) — asserted structurally, see note in output.

No fixed "must beat by X%" threshold: a real, stable improvement with no
regression is the bar.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.data.split_pipeline import split_impute_featurize
from src.features.geo_features import GEO_FEATURE_COLUMNS

_REPORTS = _PROJECT_ROOT / "reports"


def _load(name: str) -> dict | None:
    p = _REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _run_pytest(targets: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *targets],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    last = (proc.stdout.strip().splitlines() or ["(no output)"])[-1]
    return proc.returncode == 0, last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-tests", action="store_true", help="Also run the full test suite.")
    args = parser.parse_args()

    checks: list[tuple[str, bool, str]] = []

    cand = _load("geo_final_candidates.json")
    if not cand:
        print("ERROR: reports/geo_final_candidates.json missing — run compare_geo_uplift.py first.")
        raise SystemExit(2)

    prim = cand["primary_evaluation"]
    cands = prim["candidates"]
    pre = cands["production_pre_geo"]["metrics"]
    deep = cand["winner_deep_dive_location"]
    winner_name = deep["winner"]
    win = cands[winner_name]["metrics"]

    # 1. MAE
    checks.append((
        "Location holdout MAE(geo winner) < MAE(production pre-geo)",
        win["mae"] < pre["mae"],
        f"{win['mae']:,.0f} vs {pre['mae']:,.0f}  ({100 * (win['mae'] / pre['mae'] - 1):+.1f}%)",
    ))

    # 2. MAPE
    checks.append((
        "Location holdout MAPE(geo winner) not worse than pre-geo (<= +0.3 pp)",
        win["mape"] <= pre["mape"] + 0.3,
        f"{win['mape']:.2f}% vs {pre['mape']:.2f}%",
    ))

    # 3. per-city
    city_ok = True
    city_detail = []
    for c, v in deep["per_city"].items():
        ok = v["winner_mae"] <= v["pre_geo_mae"] * 1.02
        city_ok = city_ok and ok
        city_detail.append(
            f"{c}: {v['pre_geo_mae']:,.0f}->{v['winner_mae']:,.0f} {'OK' if ok else 'REGRESSION'}"
        )
    checks.append(("No city regression (per-city MAE within +2%)", city_ok, "; ".join(city_detail)))

    # 4. has_coordinates=0 (robust): median AE within +10%, MAPE within +3 pp
    nc = deep["by_coordinate_availability"].get("has_coordinates_0")
    if nc:
        med_ok = nc["winner_median_ae"] <= nc["pre_geo_median_ae"] * 1.10
        mape_ok = nc["winner_mape"] <= nc["pre_geo_mape"] + 3.0
        mean_delta = 100 * (nc["winner_mae"] / nc["pre_geo_mae"] - 1)
        checks.append((
            "has_coordinates=0 rows not materially hurt (robust: median AE + MAPE)",
            med_ok and mape_ok,
            f"n={nc['n']}  median AE {nc['pre_geo_median_ae']:,.0f}->{nc['winner_median_ae']:,.0f}, "
            f"MAPE {nc['pre_geo_mape']:.1f}%->{nc['winner_mape']:.1f}%  "
            f"(mean MAE {mean_delta:+.0f}% — tail-sensitive on {nc['n']} rows, not gated)",
        ))
    else:
        checks.append(("has_coordinates=0 rows not materially hurt", True, "no such rows in holdout"))

    # 5. leakage
    suspicious = deep.get("leakage_scan", {}).get("suspicious_feature_names_in_matrix", [])
    leak = _load("leakage_reverification_real.json")
    loc_cross = (leak or {}).get("groups_crossing_location_split", {})
    cross_total = (
        loc_cross.get("near_duplicate_groups", 1)
        + loc_cross.get("same_coordinate_building_clusters", 1)
    )
    leak_ok = not suspicious and leak is not None and cross_total == 0
    checks.append((
        "No leakage (clean feature matrix + 0 groups cross the location split)",
        leak_ok,
        f"suspicious cols={suspicious or 'none'}; "
        + (
            f"location-split crossings near_dup={loc_cross.get('near_duplicate_groups')}, "
            f"coord={loc_cross.get('same_coordinate_building_clusters')}"
            if leak
            else "leakage_reverification_real.json missing — run verify_split_leakage.py"
        ),
    ))

    # 6. feature schema parity
    from scripts.promote_model import _expected_feature_names

    df = pd.read_csv(_PROJECT_ROOT / "data" / "processed" / "real_estate_cleaned.csv")
    split_feats = split_impute_featurize(df, split_strategy="location", random_state=42).feature_names
    expected = _expected_feature_names(with_geo=True)
    schema_ok = (
        split_feats == expected
        and all(c in expected for c in GEO_FEATURE_COLUMNS)
    )
    checks.append((
        "Training / artefact / Predictor feature schema agree (geo)",
        schema_ok,
        f"split has {len(split_feats)} features, live expected has {len(expected)}, "
        f"match={split_feats == expected}, all geo cols present={all(c in expected for c in GEO_FEATURE_COLUMNS)}",
    ))

    # 7. inference parity + geo predictor tests
    ip_ok, ip_last = _run_pytest([
        "tests/test_inference_parity.py",
        "tests/test_geo_features.py",
    ])
    checks.append(("Inference parity + geo Predictor tests pass", ip_ok, ip_last))

    # 8. full suite (optional)
    if args.full_tests:
        full_ok, full_last = _run_pytest(["tests/"])
        checks.append(("Full test suite passes", full_ok, full_last))

    # 9. reproducibility (structural)
    repro_ok = (
        (leak or {}).get("xgboost_params_source") is not None
        and cand.get("random_seed") == 42
    )
    checks.append((
        "Reproducible (fixed seed 42; cleaned CSV regenerable via run_feature_engineering_real.py)",
        repro_ok,
        f"seed={cand.get('random_seed')}, params_source={(leak or {}).get('xgboost_params_source')}",
    ))

    n_pass = sum(1 for _, ok, _ in checks if ok)
    verdict = "PROMOTION_READY" if n_pass == len(checks) else "PROMOTION_NOT_RECOMMENDED"

    print("=" * 78)
    print(f"GEO PROMOTION GATE — winner: {winner_name}")
    print("=" * 78)
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}")
    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}  ({n_pass}/{len(checks)} checks passed)")
    print("models/current_model.json was NOT modified. This is a recommendation only.")
    print("=" * 78)

    out = {
        "winner": winner_name,
        "verdict": verdict,
        "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in checks],
        "winner_metrics_location_holdout": win,
        "production_pre_geo_metrics_location_holdout": pre,
    }
    (_REPORTS / "geo_promotion_gate.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if verdict == "PROMOTION_READY":
        params_flag = (
            "--params-from-manifest"
            if winner_name == "geo_current_params"
            else "--params-from-study xgboost"
        )
        print(
            f"\nWinning candidate: {winner_name} "
            + (
                "(current production hyperparameters + geo features — re-tuning on the "
                "location split did NOT beat them)"
                if winner_name == "geo_current_params"
                else "(freshly tuned params)"
            )
            + "\n\nTo promote (only after the maintainer confirms):\n"
            f"  python scripts/run_model_training_real.py --model xgboost "
            f"--split-strategy location {params_flag}\n"
            "  python scripts/promote_model.py --candidate models/<the new artefact>.pkl --dry-run\n"
            "  python scripts/promote_model.py --candidate models/<the new artefact>.pkl\n"
        )


if __name__ == "__main__":
    main()
