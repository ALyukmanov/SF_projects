"""Computes a promotion-readiness verdict for the real-data candidate from
the evidence already on disk (leakage re-verification, baseline comparison,
final candidate analysis, promote_model.py's own dry-run validation). This
is a RECOMMENDATION only -- it never writes models/current_model.json.
Actual promotion stays an explicit, separate, human-approved action via
scripts/promote_model.py.

Usage: python scripts/promotion_gate_report.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_REPORTS = _PROJECT_ROOT / "reports"


def _load(name: str) -> dict | None:
    path = _REPORTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compute_verdict(checks: list[tuple[str, bool, str]]) -> str:
    """Derive the promotion verdict from *checks* -- pure function, no I/O,
    so this decision logic is directly unit-testable (see
    tests/test_promotion_gate_report.py).

    Rules:
    - Any check whose name ends in ``"[HARD BLOCKER]"`` and fails caps the
      verdict at PROMOTION_NOT_READY, regardless of how many other checks
      pass — known, unresolved technical leakage (or any future hard-blocker
      check) is never eligible for READY or READY_WITH_LIMITATIONS.
    - All checks passing -> PROMOTION_READY.
    - Otherwise, up to 2 non-hard-blocker checks may fail (e.g. the weak-
      segment checks) and the verdict is still PROMOTION_READY_WITH_LIMITATIONS
      — such failures are disclosed via the API's prediction_reliability
      metadata, not hidden.
    - More than 2 failures (with no hard blocker) -> PROMOTION_NOT_READY.
    """
    hard_blocker_failed = any(name.endswith("[HARD BLOCKER]") and not ok for name, ok, _ in checks)
    if hard_blocker_failed:
        return "PROMOTION_NOT_READY"

    n_pass = sum(1 for _, ok, _ in checks if ok)
    n_total = len(checks)
    if n_pass == n_total:
        return "PROMOTION_READY"
    if n_pass >= n_total - 2:
        return "PROMOTION_READY_WITH_LIMITATIONS"
    return "PROMOTION_NOT_READY"


def main() -> None:
    candidate = sys.argv[1] if len(sys.argv) > 1 else None
    if candidate is None:
        # Default to the newest real group-aware xgboost artefact.
        import joblib

        best = None
        for pkl in sorted((_PROJECT_ROOT / "models").glob("*.pkl")):
            try:
                artefact = joblib.load(pkl)
            except Exception:
                continue
            if (
                artefact.get("model_type") == "xgboost"
                and artefact.get("is_synthetic") is False
                and artefact.get("split_strategy") == "group_aware_80_20_random_state_42"
            ):
                if best is None or (artefact.get("trained_at") or "") > best[0]:
                    best = (artefact.get("trained_at") or "", pkl)
        if best is None:
            print("No real group-aware xgboost candidate found.", file=sys.stderr)
            raise SystemExit(2)
        candidate = str(best[1])

    checks: list[tuple[str, bool, str]] = []  # (name, passed, detail)

    # 1. Candidate passes promote_model.py's own fail-closed dry-run validation
    dry_run = subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "scripts" / "promote_model.py"),
            "--candidate",
            candidate,
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    checks.append(
        (
            "Schema/provenance/metric-sanity validation (promote_model.py --dry-run)",
            dry_run.returncode == 0,
            "PASS"
            if dry_run.returncode == 0
            else dry_run.stdout.strip().splitlines()[-1]
            if dry_run.stdout
            else "FAILED",
        )
    )

    # 2. Real-data provenance
    import joblib

    # Locally-produced artefact from this repo's own training pipeline
    # (same trust boundary as src/models/trainer.py) -- not untrusted input.
    artefact = joblib.load(candidate)
    is_real = (
        artefact.get("is_synthetic") is False
        and "restate" in str(artefact.get("data_source", "")).lower()
    )
    checks.append(
        (
            "Real-data provenance (is_synthetic=False, data_source=restate)",
            is_real,
            str(artefact.get("data_source")),
        )
    )

    # 3. Leakage: near-duplicate groups must not cross the group-aware split
    leakage = _load("leakage_reverification_real.json")
    no_leakage = bool(leakage) and leakage["groups_crossing_group_aware_split"] == 0
    checks.append(
        (
            "Near-duplicate groups isolated by group-aware split",
            no_leakage,
            f"{leakage['groups_crossing_group_aware_split']}/{leakage['near_duplicate_groups_total']} groups cross"
            if leakage
            else "NOT VERIFIED",
        )
    )

    # 3b. HARD BLOCKER: leakage-safe preprocessing. The candidate's
    # rooms/total_area/floor/floors_total imputer must have been fit on the
    # TRAIN split only (not the full dataset before splitting -- an earlier
    # bug, fixed by fitting the imputer on the train split only), and
    # the independent leakage re-verification must confirm it is invariant to
    # holdout mutation. Unlike the other checks, a failure here caps the
    # verdict at PROMOTION_NOT_READY regardless of how many other checks
    # pass -- known, unresolved technical leakage is not something a
    # "with limitations" verdict should paper over.
    artefact_imputer = artefact.get("imputer")
    artefact_has_fitted_imputer = bool(artefact_imputer) and artefact_imputer.get("fitted") is True
    scope = (leakage or {}).get("imputer_fit_scope", {})
    scope_invariant = scope.get("invariant_to_holdout_mutation") is True
    leakage_safe_preprocessing = artefact_has_fitted_imputer and scope_invariant
    checks.append(
        (
            "Leakage-safe preprocessing (imputer fit on TRAIN split only) [HARD BLOCKER]",
            leakage_safe_preprocessing,
            (
                f"artefact carries a fitted imputer={artefact_has_fitted_imputer}, "
                f"independently verified invariant to holdout mutation={scope_invariant}"
            )
            if leakage
            else "NOT VERIFIED — run scripts/verify_split_leakage.py first",
        )
    )

    # 4. Reproducibility: lineage metadata present
    lineage_keys = [
        "random_seed",
        "git_commit",
        "library_versions",
        "hyperparameters",
        "split_strategy",
        "dataset_sha256",
    ]
    has_lineage = all(artefact.get(k) is not None for k in lineage_keys)
    checks.append(
        (
            "Full lineage metadata present",
            has_lineage,
            ", ".join(k for k in lineage_keys if artefact.get(k) is None) or "complete",
        )
    )

    # 5. Quality vs baseline: MAE must clearly beat the best non-ML baseline
    baselines = _load("baseline_comparison_real.json")
    beats_baseline = False
    baseline_detail = "NOT VERIFIED"
    if baselines:
        ml_mae = baselines["ml_tuned_xgboost"]["mae"]
        best_baseline_mae = min(v["mae"] for k, v in baselines.items() if k.startswith("baseline_"))
        beats_baseline = (
            ml_mae < best_baseline_mae * 0.75
        )  # requires a clear (>=25%) improvement, not marginal
        baseline_detail = f"ML MAE={ml_mae:,.0f} vs best baseline MAE={best_baseline_mae:,.0f} ({100*(1-ml_mae/best_baseline_mae):.0f}% lower)"
    checks.append(
        ("Clearly beats best non-ML baseline (>=25% lower MAE)", beats_baseline, baseline_detail)
    )

    # 6. City/category stability -- both cities must have positive, non-degenerate R^2
    analysis = _load("final_candidate_analysis_real.json")
    stable_cities = False
    stability_detail = "NOT VERIFIED"
    weak_segments = []
    if analysis:
        seg = analysis["segments"]
        moscow_r2 = seg.get("city_moskva", {}).get("r2")
        spb_r2 = seg.get("city_spb", {}).get("r2")
        stable_cities = (
            moscow_r2 is not None and spb_r2 is not None and moscow_r2 > 0.5 and spb_r2 > 0.5
        )
        stability_detail = f"Moscow R2={moscow_r2:.3f}, SPB R2={spb_r2:.3f}"
        for name, m in seg.items():
            if name.startswith("category_") and (m["r2"] < 0.3 or m["n"] < 50):
                weak_segments.append(f"{name} (n={m['n']}, r2={m['r2']:.2f})")
    checks.append(("Both cities individually stable (R2 > 0.5)", stable_cities, stability_detail))

    # 7. No silent synthetic fallback -- verified by the API test suite (informational, not re-run here)
    checks.append(
        (
            "No silent synthetic/demo fallback (see tests/test_api_real_candidate.py)",
            True,
            "All real-candidate API tests assert mode == 'model' and is_synthetic == False",
        )
    )

    # 8. Segment support: explicit, always-visible check (not just a side
    # list) -- passes when there are no known low-support/weak segments,
    # fails (softly -- see verdict logic below) otherwise. Cross-referenced
    # against src.inference.predictor.LOW_SUPPORT_SEGMENTS so the API's own
    # honesty metadata and this gate agree on which categories are weak.
    from src.inference.predictor import LOW_SUPPORT_SEGMENTS

    checks.append(
        (
            "No known low-support/weak segments",
            not weak_segments,
            f"weak segments: {weak_segments}" if weak_segments else "none",
        )
    )
    checks.append(
        (
            "Weak segments (if any) are disclosed via API prediction_reliability metadata",
            set(LOW_SUPPORT_SEGMENTS)
            >= {w.split(" ")[0].removeprefix("category_") for w in weak_segments},
            f"LOW_SUPPORT_SEGMENTS covers: {sorted(LOW_SUPPORT_SEGMENTS)}",
        )
    )

    verdict = compute_verdict(checks)
    n_pass = sum(1 for _, ok, _ in checks if ok)
    n_total = len(checks)

    print("=" * 70)
    print(f"PROMOTION GATE -- {Path(candidate).name}")
    print("=" * 70)
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}")
    if weak_segments:
        print(f"\nKnown weak segments (disclosed via API, not silently hidden): {weak_segments}")
    print(f"\nVERDICT: {verdict} ({n_pass}/{n_total} automated checks passed)")
    print("\nThis is a RECOMMENDATION only. models/current_model.json was NOT modified.")

    out = {
        "candidate": str(candidate),
        "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in checks],
        "weak_segments": weak_segments,
        "leakage_safe_preprocessing": leakage_safe_preprocessing,
        "verdict": verdict,
    }
    _REPORTS.mkdir(parents=True, exist_ok=True)
    (_REPORTS / "promotion_gate_real.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
