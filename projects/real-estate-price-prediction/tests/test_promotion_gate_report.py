"""Tests for scripts/promotion_gate_report.py's verdict logic and its
read-only-with-respect-to-current_model.json guarantee.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from promotion_gate_report import compute_verdict


def _all_pass(n: int) -> list[tuple[str, bool, str]]:
    return [(f"check_{i}", True, "ok") for i in range(n)]


class TestComputeVerdict:
    def test_all_checks_pass_is_promotion_ready(self):
        assert compute_verdict(_all_pass(8)) == "PROMOTION_READY"

    def test_two_soft_failures_is_ready_with_limitations(self):
        checks = _all_pass(8)
        checks[6] = ("weak_segments", False, "n/a")
        checks[7] = ("weak_segments_disclosed", False, "n/a")
        assert compute_verdict(checks) == "PROMOTION_READY_WITH_LIMITATIONS"

    def test_three_soft_failures_is_not_ready(self):
        checks = _all_pass(8)
        checks[5] = ("x", False, "n/a")
        checks[6] = ("y", False, "n/a")
        checks[7] = ("z", False, "n/a")
        assert compute_verdict(checks) == "PROMOTION_NOT_READY"

    def test_hard_blocker_failure_is_never_ready_even_if_everything_else_passes(self):
        """The central regression test: a candidate whose preprocessing
        has unresolved, known leakage must never be PROMOTION_READY or
        PROMOTION_READY_WITH_LIMITATIONS, no matter how many other checks
        pass."""
        checks = _all_pass(8)
        checks[3] = ("Leakage-safe preprocessing [HARD BLOCKER]", False, "leaky")
        assert compute_verdict(checks) == "PROMOTION_NOT_READY"

    def test_hard_blocker_passing_does_not_by_itself_force_ready(self):
        checks = [("Leakage-safe preprocessing [HARD BLOCKER]", True, "ok")] + [
            (f"check_{i}", False, "fail") for i in range(5)
        ]
        assert compute_verdict(checks) == "PROMOTION_NOT_READY"

    def test_empty_checks_list_is_promotion_ready(self):
        """Degenerate case: vacuously all-pass. Not expected in real usage
        (main() always appends a fixed set of checks) but must not crash."""
        assert compute_verdict([]) == "PROMOTION_READY"


class TestPromotionGateScriptDoesNotTouchProductionPointer:
    def test_running_the_gate_leaves_current_model_json_byte_identical(self):
        manifest_path = _PROJECT_ROOT / "models" / "current_model.json"
        if not manifest_path.exists():
            pytest.skip("models/current_model.json not present in this checkout.")
        before = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

        result = subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "scripts" / "promotion_gate_report.py")],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 2:
            pytest.skip(
                "No real candidate artefact available in this checkout to run the gate against."
            )
        assert result.returncode == 0, result.stderr

        after = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        assert before == after, "promotion_gate_report.py must never modify current_model.json"
