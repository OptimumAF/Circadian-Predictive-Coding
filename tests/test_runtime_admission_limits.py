"""Complete real counterexamples distinguish observations from admission proof."""

import json
import ctypes
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    sys.platform != "win32" or ctypes.sizeof(ctypes.c_void_p) != 8,
    reason="complete native runtime observations require 64-bit Windows",
)
def test_should_demonstrate_runtime_admission_limits_with_complete_actual_records(
    tmp_path, record_property
):
    script = Path(__file__).with_name("runtime_admission_limit_fixtures.py")
    completed = subprocess.run(
        [sys.executable, "-B", str(script), str(tmp_path / "admission-limits")],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    assert len(rows) == 1 and rows[0]["passed"] and rows[0]["group"] == "admission-limits"
    assert rows[0]["actual_complete_observations"] == 3 and rows[0]["science_calls"] == {}
    controls = rows[0]["all_control_names_and_outcomes"]
    assert len(controls) == 7 and all(row["passed"] for row in controls)
    assert sum(row.get("admission_gap_demonstrated", False) for row in controls) == 5
    record_property("complete_runtime_artifacts", str(tmp_path / "admission-limits"))
