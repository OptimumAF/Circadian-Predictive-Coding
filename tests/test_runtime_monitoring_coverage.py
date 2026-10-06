"""Actual installed guard coverage must precede continuous admission claims."""

import json
from pathlib import Path
import subprocess
import sys


def test_should_measure_installed_guard_coverage_with_complete_actual_records(
    tmp_path, record_property
):
    folder = tmp_path / "monitoring-coverage"
    record_property("complete_runtime_artifacts", str(folder))
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-X",
            "faulthandler",
            str(Path(__file__).with_name("runtime_monitoring_coverage_fixtures.py")),
            str(folder),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    assert len(rows) == 1 and rows[0]["passed"] and rows[0]["group"] == "monitoring-coverage"
    assert rows[0]["actual_complete_observations"] == 3 and rows[0]["science_calls"] == {}
    assert len(rows[0]["all_control_names_and_outcomes"]) == 7
    assert all(row["passed"] for row in rows[0]["all_control_names_and_outcomes"])
