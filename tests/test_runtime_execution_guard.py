"""The global enforcement mechanism must deny before actual mutation or execution."""

import json
from pathlib import Path
import subprocess
import sys


def test_should_deny_mutations_persist_after_caught_failure_and_release_actual_callbacks(
    tmp_path, record_property
):
    folder = tmp_path / "execution-guard"
    record_property("complete_runtime_artifacts", str(folder))
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-X",
            "faulthandler",
            str(Path(__file__).with_name("runtime_execution_guard_fixtures.py")),
            str(folder),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(rows) == 1 and rows[0]["passed"] and rows[0]["group"] == "execution-guard"
    assert rows[0]["science_calls"] == {}
    if rows[0]["supported_interpreter"]:
        assert rows[0]["actual_complete_observations"] == 3
        assert len(rows[0]["all_control_names_and_outcomes"]) == 23
        assert all(row["passed"] for row in rows[0]["all_control_names_and_outcomes"])
    else:
        assert rows[0]["actual_complete_observations"] == 0
