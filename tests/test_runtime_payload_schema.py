"""Full actual process records and failure lifecycle controls in bounded children."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.core.runtime_record_schema import RuntimeSchemaBindings
from src.core.runtime_value_schema import validate_runtime_bound


@pytest.mark.parametrize("group", ["payload", "continuity", "lifecycle"])
def test_should_validate_whole_actual_runtime_records_without_science(
    group, tmp_path, record_property
):
    script = Path(__file__).with_name("runtime_payload_fixtures.py")
    completed = subprocess.run(
        [sys.executable, "-B", str(script), group, str(tmp_path / group)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    assert len(rows) == 1 and rows[0]["passed"] and rows[0]["group"] == group
    assert rows[0]["science_calls"] == {} and rows[0]["actual_complete_observations"] >= 2
    assert rows[0]["all_control_names_and_outcomes"]
    assert all(row["passed"] for row in rows[0]["all_control_names_and_outcomes"])
    record_property("complete_runtime_artifacts", str(tmp_path / group))


def test_should_validate_canonical_strings_without_loading_code(monkeypatch):
    def deny_compile(*args, **kwargs):
        raise AssertionError("passive string validation must not compile input")

    monkeypatch.setattr("builtins.compile", deny_compile)
    bindings = RuntimeSchemaBindings()
    for value in (
        "",
        "'",
        '"',
        "both'\"",
        "literal\\u0061",
        "\\",
        "\x00\t\r\n\b\a\f\v",
        "\x1f\x7f\x85\xa0\xad",
        "é€漢😀",
        "\ud800\udfff",
        "a\u2028\u2029b",
    ):
        validate_runtime_bound(["str", repr(value)], bindings, "string codec control")
    for text in (
        "a",
        '"a"',
        "'\\u0061'",
        "'\\x61'",
        "'\\q'",
        "'\\uXYZ0'",
        "'\\U00110000'",
        "'unterminated",
        "'''a'''",
        "b'a'",
        "'a' + 'b'",
        "'line\nbreak'",
    ):
        with pytest.raises(ValueError, match="runtime payload"):
            validate_runtime_bound(["str", text], bindings, "string codec control")
