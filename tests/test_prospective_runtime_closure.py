"""Behavioral live-process controls; full fixtures execute in bounded subprocesses."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from src.app.prospective_runtime_closure import validate_runtime_observation
from src.core.prospective_runtime_closure import RuntimeCodeObservation


@pytest.mark.parametrize("group", ["positive", "python_a", "python_b", "native", "failure"])
def test_should_observe_or_deny_complete_live_runtime_without_science(group):
    script = Path(__file__).with_name("prospective_runtime_fixtures.py")
    completed = subprocess.run(
        [sys.executable, "-B", str(script), group],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"PASS {group}" in completed.stdout


@pytest.mark.parametrize(
    "field,value",
    [
        ("process_id", True),
        ("process_id", 1.0),
        ("process_id", 0),
        ("sequence", True),
        ("sequence", 0.0),
        ("sequence", -1),
        ("lease_nonce", "invalid"),
        ("observed_utc", "2026-10-05T12:00:00"),
        ("runtime_json", "{}"),
        ("runtime_json", "false"),
        ("entrypoints", []),
        ("entrypoints", ("invalid",)),
        ("complete_process_membership_observed", False),
        ("runtime_source_version_attested", True),
    ],
)
def test_should_reject_foreign_or_coerced_runtime_observation_before_use(field, value):
    # Only invalid port records are fabricated here; positive controls use the
    # real process/file/native owner observer in the independent child.
    from prospective_runtime_fixtures import invalid_observation_fixture

    snapshot, observation = invalid_observation_fixture()
    with pytest.raises(ValueError):
        validate_runtime_observation(snapshot, replace(observation, **{field: value}))


def test_should_reject_foreign_runtime_port_record():
    from prospective_runtime_fixtures import invalid_observation_fixture

    snapshot, _ = invalid_observation_fixture()
    assert getattr(RuntimeCodeObservation, "__dataclass_params__").frozen
    with pytest.raises(ValueError, match="foreign"):
        validate_runtime_observation(snapshot, object())
