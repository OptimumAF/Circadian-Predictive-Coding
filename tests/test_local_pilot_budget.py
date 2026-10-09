"""A bounded request cannot expand the fixed first-pilot execution scope."""

from dataclasses import FrozenInstanceError, replace
from math import inf, nan

import pytest

from src.core.local_pilot_budget import (
    FIRST_LOCAL_PILOT_LIMITS,
    PilotBudgetExceeded,
    PilotRequest,
    PilotResources,
    validate_local_pilot_request,
)


def test_should_accept_exact_limits_and_zero_optional_work() -> None:
    request = PilotRequest()
    assert validate_local_pilot_request(request) is request
    empty_work = replace(request.resources, training_updates=0, environment_steps=0, replay_bytes=0)
    assert validate_local_pilot_request(PilotRequest(empty_work)).resources == empty_work


@pytest.mark.parametrize(
    "field",
    [
        "wall_seconds",
        "training_updates",
        "environment_steps",
        "cpu_threads",
        "process_memory_bytes",
        "replay_bytes",
        "model_download_bytes",
        "storage_bytes",
        "gpu_memory_bytes",
    ],
)
def test_should_refuse_each_excess_before_accepting_a_plan(field: str) -> None:
    resources = replace(
        FIRST_LOCAL_PILOT_LIMITS, **{field: getattr(FIRST_LOCAL_PILOT_LIMITS, field) + 1}
    )
    with pytest.raises(PilotBudgetExceeded, match=field):
        validate_local_pilot_request(PilotRequest(resources))


def test_should_report_all_excesses_and_leave_fixed_limits_unchanged() -> None:
    request = PilotRequest(
        replace(FIRST_LOCAL_PILOT_LIMITS, training_updates=513, replay_bytes=1048577)
    )
    with pytest.raises(PilotBudgetExceeded) as caught:
        validate_local_pilot_request(request)
    assert len(caught.value.violations) == 2
    assert FIRST_LOCAL_PILOT_LIMITS.training_updates == 512
    with pytest.raises(FrozenInstanceError):
        setattr(FIRST_LOCAL_PILOT_LIMITS, "training_updates", 1000)


@pytest.mark.parametrize("value", [0, -1, inf, nan, True, 10**1000])
def test_should_refuse_invalid_duration(value: float) -> None:
    with pytest.raises(ValueError, match="wall_seconds"):
        PilotResources(wall_seconds=value)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_should_refuse_invalid_work_count(value: int) -> None:
    with pytest.raises(ValueError, match="training_updates"):
        PilotResources(training_updates=value)


@pytest.mark.parametrize(
    "pilot_request",
    [
        PilotRequest(execution_target="cloud"),
        PilotRequest(environment_kind="hardware"),
        PilotRequest(device="cuda"),
    ],
)
def test_should_refuse_unsupported_execution_context(pilot_request: PilotRequest) -> None:
    with pytest.raises(ValueError, match="local execution, simulation and CPU"):
        validate_local_pilot_request(pilot_request)
