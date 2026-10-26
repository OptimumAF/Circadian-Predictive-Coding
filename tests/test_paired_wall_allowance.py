"""Prospective paired wall caps using literal orchestration and stub CLI calls.

No models, arrays, RNG, forwards, updates or native physics execute. Integrated
cases reuse the accepted literal fixture; only metadata and controlled clocks run.

Per invocation: 51 cases; five fake pilot calls, ten stub arm entries, forty
literal final packets and ten stub evaluations. Four CLI calls use stubs only;
all remaining cases validate metadata or refuse before execution.
"""

from decimal import Decimal
import json
import sys
from typing import Any

import pytest

import scripts.run_coupled_attribution as cli
from test_paired_quota_control import orchestration as orchestration
from src.adapters import key_door_quota_comparison as paired


class IntegerSubclass(int):
    pass


class FloatSubclass(float):
    pass


class NumericLike:
    def __float__(self):
        pytest.fail("wall allowance coerced a foreign numeric object")


@pytest.mark.parametrize("allowance", [0.25, 1, 90, 90.0, 180, 5e-324])
def test_should_describe_exact_builtin_positive_caps_without_clock_or_native_work(
    monkeypatch, allowance
):
    def forbidden(*args, **kwargs):
        pytest.fail("description attempted runtime work")

    monkeypatch.setattr(paired, "monotonic", forbidden)
    monkeypatch.setattr(paired, "perf_counter", forbidden)
    monkeypatch.setattr(paired.base, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(paired.base, "require_torch", forbidden)
    actual = paired.describe_protocol(wall_seconds=allowance)
    expected = paired.describe_protocol()
    expected["resources"]["wall_seconds"] = allowance
    assert actual == expected
    assert type(actual["resources"]["wall_seconds"]) is type(allowance)


@pytest.mark.parametrize(
    "allowance",
    [
        True,
        False,
        0,
        -1,
        180.001,
        10**1000,
        float("nan"),
        float("inf"),
        None,
        "120",
        IntegerSubclass(90),
        FloatSubclass(90),
        Decimal("90"),
        NumericLike(),
    ],
)
@pytest.mark.parametrize("operation", ["describe", "run"])
def test_should_reject_invalid_allowance_before_clocks_resources_or_data(
    monkeypatch, allowance, operation
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid allowance admitted work")

    monkeypatch.setattr(paired, "monotonic", forbidden)
    monkeypatch.setattr(paired, "perf_counter", forbidden)
    monkeypatch.setattr(paired.base, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(paired.base, "require_torch", forbidden)
    monkeypatch.setattr(paired.base.shared, "_collect", forbidden)
    selected = paired.describe_protocol if operation == "describe" else paired.run_pilot
    with pytest.raises(ValueError, match="wall_seconds"):
        selected(wall_seconds=allowance)


@pytest.fixture
def controlled_pair(monkeypatch, orchestration):
    clock: dict[str, Any] = {"now": 100.0, "end": 101.0, "reads": []}
    deadlines: list[float] = []
    original = paired.base._run_arm

    def monotonic():
        clock["reads"].append(clock["now"])
        return clock["now"]

    def run_arm(row, final, deadline, sampler, **options):
        deadlines.append(deadline)
        return original(row, final, deadline, sampler, **options)

    def finish(arm):
        if arm == "trained":
            clock["now"] = clock["end"]

    monkeypatch.setattr(paired, "monotonic", monotonic)
    monkeypatch.setattr(paired, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(paired.base, "_run_arm", run_arm)
    orchestration["on_finish"] = finish
    return orchestration, clock, deadlines


@pytest.mark.parametrize("explicit", [False, True])
def test_should_use_one_original_birth_and_uniform_deadline_for_both_arms(
    controlled_pair, explicit
):
    evidence, clock, deadlines = controlled_pair
    options: dict[str, Any] = {"wall_seconds": 120.5} if explicit else {}
    result = paired.run_pilot(**options)
    expected = 120.5 if explicit else 90
    assert result["status"] == "complete"
    assert deadlines == [100.0 + expected] * 2
    assert clock["reads"] == [100.0, 101.0]
    assert result["protocol"]["resources"]["wall_seconds"] == expected
    assert evidence["physics"] == {"frozen": 96, "trained": 96}
    assert len(evidence["final_packets"]) == 8 and len(evidence["evaluations"]) == 2
    assert [row["wake_updates"] for row in result["runs"]] == [0, 16]
    assert result["scientific_advantage_verified"] is False


@pytest.mark.parametrize("elapsed,accepted", [(5.0, True), (5.25, False)])
def test_should_apply_same_original_cap_to_final_completion_after_paid_phases(
    controlled_pair, elapsed, accepted
):
    evidence, clock, deadlines = controlled_pair
    clock["end"] = 100.0 + elapsed
    result = paired.run_pilot(wall_seconds=5)
    assert (result["status"] == "complete") is accepted
    assert result["total_seconds"] == elapsed
    assert deadlines == [105.0, 105.0]
    assert [row["status"] for row in result["runs"]] == ["complete", "complete"]
    assert [row["wake_updates"] for row in result["runs"]] == [0, 16]
    assert len(evidence["evaluations"]) == 2 and len(result["comparison"]) == 40


def test_should_not_renew_deadline_or_completion_cap_from_mutated_protocol_metadata(
    monkeypatch, controlled_pair
):
    evidence, clock, deadlines = controlled_pair
    original = paired.describe_protocol
    captured = []

    def describe(**options):
        result = original(**options)
        captured.append(result)
        return result

    def mutate(arm):
        if arm == "frozen":
            captured[0]["resources"]["wall_seconds"] = 180

    monkeypatch.setattr(paired, "describe_protocol", describe)
    evidence["on_arm"] = mutate
    clock["end"] = 113.0
    result = paired.run_pilot(wall_seconds=12)
    assert result["status"] == "incomplete" and result["total_seconds"] == 13.0
    assert deadlines == [112.0, 112.0]
    assert clock["reads"] == [100.0, 113.0]
    assert result["protocol"]["resources"]["wall_seconds"] == 180
    assert [row["wake_updates"] for row in result["runs"]] == [0, 16]
    assert len(evidence["evaluations"]) == 2


@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize("allowance", [None, "120.5"])
def test_should_forward_only_explicit_wall_option_for_describe_and_run(
    monkeypatch, capsys, describe, allowance
):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "complete", "metadata_only": True}

    monkeypatch.setattr(
        cli,
        "describe_quota_comparison_protocol" if describe else "run_quota_comparison_pilot",
        selected,
    )
    arguments = [
        "pilot",
        "--scenario",
        "torch-key-door-paired-interaction-quota",
        "--environment-seed",
        "17",
        "--heldout-environment-seed",
        "6",
    ]
    if describe:
        arguments.append("--describe")
    if allowance is not None:
        arguments.extend(["--paired-wall-seconds", allowance])
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    expected: dict[str, Any] = {"environment_seed": 17, "heldout_environment_seed": 6}
    if allowance is not None:
        expected["wall_seconds"] = 120.5
    assert calls == [expected]
    assert json.loads(capsys.readouterr().out)["metadata_only"] is True


@pytest.mark.parametrize("allowance", ["0", "-1", "180.001", "nan", "inf", "invalid"])
def test_should_refuse_bad_cli_allowance_before_invoking_either_adapter(monkeypatch, allowance):
    def forbidden(**options):
        pytest.fail("invalid CLI allowance reached adapter")

    monkeypatch.setattr(cli, "run_quota_comparison_pilot", forbidden)
    monkeypatch.setattr(cli, "describe_quota_comparison_protocol", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pilot",
            "--scenario",
            "torch-key-door-paired-interaction-quota",
            "--paired-wall-seconds",
            allowance,
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "scenario",
    ["torch-key-door-legal-interaction-quota", "torch-key-door-frozen-interaction-quota"],
)
def test_should_refuse_wall_option_on_standalone_quota_routes_before_work(monkeypatch, scenario):
    def forbidden(**options):
        pytest.fail("foreign route admitted paired wall option")

    monkeypatch.setattr(cli, "run_legal_reward_pilot", forbidden)
    monkeypatch.setattr(cli, "describe_legal_reward_protocol", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["pilot", "--scenario", scenario, "--paired-wall-seconds", "120"]
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
