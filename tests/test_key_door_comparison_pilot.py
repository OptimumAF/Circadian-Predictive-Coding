"""Fixed repeated-instance dispatch without running a native learner."""

from copy import deepcopy
import json
import sys
from typing import Any

import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_comparison_pilot as pilot
import src.adapters.key_door_reward_pilot as single
from test_torch_feature_rehearsal_pilot import forbidden


PAIRS = ((9, 14), (18, 5))
MODES = ("reward", "frozen")
IDENTITIES = tuple((seed, heldout, mode) for seed, heldout in PAIRS for mode in MODES)
CASE_IDS = tuple(f"env{seed}-final{heldout}-{mode}" for seed, heldout, mode in IDENTITIES)


def expected_calls():
    calls = []
    for seed, heldout, mode in IDENTITIES:
        options = dict(
            condition_on_legal_actions=True,
            action_remapping=True,
            fill_interaction_quota=True,
            environment_seed=seed,
            heldout_environment_seed=heldout,
        )
        if mode == "frozen":
            options["frozen_recurrent_control"] = True
        calls.append(options)
    return calls


def stub_report(index, *, status="complete"):
    # Routing tests use an explicit stub projection; they make no scientific claim.
    return dict(status=status, marker=index, valid=True, owned={"values": [index]})


def routing_environment(monkeypatch, source=None):
    calls, projected = [], []
    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)

    def selected(**options):
        calls.append(deepcopy(options))
        if source is not None:
            return source(len(calls) - 1, options)
        return stub_report(len(calls) - 1)

    def project(cases, *args, **options):
        projected.append(deepcopy(cases))
        for case in cases:
            if case["report"] is not None and not case["report"].get("valid", False):
                raise ValueError("preceding case evidence is invalid")
        return dict(
            status="complete"
            if all(case["status"] == "complete" for case in cases)
            else "incomplete"
        )

    monkeypatch.setattr(pilot, "run_single_arm", selected)
    monkeypatch.setattr(pilot, "project", project)
    return calls, projected


def assert_original_slots(report):
    assert [case["case_id"] for case in report["cases"]] == list(CASE_IDS)
    assert [
        (case["environment_seed"], case["heldout_environment_seed"], case["mode"])
        for case in report["cases"]
    ] == list(IDENTITIES)


def test_should_describe_fixed_cases_and_aggregate_limits_without_running_or_projecting(
    monkeypatch,
):
    calls = []

    def describe(**options):
        calls.append(deepcopy(options))
        return dict(options=deepcopy(options))

    monkeypatch.setattr(pilot, "describe_single_arm", describe)
    monkeypatch.setattr(pilot, "run_single_arm", forbidden)
    monkeypatch.setattr(pilot, "project", forbidden)
    monkeypatch.setattr(pilot, "monotonic", forbidden)
    protocol = pilot.describe_protocol()
    assert calls == expected_calls()
    assert protocol["protocol_id"] == "actual_key_door_repeated_reward_control_v1"
    assert [case["case_id"] for case in protocol["case_matrix"]] == list(CASE_IDS)
    assert list(protocol["arm_protocols"]) == list(CASE_IDS)
    assert [value["options"] for value in protocol["arm_protocols"].values()] == expected_calls()
    assert protocol["learner_seed"] == 17 and protocol["sampling_seed"] == 29
    assert protocol["maximum_comparison_seconds"] == 360
    expected_work = dict(
        cases=4,
        environment_pairs=2,
        minimum_native_train_calls=16,
        maximum_native_train_calls=32,
        maximum_training_rows=192,
        exact_collection_rows=384,
        maximum_collection_episodes=64,
        score_forwards=160,
        score_rows=1280,
        live_episodes=160,
        live_sessions=160,
        maximum_live_forwards=1920,
        maximum_prediction_calls=2464,
        maximum_recurrent_unroll_steps=3776,
        maximum_environment_steps=2560,
        maximum_environment_constructions=256,
        replay_calls=0,
        probes=0,
    )
    assert protocol["planned_work"] == expected_work
    historical = protocol["historical_evidence"]
    assert historical["qualification"] == "failed_spent"
    assert historical["primary_admission"] == "excluded_descriptive_only"
    assert historical["loaded"] is historical["retry"] is False


def test_should_reuse_exact_single_arm_descriptors_without_changing_existing_routes():
    legacy = deepcopy(single.describe_protocol())
    protocol = pilot.describe_protocol()
    assert [protocol["arm_protocols"][name] for name in CASE_IDS] == [
        single.describe_protocol(**options) for options in expected_calls()
    ]
    assert single.describe_protocol() == legacy
    protocol["arm_protocols"][CASE_IDS[0]]["environment_seeds"]["training"] = 0
    assert (
        pilot.describe_protocol()["arm_protocols"][CASE_IDS[0]]["environment_seeds"]["training"]
        == 9
    )


def test_should_run_each_declared_own_arm_once_in_the_frozen_order_and_validate_before_more_work(
    monkeypatch,
):
    calls, projected = routing_environment(monkeypatch)
    original_runner = pilot.run_single_arm

    def run(**options):
        if calls:
            assert projected, "another arm began before prior evidence was projected"
            assert projected[-1][len(calls) - 1]["report"]["marker"] == len(calls) - 1
        return original_runner(**options)

    monkeypatch.setattr(pilot, "run_single_arm", run)
    report = pilot.run_pilot()
    assert calls == expected_calls()
    assert_original_slots(report)
    assert report["status"] == report["comparison"]["status"] == "complete"
    assert all(case["status"] == "complete" and case["reason"] is None for case in report["cases"])
    assert [case["report"]["marker"] for case in report["cases"]] == [0, 1, 2, 3]
    assert report["global_errors"] == []
    assert report["scientific_advantage_verified"] is report["full_R43_complete"] is False


def test_should_compose_with_the_real_pure_pair_projection_using_complete_owned_scalar_reports(
    monkeypatch,
):
    from test_key_door_reward_comparison import make_complete_report

    calls = []
    originals = []

    def source(**options):
        calls.append(deepcopy(options))
        mode = "frozen" if options.get("frozen_recurrent_control") else "reward"
        report = make_complete_report(
            options["environment_seed"], options["heldout_environment_seed"], mode
        )
        # Real descriptors catch fixture/port wire mismatches before native work.
        report["protocol"] = single.describe_protocol(**options)
        originals.append(report)
        return report

    monkeypatch.setattr(pilot, "run_single_arm", source)
    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    report = pilot.run_pilot()
    assert calls == expected_calls()
    assert_original_slots(report)
    assert report["status"] == report["comparison"]["status"] == "complete"
    assert len(report["comparison"]["pairs"]) == 2
    assert report["comparison"]["aggregate"] is not None
    assert report["comparison"] == pilot.project(report["cases"])
    for original, case in zip(originals, report["cases"], strict=True):
        assert case["report"] == original and case["report"] is not original
    originals[0]["status"] = "incomplete"
    assert report["cases"][0]["report"]["status"] == "complete"
    assert report["scientific_advantage_verified"] is report["full_R43_complete"] is False


def test_should_capture_protocol_matrix_and_runner_before_first_external_arm_callback(monkeypatch):
    before = deepcopy(pilot.describe_protocol())

    def source(index, options):
        if index == 0:
            monkeypatch.setattr(pilot, "ENVIRONMENT_PAIRS", ((0, 23),))
            monkeypatch.setattr(pilot, "MODES", ("frozen",))
            monkeypatch.setattr(pilot, "run_single_arm", forbidden)
        return stub_report(index)

    calls, _ = routing_environment(monkeypatch, source)
    report = pilot.run_pilot()
    assert calls == expected_calls()
    assert report["protocol"] == before
    assert_original_slots(report)
    assert report["status"] == "complete"


def test_should_detach_each_returned_report_when_a_source_reuses_and_mutates_the_same_object(
    monkeypatch,
):
    shared_report: dict[str, Any] = {}

    def source(index, options):
        shared_report.clear()
        shared_report.update(stub_report(index))
        return shared_report

    calls, _ = routing_environment(monkeypatch, source)
    report = pilot.run_pilot()
    assert calls == expected_calls() and report["status"] == "complete"
    assert [case["report"]["marker"] for case in report["cases"]] == [0, 1, 2, 3]
    assert [case["report"]["owned"]["values"] for case in report["cases"]] == [[0], [1], [2], [3]]
    shared_report["owned"]["values"].append(99)
    assert report["cases"][3]["report"]["owned"]["values"] == [3]
    report["cases"][0]["report"]["owned"]["values"].append(77)
    assert report["cases"][1]["report"]["owned"]["values"] == [1]


@pytest.mark.parametrize("failure_index", range(4))
def test_should_stop_at_first_incomplete_source_and_keep_raw_failed_and_unreached_slots(
    monkeypatch, failure_index
):
    def source(index, options):
        return stub_report(index, status="incomplete" if index == failure_index else "complete")

    calls, _ = routing_environment(monkeypatch, source)
    report = pilot.run_pilot()
    assert calls == expected_calls()[: failure_index + 1]
    assert_original_slots(report)
    assert report["status"] == "incomplete"
    assert report["cases"][failure_index]["status"] == "failed"
    assert report["cases"][failure_index]["report"]["status"] == "incomplete"
    for case in report["cases"][failure_index + 1 :]:
        assert case["status"] == "not_run" and case["report"] is None
    assert report["comparison"]["status"] != "complete"


@pytest.mark.parametrize("failure_index", (0, 2))
def test_should_preserve_an_exception_and_never_retry_or_replace_its_cell(
    monkeypatch, failure_index
):
    def source(index, options):
        if index == failure_index:
            raise RuntimeError("declared arm failed after paid work")
        return stub_report(index)

    calls, _ = routing_environment(monkeypatch, source)
    report = pilot.run_pilot()
    assert calls == expected_calls()[: failure_index + 1]
    assert_original_slots(report)
    assert report["status"] == "incomplete"
    failed = report["cases"][failure_index]
    assert failed["status"] == "failed" and "paid work" in failed["reason"]
    assert failed["report"] is None
    assert all(case["status"] == "not_run" for case in report["cases"][failure_index + 1 :])


@pytest.mark.parametrize(
    "bad_report", (None, [], {"status": "running"}, {"status": "complete", "valid": False})
)
def test_should_refuse_malformed_complete_or_noncomplete_source_without_launching_another_cell(
    monkeypatch, bad_report
):
    calls, _ = routing_environment(monkeypatch, lambda index, options: deepcopy(bad_report))
    report = pilot.run_pilot()
    assert len(calls) == 1
    assert_original_slots(report)
    assert report["status"] == "incomplete" and report["cases"][0]["status"] == "failed"
    assert report["cases"][0]["report"] == bad_report
    assert all(case["status"] == "not_run" for case in report["cases"][1:])


def test_should_revalidate_preceding_success_when_a_later_callback_corrupts_it(monkeypatch):
    calls, _ = routing_environment(monkeypatch)
    original_project = pilot.project
    retained: list[Any] = []
    original_runner = pilot.run_single_arm

    def project(cases, *args, **options):
        retained[:] = [cases]
        return original_project(cases, *args, **options)

    def source(**options):
        if len(calls) == 1:
            retained[0][0]["report"]["valid"] = False
        return original_runner(**options)

    monkeypatch.setattr(pilot, "project", project)
    monkeypatch.setattr(pilot, "run_single_arm", source)
    report = pilot.run_pilot()
    assert len(calls) == 2
    assert report["status"] == "incomplete"
    assert all(case["status"] == "not_run" for case in report["cases"][2:])
    assert report["cases"][0]["report"]["valid"] is False
    assert report["cases"][1]["report"]["marker"] == 1


def test_should_stop_after_the_original_aggregate_clock_is_spent_without_renewing(monkeypatch):
    now = [0.0]

    def source(index, options):
        now[0] = 361.0
        return stub_report(index)

    calls, _ = routing_environment(monkeypatch, source)
    monkeypatch.setattr(pilot, "monotonic", lambda: now[0])
    report = pilot.run_pilot()
    assert len(calls) == 1
    assert_original_slots(report)
    assert report["status"] == "incomplete" and report["total_seconds"] >= 361
    assert report["cases"][0]["report"]["marker"] == 0
    assert all(case["status"] == "not_run" for case in report["cases"][1:])


@pytest.mark.parametrize("selected", ("describe_protocol", "run_pilot"))
def test_should_not_accept_caller_seed_or_mode_selection(selected):
    with pytest.raises(TypeError):
        getattr(pilot, selected)(environment_seed=0)
    with pytest.raises(TypeError):
        getattr(pilot, selected)(mode="reward")


@pytest.mark.parametrize("describe", (False, True))
def test_should_dispatch_the_fixed_comparison_cli_with_no_options(monkeypatch, capsys, describe):
    calls = []

    def selected():
        calls.append("describe" if describe else "run")
        return dict(status="complete")

    monkeypatch.setattr(cli, "describe_comparison_protocol", selected)
    monkeypatch.setattr(cli, "run_comparison_pilot", selected)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run", "--scenario", "torch-key-door-repeated-reward-control"]
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == ["describe" if describe else "run"]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "flags",
    (
        ["--environment-seed", "9"],
        ["--heldout-environment-seed", "14"],
        ["--environment-seed", "9", "--heldout-environment-seed", "14"],
        ["--source-artifact", "old-failed-report.json"],
    ),
)
@pytest.mark.parametrize("describe", (False, True))
def test_should_refuse_caller_instance_or_source_overrides_before_comparison_invocation(
    monkeypatch, flags, describe
):
    monkeypatch.setattr(cli, "describe_comparison_protocol", forbidden)
    monkeypatch.setattr(cli, "run_comparison_pilot", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run", "--scenario", "torch-key-door-repeated-reward-control", *flags]
        + (["--describe"] if describe else []),
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "scenario,options",
    (
        (
            "torch-key-door-legal-interaction-quota",
            dict(
                condition_on_legal_actions=True, action_remapping=True, fill_interaction_quota=True
            ),
        ),
        (
            "torch-key-door-frozen-interaction-quota",
            dict(
                condition_on_legal_actions=True,
                action_remapping=True,
                fill_interaction_quota=True,
                frozen_recurrent_control=True,
            ),
        ),
    ),
)
def test_should_preserve_existing_single_arm_cli_dispatch(monkeypatch, capsys, scenario, options):
    calls = []
    monkeypatch.setattr(cli, "run_comparison_pilot", forbidden)

    def select(**values):
        calls.append(values)
        return {"status": "complete"}

    monkeypatch.setattr(cli, "run_legal_reward_pilot", select)
    monkeypatch.setattr(sys, "argv", ["run", "--scenario", scenario])
    cli.main()
    assert calls == [options]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
