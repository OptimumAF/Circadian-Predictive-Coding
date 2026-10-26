"""Arrival-order recent-only control using accepted fake rehearsal/probe fixtures."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_stream_rehearsal_pilot as cli
import src.adapters.stream_rehearsal_pilot as pilot
from src.app.coupled_attribution import run_attribution_stream
from src.app.stream_rehearsal import RecentStreamReplayPolicy, StreamRehearsal
from src.app.toy_execution_budget import ToyExecutionStopped
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from test_partitioned_replay_buffer import content_id
from test_rehearsal_learnability import RETAINED_PROTOCOL, fake_environment, forbidden
from test_stream_rehearsal import FakeLearner, batches, budgets
from test_stream_rehearsal_pilot import FakeModel, FakeSampler


def retained_probe_protocol():
    """Complete independent DATA oracle from the accepted rotating probe protocol."""
    result = deepcopy(RETAINED_PROTOCOL)
    result.update(
        protocol_id="coherent_rehearsal_reserved_sixteen_v1",
        classification="exploratory_rehearsal_with_isolated_learnability",
        probe_isolation="independent template and disposable fork for each copied post-rehearsal checkpoint; benchmark final data unavailable to probes",
        checkpoint_clocks="0/6/12/18 are wake counts; native states additionally include rehearsal updates",
        probe_feedback=False,
        learnability=json.loads(r"""{
          "source_protocol_id":"reserved_fresh_task_learnability_sixteen_v1",
          "probe_target":"x0*x1 >= 0; fixed nonlinear composition of A/B inputs",
          "probe_training_rng":17031,"probe_evaluation_rng":17033,"diagnostic_features_rng":17035,
          "probe_training_batches":16,"probe_training_batch_rows":8,"probe_evaluation_rows":32,
          "probe_checkpoint_updates":[0,1,2,4,8,16],"diagnostic_feature_rows":32,
          "feature_health_definition":"mean absolute tanh activation; dead <=1e-3; active complement; saturated entry abs>=.99",
          "effective_rank_definition":"centered SVD singular VALUES entropy; s>eps*max(shape)*max(1,max(s)); numerical zero rank0",
          "effective_rank_limit":"absolute numerical cutoff is not invariant at tiny scales; diagnostic, not causal proof",
          "unavailable_diagnostics":{"importance":"Coupled models do not store native per-neuron importance.","neuron_age":"Coupled models do not store native per-neuron ages."},
          "reserved_data_array_bytes":3840,"diagnostic_reference_array_bytes":512,
          "score_definitions":{"bce":"probability BCE clip1e-12 lower better","accuracy":"threshold >=.5 higher better"},
          "comparison_directions":{"fresh_gain":"probe BCE at0 minus BCE at16; retain initial level and entire curve","retention_loss":"A BCE afterB minus A BCE afterA; lower better","return_gain":"A BCE afterB minus A BCE afterReturn; higher better","new_B_gain":"B BCE initial minus B BCE afterB; higher better"}
        }"""),
    )
    result["resources"]["training_updates"] = 376
    result["planned_work"].update(
        optimizer_updates=376,
        probe_updates=256,
        probe_row_exposures=2048,
        total_row_exposures=2672,
        probe_clones=16,
        probe_score_forwards=96,
        diagnostic_feedforward_passes=16,
        probes=16,
        svd=32,
    )
    return result


def recent_policy(**changes):
    values: dict[str, Any] = dict(
        every_updates=2, rows_per_event=2, retention=ReplayRetentionBudget(4, 96)
    )
    values.update(changes)
    return RecentStreamReplayPolicy(**values)


def setup(*, cap=4, **faults):
    seconds = [0.0]
    wake, replay = budgets(cap, seconds=seconds)
    source = FakeLearner()
    source.control.update(
        replay_budget=replay, seconds=seconds, error=RuntimeError("original recent fault"), **faults
    )
    helper = StreamRehearsal(
        source, wake, recent_policy(), replay_budget=replay, total_updates=4, input_dim=2
    )
    return source, wake, replay, helper


def run(source, wake, helper, training=None):
    original, heldout = batches()
    return run_attribution_stream(
        source,
        original if training is None else training,
        heldout,
        checkpoint_updates=(0, 2, 4),
        label_delay=2,
        budget=wake,
        rehearsal=helper,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("every_updates", 0),
        ("every_updates", True),
        ("every_updates", 513),
        ("rows_per_event", 0),
        ("rows_per_event", True),
        ("rows_per_event", 129),
        ("retention", object()),
        ("retention", ReplayRetentionBudget(129, 96)),
        ("retention", ReplayRetentionBudget(4, 1024**2 + 1)),
        ("retention", "tampered"),
    ],
)
def test_should_refuse_nonexact_recent_policy_fields_and_nested_limits(field, value):
    if value == "tampered":
        value = ReplayRetentionBudget(4, 96)
        object.__setattr__(value, "max_bytes", 0)
    with pytest.raises(ValueError):
        recent_policy(**{field: value})


@pytest.mark.parametrize("fault", ["foreign", "subclass"])
def test_should_require_an_exact_supported_policy_before_forking(fault):
    class Foreign(RecentStreamReplayPolicy):
        pass

    source = FakeLearner()
    wake, replay = budgets()
    controls: Any = object() if fault == "foreign" else Foreign(2, 2, ReplayRetentionBudget(4, 96))
    with pytest.raises(ValueError):
        StreamRehearsal(source, wake, controls, replay_budget=replay, total_updates=4, input_dim=2)
    assert source.control["forks"] == 0


def test_should_refuse_retention_that_cannot_hold_a_labeled_row_before_fork():
    source = FakeLearner()
    wake, replay = budgets()
    with pytest.raises(ValueError):
        StreamRehearsal(
            source,
            wake,
            recent_policy(retention=ReplayRetentionBudget(4, 23)),
            replay_budget=replay,
            total_updates=4,
            input_dim=2,
        )
    assert source.control["forks"] == 0


def test_should_own_only_applicable_recent_controls_and_detach_both_policy_boundaries():
    retained = ReplayRetentionBudget(4, 96)
    controls = recent_policy(retention=retained)
    assert set(asdict(controls)) == {"every_updates", "rows_per_event", "retention"}
    object.__setattr__(retained, "max_examples", 1)
    assert controls.retention.max_examples == 4
    source = FakeLearner()
    wake, replay = budgets()
    source.control["replay_budget"] = replay
    helper = StreamRehearsal(
        source, wake, controls, replay_budget=replay, total_updates=4, input_dim=2
    )
    expected = asdict(controls)
    object.__setattr__(controls.retention, "max_bytes", 1)
    object.__setattr__(controls, "every_updates", 1)
    report = helper.report
    assert report["control"] == "recent_fifo_v1" and report["policy"] == expected
    report["policy"]["retention"]["max_bytes"] = 1
    assert helper.report["policy"] == expected


def test_should_rehearse_latest_two_rows_in_arrival_order_and_detach_caller_arrays():
    source, wake, replay, helper = setup(mutate_inputs=True)
    training, _ = batches()
    before = pickle.dumps(training)
    result = run(source, wake, helper, training)
    assert pickle.dumps(training) == before and source.state[0] == 0
    assert replay.replay_examples_completed == 4 and wake.updates_completed == 4
    replayed = [
        event for event in source.control["trace"] if event[0] == "train" and event[2] == "replay"
    ]
    expected = [
        (batch.features[index : index + 1], batch.targets[index : index + 1])
        for batch in (training[1], training[3])
        for index in range(2)
    ]
    for event, (features, targets) in zip(replayed, expected, strict=True):
        np.testing.assert_array_equal(event[4], features)
        np.testing.assert_array_equal(event[5], targets)
    for number, row in enumerate(result["rehearsal"]["events"]):
        assert row["selected_ids"] == [
            content_id(*pair) for pair in expected[2 * number : 2 * number + 2]
        ]
        assert row["retention"]["example_count"] == 4 and row["retention"]["retained_bytes"] == 96
        assert row["partition_retention"] is row["rotation_cursor"] is None


def test_should_refresh_duplicate_content_order_without_growing_fifo_storage():
    source, wake, _, helper = setup()
    training, _ = batches()
    modified = list(training)
    modified[2] = replace(
        training[2],
        features=np.concatenate((training[0].features[1:], training[2].features[:1])),
        targets=np.zeros((2, 1)),
    )
    modified[3] = replace(
        training[3], features=training[1].features.copy(), targets=training[1].targets.copy()
    )
    run(source, wake, helper, tuple(modified))
    events = helper.report["events"]
    expected = [
        content_id(training[1].features[index : index + 1], training[1].targets[index : index + 1])
        for index in range(2)
    ]
    assert events[0]["selected_ids"] == events[1]["selected_ids"] == expected
    retained = {
        content_id(batch.features[index : index + 1], batch.targets[index : index + 1])
        for batch in modified[2:]
        for index in range(2)
    }
    assert set(events[1]["retention"]["sample_ids"]) == retained
    assert events[1]["retention"]["example_count"] == 4


@pytest.mark.parametrize("fault", ["short_event", "train_throw", "post_commit"])
def test_should_preserve_whole_event_admission_permanent_charges_and_actual_commit_on_failure(
    fault,
):
    options = (
        {"cap": 1}
        if fault == "short_event"
        else {"fail_replay": 2}
        if fault == "train_throw"
        else {"expire_after_commit": True}
    )
    source, wake, replay, helper = setup(**options)
    with pytest.raises((RuntimeError, ToyExecutionStopped)) as caught:
        run(source, wake, helper)
    if fault == "train_throw":
        assert caught.value is source.control["error"]
    report = helper.report
    charged, returned, committed = (
        (0, 0, 0) if fault == "short_event" else (2, 1, 0) if fault == "train_throw" else (2, 2, 2)
    )
    assert (
        report["charged_exposures"]
        == report["attempted_updates"]
        == replay.replay_examples_completed
        == charged
    )
    assert report["returned_updates"] == returned and report["committed_updates"] == committed
    assert wake.updates_completed == 2 and wake.replay_examples_completed == 0
    assert report["events"][0]["status"] == "failed" and report["events"][1]["status"] == "not_run"
    assert (report["events"][0]["receipt"] is not None) is (fault == "post_commit")
    assert not any(event[0] == "predict" for event in source.control["trace"])
    assert source.state[0] == 0


@pytest.mark.parametrize("name", ["replay_policy", "describe_protocol", "run_pilot"])
@pytest.mark.parametrize("invalid", [0, 1, None])
def test_should_refuse_nonboolean_recent_mode_before_native_or_observer_work(
    name, invalid, monkeypatch
):
    for attribute in (
        "build_learner",
        "build_stream",
        "ProcessRssSampler",
        "CoupledPredictiveCodingNetwork",
    ):
        monkeypatch.setattr(pilot, attribute, forbidden)
    with pytest.raises(ValueError):
        getattr(pilot, name)(recent_only=invalid)


@pytest.mark.parametrize("action", ["describe", "run"])
@pytest.mark.parametrize("recent", [False, True])
@pytest.mark.parametrize("diagnose", [False, True])
def test_should_route_all_recent_probe_combinations_without_new_keywords_on_old_modes(
    action, recent, diagnose, monkeypatch, capsys
):
    calls = []

    def selected(**kwargs):
        calls.append(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(cli, "describe_protocol", selected if action == "describe" else forbidden)
    monkeypatch.setattr(cli, "run_pilot", selected if action == "run" else forbidden)
    arguments = ["rehearsal", f"--{action}"]
    if recent:
        arguments.append("--recent-only")
    if diagnose:
        arguments.append("--diagnose-learnability")
    monkeypatch.setattr(sys, "argv", arguments)
    assert cli.main() == 0 and json.loads(capsys.readouterr().out)["status"] == "complete"
    expected = (
        {"recent_only": True, "diagnose_learnability": diagnose}
        if recent
        else {"diagnose_learnability": True}
        if diagnose
        else {}
    )
    assert calls == [expected]


@pytest.mark.parametrize("diagnose", [False, True])
def test_should_preserve_both_complete_old_descriptors_and_old_private_dispatch_with_explicit_recent_false(
    diagnose, monkeypatch
):
    expected = retained_probe_protocol() if diagnose else RETAINED_PROTOCOL
    assert pilot.describe_protocol(diagnose_learnability=diagnose) == expected
    assert pilot.describe_protocol(diagnose_learnability=diagnose, recent_only=False) == expected
    calls = []

    def old_one(row, initial, deadline, sampler, **kwargs):
        calls.append(kwargs)
        row.update(status="failed", reason="old hook")

    monkeypatch.setattr(pilot, "_run_one", old_one)
    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    monkeypatch.setattr(pilot, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", FakeModel)
    implicit = pilot.run_pilot(diagnose_learnability=diagnose)
    explicit = pilot.run_pilot(diagnose_learnability=diagnose, recent_only=False)
    assert implicit == explicit
    assert calls == ([{"diagnose_learnability": True}] if diagnose else [{}]) * 8


@pytest.mark.parametrize("diagnose", [False, True])
def test_should_describe_the_exact_recent_control_without_ignored_allocation_or_seed(diagnose):
    report = pilot.describe_protocol(recent_only=True, diagnose_learnability=diagnose)
    old = retained_probe_protocol() if diagnose else RETAINED_PROTOCOL
    assert report["protocol_id"] == (
        "coherent_recent_rehearsal_reserved_sixteen_v1"
        if diagnose
        else "coherent_stream_recent_rehearsal_v1"
    )
    assert report["control_for"] == old["protocol_id"]
    assert report["replay"] == {
        "every_updates": 3,
        "rows_per_event": 2,
        "retention": {"max_examples": 16, "max_bytes": 384},
    }
    assert report["replay_retention"] == "recent_fifo"
    assert "retention" in report["comparison_scope"] and "order" in report["comparison_scope"]
    assert report["planned_work"] == old["planned_work"] and report["resources"] == old["resources"]
    for key in (
        "methods",
        "seeds",
        "training_rng",
        "final_rng",
        "learning_rate",
        "inference_steps",
        "inference_learning_rate",
        "checkpoint_updates",
    ):
        assert report[key] == old[key]
    if diagnose:
        assert report["learnability"] == old["learnability"]


@pytest.mark.parametrize("diagnose", [False, True])
def test_should_apply_one_common_arrival_order_plan_and_preserve_native_work_counts_and_probe_isolation(
    diagnose, monkeypatch
):
    control = fake_environment(monkeypatch, mutate_arguments=True)
    report = pilot.run_pilot(recent_only=True, diagnose_learnability=diagnose)
    assert report["status"] == "complete", [row.get("reason") for row in report["runs"]]
    assert report["protocol"]["protocol_id"] == (
        "coherent_recent_rehearsal_reserved_sixteen_v1"
        if diagnose
        else "coherent_stream_recent_rehearsal_v1"
    )
    assert report["wake_updates"] == 72 and report["replay_charged_examples"] == 48
    assert report["replay_returned_updates"] == report["replay_committed_updates"] == 48
    assert report["row_exposures_charged"] == 624
    assert report["shared_replay_plans_exact"] is report["pc_neutral_exact"] is True
    all_plans = []
    for index, row in enumerate(report["runs"]):
        helper = row["rehearsal"]
        assert helper["control"] == "recent_fifo_v1"
        assert set(helper["policy"]) == {"every_updates", "rows_per_event", "retention"}
        training, _ = control["streams"][index][0]
        expected = [
            [
                content_id(
                    training[updates - 1].features[position : position + 1],
                    training[updates - 1].targets[position : position + 1],
                )
                for position in (6, 7)
            ]
            for updates in range(3, 19, 3)
        ]
        assert [event["updates"] for event in helper["events"]] == [3, 6, 9, 12, 15, 18]
        assert [event["selected_ids"] for event in helper["events"]] == expected
        all_plans.append(expected)
        for event in helper["events"]:
            assert event["partition_retention"] is event["rotation_cursor"] is None
            assert (
                event["retention"]["example_count"] == 16
                and event["retention"]["retained_bytes"] == 384
            )
        events = [event for event in control["trace"] if event[0] == row["method"]]
        benchmark = [event for event in events if event[1:3] == ("benchmark", "train")]
        assert len(benchmark) == 30 and sum(event[3] for event in benchmark) == 156
        benchmark_predictions = [
            event for event in events if event[1:3] == ("benchmark", "predict")
        ]
        assert [event[3] for event in benchmark_predictions] == [
            0.0,
            0.0,
            10.0,
            10.0,
            20.0,
            20.0,
            30.0,
            30.0,
        ]
        assert (
            len(row["stream_result"]["applied_updates"]) == 18
            and len(row["stream_result"]["applied_consolidations"]) == 6
        )
        if diagnose:
            assert [entry["status"] for entry in row["observations"]] == ["complete"] * 4
            assert [
                state
                for template, state, _ in control["probes"]
                if template.method == row["method"]
            ] == [0.0, 10.0, 20.0, 30.0]
            final_start = min(
                position
                for position, event in enumerate(events)
                if event[1:3] == ("benchmark", "predict")
            )
            assert final_start > max(
                position
                for position, event in enumerate(events)
                if event[1] == "template" and event[2] in ("train", "predict")
            )
        else:
            assert "observations" not in row
    assert all(plan == all_plans[0] for plan in all_plans)
    assert all(source.state[0] == 0 for source in control["sources"])
    assert all(
        pickle.dumps(data) == before for data, before in (*control["streams"], *control["reserved"])
    )
    if diagnose:
        assert report["probe_attempted_updates"] == report["probe_completed_updates"] == 256
        assert report["probe_score_forward_calls_completed"] == 96
        assert (
            report["total_completed_updates"] == 376
            and report["total_training_row_exposures"] == 2672
        )
        assert report["pc_neutral_probe_curves_exact"] is True
    else:
        assert control["probes"] == [] and "probe_completed_updates" not in report
