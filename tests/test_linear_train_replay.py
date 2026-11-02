"""Tiny actual consolidation and scripted ordinary captures, never a study."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import ticket_workflow_linear as linear
from src.adapters import linear_fit_learner as adapter
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.stream_rehearsal import (
    CoverageStreamReplayPolicy,
    RecentStreamReplayPolicy,
    StreamRehearsal,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    EpisodeReplaySchema,
)
from src.core.decision_features import DecisionFeatureSpec
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed

SCRIPTED = dict(calls=0, rows=0)


def tiny_stream(mode="recent", selected="minimum_change", role="train"):
    source = linear.counted_learner(
        lambda x: None, lambda: None, **(dict(train_update=selected) if selected else {})
    )
    schema = EpisodeReplaySchema(9, 1, 6)
    retention = ReplayRetentionBudget(2, 2 * 8 * (schema.encoded_dim + 1))
    policy = (
        RecentStreamReplayPolicy(1, 1, retention)
        if mode == "recent"
        else CoverageStreamReplayPolicy(1, 1, retention, ("c0",))
    )
    wake = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    replay = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=1), wake.clock)
    helper = StreamRehearsal(
        source,
        wake,
        policy,
        replay_budget=replay,
        total_updates=1,
        input_dim=9,
        episode_schema=schema,
    )
    x = np.zeros((1, 9))
    x[0, 0] = 1.0
    episode = DemonstrationEpisode(
        "e0",
        "c0",
        role,
        (DemonstrationTransition(0, tuple(float(v) for v in x[0]), 1, 1.0, (0.0,) * 9, True, 6),),
        6,
    )
    batch = AttributionBatch(
        role, "c0", ("e0:step:0",), x, np.array([[1.0]]), episode=episode, num_classes=6
    )
    stream = AttributionStream(
        source,
        (batch,),
        (),
        label_delay=1,
        checkpoint_updates=(0, 1),
        budget=wake,
        rehearsal=helper,
        training_only=True,
    )
    return source, stream, helper, wake, replay


@pytest.mark.parametrize("mode", ["recent", "coverage"])
@pytest.mark.parametrize("selected", [None, "minimum_change"])
def test_tiny_real_replay_commits_owned_candidate_and_separate_charges(mode, selected):
    source, stream, helper, wake, replay = tiny_stream(mode, selected)
    before = source.snapshot_state()
    stream.run_until(1)
    candidate = stream.candidate_snapshot()
    assert source.snapshot_state() == stream._runtime.actor.snapshot_state().state == before
    assert candidate.completed_updates == candidate.training_examples == 2
    assert wake.updates_completed == 1 and wake.replay_examples_completed == 0
    assert replay.replay_examples_completed == 1 and replay.updates_completed == 0
    assert len(stream.applied_updates) == len(stream.applied_consolidations) == 1
    assert helper.report["committed_updates"] == helper.report["returned_updates"] == 1
    assert helper.report["charged_exposures"] == 1
    expected = adapter.MINIMUM_CHANGE_RESIDUAL_ID if selected else adapter.LINEAR_RESIDUAL_ID
    assert stream.applied_consolidations[0].diagnostic.definition == expected
    reader = linear.readout(DecisionFeatureSpec(entry.CONTEXT_FEATURES, entry.CATALOG), candidate)
    query = np.zeros((1, 9))
    query[0, 0] = 1.0
    assert np.argmax(reader.predict_logits(query), axis=1).tolist() == [1]
    stream.finish()


def test_lost_real_replay_return_keeps_wake_commit_and_paid_replay(monkeypatch):
    source, stream, helper, wake, replay = tiny_stream()
    before = source.snapshot_state()
    original = adapter.update_categorical_linear

    def lose(*args):
        original(*args)
        raise RuntimeError("lost replay return")

    monkeypatch.setattr(adapter, "update_categorical_linear", lose)
    with pytest.raises(TrainingUpdateFailed):
        stream.run_until(1)
    assert source.snapshot_state() == stream._runtime.actor.snapshot_state().state == before
    assert stream._runtime._candidate.snapshot_state().completed_updates == 1
    assert wake.updates_completed == replay.replay_examples_completed == 1
    assert len(stream.applied_updates) == 1 and not stream.applied_consolidations
    assert helper.report["attempted_updates"] == helper.report["charged_exposures"] == 1
    assert helper.report["returned_updates"] == helper.report["committed_updates"] == 0
    with pytest.raises(ValueError):
        stream.run_until(1)


@pytest.mark.parametrize("role", ["final_test", "probe_train"])
def test_nontrain_episode_refuses_before_any_tiny_fit(role):
    with pytest.raises(ValueError):
        tiny_stream(role=role)


@pytest.mark.parametrize("mode", ["recent", "coverage"])
@pytest.mark.parametrize("selected", [None, "minimum_change"])
def test_cli_replay_transport_and_existing_output_refusal(mode, selected, tmp_path, monkeypatch):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_example", run)
    output = tmp_path / "result.json"
    flags = [
        "--method",
        "linear_fit",
        "--context-return",
        "--replay",
        mode,
        "--output",
        str(output),
    ]
    if selected:
        flags += ["--train-update", selected]
    assert entry.main(flags) == 0
    assert calls == [
        dict(
            method="linear_fit",
            context_return=True,
            replay=mode,
            **(dict(train_update=selected) if selected else {}),
        )
    ]
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        entry.main(flags)
    assert len(calls) == 1 and output.read_bytes() == saved


@pytest.mark.parametrize(
    "options",
    [
        dict(separate_context_heads=True),
        dict(method="bp"),
        dict(context_return=False),
        dict(paired_serving="joint"),
        dict(reserved_probe=True),
        dict(private_adaptation=True),
        dict(catalog_study=True),
        dict(record_train_fit=True),
        dict(replay=True),
        dict(replay=[]),
    ],
)
def test_unsupported_combinations_refuse_before_data(options, monkeypatch):
    monkeypatch.setattr(entry, "training_cases", lambda **kw: pytest.fail("source data"))
    kwargs = dict(method="linear_fit", context_return=True, replay="recent")
    kwargs.update(options)
    with pytest.raises(ValueError):
        entry.run_example(**kwargs)


def scripted_capture(monkeypatch, *, fail_replay=False):
    """Actual role/codec/runtime/receipt path, scripted fits and false task outcomes."""
    holder: dict[str, Any] = dict(calls=0, report=None)
    original_stream = entry.AttributionStream
    original_intent = linear.require_train_update

    def make_stream(*args, **kwargs):
        assert kwargs["training_only"] is True and args[2] == ()
        assert all(b.role == b.episode.role == "train" for b in args[1])
        holder["source"] = args[0]
        holder["stream"] = original_stream(*args, **kwargs)
        holder["wake"] = kwargs["budget"]
        holder["birth"] = kwargs["budget"].started_at
        return holder["stream"]

    def intent(report, selected):
        holder["report"] = report
        original_intent(report, selected)

    def fit(self, features, targets, *, minimum_change):
        SCRIPTED["calls"] += 1
        SCRIPTED["rows"] += len(features)
        assert SCRIPTED["calls"] <= 512 and SCRIPTED["rows"] <= 8192
        holder["calls"] += 1
        assert features.shape == (6, 9) and targets.shape == (6, 1)
        definition = (
            adapter.MINIMUM_CHANGE_RESIDUAL_ID if minimum_change else adapter.LINEAR_RESIDUAL_ID
        )
        if fail_replay and holder["calls"] == 3:
            raise TrainingUpdateFailed(
                "scripted lost replay return",
                definition,
                dict(
                    fit_attempts=1,
                    fit_rows_attempted=6,
                    fit_returns=0,
                    candidate_commits=0,
                    unreturned_work_unknown=True,
                ),
            )
        prior = self.snapshot_state()
        model = CategoricalLinearFit(9, 6, 6, ((0.0,) * 6,) * 10, 0, (0.0,) * 6, 0.0)
        self._state = replace(
            prior,
            model=model,
            completed_updates=prior.completed_updates + 1,
            training_examples=prior.training_examples + 6,
        )
        return TrainingDiagnostic(definition, 0.0)

    monkeypatch.setattr(entry, "AttributionStream", make_stream)
    monkeypatch.setattr(linear, "require_train_update", intent)
    monkeypatch.setattr(adapter.LinearFitLearner, "_train_batch", fit)
    monkeypatch.setattr(
        entry,
        "run_workflow_episode",
        lambda *a, **kw: dict(
            complete=True,
            success=False,
            scenario=kw.get("scenario", "ordinary"),
            training_updates=0,
            work=dict(prediction_calls_validated=0),
        ),
    )
    return holder


@pytest.mark.parametrize("mode", ["recent", "coverage"])
@pytest.mark.parametrize("selected", [None, "minimum_change"])
def test_scripted_full_capture_records_wake_replay_roles_costs_and_original_authority(
    mode, selected, monkeypatch
):
    holder = scripted_capture(monkeypatch)
    result = entry.run_example(
        method="linear_fit",
        context_return=True,
        replay=mode,
        **(dict(train_update=selected) if selected else {}),
    )
    assert result["complete"], result.get("error")
    assert holder["calls"] == 18
    assert len(result["applied_updates"]) == 12 and len(result["published_replay_receipts"]) == 6
    assert (
        result["training_work"]["attempted_updates"]
        == result["training_work"]["returned_updates"]
        == 18
    )
    assert result["training_work"]["attempted_rows"] == 108
    assert result["replay_examples"] == result["replay_budget"]["replay_examples_completed"] == 36
    assert (
        result["budget"]["updates_completed"] == 12
        and result["budget"]["replay_examples_completed"] == 0
    )
    assert result["budget"]["started_at"] == holder["birth"]
    assert (
        result["budget"]["limits"]["max_wall_seconds"]
        == result["replay_budget"]["limits"]["max_wall_seconds"]
        == 30
    )
    breakdown = result["costs"]["fit_exposure_breakdown"]
    assert breakdown["wake_committed_updates"] == 12 and breakdown["wake_committed_rows"] == 72
    assert (
        breakdown["replay_admissions"]
        == breakdown["replay_returned_updates"]
        == breakdown["replay_committed_updates"]
        == 6
    )
    assert breakdown["replay_charged_rows"] == 36
    assert all(
        result[k]
        for k in (
            "source_unchanged",
            "actor_unchanged",
            "numpy_rng_unchanged",
            "python_rng_unchanged",
        )
    )
    assert len(result["episodes"]) == 24 and all(
        not r["episode"]["success"] for r in result["episodes"]
    )
    assert result["final_examples"] == result["probe_updates"] == 0
    assert all(
        ep["role"] == "train"
        for event in result["replay"]["events"]
        for ep in event["selected_episodes"]
    )


def test_scripted_lost_replay_capture_keeps_paid_exposure_and_unknown_physical_work(monkeypatch):
    holder = scripted_capture(monkeypatch, fail_replay=True)
    result = entry.run_example(
        method="linear_fit", context_return=True, replay="recent", train_update="minimum_change"
    )
    assert result["complete"] is False and holder["calls"] == 3
    assert (
        result["training_work"]["attempted_updates"] == 3
        and result["training_work"]["returned_updates"] == 2
    )
    assert result["budget"]["updates_completed"] == 2 and result["replay_examples"] == 6
    assert result["costs"]["physical_lstsq_calls"] is None
    breakdown = result["costs"]["fit_exposure_breakdown"]
    assert breakdown["wake_committed_rows"] == 12 and breakdown["replay_charged_rows"] == 6
    assert breakdown["replay_admissions"] == 1 and breakdown["replay_returned_updates"] == 0
    assert not result["published_replay_receipts"]
    assert holder["source"].snapshot_state().model is None
    assert holder["stream"]._runtime.actor.snapshot_state().state.model is None


@pytest.mark.parametrize("fault", ["checkpoint", "replay_receipt"])
def test_scripted_prior_corruption_stops_before_any_more_wake_or_replay(fault, monkeypatch):
    holder = scripted_capture(monkeypatch)
    original = linear.checkpoint

    def corrupt(point, state):
        if point == 4:
            if fault == "checkpoint":
                holder["report"]["checkpoints"][0]["source_updates"] = 99
            else:
                receipts = holder["stream"]._runtime._consolidations
                receipts[0] = replace(receipts[0], diagnostic=TrainingDiagnostic("foreign", 0.0))
        return original(point, state)

    monkeypatch.setattr(linear, "checkpoint", corrupt)
    result = entry.run_example(
        method="linear_fit", context_return=True, replay="coverage", train_update="minimum_change"
    )
    assert result["complete"] is False and holder["calls"] == 6
    assert result["budget"]["updates_completed"] == 4 and result["replay_examples"] == 12
    assert result["costs"]["fit_exposure_breakdown"]["wake_committed_rows"] == 24


def test_missing_or_malformed_breakdown_fields_stay_unknown():
    report: dict[str, Any] = dict(training_work={}, budget=None, replay=None, training_data=None)
    result = linear.replay_costs(report)
    assert all(
        result[k] is None
        for k in (
            "wake_committed_updates",
            "wake_committed_rows",
            "replay_admissions",
            "replay_returned_updates",
            "replay_charged_rows",
        )
    )


@pytest.mark.parametrize("field", ["replay_mode", "replay_consent"])
def test_changed_replay_intent_or_consent_refuses_before_more_paid_work(field, monkeypatch):
    holder = scripted_capture(monkeypatch)
    original = linear.checkpoint

    def corrupt(point, state):
        if point == 4:
            holder["report"][field] = "foreign"
        return original(point, state)

    monkeypatch.setattr(linear, "checkpoint", corrupt)
    result = entry.run_example(
        method="linear_fit", context_return=True, replay="coverage", train_update="minimum_change"
    )
    assert result["complete"] is False and "replay intent or consent changed" in result["error"]
    assert holder["calls"] == 6 and result["budget"]["updates_completed"] == 4
    assert result["replay_examples"] == 12 and result[field] == "foreign"


@pytest.mark.parametrize("forged_rows", [0, 999])
def test_refused_display_row_edit_does_not_rewrite_committed_wake_cost(forged_rows, monkeypatch):
    holder = scripted_capture(monkeypatch)
    original = linear.checkpoint

    def corrupt(point, state):
        if point == 4:
            holder["report"]["training_data"][0]["rows"] = forged_rows
        return original(point, state)

    monkeypatch.setattr(linear, "checkpoint", corrupt)
    result = entry.run_example(
        method="linear_fit", context_return=True, replay="coverage", train_update="minimum_change"
    )
    assert result["complete"] is False and "context TRAIN input changed" in result["error"]
    assert holder["calls"] == 6 and result["training_data"][0]["rows"] == forged_rows
    costs = result["costs"]["fit_exposure_breakdown"]
    assert costs["wake_committed_updates"] == 4 and costs["wake_committed_rows"] == 24
    assert costs["replay_charged_rows"] == 12
    assert result["training_work"]["attempted_rows"] == 36
