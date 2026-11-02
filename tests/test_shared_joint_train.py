"""Ordinary shared TRAIN replay: tiny real ports and explicitly scripted capture."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import ticket_workflow_linear as linear
from src.adapters import linear_fit_learner as adapter
from src.app.context_joint_rehearsal import prepare_shared_rehearsal
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.stream_rehearsal import (
    StreamRehearsal,
    CoverageStreamReplayPolicy,
    RecentStreamReplayPolicy,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    EpisodeReplaySchema,
    encode_episode,
)
from src.core.learner_ports import TrainingDiagnostic
from src.core.shared_linear_replay import SharedJointReplayPolicy
from src.core.shared_replay_schedule import SharedReplaySelection
import test_linear_train_replay as legacy

SCRIPTED = dict(calls=0, rows=0)
SCHEMA = EpisodeReplaySchema(9, 1, 6)


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_linux_integration_preserves_existing_contextual_joint_route(control):
    from src.adapters.context_linear_fit_learner import ContextLinearFitLearner
    from src.core.context_linear_state import ContextLinearFitPolicy
    from src.core.context_joint_replay import JointContextReplayPolicy

    source = ContextLinearFitLearner(ContextLinearFitPolicy(9, 6, (7, 8)))
    pin = source.snapshot_state()
    wake = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    replay = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=2), wake.clock)
    helper = StreamRehearsal(
        source,
        wake,
        RecentStreamReplayPolicy(1, 2, ReplayRetentionBudget(2, 4096)),
        replay_budget=replay,
        total_updates=2,
        input_dim=9,
        episode_schema=SCHEMA,
        joint_context=JointContextReplayPolicy(9, 6, (7, 8), control),
    )
    batches = []
    for i in (0, 1):
        ep, x, y = episode(i)
        x[0, 7] = 1.0
        ep = replace(
            ep, transitions=(replace(ep.transitions[0], observation=tuple(float(v) for v in x[0])),)
        )
        batches.append(
            AttributionBatch(
                "train", ep.context, (f"e{i}:step:0",), x, y, num_classes=6, episode=ep
            )
        )
    stream = AttributionStream(
        source,
        tuple(batches),
        (),
        checkpoint_updates=(0, 2),
        label_delay=1,
        budget=wake,
        rehearsal=helper,
        training_only=True,
    )
    stream.run_until(2)
    assert source.snapshot_state() == pin == stream._runtime.actor.snapshot_state().state
    assert helper.report["committed_updates"] == 1
    assert "joint_context" in helper.report and "joint_shared" not in helper.report
    assert replay.replay_examples_completed == 2


def test_linux_integration_shared_continuation_preserves_bank_and_paid_prefix():
    source, stream, helper, wake, replay = tiny()
    stream.run_until(1)
    birth = (wake.started_at, replay.started_at)
    token = stream.pause()
    stream.resume(token)
    stream.run_until(2)
    assert (wake.started_at, replay.started_at) == birth
    assert helper.report["committed_updates"] == 1 and replay.replay_examples_completed == 2
    assert source.snapshot_state() == stream._runtime.actor.snapshot_state().state


def episode(index, role="train"):
    x = np.zeros((1, 9))
    x[0, index] = 1
    ep = DemonstrationEpisode(
        f"e{index}",
        f"c{index}",
        role,
        (
            DemonstrationTransition(
                0, tuple(float(v) for v in x[0]), index + 1, 1.0, (0.0,) * 9, True, 6
            ),
        ),
        6,
    )
    return ep, x, np.array([[float(index + 1)]])


def tiny(control="joint", *, consent=True, **options):
    source = linear.counted_learner(lambda x: None, lambda: None, train_update="minimum_change")
    wake = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    replay = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=2), wake.clock)
    retention = ReplayRetentionBudget(2, 2 * 8 * (SCHEMA.encoded_dim + 1))
    config: dict[str, Any] = dict(
        policy=CoverageStreamReplayPolicy(1, 2, retention, ("c0", "c1")),
        replay_budget=replay,
        total_updates=2,
        input_dim=9,
        episode_schema=SCHEMA,
        joint_shared=SharedJointReplayPolicy(9, 6, control),
    )
    config.update(options)
    helper = StreamRehearsal(source, wake, **config)
    batches = []
    for i in (0, 1):
        ep, x, y = episode(i)
        batches.append(
            AttributionBatch(
                "train", ep.context, (f"e{i}:step:0",), x, y, num_classes=6, episode=ep
            )
        )
    stream = AttributionStream(
        source,
        tuple(batches),
        (),
        label_delay=1,
        checkpoint_updates=(0, 1, 2),
        budget=wake,
        rehearsal=helper,
        training_only=True,
        replay_consent=consent,
    )
    return source, stream, helper, wake, replay


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_real_tiny_shared_fit_keeps_source_actor_and_original_authority(control):
    source, stream, helper, wake, replay = tiny(control)
    before = source.snapshot_state()
    stream.run_until(1)
    assert helper.report["events"][0]["status"] == "skipped"
    assert replay.replay_examples_completed == 0
    stream.run_until(2)
    state = stream.candidate_snapshot()
    assert state.completed_updates == 3 and state.training_examples == 4
    assert source.snapshot_state() == before == stream._runtime.actor.snapshot_state().state
    assert wake.updates_completed == 2 and wake.replay_examples_completed == 0
    assert replay.updates_completed == 0 and replay.replay_examples_completed == 2
    row = helper.report["events"][1]
    assert (
        row["proposal_applied_updates"] == row["committed_updates"] == row["returned_updates"] == 1
    )
    assert row["proposal_application_unknown_updates"] == 0
    assert row["joint_shared"]["retained_train_rows_used"] == (1 if control == "joint" else 0)
    assert row["joint_shared"]["current_rows_used"] == (1 if control == "joint" else 2)
    assert row["joint_shared"]["packet_array_bytes"] == 160
    assert helper.retention.example_count == 2
    stream.finish()


@pytest.mark.parametrize("fault", ["role", "recent", "geometry", "budget_policy"])
def test_shared_constructor_refuses_incompatible_scope_before_any_fit(fault):
    options: dict[str, Any]
    if fault == "role":
        options = dict(wake_role="probe_train")
    elif fault == "recent":
        options = dict(policy=RecentStreamReplayPolicy(1, 2, ReplayRetentionBudget(2, 4096)))
    elif fault == "geometry":
        options = dict(joint_shared=SharedJointReplayPolicy(8, 6))
    else:
        options = dict(
            policy=CoverageStreamReplayPolicy(1, 1, ReplayRetentionBudget(2, 4096), ("c0", "c1"))
        )
    with pytest.raises(ValueError):
        tiny(**options)


def test_shared_stream_requires_original_consent_before_any_fit():
    with pytest.raises(ValueError, match="original explicit TRAIN replay consent"):
        tiny(consent=False)


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_preparation_selects_original_distinct_cross_context_and_owns_arrays(control):
    a, x, y = episode(0)
    b, bx, by = episode(1)
    wires = tuple(encode_episode(ep, SCHEMA) for ep in (a, b))
    selected = SharedReplaySelection(("a", "b"), wires)
    before = deepcopy(wires)
    chosen, packet, detail = prepare_shared_rehearsal(
        SharedJointReplayPolicy(9, 6, control), SCHEMA, selected, (bx, by), 2, "train"
    )
    assert chosen.sample_ids == ("a",) and detail["selected_train_ids"] == ["a"]
    assert packet[0].shape == (2, 9) and packet[1].shape == (2, 1)
    assert np.array_equal(packet[0][0], bx[0])
    assert np.array_equal(packet[0][1], x[0] if control == "joint" else bx[0])
    assert not np.shares_memory(packet[0], bx)
    assert all(
        np.array_equal(v, pin) for pair, pins in zip(wires, before) for v, pin in zip(pair, pins)
    )


@pytest.mark.parametrize("fault", ["role", "labels", "shape", "foreign"])
def test_preparation_refuses_bad_current_or_retained_information(fault):
    a, x, y = episode(0)
    b, bx, by = episode(1)
    wires = (encode_episode(a, SCHEMA), encode_episode(b, SCHEMA))
    role = "train"
    if fault == "role":
        role = "final_test"
    elif fault == "labels":
        by[0, 0] = 99
    elif fault == "shape":
        bx = bx[:, :8]
    else:
        wires = (wires[0], (np.full_like(wires[1][0], np.nan), wires[1][1]))
    with pytest.raises(ValueError):
        prepare_shared_rehearsal(
            SharedJointReplayPolicy(9, 6),
            SCHEMA,
            SharedReplaySelection(("a", "b"), wires),
            (bx, by),
            2,
            role,
        )


def scripted(monkeypatch, failure=None):
    holder = legacy.scripted_capture(monkeypatch)

    def fit(self, features, targets, *, minimum_change):
        SCRIPTED["calls"] += 1
        SCRIPTED["rows"] += len(features)
        assert len(features) in (6, 12) and features.shape[1] == 9
        assert targets.shape == (len(features), 1)
        prior = self.snapshot_state()
        model = CategoricalLinearFit(
            9, 6, len(features), ((0.0,) * 6,) * 10, 0, (0.0,) * min(len(features), 10), 0.0
        )
        if len(features) == 12 and failure == "counter_only":
            model = prior.model
        self._state = replace(
            prior,
            model=model,
            completed_updates=prior.completed_updates + 1,
            training_examples=prior.training_examples + len(features),
        )
        if len(features) == 12 and failure in ("applied", "unknown"):
            if failure == "unknown":

                def unreadable():
                    raise RuntimeError("lost proposal state")

                self.snapshot_state = unreadable
            raise RuntimeError("lost shared fit return")
        return TrainingDiagnostic(
            adapter.MINIMUM_CHANGE_RESIDUAL_ID if minimum_change else adapter.LINEAR_RESIDUAL_ID,
            0.0,
        )

    monkeypatch.setattr(adapter.LinearFitLearner, "_train_batch", fit)
    return holder


def test_counter_only_return_is_unknown_and_cannot_publish(monkeypatch):
    holder = scripted(monkeypatch, "counter_only")
    result = entry.run_example(
        method="linear_fit",
        context_return=True,
        replay="coverage",
        replay_joint="joint",
        train_update="minimum_change",
    )
    assert not result["complete"]
    row = result["replay"]["events"][2]
    assert row["attempted_updates"] == row["returned_updates"] == 1
    assert row["proposal_applied_updates"] == 0
    assert row["proposal_application_unknown_updates"] == 1
    assert row["committed_updates"] == 0 and row["charged_exposures"] == 12
    assert row["unreturned_fit_work_unknown"] is False
    assert holder["stream"]._runtime._candidate.snapshot_state().completed_updates == 6
    assert holder["wake"].updates_completed == 6 and result["replay_examples"] == 12


@pytest.mark.parametrize("failure", ["applied", "unknown"])
def test_scripted_lost_proposal_preserves_charge_and_application_scope(failure, monkeypatch):
    holder = scripted(monkeypatch, failure)
    result = entry.run_example(
        method="linear_fit",
        context_return=True,
        replay="coverage",
        replay_joint="joint",
        train_update="minimum_change",
    )
    assert not result["complete"]
    row = result["replay"]["events"][2]
    assert row["status"] == "failed" and row["charged_exposures"] == 12
    assert (
        row["attempted_updates"] == 1 and row["returned_updates"] == row["committed_updates"] == 0
    )
    assert row["proposal_applied_updates"] == (1 if failure == "applied" else 0)
    assert row["proposal_application_unknown_updates"] == (1 if failure == "unknown" else 0)
    assert row["unreturned_fit_work_unknown"] is True
    assert holder["wake"].updates_completed == 6
    assert result["replay_examples"] == 12
    assert "source_unchanged" not in result and "actor_unchanged" not in result
    assert (
        holder["source"].snapshot_state() == holder["stream"]._runtime.actor.snapshot_state().state
    )
    assert holder["source"].snapshot_state().completed_updates == 0


@pytest.mark.parametrize("control", ["joint", "current_only"])
@pytest.mark.parametrize("operation", [None, "minimum_change"])
def test_scripted_ordinary_capture_has_skips_then_four_paid_shared_consolidations(
    control, operation, monkeypatch
):
    holder = scripted(monkeypatch)
    result = entry.run_example(
        method="linear_fit",
        context_return=True,
        replay="coverage",
        replay_joint=control,
        train_update=operation,
    )
    assert result["complete"] is True, result.get("error")
    assert result["linear_replay_joint"] == control
    assert [row["status"] for row in result["replay"]["events"]] == ["skipped"] * 2 + [
        "complete"
    ] * 4
    assert (
        result["replay_examples"] == 48
        and result["replay_budget"]["limits"]["max_replay_examples"] == 72
    )
    assert {
        key: result["training_work"][key]
        for key in ("attempted_updates", "returned_updates", "attempted_rows")
    } == dict(attempted_updates=16, returned_updates=16, attempted_rows=120)
    assert result["training_work"]["budget_committed_updates"] == 12
    assert result["training_work"]["unreturned_work_unknown"] is False
    assert holder["wake"].started_at == holder["birth"]
    assert holder["wake"].updates_completed == 12
    assert (
        holder["source"].snapshot_state() == holder["stream"]._runtime.actor.snapshot_state().state
    )
    assert all(row["episode"]["success"] is False for row in result["episodes"])
    assert (
        result["source_unchanged"] and result["actor_unchanged"] and result["numpy_rng_unchanged"]
    )


@pytest.mark.parametrize("fault", ["row", "replace", "bank", "array", "budget", "intent"])
def test_scripted_corrupt_prior_evidence_refuses_before_more_fits(fault, monkeypatch):
    holder = scripted(monkeypatch)
    original = linear.checkpoint

    def corrupt(point, state):
        value = original(point, state)
        if point == 8:
            helper = holder["stream"]._rehearsal
            if fault == "row":
                helper._events[2]["charged_exposures"] = 0
            elif fault == "replace":
                helper._events[2] = deepcopy(helper._events[2])
            elif fault == "bank":
                helper._buffer = deepcopy(helper._buffer)
            elif fault == "array":
                helper._buffer._rows = deepcopy(helper._buffer._rows)
            elif fault == "budget":
                helper._replay_budget.replay_examples_completed += 1
            else:
                holder["report"]["linear_replay_joint"] = "current_only"
            holder["at_corruption"] = SCRIPTED["calls"]
        return value

    monkeypatch.setattr(linear, "checkpoint", corrupt)
    result = entry.run_example(
        method="linear_fit",
        context_return=True,
        replay="coverage",
        replay_joint="joint",
        train_update="minimum_change",
    )
    assert not result["complete"]
    assert SCRIPTED["calls"] == holder["at_corruption"]


@pytest.mark.parametrize(
    "options",
    [
        dict(replay="recent"),
        dict(replay="none"),
        dict(method="bp"),
        dict(separate_context_heads=True),
        dict(reserved_probe=True),
        dict(private_adaptation=True),
        dict(paired_serving="joint"),
        dict(context_return=False),
        dict(probe_update="minimum_change"),
        dict(replay_joint=True),
    ],
)
def test_invalid_joint_mode_refuses_before_data(options, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("data before refusal")

    monkeypatch.setattr(entry, "training_cases", forbidden)
    kwargs = dict(method="linear_fit", context_return=True, replay="coverage", replay_joint="joint")
    kwargs.update(options)
    with pytest.raises(ValueError):
        entry.run_example(**kwargs)


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_cli_transports_only_explicit_joint_intent_and_protects_output(
    control, tmp_path, monkeypatch
):
    calls = []

    def run(**kw):
        calls.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_example", run)
    target = tmp_path / "out.json"
    flags = [
        "--method",
        "linear_fit",
        "--context-return",
        "--replay",
        "coverage",
        "--replay-joint",
        control,
        "--output",
        str(target),
    ]
    assert entry.main(flags) == 0
    assert calls == [
        dict(context_return=True, method="linear_fit", replay="coverage", replay_joint=control)
    ]
    saved = target.read_bytes()
    with pytest.raises(FileExistsError):
        entry.main(flags)
    assert len(calls) == 1 and target.read_bytes() == saved
