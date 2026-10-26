"""Paid replay composition using scripted native ports, never numerical Torch."""

from copy import deepcopy
from dataclasses import asdict, replace
from time import monotonic
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.run_minigrid_linear_fit_control import read_training_sources
from src.adapters import minigrid_learning_pilot as shared
from src.adapters import minigrid_public_cue_recurrent_pilot as pilot
from src.adapters import minigrid_public_cue_recurrent_learner as port
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import minigrid_recurrent_replay as replay
from src.app.coupled_attribution import AttributionStream, _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coverage_replay import CoverageReplayBuffer
from src.core.shared_replay_schedule import SharedReplayBuffer
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.demonstration_episode import encode_episode, decode_episode
from src.core.learner_ports import TrainingUpdateFailed
from src.infra.minigrid_affine_example import example_affine_training
from src.shared.process_memory import ProcessRssSampler
from test_public_cue_recurrent_learner import configure_scripted_native, ScriptedNative


@pytest.fixture(scope="module")
def sources():
    with example_affine_training() as path:
        return pilot._typed_sources(read_training_sources(path, raw_frames=True))


@pytest.fixture(scope="module")
def packets(sources):
    original = shared._packets(sources)
    return replay.prepare_packets(
        tuple(original[i] for start in (0, 32, 64, 96) for i in range(start, start + 8))
    )


@pytest.fixture
def scripted(monkeypatch):
    configure_scripted_native(monkeypatch)
    monkeypatch.setattr(shared, "_same_state", _same_value)


def setup(packets, memory, mode="coverage"):
    source = port.PublicCueRecurrentLearner()
    wake = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=32, max_wall_seconds=90, max_process_rss_bytes=512 * 1024**2
        ),
        monotonic,
    )
    wake.attach_memory(memory)
    witness = replay.RecurrentReplayWitness(source, wake, packets, mode, memory)
    stream = AttributionStream(
        source,
        packets,
        (),
        checkpoint_updates=(0, 32),
        label_delay=1,
        budget=wake,
        rehearsal=witness.helper,
        training_only=True,
    )
    return source, wake, witness, stream


def advance(stream, witness, until):
    for point in range(stream.completed_updates + 1, until + 1):
        witness.admit(point - 1)
        stream.run_until(point)
        witness.admit(point)


@pytest.mark.parametrize("mode", replay.MODES)
def test_should_match_entire_causal_selection_and_per_head_dose_without_models(sources, mode):
    packets = replay.prepare_packets(shared._packets(sources))
    limits = ReplayRetentionBudget(6, 256 * 1024)
    bank: Any = (
        CoverageReplayBuffer(
            replay.SCHEMA.encoded_dim, limits, schema=replay.SCHEMA, contexts=replay.KEYS
        )
        if mode == "coverage"
        else SharedReplayBuffer(
            replay.SCHEMA.encoded_dim, limits, ReplayRetentionPolicy("recent_fifo")
        )
    )
    cues = [0, 0]
    for point, packet in enumerate(packets, 1):
        assert packet.episode is not None
        bank.observe_train_batch(*encode_episode(packet.episode, replay.SCHEMA))
        if point % 8 == 0:
            selected = bank.select_coverage(2) if mode == "coverage" else bank.select_recent(2)
            episodes = tuple(
                decode_episode(x, y, replay.SCHEMA) for x, y in selected.training_batches()
            )
            assert episodes == replay.expected_episodes(packets, mode, point)
            assert [replay._cue(shared.episode_arrays(e)[0][0]) for e in episodes] == [0, 1]
            for i in (0, 1):
                cues[i] += 1
        assert bank.retention.example_count <= 6
        assert bank.retention.retained_bytes <= 256 * 1024
    assert cues == [16, 16]
    if mode == "coverage":
        assert [e.context for e in replay.expected_episodes(packets, mode, 48)] == [
            "A-blue",
            "A-red",
        ]
    else:
        assert [e.context for e in replay.expected_episodes(packets, mode, 48)] == [
            "B-blue",
            "B-red",
        ]


@pytest.mark.parametrize("mode", replay.MODES)
def test_should_run_existing_training_arm_with_original_authorities_and_real_replay_clocks(
    scripted, packets, mode
):
    row = shared._row(port.METHOD)
    with ProcessRssSampler() as memory:
        arm = shared._train_arm(
            row,
            packets,
            memory,
            monotonic() + 90,
            lambda: None,
            learner_factory=lambda _: port.PublicCueRecurrentLearner(),
            checkpoint_updates=(0, 32),
            recurrent_replay=mode,
        )
        arm["guard"]()
        saved = replay.check_snapshot(arm["states"][-1][1], 32)
        assert saved.completed_updates == 40 and saved.training_examples == 360
        assert [h.completed_updates for h in saved.heads] == [20, 20]
        assert row["charged_updates"] == 32 and row["native_training_examples"] == 360
        assert row["source_unchanged"] is row["actor_unchanged"] is True
        assert row["replay_budget"]["charged_examples"] == 72
        assert row["replay"]["verified_completed_events"] == 4
        assert row["replay"]["prior_evidence_preserved"] is True
        assert len(ScriptedNative.calls) == 40
        assert all(c[0] == "train" and c[1].shape == (9, 148) for c in ScriptedNative.calls)
        assert (
            arm["budget"].updates_completed == 32 and arm["budget"].replay_examples_completed == 0
        )


@pytest.mark.parametrize("fault", ["mode", "role", "order", "untagged", "class"])
def test_should_refuse_bad_input_before_factory(packets, fault):
    values, mode = packets, "coverage"
    if fault == "mode":
        mode = "other"
    elif fault == "role":
        values = (replace(packets[0], role="final_test"), *packets[1:])
    elif fault == "order":
        values = packets[::-1]
    elif fault == "untagged":
        values = (replace(packets[0], context="A"), *packets[1:])
    else:
        values = (replace(packets[0], num_classes=2), *packets[1:])
    with pytest.raises(ValueError):
        shared._train_arm(
            shared._row(port.METHOD),
            values,
            None,
            0,
            lambda: None,
            learner_factory=lambda _: pytest.fail("factory forbidden"),
            checkpoint_updates=(0, 32),
            recurrent_replay=mode,
        )


@pytest.mark.parametrize(
    "fault",
    ["helper", "budget", "birth", "clock", "progress", "charge", "input", "source", "future"],
)
def test_should_refuse_replaced_authorities_or_unobserved_changes(scripted, packets, fault):
    with ProcessRssSampler() as memory:
        source, wake, witness, stream = setup(packets, memory)
        if fault == "helper":
            witness.helper._buffer = deepcopy(witness.helper._buffer)
        elif fault == "budget":
            wake.budget = replace(wake.budget)
        elif fault == "birth":
            wake.started_at -= 1
        elif fault == "clock":
            wake.last_clock -= 1
        elif fault == "progress":
            wake.progress = object()
        elif fault == "charge":
            witness.replay_budget.replay_examples_completed = 9
        elif fault == "input":
            original = packets[0].features[0, 0]
            packets[0].features[0, 0] += 0.125
        elif fault == "source":
            source._state = replace(
                source.snapshot_state(), failed_work=port.FailureWork("train", 0)
            )
        else:
            witness.helper._events[-1]["reason"] = "unobserved"
        try:
            with pytest.raises((ValueError, TypeError)):
                witness.admit(0)
        finally:
            if fault == "input":
                packets[0].features[0, 0] = original
        assert ScriptedNative.calls == []


@pytest.mark.parametrize("replace_row", [False, True])
def test_should_refuse_corrupted_or_replaced_completed_prefix_before_more_work(
    scripted, packets, replace_row
):
    with ProcessRssSampler() as memory:
        _, _, witness, stream = setup(packets, memory)
        advance(stream, witness, 8)
        original = deepcopy(witness.helper._events[0])
        if replace_row:
            witness.helper._events[0] = deepcopy(original)
        else:
            witness.helper._events[0]["diagnostics"][0]["objective"] = 999
        with pytest.raises(ValueError):
            witness.admit(8)
        assert len(ScriptedNative.calls) == 10
        evidence = witness.evidence()
        assert evidence["prior_evidence_preserved"] is None
        assert evidence["verified_completed_prefix"] == [original]
        assert witness.replay_budget.replay_examples_completed == 18


def test_should_charge_failed_replay_and_preserve_unknown_native_work(
    scripted, packets, monkeypatch
):
    original = ScriptedNative.train_batch

    def fail(self, x, y):
        result = original(self, x, y)
        if len(ScriptedNative.calls) == 9:
            raise RuntimeError("native returned internally then lost")
        return result

    monkeypatch.setattr(ScriptedNative, "train_batch", fail)
    row = shared._row(port.METHOD)
    with ProcessRssSampler() as memory:
        with pytest.raises(TrainingUpdateFailed):
            shared._train_arm(
                row,
                packets,
                memory,
                monotonic() + 90,
                lambda: None,
                learner_factory=lambda _: port.PublicCueRecurrentLearner(),
                checkpoint_updates=(0, 32),
                recurrent_replay="coverage",
            )
    assert len(ScriptedNative.calls) == 9
    assert row["charged_updates"] == 8 and row["replay_budget"]["charged_examples"] == 9
    assert row["replay"]["events"][0]["attempted_updates"] == 1
    assert row["replay"]["events"][0]["committed_updates"] == 0
    assert row["replay_failure_work"]["failure_work_may_be_unobserved"] is True


@pytest.mark.parametrize("mode", replay.MODES)
def test_should_admit_native_replay_clock_only_under_explicit_mode(
    scripted, sources, monkeypatch, mode
):
    owner = port.PublicCueRecurrentLearner()
    cold = owner.snapshot_state()
    saved = replace(
        cold,
        heads=(
            replace(cold.heads[0], completed_updates=20, training_examples=180),
            replace(cold.heads[1], completed_updates=20, training_examples=180),
        ),
    )
    seen = []

    def kernel(*a, **k):
        seen.append(k)
        return k

    monkeypatch.setattr(evaluation, "_run_checkpoint", kernel)
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=0,
            max_replay_examples=0,
            max_wall_seconds=200,
            max_process_rss_bytes=pilot.RSS_LIMIT,
        ),
        monotonic,
    )
    options = dict(
        updates=32,
        sources=sources,
        budget=budget,
        check_boundary=lambda: None,
        recurring_training=True,
    )
    result = evaluation.evaluate_public_cue_recurrent_checkpoint(
        owner, saved, recurrent_replay=mode, **options
    )
    assert result["return_metadata"]["native_training_updates"] == 40
    assert result["return_metadata"]["logical_wake_updates"] == 32
    with pytest.raises(ValueError):
        evaluation.evaluate_public_cue_recurrent_checkpoint(owner, saved, **options)
    assert len(seen) == 1


def test_should_keep_raw_readout_metadata_independent_of_history(scripted):
    owner = port.PublicCueRecurrentLearner()
    saved = owner.snapshot_state()
    with ProcessRssSampler() as memory:
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=0,
                max_replay_examples=0,
                max_wall_seconds=200,
                max_process_rss_bytes=pilot.RSS_LIMIT,
            ),
            monotonic,
        )
        budget.attach_memory(memory)
        result = evaluation._run_checkpoint(
            owner,
            saved,
            method=port.METHOD,
            updates=0,
            layouts=(),
            role="train",
            sources=(),
            budget=budget,
            check_boundary=lambda: None,
            memory_copier=port.copy_snapshot,
            state_equal=_same_value,
            public_history=False,
            live_plan=(),
            return_metadata=dict(native_training_updates=0),
        )
    assert result["status"] == "complete" and result["native_training_updates"] == 0
    assert "public_history" not in result and ScriptedNative.calls == []


@pytest.mark.parametrize("mode", replay.MODES)
def test_should_declare_actual_counts_and_preserve_default_protocol(mode):
    old = pilot.describe_protocol(typed_input=True, recurring_training=True)
    new = pilot.describe_protocol(typed_input=True, recurring_training=True, recurrent_replay=mode)
    assert old["planned_work"]["updates"] == 128 and "replay" not in old
    assert new["planned_work"]["updates"] == 160 and new["planned_work"]["training_rows"] == 1440
    assert asdict(replay.policy(mode))["retention"] == asdict(replay.RETENTION)
    with pytest.raises(ValueError):
        pilot.describe_protocol(typed_input=True, recurrent_replay=mode)


def test_should_refuse_prior_diagnostic_corruption_inside_later_native_return(
    scripted, packets, monkeypatch
):
    original = ScriptedNative.train_batch
    captured: dict[str, Any] = {}

    def train(self, x, y):
        diagnostic = original(self, x, y)
        if len(ScriptedNative.calls) == 10:
            captured["witness"].helper._events[0]["diagnostics"][0]["value"] += 1
        return diagnostic

    monkeypatch.setattr(ScriptedNative, "train_batch", train)
    with ProcessRssSampler() as memory:
        _, wake, witness, stream = setup(packets, memory)
        captured["witness"] = witness
        with pytest.raises(ValueError, match="diagnostic"):
            advance(stream, witness, 8)
        assert wake.updates_completed == 8 and witness.replay_budget.replay_examples_completed == 18
        assert witness.helper._events[0]["committed_updates"] == 0
        assert witness.helper._events[0]["returned_updates"] == 2
        assert stream._runtime._candidate.completed_updates == 8


def test_should_refuse_callback_replacement_and_unsupported_guard_continuation(scripted, packets):
    with ProcessRssSampler() as memory:
        _, _, witness, stream = setup(packets, memory)
        advance(stream, witness, 1)
        with pytest.raises(ValueError, match="continuation"):
            stream.pause()
        witness.helper._check_boundary = lambda: None
        with pytest.raises(ValueError):
            witness.helper._before_native()
        assert len(ScriptedNative.calls) == 1


@pytest.mark.parametrize("counts", [(16, 16), (19, 21)])
def test_should_refuse_lost_replay_or_unmatched_per_head_clocks(scripted, counts):
    cold = port.PublicCueRecurrentLearner().snapshot_state()
    left, right = counts
    saved = replace(
        cold,
        heads=(
            replace(cold.heads[0], completed_updates=left, training_examples=left * 9),
            replace(cold.heads[1], completed_updates=right, training_examples=right * 9),
        ),
    )
    with pytest.raises(ValueError, match="clocks"):
        replay.check_snapshot(saved, 32)


@pytest.mark.parametrize("fault", ["none", "fit", "readout", "row", "arm", "budget", "source"])
def test_should_fit_both_controls_before_readouts_and_preserve_failed_prefix(
    scripted, sources, monkeypatch, fault
):
    from test_recurrent_recurring_control import readout

    class Rng:
        def clone(self):
            return self

    rng = Rng()
    monkeypatch.setattr(
        pilot,
        "require_torch",
        lambda: SimpleNamespace(get_rng_state=lambda: rng, equal=lambda a, b: a is b),
    )
    calls: list[tuple] = []
    captured = []

    def train(row, packets, memory, deadline, check, *, learner_factory, recurrent_replay):
        check()
        if recurrent_replay == "coverage" and fault == "fit":
            row.update(charged_updates=1, status="incomplete")
            raise TrainingUpdateFailed("lost", "scripted", {"calls": 1})
        source = learner_factory(port.METHOD)
        cold = source.snapshot_state()
        states = []
        for point in shared.POINTS:
            n = replay.native_updates(point) // 2
            states.append(
                (
                    point,
                    replace(
                        cold,
                        heads=(
                            replace(cold.heads[0], completed_updates=n, training_examples=9 * n),
                            replace(cold.heads[1], completed_updates=n, training_examples=9 * n),
                        ),
                    ),
                )
            )
        row.update(status="trained", charged_updates=128, native_training_examples=1440)
        arm = dict(
            source=source,
            states=states,
            stream=object(),
            budget=object(),
            replay=object(),
            guard=lambda: None,
        )
        captured.append((row, arm))
        calls.append(("fit", recurrent_replay))
        return arm

    def observe(source, state, **kw):
        mode, point = kw["recurrent_replay"], kw["updates"]
        assert calls[:2] == [("fit", "recent"), ("fit", "coverage")]
        calls.append(("read", mode, point))
        kw["check_boundary"]()
        if mode == "recent" and point == 32:
            if fault == "readout":
                raise evaluation.MiniGridEvaluationFailed(
                    dict(readout(point), status="failed", reason="lost")
                )
            if fault == "row":
                captured[0][0]["charged_updates"] = 0
            if fault == "arm":
                captured[0][1]["states"] = deepcopy(captured[0][1]["states"])
            if fault == "budget":
                kw["budget"].updates_completed = 1
            if fault == "source":
                kw["sources"][0].features[0, 0] += 0.125
        return deepcopy(readout(point))

    monkeypatch.setattr(shared, "_train_arm", train)
    monkeypatch.setattr(pilot, "evaluate_public_cue_recurrent_checkpoint", observe)
    result = pilot.run_replay_comparison(tuple(p.episode for p in sources))
    assert ScriptedNative.calls == []
    if fault == "none":
        assert result["status"] == "complete", result.get("reason")
        assert calls[2:] == [("read", m, p) for m in replay.MODES for p in shared.POINTS]
        assert result["final_preservation"] is True
        assert all(c["readout_budget"]["updates_completed"] == 0 for c in result["controls"])
    else:
        assert result["status"] == "incomplete" and result["final_preservation"] is None
        assert all(c["outcomes"] is None for c in result["controls"])
        assert result["controls"][0]["training"]["charged_updates"] == 128
        if fault == "fit":
            assert calls == [("fit", "recent")]
            assert result["controls"][1]["training"]["charged_updates"] == 1
        else:
            assert calls[-1] == ("read", "recent", 32) and len(calls) == 4
            assert len(result["controls"][0]["evaluations"]) == 2
