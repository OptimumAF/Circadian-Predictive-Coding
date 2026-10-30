"""Pure/mocked protocol and authority checks; no teacher, fit or rollout work."""

from dataclasses import replace
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import compare_joint_context_retention as comparison
from src.app.coupled_attribution import AttributionBatch
from src.core.circadian_predictive_coding import ReplayRetentionSnapshot


@pytest.fixture(autouse=True)
def refuse_real_work(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("ZERO-MODEL engineering allowance: real work refused")

    monkeypatch.setattr(comparison, "training_cases", refuse)
    monkeypatch.setattr(comparison, "run_workflow_episode", refuse)
    monkeypatch.setattr(np.linalg, "lstsq", refuse)


def batch(role="train"):
    x = np.zeros((3, 9))
    x[:, 4] = 1
    return AttributionBatch(
        role, "v1-invoice", ("a", "b", "c"), x, np.array([[1.0], [2.0], [3.0]]), num_classes=6
    )


def test_should_preview_fixed_protocol_without_opening_training_or_models(monkeypatch, capsys):
    monkeypatch.setattr(comparison, "prepare_training", lambda: pytest.fail("TRAIN opened"))
    assert comparison.main([]) == 0
    p = json.loads(capsys.readouterr().out)
    assert p["points"] == [0, 4, 6, 10]
    assert p["controls"] == ["joint", "current_only"]
    assert p["training_order"] == p["tasks"] + p["tasks"][:4]
    p["points"].append(99)
    assert comparison.protocol()["points"] == [0, 4, 6, 10]


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final"])
def test_should_refuse_non_train_episode_without_converting_role(role):
    with pytest.raises(ValueError, match="original TRAIN"):
        comparison.episode_packet(batch(role), "one")


def test_should_own_complete_canonical_train_episode_and_refuse_fractional_action():
    original = batch()
    packet = comparison.episode_packet(original, "one")
    assert packet.role == packet.episode.role == "train"
    assert packet.row_ids == ("one:step:0", "one:step:1", "one:step:2")
    assert packet.episode.transitions[-1].terminal
    assert not np.shares_memory(packet.features, original.features)
    assert not np.shares_memory(packet.targets, original.targets)
    with pytest.raises(ValueError):
        comparison.episode_packet(replace(original, targets=np.full((3, 1), 1.5)), "bad")


def test_should_prepare_only_original_train_cases_in_fixed_task_order(monkeypatch):
    calls, selections = [], []

    def teachers(**kwargs):
        calls.append(kwargs)
        return tuple((kwargs["policy_version"], i) for i in range(12))

    def build(cases, spec, context, *, allow_subsets):
        selections.append((cases, context, allow_subsets))
        return replace(batch(), context=context)

    monkeypatch.setattr(comparison, "training_cases", teachers)
    monkeypatch.setattr(comparison, "build_decision_training_batch", build)
    _, training = comparison.prepare_training()
    assert calls == [dict(policy_version=v, include_policy_context=True) for v in ("v1", "v2")]
    assert len(selections) == 6 and len(training) == 10
    for (cases, context, subsets), (version, concept) in zip(
        selections, comparison.TASKS, strict=True
    ):
        offset = 3 * comparison.VOCABULARY.index(concept)
        assert cases == tuple((version, offset + i) for i in range(3))
        assert context == f"{version}-{concept}" and subsets
    assert all(t.role == t.episode.role == "train" for t in training)
    assert len({t.episode.episode_id for t in training}) == 10


def rows(successes):
    return [dict(tasks=[dict(success=s) for s in phase]) for phase in successes]


def test_should_report_negative_and_null_acquisition_retention_recovery_separately():
    points = rows(
        (
            (False,) * 6,
            (True, True, False, True, False, True),
            (False, True, False, True, True, True),
            (True, True, False, True, False, True),
        )
    )
    result = comparison.summarize(points)
    assert result["old_successes"] == [0, 3, 2, 3]
    assert result["new_successes"] == [0, 1, 2, 1]
    assert result["per_task"][0]["lost_after_new"]
    assert result["per_task"][0]["recovered_at_return"]
    assert not result["per_task"][1]["recovered_at_return"]
    assert result["per_task"][4]["acquired_at_new"]
    assert not result["per_task"][5]["acquired_at_new"]
    with pytest.raises(ValueError):
        comparison.summarize(points[:-1])


@pytest.mark.parametrize("fault", [None, "incomplete", "mutate"])
def test_should_finish_both_training_arms_before_any_readout(monkeypatch, fault):
    training, calls = (batch(),), []
    monkeypatch.setattr(comparison, "prepare_training", lambda: ("spec", training))

    def train(control, data, sampler, progress):
        calls.append(("train", control))
        progress.update(complete=fault != "incomplete")
        if fault == "mutate":
            data[0].features[0, 0] += 1
        return tuple(range(4))

    def evaluate(control, states, spec, progress):
        assert calls[:2] == [("train", c) for c in comparison.CONTROLS]
        calls.append(("read", control))
        return rows(((False,) * 6,) * 4)

    monkeypatch.setattr(comparison, "train_arm", train)
    monkeypatch.setattr(comparison, "evaluate_checkpoints", evaluate)
    report: dict[str, Any] = {}
    if fault:
        with pytest.raises(ValueError):
            comparison.run_comparison(report, None)
        assert all(c[0] == "train" for c in calls)
    else:
        comparison.run_comparison(report, None)
        assert report["complete"] and calls == [("train", c) for c in comparison.CONTROLS] + [
            ("read", c) for c in comparison.CONTROLS
        ]


def test_should_preserve_failed_case_and_refuse_existing_output_before_work(tmp_path, monkeypatch):
    calls = []

    def fail(report, sampler):
        calls.append(1)
        raise ValueError("deliberate pure admission refusal")

    monkeypatch.setattr(comparison, "run_comparison", fail)
    path = tmp_path / "actual.json"
    assert comparison.main(["--output", str(path)]) == 1
    saved = path.read_bytes()
    result = json.loads(saved)
    assert not result["complete"] and result["disposition"] == "FAILED/SPENT"
    assert all(v == 0 for v in result["work"].values())
    with pytest.raises(FileExistsError):
        comparison.main(["--output", str(path)])
    assert path.read_bytes() == saved and calls == [1]


def test_should_stop_before_native_call_and_keep_exceeded_attempt_charge():
    counts = dict.fromkeys(comparison.LIMITS, 0)
    counts["fit_attempts"] = 36
    profile = comparison.observe_work(counts, comparison.monotonic(), SimpleNamespace(peak_bytes=1))
    frame = SimpleNamespace(
        f_code=SimpleNamespace(
            co_name="fit_categorical_linear", co_filename="src/core/categorical_linear_fit.py"
        ),
        f_locals=dict(features=np.zeros((3, 9))),
    )
    with pytest.raises(comparison._StudySpent, match="SPENT"):
        profile(frame, "call", None)
    assert counts["fit_attempts"] == 37 and counts["lstsq_calls"] == 0


def test_should_capture_initial_state_without_advancing_stream_to_zero(monkeypatch):
    advanced: list[int] = []
    source = SimpleNamespace(snapshot_state=lambda: "original")
    wake = SimpleNamespace(updates_completed=10)
    replay = SimpleNamespace(replay_examples_completed=66)
    helper = SimpleNamespace(report={}, retention=ReplayRetentionSnapshot((), 0, 0))
    stream = SimpleNamespace(
        run_until=advanced.append,
        candidate_snapshot=lambda: SimpleNamespace(completed_updates=18, training_examples=96),
        finish=lambda: {},
        _runtime=SimpleNamespace(
            actor=SimpleNamespace(snapshot_state=lambda: SimpleNamespace(state="original"))
        ),
    )
    monkeypatch.setattr(comparison, "ContextLinearFitLearner", lambda policy: source)
    monkeypatch.setattr(comparison, "_sessions", lambda sampler: (wake, replay))
    monkeypatch.setattr(comparison, "StreamRehearsal", lambda *args, **kwargs: helper)
    monkeypatch.setattr(comparison, "AttributionStream", lambda *args, **kwargs: stream)
    monkeypatch.setattr(comparison, "_model_bytes", lambda state: 0)
    progress: dict[str, Any] = {}
    assert len(comparison.train_arm("joint", (), None, progress)) == 4
    assert advanced == [4, 6, 10] and progress["complete"]
    with pytest.raises(ValueError, match="control"):
        comparison.train_arm("unknown", (), None, {})


def test_should_not_count_new_task_loss_and_reappearance_as_old_task_recovery():
    points = rows(
        (
            (False,) * 6,
            (False, False, False, False, True, True),
            (True, False, False, False, False, True),
            (True, False, False, False, True, True),
        )
    )
    result = comparison.summarize(points)
    assert not result["per_task"][0]["acquired_at_new"]
    assert not result["per_task"][4]["lost_after_new"]
    assert not result["per_task"][4]["recovered_at_return"]


def test_should_admit_real_budget_helper_and_stream_without_any_fit_or_readout():
    # Why: mocks missed the runtime distinction between an absent cap and zero.
    with comparison.ProcessRssSampler(interval_seconds=0.02) as sampler:
        wake, replay = comparison._sessions(sampler)
        assert wake.budget.max_training_updates == 10
        assert wake.budget.max_replay_examples is None
        assert replay.budget.max_training_updates is None
        assert replay.budget.max_replay_examples == 66
        source = comparison.ContextLinearFitLearner(comparison.ContextLinearFitPolicy(9, 6, (4, 5)))
        before = source.snapshot_state()
        packets = tuple(comparison.episode_packet(batch(), f"admission-{i}") for i in range(10))
        helper = comparison.StreamRehearsal(
            source,
            wake,
            comparison.RecentStreamReplayPolicy(1, 12, comparison.ReplayRetentionBudget(30, 65536)),
            replay_budget=replay,
            total_updates=10,
            input_dim=9,
            episode_schema=comparison.EpisodeReplaySchema(9, 3, 6),
            joint_context=comparison.JointContextReplayPolicy(9, 6, (4, 5), "joint"),
        )
        stream = comparison.AttributionStream(
            source,
            packets,
            (),
            checkpoint_updates=comparison.POINTS,
            label_delay=1,
            budget=wake,
            rehearsal=helper,
            training_only=True,
        )
        assert stream.candidate_snapshot() == before == source.snapshot_state()
        assert all(h.model is None for h in before.heads)
        assert wake.updates_completed == replay.replay_examples_completed == 0
