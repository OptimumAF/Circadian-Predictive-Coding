"""Saved FIFO36 F16/A16 orchestration with scripted native ports.

The decoded snapshots are genuine saved inputs, while collected F packets,
updates and probabilities are deliberately scripted. No teacher, physics,
native count update or native prediction supplies these behavior checks.
"""

import builtins
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
import random

import numpy as np
import pytest

from src.adapters import minigrid_action_memory_probes as pilot
from src.adapters.invariant_action_memory_learner import InvariantActionMemoryLearner
from src.adapters.minigrid_saved_fifo_inputs import decode_saved_fifo_inputs
from src.app import toy_execution_budget as budget_module
from src.core.action_memory import ActionMemoryEntry
from src.core.learner_ports import TrainingDiagnostic
from src.shared.process_memory import ProcessRssSegment


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "artifacts/runs/minigrid-recurring-fifo-20261017/native.json"
)


def forbidden(*args, **kwargs):
    pytest.fail("scripted saved FIFO assay reached an unselected resource")


@pytest.fixture
def assay(monkeypatch):
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    inputs = decode_saved_fifo_inputs(json.loads(SOURCE.read_text(encoding="utf-8")))
    work = dict(
        inputs=inputs,
        helpers=[],
        children=[],
        attempts=[],
        trains=[],
        predictions=[],
        collections=[],
        budgets=[],
        clock=1.0,
    )
    native_helper = pilot.run_fresh_task_probe
    native_fork = InvariantActionMemoryLearner.fork
    native_import = builtins.__import__
    canonical_key = inputs.snapshots[-1].native_state.entries[0].features

    class Sampler:
        start_bytes = 100

        def __enter__(self):
            work["sampler"] = self
            return self

        def __exit__(self, *args):
            callback = work.get("on_close")
            if callback is not None:
                callback()
            return False

        def sample(self):
            callback = work.get("on_sample")
            if callback is not None:
                callback()
            return 100

        def snapshot(self):
            return ProcessRssSegment(1, 100, work.get("rss", 100), 1, 0.01)

    def collector(check_boundary, reports, *, distinct_probe=False):
        assert distinct_probe is True
        check_boundary()
        originals = {}
        for role in ("probe_train", "probe_test"):
            for original, cue in zip(inputs.episodes[:2], ("blue", "red"), strict=True):
                originals[role, cue] = replace(
                    original,
                    role=role,
                    context="F-" + cue,
                    episode_id=f"minigrid-probe-source-{role}-{cue}",
                )
                reports.append(
                    dict(
                        status="complete",
                        role=role,
                        context="F-" + cue,
                        cue=cue,
                        environment_steps_attempted=9,
                        environment_steps=9,
                        failure_work_may_be_unobserved=False,
                    )
                )
        training = tuple(
            pilot.reserved._packet(
                replace(
                    originals["probe_train", ("blue", "red")[i % 2]],
                    episode_id=f"minigrid-probe-source-probe_train-{('blue', 'red')[i % 2]}:copy:{i}",
                )
            )
            for i in range(16)
        )
        evaluation = tuple(
            pilot.reserved._packet(originals["probe_test", cue]) for cue in ("blue", "red")
        )
        work["collections"].append((training, evaluation))
        callback = work.get("on_collect")
        if callback is not None:
            callback(training, evaluation)
        check_boundary()
        return training, evaluation

    def helper(source, state, training, evaluation, **options):
        entry = dict(
            source=source, state=state, training=training, evaluation=evaluation, **options
        )
        work["helpers"].append(entry)
        work["budgets"].extend((options["budget"], options["return_budget"]))
        callback = work.get("before_helper")
        if callback is not None:
            callback(entry)
        raw = native_helper(source, state, training, evaluation, **options)
        entry["report"] = raw
        callback = work.get("after_helper")
        if callback is not None:
            callback(entry)
        return raw

    def fork(owner):
        child = native_fork(owner)
        work["children"].append(child)
        return child

    def train(owner, features, targets):
        before = owner.snapshot_state()
        work["attempts"].append((owner, before, features.copy(), targets.copy()))
        if work.get("fail_attempt") == len(work["attempts"]):
            raise RuntimeError("scripted update failed before commit")
        state = before.native_state
        entries = list(state.entries)
        if entries:
            first = entries[0]
            counts = list(first.counts)
            counts[0] += 9
            entries[0] = replace(first, counts=tuple(counts))
        else:
            entries.append(ActionMemoryEntry(canonical_key, (9, 0, 0, 0, 0, 0, 0)))
        after = replace(
            before,
            native_state=replace(
                state,
                entries=tuple(entries),
                completed_updates=state.completed_updates + 1,
                training_examples=state.training_examples + 9,
            ),
        )
        owner.restore_state(after)
        work["trains"].append((owner, before, after, features.copy(), targets.copy()))
        callback = work.get("on_train")
        if callback is not None:
            callback(owner)
        return TrainingDiagnostic("scripted_count_clock_only", 0.0)

    def predict(owner, features):
        work["predictions"].append((owner, owner.snapshot_state(), features.copy()))
        callback = work.get("on_predict")
        if callback is not None:
            callback(owner)
        return np.full((len(features), 7), 1 / 7, dtype=np.float64)

    def no_torch(name, *args, **options):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("saved invariant FIFO assay imported Torch")
        return native_import(name, *args, **options)

    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budget_module, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "monotonic", lambda: work["clock"])
    monkeypatch.setattr(pilot.reserved, "_collect_probe_sources", collector)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", helper)
    monkeypatch.setattr(InvariantActionMemoryLearner, "fork", fork)
    monkeypatch.setattr(InvariantActionMemoryLearner, "train_batch", train)
    monkeypatch.setattr(InvariantActionMemoryLearner, "predict", predict)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    yield work
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)


def test_should_describe_fixed_saved_fifo_assay_without_loading_or_resource_work(monkeypatch):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(pilot.reserved, "_collect_probe_sources", forbidden)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", forbidden)
    description = pilot.describe_saved_fifo_return_protocol()
    assert description == pilot.describe_saved_fifo_return_protocol()
    assert "36" in str(description) and "F" in str(description)
    assert "128" in str(description) and "87264" in str(description)


def test_should_use_two_owned_forks_and_four_original_phase_budgets(assay):
    before = pickle.dumps(assay["inputs"], protocol=5)
    result = pilot.run_saved_fifo_return_pilot(assay["inputs"])
    assert result["status"] == "complete", result["reason"]
    assert len(result["cohorts"]) == len(assay["helpers"]) == len(assay["children"]) == 2
    assert len(assay["collections"]) == 1
    assert len(assay["trains"]) == len(assay["attempts"]) == 64
    assert len(assay["predictions"]) == 64
    assert len(assay["budgets"]) == len({id(b) for b in assay["budgets"]}) == 4
    assert [b.updates_completed for b in assay["budgets"]] == [16, 16, 16, 16]
    assert all(b.budget.max_training_updates == 16 for b in assay["budgets"])
    assert all(b.started_at == 1.0 and b.replay_examples_completed == 0 for b in assay["budgets"])
    assert [r["source_updates"] for r in result["cohorts"]] == [0, 128]
    assert [r["native_training_examples_before"] for r in result["cohorts"]] == [0, 1152]
    for row, helper, child, origin in zip(
        result["cohorts"], assay["helpers"], assay["children"], (0, 128), strict=True
    ):
        assert row["status"] == "complete" and row["source_unchanged"] is True
        assert row["budget_binding_preserved"] is True
        assert row["phase_charges"] == {"F": 16, "A_return": 16}
        assert helper["source"] is not child
        assert helper["source"].snapshot_state() == helper["state"]
        assert child.snapshot_state().completed_updates == origin + 32
        assert child.snapshot_state().training_examples == 9 * (origin + 32)
        assert row["end_snapshot"] == asdict(child.snapshot_state())
        raw = row["report"]
        assert raw["completed_updates"] == raw["return_phase"]["completed_updates"] == 16
        assert raw["score_forward_calls_completed"] == 8
        assert raw["return_phase"]["score_forward_calls_completed"] == 8
        assert [p["updates"] for p in raw["checkpoints"]] == [0, 4, 8, 16]
        assert [p["updates"] for p in raw["return_phase"]["checkpoints"]] == [0, 4, 8, 16]
        assert raw["training_readouts"]["score_forward_calls_completed"] == 8
        assert raw["return_phase"]["training_readouts"]["score_forward_calls_completed"] == 8
    assert pickle.dumps(assay["inputs"], protocol=5) == before
    assert result["scientific_advantage_verified"] is False


def test_should_keep_F_probe_roles_and_known_A_TRAIN_roles_disjoint(assay):
    result = pilot.run_saved_fifo_return_pilot(assay["inputs"])
    assert result["status"] == "complete", result["reason"]
    for helper in assay["helpers"]:
        training, evaluation = helper["training"], helper["evaluation"]
        monitors, returning = helper["training_readouts"], helper["return_training"]
        assert [p.context for p in training] == ["F-blue", "F-red"] * 8
        assert all(p.role == p.episode.role == "probe_train" for p in training)
        assert all(p.role == p.episode.role == "probe_test" for p in evaluation)
        assert all(p.role == p.episode.role == "train" for p in (*monitors, *returning))
        assert len(monitors) == len(returning) == 2
        packets = (*training, *evaluation, *monitors, *returning)
        assert all(p.features.shape == (9, 296) and p.num_classes == 7 for p in packets)
        ids = [row for packet in packets for row in packet.row_ids]
        assert len(ids) == len(set(ids))
        arrays = [array for packet in packets for array in (packet.features, packet.targets)]
        assert all(
            not np.shares_memory(left, right)
            for i, left in enumerate(arrays)
            for right in arrays[i + 1 :]
        )
        for monitor, template in zip(monitors, returning, strict=True):
            assert np.array_equal(monitor.features, template.features)
            assert np.array_equal(monitor.targets, template.targets)


@pytest.mark.parametrize("seconds", [0, -1, 120.01, float("inf"), float("nan"), True, "120"])
def test_should_refuse_invalid_body_allowance_before_resource_or_collection_work(assay, seconds):
    with pytest.raises(ValueError):
        pilot.run_saved_fifo_return_pilot(assay["inputs"], max_seconds=seconds)
    assert assay["helpers"] == assay["collections"] == []


@pytest.mark.parametrize("target", ["clock", "policy", "middle", "A_role", "origin"])
def test_should_refuse_corrupted_full_saved_inputs_before_new_work(assay, target):
    inputs = assay["inputs"]
    if target == "clock":
        object.__setattr__(inputs.snapshots[-1].native_state, "completed_updates", 127)
    elif target == "policy":
        object.__setattr__(inputs.snapshots[-1].policy.native_policy, "max_entries", 64)
    elif target == "middle":
        object.__setattr__(inputs.snapshots[2].native_state, "retired_examples", True)
    elif target == "A_role":
        object.__setattr__(inputs.episodes[0], "role", "final")
    else:
        object.__setattr__(inputs, "original_training_updates", 129)
    with pytest.raises(ValueError):
        pilot.run_saved_fifo_return_pilot(inputs)
    assert assay["helpers"] == assay["collections"] == []


@pytest.mark.parametrize("phase,expected", [("F", 2), ("A", 18)])
def test_should_keep_paid_prefix_and_stop_before_later_cohort_on_update_failure(
    assay, phase, expected
):
    assay["fail_attempt"] = 3 if phase == "F" else 19
    result = pilot.run_saved_fifo_return_pilot(assay["inputs"])
    assert result["status"] == "incomplete"
    assert len(assay["trains"]) == expected
    assert len(assay["helpers"]) == len(assay["children"]) == 1
    first, second = result["cohorts"]
    assert first["status"] == "failed" and second["status"] == "not_run"
    assert "scripted update failed" in str(first)
    assert sum(b.updates_completed for b in assay["budgets"]) == expected


@pytest.mark.parametrize(
    "target",
    ["source", "input", "data", "limit", "clock", "last_clock", "child", "rng", "numpy_rng"],
)
def test_should_refuse_mutation_after_paid_F_update_and_preserve_observed_charge(assay, target):
    def mutate(owner):
        if len(assay["trains"]) != 1:
            return
        helper = assay["helpers"][0]
        if target == "source":
            helper["source"].restore_state(owner.snapshot_state())
        elif target == "input":
            object.__setattr__(assay["inputs"], "origin_reason", "rewritten past")
        elif target == "data":
            helper["training"][0].features[0, 0] += 0.125
        elif target == "limit":
            object.__setattr__(helper["budget"].budget, "max_training_updates", 15)
        elif target == "clock":
            helper["budget"].clock = lambda: 1.0
        elif target == "last_clock":
            helper["budget"].last_clock = 2.0
        elif target == "child":
            helper["source"].child = deepcopy(owner)
        elif target == "rng":
            random.random()
        else:
            np.random.random()

    assay["on_train"] = mutate
    result = pilot.run_saved_fifo_return_pilot(assay["inputs"])
    assert result["status"] == "incomplete"
    assert len(assay["trains"]) == 1 and len(assay["helpers"]) == 1
    assert result["cohorts"][1]["status"] == "not_run"
    assert assay["budgets"][1].updates_completed == 0


@pytest.mark.parametrize("target", ["F_counter", "F_clock", "A_limit", "A_clock"])
def test_should_preserve_both_original_budget_bindings_during_paid_A_return(assay, target):
    def mutate(owner):
        if len(assay["trains"]) != 17:
            return
        helper = assay["helpers"][0]
        if target == "F_counter":
            helper["budget"].updates_completed = 0
        elif target == "F_clock":
            helper["budget"].clock = lambda: 1.0
        elif target == "A_limit":
            object.__setattr__(helper["return_budget"].budget, "max_training_updates", 17)
        else:
            helper["return_budget"].last_clock = 2.0

    assay["on_train"] = mutate
    result = pilot.run_saved_fifo_return_pilot(assay["inputs"])
    assert result["status"] == "incomplete"
    assert len(assay["trains"]) == 17 and len(assay["helpers"]) == 1
    assert result["cohorts"][0]["status"] == "failed"
    assert result["cohorts"][1]["status"] == "not_run"


def test_should_refuse_preceding_helper_report_corruption_before_more_work(assay):
    def mutate(entry):
        if len(assay["helpers"]) == 2:
            assay["helpers"][0]["report"]["completed_updates"] = 15

    assay["before_helper"] = mutate
    result = pilot.run_saved_fifo_return_pilot(assay["inputs"])
    assert result["status"] == "incomplete"
    assert len(assay["trains"]) == 32
    assert result["cohorts"][0]["report"]["completed_updates"] == 16
    assert result["cohorts"][1]["status"] == "failed"


@pytest.mark.parametrize("target", ["input", "budget", "wall", "rss"])
def test_should_recheck_closing_resource_callback_before_complete_publication(assay, target):
    def mutate():
        if target == "input":
            object.__setattr__(assay["inputs"], "origin_status", "complete")
        elif target == "budget":
            assay["budgets"][0].updates_completed = 0
        elif target == "wall":
            assay["clock"] = 122.0
        else:
            assay["rss"] = pilot.shared.RSS_LIMIT + 1

    assay["on_close"] = mutate
    result = pilot.run_saved_fifo_return_pilot(assay["inputs"])
    assert result["status"] == "incomplete"
    assert len(assay["trains"]) == 64


def test_should_keep_missing_helper_return_unknown_with_original_observed_charge(
    assay, monkeypatch
):
    def missing(source, state, training, evaluation, **options):
        child = source.fork()
        options["check_boundary"]()
        child.train_batch(training[0].features, training[0].targets)
        options["budget"].record_update()
        options["check_boundary"]()
        raise RuntimeError("scripted helper lost its publication")

    monkeypatch.setattr(pilot, "run_fresh_task_probe", missing)
    result = pilot.run_saved_fifo_return_pilot(assay["inputs"])
    first, second = result["cohorts"]
    assert result["status"] == "incomplete" and second["status"] == "not_run"
    assert first["unreported_native_work_may_be_unobserved"] is True
    assert first["charged_updates"] == 1
    assert first["report"] is None
    assert result["work"]["probe_totals_are_unknown"] is True


@pytest.mark.parametrize("target", ["start", "last", "config_bool", "config_float", "clock_port"])
def test_should_refuse_same_valued_return_budget_type_or_port_replacement(assay, target):
    class EqualClock:
        def __call__(self):
            return 1.0

        def __eq__(self, other):
            return True

    def mutate(entry):
        budget = entry["return_budget"]
        if target == "start":
            budget.started_at = True
        elif target == "last":
            budget.last_clock = True
        elif target == "config_bool":
            object.__setattr__(budget.budget, "max_replay_examples", False)
        elif target == "config_float":
            object.__setattr__(budget.budget, "max_training_updates", 16.0)
        else:
            budget.clock = EqualClock()

    assay["before_helper"] = mutate
    result = pilot.run_saved_fifo_return_pilot(assay["inputs"])
    assert result["status"] == "incomplete"
    assert assay["trains"] == []
    assert result["cohorts"][1]["status"] == "not_run"


@pytest.mark.parametrize("target", ["child", "fork_zero", "fork_bool"])
def test_should_recheck_captured_fork_after_last_source_read_callback(assay, target):
    def install(entry):
        source = entry["source"]
        original_snapshot = source.snapshot_state
        observed_with_child = 0

        def snapshot():
            nonlocal observed_with_child
            value = original_snapshot()
            if source.child is not None:
                observed_with_child += 1
                if observed_with_child == 2:
                    if target == "child":
                        source.child = deepcopy(source.child)
                    else:
                        source.fork_calls = 0 if target == "fork_zero" else True
            return value

        source.snapshot_state = snapshot

    assay["before_helper"] = install
    result = pilot.run_saved_fifo_return_pilot(assay["inputs"])
    assert result["status"] == "incomplete"
    assert len(assay["helpers"]) == 1 and len(assay["children"]) == 1
    assert assay["trains"] == []
    assert result["cohorts"][1]["status"] == "not_run"


def test_should_refuse_equal_valued_preceding_helper_container_replacement(assay):
    def mutate(entry):
        if len(assay["helpers"]) == 2:
            raw = assay["helpers"][0]["report"]
            raw["checkpoints"] = deepcopy(raw["checkpoints"])

    assay["before_helper"] = mutate
    result = pilot.run_saved_fifo_return_pilot(assay["inputs"])
    assert result["status"] == "incomplete"
    assert len(assay["trains"]) == 32
    assert result["cohorts"][0]["report"]["completed_updates"] == 16
    assert result["cohorts"][1]["status"] == "failed"


@pytest.mark.parametrize(
    "field", ["score_slots", "game_slots", "completed_games", "censored_games"]
)
def test_should_refuse_corrupted_saved_cell_metadata_before_collection(assay, field):
    object.__setattr__(assay["inputs"], field, ((128, "final", "blue"),))
    with pytest.raises(ValueError):
        pilot.run_saved_fifo_return_pilot(assay["inputs"])
    assert assay["collections"] == assay["helpers"] == []


def test_should_refuse_float_relabeling_of_original_censored_native_clock(assay):
    object.__setattr__(assay["inputs"], "censored_games", ((64.0, "B", "blue"),))
    with pytest.raises(ValueError):
        pilot.run_saved_fifo_return_pilot(assay["inputs"])
    assert assay["helpers"] == assay["collections"] == []
