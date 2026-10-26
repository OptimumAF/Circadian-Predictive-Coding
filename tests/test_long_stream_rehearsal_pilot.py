"""Fixed long-stream data, actual replay plans, and preserved partial work.

One full four-arm run uses explicit fake learners and the real application replay
path. Four tiny native binding cases do not execute the long native pilot.
"""

from dataclasses import dataclass, replace
from hashlib import sha256
import pickle
import struct
from typing import Any

import numpy as np
import pytest

import src.adapters.long_stream_rehearsal_pilot as pilot
import src.app.toy_execution_budget as budget_module
from src.adapters.coupled_attribution_pilot import build_stream
from src.core.coupled_circadian_predictive_coding import CoupledCircadianPredictiveCodingNetwork
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.learner_ports import TrainingDiagnostic
from src.core.matched_backprop import MatchedBackpropNetwork
from test_stream_rehearsal_pilot import FakeSampler


METHODS = ["bp", "pc", "neutral_cpc", "gated_cpc"]
EVENTS = list(range(3, 64, 3))


def forbidden(*args, **kwargs):
    pytest.fail("unexpected native or unselected pilot work")


def content_id(features, targets):
    # Independent byte-format oracle, not the production content-ID function.
    digest = sha256(b"numpy_labeled_replay_row_v1")
    digest.update(struct.pack("<qq", 1, 2))
    digest.update(np.asarray(features, dtype="<f8").reshape(1, 2).tobytes())
    digest.update(struct.pack("<qq", 1, 1))
    digest.update(np.asarray(targets, dtype="<f8").reshape(1, 1).tobytes())
    return digest.hexdigest()


def rank(key):
    return sha256(
        b"numpy_replay_bottom_k_v1" + (53).to_bytes(8, "little") + bytes.fromhex(key)
    ).digest(), key


def reference_plans(training):
    seen: set[str] = set()
    recent: list[str] = []
    plans: list[dict[str, Any]] = []
    after: str | None = None
    for update, batch in enumerate(training, 1):
        for x, y in zip(batch.features, batch.targets, strict=True):
            key = content_id(x, y)
            seen.add(key)
            if key in recent:
                recent.remove(key)
            recent.append(key)
            recent = recent[-8:]
        reservoir = sorted(seen, key=rank)[:8]
        if update % 3:
            continue
        # Every event yields two unique rows, so the persistent requested turn
        # returns to recent. The reservoir resumes strictly after its last scan.
        start = (
            0
            if after is None
            else next(
                (i for i, key in enumerate(reservoir) if rank(key) > rank(after)), len(reservoir)
            )
        )
        rotation = reservoir[start:] + reservoir[:start]
        selected = [recent[-1]]
        after = next(key for key in rotation if key not in selected)
        selected.append(after)
        plans.append(
            {
                "updates": update,
                "selected_ids": selected,
                "retained": sorted(set(recent) | set(reservoir)),
                "recent": sorted(recent),
                "reservoir": sorted(reservoir),
                "overlap": len(set(recent) & set(reservoir)),
                "after": after,
            }
        )
    return plans


@dataclass
class Parameters:
    weights: Any


class FakeModel:
    def __init__(self, *args, **kwargs):
        pass

    def snapshot_parameters(self):
        return Parameters(np.zeros((2, 1)))


class FakeLearner:
    def __init__(self, trace, state=0, *, failure=None, mutate=False):
        self.trace, self.state, self.failure, self.mutate = trace, state, failure, mutate

    def fork(self):
        return FakeLearner(self.trace, self.state, failure=self.failure, mutate=self.mutate)

    def snapshot_state(self):
        return self.state

    def restore_state(self, state):
        self.state = state

    def train_batch(self, features, targets):
        self.trace.append(("train", features.copy(), targets.copy()))
        if len(features) == 1 and self.failure == "before":
            raise RuntimeError("replay failed before return")
        self.state += 1
        if len(features) == 1 and self.failure == "after":
            raise RuntimeError("replay failed after private update")
        if self.mutate:
            features.fill(999)
            targets.fill(999)
        return TrainingDiagnostic("fake_long_stream_count", float(self.state))

    def predict(self, features):
        assert len([item for item in self.trace if item[0] == "train"]) == 106
        self.trace.append(("score", self.state, features.copy()))
        result = np.full((len(features), 1), 0.5 + self.state / 1000)
        if self.mutate:
            features.fill(999)
        return result


def fake_environment(monkeypatch, *, failure=None, mutate=False):
    traces: list[list[Any]] = []
    learners: list[FakeLearner] = []
    initials: list[Any] = []

    def build(method, initial):
        trace: list[Any] = []
        owned = FakeLearner(trace, failure=failure, mutate=mutate)
        traces.append(trace)
        learners.append(owned)
        initials.append(initial)
        return owned

    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    monkeypatch.setattr(pilot, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(budget_module, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", FakeModel)
    monkeypatch.setattr(pilot, "build_learner", build)
    return traces, learners, initials


def test_should_describe_all_fixed_work_and_bounds_without_data_or_model_calls(monkeypatch):
    for name in (
        "build_long_stream",
        "build_stream",
        "build_learner",
        "CoupledPredictiveCodingNetwork",
        "ProcessRssSampler",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    report = pilot.describe_protocol()
    assert report["protocol_id"] == "coherent_long_stream_shared_rehearsal_v1"
    assert report["methods"] == METHODS and report["seeds"] == [23]
    assert report["phase_lengths"] == [16, 32, 16] and report["phase_contexts"] == ["A", "B", "A"]
    assert report["training_batches"] == 64 and report["batch_rows"] == 8
    assert report["checkpoint_updates"] == [0, 16, 48, 64] and report["label_delay"] == 2
    assert report["replay"] == {
        "every_updates": 3,
        "rows_per_event": 2,
        "retention": {"max_examples": 16, "max_bytes": 384},
        "allocation": {
            "recent": {"max_examples": 8, "max_bytes": 192},
            "reservoir": {"max_examples": 8, "max_bytes": 192},
        },
        "seed": 53,
    }
    assert report["planned_work"] == {
        "wake_updates": 256,
        "replay_updates": 168,
        "optimizer_updates": 424,
        "wake_row_exposures": 2048,
        "replay_row_exposures": 168,
        "total_row_exposures": 2216,
        "consolidation_events": 84,
        "final_score_forwards": 32,
        "probes": 0,
        "svd": 0,
    }
    assert report["resources"]["training_updates"] == 424
    assert (
        report["resources"]["replay_bytes"] == 384
        and report["resources"]["storage_bytes"] == 2 * 1024**2
    )
    report["phase_lengths"][0] = 999
    assert pilot.describe_protocol()["phase_lengths"] == [16, 32, 16]


def test_should_generate_exact_reproducible_disjoint_aba_data_without_global_rng_mutation():
    before = pickle.dumps(np.random.get_state())
    training, final = pilot.build_long_stream()
    repeated, repeated_final = pilot.build_long_stream()
    assert pickle.dumps(np.random.get_state()) == before
    assert type(training) is type(final) is tuple and len(training) == 64 and len(final) == 2
    generator = np.random.default_rng(7301)
    ids: set[str] = set()
    for index, (actual, again) in enumerate(zip(training, repeated, strict=True)):
        features = generator.uniform(-1, 1, (8, 2))
        context = "B" if 16 <= index < 48 else "A"
        np.testing.assert_array_equal(actual.features, features)
        np.testing.assert_array_equal(
            actual.targets, (features[:, [int(context == "B")]] >= 0).astype(np.float64)
        )
        assert actual.role == "train" and actual.context == context
        assert actual.row_ids == tuple(f"long-train-{index}-{j}" for j in range(8))
        assert not ids.intersection(actual.row_ids)
        ids.update(actual.row_ids)
        assert pickle.dumps(actual) == pickle.dumps(again)
        assert not np.shares_memory(actual.features, again.features)
        assert not np.shares_memory(actual.targets, again.targets)
    _, expected_final = build_stream("recurring")
    for actual, expected, again in zip(final, expected_final, repeated_final, strict=True):
        assert pickle.dumps(actual) == pickle.dumps(expected) == pickle.dumps(again)
        assert actual.role == "final_test" and actual.features.shape == (64, 2)
        assert not ids.intersection(actual.row_ids)
        assert not any(
            np.shares_memory(actual.features, batch.features)
            or np.shares_memory(actual.targets, batch.targets)
            for batch in training
        )
    training[0].features.fill(999)
    assert not np.any(repeated[0].features == 999)


def test_should_reuse_original_final_tuple_and_discard_short_training(monkeypatch):
    calls = []
    sentinel = (object(), object())

    def original(stream):
        calls.append(stream)
        return (object(),), sentinel

    monkeypatch.setattr(pilot, "build_stream", original)
    training, final = pilot.build_long_stream()
    assert calls == ["recurring"] and len(training) == 64 and final is sentinel


def test_should_manifest_only_training_content_with_independent_ids_and_first_arrival_phase():
    training, _ = pilot.build_long_stream()
    before = pickle.dumps(training)
    rows = pilot._train_manifest(training)
    expected = []
    for update, batch in enumerate(training, 1):
        phase = "first_A" if update <= 16 else "B" if update <= 48 else "return_A"
        for x, y in zip(batch.features, batch.targets, strict=True):
            expected.append(
                {
                    "sample_id": content_id(x, y),
                    "first_arrival_update": update,
                    "phase": phase,
                    "context": batch.context,
                }
            )
    assert rows == expected and len(rows) == 512
    assert pickle.dumps(training) == before
    rows[0]["phase"] = "caller mutation"
    assert pilot._train_manifest(training)[0]["phase"] == "first_A"


@pytest.mark.parametrize("index", [16, 48])
def test_should_refuse_ambiguous_content_reused_across_phases(index):
    training, _ = pilot.build_long_stream()
    copied = list(training)
    copied[index] = replace(
        copied[index], features=training[0].features.copy(), targets=training[0].targets.copy()
    )
    with pytest.raises(ValueError, match="across phases"):
        pilot._train_manifest(tuple(copied))


def test_should_keep_first_arrival_for_same_phase_duplicate_content():
    training, _ = pilot.build_long_stream()
    copied = list(training)
    copied[1] = replace(
        copied[1], features=training[0].features.copy(), targets=training[0].targets.copy()
    )
    rows = pilot._train_manifest(tuple(copied))
    assert len(rows) == 504 and sum(row["first_arrival_update"] == 1 for row in rows) == 8
    assert all(row["first_arrival_update"] != 2 for row in rows)


def test_should_refuse_more_than_the_declared_unique_manifest_capacity():
    training, _ = pilot.build_long_stream()
    extra = replace(
        training[-1],
        features=training[-1].features + 3,
        row_ids=tuple(f"extra-{j}" for j in range(8)),
    )
    with pytest.raises(ValueError, match="512"):
        pilot._train_manifest((*training, extra))


@pytest.mark.parametrize(
    "receipt, committed, expected",
    [(None, 2, []), ({"event_id": "paid"}, 2, ["a" * 64]), ({"event_id": "paid"}, 1, [])],
)
def test_should_distinguish_retained_selected_and_receipted_committed_exposure(
    receipt, committed, expected
):
    old, other = "a" * 64, "b" * 64
    manifest = [{"sample_id": old, "phase": "first_A"}, {"sample_id": other, "phase": "B"}]
    event = {
        "updates": 18,
        "status": "failed",
        "retention": {"sample_ids": (old, other)},
        "selected_ids": [other, old],
        "committed_updates": committed,
        "receipt": receipt,
    }
    report = {"events": [event]}
    before = pickle.dumps(report)
    rows = pilot._exposure(report, manifest)
    assert rows == [
        {
            "updates": 18,
            "status": "failed",
            "retained_first_A_ids": [old],
            "selected_first_A_ids": [old],
            "committed_first_A_ids": expected,
            "retained_first_A": 1,
            "selected_first_A": 1,
            "committed_first_A": len(expected),
        }
    ]
    assert pickle.dumps(report) == before
    rows[0]["retained_first_A_ids"].clear()
    assert event["retention"]["sample_ids"] == (old, other)


def test_should_execute_one_fake_four_arm_long_stream_with_exact_retention_and_exposure_plans(
    monkeypatch,
):
    traces, learners, initials = fake_environment(monkeypatch, mutate=True)
    constructor = pilot.StreamRehearsal
    helpers = []

    def rehearsal(*args, **kwargs):
        helper = constructor(*args, **kwargs)
        helpers.append(helper)
        return helper

    monkeypatch.setattr(pilot, "StreamRehearsal", rehearsal)
    training, final = pilot.build_long_stream()
    before = pickle.dumps((training, final))
    monkeypatch.setattr(pilot, "build_long_stream", lambda: (training, final))
    expected = reference_plans(training)
    report = pilot.run_pilot()
    assert report["status"] == "complete", report["runs"]
    assert [row["method"] for row in report["runs"]] == METHODS
    assert report["global_errors"] == []
    assert report["wake_updates"] == 256 and report["replay_charged_examples"] == 168
    assert (
        report["replay_attempted_updates"]
        == report["replay_returned_updates"]
        == report["replay_committed_updates"]
        == 168
    )
    assert (
        report["total_committed_updates"]
        == report["native_train_calls"]
        == report["known_native_train_calls"]
        == 424
    )
    assert report["row_exposures_charged"] == 2216
    assert report["known_completed_score_forwards"] == report["final_score_forwards"] == 32
    assert (
        report["shared_replay_plans_exact"]
        is report["pc_neutral_exact"]
        is report["retention_bounds_preserved"]
        is True
    )
    assert report["scientific_advantage_verified"] is report["full_R23_complete"] is False
    assert all(value is initials[0] for value in initials) and len(initials) == 4
    assert all(owned.state == 0 for owned in learners)
    assert pickle.dumps((training, final)) == before
    old_ids = {row["sample_id"] for row in report["train_manifest"] if row["phase"] == "first_A"}
    all_ids = {
        content_id(x, y)
        for batch in training
        for x, y in zip(batch.features, batch.targets, strict=True)
    }
    final_recent = {
        content_id(x, y) for x, y in zip(training[-1].features, training[-1].targets, strict=True)
    }
    final_ids = tuple(sorted(final_recent | set(sorted(all_ids, key=rank)[:8])))
    for trace, row, helper in zip(traces, report["runs"], helpers, strict=True):
        assert row["status"] == "complete" and row["source_learner_unchanged"] is True
        assert row["wake_updates"] == 64 and row["replay_charged_examples"] == 42
        assert row["planned_event_updates"] == EVENTS
        assert row["final_retention_observed_updates"] == 64
        assert row["final_retention"] == {
            "sample_ids": final_ids,
            "example_count": len(final_ids),
            "retained_bytes": 24 * len(final_ids),
        }
        snapshot = helper.retention
        assert snapshot.sample_ids == final_ids
        object.__setattr__(snapshot, "sample_ids", ())
        assert helper.retention.sample_ids == final_ids
        events = row["rehearsal"]["events"]
        assert len(events) == 21
        for event, plan in zip(events, expected, strict=True):
            assert event["updates"] == plan["updates"] and event["status"] == "complete"
            assert event["selected_ids"] == plan["selected_ids"]
            assert list(event["retention"]["sample_ids"]) == plan["retained"]
            assert event["retention"]["example_count"] == len(plan["retained"]) <= 16
            assert event["retention"]["retained_bytes"] == 24 * len(plan["retained"]) <= 384
            parts = event["partition_retention"]
            assert list(parts["recent"]["sample_ids"]) == plan["recent"]
            assert list(parts["reservoir"]["sample_ids"]) == plan["reservoir"]
            assert parts["overlap_examples"] == plan["overlap"]
            assert event["rotation_cursor"] == {
                "next_partition": 0,
                "reservoir_after": plan["after"],
            }
            assert (
                event["charged_exposures"]
                == event["attempted_updates"]
                == event["returned_updates"]
                == event["committed_updates"]
                == 2
            )
            assert event["receipt"] is not None
        assert [entry["updates"] for entry in row["B_event_retention_exposure"]] == list(
            range(18, 49, 3)
        )
        for exposure, plan in zip(row["B_event_retention_exposure"], expected[5:16], strict=True):
            retained = [key for key in plan["retained"] if key in old_ids]
            proposed = [key for key in plan["selected_ids"] if key in old_ids]
            assert exposure["retained_first_A_ids"] == retained
            assert exposure["selected_first_A_ids"] == exposure["committed_first_A_ids"] == proposed
            assert exposure["retained_first_A"] == len(retained)
            assert exposure["selected_first_A"] == exposure["committed_first_A"] == len(proposed)
        trains = [entry for entry in trace if entry[0] == "train"]
        assert len(trains) == 106 and sum(len(entry[1]) for entry in trains) == 554
        assert [entry[0] for entry in trace] == ["train"] * 106 + ["score"] * 8
        position = 0
        for update, wake in enumerate(training, 1):
            np.testing.assert_array_equal(trains[position][1], wake.features)
            np.testing.assert_array_equal(trains[position][2], wake.targets)
            position += 1
            if update % 3 == 0:
                for key in expected[update // 3 - 1]["selected_ids"]:
                    assert content_id(trains[position][1], trains[position][2]) == key
                    position += 1
        assert position == 106
        assert [entry[1] for entry in trace if entry[0] == "score"] == [
            0,
            0,
            26,
            26,
            80,
            80,
            106,
            106,
        ]
        result = row["stream_result"]
        assert len(result["applied_updates"]) == 64 and len(result["applied_consolidations"]) == 21
        points = {point["updates"]: point["scores"] for point in result["checkpoints"]}
        assert row["outcomes"] == {
            "new_B_gain": points[0]["B"]["bce"] - points[48]["B"]["bce"],
            "retention_loss": points[48]["A"]["bce"] - points[16]["A"]["bce"],
            "return_gain": points[48]["A"]["bce"] - points[64]["A"]["bce"],
        }


@pytest.mark.parametrize(
    "method, native_type",
    [
        ("bp", MatchedBackpropNetwork),
        ("pc", CoupledPredictiveCodingNetwork),
        ("neutral_cpc", CoupledCircadianPredictiveCodingNetwork),
        ("gated_cpc", CoupledCircadianPredictiveCodingNetwork),
    ],
)
def test_should_bind_original_native_families_without_running_the_long_pilot(method, native_type):
    initial = CoupledPredictiveCodingNetwork(2, (8, 8), 23).snapshot_parameters()
    before = pickle.dumps(initial)
    source: Any = pilot.build_learner(method, initial)
    assert type(source._model) is native_type
    source_before = pickle.dumps(source.snapshot_state())
    child = source.fork()
    child.train_batch(np.zeros((2, 2)), np.full((2, 1), 0.5))
    assert child.predict(np.zeros((1, 2))).shape == (1, 1)
    assert (
        pickle.dumps(initial) == before and pickle.dumps(source.snapshot_state()) == source_before
    )


@pytest.mark.parametrize("failure", ["before", "after"])
def test_should_preserve_permanent_failed_replay_charges_and_not_score_or_retry(
    monkeypatch, failure
):
    traces, learners, _ = fake_environment(monkeypatch, failure=failure)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    assert report["wake_updates"] == 12 and report["replay_charged_examples"] == 4
    assert report["replay_attempted_updates"] == 4
    assert report["replay_returned_updates"] == report["replay_committed_updates"] == 0
    assert report["total_committed_updates"] == 12 and report["row_exposures_charged"] == 100
    assert report["known_native_train_calls"] == 16 and report["native_train_calls"] is None
    assert report["known_completed_score_forwards"] == 0 and report["final_score_forwards"] is None
    assert all(owned.state == 0 for owned in learners)
    for trace, row in zip(traces, report["runs"], strict=True):
        assert row["status"] == "failed" and "replay failed" in row["reason"]
        assert len(trace) == 4 and all(item[0] == "train" for item in trace)
        assert row["source_learner_unchanged"] is True and row["stream_result"] is None
        events = row["rehearsal"]["events"]
        assert events[0]["status"] == "failed" and events[0]["charged_exposures"] == 1
        assert (
            events[0]["attempted_updates"] == 1
            and events[0]["returned_updates"] == events[0]["committed_updates"] == 0
        )
        assert events[0]["receipt"] is None and all(
            event["status"] == "not_run" for event in events[1:]
        )


def test_should_bind_distinct_fresh_wake_and_replay_budgets_to_the_same_original_learner(
    monkeypatch,
):
    _, learners, _ = fake_environment(monkeypatch)
    called = []

    def stream(source, training, final, *, checkpoint_updates, label_delay, budget, rehearsal):
        assert (
            source is learners[0] and rehearsal._learner is source and rehearsal._budget is budget
        )
        assert rehearsal._replay_budget is not budget
        assert (
            budget.budget.max_training_updates == 64 and budget.budget.max_replay_examples is None
        )
        assert rehearsal._replay_budget.budget.max_replay_examples == 42
        assert budget.updates_completed == rehearsal._replay_budget.replay_examples_completed == 0
        assert checkpoint_updates == (0, 16, 48, 64) and label_delay == 2
        called.append(True)
        return {
            "checkpoints": [
                {"updates": count, "scores": {context: {"bce": 0.5} for context in ("A", "B")}}
                for count in checkpoint_updates
            ]
        }

    monkeypatch.setattr(pilot, "run_attribution_stream", stream)
    training, final = pilot.build_long_stream()
    row = pilot._rows()[0]
    initial: Any = Parameters(np.zeros((2, 1)))
    sampler: Any = FakeSampler()
    pilot._run_one(row, initial, training, final, 60.0, sampler)
    assert called == [True] and row["source_learner_unchanged"] is True
    assert row["wake_updates"] == row["replay_charged_examples"] == 0


@pytest.mark.parametrize("phase", ["sampler", "data", "initial"])
def test_should_preserve_all_planned_rows_on_global_setup_failure(monkeypatch, phase):
    fake_environment(monkeypatch)
    monkeypatch.setattr(pilot, "build_learner", forbidden)

    def broken(*args, **kwargs):
        raise RuntimeError(f"declared {phase} failure")

    name = {
        "sampler": "ProcessRssSampler",
        "data": "build_long_stream",
        "initial": "CoupledPredictiveCodingNetwork",
    }[phase]
    monkeypatch.setattr(pilot, name, broken)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and len(report["runs"]) == 4
    assert all(
        row["status"] == "not_run" and row["planned_event_updates"] == EVENTS
        for row in report["runs"]
    )
    assert report["wake_updates"] == report["replay_charged_examples"] == 0
    assert report["native_train_calls"] is report["final_score_forwards"] is None
    assert any(f"declared {phase} failure" in error for error in report["global_errors"])


def test_should_leave_later_rows_unreached_when_the_original_global_deadline_is_spent(monkeypatch):
    fake_environment(monkeypatch)
    now = [0.0]
    monkeypatch.setattr(pilot, "monotonic", lambda: now[0])
    calls = []

    def run(row, *args):
        calls.append(row["method"])
        row.update(status="failed", reason="declared bounded first-row failure")
        now[0] = 60.0

    monkeypatch.setattr(pilot, "_run_one", run)
    report = pilot.run_pilot()
    assert calls == ["bp"]
    assert [row["status"] for row in report["runs"]] == ["failed", "not_run", "not_run", "not_run"]
    assert all(row["planned_event_updates"] == EVENTS for row in report["runs"])
    assert report["status"] == "incomplete" and report["total_seconds"] == 60.0


@pytest.mark.parametrize("action", ["describe_protocol", "build_long_stream", "run_pilot"])
@pytest.mark.parametrize("keyword", ["recent_only", "extended"])
def test_should_refuse_unconfigured_knobs_before_any_work(monkeypatch, action, keyword):
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(pilot, "build_stream", forbidden)
    with pytest.raises(TypeError):
        getattr(pilot, action)(**{keyword: True})
