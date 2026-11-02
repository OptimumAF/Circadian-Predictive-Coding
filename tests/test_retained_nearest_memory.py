"""New literal bounded-memory cohort; no dataset, SDK, optimizer or model RNG."""

from dataclasses import replace
import ctypes
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from src.adapters import continuous_stream_sequence as stream
from src.adapters.numpy_nearest_continuous_bc import (
    FixedInputNearestContinuousBCPolicy,
    NumpyNearestContinuousBCLearner,
    _snapshot as legacy_snapshot,
)
from src.adapters.numpy_retained_nearest_bc import (
    NumpyRetainedNearestBCLearner,
    retained_costs,
)
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.nearest_retention import select_retained_rows
from src.core.replay_retention import ReplayRetentionPolicy

SPEC = ContinuousActionSpec(2, ("unit",), (-1.0,), (1.0,))
FIFO = ReplayRetentionPolicy("recent_fifo")
RESERVOIR = ReplayRetentionPolicy("seeded_reservoir", 17)
WORK = dict(
    initializers=0,
    forks=0,
    updates=0,
    incoming_rows=0,
    predictions=0,
    query_rows=0,
    distance_candidates=0,
)
OBSERVED: dict[str, Any] = {}


def _process_peak_bytes() -> int:
    if sys.platform != "win32":
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (
            1 if sys.platform == "darwin" else 1024
        )

    class MemoryCounters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak",
                "working",
                "peak_paged",
                "paged",
                "peak_nonpaged",
                "nonpaged",
                "pagefile",
                "peak_pagefile",
            )
        ]

    process = ctypes.windll.kernel32.GetCurrentProcess
    process.restype = ctypes.c_void_p
    read = ctypes.windll.psapi.GetProcessMemoryInfo
    read.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryCounters), ctypes.c_ulong]
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    assert read(process(), ctypes.byref(counters), ctypes.sizeof(counters))
    return int(counters.peak)


@pytest.fixture(scope="module", autouse=True)
def finite_work():
    prior = json.loads(os.environ.get("RETAINED_MEMORY_PRIOR_WORK", "{}"))
    limits = dict(
        initializers=160,
        updates=400,
        incoming_rows=2048,
        predictions=180,
        query_rows=1024,
        distance_candidates=32768,
    )

    def charge(name, amount=1):
        WORK[name] += amount
        assert WORK[name] + prior.get(name, 0) <= limits.get(name, 160), f"literal work cap: {name}"

    def wrap(method, kind):
        def metered(owner, *args, **kwargs):
            charge(kind)
            if kind == "updates":
                charge("incoming_rows", len(args[0]))
            if kind == "predictions":
                rows = len(args[0])
                charge("query_rows", rows)
                charge("distance_candidates", rows * len(owner._state.prototype_features))
            return method(owner, *args, **kwargs)

        return metered

    with pytest.MonkeyPatch.context() as patch:
        for cls in (NumpyNearestContinuousBCLearner, NumpyRetainedNearestBCLearner):
            for method, kind in (
                ("__init__", "initializers"),
                ("train_batch", "updates"),
                ("fork", "forks"),
            ):
                patch.setattr(cls, method, wrap(getattr(cls, method), kind))
        patch.setattr(
            NumpyNearestContinuousBCLearner,
            "predict_with_work",
            wrap(NumpyNearestContinuousBCLearner.predict_with_work, "predictions"),
        )
        patch.setattr(np.random, "default_rng", lambda *a, **k: pytest.fail("model RNG forbidden"))
        yield
    receipt = os.environ.get("RETAINED_MEMORY_RECEIPT")

    peak_bytes = _process_peak_bytes()
    aggregate = {name: value + prior.get(name, 0) for name, value in WORK.items()}
    if receipt:
        with Path(receipt).open("x", encoding="utf8") as handle:
            json.dump(
                dict(
                    work=WORK,
                    aggregate=aggregate,
                    observed=OBSERVED,
                    actual_pytest_peak_bytes=peak_bytes,
                ),
                handle,
                indent=2,
                allow_nan=False,
            )
    assert peak_bytes <= 512 * 1024**2


def learner(maximum=2, retention=FIFO):
    return NumpyRetainedNearestBCLearner(
        FixedInputNearestContinuousBCPolicy(SPEC, max_prototypes=maximum), retention
    )


def pairs(*rows):
    return (
        np.array([[x, 0.0] for x, y in rows], dtype=np.float64),
        np.array([[y] for x, y in rows], dtype=np.float64),
    )


def episode(name, rows, role="train"):
    return ContinuousEpisode(
        name,
        name + "-context",
        role,
        SPEC,
        tuple(
            ContinuousTransition(
                i,
                (float(x), 0.0),
                (float(y),),
                0.0,
                (float(rows[min(i + 1, len(rows) - 1)][0]), 0.0),
                i == len(rows) - 1,
                False,
            )
            for i, (x, y) in enumerate(rows)
        ),
    )


def test_should_select_fifo_unique_content_without_refreshing_retained_duplicates():
    a, b, c, d = (f"{i:064x}" for i in range(4))
    result = select_retained_rows((a, b), (a, c, d), 2, FIFO)
    assert result.ids == (c, d) and result.indices == (3, 4)
    assert (result.duplicate_rows, result.discarded_rows) == (1, 2)
    assert result.selection_candidates == 9 and result.reservoir_rank_evaluations == 0


def test_should_keep_seeded_content_ranks_independent_of_batch_boundaries():
    ids = tuple(f"{i:064x}" for i in range(8))
    whole = select_retained_rows((), ids, 2, RESERVOIR)
    first = select_retained_rows((), ids[:3], 2, RESERVOIR)
    second = select_retained_rows(first.ids, ids[3:], 2, RESERVOIR)
    assert whole.ids == second.ids
    assert whole.reservoir_rank_evaluations == 18
    assert select_retained_rows(whole.ids, ids, 2, RESERVOIR).ids == whole.ids


def test_should_forget_old_fifo_content_then_relearn_on_return_without_renewing_exposures():
    source = learner()
    source.train_batch(*pairs((-1, -0.5)))
    child = source.fork()
    child.train_batch(*pairs((1, 0.5), (2, 0.75)))
    assert child.predict(pairs((-1, 0))[0]).tolist() == [[0.5]]
    child.train_batch(*pairs((-1, -0.5)))
    assert child.predict(pairs((-1, 0), (2, 0))[0]).tolist() == [[-0.5], [0.75]]
    state = child.snapshot_state()
    assert (state.completed_updates, state.training_examples, state.retained_rows) == (3, 4, 2)
    assert state.discarded_rows == 2 and source.snapshot_state().training_examples == 1


def test_should_pay_duplicates_and_report_bounded_numeric_and_identity_payload():
    owner = learner()
    owner.train_batch(*pairs((0, 0.5), (0, 0.5), (0, 0.5)))
    cost = retained_costs(owner.snapshot_state())
    assert (cost["training_examples"], cost["retained_rows"], cost["duplicate_rows"]) == (3, 1, 2)
    assert (cost["numeric_bytes"], cost["content_id_bytes"], cost["duplicate_pair_bytes"]) == (
        24,
        64,
        48,
    )
    assert cost["digest_payload_bytes"] == 32
    assert cost["incoming_pair_hash_bytes"] == 144


def test_should_keep_conflicting_targets_distinct_and_preserve_earliest_distance_tie():
    owner = learner()
    owner.train_batch(*pairs((0, -0.5), (0, 0.5)))
    assert len(set(owner.snapshot_state().prototype_ids)) == 2
    assert owner.predict(pairs((0, 0))[0]).tolist() == [[-0.5]]


@pytest.fixture(scope="module")
def checkpoint():
    owner = learner(retention=RESERVOIR)
    owner.train_batch(*pairs((0, -0.5), (1, 0.5), (2, 0.75)))
    return owner, owner.snapshot_state()


@pytest.mark.parametrize("damage", ["exposures", "bool", "rank", "id", "action", "overflow"])
def test_should_refuse_invalid_full_restore_without_changing_the_owner(checkpoint, damage):
    owner, saved = checkpoint
    bad = replace(saved)
    if damage == "exposures":
        bad = replace(saved, training_examples=0)
    elif damage == "bool":
        bad = replace(saved, duplicate_rows=True)
    elif damage == "rank":
        bad = replace(saved, reservoir_rank_evaluations=0)
    elif damage == "id":
        bad = replace(saved, prototype_ids=("f" * 64, *saved.prototype_ids[1:]))
    elif damage == "overflow":
        bad = replace(saved, training_examples=2**63)
    else:
        bad = replace(saved, prototype_actions=saved.prototype_actions.copy())
        bad.prototype_actions[0, 0] = -0.25
    with pytest.raises(ValueError):
        owner.restore_state(bad)
    assert stream._same(owner.snapshot_state(), saved)


def test_should_copy_all_fork_and_restore_state_and_refuse_legacy_snapshot_admission(checkpoint):
    owner, saved = checkpoint
    child = owner.fork()
    returned = child.snapshot_state()
    assert not np.shares_memory(returned.prototype_features, saved.prototype_features)
    child.train_batch(*pairs((3, -0.75)))
    child.restore_state(saved)
    assert stream._same(child.snapshot_state(), saved)
    assert stream._same(owner.snapshot_state(), saved)
    returned.prototype_features.fill(123)
    assert stream._same(child.snapshot_state(), saved)
    with pytest.raises(ValueError, match="policy differs"):
        legacy_snapshot(saved, saved.policy)


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_refuse_reserved_or_final_episode_labels_before_updates(role):
    owner = learner()
    with pytest.raises(ValueError):
        owner.train_episode(episode("reserved", [(0, 0.5)], role))
    assert owner.snapshot_state().training_examples == 0


@pytest.fixture(scope="module")
def literal_stream():
    captured, predictions = {}, []
    factory = stream.ActorShadowRuntime

    def runtime(source, **kwargs):
        captured["source"] = source
        captured["runtime"] = factory(source, **kwargs)
        return captured["runtime"]

    def observe(readout, metadata):
        result = readout.predict_with_work(pairs((-1, 0), (2, 0))[0])
        values = result.actions.tolist()
        predictions.append((metadata["phase"], metadata["episode_id"], values))
        return dict(actions=values, candidates=result.distance_candidates)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(stream, "ActorShadowRuntime", runtime)
        report = stream.run_sequence(
            episode("a", [(-1, -0.5)]),
            (episode("b", [(1, 0.5), (2, 0.75)]), episode("return-a", [(-1, -0.5)])),
            observe,
            fixed_input=True,
            retention=FIFO,
            max_prototypes=2,
        )
    OBSERVED["literal_stream"] = report
    return report, captured, predictions


def test_should_execute_old_new_return_predictions_under_storage_pressure(literal_stream):
    report, _, predictions = literal_stream
    assert report["complete"] is True, report
    assert [row[2] for row in predictions] == [
        [[-0.5], [-0.5]],
        [[-0.5], [-0.5]],
        [[0.5], [0.75]],
        [[0.5], [0.75]],
        [[-0.5], [0.75]],
    ]
    assert report["model_state"]["candidate_training_examples"] == 4
    assert report["retention"]["retained_rows"] == 2
    assert report["retention"]["discarded_rows"] == 2


def test_should_preserve_original_source_actor_and_label_arrival_budget(literal_stream):
    report, captured, _ = literal_stream
    source = captured["source"].snapshot_state()
    actor = captured["runtime"].actor.snapshot_state().state
    assert stream._same(source, actor) and source.training_examples == 1
    assert [row["tick"] for row in report["events"]] == list(range(6))
    assert [row["budget_updates"] for row in report["events"]] == [0, 0, 1, 1, 1, 2]
    assert [row["candidate_prototype_rows"] for row in report["events"]] == [1, 1, 2, 2, 2, 2]
    assert [row["arrived_at"] for row in report["applied"]] == [2, 5]
    assert report["work"]["runtime_append_rows"] == 3


def test_should_stop_after_paid_callback_failure_and_preserve_retention_counts():
    def observe(readout, metadata):
        if metadata["phase"] == "after_release":
            readout.predict(pairs((-1, 0))[0])
            raise RuntimeError("literal observer stop")
        return {}

    report = stream.run_sequence(
        episode("a", [(-1, -0.5)]),
        (episode("b", [(1, 0.5), (2, 0.75)]), episode("unadmitted", [(-1, -0.5)])),
        observe,
        fixed_input=True,
        retention=FIFO,
        max_prototypes=2,
    )
    assert report["complete"] is False and "literal observer stop" in report["error"]["message"]
    assert report["experiment_state"]["completed_updates"] == 1
    assert len(report["applied"]) == 1 and len(report["retained_applied_on_failure"]) == 1
    assert report["observed_native_on_failure"]["candidate"]["training_examples"] == 3
    assert report["observed_native_on_failure"]["candidate"]["retention"]["retained_rows"] == 2


def test_should_preserve_observed_native_commit_when_update_return_is_lost(monkeypatch):
    train = NumpyRetainedNearestBCLearner.train_batch

    def lost(owner, features, targets):
        result = train(owner, features, targets)
        if owner._state.completed_updates > 1:
            raise RuntimeError("lost literal update return")
        return result

    monkeypatch.setattr(NumpyRetainedNearestBCLearner, "train_batch", lost)
    report = stream.run_sequence(
        episode("a", [(-1, -0.5)]),
        (episode("b", [(1, 0.5), (2, 0.75)]),),
        fixed_input=True,
        retention=FIFO,
        max_prototypes=2,
    )
    assert report["complete"] is False and "lost literal update" in report["error"]["message"]
    assert report["work"]["runtime_append_attempts"] == 1
    assert report["observed_native_on_failure"]["candidate"]["training_examples"] == 3
    assert report["work"]["runtime_attempted_rows"] == 2
    assert report["unreturned_append_possible"] is True
    OBSERVED["lost_return"] = report


def test_should_refuse_corrupted_previous_receipts_before_admitting_next_packet(monkeypatch):
    held = {}
    factory = stream.ActorShadowRuntime

    def runtime(source, **kwargs):
        held["runtime"] = factory(source, **kwargs)
        return held["runtime"]

    def observe(readout, metadata):
        if metadata["phase"] == "before_release" and metadata["packet_index"] == 1:
            held["runtime"]._inbox._applied.clear()
        return {}

    monkeypatch.setattr(stream, "ActorShadowRuntime", runtime)
    report = stream.run_sequence(
        episode("a", [(-1, -0.5)]),
        (episode("b", [(1, 0.5), (2, 0.75)]), episode("unadmitted", [(-1, -0.5)])),
        observe,
        fixed_input=True,
        retention=FIFO,
        max_prototypes=2,
    )
    assert report["complete"] is False
    assert "charges changed" in report["error"]["message"]
    assert report["experiment_state"]["completed_updates"] == 1
    assert len(report["applied"]) == 1
    assert report["work"]["runtime_append_attempts"] == 1
    OBSERVED["corrupt_receipt"] = report


def test_should_admit_more_than256_arrivals_only_with_explicit_bounded_retention():
    warm = episode("warm", [(-1, -0.5)])
    incoming = episode("large", [(i, 0.0) for i in range(256)])
    with pytest.raises(ValueError, match="256 prototype"):
        stream.run_sequence(warm, (incoming,), fixed_input=True)
    report = stream.run_sequence(warm, (incoming,), fixed_input=True, retention=FIFO)
    assert report["complete"] is True, report
    assert report["retention"]["training_examples"] == 257
    assert report["retention"]["retained_rows"] == 256
    assert report["retention"]["discarded_rows"] == 1


def test_should_keep_original_append_only_capacity_refusal_atomic():
    owner = NumpyNearestContinuousBCLearner(
        FixedInputNearestContinuousBCPolicy(SPEC, max_prototypes=2)
    )
    owner.train_batch(*pairs((-1, -0.5)))
    saved = owner.snapshot_state()
    with pytest.raises(ValueError, match="no rows were evicted"):
        owner.train_batch(*pairs((1, 0.5), (2, 0.75)))
    assert stream._same(owner.snapshot_state(), saved)


@pytest.mark.parametrize(
    "options",
    [
        dict(retention=FIFO),
        dict(fixed_input=True, max_prototypes=2),
        dict(fixed_input=True, retention=FIFO, max_prototypes=True),
        dict(fixed_input=True, retention=ReplayRetentionPolicy("content_hash")),
        dict(fixed_input=True, retention=FIFO, max_prototypes=257),
    ],
)
def test_should_refuse_invalid_explicit_options_before_model_work(options):
    before = WORK["initializers"]
    with pytest.raises(ValueError):
        stream.run_sequence(episode("a", [(0, -0.5)]), (episode("b", [(1, 0.5)]),), **options)
    assert WORK["initializers"] == before
