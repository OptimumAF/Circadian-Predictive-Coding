"""Bounded ordinary TRAIN recurrence planning; no fitting or task success claims."""

from dataclasses import asdict, replace
import pickle

import numpy as np
import pytest

from src.adapters import minigrid_train_catalogue as catalogue
from src.core.demonstration_episode import episode_arrays
from test_arrived_cue_catalogue import episode
from test_catalogue_capacity import sources as legacy_sources


def schedule():
    # Independently constructed TRAIN literals, never recorded probe/final packets.
    contexts = (("A", 1, 2), ("B", 3, 5), ("C_train", 5, 1), ("G_train", 7, 4), ("A", 1, 2))
    return tuple(
        episode(f"arrival-{tick}-{cue}", cue_id, mark=mark, action=action, context=context)
        for tick, (context, mark, action) in enumerate(contexts)
        for cue, cue_id in (("blue", 2), ("red", 0))
    )


def recurring(packets):
    return catalogue.compose_arrived_cue_catalogue(packets, retained_per_cue=3, max_arrivals=10)


def test_should_plan_four_contexts_then_return_with_original_causal_spans_and_costs():
    packets = schedule()
    original, rng = pickle.dumps(packets), pickle.dumps(np.random.get_state())
    plan = recurring(packets)
    assert [len(b.features) for b in plan.training] == [9, 9, 18, 18, 27, 27, 36, 36, 36, 36]
    assert (plan.arrival_row_exposures, plan.replay_row_exposures, plan.fit_row_exposures) == (
        90,
        162,
        252,
    )
    assert plan.retained_per_cue == 3
    assert plan.input_array_bytes == 252 * 297 * 8 == 598752
    assert plan.peak_joint_template_array_bytes == 36 * 297 * 8 == 85536
    assert plan.peak_bank_rows == 54
    assert plan.peak_bank_observation_target_float64_equivalent_bytes == 54 * 593 * 8 == 256176
    for tick, (wake, batch) in enumerate(zip(plan.wakes, plan.training, strict=True)):
        old = tuple(range(tick % 2, tick, 2))[-3:]
        expected = tuple(packets[i] for i in old) + (packets[tick],)
        assert wake.observed_at == tick and wake.arrived_at == tick + 1
        assert wake.retained_episode_ids == tuple(packets[i].episode_id for i in old)
        assert wake.retained_arrival_ticks == tuple(i + 1 for i in old)
        assert all(i <= tick for i in wake.retained_arrival_ticks)
        assert [s.episode_id for s in wake.source_spans] == [e.episode_id for e in expected]
        assert [(s.start, s.stop) for s in wake.source_spans] == [
            (9 * i, 9 * (i + 1)) for i in range(len(expected))
        ]
        arrays = [episode_arrays(e) for e in expected]
        np.testing.assert_array_equal(batch.features, np.concatenate([x for x, y in arrays]))
        np.testing.assert_array_equal(batch.targets, np.concatenate([y for x, y in arrays]))
        assert batch.role == "train" and batch.episode is None and batch.num_classes == 7
        assert wake.current_rows == 9 and wake.replay_rows == 9 * len(old)
        assert wake.bank_rows_after <= 54 and len(wake.bank_episode_ids_after) <= 6
    assert plan.wakes[6].retained_episode_ids == tuple(packets[i].episode_id for i in (0, 2, 4))
    assert plan.wakes[8].retained_episode_ids == tuple(packets[i].episode_id for i in (2, 4, 6))
    assert plan.wakes[-1].bank_episode_ids_after == tuple(
        packets[i].episode_id for i in (4, 6, 8, 5, 7, 9)
    )
    ids = [key for batch in plan.training for key in batch.row_ids]
    assert len(ids) == len(set(ids)) == 252
    assert pickle.dumps(packets) == original and pickle.dumps(np.random.get_state()) == rng


def test_should_leave_earlier_wakes_unchanged_when_only_future_train_labels_change():
    packets = schedule()
    original = recurring(packets)
    future = replace(
        packets[-1], transitions=tuple(replace(t, action=6) for t in packets[-1].transitions)
    )
    changed = recurring((*packets[:-1], future))
    for before, after in zip(original.training[:-1], changed.training[:-1], strict=True):
        np.testing.assert_array_equal(before.features, after.features)
        np.testing.assert_array_equal(before.targets, after.targets)
        assert before.row_ids == after.row_ids
    assert not np.array_equal(original.training[-1].targets, changed.training[-1].targets)
    assert original.wakes == changed.wakes


def test_should_own_disjoint_wake_arrays_without_mutating_sources_or_other_wakes():
    packets = schedule()
    before = pickle.dumps(packets)
    plan = recurring(packets)
    arrays = [a for batch in plan.training for a in (batch.features, batch.targets)]
    assert not any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1 :])
    last = plan.training[-1].features.copy()
    plan.training[0].features.fill(0)
    plan.training[0].targets.fill(0)
    np.testing.assert_array_equal(last, plan.training[-1].features)
    assert pickle.dumps(packets) == before


def test_should_bound_single_public_cue_fifo_and_ignore_misleading_context_names():
    # Object 6 is a second cue ball, so only non-cue markers vary here.
    packets = tuple(
        episode(f"red-name-{i}", context="red", mark=(1, 3, 5, 7)[i % 4]) for i in range(10)
    )
    plan = recurring(packets)
    assert plan.fit_row_exposures == 306 and plan.replay_row_exposures == 216
    assert plan.peak_bank_rows == 27 and max(len(b.features) for b in plan.training) == 36
    for i, wake in enumerate(plan.wakes):
        assert wake.cue == "blue"
        assert wake.retained_episode_ids == tuple(e.episode_id for e in packets[max(0, i - 3) : i])
        assert wake.bank_episode_ids_after == tuple(
            e.episode_id for e in packets[max(0, i - 2) : i + 1]
        )


@pytest.mark.parametrize(
    "options",
    [
        dict(max_arrivals=True),
        dict(max_arrivals=6.0),
        dict(max_arrivals=7),
        dict(max_arrivals=11),
        dict(max_arrivals="10"),
        dict(max_arrivals=10, retained_per_cue=True),
        dict(max_arrivals=10, retained_per_cue=0),
        dict(max_arrivals=10, retained_per_cue=4),
        dict(retained_per_cue=3),
    ],
)
def test_should_refuse_ambiguous_bounds_before_source_validation_or_allocation(
    monkeypatch, options
):
    monkeypatch.setattr(catalogue, "_validate_sources", lambda *a: pytest.fail("source reached"))
    with pytest.raises(ValueError, match="exact integer"):
        catalogue.compose_arrived_cue_catalogue((), **options)


@pytest.mark.parametrize(
    "fault", ["eleven", "probe_train", "probe_test", "final_test", "short", "duplicate"]
)
def test_should_refuse_invalid_schedule_before_first_joint_array(monkeypatch, fault):
    packets = schedule()
    if fault == "eleven":
        packets = (*packets, replace(packets[0], episode_id="extra"))
    elif fault in ("probe_train", "probe_test", "final_test"):
        packets = (replace(packets[0], role=fault), *packets[1:])
    elif fault == "short":
        packets = (episode("short", rows=8), *packets[1:])
    else:
        packets = (packets[0], packets[0], *packets[2:])
    monkeypatch.setattr(
        catalogue, "compose_catalogue_training", lambda *a: pytest.fail("joint array reached")
    )
    with pytest.raises(ValueError):
        recurring(packets)


@pytest.mark.parametrize("fault", ["label", "equal_observation", "record_dictionary"])
def test_should_refuse_changed_owned_future_source_before_next_composition(monkeypatch, fault):
    packets = schedule()
    actual = catalogue.compose_catalogue_training
    calls = []

    def compose(*args):
        result = actual(*args)
        calls.append(1)
        if fault == "label":
            object.__setattr__(packets[-1].transitions[-1], "action", 6)
        elif fault == "equal_observation":
            row = packets[-1].transitions[-1]
            object.__setattr__(row, "observation", tuple(list(row.observation)))
        else:
            object.__setattr__(packets[-1], "__dict__", dict(vars(packets[-1])))
        return result

    monkeypatch.setattr(catalogue, "compose_catalogue_training", compose)
    with pytest.raises(ValueError, match="changed"):
        recurring(packets)
    assert calls == [1]


def test_should_refuse_cumulative_array_limit_before_allocating_the_next_wake(monkeypatch):
    packets = schedule()
    # Each template and all original sources fit; the accumulated wakes do not.
    monkeypatch.setattr(catalogue, "MAX_INPUT_ARRAY_BYTES", 108 * 297 * 8)
    actual = catalogue.compose_catalogue_training
    calls = []

    def compose(*args):
        calls.append(1)
        return actual(*args)

    monkeypatch.setattr(catalogue, "compose_catalogue_training", compose)
    with pytest.raises(ValueError, match="cumulative input arrays"):
        recurring(packets)
    assert len(calls) == 6


@pytest.mark.parametrize("capacity", [1, 2])
def test_should_preserve_legacy_wakes_arrays_ids_schema_and_costs_under_explicit_bounds(capacity):
    packets = legacy_sources()
    implicit = catalogue.compose_arrived_cue_catalogue(packets, retained_per_cue=capacity)
    explicit = catalogue.compose_arrived_cue_catalogue(
        packets, retained_per_cue=capacity, max_arrivals=6
    )
    recurring_bound = catalogue.compose_arrived_cue_catalogue(
        packets, retained_per_cue=capacity, max_arrivals=10
    )
    for other in (explicit, recurring_bound):
        assert [asdict(w) for w in implicit.wakes] == [asdict(w) for w in other.wakes]
        assert vars(implicit).keys() == vars(other).keys()
        for key, value in vars(implicit).items():
            if key not in ("training", "wakes"):
                assert getattr(other, key) == value
        for left, right in zip(implicit.training, other.training, strict=True):
            assert left.row_ids == right.row_ids
            np.testing.assert_array_equal(left.features, right.features)
            np.testing.assert_array_equal(left.targets, right.targets)
    if capacity == 1:
        assert implicit.fit_row_exposures == 90 and implicit.peak_bank_rows == 18
    else:
        assert implicit.fit_row_exposures == 108 and implicit.peak_bank_rows == 36
