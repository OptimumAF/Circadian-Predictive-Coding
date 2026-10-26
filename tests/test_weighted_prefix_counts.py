"""Literal metadata prefixes and source refusals; no learned task evidence."""

from copy import deepcopy
from dataclasses import replace
import sys
from typing import Any

import pytest

from src.adapters import minigrid_public_cue_recurrent_learner as public
from src.adapters import torch_train_weighted_learner as weighted
from src.app.coupled_attribution import _same_value
from src.core.demonstration_episode import episode_arrays
from test_public_cue_recurrent_learner import configure_scripted_native, episode, policy


@pytest.fixture
def scripted(monkeypatch):
    configure_scripted_native(monkeypatch)
    monkeypatch.setattr(weighted, "_same_state", _same_value)


def schedule(count=128):
    sources = tuple(episode(cue) for cue in ("blue", "red"))
    return tuple(
        replace(
            sources[int(i % 3 == 0)],
            episode_id=f"literal-{i}",
            context="same-context",
            transitions=tuple(
                replace(
                    sources[int(i % 3 == 0)].transitions[min(j, 1)], step=j, terminal=j == i % 8 + 1
                )
                for j in range(i % 8 + 2)
            ),
        )
        for i in range(count)
    )


def test_should_match_every_variable_length_prefix_without_decoding_again(scripted, monkeypatch):
    episodes = schedule()
    source = weighted.TorchTrainWeightedLearner(policy(), episodes)
    cold = source.snapshot_state()
    assert type(cold) is public.PublicCueRecurrentSnapshot
    counts, rows = [0, 0], [0, 0]
    original = public._cue
    selected = [original(episode_arrays(e)[0][0]) for e in episodes]

    def refuse(*args):
        pytest.fail("unchanged admitted metadata decoded again")

    monkeypatch.setattr(public, "_cue", refuse)
    for prefix in range(129):
        if prefix:
            head = selected[prefix - 1]
            counts[head] += 1
            rows[head] += len(episodes[prefix - 1].transitions)
        state = replace(
            cold,
            heads=(
                replace(cold.heads[0], completed_updates=counts[0], training_examples=rows[0]),
                replace(cold.heads[1], completed_updates=counts[1], training_examples=rows[1]),
            ),
        )
        source.restore_state(state)
        assert _same_value(source.snapshot_state(), state)
        assert source._public_prefixes[prefix] == (counts[0], rows[0], counts[1], rows[1])


def test_should_bound_storage_and_rebind_each_fork(scripted):
    source = weighted.TorchTrainWeightedLearner(policy(), schedule(512))
    table = source._public_prefixes
    assert len(table) == 513
    assert all(
        type(row) is tuple and len(row) == 4 and all(type(v) is int for v in row) for row in table
    )
    upper = sys.getsizeof(table) + sum(
        sys.getsizeof(row) + sum(sys.getsizeof(v) for v in row) for row in table
    )
    assert upper <= 128 * 1024
    child = source.fork()
    assert child._public_prefixes == table and child._public_prefixes is not table
    assert child._public_prefixes_owner is child._public_prefixes
    source._public_prefixes = tuple(list(table))
    with pytest.raises(ValueError, match="prefix"):
        source.snapshot_state()
    child.snapshot_state()


@pytest.mark.parametrize("target", ["source", "seal"])
@pytest.mark.parametrize(
    "fault", ["observation", "next_observation", "role", "type", "extra", "order", "dictionary"]
)
def test_should_refuse_every_changed_schedule_witness_before_using_prefixes(
    scripted, target, fault
):
    source = weighted.TorchTrainWeightedLearner(policy(), schedule(3))
    saved = source.snapshot_state()
    e = (source._episodes if target == "source" else source._schedule_seal)[0]
    if fault in ("observation", "next_observation"):
        t = e.transitions[0]
        object.__setattr__(t, fault, tuple(list(getattr(t, fault))))
    elif fault == "role":
        object.__setattr__(e, "role", "final_test")
    elif fault == "type":
        object.__setattr__(e.transitions[0], "step", False)
    elif fault == "extra":
        object.__setattr__(e, "extra", True)
    elif fault == "order":
        value = vars(e).pop("context")
        vars(e)["context"] = value
    else:
        object.__setattr__(e, "__dict__", dict(vars(e)))
    with pytest.raises(ValueError):
        source.restore_state(saved)


@pytest.mark.parametrize("fault", ["equal", "list", "short", "wrong_counts", "bool"])
def test_should_refuse_replaced_prefix_data_even_if_equal(scripted, fault):
    source = weighted.TorchTrainWeightedLearner(policy(), schedule(3))
    table = source._public_prefixes
    replacement: Any
    if fault == "equal":
        replacement = tuple(list(table))
    elif fault == "list":
        replacement = list(table)
    elif fault == "short":
        replacement = table[:-1]
    else:
        replacement = ((False, 0, 0, 0) if fault == "bool" else (1, 0, 0, 0),) + table[1:]
    source._public_prefixes = replacement
    with pytest.raises(ValueError, match="prefix"):
        source.snapshot_state()


@pytest.mark.parametrize("fault", ["head", "rows", "overrun", "counter_type", "policy"])
def test_should_refuse_corrupt_snapshot_without_changing_owned_state(scripted, fault):
    source = weighted.TorchTrainWeightedLearner(policy(), schedule(3))
    before = source.snapshot_state()
    assert type(before) is public.PublicCueRecurrentSnapshot
    heads = list(before.heads)
    if fault == "policy":
        state = replace(
            before,
            policy=replace(
                before.policy, native_policy=replace(before.policy.native_policy, seed=99)
            ),
        )
    else:
        changes: dict[str, Any] = dict(completed_updates=1, training_examples=2)
        if fault == "rows":
            changes = dict(completed_updates=1, training_examples=3)
        elif fault == "overrun":
            changes = dict(completed_updates=4, training_examples=8)
        elif fault == "counter_type":
            changes = dict(completed_updates=True, training_examples=2)
        # First literal cue selects head1: head0 with same aggregate must refuse.
        heads[0 if fault == "head" else 1] = replace(heads[0 if fault == "head" else 1], **changes)
        state = replace(before, heads=(heads[0], heads[1]))
    with pytest.raises(ValueError):
        source.restore_state(state)
    assert _same_value(source.snapshot_state(), before)


def test_should_decode_actual_initial_public_cue_for_live_prediction(scripted, monkeypatch):
    source = weighted.TorchTrainWeightedLearner(policy(), schedule(3))
    before = deepcopy(source.snapshot_state())
    original = public._cue
    seen = []

    def record(frame):
        selected = original(frame)
        seen.append(selected)
        return selected

    monkeypatch.setattr(public, "_cue", record)
    source.predict(episode_arrays(episode("red"))[0])
    assert seen and set(seen) == {1}
    assert _same_value(source.snapshot_state(), before)
