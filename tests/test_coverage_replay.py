"""Whole-episode coverage retention and deterministic detached replay selection."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.core.circadian_predictive_coding import ReplayRetentionBudget, replay_sample_id
from src.core.coverage_replay import CoverageReplayBuffer
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    EpisodeReplaySchema,
    decode_episode,
    encode_episode,
)
from src.core.shared_replay_schedule import SharedReplayBuffer


SCHEMA = EpisodeReplaySchema(2, 2, 5)
CONTEXTS = ("A-red", "A-blue", "B-red", "B-blue", "C_train-red", "C_train-blue")
ROW_BYTES = 8 * (SCHEMA.encoded_dim + 1)


def episode(context="A-red", identity="packet-0", role="train"):
    return DemonstrationEpisode(
        identity,
        context,
        role,
        (
            DemonstrationTransition(0, (0.1, 0.2), 0, 0, (0.3, 0.4), False, 5),
            DemonstrationTransition(1, (0.3, 0.4), 4, 1, (0.5, 0.6), True, 5),
        ),
        5,
    )


def buffer(contexts=CONTEXTS) -> CoverageReplayBuffer:
    return CoverageReplayBuffer(
        SCHEMA.encoded_dim,
        ReplayRetentionBudget(len(contexts), ROW_BYTES * len(contexts)),
        schema=SCHEMA,
        contexts=contexts,
    )


def observe(retained, *episodes):
    rows = [encode_episode(packet, SCHEMA) for packet in episodes]
    retained.observe_train_batch(
        np.concatenate([row for row, _ in rows]), np.concatenate([target for _, target in rows])
    )


def selected_keys(selection):
    return tuple(
        decode_episode(row, target, SCHEMA).context for row, target in selection.training_batches()
    )


def selected_ids(selection):
    return tuple(
        decode_episode(row, target, SCHEMA).episode_id
        for row, target in selection.training_batches()
    )


def snapshot(retained):
    return (
        retained.retention,
        retained.coverage_keys,
        retained.coverage_cursor,
        tuple(
            (identity, row.tobytes(), target.tobytes())
            for identity, (row, target) in retained._rows.items()
        ),
    )


def test_should_retain_one_whole_latest_packet_per_declared_key_and_standard_content_id():
    retained = buffer()
    assert isinstance(retained, SharedReplayBuffer)
    packets = [episode(context, f"packet-{index}") for index, context in enumerate(CONTEXTS)]
    observe(retained, *reversed(packets))
    assert retained.coverage_keys == CONTEXTS
    assert (
        retained.retention.example_count == 6 and retained.retention.retained_bytes == 6 * ROW_BYTES
    )
    assert selected_ids(retained.select_coverage(99)) == tuple(
        packet.episode_id for packet in packets
    )
    replacement = episode(CONTEXTS[1], "new-blue")
    observe(retained, replacement)
    selection = retained.select_coverage(6)
    assert selected_ids(selection) == (
        "packet-0",
        "new-blue",
        "packet-2",
        "packet-3",
        "packet-4",
        "packet-5",
    )
    assert replay_sample_id(*encode_episode(replacement, SCHEMA)) in selection.sample_ids
    assert (
        replay_sample_id(*encode_episode(packets[1], SCHEMA)) not in retained.retention.sample_ids
    )
    assert retained.retention.example_count == 6


def test_should_continue_round_robin_after_last_selected_key_across_calls():
    retained = buffer()
    observe(
        retained, *(episode(context, f"packet-{index}") for index, context in enumerate(CONTEXTS))
    )
    assert retained.coverage_cursor is None
    assert [selected_keys(retained.select_coverage(1))[0] for _ in range(8)] == list(
        (*CONTEXTS, *CONTEXTS[:2])
    )
    assert retained.coverage_cursor == CONTEXTS[1]
    assert selected_keys(retained.select_coverage(3)) == CONTEXTS[2:5]
    assert selected_keys(retained.select_coverage(3)) == (CONTEXTS[5], CONTEXTS[0], CONTEXTS[1])


def test_should_skip_absent_keys_and_never_duplicate_a_key_with_large_selection_budget():
    retained = buffer()
    assert retained.select_coverage(4).sample_ids == () and retained.coverage_cursor is None
    observe(retained, episode(CONTEXTS[4], "c"), episode(CONTEXTS[1], "a"))
    assert retained.coverage_keys == (CONTEXTS[1], CONTEXTS[4])
    assert selected_keys(retained.select_coverage(1)) == (CONTEXTS[1],)
    assert selected_keys(retained.select_coverage(99)) == (CONTEXTS[4], CONTEXTS[1])
    observe(retained, episode(CONTEXTS[0], "new-a"))
    assert selected_keys(retained.select_coverage(3)) == (CONTEXTS[4], CONTEXTS[0], CONTEXTS[1])


def test_should_replace_repeated_key_with_last_row_atomically_without_moving_cursor():
    retained = buffer()
    observe(retained, episode(identity="initial"))
    retained.select_coverage(1)
    observe(retained, episode(identity="first-in-batch"), episode(identity="last-in-batch"))
    assert retained.coverage_cursor == "A-red"
    assert selected_ids(retained.select_coverage(1)) == ("last-in-batch",)
    assert retained.retention.example_count == 1
    for index in range(100):
        observe(retained, episode(identity=f"replacement-{index}"))
    assert len(retained._coverage_ids) == len(retained._rows) == 1
    assert retained.retention.retained_bytes == ROW_BYTES


def test_should_own_observed_rows_and_every_selection_copy():
    retained = buffer()
    row, target = encode_episode(episode(), SCHEMA)
    retained.observe_train_batch(row, target)
    row.fill(999)
    target.fill(999)
    selection = retained.select_coverage(1)
    assert selected_ids(selection) == ("packet-0",)
    first_row, first_target = selection.training_batches()[0]
    first_row.fill(999)
    first_target.fill(999)
    assert selected_ids(selection) == ("packet-0",)
    assert selected_ids(retained.select_coverage(1)) == ("packet-0",)
    assert not np.shares_memory(selection._rows[0][0], next(iter(retained._rows.values()))[0])


def test_should_deepcopy_complete_cursor_membership_and_owned_rows():
    retained = buffer()
    observe(
        retained, *(episode(context, f"packet-{index}") for index, context in enumerate(CONTEXTS))
    )
    retained.select_coverage(2)
    copied = deepcopy(retained)
    assert copied.coverage_cursor == retained.coverage_cursor == "A-blue"
    assert selected_keys(copied.select_coverage(3)) == selected_keys(retained.select_coverage(3))
    observe(copied, episode("B-red", "copy-only"))
    assert "copy-only" in selected_ids(copied.select_coverage(6))
    assert "copy-only" not in selected_ids(retained.select_coverage(6))
    for identity in copied._rows:
        if identity in retained._rows:
            assert not np.shares_memory(copied._rows[identity][0], retained._rows[identity][0])


def test_should_detach_budget_and_schema_configuration_at_admission():
    budget = ReplayRetentionBudget(6, 6 * ROW_BYTES)
    schema = replace(SCHEMA)
    retained = CoverageReplayBuffer(schema.encoded_dim, budget, schema=schema, contexts=CONTEXTS)
    object.__setattr__(budget, "max_examples", 1)
    object.__setattr__(schema, "observation_dim", 3)
    observe(
        retained, *(episode(context, f"packet-{index}") for index, context in enumerate(CONTEXTS))
    )
    assert retained.retention.example_count == 6
    assert retained.budget.max_examples == 6


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_nontrain_packets_at_existing_required_codec_admission(role):
    retained = buffer()
    observe(retained, episode(identity="original"))
    before = snapshot(retained)
    with pytest.raises(ValueError, match="only training"):
        row, target = encode_episode(episode(identity="reserved", role=role), SCHEMA)
        retained.observe_train_batch(row, target)
    assert snapshot(retained) == before


@pytest.mark.parametrize(
    "kind", ["unknown", "dummy", "nonfinite", "terminal", "continuity", "padding"]
)
def test_should_refuse_whole_partly_invalid_batch_without_any_replacement_or_cursor_change(kind):
    retained = buffer()
    observe(retained, episode(identity="original"))
    retained.select_coverage(1)
    before = snapshot(retained)
    first = encode_episode(episode(identity="would-replace"), SCHEMA)
    second = encode_episode(
        episode("unknown" if kind == "unknown" else "A-blue", "invalid-second"), SCHEMA
    )
    rows, targets = np.concatenate([first[0], second[0]]), np.concatenate([first[1], second[1]])
    if kind == "dummy":
        targets[1, 0] = 1
    elif kind == "nonfinite":
        rows[1, -1] = np.nan
    elif kind == "terminal":
        rows[1, -1] = 0
    elif kind == "continuity":
        rows[1, 99 + 7] = 0.9
    elif kind == "padding":
        rows[1, 64] = 1
    with pytest.raises(ValueError):
        retained.observe_train_batch(rows, targets)
    assert snapshot(retained) == before


class TupleSubclass(tuple):
    pass


@pytest.mark.parametrize(
    "contexts",
    [
        [],
        (),
        ("A",) * 7,
        ("A", "A"),
        ("",),
        (" A",),
        (1,),
        ("é" * 17,),
        ("\ud800",),
        TupleSubclass(("A",)),
    ],
)
def test_should_require_exact_bounded_unique_utf8_declared_keys(contexts):
    with pytest.raises(ValueError):
        buffer(contexts)


@pytest.mark.parametrize(
    "kind",
    [
        "schema_type",
        "schema_corrupt",
        "budget_type",
        "budget_corrupt",
        "row_count",
        "bytes",
        "dimension",
        "bool_dimension",
    ],
)
def test_should_require_valid_schema_and_capacity_for_every_declared_key(kind):
    schema: Any = replace(SCHEMA)
    budget: Any = ReplayRetentionBudget(6, 6 * ROW_BYTES)
    dimension: Any = SCHEMA.encoded_dim
    if kind == "schema_type":
        schema = object()
    elif kind == "schema_corrupt":
        object.__setattr__(schema, "steps_per_episode", 0)
    elif kind == "budget_type":
        budget = object()
    elif kind == "budget_corrupt":
        object.__setattr__(budget, "max_examples", 0)
    elif kind == "row_count":
        budget = ReplayRetentionBudget(5, 6 * ROW_BYTES)
    elif kind == "bytes":
        budget = ReplayRetentionBudget(6, 6 * ROW_BYTES - 1)
    elif kind == "dimension":
        dimension += 1
    else:
        dimension = True
    with pytest.raises(ValueError):
        CoverageReplayBuffer(dimension, budget, schema=schema, contexts=CONTEXTS)


@pytest.mark.parametrize("value", [True, 0, -1, 1.0, np.int64(1), None])
def test_should_refuse_nonpositive_or_nonnative_selection_budget_without_moving_cursor(value):
    retained = buffer()
    observe(retained, episode())
    before = snapshot(retained)
    with pytest.raises(ValueError):
        retained.select_coverage(value)
    assert snapshot(retained) == before


@pytest.mark.parametrize(
    "kind", ["empty", "oversized", "list", "dtype", "width", "target_dtype", "target_shape"]
)
def test_should_refuse_bad_encoded_batch_envelopes_without_mutation(kind):
    retained = buffer()
    row: Any
    target: Any
    row, target = encode_episode(episode(), SCHEMA)
    if kind == "empty":
        row, target = row[:0], target[:0]
    elif kind == "oversized":
        row, target = np.repeat(row, 513, axis=0), np.repeat(target, 513, axis=0)
    elif kind == "list":
        row = row.tolist()
    elif kind == "dtype":
        row = row.astype(np.float32)
    elif kind == "width":
        row = row[:, :-1]
    elif kind == "target_dtype":
        target = target.astype(np.int64)
    else:
        target = target[:, 0]
    with pytest.raises(ValueError):
        retained.observe_train_batch(row, target)
    assert retained.retention.example_count == 0 and retained.coverage_cursor is None
