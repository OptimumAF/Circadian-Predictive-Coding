"""Read saved evidence only: no learner updates, prediction calls or physics."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from src.adapters.minigrid_saved_fifo_inputs import decode_saved_fifo_inputs

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "runs"
    / "minigrid-recurring-fifo-20261017"
    / "native.json"
)


@pytest.fixture(scope="module")
def original():
    return json.loads(SOURCE.read_text(encoding="utf-8"))


def test_should_reuse_complete_saved_cells_and_preserve_paid_censoring(original):
    value = decode_saved_fifo_inputs(original)
    assert tuple(state.completed_updates for state in value.snapshots) == (0, 32, 64, 96, 128)
    assert tuple(state.training_examples for state in value.snapshots) == (0, 288, 576, 864, 1152)
    assert tuple(len(state.native_state.entries) for state in value.snapshots) == (
        0,
        18,
        36,
        36,
        36,
    )
    assert tuple(state.native_state.retired_examples for state in value.snapshots) == (
        0,
        0,
        0,
        288,
        576,
    )
    assert tuple(state.native_state.evictions for state in value.snapshots) == (0, 0, 0, 18, 36)
    pairs = tuple((context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red"))
    assert value.score_slots == tuple((point, *pair) for point in (96, 128) for pair in pairs)
    assert value.game_slots == tuple((64, *pair) for pair in pairs[2:]) + tuple(
        (point, *pair) for point in (96, 128) for pair in pairs
    )
    assert len(value.completed_games) == 14
    assert value.censored_games == ((64, "B", "blue"),)
    assert value.censored_games[0] in value.game_slots
    assert value.origin_status == "incomplete"
    assert value.origin_reason == original["reason"]
    assert value.original_elapsed_seconds == original["total_seconds"]
    assert value.original_training_updates == 128 and value.original_training_rows == 1152
    assert len(value.episodes) == 6
    assert all(
        episode.role == "train" and len(episode.transitions) == 9 and episode.num_classes == 7
        for episode in value.episodes
    )


def test_should_own_full_snapshot_entries_and_sources_without_reconstructing(original):
    payload = deepcopy(original)
    result = decode_saved_fifo_inputs(payload)
    feature = result.snapshots[4].native_state.entries[0].features[0]
    count = result.snapshots[4].native_state.entries[0].counts[0]
    observation = result.episodes[0].transitions[0].observation[0]
    payload["observations"][4]["snapshot"]["native_state"]["entries"][0]["features"][0] = 123
    payload["observations"][4]["snapshot"]["native_state"]["entries"][0]["counts"][0] = 999
    payload["training_sources"][0]["transitions"][0]["observation"][0] = 123
    assert result.snapshots[4].native_state.entries[0].features[0] == feature
    assert result.snapshots[4].native_state.entries[0].counts[0] == count
    assert result.episodes[0].transitions[0].observation[0] == observation
    with pytest.raises(FrozenInstanceError):
        result.origin_status = "complete"


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("protocol", "seed"), 18),
        (("protocol", "policy", "native_policy", "max_entries"), 64),
        (("protocol", "policy", "encoding_id"), "raw_history"),
        (("protocol", "stream", "heldout"), ["D"]),
        (("methods", 0, "source_unchanged"), False),
        (("methods", 0, "actor_unchanged"), False),
        (("methods", 0, "charged_updates"), 127),
        (("methods", 0, "native_training_examples"), 1151),
        (("methods", 0, "wake_calls_returned"), 127),
        (("methods", 0, "stream_result", "applied_updates", 0, "update_number"), True),
        (("methods", 0, "stream_result", "applied_updates", 127, "arrived_at"), 127),
        (("methods", 0, "partial_stream", "native_candidate_updates"), 127),
        (("methods", 0, "final_evaluation", "status"), "complete"),
        (("observations", 4, "snapshot", "native_state", "training_examples"), 1151),
        (("observations", 3, "snapshot", "native_state", "retired_examples"), 287),
        (("observations", 4, "snapshot", "native_state", "policy", "max_payload_bytes"), 155136),
        (("observations", 4, "snapshot", "native_state", "entries", 0, "features", 147), 1.0),
        (("observations", 4, "evictions"), 35),
        (("observations", 4, "retirement_metadata_bytes"), 0),
        (("training_sources", 0, "role"), "final_test"),
        (("training_sources", 0, "episode_id"), "minigrid-source-A-red"),
        (("source_reports", 0, "transitions", 0, "action"), 1),
        (("source_reports", 0, "closed"), False),
        (("source_reports", 0, "environment_id"), "replaced_environment"),
        (("source_reports", 0, "encoding_id"), "raw_history_replacement"),
        (("methods", 0, "readouts", 1, "scores", 0, "ce"), 0.1),
        (("methods", 0, "readouts", 1, "scores", 0, "probabilities", 0, 0), 0.2),
        (("methods", 0, "readouts", 2, "episodes", 0, "state_unchanged"), False),
        (("methods", 0, "readouts", 2, "episodes", 0, "predictions", 0, "action"), 0),
        (("methods", 0, "readouts", 2, "episodes", 0, "environment", "role"), "final_test"),
        (("methods", 0, "readouts", 2, "episodes", 0, "environment", "environment_id"), "other"),
        (("methods", 0, "readouts", 2, "episodes", 0, "environment", "encoding_id"), "other"),
        (("methods", 0, "readouts", 2, "episodes", 0, "return"), 0.0),
        (("methods", 0, "readouts", 2, "episodes", 0, "prediction_calls_returned"), 8),
        (("methods", 0, "readouts", 2, "episodes", 2, "reason"), None),
        (("total_seconds",), float("nan")),
    ],
)
def test_should_refuse_corrupted_or_relabelled_original_evidence(original, path, replacement):
    payload = deepcopy(original)
    parent = payload
    for part in path[:-1]:
        parent = parent[part]
    parent[path[-1]] = replacement
    with pytest.raises(ValueError):
        decode_saved_fifo_inputs(payload)


def test_should_refuse_duplicate_keys_and_summary_only_snapshot(original):
    payload = deepcopy(original)
    native = payload["observations"][4]["snapshot"]["native_state"]
    native["entries"][1] = deepcopy(native["entries"][0])
    with pytest.raises(ValueError):
        decode_saved_fifo_inputs(payload)
    payload = deepcopy(original)
    del payload["observations"][4]["snapshot"]
    with pytest.raises(ValueError, match="full snapshot"):
        decode_saved_fifo_inputs(payload)


def test_should_not_admit_more_work_by_deleting_paid_complete_games(original):
    payload = deepcopy(original)
    payload["methods"][0]["readouts"][0]["episodes"].pop()
    with pytest.raises(ValueError):
        decode_saved_fifo_inputs(payload)


def test_should_refuse_deletion_of_the_paid_censored_attempt(original):
    payload = deepcopy(original)
    payload["methods"][0]["readouts"][2]["episodes"].pop()
    with pytest.raises(ValueError, match="paid original game resets"):
        decode_saved_fifo_inputs(payload)


def test_should_refuse_positive_reward_truncation_relabelled_as_success(original):
    payload = deepcopy(original)
    game = payload["methods"][0]["readouts"][2]["episodes"][0]
    environment = game["environment"]
    environment["terminated"] = False
    environment["truncated"] = True
    environment["transitions"][-1]["terminated"] = False
    environment["transitions"][-1]["truncated"] = True
    assert game["success"] is True and game["return"] > 0
    with pytest.raises(ValueError, match="success summaries differ"):
        decode_saved_fifo_inputs(payload)


def test_should_refuse_simultaneous_termination_and_truncation(original):
    payload = deepcopy(original)
    game = payload["methods"][0]["readouts"][2]["episodes"][0]
    game["environment"]["transitions"][-1]["truncated"] = True
    game["environment"]["truncated"] = True
    with pytest.raises(ValueError, match="simultaneously"):
        decode_saved_fifo_inputs(payload)


def test_should_preserve_original_source_payload_byte_equivalence(original):
    before = json.dumps(original, allow_nan=False, separators=(",", ":"))
    decode_saved_fifo_inputs(original)
    assert json.dumps(original, allow_nan=False, separators=(",", ":")) == before
