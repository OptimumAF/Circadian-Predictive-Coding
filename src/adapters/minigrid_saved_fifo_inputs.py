"""Decode the existing FIFO36 run into bounded, owned read-only inputs.

Inputs are the complete original JSON report, including five real snapshots and
six TRAIN demonstrations. Outputs contain those typed values and fixed pending
cells. This validates internal consistency, not external historical authenticity;
there is no IO, learner update, environment, budget renewal or outcome selection.
"""

from dataclasses import asdict, dataclass
from math import isclose, isfinite, log
from typing import Any, cast

from src.adapters.invariant_action_memory_learner import (
    InvariantMemoryPolicy,
    InvariantMemorySnapshot,
    copy_invariant_memory_snapshot,
)
from src.core.action_memory import ActionMemoryEntry
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.fifo_action_memory import FifoActionMemoryPolicy, FifoActionMemorySnapshot
from src.core.minigrid_invariant_encoding import (
    INVARIANT_HISTORY_ENCODING_ID,
    canonicalize_public_history,
)
from src.core.minigrid_key_door import predefined_layouts

POINTS = (0, 32, 64, 96, 128)
PAIRS = tuple((context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red"))
Slot = tuple[int, str, str]
METHOD = "public_invariant_observation_memory_fifo"


@dataclass(frozen=True)
class SavedFifoInputs:
    snapshots: tuple[InvariantMemorySnapshot, ...]
    episodes: tuple[DemonstrationEpisode, ...]
    score_slots: tuple[Slot, ...]
    game_slots: tuple[Slot, ...]
    completed_games: tuple[Slot, ...]
    censored_games: tuple[Slot, ...]
    origin_status: str
    origin_reason: str | None
    original_elapsed_seconds: float
    original_training_updates: int
    original_training_rows: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"saved FIFO evidence: {message}")


def _record(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    _require(type(value) is dict and set(value) == fields, f"invalid exact {name} fields")
    return cast(dict[str, Any], value)


def _count(value: Any, expected: int, name: str) -> None:
    _require(type(value) is int and value == expected, f"invalid {name}")


def _number(value: Any, name: str) -> float:
    _require(type(value) in (int, float) and isfinite(value), f"nonfinite/nonnumeric {name}")
    return float(value)


def _near(value: Any, expected: float, name: str) -> None:
    _require(isclose(_number(value, name), expected, abs_tol=1e-12, rel_tol=1e-12), name)


def _policy(value: Any) -> FifoActionMemoryPolicy:
    data = _record(
        value,
        {"method", "feature_dim", "num_classes", "max_entries", "max_payload_bytes"},
        "FIFO policy",
    )
    policy = FifoActionMemoryPolicy(**data)
    _require(
        policy
        == FifoActionMemoryPolicy(
            feature_dim=296, num_classes=7, max_entries=36, max_payload_bytes=87264
        ),
        "requires the original FIFO36/296/K7 policy",
    )
    return policy


def _snapshot(value: Any, point: int) -> InvariantMemorySnapshot:
    data = _record(value, {"policy", "native_state"}, "invariant snapshot")
    outer = _record(data["policy"], {"native_policy", "encoding_id"}, "invariant policy")
    policy = InvariantMemoryPolicy(_policy(outer["native_policy"]), outer["encoding_id"])
    native = _record(
        data["native_state"],
        {
            "policy",
            "entries",
            "completed_updates",
            "training_examples",
            "evictions",
            "retired_examples",
        },
        "native snapshot",
    )
    native_policy = _policy(native["policy"])
    _require(
        type(native["entries"]) is list and len(native["entries"]) <= 36,
        "invalid bounded native entries",
    )
    entries = []
    for value in native["entries"]:
        entry = _record(value, {"features", "counts"}, "entry")
        _require(
            type(entry["features"]) is list and len(entry["features"]) == 296,
            "entry needs 296 full recorded coordinates",
        )
        _require(
            type(entry["counts"]) is list and len(entry["counts"]) == 7,
            "entry needs seven full recorded counts",
        )
        entries.append(ActionMemoryEntry(tuple(entry["features"]), tuple(entry["counts"])))
    owned = copy_invariant_memory_snapshot(
        InvariantMemorySnapshot(
            policy,
            FifoActionMemorySnapshot(
                native_policy,
                tuple(entries),
                native["completed_updates"],
                native["training_examples"],
                native["evictions"],
                native["retired_examples"],
            ),
        )
    )
    _count(owned.completed_updates, point, "snapshot update clock")
    _count(owned.training_examples, 9 * point, "snapshot row clock")
    return owned


def _snapshots(payload: dict[str, Any]) -> tuple[InvariantMemorySnapshot, ...]:
    observations = payload.get("observations")
    _require(
        type(observations) is list and len(observations) == 5,
        "five full checkpoint observations are required",
    )
    observations = cast(list[Any], observations)
    result = []
    for point, data in zip(POINTS, observations, strict=True):
        _require(type(data) is dict and "snapshot" in data, "missing full snapshot")
        _count(data.get("updates"), point, "observation clock")
        state = _snapshot(data["snapshot"], point)
        native = cast(FifoActionMemorySnapshot, state.native_state)
        _count(data.get("native_training_examples"), 9 * point, "observation rows")
        _count(data.get("entries"), len(native.entries), "observation occupancy")
        _count(data.get("logical_payload_bytes"), 2424 * len(native.entries), "observation bytes")
        _count(data.get("evictions"), native.evictions, "observation evictions")
        _count(data.get("retired_examples"), native.retired_examples, "observation retirement")
        _count(data.get("retained_examples"), 9 * point - native.retired_examples, "retained rows")
        _count(data.get("retirement_metadata_bytes"), 16, "retirement metadata")
        _require(
            data.get("model_encoding") == INVARIANT_HISTORY_ENCODING_ID,
            "observation encoding differs",
        )
        result.append(state)
    return tuple(result)


def _episode(value: Any, context: str, cue: str) -> DemonstrationEpisode:
    data = _record(
        value, {"episode_id", "context", "role", "transitions", "num_classes"}, "source episode"
    )
    _require(
        data["episode_id"] == f"minigrid-source-{context}-{cue}"
        and data["context"] == context
        and data["role"] == "train",
        "original source identity/role/order differs",
    )
    _count(data["num_classes"], 7, "source classes")
    _require(
        type(data["transitions"]) is list and len(data["transitions"]) == 9,
        "source requires nine complete transitions",
    )
    transitions = []
    for item in data["transitions"]:
        item = _record(
            item,
            {
                "step",
                "observation",
                "action",
                "reward",
                "next_observation",
                "terminal",
                "num_classes",
            },
            "source transition",
        )
        _require(
            all(
                type(item[key]) is list and len(item[key]) == 148
                for key in ("observation", "next_observation")
            ),
            "source requires raw148",
        )
        transitions.append(
            DemonstrationTransition(
                **{
                    **item,
                    "observation": tuple(item["observation"]),
                    "next_observation": tuple(item["next_observation"]),
                }
            )
        )
    return DemonstrationEpisode(data["episode_id"], context, "train", tuple(transitions), 7)


def _environment(data: Any, context: str, cue: str) -> dict[str, Any]:
    _require(type(data) is dict, "missing environment record")
    layout = next(layout for layout in predefined_layouts()[:3] if layout.context == context)
    _require(
        data.get("context") == context
        and data.get("cue") == cue
        and data.get("role") == "train"
        and data.get("layout") == asdict(layout),
        "environment role/cue/layout binding differs",
    )
    _require(
        data.get("environment_id") == "Circadian-MiniGrid-CueKeyDoor-v1"
        and data.get("encoding_id") == "minigrid_cue_key_door_public_image_direction_148_v1",
        "original environment/encoding identity differs",
    )
    for name, expected in (
        ("seed", 17),
        ("max_steps", 16),
        ("num_classes", 7),
        ("observation_dim", 148),
        ("reset_calls", 1),
    ):
        _count(data.get(name), expected, f"environment {name}")
    _require(data.get("closed") is True, "original environment was not closed")
    return cast(dict[str, Any], data)


def _sources(payload: dict[str, Any]) -> tuple[DemonstrationEpisode, ...]:
    values, reports = payload.get("training_sources"), payload.get("source_reports")
    _require(
        type(values) is list and len(values) == 6 and type(reports) is list and len(reports) == 6,
        "six original TRAIN sources required",
    )
    values, reports = cast(list[Any], values), cast(list[Any], reports)
    result = []
    for value, report, (context, cue) in zip(values, reports, PAIRS, strict=True):
        episode = _episode(value, context, cue)
        report = _environment(report, context, cue)
        _require(
            report.get("episode_id") == episode.episode_id
            and report.get("status") == "complete"
            and report.get("terminated") is True
            and report.get("truncated") is False
            and report.get("failure_work_may_be_unobserved") is False,
            "source lacks complete original collection evidence",
        )
        _count(report.get("environment_steps_attempted"), 9, "source attempted steps")
        _count(report.get("environment_steps"), 9, "source returned steps")
        raw = report.get("transitions")
        _require(type(raw) is list and len(raw) == 9, "source collection transitions missing")
        raw = cast(list[Any], raw)
        for transition, recorded in zip(episode.transitions, raw, strict=True):
            _require(
                type(recorded) is dict
                and recorded.get("attempted") is True
                and recorded.get("applied") is True
                and recorded.get("observation") == list(transition.observation)
                and recorded.get("next_observation") == list(transition.next_observation)
                and type(recorded.get("action")) is int
                and recorded["action"] == transition.action
                and recorded.get("reward") == transition.reward
                and recorded.get("terminated") is transition.terminal
                and recorded.get("truncated") is False,
                "source collection/episode mismatch",
            )
        _near(
            report.get("total_return"), sum(t.reward for t in episode.transitions), "source return"
        )
        result.append(episode)
    return tuple(result)


def _receipts(row: dict[str, Any]) -> None:
    stream = row.get("stream_result")
    _require(type(stream) is dict, "missing original stream receipts")
    stream = cast(dict[str, Any], stream)
    for key, expected in (
        ("completed_updates", 128),
        ("training_examples", 1152),
        ("score_forward_calls", 0),
        ("num_classes", 7),
    ):
        _count(stream.get(key), expected, f"stream {key}")
    _require(
        stream.get("checkpoints") == [] and stream.get("evaluation_scope") == "training_only",
        "stream evaluation isolation differs",
    )
    receipts = stream.get("applied_updates")
    _require(type(receipts) is list and len(receipts) == 128, "128 original receipts required")
    receipts = cast(list[dict[str, Any]], receipts)
    for index, receipt in enumerate(receipts):
        expected_receipt: dict[str, Any] = dict(
            sample_id=f"batch-{index}",
            episode_id=f"minigrid-training-{index}",
            event_id=f"label-{index}",
            actor_version="a0",
            learner_version="c0",
            observed_at=index,
            arrived_at=index + 1,
            applied_at=index + 1,
            update_number=index + 1,
            diagnostic=dict(
                definition="fifo_observation_action_count_increment_rows_v1", value=9.0
            ),
        )
        _require(type(receipt) is dict and receipt == expected_receipt, "original receipt differs")
        for key in ("observed_at", "arrived_at", "applied_at", "update_number"):
            _count(receipt[key], expected_receipt[key], f"receipt {key}")
    partial = row.get("partial_stream")
    if partial is not None:
        _require(
            type(partial) is dict
            and partial.get("applied_updates") == receipts
            and partial.get("current_native_state_unavailable") is False,
            "partial receipt evidence differs",
        )
        for key, expected in (
            ("completed_wake_updates", 128),
            ("native_candidate_updates", 128),
            ("native_candidate_training_examples", 1152),
            ("committed_training_examples", 1152),
        ):
            _count(partial.get(key), expected, f"partial {key}")


def _probabilities(
    state: InvariantMemorySnapshot, features: tuple[float, ...]
) -> tuple[float, ...]:
    key = canonicalize_public_history(features)
    for entry in state.native_state.entries:
        if entry.features == key:
            return tuple(count / sum(entry.counts) for count in entry.counts)
    return (1.0 / 7,) * 7


def _score(
    data: Any, state: InvariantMemorySnapshot, episode: DemonstrationEpisode, context: str, cue: str
) -> None:
    data = _record(
        data,
        {"context", "cue", "ce", "accuracy", "exact_sequence_accuracy", "probabilities"},
        "completed score",
    )
    _require(data["context"] == context and data["cue"] == cue, "score order/identity differs")
    expected = [
        _probabilities(state, episode.transitions[0].observation + transition.observation)
        for transition in episode.transitions
    ]
    _require(
        data["probabilities"] == [list(row) for row in expected],
        "score probabilities differ from complete saved count state",
    )
    actions = [max(range(7), key=row.__getitem__) for row in expected]
    matches = [
        action == transition.action
        for action, transition in zip(actions, episode.transitions, strict=True)
    ]
    ce = (
        -sum(
            log(max(row[transition.action], 1e-12))
            for row, transition in zip(expected, episode.transitions, strict=True)
        )
        / 9
    )
    _near(data["ce"], ce, "score CE differs")
    _near(data["accuracy"], sum(matches) / 9, "score accuracy differs")
    _near(data["exact_sequence_accuracy"], float(all(matches)), "score exactness differs")


def _game(data: Any, state: InvariantMemorySnapshot, point: int, context: str, cue: str) -> bool:
    _require(
        type(data) is dict
        and data.get("context") == context
        and data.get("cue") == cue
        and data.get("source_role") == "train",
        "game identity/role/order differs",
    )
    env = _environment(data.get("environment"), context, cue)
    _require(
        env.get("episode_id") == f"live-{METHOD}-{point}-{context}-{cue}",
        "game checkpoint identity differs",
    )
    status = data.get("status")
    _require(status in ("complete", "failed"), "unexpected recorded game status")
    if status == "failed":
        _require(
            type(data.get("reason")) is str and bool(data["reason"]),
            "censored game requires original failure reason",
        )
        _require(env.get("status") == "running", "censored environment status differs")
    else:
        _require(
            data.get("state_unchanged") is True
            and data.get("reason") is None
            and env.get("status") == "complete"
            and env.get("failure_work_may_be_unobserved") is False,
            "completed game lacks protected completion evidence",
        )
    predictions, transitions = data.get("predictions"), env.get("transitions")
    _require(
        type(predictions) is list
        and 1 <= len(predictions) <= 16
        and type(transitions) is list
        and len(transitions) == len(predictions),
        "completed game requires full bounded decisions/transitions",
    )
    predictions = cast(list[dict[str, Any]], predictions)
    transitions = cast(list[dict[str, Any]], transitions)
    for value in (
        data.get("prediction_calls_attempted"),
        data.get("prediction_calls_returned"),
        env.get("environment_steps_attempted"),
        env.get("environment_steps"),
    ):
        _count(value, len(predictions), "completed game work")
    _require(
        all(type(item) is dict for item in predictions + transitions),
        "game decisions/transitions must be full records",
    )
    initial = predictions[0].get("observation")
    _require(type(initial) is list and len(initial) == 148, "game initial frame missing")
    initial = cast(list[float], initial)
    reward = 0.0
    for index, (prediction, transition) in enumerate(zip(predictions, transitions, strict=True)):
        _require(
            type(prediction) is dict and type(transition) is dict,
            "game decision/transition must be full records",
        )
        observation = prediction.get("observation")
        _require(
            type(observation) is list
            and len(observation) == 148
            and transition.get("observation") == observation
            and prediction.get("features") == initial + observation,
            "game public history binding differs",
        )
        observation = cast(list[float], observation)
        probabilities = _probabilities(state, tuple(initial + observation))
        action = max(range(7), key=probabilities.__getitem__)
        _require(
            prediction.get("probabilities") == list(probabilities)
            and type(prediction.get("action")) is int
            and prediction["action"] == action
            and type(transition.get("action")) is int
            and transition["action"] == action
            and transition.get("attempted") is True
            and transition.get("applied") is True,
            "game raw probability/first-argmax/step evidence differs",
        )
        _require(
            type(transition.get("terminated")) is bool
            and type(transition.get("truncated")) is bool,
            "invalid game terminal flags",
        )
        _require(
            not (transition["terminated"] and transition["truncated"]),
            "game cannot terminate and truncate simultaneously",
        )
        done = transition["terminated"] or transition["truncated"]
        _require(
            done == (status == "complete" and index == len(predictions) - 1),
            "game terminal return disagrees with recorded completion/censoring",
        )
        if index + 1 < len(predictions):
            _require(
                transition.get("next_observation") == predictions[index + 1].get("observation"),
                "game public transitions are discontinuous",
            )
        reward += _number(transition.get("reward"), "game reward")
    _require(
        env.get("terminated") is transitions[-1]["terminated"]
        and env.get("truncated") is transitions[-1]["truncated"],
        "environment ending differs from observed last transition",
    )
    _near(env.get("total_return"), reward, "environment return differs")
    if status == "failed":
        return False
    _require(
        env.get("terminated") is transitions[-1]["terminated"]
        and env.get("truncated") is transitions[-1]["truncated"]
        and type(data.get("success")) is bool
        and data["success"]
        == (transitions[-1]["terminated"] and not transitions[-1]["truncated"] and reward > 0),
        "game terminal/success summaries differ",
    )
    _near(data.get("return"), reward, "game return differs")
    return True


def _slots(
    row: dict[str, Any],
    snapshots: tuple[InvariantMemorySnapshot, ...],
    episodes: tuple[DemonstrationEpisode, ...],
) -> tuple[tuple[Slot, ...], ...]:
    readouts = row.get("readouts")
    _require(type(readouts) is list and len(readouts) == 5, "five original readout slots required")
    readouts = cast(list[dict[str, Any]], readouts)
    scored: set[Slot] = set()
    completed: set[Slot] = set()
    censored: set[Slot] = set()
    for point, state, data in zip(POINTS, snapshots, readouts, strict=True):
        _require(type(data) is dict and data.get("role") == "train", "readout role differs")
        _count(data.get("updates"), point, "readout clock")
        _require(data.get("status") in ("complete", "failed", "not_run"), "readout status differs")
        if data["status"] == "not_run":
            _require(set(data) == {"updates", "role", "status"}, "unrun slot contains hidden work")
            continue
        _require(data.get("method") == METHOD, "readout method differs")
        scores, games = data.get("scores"), data.get("episodes")
        _require(
            type(scores) is list
            and len(scores) <= 6
            and type(games) is list
            and len(games) <= 6
            and all(type(game) is dict for game in games),
            "invalid bounded readout records",
        )
        scores = cast(list[Any], scores)
        games = cast(list[dict[str, Any]], games)
        work = data.get("work")
        _require(type(work) is dict, "readout work missing")
        work = cast(dict[str, Any], work)
        _count(work.get("raw_score_forwards_completed"), len(scores), "paid completed score calls")
        _count(work.get("raw_score_rows_completed"), 9 * len(scores), "paid completed score rows")
        _count(work.get("environment_resets_attempted"), len(games), "paid original game resets")
        _count(
            work.get("environment_resets_completed"), len(games), "returned original game resets"
        )
        _count(
            work.get("live_prediction_calls_attempted"),
            sum(game.get("prediction_calls_attempted", -1) for game in games),
            "paid original prediction calls",
        )
        _count(
            work.get("live_prediction_calls_returned"),
            sum(game.get("prediction_calls_returned", -1) for game in games),
            "returned original prediction calls",
        )
        for score, episode, (context, cue) in zip(
            scores, episodes[: len(scores)], PAIRS[: len(scores)], strict=True
        ):
            _score(score, state, episode, context, cue)
            scored.add((point, context, cue))
        for game, (context, cue) in zip(games, PAIRS[: len(games)], strict=True):
            slot = (point, context, cue)
            (completed if _game(game, state, point, context, cue) else censored).add(slot)
        if data["status"] == "complete":
            _require(
                len(scores) == len(games) == 6
                and data.get("state_unchanged") is True
                and work.get("failure_work_may_be_unobserved") is False,
                "complete readout is not completely protected",
            )
    matrix = tuple((point, context, cue) for point in POINTS for context, cue in PAIRS)
    return (
        tuple(slot for slot in matrix if slot not in scored),
        tuple(slot for slot in matrix if slot not in completed),
        tuple(slot for slot in matrix if slot in completed),
        tuple(slot for slot in matrix if slot in censored),
    )


def decode_saved_fifo_inputs(payload: dict[str, Any]) -> SavedFifoInputs:
    """Validate supplied complete state; do not infer a state from summary metrics."""
    _require(type(payload) is dict, "source must be an exact JSON object")
    protocol = payload.get("protocol")
    _require(
        type(protocol) is dict
        and protocol.get("protocol_id")
        == "minigrid_bounded_invariant_fifo36_history_action_memory_v1"
        and protocol.get("methods") == [METHOD]
        and protocol.get("seed") == 17
        and protocol.get("checkpoint_updates") == list(POINTS)
        and protocol.get("layouts") == [asdict(layout) for layout in predefined_layouts()[:3]]
        and protocol.get("stream")
        == dict(training_only=True, heldout=[], label_delay=1, updates=128),
        "original fixed TRAIN protocol binding differs",
    )
    protocol = cast(dict[str, Any], protocol)
    bound_policy = _record(
        protocol.get("policy"), {"native_policy", "encoding_id"}, "protocol policy"
    )
    _policy(bound_policy["native_policy"])
    _require(
        bound_policy["encoding_id"] == INVARIANT_HISTORY_ENCODING_ID, "protocol encoding differs"
    )
    status, reason = payload.get("status"), payload.get("reason")
    _require(
        status in ("complete", "incomplete") and (reason is None or type(reason) is str),
        "origin completion/failure status missing",
    )
    if status == "incomplete":
        _require(type(reason) is str and bool(reason), "incomplete origin requires its reason")
    elapsed = _number(payload.get("total_seconds"), "original elapsed seconds")
    _require(elapsed >= 0, "original elapsed seconds must be nonnegative")
    methods = payload.get("methods")
    _require(
        type(methods) is list and len(methods) == 1 and type(methods[0]) is dict,
        "one original method row required",
    )
    methods = cast(list[dict[str, Any]], methods)
    row = methods[0]
    _require(
        row.get("method") == METHOD
        and row.get("status") == status
        and row.get("source_unchanged") is True
        and row.get("actor_unchanged") is True
        and row.get("checkpoint_updates") == list(POINTS)
        and row.get("final_evaluation")
        == dict(status="not_applicable", role="final_test", reason="no final data in this control"),
        "original source/actor/final-isolation proofs missing",
    )
    for key, expected in (
        ("charged_updates", 128),
        ("native_training_examples", 1152),
        ("wake_calls_attempted", 128),
        ("wake_calls_returned", 128),
    ):
        _count(row.get(key), expected, f"original {key}")
    _receipts(row)
    snapshots, episodes = _snapshots(payload), _sources(payload)
    score_slots, game_slots, completed, censored = _slots(row, snapshots, episodes)
    return SavedFifoInputs(
        snapshots,
        episodes,
        score_slots,
        game_slots,
        completed,
        censored,
        cast(str, status),
        reason,
        elapsed,
        128,
        1152,
    )
