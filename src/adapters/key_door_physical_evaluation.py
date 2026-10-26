"""Read out frozen policies at distinct physical boundaries and native origins.

Inputs are five owned checkpoints, fixed final packets and forty planned live
slots. Existing scoring/live helpers own prediction and episode execution. This
adapter keeps physical exposure separate from native training indices, preserves
paid partial results, and refuses changed preceding evidence. No training,
selection, environment construction or new scoring mathematics lives here.
"""

from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable

from src.adapters import key_door_pilot as shared
from src.app.coupled_attribution import _same_value, _score_checkpoints, _validate_batch
from src.core.key_door_navigation import KeyDoorLayout, key_door_layout_record

_BOUNDARIES = (0, 24, 48, 72, 96)
_SLOT_KEYS = (
    "checkpoint_updates",
    "checkpoint_physical_steps",
    "context",
    "cue",
    "layout_partition",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _origin(value):
    _require(
        type(value) is tuple
        and len(value) == 2
        and all(type(count) is int and count >= 0 for count in value),
        "native_origin must be an exact pair of nonnegative native integers",
    )
    _require(
        value[1] == 8 * value[0],
        "native origin rows must match the complete eight-row TRAIN prefix",
    )
    # The existing scorer bounds explicit training-context histories to 512.
    _require(value[0] <= 512, "native origin exceeds the existing 512-context scoring bound")
    return value


def _prefix_contexts(row, updates):
    if updates == 0:
        return None
    training = row.get("training")
    _require(
        type(training) is list and len(training) >= updates,
        "nonzero native origin requires its original TRAIN prefix contexts",
    )
    contexts = []
    for index, slot in enumerate(training[:updates]):
        _require(
            type(slot) is dict
            and type(slot.get("index")) is int
            and slot["index"] == index
            and slot.get("phase") == "demonstration"
            and slot.get("status") == "complete"
            and slot.get("context") == "A"
            and type(slot.get("cue")) is int
            and slot["cue"] == (1 if index % 2 == 0 else -1),
            "native origin requires the complete ordered A-only TRAIN prefix",
        )
        contexts.append(f"{slot['context']}-{'red' if slot['cue'] == 1 else 'blue'}")
    return tuple(contexts)


def _admit(row, states, final, declared_layouts, check_boundary, native_origin):
    _require(callable(check_boundary), "physical evaluation requires a boundary check")
    _require(
        type(states) is list and len(states) == 5, "physical evaluation requires five checkpoints"
    )
    for pair in states:
        _require(
            type(pair) is tuple
            and len(pair) == 2
            and type(pair[0]) is int
            and pair[0] == native_origin[0],
            "frozen physical checkpoints must retain native index zero"
            if native_origin == (0, 0)
            else "frozen physical checkpoints must retain their native origin index",
        )
        for name, expected_value in zip(("completed_updates", "training_examples"), native_origin):
            value = getattr(pair[1], name, None)
            _require(
                type(value) is int and value == expected_value,
                "frozen native counters must remain zero"
                if native_origin == (0, 0)
                else "frozen native counters must remain at their original absolute values",
            )
        _require(
            _same_value(pair[1], states[0][1]),
            "frozen checkpoint state changed across physical exposure",
        )
    expected = [
        dict(physical_steps=point, native_updates=native_origin[0]) for point in _BOUNDARIES
    ]
    _require(
        type(row) is dict and row.get("quota_checkpoints") == expected,
        "physical evaluation requires the five exact original quota boundaries",
    )
    for point in row["quota_checkpoints"]:
        _require(
            type(point) is dict
            and type(point.get("physical_steps")) is int
            and type(point.get("native_updates")) is int,
            "quota boundaries require native integers",
        )
    _require(
        row.get("method") in ("recurrent_bc", "recurrent_rl")
        and row.get("replay_enabled") is False,
        "frozen physical evaluation requires the original recurrent no-replay identity",
    )
    _require(
        type(row.get("live_sessions_started")) is int and row["live_sessions_started"] == 0,
        "physical evaluation requires an unused live session counter",
    )
    _require(
        type(declared_layouts) is tuple
        and len(declared_layouts) == 4
        and all(type(layout) is KeyDoorLayout for layout in declared_layouts),
        "invalid declared layouts",
    )
    owned = tuple(replace(layout) for layout in declared_layouts)
    _require(
        tuple(layout.layout_id for layout in owned) == ("A", "B", "C", "D"), "invalid layout order"
    )
    slots = row.get("episodes")
    _require(
        type(slots) is list and len(slots) == 40, "physical evaluation requires forty live slots"
    )
    for stage, physical in enumerate(_BOUNDARIES):
        for index, (layout, cue) in enumerate((layout, cue) for layout in owned for cue in (1, -1)):
            slot = slots[8 * stage + index]
            _require(
                type(slot) is dict
                and type(slot.get("checkpoint_updates")) is int
                and slot["checkpoint_updates"] == native_origin[0]
                and type(slot.get("checkpoint_physical_steps")) is int
                and slot["checkpoint_physical_steps"] == physical
                and slot.get("context") == layout.layout_id
                and type(slot.get("cue")) is int
                and slot["cue"] == cue
                and slot.get("layout_partition")
                == ("heldout_layout" if layout.layout_id == "D" else "known_layout")
                and slot.get("status") == "not_run",
                "invalid physical live slot",
            )
    _require(
        type(final) is tuple and len(final) == 8, "physical evaluation requires eight final packets"
    )
    seen: set[str] = set()
    for batch, (layout, cue) in zip(final, ((layout, cue) for layout in owned for cue in (1, -1))):
        _validate_batch(batch, "final_test", seen)
        _require(
            batch.context == f"{layout.layout_id}-{'red' if cue == 1 else 'blue'}"
            and batch.features.shape == (8, 8)
            and batch.num_classes == 5,
            "invalid final packet schema",
        )
    return owned, _prefix_contexts(row, native_origin[0])


def _publish_scores(result, progress, physical):
    result["score_forward_calls"] += progress.get("prediction_calls_completed", 0)
    for point in progress.get("checkpoints", []):
        result["checkpoints"].append(dict(deepcopy(point), physical_steps=physical))


def _outcomes(checkpoints) -> dict[str, float]:
    by_physical = {point["physical_steps"]: point["scores"] for point in checkpoints}

    def mean(physical, context):
        return (
            sum(by_physical[physical][f"{context}-{color}"]["ce"] for color in ("red", "blue")) / 2
        )

    return dict(
        new_B_gain=mean(0, "B") - mean(48, "B"),
        new_C_gain=mean(0, "C") - mean(72, "C"),
        retention_loss=mean(72, "A") - mean(24, "A"),
        return_gain=mean(72, "A") - mean(96, "A"),
    )


def evaluate_frozen_physical_checkpoints(
    row: dict[str, Any],
    source: Any,
    states: list[tuple[int, Any]],
    final: Any,
    budget: Any,
    *,
    declared_layouts: tuple[KeyDoorLayout, ...],
    check_boundary: Callable[[], None],
    native_origin: tuple[int, int] = (0, 0),
) -> None:
    """Retain five physical groups without relabeling their absolute native clock.

    Nonzero origins require original completed A-only prefix contexts in
    row.training. The caller supplies sealed native checkpoints and preserves
    the executed prefix's own receipts separately; this helper performs no SGD.
    """
    native_origin = _origin(native_origin)
    owned_layouts, training_contexts = _admit(
        row, states, final, declared_layouts, check_boundary, native_origin
    )
    prefix_training = row.get("training") if training_contexts is not None else None
    episodes = row["episodes"]
    identities = [{key: deepcopy(slot[key]) for key in _SLOT_KEYS} for slot in episodes]
    expected_episodes = deepcopy(episodes)
    expected_states, expected_final = deepcopy(states), deepcopy(final)
    expected_boundaries = deepcopy(row["quota_checkpoints"])
    expected_layouts = [key_door_layout_record(layout) for layout in declared_layouts]
    result: dict[str, Any] = dict(
        completed_updates=native_origin[0],
        training_examples=native_origin[1],
        score_forward_calls=0,
        checkpoints=[],
        applied_updates=[],
        num_classes=5,
    )
    progress_rows: list[dict[str, Any]] = [
        dict(physical_steps=physical, status="not_run", reason="not reached", score_progress=None)
        for physical in _BOUNDARIES
    ]
    row.update(stream_result=result, physical_evaluation_progress=progress_rows, outcomes=None)
    expected_result = deepcopy(result)
    expected_progress = deepcopy(progress_rows)
    expected_live_count = 0
    active_stage = 0
    identity = (row["method"], row["replay_enabled"])

    def verify() -> None:
        if training_contexts is not None:
            _require(
                row.get("training") is prefix_training
                and _prefix_contexts(row, native_origin[0]) == training_contexts,
                "original TRAIN prefix context binding changed",
            )
        _require(
            row.get("episodes") is episodes
            and row.get("stream_result") is result
            and row.get("physical_evaluation_progress") is progress_rows,
            "physical evaluation evidence binding changed",
        )
        _require(
            _same_value((row.get("method"), row.get("replay_enabled")), identity)
            and type(row.get("live_sessions_started")) is int
            and row.get("live_sessions_started") == expected_live_count,
            "physical evaluation identity or observed session count changed",
        )
        _require(
            _same_value(row.get("quota_checkpoints"), expected_boundaries)
            and _same_value(states, expected_states)
            and _same_value(final, expected_final)
            and [key_door_layout_record(layout) for layout in declared_layouts] == expected_layouts,
            "physical evaluation source evidence changed",
        )
        _require(_same_value(result, expected_result), "preceding physical score evidence changed")
        _require(len(episodes) == 40 and len(progress_rows) == 5, "physical evidence shape changed")
        for index, slot in enumerate(episodes):
            _require(
                type(slot) is dict
                and _same_value({key: slot.get(key) for key in _SLOT_KEYS}, identities[index]),
                "physical episode identity changed",
            )
            if index // 8 != active_stage:
                _require(
                    _same_value(slot, expected_episodes[index]),
                    "preceding or unreached live evidence changed",
                )
        _require(
            _same_value(progress_rows, expected_progress),
            "physical score progress changed outside observed publication",
        )

    def boundary() -> None:
        verify()
        check_boundary()
        verify()

    for stage, (physical, state) in enumerate(zip(_BOUNDARIES, states)):
        active_stage = stage
        entry = progress_rows[stage]
        entry.update(status="running", reason=None)
        expected_progress = deepcopy(progress_rows)
        # Why: callbacks may inspect published evidence. Keeping the active
        # ledger local prevents them from editing paid counters or returned
        # scores before we publish an independently owned readback.
        progress: dict[str, Any] = {}
        try:
            boundary()
            try:
                _score_checkpoints(
                    source,
                    [state],
                    (),
                    final,
                    budget,
                    verify_state=True,
                    check_boundary=boundary,
                    score_progress=progress,
                    **(
                        {"training_contexts": training_contexts}
                        if training_contexts is not None
                        else {}
                    ),
                )
            finally:
                progress_changed = not _same_value(progress_rows, expected_progress)
                if progress_changed:
                    row["physical_evaluation_progress_before_publish"] = deepcopy(progress_rows)
                entry["score_progress"] = deepcopy(progress)
                _publish_scores(result, progress, physical)
                expected_result = deepcopy(result)
                expected_progress = deepcopy(progress_rows)
                _require(not progress_changed, "physical score progress changed before publication")
            _require(
                progress.get("status") == "complete"
                and progress.get("prediction_calls_completed") == 8
                and len(progress.get("checkpoints", [])) == 1,
                "incomplete physical score stage",
            )
            boundary()
            view = dict(row, episodes=episodes[8 * stage : 8 * stage + 8])
            try:
                shared._live(
                    view,
                    source,
                    [state],
                    budget,
                    check_boundary=boundary,
                    declared_layouts=owned_layouts,
                    condition_on_legal_actions=True,
                )
            finally:
                counter_changed = (
                    type(row.get("live_sessions_started")) is not int
                    or row["live_sessions_started"] != expected_live_count
                )
                if counter_changed:
                    row["physical_evaluation_live_counter_before_copyback"] = deepcopy(
                        row.get("live_sessions_started")
                    )
                row["live_sessions_started"] = view["live_sessions_started"]
                expected_live_count = view["live_sessions_started"]
                _require(
                    not counter_changed,
                    "parent live session counter changed outside observed sessions",
                )
            _require(
                type(expected_live_count) is int
                and expected_live_count == 8 * (stage + 1)
                and all(
                    slot.get("status") == "complete" and slot.get("state_unchanged") is True
                    for slot in episodes[8 * stage : 8 * stage + 8]
                ),
                "incomplete physical live stage",
            )
            entry.update(status="complete", reason=None)
            expected_episodes = deepcopy(episodes)
            expected_progress = deepcopy(progress_rows)
            boundary()
        except Exception as error:
            entry.update(status="failed", reason=f"{type(error).__name__}: {error}")
            raise
    outcomes = _outcomes(result["checkpoints"])
    boundary()
    row["outcomes"] = outcomes
