"""Validate saved key/door reports and derive bounded recorded-action playback.

Inputs are one saved return-policy report or its off/on envelope and a source
label. Outputs retain the original JSON and exact episode pointers, with derived
screen coordinates and explicit unknown metadata. No environment, learner, IO,
prediction or training is constructed or executed here. Board geometry is a
derived overview; it is not a global observation supplied to the recorded policy.
"""

from copy import deepcopy
import json
from math import isfinite
from typing import Any, cast

from src.app.acquired_inspector_validation import PROTOCOL_CONTROLS, validate_acquired_report
from src.core.key_door_navigation import KeyDoorLayout


_STATUSES = {"complete", "failed", "incomplete", "not_run", "running", "pending"}
_DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
_TRANSITION_FIELDS = {
    "observation",
    "next_observation",
    "legal_actions",
    "action",
    "probabilities",
    "applied",
    "reward",
    "terminal",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, minimum: int, maximum: int, name: str) -> int:
    _require(type(value) is int and minimum <= value <= maximum, f"invalid {name}")
    return value


def _number(value: Any, name: str) -> float:
    _require(type(value) in (int, float), f"invalid {name}")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"nonfinite {name}") from error
    _require(isfinite(number), f"nonfinite {name}")
    return number


def _text(value: Any, name: str, limit: int = 128) -> str:
    _require(
        type(value) is str
        and bool(value)
        and value == value.strip()
        and len(value.encode("utf-8")) <= limit,
        f"invalid {name}",
    )
    return value


def _status(value: Any) -> bool:
    return type(value) is str and value in _STATUSES


def _optional_object(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    _require(type(value) is dict, f"{name} must be an object or null")
    return cast(dict[str, Any], value)


def _records(value: Any, name: str, limit: int = 32) -> list[dict[str, Any]]:
    if value is None:
        return []
    _require(type(value) is list and len(value) <= limit, f"invalid bounded {name} array")
    _require(all(type(row) is dict for row in value), f"{name} entries must be objects")
    return cast(list[dict[str, Any]], value)


def _present_numbers(
    record: dict[str, Any], names: tuple[str, ...], *, count: bool = False
) -> None:
    for name in names:
        value = record.get(name)
        if value is not None:
            if count:
                _require(type(value) is int and value >= 0, f"invalid recorded {name} count")
            else:
                _number(value, name)


def _metric_values(record: dict[str, Any]) -> None:
    _present_numbers(record, ("ce", "bce", "mean_ce", "mean_return"))
    for name in ("accuracy", "mean_accuracy", "exact_sequence_accuracy", "success_rate"):
        if record.get(name) is not None:
            _require(0 <= _number(record[name], name) <= 1, f"invalid {name} range")
    for name in ("ce", "bce", "mean_ce"):
        if record.get(name) is not None:
            _require(record[name] >= 0, f"negative {name}")


def _probe_readouts(condition: dict[str, Any]) -> None:
    for slot in _records(condition.get("probes"), "probes"):
        _present_numbers(
            slot,
            (
                "checkpoint_updates",
                "native_updates_before",
                "native_training_examples_before",
                "charged_updates",
            ),
            count=True,
        )
        if slot.get("status") is not None:
            _require(_status(slot["status"]), "invalid probe status")
        report = _optional_object(slot.get("report"), "probe report")
        _present_numbers(
            report,
            (
                "reserved_data_bytes",
                "completed_updates",
                "completed_training_examples",
                "score_forward_calls_completed",
            ),
            count=True,
        )
        for point in _records(report.get("checkpoints"), "probe checkpoints"):
            _present_numbers(point, ("updates",), count=True)
            scores = _optional_object(point.get("scores"), "probe scores")
            _require(len(scores) <= 16, "too many probe score contexts")
            for metrics in scores.values():
                _metric_values(_optional_object(metrics, "probe context scores"))
            details = _optional_object(point.get("decision_diagnostics"), "decision diagnostics")
            _require(len(details) <= 16, "too many decision diagnostic contexts")
            for diagnostic in details.values():
                diagnostic = _optional_object(diagnostic, "context decision diagnostics")
                for decision in _records(diagnostic.get("decisions"), "probe decisions", 128):
                    _require(
                        decision.get("correct") is None or type(decision["correct"]) is bool,
                        "invalid decision correctness",
                    )


def _learning_readouts(condition: dict[str, Any]) -> None:
    for point in _records(condition.get("checkpoints"), "training checkpoints"):
        _present_numbers(point, ("updates",), count=True)
        _metric_values(point)
        _present_numbers(
            _optional_object(point.get("clocks"), "checkpoint clocks"),
            ("native_updates",),
            count=True,
        )
    evaluation = _optional_object(condition.get("policy_evaluation"), "policy evaluation")
    for point in _records(evaluation.get("outcomes"), "final outcomes"):
        _present_numbers(point, ("updates",), count=True)
        layouts = _optional_object(point.get("by_layout"), "outcomes by layout")
        _require(len(layouts) <= 32, "too many outcome layouts")
        for summary in layouts.values():
            _metric_values(_optional_object(summary, "layout outcomes"))


def _resource_readouts(condition: dict[str, Any]) -> None:
    _present_numbers(condition, ("charged_updates",), count=True)
    _present_numbers(condition, ("total_seconds",))
    _present_numbers(
        _optional_object(condition.get("partial_stream"), "partial stream"),
        ("native_candidate_updates", "native_candidate_training_examples"),
        count=True,
    )
    _present_numbers(
        _optional_object(condition.get("work"), "work"), ("score_forwards_completed",), count=True
    )
    evaluation = _optional_object(condition.get("policy_evaluation"), "policy evaluation")
    _present_numbers(
        _optional_object(evaluation.get("work"), "evaluation work"),
        ("cached_forwards_completed", "live_callback_returns"),
        count=True,
    )
    _present_numbers(
        _optional_object(condition.get("observed_process_rss"), "process RSS"),
        ("peak_bytes",),
        count=True,
    )
    replay = _optional_object(condition.get("replay"), "replay")
    _present_numbers(replay, ("charged_examples",), count=True)
    rehearsal = _optional_object(replay.get("rehearsal"), "rehearsal")
    _present_numbers(rehearsal, ("committed_updates",), count=True)
    _present_numbers(
        _optional_object(replay.get("final_retention"), "final retention"),
        ("example_count", "retained_bytes"),
        count=True,
    )
    _present_numbers(
        _optional_object(rehearsal.get("policy"), "replay policy"),
        ("every_updates", "seed"),
        count=True,
    )
    for event in _records(rehearsal.get("events"), "replay events", 512):
        _present_numbers(event, ("updates", "charged_exposures", "committed_updates"), count=True)
        for selected in _records(event.get("selected_episodes"), "selected episodes", 128):
            if selected.get("context") is not None:
                _text(selected["context"], "selected episode context", 32)


def _json_envelope(report: Any) -> None:
    _require(type(report) is dict, "report must be an exact JSON object")
    stack = [(report, 0)]
    nodes = 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        _require(nodes <= 500000 and depth <= 64, "report exceeds bounded JSON structure")
        if type(value) is dict:
            _require(all(type(key) is str for key in value), "JSON object keys must be strings")
            stack.extend((child, depth + 1) for child in value.values())
        elif type(value) is list:
            stack.extend((child, depth + 1) for child in value)
        elif type(value) in (int, float):
            _number(value, "JSON number")
        else:
            _require(type(value) in (str, bool, type(None)), "report contains non-JSON values")
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    _require(len(encoded.encode("utf-8")) <= 8 * 1024**2, "report exceeds 8 MiB")


def _observation(value: Any) -> list[float]:
    _require(type(value) is list and len(value) == 8, "observation must contain eight numbers")
    return [_number(number, "observation") for number in value]


def _rotated(position: tuple[int, int], turns: int) -> list[int]:
    x, y = position
    for _ in range(turns):
        x, y = 4 - y, x
    return [x, y]


def _tile(position: tuple[int, int], board: dict[str, Any], layout: KeyDoorLayout) -> float:
    if position == (2, 0):
        return 1.0 if board["door"] else 0.75
    if position in board["keys"]:
        return board["keys"][position] * 0.5
    if position[0] == 2 and 1 <= position[1] <= 4:
        return 0.0
    if position in ((1, layout.branch_row), (3, layout.branch_row)):
        return 0.0
    return -1.0


def _neighbor(board: dict[str, Any], action: int, rotation: int) -> tuple[int, int]:
    dx, dy = _DIRECTIONS[(action - rotation) % 4]
    return board["position"][0] + dx, board["position"][1] + dy


def _visible(board: dict[str, Any], layout: KeyDoorLayout) -> list[float]:
    return [
        *(_tile(_neighbor(board, action, layout.rotation), board, layout) for action in range(4)),
        _tile(board["position"], board, layout),
        float(board["held"]),
        float(board["cue"]),
        layout.appearance,
    ]


def _legal(board: dict[str, Any], layout: KeyDoorLayout) -> list[int]:
    actions = [
        action
        for action in range(4)
        if _tile(
            _neighbor(board, layout.movement_action_map[action], layout.rotation), board, layout
        )
        not in (-1.0, 0.75)
    ]
    if (board["position"] in board["keys"] and not board["held"]) or (
        board["position"] == (2, 1) and not board["door"] and board["held"]
    ):
        actions.append(4)
    return actions


def _frame(
    board: dict[str, Any], layout: KeyDoorLayout, step: int, observation: list[float]
) -> dict[str, Any]:
    return dict(
        step=step,
        position=_rotated(board["position"], layout.rotation),
        observation=observation.copy(),
        held_color=board["held"],
        visible_cue=board["cue"],
        door_open=board["door"],
        remaining_keys=[
            dict(position=_rotated(position, layout.rotation), color=color)
            for position, color in sorted(board["keys"].items())
        ],
        origin="derived_from_recorded_applied_actions",
        observation_origin="recorded_transition_observation"
        if step == 0
        else "recorded_transition_next_observation",
    )


def _identity(record: dict[str, Any]) -> None:
    identity = record.get("identity")
    _require(
        type(identity) is dict
        and set(identity)
        == {
            "method",
            "seed",
            "actor_version",
            "task_sequence",
            "checkpoint_updates",
            "replay_enabled",
        },
        "invalid recorded evaluation identity",
    )
    identity = cast(dict[str, Any], identity)
    sequence = identity["task_sequence"]
    _require(type(sequence) is list, "identity task_sequence must be a JSON array")
    # Why: non-game report routes must not import the NumPy evaluation runtime.
    from src.app.line_policy_evaluation import EvaluationIdentity

    admitted = EvaluationIdentity(
        identity["method"],
        identity["seed"],
        identity["actor_version"],
        tuple(sequence),
        identity["checkpoint_updates"],
        identity["replay_enabled"],
    )
    _require(
        admitted.checkpoint_updates == record["checkpoint_updates"], "checkpoint identity mismatch"
    )
    _require(
        (record["context"] in sequence) == (record["layout_partition"] == "known_layout"),
        "layout partition mismatch",
    )
    _require(
        record.get("execution_origin") == "live_environment"
        and record.get("checkpoint_origin") == "owned_native_checkpoint"
        and record.get("phase") == "evaluation"
        and record.get("learning_active") is False,
        "invalid recorded execution origin",
    )


def _apply(
    record: dict[str, Any], transition: dict[str, Any], board: dict[str, Any], layout: KeyDoorLayout
) -> tuple[float, bool]:
    action = transition["action"]
    board["cue"] = 0
    if action != 4:
        board["position"] = _neighbor(board, layout.movement_action_map[action], layout.rotation)
        return (1.0, True) if board["position"] == (2, 0) else (0.0, False)
    if board["position"] in board["keys"]:
        board["held"] = board["keys"].pop(board["position"])
    elif board["held"] == record["cue"]:
        board["door"] = True
    else:
        return -1.0, True
    return 0.0, False


def _decision(transition: dict[str, Any], legal: list[int], *, partial: bool) -> bool:
    _require(
        type(transition["applied"]) is bool and type(transition["terminal"]) is bool,
        "invalid transition flags",
    )
    action, probabilities = transition["action"], transition["probabilities"]
    if action is None or probabilities is None:
        _require(
            partial and action is None and probabilities is None and not transition["applied"],
            "unresolved decision outside partial final transition",
        )
        return False
    _integer(action, 0, 4, "action")
    _require(
        type(probabilities) is list and len(probabilities) == 5,
        "probabilities must contain five values",
    )
    values = [_number(value, "probability") for value in probabilities]
    _require(
        all(0 <= value <= 1 for value in values) and abs(sum(values) - 1) <= 1e-6,
        "invalid categorical probabilities",
    )
    _require(
        action == max(range(5), key=values.__getitem__), "action differs from argmax first tie"
    )
    _require(not transition["applied"] or action in legal, "applied action is illegal")
    _require(
        transition["applied"] or action not in legal or partial, "legal decision was not applied"
    )
    return True


def _train_decision(transition: dict[str, Any], legal: list[int], *, partial: bool) -> None:
    raw = transition["raw_probabilities"]
    if raw is None:
        _require(
            partial and transition["action"] is None and transition["probabilities"] is None,
            "unobserved TRAIN prediction has a decision",
        )
        return
    _require(type(raw) is list and len(raw) == 5, "invalid TRAIN raw probabilities")
    owned = tuple(raw)
    from src.core.legal_policy_gradient import condition_action_probabilities

    condition_action_probabilities(owned, tuple(range(5)))
    if sum(owned[action] for action in legal) == 0:
        _require(
            partial and transition["action"] is None and transition["probabilities"] is None,
            "zero legal mass outside partial TRAIN refusal",
        )
        return
    expected = condition_action_probabilities(owned, tuple(legal))
    probabilities = transition["probabilities"]
    _require(
        type(probabilities) is list
        and len(probabilities) == 5
        and all(
            abs(_number(actual, "TRAIN conditional probability") - value) <= 1e-12
            for actual, value in zip(probabilities, expected, strict=True)
        )
        and all(probabilities[action] == 0 for action in range(5) if action not in legal)
        and transition["action"] == max(range(5), key=expected.__getitem__),
        "TRAIN conditional law or argmax mismatch",
    )


def _playback(
    record: dict[str, Any], layout: KeyDoorLayout, *, train: bool = False
) -> list[dict[str, Any]]:
    transitions = record.get("transitions")
    _require(type(transitions) is list and len(transitions) <= 32, "invalid bounded transitions")
    transitions = cast(list[Any], transitions)
    maximum = _integer(record.get("max_steps"), 1, 32, "max_steps")
    _require(len(transitions) <= maximum, "too many attempted transitions")
    left = -1 if layout.swap_keys else 1
    board: dict[str, Any] = dict(
        position=(2, 4),
        held=0,
        cue=record["cue"],
        door=False,
        keys={(1, layout.branch_row): left, (3, layout.branch_row): -left},
    )
    frames: list[dict[str, Any]] = []
    total, applied, returned, terminal = 0.0, 0, 0, False
    # Both TRAIN diagnostics and explicit legal LIVE policies retain the raw law.
    conditional = train or record.get("action_conditioning") == "legal_actions_v1"
    for index, transition in enumerate(transitions):
        _require(
            type(transition) is dict
            and set(transition)
            == (_TRANSITION_FIELDS | {"raw_probabilities"} if conditional else _TRANSITION_FIELDS),
            "invalid transition fields",
        )
        _require(not terminal, "transition follows terminal outcome")
        observed = _observation(transition["observation"])
        _require(
            observed == _visible(board, layout),
            "observation continuity or recorded geometry mismatch",
        )
        if not frames:
            frames.append(_frame(board, layout, 0, observed))
        legal = transition["legal_actions"]
        _require(
            type(legal) is list
            and all(type(action) is int for action in legal)
            and legal == _legal(board, layout),
            "recorded legal actions mismatch",
        )
        partial = record["status"] != "complete" and index == len(transitions) - 1
        if conditional:
            _train_decision(transition, legal, partial=partial)
        returned += int(_decision(transition, legal, partial=partial))
        reward, ended = (
            _apply(record, transition, board, layout) if transition["applied"] else (0.0, False)
        )
        _require(
            _number(transition["reward"], "reward") == reward and transition["terminal"] is ended,
            "transition reward or terminal mismatch",
        )
        following = _observation(transition["next_observation"])
        _require(following == _visible(board, layout), "next observation mismatch")
        frames.append(_frame(board, layout, index + 1, following))
        applied += int(transition["applied"])
        total += reward
        terminal = ended
        _require(
            transition["applied"] or index == len(transitions) - 1,
            "transition follows unapplied decision",
        )
    _readout(record, transitions, maximum, total, applied, returned, terminal)
    return frames


def _readout(
    record: dict[str, Any],
    transitions: list[Any],
    maximum: int,
    total: float,
    applied: int,
    returned: int,
    terminal: bool,
) -> None:
    _require(
        _integer(record.get("steps_executed"), 0, 32, "steps_executed") == applied,
        "applied step count mismatch",
    )
    attempted = _integer(
        record.get("prediction_calls_attempted"), 0, 32, "prediction_calls_attempted"
    )
    completed = _integer(
        record.get("prediction_calls_returned"), 0, 32, "prediction_calls_returned"
    )
    _require(
        attempted == len(transitions)
        and returned <= completed <= attempted
        and completed - returned <= 1,
        "prediction counters mismatch",
    )
    _require(_number(record.get("return"), "return") == total, "recorded return mismatch")
    _require(
        type(record.get("terminal")) is bool and record["terminal"] is terminal,
        "recorded terminal mismatch",
    )
    outcome = record.get("outcome")
    _require(
        outcome in (None, "illegal_action", "step_limit", "success", "wrong_key"), "invalid outcome"
    )
    if outcome is not None or record["status"] == "complete":
        expected = (
            "success"
            if total == 1
            else "wrong_key"
            if total == -1
            else "illegal_action"
            if transitions
            and transitions[-1]["action"] is not None
            and not transitions[-1]["applied"]
            else "step_limit"
        )
        _require(outcome == expected, "recorded outcome mismatch")
        if expected == "step_limit":
            _require(applied == maximum, "step_limit precedes step budget")
        _require(completed == returned == attempted, "completed episode has unresolved predictions")
    _require(
        type(record.get("success")) is bool and record["success"] is (outcome == "success"),
        "recorded success mismatch",
    )


def _episode(
    record: Any, pointer: str, *, train: bool = False, public_history: bool = False
) -> dict[str, Any]:
    _require(type(record) is dict, "episode slot must be a JSON object")
    _require(
        record.get("action_conditioning") in (None, "legal_actions_v1"),
        "unknown recorded action conditioning",
    )
    _require(not (train and public_history), "TRAIN episode selectors cannot combine")
    if public_history:
        _require(record.get("source_role") == "train", "public history requires TRAIN episodes")
        _require(record.get("action_conditioning") is None, "public history requires raw actions")
    _require(
        train or public_history or record.get("source_role") != "train",
        "TRAIN episode requires its original readout selector",
    )
    _integer(record.get("checkpoint_updates"), 0, 512, "checkpoint_updates")
    _text(record.get("context"), "context", 32)
    _require(type(record.get("cue")) is int and record["cue"] in (-1, 1), "invalid cue")
    _require(
        record.get("layout_partition") in ("known_layout", "heldout_layout"),
        "invalid layout partition",
    )
    _require(_status(record.get("status")), "invalid episode status")
    _require(
        record.get("reason") is None or type(record["reason"]) is str, "invalid episode reason"
    )
    _require(
        record.get("state_unchanged") is None or type(record["state_unchanged"]) is bool,
        "invalid state observation",
    )
    _present_numbers(
        record,
        ("steps_executed", "prediction_calls_attempted", "prediction_calls_returned"),
        count=True,
    )
    _present_numbers(record, ("return",))
    for flag in ("terminal", "success", "learning_active"):
        _require(record.get(flag) is None or type(record[flag]) is bool, f"invalid {flag}")
    for name, expected in (("num_classes", 5), ("observation_dim", 8)):
        if record.get(name) is not None:
            _require(type(record[name]) is int and record[name] == expected, f"invalid {name}")
    if record.get("max_steps") is not None:
        _integer(record["max_steps"], 1, 32, "max_steps")
    layout = record.get("layout")
    frames = []
    if layout is None:
        if record.get("transitions") is not None:
            _require(
                type(record["transitions"]) is list, "transitions must be a JSON array or null"
            )
        _require(
            record["status"] != "complete" and not record.get("transitions"),
            "complete or observed episode requires recorded layout",
        )
        # Why this: an unobserved slot may omit data, but present success/work
        # claims must not survive without the recorded layout and transitions.
        for counter in (
            "steps_executed",
            "prediction_calls_attempted",
            "prediction_calls_returned",
        ):
            _require(record.get(counter) in (None, 0), "unobserved slot reports executed work")
        _require(record.get("return") in (None, 0), "unobserved slot reports return")
        for flag in ("success", "terminal"):
            _require(record.get(flag) in (None, False), "unobserved slot reports terminal success")
        _require(record.get("outcome") is None, "unobserved slot reports an outcome")
        if record.get("identity") is not None:
            _identity(record)
    else:
        base_fields = {"layout_id", "rotation", "branch_row", "swap_keys", "appearance"}
        mapped_fields = base_fields | {"movement_action_map", "observation_basis"}
        _require(
            type(layout) is dict and set(layout) in (base_fields, mapped_fields),
            "invalid layout fields",
        )
        if set(layout) == mapped_fields:
            _require(
                layout["observation_basis"] == "spatial_directions_v1"
                and type(layout["movement_action_map"]) is list
                and len(layout["movement_action_map"]) == 4,
                "invalid recorded control mapping or spatial observation basis",
            )
            owned = KeyDoorLayout(
                **{key: layout[key] for key in base_fields},
                movement_action_map=cast(
                    tuple[int, int, int, int], tuple(layout["movement_action_map"])
                ),
            )
        else:
            owned = KeyDoorLayout(**layout)
        _require(owned.layout_id == record["context"], "recorded layout context mismatch")
        _require(
            record.get("num_classes") == 5
            and type(record.get("num_classes")) is int
            and record.get("observation_dim") == 8
            and type(record.get("observation_dim")) is int,
            "invalid episode dimensions",
        )
        _identity(record)
        frames = _playback(record, owned, **({"train": True} if train else {}))
    return dict(pointer=pointer, record=deepcopy(record), frames=frames)


def project_key_door_report(
    report: dict[str, Any],
    *,
    source_label: str,
    resources: dict[str, Any] | None = None,
    resource_label: str | None = None,
) -> dict[str, Any]:
    """Validate all conditions/slots before returning detached recorded playback."""
    _text(source_label, "source_label", 1024)
    _json_envelope(report)
    from src.app.clm_probe_inspector_projection import REVIEW_KEYS, project_clm_probe_review

    if REVIEW_KEYS <= report.keys():
        _require(resources is None and resource_label is None, "CLM review owns its recorded costs")
        return project_clm_probe_review(report, source_label=source_label)
    if report.get("definition") == "portable_affine_private_adaptation_v1":
        from src.app.minigrid_private_affine_inspector import project_private_affine_report

        _require(
            resources is None and resource_label is None, "private report owns its recorded costs"
        )
        return project_private_affine_report(report, source_label=source_label)
    if report.get("definition") in (
        "portable_affine_train_and_play_v1",
        "portable_affine_recurring_train_v1",
    ):
        from src.app.minigrid_affine_inspector import project_affine_workflow_report

        _require(
            resources is None and resource_label is None, "affine report owns its recorded costs"
        )
        return project_affine_workflow_report(report, source_label=source_label)
    if report.get("definition") == "guarded_context_serving_v1":
        from src.app.context_serving_inspector_projection import project_context_serving_report

        return project_context_serving_report(
            report, source_label=source_label, resources=resources, resource_label=resource_label
        )
    if report.get("definition") in (
        "qualified_bp_new_contexts_v1",
        "matched_native_ticket_controls_v1",
        "fresh_bp_ticket_demo_v1",
    ):
        from src.app.ticket_workflow_inspector_projection import project_ticket_workflow_report

        return project_ticket_workflow_report(
            report, source_label=source_label, resources=resources, resource_label=resource_label
        )
    if all(key in report for key in ("comparisons", "observed_work", "baseline_source")):
        from src.app.continuous_comparison_inspector import project_continuous_comparison_report

        return project_continuous_comparison_report(
            report, source_label=source_label, resources=resources, resource_label=resource_label
        )
    if report.get("definition") in ("taskspace_memory_demo_v1", "taskspace_memory_native_v1"):
        from src.app.taskspace_memory_inspector_projection import project_taskspace_memory_report

        return project_taskspace_memory_report(
            report, source_label=source_label, resources=resources, resource_label=resource_label
        )
    if report.get("role") == "final_test" and "benchmark_updates" in report:
        from src.app.minigrid_saved_fifo_inspector import project_minigrid_heldout_fifo_report

        _require(
            resources is None and resource_label is None, "final report owns its recorded costs"
        )
        return project_minigrid_heldout_fifo_report(report, source_label=source_label)
    _require(_status(report.get("status")), "invalid report status")
    protocol = report.get("protocol")
    if (
        type(protocol) is dict
        and protocol.get("protocol_id") == "paired_key_door_own_policy_quota_v1"
    ):
        from src.app.paired_quota_inspector import project_paired_quota_report

        _require(
            resources is None and resource_label is None, "paired report owns its recorded costs"
        )
        return project_paired_quota_report(
            report, source_label=source_label, project_episode=_episode
        )
    if type(protocol) is dict and protocol.get("protocol_id") in (
        "local_key_door_bounded_action_memory_v1",
        "local_key_door_bounded_action_memory_probe_v1",
        "local_key_door_matched_public_history_live_v1",
    ):
        from src.app.public_history_inspector_projection import project_public_history_report

        return project_public_history_report(
            report, source_label=source_label, resources=resources, resource_label=resource_label
        )
    if (
        type(protocol) is dict
        and protocol.get("protocol_id") == "coherent_independent_data_replay_removal_v1"
    ):
        from src.app.coherent_stream_inspector_projection import project_coherent_stream_report

        return project_coherent_stream_report(
            report, source_label=source_label, resources=resources, resource_label=resource_label
        )
    if type(protocol) is dict and str(protocol.get("protocol", "")).startswith(
        (
            "sensor_brightness_prearrival_",
            "sensor_brightness_scheduled_",
            "sensor_brightness_recent_replay_",
            "sensor_stationary_prearrival_",
            "sensor_stationary_scheduled_",
            "sensor_stationary_recent_replay_",
        )
    ):
        from src.app.sensor_inspector_projection import project_sensor_report

        return project_sensor_report(
            report, source_label=source_label, resources=resources, resource_label=resource_label
        )
    _require(
        resources is None and resource_label is None,
        "separate resource artifacts are supported only for sensor reports",
    )
    from src.app.minigrid_saved_fifo_inspector import (
        PROTOCOL_ID as SAVED_FIFO_PROTOCOL_ID,
        project_minigrid_saved_fifo_report,
    )

    if type(protocol) is dict and protocol.get("protocol_id") == SAVED_FIFO_PROTOCOL_ID:
        _require(
            resources is None and resource_label is None,
            "saved FIFO inspector uses its own recorded costs only",
        )
        return project_minigrid_saved_fifo_report(report, source_label=source_label)

    from src.app.minigrid_learnability_inspector import (
        PROTOCOL_IDS as LEARNABILITY_PROTOCOL_IDS,
        project_minigrid_learnability_report,
    )

    if type(protocol) is dict and protocol.get("protocol_id") in LEARNABILITY_PROTOCOL_IDS:
        return project_minigrid_learnability_report(report, source_label=source_label)
    if (
        type(protocol) is dict
        and protocol.get("protocol_id") == "minigrid_fixed_a128_acquisition_v1"
    ):
        from src.app.minigrid_inspector_projection import project_minigrid_report

        return project_minigrid_report(report, source_label=source_label)
    if "conditions" in report:
        conditions = report["conditions"]
        _require(
            type(conditions) is dict
            and 1 <= len(conditions) <= 2
            and set(conditions) <= {"off", "on"},
            "invalid paired conditions",
        )
        sources = [
            (key, f"/conditions/{key}", conditions[key])
            for key in ("off", "on")
            if key in conditions
        ]
    else:
        sources = [("single", "", report)]
    projected = []
    for key, pointer, condition in sources:
        _require(type(condition) is dict and _status(condition.get("status")), "invalid condition")
        protocol = condition.get("protocol")
        _require(type(protocol) is dict, "condition requires recorded protocol")
        protocol_id = _text(protocol.get("protocol_id"), "protocol_id")
        train = "acquired_return_diagnostics" in condition
        _require(not train or key == "single", "TRAIN inspector supports a single report only")
        _require(
            train or protocol_id not in PROTOCOL_CONTROLS,
            "acquired TRAIN protocol requires its original readout selector",
        )
        _require(
            train or condition.get("source_role") != "train",
            "TRAIN source requires its original readout selector",
        )
        _probe_readouts(condition)
        _learning_readouts(condition)
        _resource_readouts(condition)
        evaluation = condition.get("policy_evaluation")
        if train:
            episodes = validate_acquired_report(condition)
        elif evaluation is None:
            _require(
                condition["status"] != "complete", "complete condition requires policy evaluation"
            )
            episodes = []
        else:
            _require(type(evaluation) is dict, "invalid policy evaluation")
            episodes = evaluation.get("episodes")
            _require(
                type(episodes) is list and len(episodes) <= 128, "episode slot count exceeds 128"
            )
        projected.append(
            dict(
                key=key,
                pointer=pointer,
                status=condition["status"],
                protocol_id=protocol_id,
                episodes=[
                    _episode(episode, f"{pointer}/policy_evaluation/episodes/{index}")
                    if not train
                    else _episode(
                        episode,
                        f"{pointer}/acquired_return_diagnostics/episodes/{index}",
                        train=True,
                    )
                    for index, episode in enumerate(episodes)
                ],
                map_origin="derived_from_recorded_layout_and_applied_actions",
                unknown=dict(topology=None, lineage=None, sleep_proposals=None),
            )
        )
        if train:
            projected[-1].update(readout_key="acquired_return_diagnostics", source_role="train")
    return dict(
        schema_id="key_door_inspector_v1",
        source_label=source_label,
        origin="recorded_artifact",
        source=deepcopy(report),
        conditions=projected,
    )
