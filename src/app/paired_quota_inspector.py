"""Project saved paired quota reports without executing a policy or environment.

The callback derives playback for recorded episodes. This module only binds those
episodes to recorded quota and score ordinals; it never fills missing raw fields,
recomputes comparisons, or interprets overlapping wall spans as additive costs.
"""

from collections.abc import Callable
from copy import deepcopy
import json
from math import isfinite
from typing import Any, cast


_PROTOCOL = "paired_key_door_own_policy_quota_v1"
_BOUNDARIES = (0, 24, 48, 72, 96)
_CONTEXTS = tuple(f"{layout}-{cue}" for layout in "ABCD" for cue in ("red", "blue"))
_STATUSES = {"complete", "failed", "incomplete", "not_run", "running", "pending"}
_ARM_STATUSES = _STATUSES | {"trained", "not_evaluated"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, name: str, maximum: int = 2**53 - 1) -> int:
    _require(type(value) is int and 0 <= value <= maximum, f"invalid {name} count")
    return value


def _number(value: Any, name: str) -> float:
    _require(type(value) in (int, float), f"invalid {name} number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"nonfinite {name}") from error
    _require(isfinite(result), f"nonfinite {name}")
    return result


def _status(value: Any) -> bool:
    return type(value) is str and value in _STATUSES


def _costs(run: dict[str, Any]) -> None:
    """Check recorded count types without recomputing scientific work totals."""
    work = run.get("work")
    if work is not None:
        _require(type(work) is dict, "invalid recorded work")
        for key in (
            "observed_checkpoint_training_unroll_steps",
            "classification_unroll_steps",
            "live_callback_attempts",
            "live_callback_returns",
            "live_applied_steps",
            "live_sessions_started",
            "collection_unroll_steps",
            "collection_physical_steps",
            "collection_physical_attempts",
            "collection_episodes",
            "native_training_updates",
        ):
            if work.get(key) is not None:
                _integer(work[key], key)
        unknown = work.get("failure_work_may_be_unobserved")
        _require(unknown is None or type(unknown) is bool, "invalid unknown-work flag")
    for key in ("live_sessions_started", "collection_attempts"):
        if run.get(key) is not None:
            _integer(run[key], key)


def _array(value: Any, name: str, maximum: int, complete: bool) -> list[Any]:
    if value is None and not complete:
        return []
    _require(type(value) is list and len(value) <= maximum, f"invalid {name} array")
    _require(not complete or len(value) == maximum, f"incomplete {name} coverage")
    return value


def _envelope(report: Any) -> None:
    """Bound work before copying or invoking the injected playback callback."""
    pending = [(report, 0)]
    nodes = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        _require(nodes <= 500_000 and depth <= 64, "paired report exceeds JSON limits")
        if type(value) is dict:
            _require(all(type(key) is str for key in value), "JSON keys must be strings")
            pending.extend((item, depth + 1) for item in value.values())
        elif type(value) is list:
            pending.extend((item, depth + 1) for item in value)
        elif type(value) in (int, float):
            _number(value, "JSON value")
        else:
            _require(value is None or type(value) in (str, bool), "invalid JSON value")
    encoded = json.dumps(report, allow_nan=False, ensure_ascii=False).encode("utf-8")
    _require(len(encoded) <= 8 * 1024 * 1024, "paired report exceeds 8 MiB")


def _quota_rows(run: dict[str, Any], complete: bool) -> list[Any]:
    rows = _array(run.get("quota_checkpoints"), "quota checkpoints", 5, complete)
    previous = 0
    for index, row in enumerate(rows):
        if row is None:
            _require(not complete, "complete quota checkpoint cannot be null")
            continue
        _require(type(row) is dict, "invalid quota checkpoint")
        physical = _integer(row.get("physical_steps"), "physical steps", 96)
        native = _integer(row.get("native_updates"), "native updates", 512)
        _require(physical == _BOUNDARIES[index], "quota physical ordinal mismatch")
        _require(native >= previous, "quota native indices must not decrease")
        _require(index != 0 or native == 0, "initial quota native index must be zero")
        _require(run["arm"] != "frozen" or native == 0, "frozen quota has active updates")
        previous = native
    return rows


def _score_rows(run: dict[str, Any], quotas: list[Any], complete: bool) -> list[Any]:
    stream = run.get("stream_result")
    if stream is None and not complete:
        return []
    _require(type(stream) is dict, "invalid stream result")
    rows = _array(cast(dict[str, Any], stream).get("checkpoints"), "score checkpoints", 5, complete)
    for index, row in enumerate(rows):
        if row is None:
            _require(not complete, "complete score checkpoint cannot be null")
            continue
        _require(type(row) is dict, "invalid score checkpoint")
        native = _integer(row.get("updates"), "score native updates", 512)
        _require(index < len(quotas) and quotas[index] is not None, "score has no quota")
        _require(native == quotas[index]["native_updates"], "score native index mismatch")
        if row.get("physical_steps") is not None:
            _require(
                _integer(row["physical_steps"], "score physical steps", 96) == _BOUNDARIES[index],
                "score physical index mismatch",
            )
        scores = row.get("scores")
        _require(type(scores) is dict, "invalid score metrics")
        _require(set(scores) <= set(_CONTEXTS), "unknown score context")
        _require(not complete or set(scores) == set(_CONTEXTS), "incomplete score coverage")
        for metrics in scores.values():
            if metrics is None:
                _require(not complete, "complete score cannot be null")
                continue
            _require(type(metrics) is dict, "invalid score metrics")
            for key in ("ce", "accuracy"):
                if metrics.get(key) is None:
                    _require(not complete, "complete score metric unavailable")
                    continue
                number = _number(metrics[key], key)
                _require(number >= 0, f"negative {key}")
                _require(key != "accuracy" or number <= 1, "accuracy outside range")
    return rows


def _episode_rows(
    run: dict[str, Any], quotas: list[Any], complete: bool, arm_protocol: dict[str, Any]
) -> list[Any]:
    rows = _array(run.get("episodes"), "episodes", 40, complete)
    for index, row in enumerate(rows):
        if row is None:
            _require(not complete, "complete episode cannot be null")
            continue
        _require(type(row) is dict, "invalid episode record")
        stage, slot = divmod(index, 8)
        context, color = _CONTEXTS[slot].split("-")
        _require(row.get("context") == context, "episode context ordinal mismatch")
        _require(type(row.get("cue")) is int, "invalid episode cue")
        _require(row["cue"] == (1 if color == "red" else -1), "episode cue mismatch")
        _require(
            row.get("layout_partition") == ("heldout_layout" if context == "D" else "known_layout"),
            "episode layout partition mismatch",
        )
        _require(_status(row.get("status")), "invalid episode status")
        observed = row["status"] == "complete"
        _require(not complete or observed, "complete arm contains unfinished episode")
        native = _integer(row.get("checkpoint_updates"), "episode native updates", 512)
        _require(stage < len(quotas) and quotas[stage] is not None, "episode has no quota")
        _require(native == quotas[stage]["native_updates"], "episode native index mismatch")
        if row.get("checkpoint_physical_steps") is not None:
            _require(
                _integer(row["checkpoint_physical_steps"], "episode physical steps", 96)
                == _BOUNDARIES[stage],
                "episode physical index mismatch",
            )
        for key in ("role", "source_role"):
            if row.get(key) is not None:
                _require(row[key] == "final_test", "episode evaluation role mismatch")
        if observed:
            _require(row.get("phase") == "evaluation", "episode is not evaluation")
            _require(row.get("learning_active") is False, "evaluation learning is active")
        identity = row.get("identity")
        if identity is not None:
            _require(type(identity) is dict, "invalid episode identity")
            _require(identity.get("method") == run["method"], "episode method mismatch")
            _require(identity.get("replay_enabled") is False, "episode replay mismatch")
            _require(
                _integer(identity.get("checkpoint_updates"), "identity native updates", 512)
                == native,
                "episode identity native mismatch",
            )
            _require(
                type(identity.get("seed")) is int and identity["seed"] == arm_protocol.get("seed"),
                "episode seed mismatch",
            )
            _require(
                identity.get("task_sequence") == arm_protocol.get("task_sequence"),
                "episode task sequence mismatch",
            )
        _require(not observed or identity is not None, "complete episode lacks identity")
    return rows


def _parameter_shapes(parameters: Any, protocol: dict[str, Any]) -> dict[str, list[int]] | None:
    """Describe saved arrays; never reconstruct native tensors or infer bytes."""
    if parameters is None:
        return None
    dimensions = [
        _integer(protocol.get(key), key, 256)
        for key in ("feature_dim", "hidden_dim", "num_classes")
    ]
    _require(all(value > 0 for value in dimensions), "empty native dimension")
    features, hidden, classes = dimensions
    expected = {
        "weight_input": [features, hidden],
        "weight_recurrent": [hidden, hidden],
        "bias_hidden": [1, hidden],
        "weight_output": [hidden, classes],
        "bias_output": [1, classes],
    }
    _require(
        type(parameters) is dict and set(parameters) == set(expected),
        "invalid native parameter names",
    )
    for name, (rows, columns) in expected.items():
        values = parameters[name]
        _require(type(values) is list and len(values) == rows, "invalid native parameter rows")
        for row in values:
            _require(type(row) is list and len(row) == columns, "invalid native parameter columns")
            for value in row:
                _number(value, "native parameter")
    return expected


def _observation_shapes(
    run: dict[str, Any], quotas: list[Any], protocol: dict[str, Any]
) -> list[dict[str, list[int]] | None]:
    """Bind optional observations to quotas before invoking any playback code."""
    observations = _array(run.get("observations"), "native observations", 5, False)
    shapes: list[dict[str, list[int]] | None] = []
    for index, observation in enumerate(observations):
        if observation is None:
            shapes.append(None)
            continue
        _require(type(observation) is dict, "invalid native observation")
        quota = quotas[index] if index < len(quotas) else None
        native = _integer(observation.get("native_updates"), "observation native updates", 512)
        if quota is not None:
            _require(native == quota["native_updates"], "observation native index mismatch")
        _require(run["arm"] != "frozen" or native == 0, "frozen observation has active updates")
        for name, expected in (("wake_updates", native), ("physical_steps", _BOUNDARIES[index])):
            if observation.get(name) is not None:
                _require(
                    _integer(observation[name], name) == expected, "observation clock mismatch"
                )
        for name in (
            "native_training_examples",
            "head_tensor_bytes",
            "optimizer_tensor_bytes",
            "gradient_tensor_bytes",
            "activation_tensor_bytes",
            "training_unroll_steps",
        ):
            if observation.get(name) is not None:
                _integer(observation[name], name)
        architecture = observation.get("architecture")
        _require(
            architecture is None or architecture == "elman_tanh", "unsupported native architecture"
        )
        scope = observation.get("memory_scope")
        _require(
            scope is None or type(scope) is str and 0 < len(scope) <= 1024, "invalid memory scope"
        )
        observed_shapes = _parameter_shapes(observation.get("parameters"), protocol)
        # A partial legacy report can retain an observation after losing its
        # quota slot. Validate it, but never present it as an admitted checkpoint.
        shapes.append(observed_shapes if quota is not None else None)
    return shapes


def _run(
    run: Any, index: int, complete: bool, arm_protocol: dict[str, Any]
) -> tuple[dict[str, Any], list[Any]]:
    _require(type(run) is dict, "invalid paired arm")
    arm = ("frozen", "trained")[index]
    _require(run.get("arm") == arm, "paired arms missing, duplicated or reordered")
    _require(
        type(run.get("status")) is str and run["status"] in _ARM_STATUSES, "invalid arm status"
    )
    _require(not complete or run["status"] == "complete", "complete pair has unfinished arm")
    _require(run.get("method") == "recurrent_rl", "invalid paired method")
    _require(run.get("replay_enabled") is False, "paired replay must be disabled")
    _require(run.get("action_mapping") == "movement_permutation_v1", "invalid action mapping")
    _require(run.get("action_conditioning") == "legal_actions_v1", "invalid action conditioning")
    _costs(run)
    arm_complete = run["status"] == "complete"
    quotas = _quota_rows(run, arm_complete)
    scores = _score_rows(run, quotas, arm_complete)
    episodes = _episode_rows(run, quotas, arm_complete, arm_protocol)
    shapes = _observation_shapes(run, quotas, arm_protocol)
    observations = run.get("observations") or []
    stages = []
    for ordinal, quota in enumerate(quotas):
        stages.append(
            dict(
                physical_steps=None if quota is None else quota["physical_steps"],
                native_updates=None if quota is None else quota["native_updates"],
                pointer=f"/runs/{index}/quota_checkpoints/{ordinal}",
                score_pointer=(
                    f"/runs/{index}/stream_result/checkpoints/{ordinal}"
                    if ordinal < len(scores) and scores[ordinal] is not None
                    else None
                ),
                episode_indices=list(range(ordinal * 8, min((ordinal + 1) * 8, len(episodes)))),
                observation_pointer=(
                    f"/runs/{index}/observations/{ordinal}"
                    if quota is not None
                    and ordinal < len(observations)
                    and observations[ordinal] is not None
                    else None
                ),
                parameter_shapes=shapes[ordinal] if ordinal < len(shapes) else None,
            )
        )
    return (
        dict(
            key=arm,
            pointer=f"/runs/{index}",
            status=run["status"],
            protocol_id=_PROTOCOL,
            map_origin="derived_from_recorded_layout_and_applied_actions",
            unknown=dict(topology=None, lineage=None, sleep_proposals=None),
            stages=stages,
            episodes=[],
        ),
        episodes,
    )


def _comparison(value: Any, runs: list[Any], complete: bool) -> Any:
    if value is None:
        _require(not complete, "complete pair lacks comparison coverage")
        return None
    rows = _array(value, "comparison", 40, complete)
    seen = set()
    for index, row in enumerate(rows):
        if row is None:
            _require(not complete, "complete comparison cannot be null")
            continue
        _require(type(row) is dict, "invalid comparison record")
        physical = _integer(row.get("physical_steps"), "comparison physical steps", 96)
        context = row.get("context")
        _require(physical in _BOUNDARIES and context in _CONTEXTS, "invalid comparison cell")
        cell = (physical, context)
        _require(cell not in seen, "duplicate comparison cell")
        seen.add(cell)
        native = row.get("native_updates")
        _require(type(native) is list and len(native) == 2, "invalid comparison native indices")
        ordinal = _BOUNDARIES.index(physical)
        for arm, number in enumerate(native):
            _integer(number, "comparison native updates", 512)
            quotas = runs[arm].get("quota_checkpoints")
            _require(
                type(quotas) is list and ordinal < len(quotas) and quotas[ordinal] is not None,
                "comparison cell has no quota",
            )
            _require(number == quotas[ordinal]["native_updates"], "comparison native mismatch")
        for name in ("ce_reduction", "accuracy_gain", "return_gain"):
            _number(row.get(name), name)
        _require(
            type(row.get("success_gain")) is int and -1 <= row["success_gain"] <= 1,
            "invalid signed success gain",
        )
        _require(type(row.get("extra_steps")) is int, "invalid signed extra steps")
    _require(not complete or len(seen) == 40, "incomplete comparison coverage")
    return [
        dict(pointer=f"/comparison/{index}", record=deepcopy(row)) for index, row in enumerate(rows)
    ]


def _timing(value: Any) -> Any:
    if value is None:
        return None
    _require(type(value) is dict, "invalid timing record")
    _require(
        value.get("definition") == "paired_quota_inclusive_wall_v1", "invalid timing definition"
    )
    _require(value.get("clock") == "perf_counter", "invalid timing clock")
    arms, guards = value.get("arms"), value.get("guards")
    _require(type(arms) is dict and set(arms) == {"frozen", "trained"}, "invalid timing arms")
    _require(type(guards) is dict, "invalid timing guards")
    _require(set(guards) <= {"pair", "preceding", "pending"}, "unknown timing guard")
    spans = list(guards.values())
    for arm in arms.values():
        _require(type(arm) is dict, "invalid arm timing")
        _require(set(arm) <= {"train", "evaluation"}, "unknown arm timing phase")
        spans.extend(arm.values())
    for span in spans:
        if span is None:
            continue
        _require(type(span) is dict, "invalid timing span")
        for name in ("attempts", "returns", "raised"):
            _integer(span.get(name), f"timing {name}")
        _require(span["returns"] + span["raised"] <= span["attempts"], "timing count mismatch")
        seconds = span.get("inclusive_seconds")
        if seconds is not None:
            _require(_number(seconds, "inclusive seconds") >= 0, "negative inclusive seconds")
        _require(
            span.get("clock_error") is None or type(span["clock_error"]) is str,
            "invalid timing clock error",
        )
    return dict(pointer="/timing", record=deepcopy(value))


def _displayed_costs(report: dict[str, Any]) -> None:
    """Validate displayed fields without reconstructing scientific work."""
    names: tuple[str, ...]
    for container, names, kind in [
        (report, ("total_seconds",), "seconds"),
        (report.get("observed_process_rss"), ("peak_bytes",), "count"),
        (report["protocol"].get("resources"), ("wall_seconds",), "seconds"),
    ]:
        if container is not None:
            _require(type(container) is dict, "invalid displayed cost record")
            for name in names:
                value = container.get(name)
                if value is not None:
                    if kind == "count":
                        _integer(value, name)
                    else:
                        _require(_number(value, name) >= 0, "negative recorded seconds")
    for run in report["runs"]:
        for container, names in [
            (run, ("wake_updates",)),
            (run.get("stream_result"), ("training_examples", "score_forward_calls")),
            (
                run.get("work"),
                (
                    "classification_unroll_steps",
                    "live_callback_returns",
                    "live_applied_steps",
                    "observed_checkpoint_training_unroll_steps",
                ),
            ),
        ]:
            if container is not None:
                _require(type(container) is dict, "invalid displayed arm cost")
                for name in names:
                    if container.get(name) is not None:
                        _integer(container[name], name)
        for container, names in [
            (run, ("source_unchanged", "actor_unchanged", "initial_state_matches_control")),
            (run.get("work"), ("failure_work_may_be_unobserved",)),
        ]:
            if container is not None:
                for name in names:
                    _require(
                        container.get(name) is None or type(container[name]) is bool,
                        f"invalid recorded {name} flag",
                    )


def project_paired_quota_report(
    report: Any,
    *,
    source_label: str,
    project_episode: Callable[[Any, str], dict[str, Any]],
) -> dict[str, Any]:
    """Return a detached projection, validating all bindings before playback.

    For example, the existing inspector supplies its recorded-episode projector
    as ``project_episode``. Missing trained physical labels stay null in source;
    quota ordinals provide the separate stage association used by the renderer.
    """
    _envelope(report)
    _require(type(report) is dict, "paired report must be an object")
    _require(
        type(source_label) is str and 0 < len(source_label.encode("utf-8")) <= 512,
        "invalid source label",
    )
    _require(callable(project_episode), "episode projector must be callable")
    protocol = report.get("protocol")
    _require(
        type(protocol) is dict and protocol.get("protocol_id") == _PROTOCOL,
        "unsupported paired protocol",
    )
    _require(protocol.get("arms") == ["frozen", "trained"], "invalid declared paired arms")
    boundaries = protocol.get("physical_boundaries")
    _require(
        type(boundaries) is list
        and all(type(item) is int for item in boundaries)
        and tuple(boundaries) == _BOUNDARIES,
        "invalid physical boundaries",
    )
    _require(
        type(protocol.get("physical_steps_per_arm")) is int
        and protocol["physical_steps_per_arm"] == 96,
        "invalid physical quota",
    )
    arm_protocol = protocol.get("arm_protocol")
    _require(type(arm_protocol) is dict, "invalid arm protocol")
    _require(
        arm_protocol.get("protocol_id") == "actual_key_door_seeded_interaction_quota_v1",
        "unsupported arm protocol",
    )
    _require(_status(report.get("status")), "invalid paired status")
    complete = report["status"] == "complete"
    runs = report.get("runs")
    _require(type(runs) is list and len(runs) == 2, "paired report requires two arms")
    validated = [_run(run, index, complete, arm_protocol) for index, run in enumerate(runs)]
    _displayed_costs(report)
    comparison = _comparison(report.get("comparison"), runs, complete)
    timing = _timing(report.get("timing"))
    # Own the source before calling external code; callback inputs are separate
    # copies so playback cannot modify the evidence the renderer exposes.
    source = deepcopy(report)
    conditions = []
    for arm_index, (condition, _) in enumerate(validated):
        episodes = source["runs"][arm_index].get("episodes") or []
        for index, row in enumerate(episodes):
            pointer = f"/runs/{arm_index}/episodes/{index}"
            projected: dict[str, Any]
            if row is None:
                projected = dict(pointer=pointer, record=None, frames=[])
            else:
                projected = project_episode(deepcopy(row), pointer)
                _require(
                    type(projected) is dict and projected.get("pointer") == pointer,
                    "episode projector changed source pointer",
                )
                _require(projected.get("record") == row, "episode projector changed source record")
            condition["episodes"].append(deepcopy(projected))
        conditions.append(condition)
    return dict(
        schema_id="paired_quota_inspector_v1",
        source_label=source_label,
        origin="recorded_artifact",
        source=source,
        conditions=conditions,
        comparison=comparison,
        timing=timing,
    )
