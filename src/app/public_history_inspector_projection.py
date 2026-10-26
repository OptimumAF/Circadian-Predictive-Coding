"""Project three bounded saved public-history reports without executing learners.

The original JSON remains independently owned evidence. Episode playback reuses
the recorded-action validator; scientific readouts are never recomputed. Separate
resource files, model loading, physics, training and unrecorded metadata are
outside this projection's scope.
"""

from copy import deepcopy
from typing import Any

from src.app.key_door_inspector import (
    _episode,
    _integer,
    _json_envelope,
    _metric_values,
    _number,
    _optional_object,
    _probe_readouts,
    _records,
    _require,
    _resource_readouts,
    _status,
    _text,
)

_MODES = {
    "local_key_door_bounded_action_memory_v1": "memory_live",
    "local_key_door_bounded_action_memory_probe_v1": "memory_probe",
    "local_key_door_matched_public_history_live_v1": "matched_live",
}
_NEURAL = ["bp", "pc", "neutral_cpc", "gated_cpc"]
_POINTS = [0, 2, 4, 6, 8]
_TRAIN_CONTEXTS = {f"{context}-{cue}" for context in "ABC" for cue in ("red", "blue")}
_MEMORY_POLICY = dict(
    method="observation_memory",
    feature_dim=16,
    num_classes=5,
    max_entries=64,
    max_payload_bytes=10752,
)
_MAX_COUNT = 2**53 - 1
_COUNTS = {
    "steps_executed",
    "sample_count",
    "entries",
    "conflicting_keys",
    "wake_updates",
    "native_updates",
    "native_training_examples",
    "checkpoint_updates",
    "update_number",
    "charged_updates",
    "collected_episodes",
    "collected_probe_episodes",
    "native_score_returns",
    "completed_updates",
    "training_examples",
    "updates",
    "reserved_data_bytes",
    "completed_training_examples",
    "attempted_training_examples",
    "attempted_updates",
    "session_construction_attempts",
    "sessions_started",
    "live_sessions_started",
    "attempted_steps",
    "encoded_frames",
    "native_prediction_returns",
    "completed_history_steps",
    "callback_attempts",
    "callback_returns",
    "applied_steps",
    "score_forwards_attempted",
    "score_forwards_completed",
    "score_rows_attempted",
    "score_rows_completed",
    "score_forward_calls",
    "score_forward_calls_attempted",
    "score_forward_calls_completed",
    "prediction_calls_attempted",
    "prediction_calls_returned",
    "native_updates_before",
    "native_training_examples_before",
    "head_tensor_bytes",
    "optimizer_tensor_bytes",
    "gradient_tensor_bytes",
    "logical_payload_bytes",
}


def _count(value: Any, name: str, maximum: int = _MAX_COUNT) -> int:
    return _integer(value, 0, maximum, name)


def _scalars(value: Any) -> None:
    """Check recorded metrics/work in place; do not fill absent or null values."""
    if type(value) is list:
        for item in value:
            _scalars(item)
    elif type(value) is dict:
        _metric_values(value)
        if value.get("reason") is not None:
            _require(type(value["reason"]) is str, "invalid recorded reason")
        if value.get("status") is not None:
            _require(_status(value["status"]), "invalid recorded status")
        for key, item in value.items():
            if item is not None and type(item) not in (dict, list):
                if key in _COUNTS or key.endswith("_bytes"):
                    _count(item, key)
                elif key.endswith("_seconds"):
                    _require(_number(item, key) >= 0, f"negative {key}")
                elif key in ("illegal_action_rate", "step_limit_rate"):
                    _require(0 <= _number(item, key) <= 1, f"invalid {key}")
                elif key in (
                    "source_unchanged",
                    "actor_unchanged",
                    "state_unchanged",
                    "budget_binding_preserved",
                    "failed_work_may_be_unobserved",
                ):
                    _require(type(item) is bool, f"invalid {key} flag")
            _scalars(item)


def _memory_policy(value: Any) -> None:
    _require(type(value) is dict and set(value) == set(_MEMORY_POLICY), "invalid memory policy")
    for key, expected in _MEMORY_POLICY.items():
        _require(
            type(value[key]) is type(expected) and value[key] == expected,
            f"invalid memory policy {key}",
        )


def _protocol(protocol: dict[str, Any], mode: str, complete: bool) -> list[str]:
    methods = _NEURAL if mode == "matched_live" else ["observation_memory"]
    for key, expected in (
        ("methods", methods),
        ("checkpoint_updates", _POINTS),
        ("task_sequence", ["A", "B", "C", "A"]),
    ):
        if complete or key in protocol:
            _require(
                type(protocol.get(key)) is list
                and protocol[key] == expected
                and all(type(a) is type(b) for a, b in zip(protocol[key], expected)),
                f"invalid public-history protocol {key}",
            )
    for key, expected_number in (("seed", 17),):
        if complete or key in protocol:
            _require(
                type(protocol.get(key)) is int and protocol[key] == expected_number,
                f"invalid protocol {key}",
            )
    if mode != "matched_live":
        if complete or "policy" in protocol:
            _memory_policy(protocol.get("policy"))
    else:
        for key, expected_number in (("feature_dim", 16), ("hidden_dim", 8), ("num_classes", 5)):
            if complete or key in protocol:
                _require(
                    type(protocol.get(key)) is int and protocol[key] == expected_number,
                    f"invalid matched protocol {key}",
                )
    stream = _optional_object(protocol.get("stream"), "protocol stream")
    for key, expected_setting in (
        ("training_only", True),
        ("heldout", []),
        ("label_delay", 1),
        ("updates_per_method" if mode == "matched_live" else "updates", 8),
    ):
        if complete or key in stream:
            _require(
                type(stream.get(key)) is type(expected_setting) and stream[key] == expected_setting,
                f"invalid TRAIN-only protocol {key}",
            )
    return methods


def _scores(point: dict[str, Any], contexts: set[str], complete: bool) -> None:
    scores = _optional_object(point.get("scores"), "recorded scores")
    _require(set(scores) <= contexts, "unknown recorded score context")
    if complete:
        _require(set(scores) == contexts, "complete score point omits a context")
    for metrics in scores.values():
        if metrics is None and not complete:
            continue
        _require(type(metrics) is dict, "context scores must be an object")
        _metric_values(metrics)
        if complete:
            _require(
                metrics.get("ce") is not None and metrics.get("accuracy") is not None,
                "complete score point omits its measurements",
            )


def _memory_observation(observation: dict[str, Any], complete: bool) -> None:
    for name, maximum in (
        ("entries", 64),
        ("logical_payload_bytes", 10752),
        ("conflicting_keys", 64),
    ):
        if observation.get(name) is not None:
            _count(observation[name], name, maximum)
    snapshot = _optional_object(observation.get("snapshot"), "memory snapshot")
    if not snapshot:
        _require(not complete, "complete memory observation omits its snapshot")
        return
    _memory_policy(snapshot.get("policy"))
    entries = _records(snapshot.get("entries"), "memory entries", 64)
    _require(type(snapshot.get("entries")) is list, "snapshot entries must be recorded")
    seen = set()
    exposures = 0
    conflicts = 0
    for entry in entries:
        _require(set(entry) == {"features", "counts"}, "invalid memory entry fields")
        features, counts = entry["features"], entry["counts"]
        _require(type(features) is list and len(features) == 16, "invalid memory key dimensions")
        feature_key = tuple(_number(value, "memory coordinate") for value in features)
        _require(
            all(-1 <= value <= 1 for value in feature_key) and feature_key not in seen,
            "invalid or duplicate memory key",
        )
        seen.add(feature_key)
        _require(type(counts) is list and len(counts) == 5, "invalid memory count dimensions")
        for value in counts:
            _count(value, "memory action count", 2**63 - 1)
        _require(sum(counts) > 0, "empty stored memory entry")
        exposures += sum(counts)
        conflicts += int(sum(value > 0 for value in counts) > 1)
    updates = _count(snapshot.get("completed_updates"), "snapshot updates", 8)
    _require(
        updates == observation["wake_updates"], "memory checkpoint clock disagrees with snapshot"
    )
    examples = _count(snapshot.get("training_examples"), "snapshot examples", 64)
    _require(exposures == examples == 8 * updates, "memory snapshot exposure clocks disagree")
    for key, expected in (
        ("entries", len(entries)),
        ("logical_payload_bytes", 168 * len(entries)),
        ("conflicting_keys", conflicts),
        ("native_updates", updates),
        ("native_training_examples", examples),
    ):
        if complete or key in observation:
            _require(
                type(observation.get(key)) is int and observation[key] == expected,
                f"memory observation {key} disagrees with its snapshot",
            )


def _training(row: dict[str, Any], memory: bool, complete: bool) -> None:
    observations = _records(row.get("observations"), "observations", 5)
    points = _records(row.get("training_points"), "TRAIN score points", 5)
    for records, name, axis in (
        (observations, "observations", "wake_updates"),
        (points, "TRAIN points", "updates"),
    ):
        indices = [_count(point.get(axis), name, 8) for point in records]
        _require(
            all(value in _POINTS for value in indices) and indices == sorted(set(indices)),
            f"invalid {name} checkpoint order",
        )
        if complete:
            _require(indices == _POINTS, f"complete run omits {name}")
    for observation in observations:
        point = observation["wake_updates"]
        for name, expected in (("native_updates", point), ("native_training_examples", 8 * point)):
            if complete or name in observation:
                _require(
                    type(observation.get(name)) is int and observation[name] == expected,
                    f"invalid observation {name} origin",
                )
        if memory:
            _memory_observation(observation, complete)
    for point in points:
        _scores(point, _TRAIN_CONTEXTS, complete)
        if complete:
            _require(
                all(
                    point.get(name) is not None
                    for name in ("mean_ce", "mean_accuracy", "exact_sequence_accuracy")
                ),
                "complete TRAIN readout omits aggregate measurements",
            )
    stream = _optional_object(row.get("stream_result"), "stream result")
    if stream:
        for key, maximum in (
            ("completed_updates", 8),
            ("training_examples", 64),
            ("score_forward_calls", 0),
        ):
            if complete or key in stream:
                value = _count(stream.get(key), key, maximum)
                if complete:
                    _require(value == maximum, f"complete stream has incomplete {key}")
        receipts = _records(stream.get("applied_updates"), "applied TRAIN updates", 8)
        _require(
            [receipt.get("update_number") for receipt in receipts]
            == list(range(1, len(receipts) + 1)),
            "invalid original TRAIN receipt order",
        )
        if complete:
            _require(
                len(receipts) == 8
                and stream.get("evaluation_scope") == "training_only"
                and stream.get("checkpoints") == [],
                "invalid complete TRAIN-only stream",
            )
    else:
        _require(not complete, "complete run omits its stream result")
    if complete:
        _require(
            row.get("wake_updates") == 8 and row.get("native_score_returns") == 30,
            "complete run omits actual TRAIN work",
        )
        work = _optional_object(row.get("work"), "TRAIN work")
        _require(
            all(
                work.get(key) == expected
                for key, expected in (
                    ("score_forwards_attempted", 30),
                    ("score_forwards_completed", 30),
                    ("score_rows_attempted", 240),
                    ("score_rows_completed", 240),
                )
            ),
            "complete TRAIN score work is incomplete",
        )
        _require(
            row.get("source_unchanged") is True and row.get("actor_unchanged") is True,
            "complete run lacks original learner isolation observations",
        )
        outcomes = _optional_object(row.get("outcomes"), "TRAIN outcomes")
        _require(
            all(
                outcomes.get(key) is not None
                for key in ("new_B_gain", "new_C_gain", "retention_loss", "return_gain")
            ),
            "complete run omits signed TRAIN outcomes",
        )


def _probe_slots(row: dict[str, Any], complete: bool) -> None:
    _probe_readouts(row)
    slots = _records(row.get("probes"), "probe slots", 2)
    origins = [_count(slot.get("checkpoint_updates"), "probe native origin", 8) for slot in slots]
    _require(
        all(value in (0, 8) for value in origins) and origins == sorted(set(origins)),
        "invalid or duplicate probe native origins",
    )
    if complete:
        _require(origins == [0, 8], "complete probe run omits a fixed cohort")
    for slot in slots:
        _require(_status(slot.get("status")), "invalid probe slot status")
        done = slot["status"] == "complete"
        origin = slot["checkpoint_updates"]
        for key, expected in (
            ("native_updates_before", origin),
            ("native_training_examples_before", 8 * origin),
        ):
            if done or key in slot:
                _require(
                    type(slot.get(key)) is int and slot[key] == expected,
                    "probe changed its original native clock",
                )
        report = _optional_object(slot.get("report"), "probe report")
        if done:
            _require(
                slot.get("charged_updates") == 16 and report.get("status") == "complete",
                "complete probe lacks its independent charges",
            )
            _require(
                all(
                    slot.get(flag) is True
                    for flag in ("source_unchanged", "state_unchanged", "budget_binding_preserved")
                ),
                "complete probe lacks original isolation observations",
            )
            for key, expected in (
                ("attempted_updates", 16),
                ("completed_updates", 16),
                ("attempted_training_examples", 128),
                ("completed_training_examples", 128),
                ("score_forward_calls_attempted", 8),
                ("score_forward_calls_completed", 8),
            ):
                _require(
                    type(report.get(key)) is int and report[key] == expected,
                    f"incomplete probe {key}",
                )
        points = _records(report.get("checkpoints"), "probe points", 4)
        indices = [_count(point.get("updates"), "relative probe update", 16) for point in points]
        _require(
            all(value in (0, 4, 8, 16) for value in indices) and indices == sorted(set(indices)),
            "invalid relative probe checkpoint order",
        )
        if done:
            _require(indices == [0, 4, 8, 16], "complete probe omits readouts")
        for point in points:
            _scores(point, {"E-red", "E-blue"}, done)
    if complete:
        _require(
            all(slot["status"] == "complete" for slot in slots),
            "complete probe run has unfinished cohorts",
        )


def _history_work(slot: dict[str, Any], complete: bool) -> None:
    work = _optional_object(slot.get("history_session_work"), "history session work")
    names = (
        "completed_history_steps",
        "native_prediction_returns",
        "encoded_frames",
        "attempted_steps",
    )
    if all(work.get(name) is not None for name in names):
        values = [_count(work[name], name, 12) for name in names]
        _require(values == sorted(values), "inconsistent history session work")
    if complete:
        _require(
            all(work.get(name) is not None for name in names)
            and work.get("session_construction_attempts") == 1
            and work.get("sessions_started") == 1,
            "complete live episode omits actual session work",
        )
        _require(
            work["attempted_steps"]
            == work["encoded_frames"]
            == work["native_prediction_returns"]
            == work["completed_history_steps"]
            == slot.get("prediction_calls_returned")
            == slot.get("prediction_calls_attempted"),
            "complete live episode history and prediction work disagree",
        )


def _live_slots(row: dict[str, Any], pointer: str, complete: bool) -> list[dict[str, Any]]:
    slots = _records(row.get("live_episodes"), "live episodes", 30)
    expected = [(point, context, cue) for point in _POINTS for context in "ABC" for cue in (1, -1)]
    seen = set()
    episodes = []
    previous = -1
    for index, slot in enumerate(slots):
        _require(
            type(slot.get("checkpoint_updates")) is int
            and slot["checkpoint_updates"] in _POINTS
            and type(slot.get("context")) is str
            and slot["context"] in "ABC"
            and type(slot.get("cue")) is int
            and slot["cue"] in (1, -1),
            "invalid known TRAIN live slot identity",
        )
        _require(
            slot.get("action_conditioning") is None,
            "public-history live episodes require raw probabilities, not legal conditioning",
        )
        identity = (slot["checkpoint_updates"], slot["context"], slot["cue"])
        _require(
            identity in expected and identity not in seen,
            "invalid or duplicate known TRAIN live slot",
        )
        order = expected.index(identity)
        _require(order > previous, "known TRAIN live slots are out of order")
        previous = order
        seen.add(identity)
        if slot.get("identity") is not None:
            recorded_identity = _optional_object(slot["identity"], "live identity")
            _require(
                recorded_identity.get("method") == row["method"], "live method identity mismatch"
            )
        episode = _episode(slot, f"{pointer}/live_episodes/{index}", public_history=True)
        _history_work(slot, complete)
        if complete:
            _require(
                slot["status"] == "complete" and slot.get("state_unchanged") is True,
                "complete run has unfinished live evidence",
            )
        episodes.append(episode)
    if complete:
        _require(
            len(slots) == 30 and row.get("live_sessions_started") == 30,
            "complete live run omits planned actual episodes",
        )
        outcomes = _records(row.get("live_outcomes"), "live summaries", 5)
        _require(
            [point.get("updates") for point in outcomes] == _POINTS,
            "complete live run omits original summaries",
        )
        for summary in outcomes:
            _require(
                all(summary.get(key) is not None for key in ("success_rate", "mean_return")),
                "complete live summary omits measurements",
            )
        work = _optional_object(row.get("live_work"), "live work")
        _require(
            work.get("sessions_started") == 30 and work.get("session_construction_attempts") == 30,
            "complete live run omits actual session counts",
        )
        for key, slot_key in (
            ("attempted_steps", "attempted_steps"),
            ("encoded_frames", "encoded_frames"),
            ("native_prediction_returns", "native_prediction_returns"),
            ("completed_history_steps", "completed_history_steps"),
        ):
            _require(
                work.get(key) == sum(slot["history_session_work"][slot_key] for slot in slots),
                "aggregate live history work disagrees with recorded episodes",
            )
        for key, slot_key in (
            ("callback_attempts", "prediction_calls_attempted"),
            ("callback_returns", "prediction_calls_returned"),
            ("applied_steps", "steps_executed"),
        ):
            _require(
                work.get(key) == sum(slot[slot_key] for slot in slots),
                "aggregate live work disagrees with recorded episodes",
            )
    return episodes


def project_public_history_report(
    report: dict[str, Any],
    *,
    source_label: str,
    resources: dict[str, Any] | None = None,
    resource_label: str | None = None,
) -> dict[str, Any]:
    """Keep source measurements and raw TRAIN episode evidence visibly separate."""
    _require(
        resources is None and resource_label is None,
        "public-history reports support embedded costs only; separate resources are unsupported",
    )
    _json_envelope(report)
    _text(source_label, "source label", 1024)
    _require(_status(report.get("status")), "invalid public-history report status")
    protocol = _optional_object(report.get("protocol"), "protocol")
    protocol_id = protocol.get("protocol_id")
    _require(
        type(protocol_id) is str and protocol_id in _MODES, "unsupported public-history protocol"
    )
    mode = _MODES[_text(protocol_id, "protocol_id")]
    complete = report["status"] == "complete"
    methods = _protocol(protocol, mode, complete)
    _scalars(report)
    _resource_readouts(report)
    for flag in (
        "final_data_constructed",
        "scientific_advantage_verified",
        "full_R42_complete",
        "full_R43_complete",
    ):
        if complete or report.get(flag) is not None:
            _require(report.get(flag) is False, f"unsupported public-history {flag} claim")
    rows = _records(report.get("runs"), "runs", len(methods))
    observed_methods = [row.get("method") for row in rows]
    _require(
        all(method in methods for method in observed_methods)
        and observed_methods == [method for method in methods if method in observed_methods],
        "invalid, duplicate or reordered public-history methods",
    )
    if complete:
        _require(
            observed_methods == methods and report.get("collected_episodes") == 8,
            "complete report omits methods or collection",
        )
        _require(report.get("global_errors") == [], "complete report retains global failures")
    projected = []
    for index, row in enumerate(rows):
        _require(_status(row.get("status")), "invalid run status")
        done = row["status"] == "complete"
        _require(not complete or done, "complete report has unfinished runs")
        _training(row, mode != "matched_live", done)
        pointer = f"/runs/{index}"
        if mode == "memory_probe":
            _require(
                not row.get("live_episodes"), "probe-only protocol cannot contain live episodes"
            )
            _probe_slots(row, done)
            episodes = []
        else:
            _require(not row.get("probes"), "live-only protocol cannot contain probes")
            episodes = _live_slots(row, pointer, done)
        projected.append(
            dict(method=row["method"], pointer=pointer, status=row["status"], episodes=episodes)
        )
    return dict(
        schema_id="public_history_inspector_v1",
        mode=mode,
        origin="recorded_artifact",
        source_label=source_label,
        source=deepcopy(report),
        runs=projected,
        unknown=dict(topology=None, lineage=None, sleep_proposals=None, promotion=None),
    )
