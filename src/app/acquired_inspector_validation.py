"""Validate saved acquired-task TRAIN selectors, clocks and derived readouts.

Inputs are already bounded finite JSON; outputs are the original episode list.
This pure validator checks consistency, not execution provenance or scientific
acceptance. Recorded-action geometry/law validation remains in the inspector.
"""

from math import isclose
from typing import Any, cast

from src.core.context_return_metrics import summarize_acquired_return


PROTOCOL_CONTROLS = {
    "key_door_train_only_acquired_task_return_v1": None,
    "key_door_acquired_task_recent48_replay_v1": "recent48",
    "key_door_acquired_task_coverage_replay_v1": "coverage",
}
_POINTS = (192, 256, 320)
_SEQUENCE = ["A", "B", "A", "C_train", "A"]
_STATUSES = {"complete", "failed", "incomplete", "not_run", "running"}


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _same(actual: Any, expected: Any) -> bool:
    if type(expected) is int:
        return type(actual) is int and actual == expected
    if type(expected) is float:
        return type(actual) in (int, float) and isclose(
            actual, expected, rel_tol=1e-12, abs_tol=1e-12
        )
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(
            _same(actual[key], value) for key, value in expected.items()
        )
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _same(a, b) for a, b in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _count(record: dict[str, Any], key: str, maximum: int) -> None:
    if record.get(key) is not None:
        _need(type(record[key]) is int and 0 <= record[key] <= maximum, f"invalid TRAIN {key}")


def _clock(point: int) -> dict[str, int]:
    return dict(
        wake_updates=point,
        replay_committed_updates=point // 2,
        replay_charged_examples=8 * (point // 2),
        native_updates=point + point // 2,
        native_training_examples=8 * (point + point // 2),
    )


def _exact_clock(value: Any, point: int) -> None:
    expected = _clock(point)
    _need(
        type(value) is dict
        and value.keys() == expected.keys()
        and all(
            type(value[key]) is int and value[key] == number for key, number in expected.items()
        ),
        "invalid recorded TRAIN replay clock",
    )


def _source_counts(
    condition: dict[str, Any], control: str | None, *, observed: bool = False
) -> None:
    complete = condition["status"] == "complete" or observed
    _count(condition, "charged_updates", 320)
    stream = condition.get("partial_stream")
    if stream is not None:
        _need(type(stream) is dict, "invalid TRAIN partial stream")
        for key, maximum in (
            ("completed_wake_updates", 320),
            ("native_candidate_updates", 480 if control else 320),
            ("native_candidate_training_examples", 3840 if control else 2560),
        ):
            _count(stream, key, maximum)
        if complete:
            _need(
                stream.get("completed_wake_updates") == 320
                and stream.get("native_candidate_updates") == (480 if control else 320)
                and stream.get("native_candidate_training_examples") == (3840 if control else 2560),
                "TRAIN completed native counts disagree",
            )
    replay = condition.get("replay")
    _need(control is not None or replay is None, "off TRAIN report has replay work")
    if replay is not None:
        _need(type(replay) is dict, "invalid TRAIN replay evidence")
        _count(replay, "charged_examples", 1280)
        rehearsal = replay.get("rehearsal")
        if rehearsal is not None:
            _need(type(rehearsal) is dict, "invalid TRAIN rehearsal evidence")
            _count(rehearsal, "committed_updates", 160)
            if complete:
                _need(
                    replay.get("charged_examples") == 1280
                    and rehearsal.get("committed_updates") == 160,
                    "TRAIN replay counts disagree",
                )
    if complete:
        _need(
            type(stream) is dict
            and condition.get("charged_updates") == 320
            and (control is None or type(replay) is dict),
            "TRAIN completed wake/replay evidence missing or contradictory",
        )
        if control is not None:
            recorded_replay = cast(dict[str, Any], replay)
            _need(
                type(recorded_replay.get("rehearsal")) is dict
                and recorded_replay.get("charged_examples") == 1280
                and recorded_replay["rehearsal"].get("committed_updates") == 160,
                "TRAIN completed replay evidence missing or contradictory",
            )


def _raw_summary(condition: dict[str, Any]) -> dict[str, Any] | None:
    raw = condition.get("acquired_return")
    if raw is None:
        return None
    _need(
        type(raw) is dict and type(raw.get("endpoint_values")) is list, "invalid TRAIN raw summary"
    )
    expected = summarize_acquired_return(tuple(raw["endpoint_values"]))
    _need(_same(raw, expected), "TRAIN raw metrics contradict their endpoints")
    checkpoints = condition.get("checkpoints")
    if checkpoints is not None:
        for endpoint in raw["endpoint_values"]:
            matching = [
                point for point in checkpoints if point.get("updates") == endpoint["updates"]
            ]
            _need(len(matching) <= 1, "duplicate TRAIN raw endpoint")
            if not matching:
                continue
            point = matching[0]
            scores, diagnostics = point.get("scores"), point.get("decision_diagnostics")
            for context in ("A", "C_train"):
                names = (context + "-red", context + "-blue")
                if scores is not None:
                    _need(type(scores) is dict, "invalid TRAIN endpoint scores")
                    for metric in ("ce", "accuracy"):
                        if all(
                            type(scores.get(name)) is dict and scores[name].get(metric) is not None
                            for name in names
                        ):
                            _need(
                                all(
                                    type(scores[name][metric]) in (int, float)
                                    and scores[name][metric] >= 0
                                    and (metric == "ce" or scores[name][metric] <= 1)
                                    for name in names
                                ),
                                "invalid TRAIN endpoint score",
                            )
                            _need(
                                _same(
                                    endpoint[context][metric],
                                    sum(scores[name][metric] for name in names) / 2,
                                ),
                                "TRAIN raw endpoint contradicts recorded cue scores",
                            )
                if diagnostics is not None:
                    _need(type(diagnostics) is dict, "invalid TRAIN endpoint decisions")
                    if all(
                        type(diagnostics.get(name)) is dict
                        and diagnostics[name].get("decisions") is not None
                        for name in names
                    ):
                        exact = []
                        for name in names:
                            decisions = diagnostics[name]["decisions"]
                            _need(
                                type(decisions) is list
                                and len(decisions) == 8
                                and all(
                                    type(decision) is dict and type(decision.get("correct")) is bool
                                    for decision in decisions
                                ),
                                "invalid TRAIN complete episode decisions",
                            )
                            for decision in decisions:
                                if (
                                    decision.get("target_class") is not None
                                    and decision.get("predicted_class") is not None
                                ):
                                    _need(
                                        all(
                                            type(decision[key]) is int and 0 <= decision[key] <= 4
                                            for key in ("target_class", "predicted_class")
                                        ),
                                        "invalid TRAIN decision classes",
                                    )
                                    _need(
                                        decision["correct"]
                                        is (
                                            decision["target_class"] == decision["predicted_class"]
                                        ),
                                        "TRAIN decision correctness contradicts classes",
                                    )
                            exact.append(all(decision["correct"] for decision in decisions))
                        _need(
                            _same(endpoint[context]["exact_sequence_accuracy"], sum(exact) / 2),
                            "TRAIN raw sequence fit contradicts recorded decisions",
                        )
    return raw


def _live_summary(
    condition: dict[str, Any],
    readout: dict[str, Any],
    episodes: list[dict[str, Any]],
    raw: dict[str, Any] | None,
) -> None:
    outcomes = readout.get("outcomes")
    if outcomes is None:
        _need(
            readout.get("metrics") is None and readout.get("initial_A_acquired") is None,
            "unobserved TRAIN outcomes have derived claims",
        )
        _need(
            condition.get("acquired_initial_condition_met") is None
            and condition.get("acquired_new_task_condition_met") is None,
            "unobserved TRAIN outcomes have acquisition premises",
        )
        _need(
            condition.get("acquired_interpretation") in (None, "not_reached"),
            "unobserved TRAIN outcomes have an interpretation",
        )
        return
    _need(
        len(episodes) == 12
        and all(
            episode.get("status") == "complete" and episode.get("state_unchanged") is True
            for episode in episodes
        ),
        "TRAIN outcomes require twelve completed unchanged episodes",
    )
    expected: list[dict[str, Any]] = []
    for point in _POINTS:
        groups = {}
        for context in ("A", "C_train"):
            selected = [
                episode
                for episode in episodes
                if episode["checkpoint_updates"] == point and episode["context"] == context
            ]
            _need(
                all(
                    type(e.get("success")) is bool
                    and type(e.get("return")) in (int, float)
                    and -1 <= e["return"] <= 1
                    for e in selected
                ),
                "invalid TRAIN live outcomes",
            )
            groups[context] = dict(
                success_rate=sum(e["success"] for e in selected) / 2,
                mean_return=sum(e["return"] for e in selected) / 2,
            )
        expected.append(dict(updates=point, by_layout=groups))
    _need(_same(outcomes, expected), "TRAIN outcomes contradict saved cue episodes")
    initial, learned, returned = [point["by_layout"] for point in expected]
    metrics = {
        metric: dict(
            new_C_gain=learned["C_train"][metric] - initial["C_train"][metric],
            old_A_loss=initial["A"][metric] - learned["A"][metric],
            return_A_gain=returned["A"][metric] - learned["A"][metric],
            return_gap=initial["A"][metric] - returned["A"][metric],
            new_C_loss_after_return=learned["C_train"][metric] - returned["C_train"][metric],
        )
        for metric in ("success_rate", "mean_return")
    }
    _need(_same(readout.get("metrics"), metrics), "TRAIN signed metrics contradict saved outcomes")
    acquired = all(e["success"] for e in episodes[:2])
    _need(
        readout.get("initial_A_acquired") is acquired,
        "TRAIN initial acquisition contradicts saved episodes",
    )
    expected_interpretation = (
        "descriptive_train_only_retention_and_relearning_not_recovery_savings"
        if acquired
        else "inconclusive_initial_A_not_acquired"
    )
    _need(
        readout.get("interpretation") == expected_interpretation,
        "TRAIN live interpretation contradicts acquisition",
    )
    if raw is not None and condition.get("acquired_initial_condition_met") is not None:
        _need(
            condition["acquired_initial_condition_met"]
            is (acquired and not raw["incomplete_initial_A_fit"]),
            "TRAIN initial premise contradicts raw/live evidence",
        )
    if condition.get("acquired_interpretation") is not None:
        interpretation = (
            (
                "descriptive_train_retention_relearning"
                if acquired and not raw["incomplete_initial_A_fit"]
                else "inconclusive_initial_A_not_acquired"
            )
            if raw is not None
            else "not_reached"
        )
        _need(
            condition["acquired_interpretation"] == interpretation,
            "TRAIN top interpretation contradicts raw/live acquisition",
        )
    if raw is not None and condition.get("acquired_new_task_condition_met") is not None:
        new_acquired = raw["endpoint_values"][1]["C_train"]["exact_sequence_accuracy"] == 1 and all(
            e["success"] for e in episodes[6:8]
        )
        _need(
            condition["acquired_new_task_condition_met"] is new_acquired,
            "TRAIN C acquisition premise contradicts raw/live evidence",
        )


def validate_acquired_report(condition: dict[str, Any]) -> list[dict[str, Any]]:
    """Refuse mixed selectors and contradictory recorded TRAIN evidence."""
    protocol = condition["protocol"]
    identifier = protocol.get("protocol_id")
    _need(identifier in PROTOCOL_CONTROLS, "unsupported acquired TRAIN protocol")
    _need(
        not any(
            key in condition
            for key in ("conditions", "policy_evaluation", "train_policy_diagnostics")
        ),
        "mixed TRAIN/final readout selectors",
    )
    _need(
        condition.get("source_role") in (None, "train")
        and condition.get("readout_key") in (None, "acquired_return_diagnostics"),
        "mislabeled TRAIN selector or role",
    )
    control = PROTOCOL_CONTROLS[identifier]
    _need(
        protocol.get("diagnostic_data_role") == "train"
        and protocol.get("action_conditioning") == "legal_actions_v1"
        and protocol.get("task_sequence") == _SEQUENCE,
        "invalid TRAIN protocol role or policy",
    )
    _need(protocol.get("diagnostic_points") == list(_POINTS), "invalid TRAIN diagnostic points")
    _need(
        protocol.get("replay_control") == control and condition.get("replay_control") == control,
        "TRAIN protocol/control disagreement",
    )
    _source_counts(condition, control)
    stream_result = condition.get("stream_result")
    if stream_result is not None:
        _need(
            type(stream_result) is dict
            and stream_result.get("evaluation_scope") == "training_only"
            and stream_result.get("score_forward_calls") == 0,
            "TRAIN stream contains final evaluation",
        )
    training = condition.get("training_summary")
    if training is not None:
        _need(type(training) is dict, "invalid TRAIN source summary")
        if training.get("roles") is not None:
            _need(
                type(training["roles"]) is list
                and all(role == "train" for role in training["roles"]),
                "TRAIN source has nontraining roles",
            )
    raw = _raw_summary(condition)
    readout = condition.get("acquired_return_diagnostics")
    if readout is None:
        _need(condition["status"] != "complete", "complete TRAIN report has no readout")
        _need(
            condition.get("acquired_initial_condition_met") is None
            and condition.get("acquired_new_task_condition_met") is None
            and condition.get("acquired_interpretation") in (None, "not_reached"),
            "unobserved TRAIN readout has acquisition claims",
        )
        return []
    _need(type(readout) is dict and readout.get("status") in _STATUSES, "invalid TRAIN readout")
    _need(
        readout.get("classification") == "train_resubstitution_acquired_task_return",
        "invalid TRAIN classification",
    )
    _need(
        readout.get("scientific_advantage_verified") in (None, False),
        "TRAIN readout does not establish scientific advantage",
    )
    _need(
        readout.get("source_role") == "train"
        and readout.get("method") == "recurrent_bc"
        and readout.get("action_conditioning") == "legal_actions_v1"
        and readout.get("replay_enabled") is (control is not None)
        and readout.get("replay_control") == control,
        "invalid TRAIN readout role/control",
    )
    clocks = readout.get("checkpoint_clocks")
    if control is None:
        _need(clocks is None, "off TRAIN native checkpoint counts were not recorded")
    else:
        _need(type(clocks) is list and len(clocks) == 3, "missing recorded TRAIN replay clocks")
        for clock, point in zip(clocks, _POINTS, strict=True):
            _exact_clock(clock, point)
    episodes = readout.get("episodes")
    _need(
        type(episodes) is list
        and len(episodes) <= 12
        and (readout["status"] != "complete" or len(episodes) == 12),
        "invalid TRAIN episode slots",
    )
    planned = [
        (point, context, cue)
        for point in _POINTS
        for context in ("A", "C_train")
        for cue in (1, -1)
    ]
    for episode, (point, context, cue) in zip(episodes, planned, strict=False):
        _need(
            type(episode) is dict
            and episode.get("source_role") == "train"
            and type(episode.get("checkpoint_updates")) is int
            and episode["checkpoint_updates"] == point
            and episode.get("context") == context
            and type(episode.get("cue")) is int
            and episode["cue"] == cue
            and episode.get("layout_partition") == "known_layout",
            "invalid TRAIN episode identity/role",
        )
        if episode.get("identity") is not None:
            identity = episode["identity"]
            _need(
                type(identity) is dict
                and identity.get("method") == "recurrent_bc"
                and identity.get("seed") == 17
                and identity.get("task_sequence") == _SEQUENCE
                and identity.get("replay_enabled") is (control is not None),
                "TRAIN evaluation identity disagrees",
            )
        if episode.get("layout") is not None:
            _need(
                episode.get("action_conditioning") == "legal_actions_v1"
                and episode.get("max_steps") == 12,
                "invalid TRAIN action law",
            )
            layout = dict(
                layout_id=context,
                rotation=0 if context == "A" else 2,
                branch_row=2 if context == "A" else 3,
                swap_keys=context != "A",
                appearance=0.0 if context == "A" else -0.75,
            )
            _need(_same(episode["layout"], layout), "TRAIN layout disagrees with fixed protocol")
    work = readout.get("work")
    if work is not None:
        _need(type(work) is dict, "invalid TRAIN work")
        for key in ("live_callback_attempts", "live_callback_returns", "live_applied_steps"):
            _count(work, key, 144)
        _count(work, "live_sessions_started", 12)
        for key, field in (
            ("live_callback_attempts", "prediction_calls_attempted"),
            ("live_callback_returns", "prediction_calls_returned"),
            ("live_applied_steps", "steps_executed"),
        ):
            if work.get(key) is not None and all(
                type(episode.get(field)) is int for episode in episodes
            ):
                _need(
                    work[key] == sum(episode[field] for episode in episodes),
                    "TRAIN work contradicts saved episodes",
                )
    _count(readout, "live_sessions_started", 12)
    if readout.get("live_sessions_started", 0) or any(
        episode.get("layout") is not None for episode in episodes
    ):
        _source_counts(condition, control, observed=True)
    if condition["status"] == "complete":
        _need(
            readout["status"] == "complete"
            and raw is not None
            and readout.get("outcomes") is not None,
            "complete TRAIN report has incomplete readouts",
        )
    for point in condition.get("checkpoints") or []:
        _need(type(point) is dict, "invalid TRAIN raw checkpoint")
        scores = point.get("scores")
        if scores is not None:
            _need(type(scores) is dict and len(scores) <= 6, "invalid TRAIN raw scores")
            for score in scores.values():
                _need(type(score) is dict, "invalid TRAIN raw context score")
                for metric in ("ce", "accuracy"):
                    if score.get(metric) is not None:
                        value = score[metric]
                        _need(
                            type(value) in (int, float)
                            and value >= 0
                            and (metric == "ce" or value <= 1),
                            "invalid TRAIN raw metric range",
                        )
        if point.get("clocks") is not None:
            _need(
                control is not None
                and type(point.get("updates")) is int
                and 0 <= point["updates"] <= 320,
                "invalid TRAIN raw clocks",
            )
            _exact_clock(point["clocks"], point["updates"])
    _live_summary(condition, readout, episodes, raw)
    return episodes
