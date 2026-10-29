"""Validate saved fixed-A MiniGrid evidence and project public-view playback.

Inputs are bounded JSON records, never environments or opaque native states.
Outputs own the unchanged source and exact observation/decision/physics pointers.
The seven-by-seven view is recorded egocentric input, not an inferred world map.
No learner, Torch, physics simulation, training, filesystem or network is used.
"""

from copy import deepcopy
import json
from math import fsum, isfinite, log
from typing import Any

PROTOCOL_ID = "minigrid_fixed_a128_acquisition_v1"
METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc", "recurrent_bc")
POINTS = (0, 32, 64, 96, 128)
STATUSES = {"complete", "incomplete", "failed", "running", "not_run", "trained"}
MISSION = "remember the cue, take its key, unlock the neutral door and reach the goal"
ACTIONS = ("left", "right", "forward", "pickup", "drop", "toggle", "done")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, name: str, low: float | None = None, high: float | None = None) -> float:
    _require(type(value) in (int, float), f"invalid {name}")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"nonfinite {name}") from error
    _require(isfinite(number), f"nonfinite {name}")
    _require(
        (low is None or number >= low) and (high is None or number <= high), f"invalid {name} range"
    )
    return number


def _integer(value: Any, low: int, high: int, name: str) -> int:
    _require(type(value) is int and low <= value <= high, f"invalid {name}")
    return value


def _boolean(value: Any, name: str) -> bool:
    _require(type(value) is bool, f"invalid {name}")
    return value


def _status(value: Any) -> bool:
    return type(value) is str and value in STATUSES


def _records(value: Any, maximum: int, name: str) -> list[dict[str, Any]]:
    _require(
        type(value) is list and len(value) <= maximum and all(type(v) is dict for v in value),
        f"invalid bounded {name}",
    )
    return value


def _close(left: Any, right: float, name: str) -> None:
    _require(abs(_number(left, name) - right) <= 1e-10, f"recorded {name} contradicts evidence")


def _envelope(report: Any, source_label: str) -> None:
    _require(
        type(source_label) is str
        and bool(source_label.strip())
        and source_label == source_label.strip()
        and len(source_label.encode("utf8")) <= 1024,
        "invalid source label",
    )
    _require(type(report) is dict, "MiniGrid report must be a JSON object")
    stack, nodes = [(report, 0)], 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        _require(nodes <= 500000 and depth <= 64, "MiniGrid JSON structure exceeds bounds")
        if type(value) is dict:
            _require(all(type(key) is str for key in value), "JSON keys must be text")
            stack.extend((child, depth + 1) for child in value.values())
        elif type(value) is list:
            stack.extend((child, depth + 1) for child in value)
        elif type(value) in (int, float):
            _number(value, "JSON number")
        else:
            _require(type(value) in (str, bool, type(None)), "non-JSON source value")
    _require(
        len(
            json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
                "utf8"
            )
        )
        <= 8 * 1024**2,
        "MiniGrid report exceeds8MiB",
    )


def decode_public_observation(observation: Any) -> tuple[list[list[list[int]]], int]:
    """Invert the recorded148 encoding, preserving native [x][y][channel] order."""
    _require(
        type(observation) is list and len(observation) == 148, "observation requires148 coordinates"
    )
    values = [_number(v, "observation", -1, 1) for v in observation]
    decoded = []
    for index, value in enumerate(values[:147]):
        raw = (value + 1.0) * (10, 5, 2)[index % 3] / 2.0
        _require(abs(raw - round(raw)) <= 1e-8, "observation does not encode public integer pixels")
        decoded.append(round(raw))
    direction = (values[147] + 1.0) * 1.5
    _require(abs(direction - round(direction)) <= 1e-8, "invalid encoded direction")
    image = [
        [[decoded[(x * 7 + y) * 3 + channel] for channel in range(3)] for y in range(7)]
        for x in range(7)
    ]
    return image, round(direction)


def _initial(observation: Any) -> list[float]:
    _require(
        type(observation) is dict and set(observation) == {"image", "direction", "mission"},
        "invalid initial public observation",
    )
    _require(
        observation["mission"] == MISSION and type(observation["mission"]) is str,
        "invalid fixed mission",
    )
    image = observation["image"]
    _require(
        type(image) is list
        and len(image) == 7
        and all(type(column) is list and len(column) == 7 for column in image),
        "invalid initial7x7 image",
    )
    values: list[float] = []
    for column in image:
        for pixel in column:
            _require(type(pixel) is list and len(pixel) == 3, "invalid image channels")
            values.extend(
                2.0 * _integer(value, 0, scale, "image code") / scale - 1.0
                for value, scale in zip(pixel, (10, 5, 2), strict=True)
            )
    values.append(2.0 * _integer(observation["direction"], 0, 3, "direction") / 3.0 - 1.0)
    return values


def _probabilities(value: Any) -> tuple[list[float], int]:
    _require(type(value) is list and len(value) == 7, "prediction requires seven probabilities")
    probabilities = [_number(v, "probability", 0, 1) for v in value]
    _require(abs(fsum(probabilities) - 1.0) <= 1e-6, "probabilities do not sum to one")
    return probabilities, max(range(7), key=probabilities.__getitem__)


def _counts(record: Any) -> None:
    if record is None:
        return
    _require(type(record) is dict, "invalid recorded work object")
    for key, value in record.items():
        if value is None:
            continue
        if key == "failure_work_may_be_unobserved":
            _boolean(value, key)
        elif key.endswith("seconds"):
            _number(value, key, 0)
        elif type(value) in (int, float, bool):
            _integer(value, 0, 2**53 - 1, key)


def _resources(report: dict[str, Any]) -> None:
    if report.get("seconds") is not None:
        _number(report["seconds"], "seconds", 0)
    retained = report.get("retained_input_array_bytes")
    if retained is not None:
        _require(type(retained) is dict, "invalid retained array byte counts")
        expected = dict(
            original_sources=6 * 9 * 149 * 8,
            training_raw=128 * 9 * 149 * 8,
            training_history=128 * 9 * 297 * 8,
            raw_history_a=2 * 9 * 297 * 8,
        )
        for key, value in retained.items():
            if value is not None:
                _integer(value, 0, 2**53 - 1, "retained array bytes")
                if key in expected and len(report.get("training_sources") or []) == 6:
                    _require(
                        value == expected[key],
                        "retained array byte count contradicts fixed source geometry",
                    )
    rss = report.get("observed_process_rss")
    if rss is not None:
        _require(type(rss) is dict, "invalid observed RSS")
        for key in ("pid", "start_bytes", "peak_bytes", "sample_count"):
            if rss.get(key) is not None:
                _integer(rss[key], 0, 2**53 - 1, key)
        if rss.get("interval_seconds") is not None:
            _number(rss["interval_seconds"], "RSS interval", 0)
        if rss.get("peak_bytes") is not None and rss.get("start_bytes") is not None:
            _require(rss["peak_bytes"] >= rss["start_bytes"], "RSS peak precedes initial RSS")
    source_reports = report.get("source_reports")
    if source_reports is None:
        return
    for source in _records(source_reports, 6, "teacher reports"):
        for key, maximum in (
            ("environment_steps", 16),
            ("environment_steps_attempted", 16),
            ("reset_calls", 1),
        ):
            if source.get(key) is not None:
                _integer(source[key], 0, maximum, key)
        if source.get("total_return") is not None:
            _number(source["total_return"], "teacher return", -16, 16)
        transitions = source.get("transitions")
        if transitions is not None:
            transitions = _records(transitions, 16, "teacher transitions")
            for transition in transitions:
                _require(
                    transition.get("applied") is None or type(transition["applied"]) is bool,
                    "invalid teacher application flag",
                )
            if source.get("environment_steps") is not None:
                _require(
                    source["environment_steps"]
                    == sum(t.get("applied") is True for t in transitions),
                    "teacher step count contradicts applied transitions",
                )
            if source.get("environment_steps_attempted") is not None:
                _require(
                    source["environment_steps_attempted"] == len(transitions),
                    "teacher attempts contradict transitions",
                )
        if (
            source.get("environment_steps") is not None
            and source.get("environment_steps_attempted") is not None
        ):
            _require(
                source["environment_steps"] <= source["environment_steps_attempted"],
                "teacher returns exceed attempts",
            )


def _sources(report: dict[str, Any], complete: bool) -> dict[str, list[int]]:
    sources = _records(report.get("training_sources", []), 6, "training sources")
    _require(
        not complete or len(sources) == 6, "complete report requires six original TRAIN sources"
    )
    targets: dict[str, list[int]] = {}
    pairs = [(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")]
    for episode, (context, cue) in zip(sources, pairs, strict=False):
        _require(
            episode.get("role") == "train"
            and type(episode.get("role")) is str
            and episode.get("context") == context
            and episode.get("num_classes") == 7
            and type(episode.get("num_classes")) is int,
            "invalid TRAIN source identity",
        )
        _require(
            type(episode.get("episode_id")) is str and episode["episode_id"].endswith(f"-{cue}"),
            "TRAIN cue identity mismatch",
        )
        transitions = _records(episode.get("transitions"), 9, "source transitions")
        _require(len(transitions) == 9, "TRAIN source requires nine complete steps")
        prior = None
        actions = []
        for index, transition in enumerate(transitions):
            _require(
                _integer(transition.get("step"), 0, 8, "source step") == index
                and type(transition.get("num_classes")) is int
                and transition["num_classes"] == 7,
                "source step/classes mismatch",
            )
            decode_public_observation(transition.get("observation"))
            decode_public_observation(transition.get("next_observation"))
            _require(
                prior is None or prior == transition["observation"], "source continuity mismatch"
            )
            _require(
                _boolean(transition.get("terminal"), "source terminal") == (index == 8),
                "invalid source terminal position",
            )
            _number(transition.get("reward"), "source reward", -1, 1)
            actions.append(_integer(transition.get("action"), 0, 6, "source action"))
            prior = transition["next_observation"]
        if context == "A":
            targets[cue] = actions
    return targets


def _method(row: dict[str, Any], method: str, complete: bool) -> None:
    _require(
        row.get("method") == method and type(row.get("method")) is str,
        "method order/identity mismatch",
    )
    _require(_status(row.get("status")), "invalid method status")
    for key, maximum in (
        ("charged_updates", 128),
        ("native_training_examples", 1152),
        ("wake_calls_attempted", 128),
        ("wake_calls_returned", 128),
    ):
        if row.get(key) is not None:
            _integer(row[key], 0, maximum, key)
    if complete or row["status"] in ("trained", "complete"):
        _require(
            row.get("checkpoint_updates") == list(POINTS)
            and row.get("charged_updates") == 128
            and row.get("native_training_examples") == 1152
            and row.get("source_unchanged") is True
            and row.get("actor_unchanged") is True,
            "completed training clocks or isolation mismatch",
        )
        stream: Any = row.get("stream_result")
        _require(
            type(stream) is dict
            and stream.get("completed_updates") == 128
            and stream.get("training_examples") == 1152
            and stream.get("score_forward_calls") == 0
            and stream.get("checkpoints") == []
            and stream.get("evaluation_scope") == "training_only"
            and stream.get("num_classes") == 7,
            "invalid completed train-only result",
        )
        _require(
            type(stream.get("applied_updates")) is list and len(stream["applied_updates"]) == 128,
            "completed training requires128 paid receipts",
        )
        for key, expected in (
            ("completed_updates", 128),
            ("training_examples", 1152),
            ("score_forward_calls", 0),
            ("num_classes", 7),
        ):
            _require(
                type(stream.get(key)) is int and stream[key] == expected,
                "completed stream count type/value mismatch",
            )
        _require(
            all(type(point) is int for point in row["checkpoint_updates"]),
            "checkpoint counts require native integers",
        )
    partial: Any = row.get("partial_stream")
    if partial is not None:
        _require(type(partial) is dict, "invalid partial stream")
        for key, maximum in (
            ("completed_wake_updates", 128),
            ("native_candidate_updates", 128),
            ("native_candidate_training_examples", 1152),
        ):
            if partial.get(key) is not None:
                _integer(partial[key], 0, maximum, key)
        if row["status"] in ("trained", "complete"):
            _require(
                partial.get("completed_wake_updates")
                == partial.get("native_candidate_updates")
                == 128
                and partial.get("native_candidate_training_examples") == 1152,
                "completed native clock mismatch",
            )
    for key in ("source_unchanged", "actor_unchanged"):
        if row.get(key) is not None:
            _boolean(row[key], key)
    _counts(row.get("work"))


def _scores(evaluation: dict[str, Any], targets: dict[str, list[int]]) -> None:
    scores = _records(evaluation.get("scores", []), 2, "raw scores")
    complete = evaluation["status"] == "complete"
    _require(not complete or len(scores) == 2, "complete readout requires both raw cue scores")
    for score, cue in zip(scores, ("blue", "red"), strict=False):
        _require(score.get("context") == "A" and score.get("cue") == cue, "raw cue order mismatch")
        probabilities: Any = score.get("probabilities")
        _require(
            type(probabilities) is list and len(probabilities) == 9,
            "raw score requires nine probability rows",
        )
        validated = [_probabilities(row) for row in probabilities]
        _require(cue in targets, "raw scores require the original A labels")
        labels = targets[cue]
        predictions = [action for _, action in validated]
        ce = (
            fsum(
                -log(max(row[label], 1e-12))
                for (row, _), label in zip(validated, labels, strict=True)
            )
            / 9
        )
        _close(score.get("ce"), ce, "raw CE")
        _close(
            score.get("accuracy"),
            sum(p == t for p, t in zip(predictions, labels, strict=True)) / 9,
            "raw accuracy",
        )
        _close(score.get("exact_sequence_accuracy"), float(predictions == labels), "raw sequence")
        if "predicted_actions" in score:
            _require(
                score["predicted_actions"] == predictions and score.get("target_actions") == labels,
                "raw decision arrays contradict probabilities/source labels",
            )
            _require(
                all(
                    type(value) is int
                    for value in (*score["predicted_actions"], *score["target_actions"])
                ),
                "raw actions require native integers",
            )


def _evaluation_work(evaluation: dict[str, Any], episodes: list[dict[str, Any]]) -> None:
    work: Any = evaluation.get("work")
    _require(type(work) is dict, "readout requires recorded work")
    endpoint = evaluation["updates"] in (0, 128)
    calls = (
        ("raw_score_forwards_attempted", "raw_score_forwards_completed")
        if endpoint
        else ("score_forward_calls_attempted", "score_forward_calls_completed")
    )
    rows = (
        ("raw_score_rows_attempted", "raw_score_rows_completed")
        if endpoint
        else ("score_rows_attempted", "score_rows_completed")
    )
    attempted = _integer(work.get(calls[0]), 0, 2, "score attempts")
    returned = _integer(work.get(calls[1]), 0, attempted, "score returns")
    _require(
        returned >= len(evaluation.get("scores", [])) and attempted - returned <= 1,
        "score costs contradict observed prefix",
    )
    _require(
        _integer(work.get(rows[0]), 0, 18, "score attempted rows") == 9 * attempted
        and _integer(work.get(rows[1]), 0, 18, "score returned rows") == 9 * returned,
        "score row costs contradict forward calls",
    )
    complete = evaluation["status"] == "complete"
    _require(
        not complete or attempted == returned == 2, "complete readout requires two paid scores"
    )
    if not endpoint:
        return
    for key, episode_key in (
        ("live_prediction_calls_attempted", "prediction_calls_attempted"),
        ("live_prediction_calls_returned", "prediction_calls_returned"),
    ):
        _require(
            _integer(work.get(key), 0, 32, key)
            == sum(episode[episode_key] for episode in episodes),
            "live prediction costs contradict episode records",
        )
    resets = _integer(work.get("environment_resets_attempted"), 0, 2, "reset attempts")
    reset_returns = _integer(work.get("environment_resets_completed"), 0, resets, "reset returns")
    environments = [
        episode["environment"] for episode in episodes if episode.get("environment") is not None
    ]
    if any(
        episode.get("environment") is None and episode["status"] == "failed" for episode in episodes
    ):
        _require(
            work.get("failure_work_may_be_unobserved") is True,
            "failed native construction/readback must retain unknown work",
        )
    _require(
        len(environments) <= reset_returns <= resets <= len(episodes),
        "reset costs contradict observed episodes",
    )
    for key, environment_key in (
        ("environment_steps_attempted", "environment_steps_attempted"),
        ("environment_steps_returned", "environment_steps"),
    ):
        recorded = _integer(work.get(key), 0, 32, key)
        known = sum(environment[environment_key] for environment in environments)
        _require(
            recorded >= known and (len(environments) != len(episodes) or recorded == known),
            "physics costs contradict recorded transitions",
        )
    sessions = _integer(work.get("live_sessions_started"), 0, 2, "session starts")
    _require(
        sessions <= len(episodes) and (evaluation["method"] == "recurrent_bc" or sessions == 0),
        "native session cost mismatch",
    )
    if complete:
        _require(
            resets == reset_returns == 2
            and sessions == (2 if evaluation["method"] == "recurrent_bc" else 0),
            "complete live reset/session cost mismatch",
        )


def _frame(
    observation: list[float],
    pointer: str,
    decision: str | None,
    transition: str | None,
    features: str | None,
) -> dict[str, Any]:
    image, direction = decode_public_observation(observation)
    return dict(
        observation=deepcopy(observation),
        image=image,
        direction=direction,
        pointer=pointer,
        decision_pointer=decision,
        transition_pointer=transition,
        features_pointer=features,
    )


def _episode(
    record: dict[str, Any],
    pointer: str,
    method: str,
    updates: int,
    cue: str,
    *,
    context: str = "A",
    role: str = "train",
    branch_row: int = 3,
    allow_observed_failure_tail: bool = False,
    expected_layout: dict[str, Any] | None = None,
    expected_episode_id: str | None = None,
) -> dict[str, Any]:
    _require(
        record.get("context") == context
        and record.get("cue") == cue
        and record.get("source_role") == role,
        "live identity/role mismatch",
    )
    _require(_status(record.get("status")), "invalid live status")
    complete = record["status"] == "complete"
    predictions = _records(record.get("predictions", []), 16, "live predictions")
    attempted = _integer(record.get("prediction_calls_attempted"), 0, 16, "prediction attempts")
    returned = _integer(record.get("prediction_calls_returned"), 0, attempted, "prediction returns")
    _require(
        (
            attempted == len(predictions)
            or (
                allow_observed_failure_tail
                and record["status"] == "failed"
                and len(predictions) == attempted + 1
                and predictions[-1].get("probabilities") is None
                and predictions[-1].get("action") is None
            )
        )
        and returned >= max(0, attempted - 1),
        "prediction counters contradict the observed prefix",
    )
    environment: Any = record.get("environment")
    _require(environment is None or type(environment) is dict, "invalid recorded environment")
    _require(
        not complete or environment is not None,
        "complete episode requires native environment evidence",
    )
    _require(
        environment is not None or not predictions or type(record.get("readback_error")) is str,
        "policy observations without native evidence require a recorded readback fault",
    )
    initial: Any = None
    transitions: list[dict[str, Any]] = []
    if environment is not None:
        if environment.get("closed") is not None:
            _boolean(environment["closed"], "native closed flag")
        _require(
            environment.get("environment_id") == "Circadian-MiniGrid-CueKeyDoor-v1"
            and environment.get("encoding_id")
            == "minigrid_cue_key_door_public_image_direction_148_v1",
            "unsupported native environment/encoding",
        )
        _require(
            environment.get("role") == role
            and environment.get("context") == context
            and environment.get("cue") == cue
            and environment.get("observation_dim") == 148
            and environment.get("num_classes") == 7,
            "native role/context/cue/width mismatch",
        )
        _require(
            environment.get("seed") == 17
            and environment.get("max_steps") == 16
            and environment.get("action_names") == list(ACTIONS),
            "native fixed policy/physics identity mismatch",
        )
        for key, expected in (
            ("seed", 17),
            ("max_steps", 16),
            ("observation_dim", 148),
            ("num_classes", 7),
        ):
            _require(
                type(environment.get(key)) is int and environment[key] == expected,
                "native identity count type mismatch",
            )
        _require(
            _status(environment.get("status"))
            and environment.get("episode_id")
            == (
                expected_episode_id
                if expected_episode_id is not None
                else f"live-{method}-{updates}-{context}-{cue}"
            ),
            "native checkpoint episode identity mismatch",
        )
        _require(
            _integer(environment.get("reset_calls"), 0, 1, "native reset count") == 1,
            "observed native episode requires one reset",
        )
        layout = environment.get("layout")
        geometry = (
            expected_layout
            if expected_layout is not None
            else dict(
                context=context,
                rotation=0,
                branch_row=branch_row,
                swap_keys=False,
                wall_color="grey",
            )
        )
        _require(
            type(layout) is dict and layout == geometry,
            "native A geometry mismatch",
        )
        _require(
            all(type(layout[key]) is type(value) for key, value in geometry.items()),
            "native layout types mismatch",
        )
        if environment.get("initial_observation") is not None:
            initial = _initial(environment["initial_observation"])
        transitions = _records(environment.get("transitions", []), 16, "native transitions")
    _require(len(transitions) <= len(predictions), "physics has no corresponding policy decision")
    for index, prediction in enumerate(predictions):
        observation: Any = prediction.get("observation")
        decode_public_observation(observation)
        if initial is None:
            initial = observation
        if index == 0:
            _require(
                all(abs(a - b) <= 1e-12 for a, b in zip(initial, observation, strict=True)),
                "first policy observation contradicts native reset",
            )
        probabilities, action = prediction.get("probabilities"), prediction.get("action")
        if probabilities is None:
            _require(
                not complete and index == len(predictions) - 1 and action is None,
                "missing nonpartial policy probabilities",
            )
        else:
            _, chosen = _probabilities(probabilities)
            _require(
                index < returned and _integer(action, 0, 6, "chosen action") == chosen,
                "action is not the first raw argmax",
            )
        if method == "recurrent_bc":
            _require("features" not in prediction, "RNN evidence must retain original148 inputs")
        else:
            features: Any = prediction.get("features")
            _require(
                type(features) is list and len(features) == 296,
                "feedforward prediction requires actual296 features",
            )
            _require(
                features == initial + observation,
                "history input must be own initial plus current public frame",
            )
            for value in features:
                _number(value, "history feature", -1, 1)
    frames = []
    if initial is not None and predictions:
        first = predictions[0]["observation"] if predictions else initial
        first_pointer = (
            f"{pointer}/predictions/0/observation"
            if predictions
            else f"{pointer}/environment/initial_observation"
        )
        frames.append(
            _frame(
                first,
                first_pointer,
                f"{pointer}/predictions/0" if predictions else None,
                f"{pointer}/environment/transitions/0" if transitions else None,
                f"{pointer}/predictions/0/features"
                if predictions and method != "recurrent_bc"
                else None,
            )
        )
    prior, ended, rewards, normal_returns = initial, False, [], 0
    for index, transition in enumerate(transitions):
        _require(not ended, "native work continues after terminal/truncated ending")
        prediction = predictions[index]
        _require(
            transition.get("observation") == prediction["observation"]
            and (prior is None or prior == transition["observation"]),
            "policy/native observation or continuity mismatch",
        )
        _require(
            prediction.get("action") is not None
            and transition.get("action") == prediction["action"]
            and type(transition.get("action")) is int,
            "native action differs from recorded policy",
        )
        _require(
            transition.get("attempted") is True
            and (transition.get("applied") is None or transition.get("applied") is True),
            "invalid native action application",
        )
        if transition.get("applied") is True:
            normal_returns += 1
        else:
            _require(
                not complete and index == len(transitions) - 1,
                "unreturned native call is not the failed prefix ending",
            )
        next_observation = transition.get("next_observation")
        reward = transition.get("reward")
        flags = (transition.get("terminated"), transition.get("truncated"))
        if transition.get("applied") is None:
            _require(
                all(value is None for value in (next_observation, reward, *flags)),
                "unreturned native call has invented result values",
            )
        if reward is not None:
            rewards.append(_number(reward, "native reward", -1, 1))
        if any(value is None for value in (next_observation, reward, *flags)):
            _require(
                not complete and index == len(transitions) - 1, "missing nonpartial native result"
            )
        for flag in flags:
            if flag is not None:
                _boolean(flag, "native ending flag")
        ended = any(flag is True for flag in flags)
        if next_observation is not None:
            decode_public_observation(next_observation)
            next_prediction = predictions[index + 1] if index + 1 < len(predictions) else None
            _require(not ended or next_prediction is None, "policy predicts after native ending")
            if next_prediction is not None:
                _require(
                    next_prediction["observation"] == next_observation,
                    "next policy input differs from returned public observation",
                )
            decision_pointer = (
                f"{pointer}/predictions/{index + 1}" if next_prediction is not None else None
            )
            transition_pointer = (
                (
                    f"{pointer}/environment/transitions/{index + 1}"
                    if index + 1 < len(transitions)
                    else None
                )
                if next_prediction is not None
                else f"{pointer}/environment/transitions/{index}"
            )
            frames.append(
                _frame(
                    next_observation,
                    f"{pointer}/environment/transitions/{index}/next_observation",
                    decision_pointer,
                    transition_pointer,
                    f"{decision_pointer}/features"
                    if decision_pointer and method != "recurrent_bc"
                    else None,
                )
            )
            prior = next_observation
    if environment is not None:
        _require(
            _integer(environment.get("environment_steps_attempted"), 0, 16, "native attempts")
            == len(transitions)
            and _integer(environment.get("environment_steps"), 0, 16, "native returns")
            == normal_returns,
            "native counters contradict transitions",
        )
        _close(environment.get("total_return"), fsum(rewards), "native total return")
        _boolean(environment.get("terminated"), "native terminated")
        _boolean(environment.get("truncated"), "native truncated")
        if ended:
            _require(
                (environment["terminated"], environment["truncated"])
                == (transitions[-1]["terminated"], transitions[-1]["truncated"]),
                "native final ending flags mismatch",
            )
        else:
            _require(
                not environment["terminated"] and not environment["truncated"],
                "unobserved native ending",
            )
        if environment["truncated"]:
            _require(normal_returns == 16, "truncation precedes the fixed native limit")
    if complete:
        _require(
            environment is not None
            and environment.get("closed") is True
            and environment.get("status") == "complete"
            and ended
            and attempted == returned == len(transitions)
            and record.get("state_unchanged") is True,
            "complete live episode lacks complete immutable native evidence",
        )
        success = bool(
            environment["terminated"]
            and not environment["truncated"]
            and environment["total_return"] > 0
        )
        _require(record.get("success") is success, "live success contradicts native ending/return")
        _close(record.get("return"), environment["total_return"], "episode return")
    else:
        for key in ("success", "state_unchanged"):
            if record.get(key) is not None:
                _boolean(record[key], key)
        if record.get("return") is not None:
            _require(environment is not None, "return without native evidence")
            _close(record["return"], environment["total_return"], "partial episode return")
        _require(record.get("success") is not True or ended, "unobserved partial success")
    return dict(
        pointer=pointer,
        record=deepcopy(record),
        updates=updates,
        cue=cue,
        frames=frames,
        initial_image=decode_public_observation(initial)[0] if initial is not None else None,
    )


def _acquisition(report: dict[str, Any], evaluations: list[dict[str, Any]]) -> None:
    acquisition: Any = report.get("acquisition")
    if acquisition is None:
        _require(report["status"] != "complete", "complete report lacks acquisition summary")
        return
    _require(type(acquisition) is dict, "invalid acquisition summary")
    rows = _records(acquisition.get("methods"), 5, "acquisition methods")
    _require(len(rows) == 5, "acquisition summary requires all methods")
    derived = {}
    for row, method in zip(rows, METHODS, strict=True):
        _require(row.get("method") == method, "acquisition method mismatch")
        points = {r["updates"]: r for r in evaluations if r["method"] == method}
        available = (
            report["status"] == "complete"
            and len(points) == 5
            and all(r["status"] == "complete" for r in points.values())
        )
        if not available:
            _require(
                row.get("status") == "unavailable"
                and all(
                    row.get(key) is None
                    for key in (
                        "raw_both_cues_exact",
                        "live_both_cues_goals",
                        "joint_acquisition",
                        "ce_gain",
                        "live_success_count_change",
                        "mean_return_gain",
                    )
                ),
                "unobserved acquisition must remain unavailable",
            )
            continue
        initial, final = points[0], points[128]
        raw = all(s["exact_sequence_accuracy"] == 1 for s in final["scores"])
        live = all(e["success"] is True for e in final["episodes"])
        _require(
            row.get("status") == "complete"
            and row.get("raw_both_cues_exact") is raw
            and row.get("live_both_cues_goals") is live
            and row.get("joint_acquisition") is (raw and live),
            "acquisition premise contradicts observed episodes",
        )
        expected = dict(
            ce_gain=fsum(s["ce"] for s in initial["scores"]) / 2
            - fsum(s["ce"] for s in final["scores"]) / 2,
            live_success_count_change=sum(e["success"] for e in final["episodes"])
            - sum(e["success"] for e in initial["episodes"]),
            mean_return_gain=fsum(e["return"] for e in final["episodes"]) / 2
            - fsum(e["return"] for e in initial["episodes"]) / 2,
        )
        for key, value in expected.items():
            _close(row.get(key), value, key)
        derived[method] = expected
    comparisons = acquisition.get("signed_comparisons")
    _require(
        type(comparisons) is dict
        and set(comparisons) == {f"gated_cpc-minus-{control}" for control in ("pc", "neutral_cpc")},
        "invalid fixed signed comparisons",
    )
    for control in ("pc", "neutral_cpc"):
        values = comparisons[f"gated_cpc-minus-{control}"]
        _require(
            type(values) is dict
            and set(values) == {"ce_gain", "live_success_count_change", "mean_return_gain"},
            "invalid signed comparison metrics",
        )
        for key, value in values.items():
            if "gated_cpc" not in derived or control not in derived:
                _require(value is None, "unobserved comparison must remain null")
            else:
                _close(value, derived["gated_cpc"][key] - derived[control][key], key)


def project_minigrid_report(report: dict[str, Any], *, source_label: str) -> dict[str, Any]:
    """Project only the declared fixed-A artifact, preserving every recorded slot."""
    _envelope(report, source_label)
    _require(_status(report.get("status")), "invalid MiniGrid report status")
    protocol: Any = report.get("protocol")
    _require(
        type(protocol) is dict and protocol.get("protocol_id") == PROTOCOL_ID,
        "unsupported MiniGrid protocol",
    )
    _require(
        not any(
            key in report
            for key in ("conditions", "policy_evaluation", "acquired_return_diagnostics")
        ),
        "mixed MiniGrid/custom report selectors",
    )
    _require(
        protocol.get("methods") == list(METHODS) and protocol.get("checkpoints") == list(POINTS),
        "fixed method/checkpoint protocol mismatch",
    )
    _require(
        protocol.get("seed") == 17
        and protocol.get("hidden_dim") == 8
        and protocol.get("num_classes") == 7
        and protocol.get("feature_dim_by_method")
        == {method: 148 if method == "recurrent_bc" else 296 for method in METHODS},
        "fixed native dimensions/seed mismatch",
    )
    for key, expected in (("seed", 17), ("hidden_dim", 8), ("num_classes", 7)):
        _require(
            type(protocol[key]) is int and protocol[key] == expected,
            "protocol dimensions require native integers",
        )
    _require(
        all(type(value) is int for value in protocol["feature_dim_by_method"].values())
        and all(type(point) is int for point in protocol["checkpoints"]),
        "protocol widths/checkpoints require native integers",
    )
    complete = report["status"] == "complete"
    methods = _records(report.get("methods"), 5, "methods")
    _require(len(methods) == 5, "report requires five method slots")
    targets = _sources(report, complete)
    _resources(report)
    conditions: list[dict[str, Any]] = []
    for index, (row, method) in enumerate(zip(methods, METHODS, strict=True)):
        _method(row, method, complete)
        conditions.append(
            dict(
                key=method,
                pointer=f"/methods/{index}",
                status=row["status"],
                source_role="train",
                episodes=[],
                unknown=dict(topology=None, lineage=None, sleep_proposals=None),
            )
        )
    evaluations = _records(report.get("evaluations", []), 25, "evaluations")
    if evaluations:
        _require(
            all(row["status"] in ("trained", "complete") for row in methods),
            "readouts precede completion of all five original streams",
        )
    _require(not complete or len(evaluations) == 25, "complete fixed-A report requires25 readouts")
    order = [(method, point) for method in METHODS for point in POINTS]
    for index, (evaluation, (method, point)) in enumerate(zip(evaluations, order, strict=False)):
        _require(
            evaluation.get("method") == method
            and _integer(evaluation.get("updates"), 0, 128, "evaluation checkpoint") == point
            and evaluation.get("role") == "train",
            "evaluation identity/role/order mismatch",
        )
        _require(
            _status(evaluation.get("status"))
            and (not complete or evaluation["status"] == "complete"),
            "invalid evaluation status",
        )
        if index < len(evaluations) - 1:
            _require(evaluation["status"] == "complete", "work continues after failed readout")
        if evaluation["status"] == "complete":
            _require(
                evaluation.get("state_unchanged") is True, "completed readout state not preserved"
            )
        _scores(evaluation, targets)
        _counts(evaluation.get("work"))
        episodes = _records(evaluation.get("episodes", []), 2, "live episodes")
        _require(point in (0, 128) or not episodes, "live work at undeclared checkpoint")
        if evaluation["status"] == "complete" and point in (0, 128):
            _require(len(episodes) == 2, "complete endpoint requires both live cue episodes")
        for episode_index, (episode, cue) in enumerate(
            zip(episodes, ("blue", "red"), strict=False)
        ):
            _require(
                not evaluation["status"] == "complete" or episode.get("status") == "complete",
                "complete readout contains failed episode",
            )
            if episode_index < len(episodes) - 1:
                _require(
                    episode.get("status") == "complete", "live work continues after failed episode"
                )
            conditions[METHODS.index(method)]["episodes"].append(
                _episode(
                    episode, f"/evaluations/{index}/episodes/{episode_index}", method, point, cue
                )
            )
        _evaluation_work(evaluation, episodes)
    _acquisition(report, evaluations)
    return dict(
        schema_id="minigrid_inspector_v1",
        source_label=source_label,
        origin="recorded_artifact",
        source=deepcopy(report),
        conditions=conditions,
    )
