"""Compare a fixed repeated-instance reward/nonlearning family from saved data.

Inputs are the four prospectively ordered native report records. Outputs retain
both physical-stage curves, signed differences, paid costs and incomplete cells.
No learners, adapters, physics, IO, tuning or statistical inference belongs here.
Catalogue instances are finite related configurations, not independent draws.
"""

from copy import deepcopy
from math import isfinite
from typing import Any

from src.core.key_door_instances import seeded_action_remapping_layouts
from src.core.key_door_navigation import key_door_layout_record

ENVIRONMENT_PAIRS = ((9, 14), (18, 5))
MODES = ("reward", "frozen")
_PHYSICAL = (0, 24, 48, 72, 96)
_CONTEXTS = ("A", "B", "C", "D")
_TASKS = ("A", "B", "C", "A")
_OUTCOMES = ("new_B_gain", "new_C_gain", "retention_loss", "return_gain")
_LIVE_METRICS = ("success_rate", "mean_return", "mean_ce", "mean_accuracy")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, name: str, low: int = 0, high: int = 10000) -> int:
    _require(type(value) is int and low <= value <= high, f"invalid integer {name}")
    return value


def _number(value: Any, name: str, low: float | None = None, high: float | None = None) -> float:
    _require(type(value) in (int, float), f"invalid number {name}")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"nonfinite {name}") from error
    _require(
        isfinite(number) and (low is None or number >= low) and (high is None or number <= high),
        f"invalid range {name}",
    )
    return number


def _normal(value):
    if type(value) in (tuple, list):
        return [_normal(item) for item in value]
    if type(value) is dict:
        return {key: _normal(item) for key, item in value.items()}
    return value


def _same(left, right) -> bool:
    left, right = _normal(left), _normal(right)
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_same(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return bool(left == right)


def _finite_tree(value) -> None:
    stack, count = [(value, 0)], 0
    while stack:
        item, depth = stack.pop()
        count += 1
        _require(count <= 1000000 and depth <= 64, "comparison report exceeds structural bounds")
        if type(item) is dict:
            _require(all(type(key) is str for key in item), "report keys must be text")
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) in (list, tuple):
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is float:
            _require(isfinite(item), "report contains nonfinite evidence")
        else:
            _require(
                type(item) in (str, int, bool, type(None)), "report contains unsupported evidence"
            )


def _records(value, size: int, name: str):
    _require(
        type(value) in (tuple, list)
        and len(value) == size
        and all(type(item) is dict for item in value),
        f"invalid {name}",
    )
    return value


def _protocol(report, seed, heldout, mode):
    protocol = report.get("protocol")
    _require(type(protocol) is dict, "complete case lacks protocol")
    expected = dict(
        protocol_id="actual_key_door_frozen_interaction_quota_v1"
        if mode == "frozen"
        else "actual_key_door_seeded_interaction_quota_v1",
        seed=17,
        sampling_seed=29,
        feature_dim=8,
        hidden_dim=8,
        num_classes=5,
        environment_seeds=dict(training=seed, final_only=heldout),
        methods=["recurrent_rl"],
        replay_modes=[False],
        task_sequence=list(_TASKS),
        cues=[1, -1],
        checkpoint_physical_steps=list(_PHYSICAL),
        native_defaults=dict(learning_rate=0.03, momentum=0.9, max_episode_steps=16, discount=1.0),
        action_conditioning="legal_actions_v1",
        action_mapping="movement_permutation_v1",
        interaction_quota="twelve_steps_per_phase_v1",
        heldout_layouts=["D"],
        layouts=[
            key_door_layout_record(layout)
            for layout in seeded_action_remapping_layouts(seed, heldout)
        ],
    )
    _require(
        all(key in protocol and _same(protocol[key], value) for key, value in expected.items()),
        "unmatched protocol, initialization, environment roles or layouts",
    )
    _require(
        protocol.get("learning_mode")
        == ("frozen_recurrent_control_v1" if mode == "frozen" else None),
        "invalid protocol learning mode",
    )
    return protocol


def _quota(value, attempted):
    expected = dict(
        maximum_steps=12, attempted_steps=attempted, completed_steps=attempted, failed_steps=0
    )
    _require(_same(value, expected), "incomplete or invalid physical quota")


def _training(row, mode, layouts):
    training = row.get("training")
    _require(
        type(training) in (list, tuple) and 8 <= len(training) <= 16, "invalid collection count"
    )
    phase_steps = [0] * 8
    phase_counts = [0] * 8
    previous_phase = 0
    cumulative = 0
    for index, slot in enumerate(training):
        _require(
            type(slot) is dict and slot.get("status") == "complete", "incomplete collection slot"
        )
        _require(
            _integer(slot.get("index"), "collection index") == index, "invalid collection order"
        )
        phase = _integer(slot.get("quota_phase"), "quota phase", 0, 7)
        _require(previous_phase <= phase <= previous_phase + 1, "invalid quota phase order")
        previous_phase = phase
        _require(
            _integer(slot.get("episode_in_quota"), "episode in quota", 0, 1) == phase_counts[phase],
            "invalid phase episode count",
        )
        context, cue = _TASKS[phase // 2], 1 if phase % 2 == 0 else -1
        _require(
            slot.get("context") == context
            and type(slot.get("cue")) is int
            and slot["cue"] == cue
            and _same(slot.get("layout"), layouts[_CONTEXTS.index(context)]),
            "unmatched collection context or layout",
        )
        collection = slot.get("collection")
        _require(
            type(collection) is dict
            and collection.get("status") == "complete"
            and collection.get("sampler_binding_preserved") is True
            and collection.get("action_conditioning") == "legal_actions_v1",
            "invalid collection evidence",
        )
        count = _integer(collection.get("environment_steps"), "collection steps", 1, 12)
        for key in (
            "environment_steps_attempted",
            "prediction_calls_attempted",
            "prediction_calls_returned",
            "sampling_draws",
        ):
            _require(_integer(collection.get(key), key) == count, "unmatched paid collection work")
        transitions = _records(collection.get("transitions"), count, "collection transitions")
        _require(
            all(t.get("applied") is True for t in transitions), "unapplied collection transition"
        )
        _quota(collection.get("interaction_quota_before"), phase_steps[phase])
        phase_steps[phase] += count
        _quota(collection.get("interaction_quota_after"), phase_steps[phase])
        packet = collection.get("episode")
        _require(
            type(packet) is dict
            and type(packet.get("episode")) is dict
            and packet["episode"].get("role") == "train",
            "collection must retain its TRAIN role",
        )
        native_before = 0 if mode == "frozen" else index
        rows_before = 0 if mode == "frozen" else cumulative
        for key, expected in (
            ("native_updates_before", native_before),
            ("native_training_examples_before", rows_before),
            ("charged_updates_before", native_before),
            ("charged_updates_after", 0 if mode == "frozen" else index + 1),
        ):
            _require(
                _integer(slot.get(key), key) == expected,
                "unmatched original native budget progression",
            )
        _require(
            _integer(collection.get("behavior_native_updates"), "behavior updates") == native_before
            and _integer(collection.get("behavior_training_examples"), "behavior rows")
            == rows_before,
            "collection behavior native counters changed",
        )
        for key in ("update_attempted", "update_committed", "commit_validated"):
            _require(slot.get(key) is (mode == "reward"), "invalid native commit flags")
        if mode == "reward":
            receipt = slot.get("receipt")
            _require(
                type(receipt) is dict
                and _integer(receipt.get("update_number"), "receipt update") == index + 1,
                "missing validated reward receipt",
            )
        else:
            _require(
                slot.get("receipt") is None
                and slot.get("native_stages") == []
                and slot.get("collection_only") is True
                and slot.get("collection_validated") is True,
                "frozen control admits no native updates or receipts",
            )
        cumulative += count
        phase_counts[phase] += 1
    _require(
        phase_steps == [12] * 8 and all(1 <= count <= 2 for count in phase_counts),
        "collection lacks exact96 physical steps",
    )
    return training, [0] + [
        sum(phase_counts[:end]) if mode == "reward" else 0 for end in (2, 4, 6, 8)
    ]


def _scores(row, native, mode):
    boundaries = _records(row.get("quota_checkpoints"), 5, "quota boundaries")
    points = _records(row.get("stream_result", {}).get("checkpoints"), 5, "score checkpoints")
    result = []
    for physical, updates, boundary, point in zip(_PHYSICAL, native, boundaries, points):
        _require(
            _same(boundary, dict(physical_steps=physical, native_updates=updates)),
            "unmatched physical/native boundary",
        )
        _require(
            _integer(point.get("updates"), "score native index") == updates,
            "score native index changed",
        )
        _require(
            "status" not in point or point["status"] == "complete", "incomplete score checkpoint"
        )
        if mode == "frozen" or "physical_steps" in point:
            _require(
                _integer(point.get("physical_steps"), "score physical stage") == physical,
                "frozen physical stages cannot collapse native0",
            )
        scores = point.get("scores")
        _require(
            type(scores) is dict
            and set(scores) == {f"{c}-{color}" for c in _CONTEXTS for color in ("red", "blue")},
            "incomplete final context scores",
        )
        for metrics in scores.values():
            _require(
                type(metrics) is dict and set(metrics) == {"ce", "accuracy"},
                "invalid categorical metrics",
            )
            _number(metrics["ce"], "CE", 0)
            _number(metrics["accuracy"], "accuracy", 0, 1)
        result.append(scores)
    return result


def _signed(scores):
    def mean(stage, context):
        return sum(scores[stage][f"{context}-{color}"]["ce"] for color in ("red", "blue")) / 2

    return dict(
        new_B_gain=mean(0, "B") - mean(2, "B"),
        new_C_gain=mean(0, "C") - mean(3, "C"),
        retention_loss=mean(3, "A") - mean(1, "A"),
        return_gain=mean(3, "A") - mean(4, "A"),
    )


def _live(row, native, scores, layouts):
    episodes = _records(row.get("episodes"), 40, "live episodes")
    stages = []
    for stage, physical in enumerate(_PHYSICAL):
        contexts = []
        for context_index, context in enumerate(_CONTEXTS):
            slots = episodes[8 * stage + 2 * context_index : 8 * stage + 2 * context_index + 2]
            outcomes: dict[str, int] = {}
            for cue, slot in zip((1, -1), slots):
                _require(
                    slot.get("status") == "complete"
                    and slot.get("state_unchanged") is True
                    and slot.get("execution_origin") == "live_environment"
                    and slot.get("phase") == "evaluation"
                    and slot.get("learning_active") is False
                    and slot.get("context") == context
                    and type(slot.get("cue")) is int
                    and slot["cue"] == cue
                    and _same(slot.get("layout"), layouts[context_index])
                    and slot.get("layout_partition")
                    == ("heldout_layout" if context == "D" else "known_layout"),
                    "unmatched live evaluation identity",
                )
                _require(
                    _integer(slot.get("checkpoint_updates"), "live native index") == native[stage],
                    "unmatched live native stage",
                )
                if "checkpoint_physical_steps" in slot:
                    _require(
                        _integer(slot["checkpoint_physical_steps"], "live physical stage")
                        == physical,
                        "unmatched live physical stage",
                    )
                elif native == [0] * 5:
                    raise ValueError("frozen live episodes require explicit physical stages")
                _require(
                    type(slot.get("success")) is bool and type(slot.get("terminal")) is bool,
                    "invalid live outcome flags",
                )
                outcome = slot.get("outcome")
                _require(
                    type(outcome) is str
                    and outcome in ("success", "wrong_key", "illegal_action", "step_limit")
                    and slot["success"] is (outcome == "success")
                    and slot["terminal"] is (outcome in ("success", "wrong_key")),
                    "invalid live outcome",
                )
                returned = _integer(
                    slot.get("prediction_calls_returned"), "live predictions", 1, 12
                )
                _require(
                    _integer(slot.get("prediction_calls_attempted"), "live prediction attempts")
                    == returned,
                    "incomplete live predictions",
                )
                steps = _integer(slot.get("steps_executed"), "live physical steps", 0, 12)
                transitions = _records(slot.get("transitions"), returned, "live transitions")
                _require(
                    all(type(t.get("applied")) is bool for t in transitions)
                    and sum(t["applied"] for t in transitions) == steps,
                    "unmatched live physical count",
                )
                total_return = _number(slot.get("return"), "live return", -1, 1)
                _require(
                    sum(_number(t.get("reward"), "transition reward", -1, 1) for t in transitions)
                    == total_return,
                    "unmatched observed live return",
                )
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
            metrics = [scores[stage][f"{context}-{color}"] for color in ("red", "blue")]
            contexts.append(
                dict(
                    context=context,
                    episode_count=2,
                    success_count=sum(s["success"] for s in slots),
                    success_rate=sum(s["success"] for s in slots) / 2,
                    mean_return=sum(s["return"] for s in slots) / 2,
                    mean_ce=sum(m["ce"] for m in metrics) / 2,
                    mean_accuracy=sum(m["accuracy"] for m in metrics) / 2,
                    outcome_counts=outcomes,
                    live=[
                        {
                            key: deepcopy(s[key])
                            for key in (
                                "cue",
                                "success",
                                "outcome",
                                "return",
                                "steps_executed",
                                "prediction_calls_attempted",
                                "prediction_calls_returned",
                            )
                        }
                        for s in slots
                    ],
                )
            )
        stages.append(
            dict(physical_steps=physical, native_updates=native[stage], contexts=contexts)
        )
    return stages


def _observations(row, native, mode):
    observations = _records(row.get("observations"), 5, "native observations")
    for physical, updates, observation in zip(_PHYSICAL, native, observations):
        _require(
            _integer(observation.get("native_updates"), "observed native updates") == updates
            and _integer(observation.get("native_training_examples"), "observed native rows")
            == (0 if mode == "frozen" else physical),
            "unmatched observed native progression",
        )
        _require(
            type(observation.get("parameters")) is dict and bool(observation["parameters"]),
            "missing initialized parameter observation",
        )
        if mode == "frozen":
            _require(
                _integer(observation.get("physical_steps"), "observed physical stage") == physical
                and _same(observation["parameters"], observations[0]["parameters"]),
                "frozen observed parameters or physical stage changed",
            )
    return observations[0]["parameters"]


def _validate_complete(case):
    report, mode = case["report"], case["mode"]
    _require(
        case.get("reason") is None
        and type(report) is dict
        and report.get("status") == "complete"
        and report.get("global_errors") == [],
        "complete case lacks complete native report",
    )
    protocol = _protocol(report, case["environment_seed"], case["heldout_environment_seed"], mode)
    row = _records(report.get("runs"), 1, "native runs")[0]
    _require(
        row.get("status") == "complete"
        and row.get("source_unchanged") is True
        and row.get("actor_unchanged") is True
        and row.get("method") == "recurrent_rl"
        and row.get("replay_enabled") is False
        and row.get("readback_error") is None
        and _same(row.get("environment_seeds"), protocol["environment_seeds"]),
        "invalid native isolation or environment roles",
    )
    _require(
        row.get("learning_mode") == ("frozen_recurrent_control_v1" if mode == "frozen" else None),
        "invalid native learning mode",
    )
    _require(
        row.get("action_conditioning") == "legal_actions_v1"
        and row.get("action_mapping") == "movement_permutation_v1"
        and row.get("interaction_quota_mode") == "twelve_steps_per_phase_v1",
        "unmatched native policy law",
    )
    for quota in _records(row.get("physical_quotas"), 8, "physical quotas"):
        _quota(quota, 12)
    training, native = _training(row, mode, protocol["layouts"])
    updates = len(training) if mode == "reward" else 0
    _require(
        _integer(row.get("wake_updates"), "wake updates") == updates
        and _integer(row.get("live_sessions_started"), "live sessions") == 40,
        "unmatched native work",
    )
    stream = row.get("stream_result")
    _require(
        type(stream) is dict
        and _integer(stream.get("completed_updates"), "native updates") == updates
        and _integer(stream.get("training_examples"), "native training rows")
        == (96 if mode == "reward" else 0)
        and _integer(stream.get("score_forward_calls"), "score calls") == 40,
        "unmatched native training or final score cost",
    )
    _require(
        _same(
            stream.get("applied_updates"),
            [s["receipt"] for s in training] if mode == "reward" else [],
        ),
        "unmatched native receipts",
    )
    scores = _scores(row, native, mode)
    initial_parameters = _observations(row, native, mode)
    outcomes = _signed(scores)
    _require(
        type(row.get("outcomes")) is dict and set(row["outcomes"]) == set(_OUTCOMES),
        "missing signed CE outcomes",
    )
    for key, value in outcomes.items():
        _require(
            _number(row["outcomes"][key], key) == value,
            "reported signed CE outcome contradicts physical scores",
        )
    stages = _live(row, native, scores, protocol["layouts"])
    if report.get("total_seconds") is not None:
        _number(report["total_seconds"], "reported time", 0)
    return dict(
        protocol=protocol,
        initial_parameters=deepcopy(initial_parameters),
        outcomes=outcomes,
        stages=stages,
        costs=dict(
            native_updates=updates,
            native_training_rows=96 if mode == "reward" else 0,
            collection_physical_steps=96,
            collection_episodes=len(training),
            score_forward_calls=40,
            live_sessions_started=40,
            live_prediction_calls=sum(s["prediction_calls_returned"] for s in row["episodes"]),
            live_physical_steps=sum(s["steps_executed"] for s in row["episodes"]),
            reported_work=deepcopy(row.get("work")),
            total_seconds=report.get("total_seconds"),
            observed_process_rss=deepcopy(report.get("observed_process_rss")),
        ),
    )


def _partial_costs(case):
    report = case.get("report")
    if type(report) is not dict:
        return None
    rows: Any = report.get("runs")
    row = (
        rows[0] if type(rows) in (list, tuple) and len(rows) == 1 and type(rows[0]) is dict else {}
    )
    return dict(
        reported_work=deepcopy(row.get("work")),
        wake_updates=deepcopy(row.get("wake_updates")),
        total_seconds=deepcopy(report.get("total_seconds")),
        observed_process_rss=deepcopy(report.get("observed_process_rss")),
    )


def _pair(seed, heldout, cases, validated):
    result = dict(
        environment_seed=seed,
        heldout_environment_seed=heldout,
        case_ids=[case["case_id"] for case in cases],
        statuses={c["mode"]: c["status"] for c in cases},
        reasons={c["mode"]: c["reason"] for c in cases},
        status="incomplete",
        outcomes=None,
        stages=None,
    )
    if not all(case["status"] == "complete" for case in cases):
        if any(case["status"] == "failed" for case in cases):
            result["status"] = "failed"
        return result
    reward, frozen = (validated[case["case_id"]] for case in cases)
    for key in ("seed", "sampling_seed", "native_defaults", "layouts", "environment_seeds"):
        _require(
            _same(reward["protocol"][key], frozen["protocol"][key]), "unmatched paired protocol"
        )
    _require(
        _same(reward["initial_parameters"], frozen["initial_parameters"]),
        "unmatched paired initial parameters",
    )
    for first, second in zip(reward["stages"][0]["contexts"], frozen["stages"][0]["contexts"]):
        _require(
            all(_same(first[key], second[key]) for key in ("mean_ce", "mean_accuracy", "live")),
            "unmatched paired initial evaluation",
        )
    result["status"] = "complete"
    result["outcomes"] = dict(
        reward=deepcopy(reward["outcomes"]),
        frozen=deepcopy(frozen["outcomes"]),
        reward_minus_frozen={
            key: reward["outcomes"][key] - frozen["outcomes"][key] for key in _OUTCOMES
        },
    )
    result["stages"] = [
        dict(
            physical_steps=physical,
            contexts=[
                dict(
                    context=context,
                    reward=deepcopy(reward["stages"][stage]["contexts"][index]),
                    frozen=deepcopy(frozen["stages"][stage]["contexts"][index]),
                    reward_minus_frozen={
                        key: reward["stages"][stage]["contexts"][index][key]
                        - frozen["stages"][stage]["contexts"][index][key]
                        for key in _LIVE_METRICS
                    },
                )
                for index, context in enumerate(_CONTEXTS)
            ],
        )
        for stage, physical in enumerate(_PHYSICAL)
    ]
    return result


def compare_key_door_reward_cases(cases) -> dict[str, Any]:
    """Validate all fixed cells, then aggregate only two complete declared pairs."""
    _finite_tree(cases)
    cases = _records(cases, 4, "fixed comparison cells")
    validated = {}
    costs = []
    for case, (seed, heldout, mode) in zip(
        cases, ((s, h, m) for s, h in ENVIRONMENT_PAIRS for m in MODES)
    ):
        _require(
            case.get("case_id") == f"env{seed}-final{heldout}-{mode}"
            and type(case.get("environment_seed")) is int
            and case["environment_seed"] == seed
            and type(case.get("heldout_environment_seed")) is int
            and case["heldout_environment_seed"] == heldout
            and case.get("mode") == mode,
            "comparison cells must retain the fixed ordered identities",
        )
        _require(
            type(case.get("status")) is str
            and case["status"] in ("not_run", "running", "complete", "failed")
            and "reason" in case
            and (case["reason"] is None or type(case["reason"]) is str)
            and "report" in case
            and (case["report"] is None or type(case["report"]) is dict),
            "invalid comparison status or report",
        )
        if case["status"] == "complete":
            validated[case["case_id"]] = _validate_complete(case)
        costs.append(
            dict(
                case_id=case["case_id"],
                status=case["status"],
                costs=deepcopy(
                    validated[case["case_id"]]["costs"]
                    if case["case_id"] in validated
                    else _partial_costs(case)
                ),
            )
        )
    pairs = [
        _pair(seed, heldout, cases[2 * i : 2 * i + 2], validated)
        for i, (seed, heldout) in enumerate(ENVIRONMENT_PAIRS)
    ]
    complete = all(pair["status"] == "complete" for pair in pairs)
    aggregate = None
    if complete:
        aggregate = dict(
            pair_count=2,
            outcomes={
                key: sum(pair["outcomes"]["reward_minus_frozen"][key] for pair in pairs) / 2
                for key in _OUTCOMES
            },
            stages=[
                dict(
                    physical_steps=physical,
                    contexts=[
                        dict(
                            context=context,
                            reward_minus_frozen={
                                key: sum(
                                    pair["stages"][stage]["contexts"][index]["reward_minus_frozen"][
                                        key
                                    ]
                                    for pair in pairs
                                )
                                / 2
                                for key in _LIVE_METRICS
                            },
                        )
                        for index, context in enumerate(_CONTEXTS)
                    ],
                )
                for stage, physical in enumerate(_PHYSICAL)
            ],
        )
    return dict(
        status="complete" if complete else "incomplete",
        pairs=pairs,
        aggregate=aggregate,
        case_costs=costs,
        directions=dict(
            difference="reward_minus_frozen",
            gains="positive favors reward",
            retention_loss="negative favors reward",
            success_and_return="higher is better",
            ce="lower is better",
            accuracy="higher is better",
            C_scope="identity-law return, not a novel third task",
        ),
        limitations="two prespecified finite related catalogue pairs; descriptive only; frozen is NONLEARNING, not a trained matched baseline; unequal SGD/computation; no significance, generalization or circadian advantage",
    )
