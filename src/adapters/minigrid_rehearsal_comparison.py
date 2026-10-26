"""Compare complete saved original MiniGrid F/old-A/return reports.

Inputs are plain pilot DTOs; output is descriptive paired curves and paid costs.
No IO, learner construction, scoring, selection, or final evaluation occurs here.
Why: unchanged benchmark probability anchors qualify historical control reuse;
extra TRAIN replay remains explicitly different exposure and computation.
"""

from copy import deepcopy
import math
from typing import Any

from src.adapters import minigrid_probe_rehearsal as rehearsal_option

_NEUTRAL_CONFIG = {
    "chemical_decay": 0.995,
    "chemical_buildup_rate": 0.02,
    "use_saturating_chemical": False,
    "chemical_max_value": 2.5,
    "chemical_saturation_gain": 1.0,
    "use_dual_chemical": False,
    "dual_fast_mix": 0.7,
    "slow_chemical_decay": 0.999,
    "slow_buildup_scale": 0.25,
    "plasticity_sensitivity": 0.7,
    "use_adaptive_plasticity_sensitivity": False,
    "plasticity_sensitivity_min": 0.35,
    "plasticity_sensitivity_max": 1.2,
    "plasticity_importance_mix": 0.5,
    "min_plasticity": 1.0,
    "use_reward_modulated_learning": False,
    "reward_baseline_decay": 0.95,
    "reward_difficulty_exponent": 1.0,
    "reward_scale_min": 0.75,
    "reward_scale_max": 1.5,
    "use_adaptive_thresholds": False,
    "adaptive_split_percentile": 85.0,
    "adaptive_prune_percentile": 20.0,
    "split_threshold": 0.8,
    "prune_threshold": 0.08,
    "split_hysteresis_margin": 0.0,
    "prune_hysteresis_margin": 0.0,
    "split_cooldown_steps": 0,
    "prune_cooldown_steps": 0,
    "split_weight_norm_mix": 0.3,
    "prune_weight_norm_mix": 0.3,
    "split_importance_mix": 0.2,
    "prune_importance_mix": 0.35,
    "importance_ema_decay": 0.95,
    "max_split_per_sleep": 0,
    "max_prune_per_sleep": 0,
    "split_noise_scale": 0.0,
    "sleep_reset_factor": 1.0,
    "sleep_warmup_steps": 0,
    "sleep_split_only_until_fraction": 0.5,
    "sleep_prune_only_after_fraction": 0.85,
    "sleep_max_change_fraction": 1.0,
    "sleep_min_change_count": 1,
    "prune_min_age_steps": 0,
    "homeostatic_downscale_factor": 1.0,
    "homeostasis_target_input_norm": 0.0,
    "homeostasis_target_output_norm": 0.0,
    "homeostasis_strength": 0.5,
    "use_adaptive_sleep_trigger": False,
    "min_sleep_steps": 40,
    "sleep_energy_window": 32,
    "sleep_plateau_delta": 0.0001,
    "sleep_chemical_variance_threshold": 0.02,
    "use_adaptive_sleep_budget": False,
    "adaptive_sleep_budget_min_scale": 0.25,
    "adaptive_sleep_budget_max_scale": 1.0,
    "adaptive_sleep_budget_plateau_weight": 0.6,
    "adaptive_sleep_budget_variance_weight": 0.4,
    "sleep_mode": "legacy",
    "sleep_enable_chemical_reset": True,
    "sleep_enable_homeostasis": True,
    "sleep_enable_split": True,
    "sleep_enable_prune": True,
}
_ACTIVE_CONFIG = {
    **_NEUTRAL_CONFIG,
    **{
        "min_plasticity": 0.2,
        "max_split_per_sleep": 2,
        "max_prune_per_sleep": 2,
        "split_noise_scale": 0.01,
        "sleep_reset_factor": 0.45,
    },
}

METRICS = ("ce", "accuracy", "exact_sequence_accuracy")
SOURCE_KEYS = (
    "episode_id",
    "role",
    "context",
    "seed",
    "transitions",
    "initial_observation",
    "layout",
    "cue",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _near(actual: Any, expected: float, message: str) -> None:
    _require(
        type(actual) in (int, float)
        and math.isfinite(actual)
        and math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10),
        message,
    )


def _complete(value: dict[str, Any]) -> None:
    _require(
        value.get("status") == "complete" and value.get("reason") is None,
        "complete successful evidence required",
    )


def _metrics(losses: list[float], correct: list[bool]) -> dict[str, float]:
    return dict(
        ce=sum(losses) / len(losses),
        accuracy=sum(correct) / len(correct),
        exact_sequence_accuracy=float(all(correct)),
    )


def _score(score: dict[str, Any], values: dict[str, float]) -> None:
    for key in METRICS:
        if key in score or key != "exact_sequence_accuracy":
            _near(score[key], values[key], f"inconsistent {key}")


def _decision(
    detail: dict[str, Any], score: dict[str, Any], sources: dict[str, Any]
) -> dict[str, float]:
    source = sources[detail["episode_id"]]
    labels = [row["action"] for row in source["transitions"]]
    _require(
        detail["num_classes"] == 7 and detail["rows"] == len(labels) == 9, "decision shape differs"
    )
    _require(
        detail["row_ids"] == [source["episode_id"] + f":step:{i}" for i in range(9)],
        "decision row identity differs",
    )
    losses, correct = [], []
    confusion = [[0] * 7 for _ in range(7)]
    counts, predicted_counts = [0] * 7, [0] * 7
    per_class: list[list[float]] = [[] for _ in range(7)]
    _require(
        len(detail["decisions"]) == 9 and len(detail["per_class"]) == 7,
        "decision records incomplete",
    )
    for index, (decision, target) in enumerate(zip(detail["decisions"], labels, strict=True)):
        predicted, loss = decision["predicted_class"], decision["loss"]
        _require(type(predicted) is int and 0 <= predicted < 7, "invalid predicted class")
        _require(
            decision["index"] == index
            and decision["target"] == decision["target_class"] == target
            and decision["correct"] is (predicted == target),
            "decision label or correctness differs",
        )
        _require(
            type(loss) in (int, float) and math.isfinite(loss) and 0 <= loss <= -math.log(1e-12),
            "invalid saved loss",
        )
        losses.append(loss)
        correct.append(predicted == target)
        counts[target] += 1
        predicted_counts[predicted] += 1
        confusion[target][predicted] += 1
        per_class[target].append(loss)
    _require(
        detail["confusion"] == confusion
        and detail["class_counts"] == counts
        and detail["predicted_class_counts"] == predicted_counts,
        "decision aggregate differs",
    )
    for index, row in enumerate(detail["per_class"]):
        support = counts[index]
        _require(
            row["class_index"] == index
            and row["support"] == support
            and row["predicted_count"] == predicted_counts[index]
            and row["correct"] == confusion[index][index],
            "per-class counts differ",
        )
        if support:
            _near(row["accuracy"], confusion[index][index] / support, "per-class accuracy differs")
            _near(row["mean_loss"], sum(per_class[index]) / support, "per-class loss differs")
        else:
            _require(
                row["accuracy"] is None and row["mean_loss"] is None, "unsupported class fabricated"
            )
    values = _metrics(losses, correct)
    _score(score, values)
    return values


def _curve(
    points: list[dict[str, Any]], sources: dict[str, Any], expected: dict[str, str]
) -> list[dict[str, Any]]:
    _require(
        [point["updates"] for point in points] == [0, 4, 8, 16], "probe checkpoint schedule differs"
    )
    result = []
    for point in points:
        _require(
            set(point["scores"]) == set(point["decision_diagnostics"]) == set(expected),
            "probe phase membership differs",
        )
        _require(
            all(
                point["decision_diagnostics"][key]["episode_id"] == episode_id
                for key, episode_id in expected.items()
            ),
            "probe phase source identity differs",
        )
        values = [
            _decision(point["decision_diagnostics"][name], score, sources)
            for name, score in point["scores"].items()
        ]
        result.append(
            dict(
                updates=point["updates"],
                **{key: sum(value[key] for value in values) / 2 for key in METRICS},
            )
        )
    return result


def _outcome(outcome: dict[str, Any], curve: list[dict[str, Any]]) -> None:
    _require(len(outcome["curve"]) == 4, "outcome curve incomplete")
    for actual, expected in zip(outcome["curve"], curve, strict=True):
        _require(actual["updates"] == expected["updates"], "outcome updates differ")
        for key in METRICS:
            _near(actual[key], expected[key], "outcome curve differs")
    for name, key in (
        ("ce_gain", "ce"),
        ("accuracy_gain", "accuracy"),
        ("exact_sequence_gain", "exact_sequence_accuracy"),
    ):
        gain = curve[0][key] - curve[-1][key] if key == "ce" else curve[-1][key] - curve[0][key]
        _near(outcome[name], gain, "outcome gain differs")


def _phase(phase: dict[str, Any]) -> None:
    _complete(phase)
    for key in ("attempted_updates", "completed_updates"):
        _require(type(phase[key]) is int and phase[key] == 16, "native phase dose differs")
    for key in ("attempted_training_examples", "completed_training_examples"):
        _require(phase[key] == 144, "native phase rows differ")
    _require(
        len(phase["diagnostics"]) == 16 and phase["timing_status"] == "complete",
        "phase work unavailable",
    )
    _require(
        phase["score_forward_calls_attempted"] == phase["score_forward_calls_completed"] == 8,
        "phase predictions differ",
    )
    monitor = phase["training_readouts"]
    _complete(monitor)
    _require(
        monitor["timing_status"] == "complete"
        and monitor["failure_work_may_be_unobserved"] is False,
        "monitor work unavailable",
    )
    _require(
        monitor["score_forward_calls_attempted"] == monitor["score_forward_calls_completed"] == 8
        and monitor["attempted_rows"] == monitor["completed_rows"] == 72,
        "monitor dose differs",
    )


def _history_episode(source: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(source)
    initial = source["transitions"][0]["observation"]
    for row in result["transitions"]:
        row["observation"] = initial + row["observation"]
        row["next_observation"] = initial + row["next_observation"]
    return result


def _replay(slot: dict[str, Any], sources: dict[str, Any]) -> None:
    expected_ids = ["minigrid-source-A-blue", "minigrid-source-A-red"]
    expected_episodes = [_history_episode(sources[name]) for name in expected_ids]
    work, replay = slot["rehearsal_work"], slot["report"]["rehearsal"]
    _require(
        work["budget_binding_preserved"] is True and work["work_unknown"] is False,
        "replay binding or work unavailable",
    )
    for key, value in dict(
        charged_replay_examples=288,
        attempted_updates=32,
        returned_updates=32,
        committed_updates=32,
        returned_training_rows=288,
        bank_encoded_bytes=87280,
        validation_projection_rows=36,
        validation_projected_array_equivalent_bytes=85248,
    ).items():
        _require(type(work[key]) is int and work[key] == value, f"replay {key} differs")
    _require(
        replay["bound"] is True
        and replay["stopped"] is False
        and replay["reason"] is None
        and replay["wake_role"] == "probe_train"
        and replay["control"] == "fixed_train_episodes_v1",
        "replay route differs",
    )
    _require(
        replay["total_updates"] == replay["wake_updates_observed"] == replay["planned_events"] == 16
        and replay["input_dim"] == 296,
        "replay schedule differs",
    )
    bank = replay["fixed_bank"]
    _require(
        bank
        == dict(
            episode_ids=expected_ids,
            episode_count=2,
            native_transition_count=18,
            encoded_array_bytes=87280,
            construction_updates=0,
        ),
        "fixed TRAIN bank differs",
    )
    for key, value in dict(
        charged_exposures=288,
        replay_budget_charged_exposures=288,
        attempted_updates=32,
        returned_updates=32,
        committed_updates=32,
    ).items():
        _require(replay[key] == value, "replay totals differ")
    _require(len(replay["events"]) == 16, "replay events incomplete")
    for index, event in enumerate(replay["events"], 1):
        _complete(event)
        _require(
            event["updates"] == index
            and event["event_id"] == f"rehearsal-{index}"
            and event["selected_episodes"] == expected_episodes,
            "replay event membership differs",
        )
        _require(
            event["charged_exposures"] == 18
            and event["attempted_updates"]
            == event["returned_updates"]
            == event["committed_updates"]
            == 2,
            "replay event cost differs",
        )
    stream = slot["report"]["probe_stream"]
    _require(
        stream["training_role"] == "probe_train"
        and stream["completed_updates"] == 16
        and stream["evaluation_scope"] == "training_only",
        "probe stream isolation differs",
    )


def _probe(slot: dict[str, Any], sources: dict[str, Any], replay: bool) -> dict[str, Any]:
    _complete(slot)
    for key in (
        "source_unchanged",
        "state_unchanged",
        "budget_binding_preserved",
        "return_budget_binding_preserved",
    ):
        _require(slot[key] is True, f"probe {key} unavailable")
    _require(
        slot["work_unknown"] is slot["return_work_unknown"] is False
        and slot["charged_updates"] == slot["return_charged_updates"] == 16,
        "probe work unavailable",
    )
    checkpoint = slot["checkpoint_updates"]
    _require(
        slot["native_updates_before"] == checkpoint
        and slot["native_training_examples_before"]
        == slot["expected_training_examples"]
        == checkpoint * 54
        and slot["catalogue_rows_per_update"] == 54,
        "probe source dose differs",
    )
    fresh, returned = slot["report"], slot["report"]["return_phase"]
    _phase(fresh)
    _phase(returned)
    _require(
        returned["schedule"] == [0, 1] * 8 and returned["checkpoint_updates"] == [0, 4, 8, 16],
        "A return schedule differs",
    )
    f_ids = {f"F-{cue}": f"minigrid-probe-source-probe_test-{cue}" for cue in ("blue", "red")}
    a_ids = {f"minigrid-source-A-{cue}": f"minigrid-source-A-{cue}" for cue in ("blue", "red")}
    scope = fresh["episode_scope"]
    _require(
        scope["training_episode_ids"]
        == [
            f"minigrid-probe-source-probe_train-{'blue' if index % 2 == 0 else 'red'}:copy:{index}"
            for index in range(16)
        ]
        and scope["evaluation_episode_ids"] == list(f_ids.values())
        and scope["training_transitions"] == 144
        and scope["evaluation_transitions"] == 18,
        "F episode schedule differs",
    )
    _require(
        all(
            sources[value]["role"] == "probe_test" and sources[value]["context"] == key
            for key, value in f_ids.items()
        )
        and all(
            sources[value]["role"] == "train" and sources[value]["context"] == "A"
            for value in a_ids
        ),
        "phase source role differs",
    )
    curves = [
        _curve(phase["checkpoints"], sources, expected)
        for phase, expected in (
            (fresh, f_ids),
            (fresh["training_readouts"], a_ids),
            (returned, f_ids),
            (returned["training_readouts"], a_ids),
        )
    ]
    for outcome, curve in zip(
        (
            slot["outcomes"],
            slot["old_task_readout_outcomes"],
            slot["return_fit_outcomes"]["F"],
            slot["return_fit_outcomes"]["A"],
        ),
        curves,
        strict=True,
    ):
        _outcome(outcome, curve)
    for before, after in (
        (fresh, returned),
        (fresh["training_readouts"], returned["training_readouts"]),
    ):
        for key in ("scores", "decision_diagnostics"):
            _require(
                before["checkpoints"][-1][key] == after["checkpoints"][0][key],
                "A return did not resume preceding candidate",
            )
    if replay:
        _replay(slot, sources)
    else:
        _require(
            "rehearsal_work" not in slot and "rehearsal" not in fresh, "control contains replay"
        )
    return dict(
        source_updates=checkpoint,
        F=curves[0],
        old_A_during_F=curves[1],
        F_during_return=curves[2],
        A_return=curves[3],
        F_ce_gain=slot["outcomes"]["ce_gain"],
        old_A_ce_loss=-slot["old_task_readout_outcomes"]["ce_gain"],
        A_return_ce_gain=slot["return_fit_outcomes"]["A"]["ce_gain"],
        old_A_acquired_before_F=curves[1][0]["exact_sequence_accuracy"] == 1,
    )


def _benchmark(row: dict[str, Any], sources: dict[str, Any]) -> None:
    _complete(row)
    _require(
        row["source_unchanged"] is row["actor_unchanged"] is True
        and row["charged_updates"]
        == row["wake_calls_attempted"]
        == row["wake_calls_returned"]
        == 128
        and row["native_training_examples"] == 6912,
        "benchmark work differs",
    )
    _require(
        row["final_evaluation"] == dict(updates=128, role="final_test", status="not_run"),
        "final evaluation must remain unavailable",
    )
    _require(
        row["checkpoint_updates"] == [0, 32, 64, 96, 128]
        and [point["updates"] for point in row["readouts"]] == row["checkpoint_updates"],
        "benchmark checkpoints differ",
    )
    _require(len(row["outcomes"]) == 5, "benchmark outcomes incomplete")
    expected_ids = [
        f"minigrid-source-{context}-{cue}"
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    ]
    for point, outcome in zip(row["readouts"], row["outcomes"], strict=True):
        _complete(point)
        _require(
            point["source_unchanged"] is point["state_unchanged"] is True
            and [score["episode_id"] for score in point["scores"]] == expected_ids,
            "benchmark readout cohort differs",
        )
        contexts: dict[str, list[dict[str, float]]] = {}
        for score in point["scores"]:
            source = sources[score["episode_id"]]
            labels = [transition["action"] for transition in source["transitions"]]
            _require(
                score["context"] == source["context"]
                and score["target_actions"] == labels
                and len(score["probabilities"]) == 9,
                "benchmark labels differ",
            )
            losses, correct, predictions = [], [], []
            for values, target in zip(score["probabilities"], labels, strict=True):
                _require(
                    len(values) == 7
                    and all(
                        type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                        for value in values
                    )
                    and abs(sum(values) - 1) <= 1e-6,
                    "invalid benchmark probabilities",
                )
                predicted = max(range(7), key=lambda index: values[index])
                predictions.append(predicted)
                losses.append(-math.log(max(1e-12, min(1, values[target]))))
                correct.append(predicted == target)
            _require(score["predicted_actions"] == predictions, "benchmark argmax differs")
            values = _metrics(losses, correct)
            _score(score, values)
            contexts.setdefault(source["context"], []).append(values)
        _require(outcome["updates"] == point["updates"], "benchmark outcome checkpoint differs")
        for context, values in contexts.items():
            for key in METRICS:
                _near(
                    outcome[context][key],
                    sum(value[key] for value in values) / 2,
                    "benchmark outcome differs",
                )
    _require(
        row["joint_catalogue_acquired"]
        is all(score["exact_sequence_accuracy"] == 1 for score in row["readouts"][-1]["scores"]),
        "acquisition premise differs",
    )


def _work(report: dict[str, Any], replay: bool) -> None:
    work = report["work"]
    for key, expected in dict(
        native_train_calls=256 if replay else 192,
        training_rows=8064 if replay else 7488,
        total_score_forwards_completed=94,
        total_score_rows_completed=846,
        teacher_episodes_observed=10,
        source_environment_steps_attempted=90,
        source_environment_steps_returned=90,
        final_episodes=0,
        learner_live_episodes=0,
        replay_rows=576 if replay else 0,
    ).items():
        _require(type(work[key]) is int and work[key] == expected, f"observed {key} differs")
    _require(
        work["failure_work_may_be_unobserved"]
        is work["source_environment_counts_are_lower_bounds"]
        is False,
        "unknown work cannot qualify comparison",
    )
    for name in ("training_readouts", "return_phase"):
        _require(
            work[name]["failure_work_may_be_unobserved"]
            is work[name]["storage_may_be_unobserved"]
            is False,
            "delegated work unavailable",
        )
    if replay:
        observed_replay = rehearsal_option.aggregate(report["methods"])
        _require(
            work["old_a_rehearsal"] == observed_replay
            and observed_replay["partial_work_unknown"] is False
            and observed_replay["budget_binding_preserved"] is True,
            "aggregate replay work differs",
        )


def compare_reports(
    control: dict[str, Any], rehearsal: dict[str, Any], *, method: str
) -> dict[str, Any]:
    """Validate the frozen pair and return descriptive outcomes; malformed DTOs refuse."""
    try:
        return _compare(control, rehearsal, method)
    except (KeyError, TypeError, IndexError, OverflowError, AttributeError) as error:
        raise ValueError(f"malformed MiniGrid saved comparison evidence: {error}") from error


def _compare(control: dict[str, Any], rehearsal: dict[str, Any], method: str) -> dict[str, Any]:
    _require(
        type(control) is dict and type(rehearsal) is dict and method in rehearsal_option.METHODS,
        "original method and plain saved reports required",
    )
    native = f"public_invariant_{method}"
    protocol = control["protocol"]
    for key, expected in dict(
        feature_dim=296,
        hidden_dim=8,
        num_classes=7,
        seed=17,
        learning_rate=0.03,
        momentum=0.9,
        inference_steps=2,
        inference_learning_rate=0.04,
        methods=[native],
        checkpoint_updates=[0, 32, 64, 96, 128],
        stream=dict(training_only=True, heldout=[], label_delay=1),
    ).items():
        _require(protocol[key] == expected, f"frozen recipe {key} differs")
    _require(
        protocol["invariant_history"]["encoding_id"]
        == "minigrid_public_wall_direction_invariant_history_296_v1"
        and "cue_relative_history" not in protocol
        and "old_a_rehearsal" not in protocol
        and "train_fit_bounds" not in protocol,
        "unqualified original protocol",
    )
    expected_configs = (
        {}
        if method in ("bp", "pc")
        else {method: _NEUTRAL_CONFIG if method == "neutral_cpc" else _ACTIVE_CONFIG}
    )
    _require(protocol["cpc_configs"] == expected_configs, "frozen native CPC configuration differs")
    expected_protocol = deepcopy(protocol)
    rehearsal_option.describe(expected_protocol)
    _require(
        rehearsal["protocol"] == expected_protocol,
        "protocol differs beyond declared replay intervention",
    )
    for key in ("training_sources", "catalogue_inputs"):
        _require(control[key] == rehearsal[key], f"original {key} differs")
    _require(
        control["probe_data"]["packets"] == rehearsal["probe_data"]["packets"], "F packets differ"
    )
    source_ids = [
        f"minigrid-source-{context}-{cue}"
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    ]
    _require(
        [source["episode_id"] for source in control["training_sources"]] == source_ids
        and all(
            source["role"] == "train" and len(source["transitions"]) == 9
            for source in control["training_sources"]
        ),
        "TRAIN membership differs",
    )
    for name, count in (("source_reports", 6), ("probe_source_reports", 4)):
        _require(len(control[name]) == len(rehearsal[name]) == count, "source report count differs")
        for left, right in zip(control[name], rehearsal[name], strict=True):
            _require(
                all(left[key] == right[key] for key in SOURCE_KEYS), "source provenance differs"
            )
            for source in (left, right):
                _complete(source)
                _require(
                    source["closed"] is True
                    and source["failure_work_may_be_unobserved"] is False
                    and source["reset_calls"] == 1
                    and source["environment_steps_attempted"] == source["environment_steps"] == 9,
                    "source collection incomplete",
                )
    sources = {
        source["episode_id"]: source
        for source in (*control["training_sources"], *control["probe_source_reports"])
    }
    _require(len(sources) == 10, "source identity collision")
    summaries = []
    for report, replay in ((control, False), (rehearsal, True)):
        _complete(report)
        _require(
            report["scientific_advantage_verified"] is False
            and report["comparisons"] is None
            and len(report["methods"]) == 1,
            "unsupported scientific claim or extra methods",
        )
        row = report["methods"][0]
        _require(
            row["method"] == native
            and [slot["checkpoint_updates"] for slot in row["probes"]] == [0, 128],
            "method or probe schedule differs",
        )
        _benchmark(row, sources)
        _work(report, replay)
        summaries.append(
            dict(
                joint_catalogue_acquired=row["joint_catalogue_acquired"],
                train_curve=deepcopy(row["outcomes"]),
                probes=[_probe(slot, sources, replay) for slot in row["probes"]],
                work=deepcopy(report["work"]),
                total_seconds=report["total_seconds"],
                observed_process_rss=deepcopy(report["observed_process_rss"]),
            )
        )
    maximum = 0.0
    for left, right in zip(
        control["methods"][0]["readouts"], rehearsal["methods"][0]["readouts"], strict=True
    ):
        for a, b in zip(left["scores"], right["scores"], strict=True):
            _require(
                a["target_actions"] == b["target_actions"]
                and a["predicted_actions"] == b["predicted_actions"],
                "benchmark anchors changed decisions",
            )
            for av, bv in zip(a["probabilities"], b["probabilities"], strict=True):
                for x, y in zip(av, bv, strict=True):
                    maximum = max(maximum, abs(x - y))
    _require(maximum <= 1e-7, "benchmark probability anchors changed")
    deltas = [
        dict(
            source_updates=a["source_updates"],
            **{key: b[key] - a[key] for key in ("F_ce_gain", "old_A_ce_loss", "A_return_ce_gain")},
        )
        for a, b in zip(summaries[0]["probes"], summaries[1]["probes"], strict=True)
    ]
    return dict(
        status="complete",
        method=method,
        control=summaries[0],
        rehearsal=summaries[1],
        replay_minus_control=deltas,
        probability_anchor=dict(rows=270, tolerance=1e-7, maximum_difference=maximum),
        source_inputs_equal=True,
        extra_native_calls=64,
        extra_training_rows=576,
        dose_and_compute_matched=False,
        scientific_advantage_verified=False,
        limitations="Descriptive fixed seed/recipe comparison. Replay adds old-A exposure and compute. Positive F/A-return gain differences and negative old-A loss differences are desirable. Acquisition must precede retention/recovery claims. Probe class/loss arithmetic is checked, but absent saved probe probability vectors prevent independent native argmax/log-probability reconstruction. Invocation wall/RSS are separate observations, not performance comparisons.",
    )
