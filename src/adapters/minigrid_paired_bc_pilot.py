"""Fixed row-exposure paired recurrent acquisition through the existing stream.

Reuse accepted original TRAIN sources/reference and native metrics. Pairing is
explicitly row matched, not optimizer-step or momentum-schedule matched. No data
collection, final/probe access, tuning, promotion or new runner lives here.
"""

from copy import deepcopy
from dataclasses import asdict
from math import fsum, isfinite
import json
from pathlib import Path
import pickle
from time import monotonic
from typing import Any

import numpy as np

from src.adapters import minigrid_learning_pilot as shared
from src.adapters.key_door_reward_pilot import _binding, _check_budget
from src.adapters.minigrid_learnability import _score_checkpoint
from src.adapters.minigrid_policy_evaluation import MiniGridEvaluationFailed, evaluate_checkpoint
from src.adapters.minigrid_weighted_bc_pilot import (
    _admit_reference,
    _check_initial_condition,
    _load_reference,
)
from src.adapters.torch_recurrent_episode_batch import TORCH_RECURRENT_EPISODE_PAIR_CE_ID
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.adapters.torch_train_episode_batch import TorchTrainEpisodeBatchLearner
from src.app.coupled_attribution import AttributionBatch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.minigrid_key_door import predefined_layouts
from src.shared.process_memory import ProcessRssSampler

METHOD = "recurrent_paired_bc"
POINTS = (0, 16, 32, 48, 64)
ENDPOINT_FIRST_POINTS = (0, 64, 16, 32, 48)
BODY_SECONDS, TRAIN_SECONDS, READOUT_SECONDS = 75.0, 45.0, 20.0
RSS_LIMIT = 768 * 1024**2
PAIRED_REFERENCE_PATH = "artifacts/runs/r21-paired-endpoint-first-20261017/native-result.json"


def describe_protocol(
    *, endpoint_first: bool = False, public_history: bool = False
) -> dict[str, Any]:
    """Preview fixed inputs, comparisons and costs without native work."""
    if type(endpoint_first) is not bool:
        raise ValueError("endpoint_first must be an exact boolean")
    if type(public_history) is not bool or (public_history and endpoint_first):
        raise ValueError("public_history must be an exact boolean and excludes endpoint_first")
    result: dict[str, Any] = dict(
        protocol_id="minigrid_fixed_a64_episode_pair_acquisition_v1",
        classification="exploratory_single_seed_train_only_row_matched_acquisition",
        method=METHOD,
        objective=TORCH_RECURRENT_EPISODE_PAIR_CE_ID,
        feature_dim=148,
        hidden_dim=8,
        num_classes=7,
        seed=17,
        learning_rate=0.03,
        momentum=0.9,
        max_episode_steps=16,
        checkpoints=list(POINTS),
        reference_checkpoints=[2 * p for p in POINTS],
        reference_path="artifacts/runs/r21-fixed-a-acquisition-20261016/native-result.json",
        training="original128 numbered A TRAIN episodes paired by adjacent position;64SGD/128exposures/1152rows through delayed AttributionStream; independent hidden reset per episode",
        evaluation="after all training:6originalTRAIN raw packets and2Araw+2target-free live episodes per checkpoint; initial saved native law must match before later readouts",
        acquisition="endpoint both exact teacher sequences AND both native goal successes",
        comparison="paired-minus-original CE gain/success-count change/mean-return gain; positive gain better; rows matched but optimizer steps and momentum per-row schedule NOT matched",
        limitations="no final/probes/B-C training/replay/tuning/selection; no CPC/retention/recovery/generalization claim",
        planned_work=dict(
            native_training_updates=64,
            episode_exposures=128,
            training_rows=1152,
            raw_scorer_calls=30,
            raw_scorer_rows=270,
            live_evaluator_raw_calls=10,
            live_evaluator_raw_rows=90,
            live_episodes=10,
            max_live_predictions=160,
            teacher_episodes=0,
            final_episodes=0,
            probe_updates=0,
            gradient_pairs=0,
            activation_diagnostics=0,
        ),
        limits=dict(
            body_seconds=BODY_SECONDS,
            parent_seconds=90,
            training_seconds=TRAIN_SECONDS,
            readout_seconds=READOUT_SECONDS,
            process_rss_bytes=RSS_LIMIT,
            output_bytes=16 * 1024**2,
            cpu_threads=1,
            interop_threads=1,
            gpu=0,
        ),
    )
    if endpoint_first:
        result.update(
            protocol_id="minigrid_fixed_a64_episode_pair_endpoint_first_v1",
            readout_order=list(ENDPOINT_FIRST_POINTS),
            endpoint_evidence="separate validated fixed TRAIN endpoints; all5 still required for full completion",
        )
    if public_history:
        result.update(
            protocol_id="minigrid_fixed_a64_paired_public_history_endpoints_v1",
            classification="exploratory_single_seed_distinct_public_information_and_capacity_control",
            feature_dim=296,
            training_checkpoints=list(POINTS),
            checkpoints=[0, 64],
            reference_checkpoints=[0, 64],
            reference_path=PAIRED_REFERENCE_PATH,
            source_reference_path="artifacts/runs/r21-fixed-a-acquisition-20261016/native-result.json",
            initialization="copy raw148 cold parameters; added first-frame weights zero; before any SGD compare both TRAIN A probability matrices at abs1e-7/rtol0 and identical argmax, including retained raw0 law;4forwards36rows",
            parameter_counts=dict(raw=1319, history=2503, added=1184),
            evaluation="after all64updates: original2A raw scores and2target-free native episodes at0/64 only; no separate B/C or middle readouts; five stream states still captured",
            comparison="history-minus-raw-paired CE gain/success-count change/mean-return gain; positive better; same64steps/128episodes/1152rows, different capacity/public inputs/compute",
            limitations="new endpoint-only condition does not complete old full-five protocols; no final/probes/replay/teachers/tuning/selection/CPC/retention/recovery/generalization claim",
        )
        result["planned_work"].update(
            initialization_forward_calls=4,
            initialization_forward_rows=36,
            raw_scorer_calls=0,
            raw_scorer_rows=0,
            live_evaluator_raw_calls=4,
            live_evaluator_raw_rows=36,
            live_episodes=4,
            max_live_predictions=64,
            max_native_forward_calls=200,
            max_native_forward_rows=1288,
        )
        result["limits"]["output_bytes"] = 8 * 1024**2
    return result


def _load_paired_reference() -> dict[str, Any]:
    """Admit the already accepted fixed endpoints, keeping full failure explicit."""
    from src.infra.key_door_inspector_files import _finite_float, _reject_constant, _unique_object

    path = Path(__file__).resolve().parents[2] / PAIRED_REFERENCE_PATH
    with path.open("rb") as handle:
        payload = handle.read(8 * 1024**2 + 1)
    if len(payload) > 8 * 1024**2:
        raise ValueError("paired reference exceeds8MiB")
    saved = json.loads(
        payload.decode("utf8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )
    protocol, training, endpoint = saved["protocol"], saved["training"], saved["endpoint_evidence"]
    expected = describe_protocol(endpoint_first=True)
    if (
        protocol != expected
        or saved["status"] != "incomplete"
        or saved["initial_parity"] is not True
        or saved["readback_error"] is not None
        or endpoint["status"] != "validated"
        or endpoint["checkpoint_updates"] != [0, 64]
        or any(
            type(training[k]) is not int or training[k] != value
            for k, value in (
                ("charged_updates", 64),
                ("native_training_episodes", 128),
                ("native_training_examples", 1152),
            )
        )
        or training["source_unchanged"] is not True
        or training["actor_unchanged"] is not True
    ):
        raise ValueError("retained raw paired reference no longer binds accepted endpoint evidence")
    records = deepcopy(_fixed_endpoints(saved["evaluations"], 64))
    metrics = _metrics(records, final_point=64)
    if metrics != endpoint["outcomes"]["paired"]:
        raise ValueError("retained paired endpoint metrics changed")
    return dict(
        evaluations=records,
        metrics=metrics,
        provenance=dict(
            path=PAIRED_REFERENCE_PATH,
            full_status="incomplete",
            endpoint_status="validated",
            checkpoint_updates=[0, 64],
        ),
    )


def _history_learner(
    schedule: tuple, sources: tuple, *, diagnostics: dict[str, Any] | None = None
) -> tuple[Any, dict[str, Any]]:
    """Preserve the actual cold law, measuring its parity before any SGD."""
    from src.adapters.minigrid_learnability import _history_packets
    from src.adapters.torch_recurrent_episode_batch import TorchRecurrentEpisodeBatchLearner
    from src.adapters.torch_recurrent_public_history import initialize_public_history_control

    details = {} if diagnostics is None else diagnostics
    if type(details) is not dict or details:
        raise ValueError("history initialization requires an empty private diagnostic buffer")
    details.update(
        status="running",
        before_training=True,
        initial_law_matches=False,
        forward_calls_attempted=0,
        forward_rows_attempted=0,
        forward_calls=0,
        forward_rows=0,
        curves=[],
    )
    try:
        raw = TorchRecurrentEpisodeBatchLearner(TorchRecurrentPolicy(148, 8, 7, 17))
        raw_before = raw.snapshot_state()
        state = initialize_public_history_control(raw_before)
        learner = TorchTrainEpisodeBatchLearner(state.policy, schedule)
        learner.restore_state(state)
        matches = True
        for packet, encoded in zip(sources[:2], _history_packets(sources[:2]), strict=True):
            curve: dict[str, Any] = dict(
                context=packet.context,
                cue=shared._episode(packet).episode_id.rsplit("-", 1)[-1],
                raw_probabilities=None,
                history_probabilities=None,
            )
            details["curves"].append(curve)
            predictions = []
            for owner, features, key in (
                (raw, packet.features, "raw_probabilities"),
                (learner, encoded.features, "history_probabilities"),
            ):
                details["forward_calls_attempted"] += 1
                details["forward_rows_attempted"] += len(features)
                values = owner.predict(features)
                details["forward_calls"] += 1
                details["forward_rows"] += len(features)
                curve[key] = values.tolist()
                predictions.append(values)
            before, after = predictions
            matches = matches and bool(
                np.allclose(before, after, rtol=0, atol=1e-7)
                and np.array_equal(np.argmax(before, axis=1), np.argmax(after, axis=1))
            )
            curve["maximum_absolute_difference"] = float(np.max(np.abs(before - after)))
        details.update(
            status="complete",
            initial_law_matches=matches,
            source_unchanged=shared._same_state(state, learner.snapshot_state()),
            raw_source_unchanged=shared._same_state(raw_before, raw.snapshot_state()),
        )
        return learner, details
    except Exception as error:
        details.update(status="failed", reason=f"{type(error).__name__}: {error}")
        raise


def _pair_packets(original: tuple[AttributionBatch, ...]) -> tuple[AttributionBatch, ...]:
    if type(original) is not tuple or len(original) != 128:
        raise ValueError("paired acquisition requires all128 original fixed-A episode packets")
    result = []
    for index in range(64):
        pair = original[2 * index : 2 * index + 2]
        for packet in pair:
            episode = shared._episode(packet)
            if (
                packet.role != "train"
                or episode.role != "train"
                or episode.context != "A"
                or len(episode.transitions) != 9
            ):
                raise ValueError("paired transport requires original nine-row TRAIN A episodes")
        result.append(
            AttributionBatch(
                "train",
                "A",
                tuple(f"paired-{index}-{row}" for p in pair for row in p.row_ids),
                np.concatenate([p.features for p in pair]),
                np.concatenate([p.targets for p in pair]),
                None,
                7,
            )
        )
    return tuple(result)


def _fixed_endpoints(records: list[dict[str, Any]], final_point: int) -> list[dict[str, Any]]:
    if type(final_point) is not int or final_point not in (64, 128):
        raise ValueError("endpoint metrics require fixed64 paired or128 original endpoint")
    selected = []
    for point in (0, final_point):
        matches = [r for r in records if r.get("updates") == point]
        if len(matches) != 1:
            raise ValueError("endpoint metrics require exactly one initial and fixed final record")
        record = matches[0]
        if (
            type(record.get("updates")) is not int
            or record.get("status") != "complete"
            or record.get("role") != "train"
            or record.get("method") != (METHOD if final_point == 64 else "recurrent_bc")
            or type(record.get("scores")) is not list
            or type(record.get("episodes")) is not list
            or len(record["scores"]) != 2
            or len(record["episodes"]) != 2
        ):
            raise ValueError("endpoint metrics require complete owned TRAIN A raw/live records")
        for cue, score, episode in zip(
            ("blue", "red"), record["scores"], record["episodes"], strict=True
        ):
            if (
                score.get("context") != "A"
                or score.get("cue") != cue
                or episode.get("context") != "A"
                or episode.get("cue") != cue
                or episode.get("status") != "complete"
                or type(episode.get("success")) is not bool
                or type(score.get("exact_sequence_accuracy")) not in (int, float)
                or score["exact_sequence_accuracy"] not in (0, 1)
                or type(score.get("ce")) not in (int, float)
                or not isfinite(score["ce"])
                or score["ce"] < 0
                or type(episode.get("return")) not in (int, float)
                or not isfinite(episode["return"])
            ):
                raise ValueError(
                    "endpoint metrics require both finite complete native cue outcomes"
                )
        selected.append(record)
    return selected


def _metrics(records: list[dict[str, Any]], *, final_point: int | None = None) -> dict[str, Any]:
    if final_point is not None:
        records = _fixed_endpoints(records, final_point)
    initial, final = records[0], records[-1]
    raw = all(s["exact_sequence_accuracy"] == 1 for s in final["scores"])
    live = all(e["success"] is True for e in final["episodes"])
    return dict(
        raw_both_cues_exact=raw,
        live_both_cues_goals=live,
        joint_acquisition=raw and live,
        ce_gain=fsum(s["ce"] for s in initial["scores"]) / 2
        - fsum(s["ce"] for s in final["scores"]) / 2,
        live_success_count_change=sum(e["success"] for e in final["episodes"])
        - sum(e["success"] for e in initial["episodes"]),
        mean_return_gain=fsum(e["return"] for e in final["episodes"]) / 2
        - fsum(e["return"] for e in initial["episodes"]) / 2,
    )


def _comparison(records: list, reference: list, *, endpoint_first: bool) -> dict[str, Any]:
    paired = _metrics(records, final_point=64) if endpoint_first else _metrics(records)
    original = _metrics(reference, final_point=128) if endpoint_first else _metrics(reference)
    return dict(
        paired=paired,
        original=original,
        paired_minus_original={
            key: paired[key] - original[key]
            for key in ("ce_gain", "live_success_count_change", "mean_return_gain")
        },
    )


def _history_comparison(records: list, reference: list) -> dict[str, Any]:
    history, raw = _metrics(records, final_point=64), _metrics(reference, final_point=64)
    return dict(
        history=history,
        raw_paired=raw,
        history_minus_raw_paired={
            key: history[key] - raw[key]
            for key in ("ce_gain", "live_success_count_change", "mean_return_gain")
        },
    )


def _wall_stop(error: BaseException | None) -> bool:
    if isinstance(error, MiniGridEvaluationFailed):
        if error.report.get("readback_error") is not None or any(
            episode.get("close_error") is not None or episode.get("readback_error") is not None
            for episode in error.report.get("episodes", ())
        ):
            return False
        error = error.__cause__
    return type(error) is ToyExecutionStopped and error.stop.reason == "max_wall_seconds"


def _work(training: dict[str, Any], raw: list, live: list, failed: bool) -> dict[str, Any]:
    charged = training.get("charged_updates") or 0
    failure_work = training.get("paired_failure_work") or {}
    raw_keys = (
        "score_forward_calls_attempted",
        "score_forward_calls_completed",
        "score_rows_attempted",
        "score_rows_completed",
    )
    result = dict(
        charged_updates=charged,
        committed_training_rows=18 * charged,
        committed_episode_exposures=2 * charged,
        observed_physical_optimizer_steps=charged + failure_work.get("optimizer_steps", 0),
        failed_native_work=deepcopy(failure_work),
        failure_work_may_be_unobserved=failed,
    )
    result["raw_scores"] = {k: sum(r["work"].get(k, 0) for r in raw) for k in raw_keys}
    keys = (
        "raw_score_forwards_attempted",
        "raw_score_forwards_completed",
        "raw_score_rows_attempted",
        "raw_score_rows_completed",
        "live_prediction_calls_attempted",
        "live_prediction_calls_returned",
        "live_sessions_started",
        "environment_resets_completed",
        "environment_steps_returned",
    )
    result["live_evaluation"] = {k: sum(r["work"].get(k, 0) for r in live) for k in keys}
    result["observed_forward_rows"] = (
        result["committed_training_rows"]
        + failure_work.get("forward_rows_completed", 0)
        + result["raw_scores"]["score_rows_completed"]
        + result["live_evaluation"]["raw_score_rows_completed"]
        + result["live_evaluation"]["live_prediction_calls_returned"]
    )
    result["observed_backward_rows"] = result["committed_training_rows"] + failure_work.get(
        "backward_rows_completed", 0
    )
    return result


def run_pilot(*, endpoint_first: bool = False, public_history: bool = False) -> dict[str, Any]:
    """Execute one frozen recipe; preserve paid prefixes and refuse changed owners."""
    from scripts.run_minigrid_history_evaluation_smoke import _fixed_a_packets

    if type(endpoint_first) is not bool:
        raise ValueError("endpoint_first must be an exact boolean")
    if type(public_history) is not bool or (public_history and endpoint_first):
        raise ValueError("public_history must be an exact boolean and excludes endpoint_first")
    started = monotonic()
    report: dict[str, Any] = dict(
        protocol=describe_protocol(public_history=True)
        if public_history
        else (describe_protocol(endpoint_first=True) if endpoint_first else describe_protocol()),
        status="running",
        reason=None,
        reference={},
        training=shared._row(METHOD),
        raw_readouts=[],
        evaluations=[],
        initial_parity=None,
        outcomes=None,
        work=None,
        readout_budget=None,
        observed_process_rss=None,
        total_seconds=None,
        readback_error=None,
    )
    if endpoint_first:
        report["endpoint_evidence"] = None
    initialization: dict[str, Any] = {}
    initialization_payload = pickle.dumps(initialization, protocol=5)
    paired_reference: dict[str, Any] | None = None
    paired_reference_payload: bytes | None = None
    if public_history:
        report["initialization"] = initialization
    protocol, reference = report["protocol"], report["reference"]
    training: dict[str, Any] = report["training"]
    training.pop("final_evaluation", None)
    training["readouts"] = []
    raw, live = report["raw_readouts"], report["evaluations"]
    protocol_payload = pickle.dumps(protocol, protocol=5)
    reference_payload = pickle.dumps(reference, protocol=5)
    training_payload = None
    raw_payload = live_payload = pickle.dumps([], protocol=5)
    sources = packets = original = None
    signatures: list[tuple[Any, Any]] = []
    arm: Any = None
    owners: Any = None
    memory: Any = None
    budget: Any = None
    birth = last_clock = None
    budget_payload = None
    frozen_budget = None
    failure = None
    failure_error: Exception | None = None
    endpoint_payload: bytes | None = None
    outcomes = None
    initial_parity = None
    observed_rss = None

    def evidence() -> None:
        if (
            report.get("protocol") is not protocol
            or pickle.dumps(protocol, protocol=5) != protocol_payload
            or report.get("reference") is not reference
            or pickle.dumps(reference, protocol=5) != reference_payload
            or report.get("training") is not training
            or (
                endpoint_first
                and ("endpoint_evidence" not in report or report["endpoint_evidence"] is not None)
            )
            or training.get("method") != METHOD
            or (
                training_payload is not None
                and pickle.dumps(training, protocol=5) != training_payload
            )
            or report.get("raw_readouts") is not raw
            or pickle.dumps(raw, protocol=5) != raw_payload
            or report.get("evaluations") is not live
            or pickle.dumps(live, protocol=5) != live_payload
            or any(
                not shared._check_signature(values, signature) for values, signature in signatures
            )
            or (
                public_history
                and (
                    report.get("initialization") is not initialization
                    or pickle.dumps(initialization, protocol=5) != initialization_payload
                    or (
                        paired_reference is not None
                        and pickle.dumps(paired_reference, protocol=5) != paired_reference_payload
                    )
                )
            )
        ):
            raise ValueError("original paired protocol/input/paid evidence changed")
        if arm is not None and any(arm.get(k) is not v for k, v in owners.items()):
            raise ValueError("original paired completed runtime bindings changed")
        if budget is not None:
            _check_budget(budget, birth, 0, last_clock)
            if budget.process_rss_sampler is not memory:
                raise ValueError("original paired readout sampler changed")
            if (
                frozen_budget is not None
                and pickle.dumps(record_budget(), protocol=5) != frozen_budget
            ):
                raise ValueError("completed paired readout budget changed")

    def boundary() -> None:
        evidence()
        if monotonic() >= started + BODY_SECONDS:
            raise ValueError("original75second paired pilot deadline exhausted")
        if memory is not None:
            memory.sample()
            if memory.snapshot().peak_bytes > RSS_LIMIT:
                raise ValueError("paired pilot process RSS cap exhausted")
        # Resource callbacks may change caller-owned records; prove them again.
        evidence()

    def record_budget() -> dict[str, Any]:
        return dict(
            limits=asdict(budget.budget),
            started_at=budget.started_at,
            last_clock=budget.last_clock,
            updates_completed=budget.updates_completed,
            replay_examples_completed=budget.replay_examples_completed,
            elapsed_seconds=budget.last_clock - budget.started_at,
            process_rss=None
            if budget.process_rss_segment is None
            else asdict(budget.process_rss_segment),
        )

    def guard() -> None:
        nonlocal last_clock, budget_payload
        boundary()
        if arm is not None:
            arm["guard"]()
        boundary()
        if budget is not None and frozen_budget is None:
            budget.before_final()
            _check_budget(budget, birth, 0, last_clock)
            last_clock = budget.last_clock
            boundary()
            budget_payload = pickle.dumps(record_budget(), protocol=5)

    try:
        with ProcessRssSampler() as active_memory:
            memory = active_memory
            admitted = _admit_reference(_load_reference())
            reference.update(deepcopy(admitted["provenance"]))
            if public_history:
                paired_reference = _load_paired_reference()
                paired_reference_payload = pickle.dumps(paired_reference, protocol=5)
                _check_initial_condition(
                    admitted["evaluations"][0], paired_reference["evaluations"][0]
                )
                reference["paired_raw"] = deepcopy(paired_reference["provenance"])
                reference["paired_raw_metrics"] = deepcopy(paired_reference["metrics"])
            reference_payload = pickle.dumps(reference, protocol=5)
            sources = admitted["sources"]
            original = _fixed_a_packets(sources)
            if public_history:
                from src.adapters.minigrid_learnability import _history_packets

                signatures.append((original, shared._signature(original)))
                original = _history_packets(original)
            packets = _pair_packets(original)
            schedule = tuple(shared._episode(p) for p in original)
            signatures.extend((v, shared._signature(v)) for v in (sources, original, packets))

            def factory(method: str) -> Any:
                nonlocal initialization_payload
                if method != METHOD:
                    raise ValueError("paired factory refuses other methods")
                if public_history:
                    private_details: dict[str, Any] = {}
                    try:
                        learner, details = _history_learner(
                            schedule, sources, diagnostics=private_details
                        )
                        if details is not private_details:
                            raise ValueError(
                                "history initialization changed its private result owner"
                            )
                    finally:
                        initialization.clear()
                        initialization.update(deepcopy(private_details))
                        initialization_payload = pickle.dumps(initialization, protocol=5)
                    if (
                        initialization.get("status") != "complete"
                        or any(
                            initialization.get(k) is not True
                            for k in (
                                "before_training",
                                "initial_law_matches",
                                "source_unchanged",
                                "raw_source_unchanged",
                            )
                        )
                        or initialization.get("forward_calls") != 4
                        or initialization.get("forward_rows") != 36
                    ):
                        raise ValueError(
                            "history initial-law/state qualification failed before training"
                        )
                    assert paired_reference is not None
                    for curve, retained in zip(
                        initialization["curves"],
                        paired_reference["evaluations"][0]["scores"],
                        strict=True,
                    ):
                        a, b = (
                            np.asarray(curve["raw_probabilities"]),
                            np.asarray(retained["probabilities"]),
                        )
                        if (
                            (curve["context"], curve["cue"])
                            != (retained["context"], retained["cue"])
                            or a.shape != (9, 7)
                            or b.shape != a.shape
                            or not np.allclose(a, b, rtol=0, atol=1e-7)
                            or not np.array_equal(np.argmax(a, axis=1), np.argmax(b, axis=1))
                        ):
                            raise ValueError("pre-SGD cold law differs from retained raw paired0")
                    boundary()
                    return learner
                return TorchTrainEpisodeBatchLearner(TorchRecurrentPolicy(148, 8, 7, 17), schedule)

            try:
                arm = shared._train_arm(
                    training,
                    packets,
                    memory,
                    started + TRAIN_SECONDS,
                    boundary,
                    learner_factory=factory,
                    checkpoint_updates=POINTS,
                    **({"public_history": True} if public_history else {}),
                )
            finally:
                training_payload = pickle.dumps(training, protocol=5)
            owners = {k: arm[k] for k in ("source", "states", "guard", "budget", "stream")}
            guard()
            budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=0,
                    max_wall_seconds=min(
                        READOUT_SECONDS, max(0.0, started + BODY_SECONDS - monotonic())
                    ),
                    max_process_rss_bytes=RSS_LIMIT,
                ),
                monotonic,
            )
            birth, last_clock = _binding(budget), budget.last_clock
            budget.attach_memory(memory)
            guard()
            states = arm["states"]
            if public_history:
                if [point for point, _ in states] != list(POINTS):
                    raise ValueError("history acquisition requires all five captured stream states")
                by_point = dict(states)
                states = [(point, by_point[point]) for point in (0, 64)]
            if endpoint_first:
                if [point for point, _ in states] != list(POINTS):
                    raise ValueError(
                        "endpoint-first requires all original sorted training checkpoints"
                    )
                by_point = dict(states)
                states = [(point, by_point[point]) for point in ENDPOINT_FIRST_POINTS]
            for point, state in states:
                guard()
                if not public_history:
                    current = _score_checkpoint(
                        arm["source"],
                        state,
                        method=METHOD,
                        updates=point,
                        sources=sources,
                        budget=budget,
                        check_boundary=guard,
                    )
                    raw.append(current)
                    raw_payload = pickle.dumps(raw, protocol=5)
                    guard()
                    if endpoint_first and (
                        current.get("method") != METHOD
                        or type(current.get("updates")) is not int
                        or current["updates"] != point
                        or current.get("role") != "train"
                    ):
                        raise ValueError(
                            "endpoint-first raw readout metadata differs from its checkpoint"
                        )
                    if current["status"] != "complete":
                        raise ValueError(f"paired raw readout failed: {current.get('reason')}")
                try:
                    current = evaluate_checkpoint(
                        arm["source"],
                        state,
                        method=METHOD,
                        updates=point,
                        layouts=predefined_layouts()[:1],
                        role="train",
                        sources=sources[:2],
                        budget=budget,
                        check_boundary=guard,
                        **({"public_history": True} if public_history else {}),
                    )
                except MiniGridEvaluationFailed as error:
                    live.append(deepcopy(error.report))
                    live_payload = pickle.dumps(live, protocol=5)
                    raise
                live.append(current)
                live_payload = pickle.dumps(live, protocol=5)
                guard()
                if endpoint_first and (
                    current.get("status") != "complete"
                    or current.get("method") != METHOD
                    or type(current.get("updates")) is not int
                    or current["updates"] != point
                    or current.get("role") != "train"
                ):
                    raise ValueError(
                        "endpoint-first requires each declared live readout to complete"
                    )
                if point == 0:
                    _check_initial_condition(
                        paired_reference["evaluations"][0]
                        if public_history and paired_reference is not None
                        else admitted["evaluations"][0],
                        current,
                    )
                    initial_parity = True
                    guard()
                if endpoint_first and point == 64:
                    # Why: the fixed scientific endpoint must be observed before
                    # descriptive middle states consume the same original clock.
                    endpoint = _comparison(live, admitted["evaluations"], endpoint_first=True)
                    guard()
                    endpoint_payload = pickle.dumps(
                        dict(
                            status="validated",
                            checkpoint_updates=[0, 64],
                            outcomes=endpoint,
                        ),
                        protocol=5,
                    )
            if public_history:
                if raw or [r.get("updates") for r in live] != [0, 64] or initial_parity is not True:
                    raise ValueError(
                        "history recipe requires exactly both fixed endpoints and initial parity"
                    )
            elif len(raw) != 5 or len(live) != 5 or initial_parity is not True:
                raise ValueError(
                    "paired recipe requires all five complete readouts and initial parity"
                )
            if endpoint_first and any(
                [record.get("updates") for record in records] != list(ENDPOINT_FIRST_POINTS)
                for records in (raw, live)
            ):
                raise ValueError(
                    "endpoint-first full completion requires all five declared readouts in order"
                )
            if public_history:
                assert paired_reference is not None
                outcomes = _history_comparison(live, paired_reference["evaluations"])
            else:
                outcomes = _comparison(live, admitted["evaluations"], endpoint_first=endpoint_first)
            budget.complete_memory()
            guard()
            frozen_budget = pickle.dumps(record_budget(), protocol=5)
            budget_payload = frozen_budget
            guard()
    except Exception as error:
        failure_error = error
        failure = f"{type(error).__name__}: {error}"
    try:
        if memory is not None:
            observed_rss = asdict(memory.snapshot())
        report["total_seconds"] = monotonic() - started
        if failure is None:
            guard()
    except Exception as error:
        report["readback_error"] = f"{type(error).__name__}: {error}"
        failure_error = error
        failure = failure or report["readback_error"]
    endpoint_evidence = None
    if endpoint_payload is not None:
        try:
            if failure is not None:
                if report["readback_error"] is not None or not _wall_stop(failure_error):
                    raise ValueError(
                        "later non-wall failure prevents endpoint integrity acceptance"
                    )
                # Integrity readback only: the completed arm still enforces its
                # original45s/source/actor/native proofs. Never re-admit work or
                # call the expired diagnostic20s budget through outer guard().
                boundary()
                arm["guard"]()
                report["total_seconds"] = monotonic() - started
                if report["total_seconds"] >= BODY_SECONDS:
                    raise ValueError(
                        "original75second deadline prevents endpoint integrity acceptance"
                    )
                evidence()
            endpoint_evidence = pickle.loads(endpoint_payload)
        except Exception as error:
            endpoint_evidence = dict(
                status="invalid_or_unverified",
                checkpoint_updates=[0, 64],
                outcomes=None,
                reason=f"{type(error).__name__}: {error}",
            )
    paid_training = pickle.loads(training_payload) if training_payload is not None else {}
    paid_raw, paid_live = pickle.loads(raw_payload), pickle.loads(live_payload)
    report.update(
        protocol=pickle.loads(protocol_payload),
        reference=pickle.loads(reference_payload),
        training=paid_training,
        raw_readouts=paid_raw,
        evaluations=paid_live,
        status="complete" if failure is None else "incomplete",
        reason=failure,
        initial_parity=initial_parity,
        outcomes=outcomes if failure is None else None,
        work=_work(paid_training, paid_raw, paid_live, failure is not None),
        readout_budget=None if budget_payload is None else pickle.loads(budget_payload),
        observed_process_rss=observed_rss,
    )
    if endpoint_first:
        report["endpoint_evidence"] = endpoint_evidence
    if public_history:
        report["initialization"] = pickle.loads(initialization_payload)
        report["work"]["initialization"] = dict(
            forward_calls=report["initialization"].get("forward_calls", 0),
            forward_rows=report["initialization"].get("forward_rows", 0),
        )
        report["work"]["observed_forward_rows"] += report["work"]["initialization"]["forward_rows"]
    return report
