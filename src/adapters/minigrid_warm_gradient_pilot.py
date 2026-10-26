"""Measure TRAIN cue-gradient geometry on the accepted reconstructed A128 RNN.

This thin use case composes the existing warm stream/parity helper and raw
checkpoint scorer. It introduces no trainer, model math, data collection,
final/probe access or tuning; partial paid evidence remains available on failure.
"""

from copy import deepcopy
from dataclasses import asdict
import pickle
from time import monotonic
from typing import Any

from src.adapters import minigrid_learning_pilot as shared
from src.adapters.key_door_reward_pilot import _binding, _check_budget
from src.adapters.minigrid_gradient_readouts import empty_work, validate_sources
from src.adapters.minigrid_learnability import _score_checkpoint
from src.adapters.minigrid_warm_start import prepare_warm_start
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.process_memory import ProcessRssSampler

BODY_SECONDS = 75.0
WARM_SECONDS = 45.0
DIAGNOSTIC_SECONDS = 20.0
RSS_LIMIT = 768 * 1024**2


def describe_protocol(*, cue_updates: bool = False) -> dict[str, Any]:
    """Declare fixed TRAIN inputs, mean-CE diagnostic and resources without work."""
    if type(cue_updates) is not bool:
        raise ValueError("cue_updates must be an exact bool")
    protocol: dict[str, Any] = dict(
        protocol_id="minigrid_warm_a128_cue_gradient_geometry_v1",
        classification="exploratory_train_only_frozen_recurrent_gradient_measurement",
        method="recurrent_bc",
        source_role="train",
        updates=128,
        training_rows=1152,
        feature_dim=148,
        hidden_dim=8,
        num_classes=7,
        seed=17,
        learning_rate=0.03,
        momentum=0.9,
        max_episode_steps=16,
        reference_path="artifacts/runs/r21-fixed-a-acquisition-20261016/native-result.json",
        warm_start="newly charged original A128 reconstruction; all five raw and both0/128native cue-pair parity checks EXACT; restored parameters/gradients/optimizer/momentum",
        diagnostic="after warm acceptance: original six nine-row TRAIN148/K7 A/B/C_train blue/red raw scores, then three detached original mean-CE gradient pairs; no SGD",
        definition="torch_recurrent_train_mean_ce_gradient_geometry_v1",
        budget_scope="independent zero-update diagnostic budget does not renew completed warm clock/charges; original45s warm guard and75s whole deadline remain binding",
        planned_work=dict(
            native_train_calls=128,
            training_rows=1152,
            warm_raw_score_forwards=10,
            warm_raw_score_rows=90,
            warm_live_episodes=4,
            maximum_warm_live_predictions=64,
            new_teacher_episodes=0,
            diagnostic_raw_forwards=6,
            diagnostic_raw_rows=54,
            gradient_pairs=3,
            gradient_forward_calls=6,
            gradient_forward_rows=54,
            gradient_backward_calls=6,
            gradient_backward_rows=54,
            gradient_geometry_reductions=18,
            gradient_input_array_equivalent_bytes=64368,
            diagnostic_optimizer_steps=0,
            final_episodes=0,
            probe_updates=0,
        ),
        limits=dict(
            body_seconds=BODY_SECONDS,
            parent_seconds=90,
            warm_seconds=WARM_SECONDS,
            diagnostic_seconds=DIAGNOSTIC_SECONDS,
            process_rss_bytes=RSS_LIMIT,
            output_bytes=32 * 1024**2,
            cpu_threads=1,
            interop_threads=1,
            gpu=0,
        ),
        limitations="TRAIN-only one frozen checkpoint; signed/undefined whole-episode geometry may hide cue-local conflict; no causal interference, useful acquisition, retention/recovery, generalization or CPC advantage established",
    )
    if cue_updates:
        protocol.update(
            protocol_id="minigrid_warm_a128_isolated_cue_updates_v1",
            classification="exploratory_train_only_two_isolated_native_sgd_updates",
            intervention="two full-state A128 forks with original momentum; one original A-blue mean-CE update versus one original A-red mean-CE update; score all six TRAIN packets on both129/1161 states; keep both arms",
            comparison="per-packet after-minus-before CE and blue-arm-minus-red-arm CE; lower CE is better; shared carried momentum affects absolute changes",
            limitations="one TRAIN-only state/two single-update arms; no pure-gradient causal attribution, useful acquisition, fresh-task learning, retention/recovery, generalization or CPC advantage established",
        )
        protocol["planned_work"].update(
            native_train_calls=130,
            training_rows=1170,
            isolated_cue_updates=2,
            isolated_training_rows=18,
            isolated_raw_forwards=12,
            isolated_raw_rows=108,
            isolated_update_budget=2,
            isolated_update_seconds=20.0,
        )
    return protocol


def _new_report() -> dict[str, Any]:
    return dict(
        protocol=describe_protocol(),
        status="running",
        reason=None,
        warm_start={},
        gradient_readout=dict(status="not_run", reason="not reached"),
        diagnostic_budget=None,
        work=None,
        observed_process_rss=None,
        total_seconds=None,
        readback_error=None,
    )


def _work(warm: dict[str, Any], readout: dict[str, Any], failed: bool) -> dict[str, Any]:
    warm_work = deepcopy(warm.get("work"))
    raw = readout.get("work", {})
    raw_keys = (
        "score_forward_calls_attempted",
        "score_forward_calls_completed",
        "score_rows_attempted",
        "score_rows_completed",
    )
    return dict(
        warm_start=warm_work,
        raw_scores={key: raw.get(key, 0) for key in raw_keys},
        cue_gradient_geometry=deepcopy(raw.get("cue_gradient_geometry", empty_work())),
        native_train_calls=None if warm_work is None else warm_work["charged_updates"],
        training_rows=None if warm_work is None else warm_work["observed_native_training_rows"],
        failure_work_may_be_unobserved=failed
        or warm_work is None
        or warm_work.get("failure_work_may_be_unobserved", False)
        or raw.get("failure_work_may_be_unobserved", False),
    )


def run_pilot(*, cue_updates: bool = False) -> dict[str, Any]:
    """Reconstruct once, verify parity, and retain one budgeted TRAIN readout."""
    if type(cue_updates) is not bool:
        raise ValueError("cue_updates must be an exact bool")
    started = monotonic()
    deadline = started + BODY_SECONDS
    report = _new_report()
    cue_readout = dict(status="not_run", reason="not reached")
    cue_payload = pickle.dumps(cue_readout, protocol=5)
    cue_active = False
    if cue_updates:
        report["protocol"] = describe_protocol(cue_updates=True)
        report["cue_update_readout"] = cue_readout
    protocol, warm, readout = report["protocol"], report["warm_start"], report["gradient_readout"]
    protocol_payload = pickle.dumps(protocol, protocol=5)
    readout_payload = pickle.dumps(readout, protocol=5)
    warm_payload = None
    memory: Any = None
    prepared: Any = None
    source: Any = None
    sources: Any = None
    source_before = signature = None
    warm_guard: Any = None
    budget: Any = None
    birth = last_clock = frozen_clock = frozen_rss = None
    frozen_rss_owner = None
    budget_payload = None
    failure = None
    observed_rss = None

    def evidence() -> None:
        if (
            report.get("protocol") is not protocol
            or pickle.dumps(protocol, protocol=5) != protocol_payload
            or report.get("warm_start") is not warm
            or (warm_payload is not None and pickle.dumps(warm, protocol=5) != warm_payload)
            or report.get("gradient_readout") is not readout
            or pickle.dumps(readout, protocol=5) != readout_payload
            or (
                cue_updates
                and (
                    report.get("cue_update_readout") is not cue_readout
                    or (not cue_active and pickle.dumps(cue_readout, protocol=5) != cue_payload)
                )
            )
        ):
            raise ValueError("original warm-gradient protocol or paid evidence changed")
        if prepared is not None and (
            prepared.get("source") is not source
            or prepared.get("guard") is not warm_guard
            or prepared.get("sources") is not sources
            or type(prepared.get("updates")) is not int
            or prepared["updates"] != 128
            or type(prepared.get("rows")) is not int
            or prepared["rows"] != 1152
        ):
            raise ValueError("original returned warm source/guard/input binding changed")
        if signature is not None and not shared._check_signature(sources, signature):
            raise ValueError("original six TRAIN source packets changed")
        if budget is not None:
            _check_budget(budget, birth, 0, last_clock)
            if budget.process_rss_sampler is not memory or (
                frozen_clock is not None
                and (
                    budget.last_clock != frozen_clock
                    or budget.process_rss_segment is not frozen_rss_owner
                    or not _same_state(frozen_rss, budget.process_rss_segment)
                )
            ):
                raise ValueError(
                    "original zero-update diagnostic budget/clock/resource binding changed"
                )

    def boundary() -> None:
        evidence()
        if monotonic() >= deadline:
            raise ValueError("original75s warm-gradient whole deadline exhausted")
        if memory is not None:
            memory.sample()
            if memory.snapshot().peak_bytes > RSS_LIMIT:
                raise ValueError("warm-gradient process RSS cap exhausted")
        evidence()

    def record_budget() -> dict[str, Any]:
        return dict(
            limits=asdict(budget.budget),
            started_at=budget.started_at,
            last_clock=budget.last_clock,
            elapsed_seconds=budget.last_clock - budget.started_at,
            updates_completed=budget.updates_completed,
            replay_examples_completed=budget.replay_examples_completed,
            process_rss=None
            if budget.process_rss_segment is None
            else asdict(budget.process_rss_segment),
        )

    def guard() -> None:
        nonlocal last_clock, budget_payload
        boundary()
        if warm_guard is not None:
            warm_guard()
        if source_before is not None and not _same_state(source_before, source.snapshot_state()):
            raise ValueError("warm-gradient source full native state changed")
        evidence()
        if budget is not None and frozen_clock is None:
            budget.before_final()
            _check_budget(budget, birth, 0, last_clock)
            last_clock = budget.last_clock
            boundary()
            if source_before is not None and not _same_state(
                source_before, source.snapshot_state()
            ):
                raise ValueError("warm-gradient source changed after budget callback")
            evidence()
            budget_payload = pickle.dumps(record_budget(), protocol=5)

    try:
        with ProcessRssSampler() as active_memory:
            memory = active_memory
            try:
                prepared = prepare_warm_start(
                    warm, memory, min(deadline, started + WARM_SECONDS), boundary
                )
            finally:
                warm_payload = pickle.dumps(warm, protocol=5)
            source, warm_guard, sources = prepared["source"], prepared["guard"], prepared["sources"]
            if (
                warm.get("status") != "complete"
                or warm.get("parity_matched") is not True
                or warm.get("full_state_restored") is not True
                or type(warm.get("parity_checks")) is not list
                or len(warm["parity_checks"]) != 5
                or any(value is not True for value in warm["parity_checks"])
                or not callable(warm_guard)
            ):
                raise ValueError(
                    "all original warm parity/full-state criteria must pass before diagnostics"
                )
            guard()
            source_before = deepcopy(source.snapshot_state())
            shared._counter(source_before, 128)
            signature = validate_sources(sources)
            guard()
            budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=0,
                    max_wall_seconds=min(DIAGNOSTIC_SECONDS, max(0.0, deadline - monotonic())),
                    max_process_rss_bytes=RSS_LIMIT,
                ),
                monotonic,
            )
            birth, last_clock = _binding(budget), budget.last_clock
            budget.attach_memory(memory)
            guard()
            current = _score_checkpoint(
                source,
                deepcopy(source_before),
                method="recurrent_bc",
                updates=128,
                sources=sources,
                budget=budget,
                check_boundary=guard,
                gradient_geometry=True,
            )
            # Retain the paid result before any subsequent external readback.
            readout = current
            readout_payload = pickle.dumps(current, protocol=5)
            report["gradient_readout"] = readout
            guard()
            if current["status"] != "complete":
                raise ValueError(f"warm-gradient readout incomplete: {current.get('reason')}")
            if cue_updates:
                from src.adapters.minigrid_cue_updates import run_cue_updates

                cue_active = True
                cue_readout.clear()
                try:
                    run_cue_updates(
                        cue_readout,
                        source,
                        source_before,
                        sources,
                        readout,
                        budget,
                        check_boundary=guard,
                    )
                finally:
                    # Retain the helper's private paid prefix before later callbacks.
                    cue_payload = pickle.dumps(cue_readout, protocol=5)
                    cue_active = False
                guard()
                if cue_readout["status"] != "complete":
                    raise ValueError(
                        f"isolated cue updates incomplete: {cue_readout.get('reason')}"
                    )
            budget.complete_memory()
            guard()
            frozen_clock = budget.last_clock
            frozen_rss_owner = budget.process_rss_segment
            frozen_rss = deepcopy(frozen_rss_owner)
            budget_payload = pickle.dumps(record_budget(), protocol=5)
            guard()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    try:
        if memory is not None:
            observed_rss = asdict(memory.snapshot())
        report["total_seconds"] = monotonic() - started
        if failure is None:
            guard()
    except Exception as error:
        report["readback_error"] = f"{type(error).__name__}: {error}"
        failure = failure or report["readback_error"]
    # Publish private paid captures only after every external callback finishes.
    paid_warm = pickle.loads(warm_payload) if warm_payload is not None else {}
    paid_readout = pickle.loads(readout_payload)
    report.update(
        protocol=pickle.loads(protocol_payload),
        warm_start=paid_warm,
        gradient_readout=paid_readout,
        diagnostic_budget=None if budget_payload is None else pickle.loads(budget_payload),
        status="complete" if failure is None else "incomplete",
        reason=failure,
        work=_work(paid_warm, paid_readout, failure is not None),
        observed_process_rss=observed_rss,
    )
    if cue_updates:
        paid_cues = pickle.loads(cue_payload)
        report["cue_update_readout"] = paid_cues
        cue_work = paid_cues.get("work", {})
        report["work"]["isolated_cue_updates"] = deepcopy(cue_work)
        for key, cue_key in (
            ("native_train_calls", "native_updates_committed"),
            ("training_rows", "committed_training_rows"),
        ):
            if report["work"][key] is not None:
                report["work"][key] += cue_work.get(cue_key, 0)
        report["work"]["failure_work_may_be_unobserved"] |= cue_work.get(
            "failure_work_may_be_unobserved", False
        )
    return report
