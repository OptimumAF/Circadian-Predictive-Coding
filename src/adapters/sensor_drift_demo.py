"""Small recurring sensor qualification on the existing delayed stream.

Inputs are the fixed public fixture and explicit existing native learner recipe. Output contains
shadow prearrival predictions, scores joined only after label arrival, and costs.
Opt-in monitoring fits only the first stationary-sensor prefix. No promotion,
model selection. Optional heldout evaluation runs after all training and probes.
Optional learnability probes use separate
disposable forks; their updates never reach the benchmark or actor.
"""

from dataclasses import asdict
from time import monotonic
from typing import Any


def _scheduled_report(stream: Any) -> dict[str, Any]:
    from src.adapters.scheduled_backprop_learner import ScheduledBackpropLearner

    owner = stream._runtime._candidate
    if type(owner) is not ScheduledBackpropLearner:
        raise ValueError("scheduled stream requires its scheduled learner owner")
    return owner.report()


def _replay_policy() -> Any:
    from src.app.stream_rehearsal import RecentStreamReplayPolicy
    from src.core.circadian_predictive_coding import ReplayRetentionBudget

    return RecentStreamReplayPolicy(4, 4, ReplayRetentionBudget(8, 1088))


def _validate_modes(
    monitoring: bool,
    scheduled: bool,
    replay: bool,
    method: str,
    learnability: bool,
    heldout: bool,
    incident_replay: bool,
    stationary: bool = False,
    contrast_inversion: bool = False,
) -> None:
    from src.adapters.sensor_learners import validate_sensor_method

    validate_sensor_method(method)
    if any(
        type(mode) is not bool
        for mode in (
            monitoring,
            scheduled,
            replay,
            learnability,
            heldout,
            incident_replay,
            stationary,
            contrast_inversion,
        )
    ):
        raise ValueError(
            "sensor monitoring, scheduled, replay, learnability, heldout, incident_replay, stationary and contrast_inversion flags must be exact bools"
        )
    if stationary and contrast_inversion:
        raise ValueError("stationary and contrast inversion are separate sensor conditions")
    if stationary and not monitoring:
        raise ValueError("stationary sensor control requires monitoring")
    if stationary and (learnability or heldout):
        raise ValueError(
            "stationary control does not use the recurring heldout or learnability protocols"
        )
    if incident_replay and not (replay and monitoring):
        raise ValueError("incident replay requires explicit replay and monitoring")
    if scheduled and replay:
        raise ValueError("scheduled finetuning and recent replay are separate controls")
    if method != "bp" and (scheduled or replay):
        raise ValueError("scheduled and replay sensor controls require method bp")
    if learnability and (scheduled or replay):
        raise ValueError(
            "sensor learnability requires ordinary methods without scheduled or replay"
        )


def describe_protocol(
    *,
    monitoring: bool = False,
    scheduled: bool = False,
    replay: bool = False,
    method: str = "bp",
    learnability: bool = False,
    heldout: bool = False,
    incident_replay: bool = False,
    stationary: bool = False,
    contrast_inversion: bool = False,
) -> dict[str, Any]:
    _validate_modes(
        monitoring,
        scheduled,
        replay,
        method,
        learnability,
        heldout,
        incident_replay,
        stationary,
        contrast_inversion,
    )
    result: dict[str, Any] = dict(
        protocol="sensor_brightness_prearrival_bp_v1",
        purpose="engineering qualification; no comparative scientific claim",
        method="matched_bp",
        seed=17,
        dimensions=[16, 8, 1],
        learning_rate=0.1,
        phases=["A", "brightness_plus_0.25", "A_return"],
        batches_per_phase=4,
        rows_per_batch=4,
        label_delay=2,
        prediction="current shadow candidate before same-tick label arrival/update",
        scoring="join retained probabilities only to arrived training labels",
        actor="frozen; no deployment or promotion",
        training_updates=12,
        training_rows=48,
        diagnostic_forward_calls=12,
        diagnostic_forward_rows=48,
        training_forward_backward_evaluations=24,
        training_evaluated_rows=96,
        training_evaluation_reason="preupdate gradient plus mandatory candidate gradient validation",
        body_seconds=30,
        process_rss_bytes=256 * 1024**2,
        final_evaluation=False,
        comparison=False,
    )
    if monitoring:
        from src.app.sensor_monitoring import monitoring_configuration

        result.update(
            protocol="sensor_brightness_prearrival_monitoring_bp_v1",
            monitoring=monitoring_configuration(stationary=True)
            if stationary
            else monitoring_configuration(),
        )
    if scheduled:
        result.update(
            protocol=(
                "sensor_brightness_scheduled_monitoring_bp_v1"
                if monitoring
                else "sensor_brightness_scheduled_bp_v1"
            ),
            method="scheduled_matched_bp",
            training_updates=3,
            wake_receipts=12,
            batches_per_optimizer_step=4,
            optimizer_batch_rows=16,
            training_forward_backward_evaluations=6,
            buffering_diagnostic="scheduled_bp_buffered_no_optimizer_v1; no task loss observed",
        )
    if replay:
        result.update(
            protocol=(
                "sensor_brightness_recent_replay_monitoring_bp_v1"
                if monitoring
                else "sensor_brightness_recent_replay_bp_v1"
            ),
            method="recent_replay_matched_bp",
            training_updates=22,
            training_rows=58,
            wake_receipts=12,
            wake_optimizer_steps=12,
            wake_training_rows=48,
            replay_training_updates=10,
            replay_training_rows=10,
            replay_allowance=12,
            replay_policy=asdict(_replay_policy()),
            replay_semantics="content-deduplicated arrived rows; one native update per selected row",
            training_forward_backward_evaluations=44,
            training_evaluated_rows=116,
        )
    if incident_replay:
        result.update(
            protocol=result["protocol"].removesuffix("_v1") + "_incident_feature_v1",
            incident_replay=dict(
                channel="feature",
                admission="one active onset observed strictly before next declared boundary",
                clear="expires unconsumed onset; continuation does not retrigger",
                timing="monitor observes arrived labels after rehearsal; never same-wake feedback",
                exposure_scope="existing permanent replay allowance; training counts above are upper bounds",
                limitations="engineering trigger qualification; no statistical FPR, recovery or benefit claim",
            ),
        )
        result["monitoring"]["feedback"] = True
        result["monitoring"]["feedback_scope"] = (
            "arrived feature incident to future replay boundary only"
        )
    if method != "bp":
        from src.core.coupled_energy import COUPLED_ENERGY_ID

        result.update(
            protocol=(
                f"sensor_brightness_prearrival_monitoring_{method}_v1"
                if monitoring
                else f"sensor_brightness_prearrival_{method}_v1"
            ),
            method=method,
            training_objective=COUPLED_ENERGY_ID,
            inference_steps=4,
            inference_learning_rate=0.2,
            inference_policy=None,
            precisions=[1.0],
            training_forward_backward_evaluations=72,
            training_evaluated_rows=288,
            training_latent_initializations=12,
            training_latent_initialization_rows=48,
            training_evaluation_reason=(
                "four fixed latent steps, final relaxed energy/gradient, "
                "and mandatory candidate energy/gradient validation"
            ),
        )
        if method != "pc":
            from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig

            result["chemistry"] = asdict(
                CoupledCircadianConfig.matched_pc_control()
                if method == "neutral_cpc"
                else CoupledCircadianConfig()
            )
    if learnability:
        from src.adapters.sensor_learnability import describe_learnability

        result.update(
            protocol=result["protocol"].removesuffix("_v1") + "_learnability_v1",
            learnability=describe_learnability(),
            total_training_updates=28,
            total_training_rows=176,
        )
    if heldout:
        from src.core.sensor_retention import describe_sensor_heldout

        result.update(
            protocol=result["protocol"].removesuffix("_v1") + "_heldout_v1",
            final_evaluation=True,
            heldout=describe_sensor_heldout(contrast_inversion=True)
            if contrast_inversion
            else describe_sensor_heldout(),
        )
    if contrast_inversion:
        result.update(
            protocol=result["protocol"].replace("sensor_brightness_", "sensor_contrast_inversion_"),
            contrast_inversion=True,
            phases=["A", "contrast_inversion", "A_return"],
            fixture="B pixels = 1 - original A pixels; semantic labels unchanged; A returns exactly",
            limitations="fixed camera contrast reversal; repeated training frames, one initialization, no comparative benefit established",
        )
    if stationary:
        result.update(
            protocol=result["protocol"].replace("sensor_brightness_", "sensor_stationary_"),
            stationary=True,
            phases=["A", "A", "A"],
            fixture="twelve exact baseline-A batches; no added noise or physical shift",
            limitations="eight post-calibration stationary opportunities; dependent repeated frames, not statistical FPR control or recovery evidence",
        )
        if replay:
            # Why: removing the shifted phase leaves only two unique arrived
            # rows. Keep the original content deduplication and allowance.
            result.update(
                training_updates=18,
                training_rows=54,
                replay_training_updates=6,
                replay_training_rows=6,
                training_forward_backward_evaluations=36,
                training_evaluated_rows=108,
            )
    return result


def run_demo(
    *,
    monitoring: bool = False,
    scheduled: bool = False,
    replay: bool = False,
    method: str = "bp",
    learnability: bool = False,
    heldout: bool = False,
    incident_replay: bool = False,
    stationary: bool = False,
    contrast_inversion: bool = False,
) -> dict[str, Any]:
    """Run one fixed small qualification; failures retain attempted diagnostic work."""
    _validate_modes(
        monitoring,
        scheduled,
        replay,
        method,
        learnability,
        heldout,
        incident_replay,
        stationary,
        contrast_inversion,
    )
    started = monotonic()
    from src.adapters.sensor_learners import build_sensor_learner
    from src.app.coupled_attribution import (
        AttributionBatch,
        AttributionStream,
        PrearrivalObservation,
        _same_value,
    )
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.core.actor_ports import CandidateState
    from src.core.binary_probability_scores import probability_scores
    from src.core.coupled_energy import predict_coupled
    from src.core.coupled_predictive_coding import (
        CoupledPredictiveCodingNetwork,
        CoupledSnapshot,
        CoupledWakeState,
    )
    from src.core.matched_backprop import MatchedBackpropNetwork, MatchedBackpropSnapshot
    from src.core.sensor_stream import make_sensor_stream
    from src.shared.process_memory import ProcessRssSampler

    protocol = describe_protocol(
        monitoring=monitoring,
        scheduled=scheduled,
        replay=replay,
        method=method,
        learnability=learnability,
        heldout=heldout,
        incident_replay=incident_replay,
        **({"stationary": True} if stationary else {}),
        **({"contrast_inversion": True} if contrast_inversion else {}),
    )
    pending: dict[tuple[str, ...], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    work = dict(
        prediction_calls_attempted=0,
        prediction_calls_completed=0,
        prediction_rows_attempted=0,
        prediction_rows_completed=0,
    )
    report: dict[str, Any] = dict(
        status="incomplete",
        reason=None,
        protocol=protocol,
        rows=rows,
        diagnostics=work,
        stream=None,
    )
    stream = None
    monitor = None
    rehearsal = None
    replay_budget = None
    learn_observer = None
    readout_progress: dict[str, Any] | None = {} if heldout else None
    retention = None
    with ProcessRssSampler() as memory:
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=12, max_wall_seconds=30, max_process_rss_bytes=256 * 1024**2
            ),
            monotonic,
        )
        budget.attach_memory(memory)
        try:
            if monitoring:
                from src.app.sensor_monitoring import SensorMonitoring

                monitor = SensorMonitoring(stationary=True) if stationary else SensorMonitoring()
                work["pending_feature_peak_bytes"] = 0
            fixture_options = {"stationary": True} if stationary else {}
            if contrast_inversion:
                fixture_options["contrast_inversion"] = True
            fixture = make_sensor_stream(**fixture_options)
            training = tuple(
                AttributionBatch(
                    "train",
                    "A" if stationary else ("A", "B", "A")[tick // 4],
                    tuple(f"sensor-{tick}-{row}" for row in range(4)),
                    batch.features,
                    batch.targets,
                )
                for tick, batch in enumerate(fixture)
            )
            final_batches: tuple[AttributionBatch, ...] = ()
            if heldout:
                from src.core.sensor_retention import make_sensor_heldout

                heldout_fixture = (
                    make_sensor_heldout(contrast_inversion=True)
                    if contrast_inversion
                    else make_sensor_heldout()
                )
                final_batches = tuple(
                    AttributionBatch(
                        "final_test",
                        context,
                        tuple(f"sensor-final-{context}-{i}" for i in range(32)),
                        batch.features,
                        batch.targets,
                    )
                    for context, batch in zip(("A", "B"), heldout_fixture, strict=True)
                )
            parameters = CoupledPredictiveCodingNetwork(16, (8,), seed=17).snapshot_parameters()
            learner: Any
            if scheduled:
                from src.adapters.scheduled_backprop_learner import ScheduledBackpropLearner

                learner = ScheduledBackpropLearner(
                    MatchedBackpropNetwork(parameters), learning_rate=0.1
                )
            else:
                learner = build_sensor_learner(method, parameters)
            source_before = learner.snapshot_state()
            if learnability:
                from src.adapters.sensor_learnability import SensorLearnability

                # The probe receives a separate initial template, never the live
                # benchmark learner or actor. Checkpoints are detached by stream.
                learn_observer = SensorLearnability(
                    build_sensor_learner(method, parameters),
                    deadline=started + 30,
                    memory=memory,
                    check_boundary=budget.before_final,
                )
            if replay:
                from src.app.stream_rehearsal import StreamRehearsal

                replay_budget = ToyBudgetSession(
                    ToyExecutionBudget(
                        max_replay_examples=12,
                        max_wall_seconds=30,
                        max_process_rss_bytes=256 * 1024**2,
                    ),
                    monotonic,
                )
                replay_budget.attach_memory(memory)
                incident_options: dict[str, Any] = (
                    {"incident_channel": "feature"} if incident_replay else {}
                )
                rehearsal = StreamRehearsal(
                    learner,
                    budget,
                    _replay_policy(),
                    replay_budget=replay_budget,
                    total_updates=12,
                    input_dim=16,
                    **incident_options,
                )

            def predict(observation: PrearrivalObservation, candidate: CandidateState[Any]) -> None:
                native = candidate.state
                if scheduled:
                    from src.adapters.scheduled_backprop_learner import ScheduledBackpropSnapshot

                    if (
                        type(native) is not ScheduledBackpropSnapshot
                        or native.arrived_batches != candidate.wake_updates
                    ):
                        raise ValueError(
                            "scheduled snapshot arrival count differs from wake receipts"
                        )
                    native = native.native
                if method == "bp":
                    if type(native) is not MatchedBackpropSnapshot:
                        raise ValueError("sensor demo requires the native matched BP snapshot")
                elif type(native) is not CoupledSnapshot:
                    raise ValueError("sensor PC/CPC requires the full native coupled snapshot")
                elif method == "pc":
                    if native.wake_state is not None:
                        raise ValueError("sensor PC snapshot cannot contain circadian wake state")
                elif (
                    type(native.wake_state) is not CoupledWakeState
                    or native.wake_state.wake_updates != candidate.wake_updates
                ):
                    raise ValueError("sensor CPC snapshot wake count differs from wake receipts")
                if work["prediction_calls_attempted"] >= 12:
                    raise ValueError("sensor diagnostic forward allowance exhausted")
                work["prediction_calls_attempted"] += 1
                work["prediction_rows_attempted"] += len(observation.features)
                probabilities = predict_coupled(native.parameters, observation.features)
                work["prediction_calls_completed"] += 1
                work["prediction_rows_completed"] += len(observation.features)
                if observation.row_ids in pending:
                    raise ValueError("sensor prediction IDs must be observed once")
                pending[observation.row_ids] = dict(
                    feature_tick=observation.tick,
                    prediction_state_updates=candidate.wake_updates,
                    actor_version=candidate.actor_version,
                    learner_version=candidate.learner_version,
                    probabilities=probabilities.copy(),
                )
                if scheduled:
                    pending[observation.row_ids]["prediction_optimizer_steps"] = (
                        candidate.state.optimizer_steps_completed
                    )
                if rehearsal is not None:
                    pending[observation.row_ids]["prediction_replay_updates"] = rehearsal.report[
                        "committed_updates"
                    ]
                if monitor is not None:
                    pending[observation.row_ids]["features"] = observation.features.copy()
                    work["pending_feature_peak_bytes"] = max(
                        work["pending_feature_peak_bytes"],
                        sum(row["features"].nbytes for row in pending.values()),
                    )

            def score(updates: int, state: Any, arrived: AttributionBatch | None) -> None:
                if updates == 0:
                    if arrived is not None:
                        raise ValueError("initial sensor callback cannot carry labels")
                    if learn_observer is not None:
                        learn_observer.observe(updates, state, arrived)
                    return
                if arrived is None or arrived.row_ids not in pending:
                    raise ValueError("arrived sensor labels require their earlier prediction")
                prediction = pending.pop(arrived.row_ids)
                probabilities = prediction.pop("probabilities")
                features = prediction.pop("features", None)
                # Why this: scoring never recomputes a prediction after seeing its label.
                rows.append(
                    dict(
                        **prediction,
                        label_tick=prediction["feature_tick"] + 2,
                        arrival_update=updates,
                        row_ids=list(arrived.row_ids),
                        probabilities=probabilities.tolist(),
                        targets=arrived.targets.tolist(),
                        scores=probability_scores(probabilities, arrived.targets),
                    )
                )
                # Preserve the paid forecast/score even if a later diagnostic fails.
                if monitor is not None:
                    assert features is not None
                    monitoring_row = monitor.observe(
                        prediction["feature_tick"],
                        features,
                        probabilities.copy(),
                        arrived.targets.copy(),
                    )
                    if incident_replay:
                        from src.core.shift_incidents import ShiftIncidentDecision

                        assert rehearsal is not None
                        incident = monitoring_row["incident"]
                        rehearsal.observe_incident(
                            ShiftIncidentDecision(
                                incident["batch_index"],
                                tuple(incident["states"]),
                                tuple(incident["onset_batches"]),
                            ),
                            completed_updates=updates,
                        )
                if learn_observer is not None and updates in (4, 8, 12):
                    learn_observer.observe(updates, state, arrived)

            stream = AttributionStream(
                learner,
                training,
                final_batches,
                checkpoint_updates=tuple(range(13)),
                label_delay=2,
                budget=budget,
                observe_features=predict,
                observe_checkpoint=score,
                training_only=not heldout,
                rehearsal=rehearsal,
            )
            actor_before = stream._runtime.actor.snapshot_state()
            stream.run_until(12)
            if heldout:
                # Final data cannot be admitted while requested probes are incomplete.
                if learn_observer is not None and learn_observer.report()["status"] != "complete":
                    raise ValueError("sensor probes incomplete before final evaluation")
                report["stream"] = stream.finish(score_progress=readout_progress)
                from src.core.sensor_retention import summarize_sensor_retention

                retention = summarize_sensor_retention(tuple(report["stream"]["checkpoints"]))
            else:
                report["stream"] = stream.finish()
            if learn_observer is not None and learn_observer.report()["status"] != "complete":
                raise ValueError("sensor learnability did not complete its fixed checkpoints")
            if rehearsal is not None:
                replay_work = rehearsal.report
                if incident_replay:
                    rehearsal.check_incident_progress(12)
                    if any(
                        event["status"] not in {"complete", "skipped"}
                        for event in replay_work["events"]
                    ):
                        raise ValueError("incident replay has unfinished boundary evidence")
                if (
                    (
                        not incident_replay
                        and replay_work["charged_exposures"] != (6 if stationary else 10)
                    )
                    or replay_work["committed_updates"] != replay_work["charged_exposures"]
                    or replay_work["wake_updates_observed"] != 12
                ):
                    raise ValueError("recent replay did not complete its fixed arrived-row work")
            if scheduled:
                schedule = _scheduled_report(stream)
                if (
                    schedule["arrived_batches"] != 12
                    or schedule["optimizer_steps_completed"] != 3
                    or schedule["training_rows_completed"] != 48
                    or schedule["buffered_batches"] != 0
                ):
                    raise ValueError("scheduled stream did not complete its fixed optimizer work")
            if pending or len(rows) != 12:
                raise ValueError("sensor stream has unscored predictions or missing arrived labels")
            report["source_unchanged"] = _same_value(source_before, learner.snapshot_state())
            report["actor_unchanged"] = _same_value(
                actor_before, stream._runtime.actor.snapshot_state()
            )
            if not report["source_unchanged"] or not report["actor_unchanged"]:
                raise ValueError("sensor demo source or serving actor changed")
            report["input_array_bytes"] = sum(b.features.nbytes + b.targets.nbytes for b in fixture)
            report["parameter_count"] = 145
            budget.before_final()
            report["status"] = "complete"
        except Exception as error:
            report.update(status="incomplete", reason=f"{type(error).__name__}: {error}")
        finally:
            report["charged_training_updates"] = budget.updates_completed
            report["pending_prediction_batches"] = len(pending)
            if stream is not None:
                report["prearrival_observation"] = stream.prearrival_report
                if scheduled:
                    report["scheduled_training"] = _scheduled_report(stream)
            if monitoring:
                report["sensor_monitoring"] = None if monitor is None else monitor.report()
                if incident_replay and report["sensor_monitoring"] is not None:
                    report["sensor_monitoring"]["configuration"]["feedback"] = True
                    report["sensor_monitoring"]["configuration"]["feedback_scope"] = (
                        "arrived feature incident to future replay boundary only"
                    )
            if learnability:
                report["learnability"] = None if learn_observer is None else learn_observer.report()
            if heldout:
                from copy import deepcopy

                report["heldout_evaluation"] = dict(
                    progress=deepcopy(readout_progress)
                    if readout_progress
                    else {
                        "status": "not_run",
                        "reason": "training or probes did not complete",
                        "prediction_calls_attempted": 0,
                        "prediction_calls_completed": 0,
                        "prediction_rows_attempted": 0,
                        "prediction_rows_completed": 0,
                        "checkpoints": [],
                    },
                    retention=retention,
                    heldout_array_bytes=8704,
                    training_receipts=[]
                    if stream is None
                    else [asdict(receipt) for receipt in stream._runtime.applied_updates],
                )
            if replay or method != "bp" or learnability or heldout:
                # Retain already-paid targetless forecasts on coupled failure too.
                report["pending_forecasts"] = [
                    dict(
                        **{
                            key: value
                            for key, value in prediction.items()
                            if key not in ("features", "probabilities")
                        },
                        row_ids=list(row_ids),
                        probabilities=prediction["probabilities"].tolist(),
                    )
                    for row_ids, prediction in pending.items()
                ]
            if replay:
                report["rehearsal"] = None if rehearsal is None else rehearsal.report
                report["replay_budget_charged_exposures"] = (
                    0 if replay_budget is None else replay_budget.replay_examples_completed
                )
            memory.sample()
            report["process_rss"] = asdict(memory.snapshot())
            report["elapsed_seconds"] = monotonic() - started
            if (
                report["elapsed_seconds"] >= 30
                or report["process_rss"]["peak_bytes"] > 256 * 1024**2
            ):
                report.update(status="incomplete", reason="sensor final resource cap exceeded")
    return report
