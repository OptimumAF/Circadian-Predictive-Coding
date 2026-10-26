"""Fixed arrived-only shift observations on the existing matched stream runner.

Owns native binding, fixed synthetic data/settings and resource reporting. The
observer never receives benchmark final data. Explicit incident replay admits
bounded train-only work at a later boundary; default signals change no policy.
"""

from dataclasses import asdict, fields
import pickle
from time import monotonic
from typing import Any

from src.adapters import learnability_pilot as probe_pilot
from src.adapters.coupled_attribution_pilot import build_learner, build_stream
from src.app.arrived_shift_observer import ArrivedShiftObserver
from src.app.coupled_attribution import AttributionStream, run_attribution_stream
from src.app.coupled_attribution_summary import METHODS
from src.app.stream_rehearsal import RecentStreamReplayPolicy, StreamRehearsal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_energy import Array, predict_coupled
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork, CoupledSnapshot
from src.core.local_pilot_budget import PilotRequest, PilotResources, validate_local_pilot_request
from src.core.matched_backprop import MatchedBackpropSnapshot
from src.core.shift_calibration import ShiftCalibration
from src.core.shift_incidents import ShiftIncidentDecision
from src.core.shift_signals import ShiftPolicy
from src.shared.process_memory import ProcessRssSampler

PROTOCOL_ID = "arrived_only_shift_observations_v1"
RESOURCES = PilotResources(training_updates=144, environment_steps=0, replay_bytes=0)
INCIDENT_RESOURCES = PilotResources(training_updates=176, environment_steps=0, replay_bytes=192)
LEARNABILITY_RESOURCES = PilotResources(training_updates=432, environment_steps=0, replay_bytes=192)
PROBE_CHECKPOINTS = (0, 6, 12, 18)
CHECKPOINTS = tuple(range(19))
CHANNELS = ("uncertainty", "feature", "residual")


def describe_protocol(
    *,
    calibrated: bool = False,
    incidents: bool = False,
    incident_replay: bool = False,
    learnability: bool = False,
) -> dict[str, Any]:
    if type(calibrated) is not bool:
        raise ValueError("calibrated mode must be boolean")
    if type(incidents) is not bool:
        raise ValueError("incidents mode must be boolean")
    if type(incident_replay) is not bool:
        raise ValueError("incident_replay mode must be boolean")
    if incident_replay and not (calibrated and incidents):
        raise ValueError("incident replay requires calibrated incidents")
    if type(learnability) is not bool:
        raise ValueError("learnability mode must be boolean")
    if learnability and not incident_replay:
        raise ValueError("learnability requires calibrated incidents and incident replay")
    report: dict[str, Any] = dict(
        protocol_id=PROTOCOL_ID,
        classification="exploratory_fixed_data_signal_engineering_smoke",
        methods=list(METHODS),
        streams=["stationary", "recurring"],
        seeds=[23],
        training_rng=7301,
        final_rng=9101,
        hidden_dims=[8, 8],
        learning_rate=0.03,
        inference_steps=4,
        inference_learning_rate=0.2,
        precisions=[1.0, 1.0],
        active_cpc_config=asdict(CoupledCircadianConfig()),
        neutral_cpc_config=asdict(CoupledCircadianConfig.matched_pc_control()),
        training_batches=18,
        batch_rows=8,
        final_rows_per_context=64,
        label_delay=2,
        checkpoint_updates=list(CHECKPOINTS),
        policy=asdict(ShiftPolicy()),
        channels=list(CHANNELS),
        uncertainty="mean binary entropy / log(2); target-free prediction",
        feature="raw input column means; mean absolute window/reference difference",
        residual="mean absolute probability error; labels already arrived",
        reference="first3 batches frozen; window2 post-reference batches; batch-weighted means",
        persistence="strict threshold exceedance for2 consecutive windows; reset on nonexceedance",
        prediction_timing="computed AFTER update k using copied parameters from k-1; not a live before-update trigger",
        score_release="all benchmark final scoring after every observer; no feedback",
        comparison="recurring states0/6/12/18 and receipts must match retained original pilot",
        contextual_annotations="added after stream completion; not detector inputs",
        detection_delay="first alert update minus first changed-phase update (7 or13); None if missed",
        limitations="one seed/fixed data; overlapping windows not independent evidence; unchanged input distribution can hide concept-only shifts; no deployable efficacy/advantage claim",
        planned_work=dict(
            benchmark_updates=144,
            training_row_exposures=1152,
            observation_predictions=144,
            final_score_forwards=304,
            predict_coupled_calls=448,
            detector_steps=144,
            probe_updates=0,
            svd_calls=0,
            replay_examples=0,
        ),
        resources=asdict(RESOURCES),
        stop_policy="one invocation; preserve all failed/unreached rows; no retry or tuning",
    )
    if calibrated:
        report.update(
            protocol_id="arrived_only_prefix_calibration_v1",
            classification="outcome_motivated_exploratory_prefix_calibration",
            calibration=asdict(ShiftCalibration()),
            calibration_population="evolving copied k-1 parameters; first6 declared stationary arrived-training batches; no final data",
            calibration_rule="max(0, observed calibration window scores) + margin; capped only at channel domain upper(1,2,1)",
            reachability="threshold strictly below algebraic score ceiling; unreachable channels reported without retuning",
            calibration_freeze="fit at update6 before monitoring7..18; no further fit or policy feedback",
            calibration_identity="caller-declared method/seed run plus prediction_state_updates and arrived row IDs; not parameter attestation",
            comparison="all19 benchmark checkpoint score reports and receipts match retained uncalibrated v1",
            limitations=report["limitations"]
            + "; calibration has only2 full windows; no false-positive calibration guarantee; prefix stationarity assumed, not detected",
        )
        report["planned_work"] = dict(
            report["planned_work"],
            detector_steps=192,
            calibration_fits=8,
            calibration_batches=48,
            calibration_rows_in_training_total=384,
            replayed_summary_observations=48,
            monitoring_observations=96,
        )
    if incidents:
        report.update(
            protocol_id=report["protocol_id"] + "_incidents_v1",
            incidents="per-channel inactive/onset/continued/cleared; onset index links each incident; no stage-boundary reset",
            incident_outcomes="new onset only for B/return detection; original signal_outcomes retain alarm-level annotations",
        )
        report["planned_work"] = dict(
            report["planned_work"], incident_steps=144, incident_channel_transitions=432
        )
    if incident_replay:
        report.update(
            protocol_id=report["protocol_id"] + "_feature_replay_v1",
            classification="exploratory_fixed_feature_incident_feedback",
            resources=asdict(INCIDENT_RESOURCES),
            calibration_freeze="fit at update6 before monitoring7..18; no further fit",
            score_release="per-case final scoring after all18 wakes and completed observers",
            comparison="retained no-feedback cases; extra replay work is not equal compute",
            replay=dict(
                channel="feature",
                every_updates=4,
                rows_per_event=4,
                retention_examples=8,
                retention_numeric_bytes=192,
                max_charged_rows_per_case=4,
                admission="prior observed active onset only; clear expires; consume before work",
                selection="existing recent FIFO, arrived training rows only",
            ),
            policy_feedback=True,
        )
        report["planned_work"] = dict(
            report["planned_work"],
            replay_examples=32,
            native_training_updates_upper=176,
            training_row_exposures=1184,
        )
    if learnability:
        report.update(
            protocol_id=report["protocol_id"] + "_isolated_c_probe_v1",
            resources=asdict(LEARNABILITY_RESOURCES),
            learnability=dict(
                streams=["recurring"],
                checkpoint_updates=list(PROBE_CHECKPOINTS),
                probe_checkpoint_updates=[0, 1, 2, 4, 8, 16],
                task="existing C: x0*x1 >= 0; two float64 inputs",
                training_rng=17031,
                evaluation_rng=17033,
                reference_rng=17035,
                training_batches=16,
                training_rows_per_batch=8,
                evaluation_rows=32,
                reference_rows=32,
                reserved_data_array_bytes=3840,
                reference_array_bytes=512,
                source="existing learnability_pilot.create_sixteen_probe_observer",
                gain="C BCE at0 minus C BCE after exactly16 probe updates; retain entire curve",
                isolation="independent template/disposable full-state forks; no benchmark/final feedback",
                budgets="four distinct original16-update sessions at recurring-case setup; shared clock/RSS/global deadline",
                diagnostics="existing tanh activity/dead/saturation/centered SVD effective rank; no causal plasticity claim",
            ),
            score_release="per-case final scoring after all18 wakes, arrived observers and required isolated probes",
        )
        report["planned_work"] = dict(
            report["planned_work"],
            probe_updates=256,
            probe_training_row_exposures=2048,
            probe_clones=16,
            probe_score_forward_calls=96,
            probe_score_forward_rows=3072,
            diagnostic_feedforward_passes=16,
            diagnostic_svd_calls=32,
            svd_calls=32,
            native_training_updates_upper=432,
            training_row_exposures=3232,
            predict_coupled_calls=544,
        )
    return report


def _predict_snapshot(state: Any, features: Array) -> Array:
    if type(state) not in (CoupledSnapshot, MatchedBackpropSnapshot):
        raise ValueError("shift prediction requires an exact matched native snapshot")
    if type(state.format_version) is not int or state.format_version != 1:
        raise ValueError("shift prediction requires snapshot format1")
    # Prediction uses only parameters. This does not attest complete native state.
    return predict_coupled(state.parameters, features)


def _snapshot_array_bytes(state: Any) -> int:
    parameters = state.parameters
    arrays = (
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    )
    total = sum(a.nbytes for a in arrays)
    if type(state) is CoupledSnapshot and state.wake_state is not None:
        total += state.wake_state.chemical_state.nbytes + state.wake_state.plasticity_state.nbytes
    return total


def _signal_outcomes(observations, stream):
    if any(row["status"] != "complete" for row in observations["rows"]):
        return None
    result = {}
    for index, channel in enumerate(CHANNELS):
        alerts = [
            row["updates"] for row in observations["rows"] if row["decision"]["alarms"][index]
        ]
        b = next((n for n in alerts if 7 <= n <= 12), None) if stream == "recurring" else None
        returning = (
            next((n for n in alerts if 13 <= n <= 18), None) if stream == "recurring" else None
        )
        result[channel] = dict(
            alert_updates=alerts,
            stationary_false_alerts=alerts if stream == "stationary" else None,
            first_phase_false_alerts=[n for n in alerts if n <= 6],
            first_B_alert_update=b,
            B_delay_updates=None if b is None else b - 7,
            first_return_alert_update=returning,
            return_delay_updates=None if returning is None else returning - 13,
        )
    return result


def _learning_outcomes(result):
    points = {row["updates"]: row["scores"] for row in result["checkpoints"]}
    return dict(
        new_B_gain=points[0]["B"]["bce"] - points[12]["B"]["bce"],
        retention_loss=points[12]["A"]["bce"] - points[6]["A"]["bce"],
        return_gain=points[12]["A"]["bce"] - points[18]["A"]["bce"],
    )


def _incident_outcomes(observations, stream):
    """Annotate only new onsets after observation; contexts never enter the tracker."""
    if any(row["status"] != "complete" for row in observations["rows"]):
        return None
    result = {}
    for index, channel in enumerate(CHANNELS):
        rows = observations["rows"]
        transitions = {
            state: [row["updates"] for row in rows if row["incident"]["states"][index] == state]
            for state in ("onset", "continued", "cleared")
        }
        onsets = transitions["onset"]
        b = next((n for n in onsets if 7 <= n <= 12), None) if stream == "recurring" else None
        returning = (
            next((n for n in onsets if 13 <= n <= 18), None) if stream == "recurring" else None
        )
        result[channel] = dict(
            onset_updates=onsets,
            continued_updates=transitions["continued"],
            cleared_updates=transitions["cleared"],
            stationary_false_incidents=onsets if stream == "stationary" else None,
            first_phase_false_incidents=[n for n in onsets if n <= 6],
            first_B_onset_update=b,
            B_onset_delay_updates=None if b is None else b - 7,
            first_return_onset_update=returning,
            return_onset_delay_updates=None if returning is None else returning - 13,
        )
    return result


def _observer(method, stream, *, calibrated, incidents=False):
    if not calibrated:
        return ArrivedShiftObserver(_predict_snapshot, max_updates=18, incidents=incidents)
    return ArrivedShiftObserver(
        _predict_snapshot,
        max_updates=18,
        calibration=ShiftCalibration(),
        calibration_model_id=f"{method}:{stream}:seed23:evolving-previous-parameters",
        incidents=incidents,
    )


def _new_row(method, stream, *, calibrated=False, incidents=False, learnability=False):
    row = dict(
        method=method,
        stream=stream,
        seed=23,
        status="not_run",
        reason="not reached",
        observations=_observer(method, stream, calibrated=calibrated, incidents=incidents).report(),
        outcomes=None,
        signal_outcomes=None,
    )
    if incidents:
        row["incident_outcomes"] = None
    if learnability:
        row["learnability"] = (
            probe_pilot.empty_learnability_observations() if stream == "recurring" else None
        )
    return row


def _run_incident_stream(
    learner,
    training,
    final,
    budget,
    helper,
    observer,
    provenance,
    progress,
    *,
    probe_observer=None,
    probe_observations=None,
):
    """Route completed arrived decisions; preserve the previous snapshot's cost.

    Why: rehearsal precedes the callback, but prediction uses the snapshot saved
    by its preceding callback. The current replay count would mislabel row12.
    """
    previous_replay = 0

    def observe(updates, state, arrived):
        nonlocal previous_replay
        if updates:
            provenance[updates] = previous_replay
        observer(updates, state, arrived)
        if not updates:
            if probe_observer is not None:
                if not observer.report()["primed"]:
                    raise RuntimeError("probe requires a primed arrived observer")
                _observe_probe(probe_observer, probe_observations, updates, state, arrived)
            return
        report = observer.report()
        observation = report["rows"][updates - 1]
        if report["stopped"] or observation["status"] != "complete":
            raise RuntimeError("incident feedback requires a completed arrived observation")
        decision = observation["incident"]
        helper.observe_incident(
            ShiftIncidentDecision(
                decision["batch_index"], tuple(decision["states"]), tuple(decision["onset_batches"])
            ),
            completed_updates=updates,
        )
        previous_replay = helper.report["committed_updates"]
        if probe_observer is not None and updates in PROBE_CHECKPOINTS:
            _observe_probe(probe_observer, probe_observations, updates, state, arrived)

    stream = AttributionStream(
        learner,
        training,
        final,
        checkpoint_updates=CHECKPOINTS,
        label_delay=2,
        budget=budget,
        observe_checkpoint=observe,
        rehearsal=helper,
    )
    stream.run_until(len(training))
    report = observer.report()
    if not report["primed"] or any(r["status"] != "complete" for r in report["rows"]):
        raise RuntimeError("final scoring requires every completed arrived observation")
    if probe_observer is not None:
        for point in PROBE_CHECKPOINTS:
            _require_probe(probe_observations, point)
    return stream.finish(score_progress=progress)


def _require_probe(observations, updates):
    row = observations[PROBE_CHECKPOINTS.index(updates)]
    report = row.get("probe")
    if (
        row["status"] != "complete"
        or type(report) is not dict
        or report.get("status") != "complete"
        or report.get("completed_updates") != 16
    ):
        raise RuntimeError("benchmark requires every selected isolated probe complete")


def _observe_probe(callback, observations, updates, state, arrived):
    try:
        callback(updates, state, arrived)
        _require_probe(observations, updates)
    except Exception as error:
        observations[PROBE_CHECKPOINTS.index(updates)].update(
            status="failed", reason=f"{type(error).__name__}: {error}"
        )
        raise


def _prepare_probe_budgets(deadline, sampler):
    sessions = []
    for _ in PROBE_CHECKPOINTS:
        session = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=16,
                max_wall_seconds=max(0.0, deadline - monotonic()),
                max_process_rss_bytes=LEARNABILITY_RESOURCES.process_memory_bytes,
            ),
            monotonic,
        )
        session.attach_memory(sampler)
        sessions.append(session)
    return tuple(sessions)


def _record_probe_costs(row, sessions):
    slots = row["learnability"] or []
    reports = [slot["probe"] for slot in slots if type(slot.get("probe")) is dict]
    unknown = any(
        slot["status"] == "failed" and type(slot.get("probe")) is not dict for slot in slots
    )
    row["probe_work_unknown"] = unknown
    row["probe_attempted_updates"] = (
        None if unknown else sum(p["attempted_updates"] for p in reports)
    )
    row["probe_reported_completed_updates"] = sum(p["completed_updates"] for p in reports)
    row["probe_completed_updates"] = sum(session.updates_completed for session in sessions)
    row["probe_training_row_exposures"] = (
        None if unknown else sum(p["attempted_training_examples"] for p in reports)
    )
    row["probe_completed_training_row_exposures"] = 8 * row["probe_completed_updates"]
    row["probe_budget_completed_updates"] = [s.updates_completed for s in sessions]
    for phase in ("attempted", "completed"):
        row[f"probe_score_forward_calls_{phase}"] = (
            None if unknown else sum(p[f"score_forward_calls_{phase}"] for p in reports)
        )
    health_unknown = any(
        slot["status"] == "failed" and "feature_health" not in slot for slot in slots
    )
    health_completed = sum("feature_health" in slot for slot in slots)
    row["health_work_unknown"] = health_unknown
    row["health_forward_calls_completed"] = None if health_unknown else health_completed
    row["health_svd_calls_completed"] = None if health_unknown else 2 * health_completed
    row["probe_reserved_data_bytes_by_checkpoint"] = [
        None if slot.get("probe") is None else slot["probe"].get("reserved_data_bytes")
        for slot in slots
    ]
    row["probe_seconds"] = sum(slot.get("probe_seconds", 0.0) for slot in slots)
    row["learnability_diagnostic_seconds"] = sum(
        slot.get("diagnostic_seconds", 0.0) for slot in slots
    )
    row["learnability_cost_scope"] = (
        "completed fixed-eight-row probe updates from original session counters; "
        "attempts and scores from returned reports; failed delegate without report "
        "has unknown work; health/SVD partial failure counts unavailable; all work included in case time/RSS"
    )


def _probe_comparison(slots):
    """Compare semantics after the caller validates differing native storage."""
    if slots is None:
        return None
    results = []
    for slot in slots:
        copied = {
            k: v
            for k, v in slot.items()
            if k not in {"diagnostic_seconds", "probe_seconds", "checkpoint_snapshot_array_bytes"}
        }
        if type(copied.get("probe")) is dict:
            copied["probe"] = {k: v for k, v in copied["probe"].items() if k != "elapsed_seconds"}
        results.append(copied)
    return results


def _learnability_parity(pc, neutral):
    if pc is None or neutral is None:
        return pc is neutral
    # Why: 105 float64 parameters use840B. Neutral CPC additionally owns
    # eight chemistry and eight plasticity values (128B), even at identity gates.
    for slots, expected in ((pc, 840), (neutral, 968)):
        if len(slots) != 4 or any(
            type(slot.get("checkpoint_snapshot_array_bytes")) is not int
            or slot["checkpoint_snapshot_array_bytes"] != expected
            for slot in slots
        ):
            return False
    return _probe_comparison(pc) == _probe_comparison(neutral)


def _run_one(
    row,
    initial,
    deadline,
    sampler,
    *,
    calibrated=False,
    incidents=False,
    incident_replay=False,
    learnability=False,
):
    started = monotonic()
    observer = _observer(row["method"], row["stream"], calibrated=calibrated, incidents=incidents)
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=18,
            max_wall_seconds=max(0.0, deadline - started),
            max_process_rss_bytes=RESOURCES.process_memory_bytes,
        ),
        monotonic,
    )
    helper = None
    replay_budget = None
    provenance: dict[int, int] = {}
    progress: dict[str, Any] = {}
    probe_budgets: tuple[ToyBudgetSession, ...] = ()
    probe_observer = None
    if learnability:
        row["learnability"] = (
            probe_pilot.empty_learnability_observations() if row["stream"] == "recurring" else None
        )
    try:
        budget.attach_memory(sampler)
        budget.before_final()
        if incident_replay:
            # Both sessions are born before learner work and share the deadline,
            # original clock and actual process sampler. No allowance is renewed.
            replay_budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=4,
                    max_replay_examples=4,
                    max_wall_seconds=max(0.0, deadline - monotonic()),
                    max_process_rss_bytes=INCIDENT_RESOURCES.process_memory_bytes,
                ),
                monotonic,
            )
            replay_budget.attach_memory(sampler)
        if learnability and row["stream"] == "recurring":
            probe_budgets = _prepare_probe_budgets(deadline, sampler)
        training, final = build_stream(row["stream"])
        learner = build_learner(row["method"], initial)
        if probe_budgets:
            template = build_learner(row["method"], initial)
            if template is learner:
                raise ValueError("probe template must be independent of benchmark learner")
            probe_observer = probe_pilot.create_sixteen_probe_observer(
                template, row["learnability"], deadline, sampler, probe_budgets=probe_budgets
            )
        source_state = learner.snapshot_state()
        before = pickle.dumps(source_state)
        row["previous_snapshot_array_bytes"] = _snapshot_array_bytes(source_state)
        if incident_replay:
            assert replay_budget is not None
            helper = StreamRehearsal(
                learner=learner,
                budget=budget,
                replay_budget=replay_budget,
                policy=RecentStreamReplayPolicy(4, 4, ReplayRetentionBudget(8, 192)),
                total_updates=18,
                input_dim=2,
                incident_channel="feature",
            )
            probe_options: dict[str, Any] = {}
            if probe_observer is not None:
                probe_options.update(
                    probe_observer=probe_observer, probe_observations=row["learnability"]
                )
            result = _run_incident_stream(
                learner,
                training,
                final,
                budget,
                helper,
                observer,
                provenance,
                progress,
                **probe_options,
            )
        else:
            result = run_attribution_stream(
                learner,
                training,
                final,
                checkpoint_updates=CHECKPOINTS,
                label_delay=2,
                budget=budget,
                observe_checkpoint=observer,
            )
        if pickle.dumps(learner.snapshot_state()) != before:
            raise RuntimeError("diagnostics or stream changed the source learner")
        row.update(
            status="complete",
            reason=None,
            stream_result=result,
            source_learner_unchanged=True,
            outcomes=_learning_outcomes(result),
        )
    except Exception as error:
        row.update(status="failed", reason=f"{type(error).__name__}: {error}", outcomes=None)
    row["observations"] = observer.report()
    if incident_replay:
        for observation in row["observations"]["rows"]:
            observation["prediction_replay_updates"] = provenance.get(observation["updates"])
        row["rehearsal"] = None if helper is None else helper.report
        row["replay_budget_charged_exposures"] = (
            0 if replay_budget is None else replay_budget.replay_examples_completed
        )
        row["replay_policy_feedback"] = True
        row["final_score_progress"] = progress
    row["signal_outcomes"] = _signal_outcomes(row["observations"], row["stream"])
    if incidents:
        row["incident_outcomes"] = _incident_outcomes(row["observations"], row["stream"])
    row["benchmark_completed_updates"] = budget.updates_completed
    row["elapsed_seconds"] = monotonic() - started
    row["diagnostic_seconds"] = row["observations"]["total_seconds"]
    row["benchmark_and_orchestration_seconds"] = row["elapsed_seconds"] - row["diagnostic_seconds"]
    if learnability:
        _record_probe_costs(row, probe_budgets)
        row["benchmark_and_orchestration_seconds"] -= (
            row["probe_seconds"] + row["learnability_diagnostic_seconds"]
        )


def run_pilot(
    *,
    calibrated: bool = False,
    incidents: bool = False,
    incident_replay: bool = False,
    learnability: bool = False,
) -> dict[str, Any]:
    if type(learnability) is not bool:
        raise ValueError("learnability mode must be boolean")
    probe_options: dict[str, Any] = {"learnability": True} if learnability else {}
    protocol = describe_protocol(
        calibrated=calibrated, incidents=incidents, incident_replay=incident_replay, **probe_options
    )
    resources = (
        LEARNABILITY_RESOURCES
        if learnability
        else INCIDENT_RESOURCES
        if incident_replay
        else RESOURCES
    )
    validate_local_pilot_request(PilotRequest(resources))
    started = monotonic()
    rows: list[dict[str, Any]] = [
        _new_row(m, s, calibrated=calibrated, incidents=incidents, **probe_options)
        for s in ("stationary", "recurring")
        for m in METHODS
    ]
    sampler = None
    errors = []
    initial_data = None
    try:
        sampler = ProcessRssSampler()
        with sampler:
            initial = CoupledPredictiveCodingNetwork(2, (8, 8), 23).snapshot_parameters()
            initial_data = {
                field.name: (
                    [a.tolist() for a in value] if type(value) is tuple else value.tolist()
                )
                for field in fields(initial)
                for value in (getattr(initial, field.name),)
            }
            for row in rows:
                if monotonic() - started >= resources.wall_seconds:
                    break
                options: dict[str, Any] = dict(calibrated=calibrated, incidents=incidents)
                if incident_replay:
                    options["incident_replay"] = True
                options.update(probe_options)
                _run_one(
                    row,
                    initial,
                    started + resources.wall_seconds,
                    sampler,
                    **options,
                )
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        if sampler is None:
            raise RuntimeError("RSS sampler setup unavailable")
        memory = asdict(sampler.snapshot())
    except Exception as error:
        memory = None
        errors.append(f"{type(error).__name__}: {error}")
    elapsed = monotonic() - started
    updates = sum(r.get("benchmark_completed_updates", 0) for r in rows)
    observed = sum(o["status"] == "complete" for r in rows for o in r["observations"]["rows"])
    parity = {}
    for stream in ("stationary", "recurring"):
        pc, neutral = [
            r for r in rows if r["stream"] == stream and r["method"] in ("pc", "neutral_cpc")
        ]
        parity[stream] = (
            (
                pc["stream_result"] == neutral["stream_result"]
                and [(o["metrics"], o["decision"]) for o in pc["observations"]["rows"]]
                == [(o["metrics"], o["decision"]) for o in neutral["observations"]["rows"]]
                and (
                    not incidents
                    or [o["incident"] for o in pc["observations"]["rows"]]
                    == [o["incident"] for o in neutral["observations"]["rows"]]
                )
                and (not incident_replay or pc["rehearsal"] == neutral["rehearsal"])
                and (
                    not learnability
                    or _learnability_parity(pc["learnability"], neutral["learnability"])
                )
            )
            if pc["status"] == neutral["status"] == "complete"
            else None
        )
    complete = (
        not errors
        and all(r["status"] == "complete" for r in rows)
        and updates == observed == 144
        and elapsed <= resources.wall_seconds
        and memory is not None
        and memory["peak_bytes"] <= resources.process_memory_bytes
        and all(v is True for v in parity.values())
    )
    report = dict(
        protocol=protocol,
        status="complete" if complete else "incomplete",
        initial_parameters=initial_data,
        runs=rows,
        global_errors=errors,
        total_seconds=elapsed,
        observed_process_rss=memory,
        benchmark_completed_updates=updates,
        training_row_exposures=8 * updates,
        observation_predictions_attempted=sum(
            r["observations"]["prediction_calls_attempted"] for r in rows
        ),
        observation_predictions_completed=sum(
            r["observations"]["prediction_calls_completed"] for r in rows
        ),
        completed_observations=observed,
        pc_neutral_exact=parity,
        policy_feedback=incident_replay,
        scientific_advantage_verified=False,
        full_R27_complete=False,
    )
    if incident_replay:
        replay = sum(r.get("replay_budget_charged_exposures", 0) for r in rows)
        report["native_training_updates"] = updates + replay
        report["training_row_exposures"] += replay
        report["native_training_updates_committed"] = updates + sum(
            r["rehearsal"]["committed_updates"] for r in rows if r.get("rehearsal") is not None
        )
        report["training_costs_complete"] = all(r["status"] == "complete" for r in rows)
        report["training_cost_scope"] = (
            "completed wakes plus permanently charged one-row replay attempts; "
            "a failed wake may have unobserved native work"
        )
    if learnability:
        unknown = any(r.get("probe_work_unknown", False) for r in rows)
        report["pc_neutral_snapshot_storage"] = dict(
            pc_bytes=840,
            neutral_cpc_bytes=968,
            neutral_auxiliary_bytes=128,
            scope="exact per-checkpoint sizes validated separately from shared learning/probe semantics",
        )
        attempted = sum(r.get("probe_attempted_updates") or 0 for r in rows)
        completed = sum(r.get("probe_completed_updates", 0) for r in rows)
        attempted_rows = sum(r.get("probe_training_row_exposures") or 0 for r in rows)
        report["probe_attempted_updates"] = None if unknown else attempted
        report["probe_completed_updates"] = completed
        report["probe_training_row_exposures"] = None if unknown else attempted_rows
        report["probe_completed_training_row_exposures"] = sum(
            r.get("probe_completed_training_row_exposures", 0) for r in rows
        )
        report["native_training_updates"] = (
            None if unknown else report["native_training_updates"] + attempted
        )
        report["native_training_updates_committed"] += completed
        report["training_row_exposures"] = (
            None if unknown else report["training_row_exposures"] + attempted_rows
        )
        report["probe_work_unknown"] = unknown
        for phase in ("attempted", "completed"):
            report[f"probe_score_forward_calls_{phase}"] = (
                None
                if unknown
                else sum(r.get(f"probe_score_forward_calls_{phase}", 0) for r in rows)
            )
        health_unknown = any(r.get("health_work_unknown", False) for r in rows)
        report["health_work_unknown"] = health_unknown
        for kind in ("forward", "svd"):
            report[f"health_{kind}_calls_completed"] = (
                None
                if health_unknown
                else sum(r.get(f"health_{kind}_calls_completed", 0) for r in rows)
            )
        report["probe_seconds"] = sum(r.get("probe_seconds", 0.0) for r in rows)
        report["learnability_diagnostic_seconds"] = sum(
            r.get("learnability_diagnostic_seconds", 0.0) for r in rows
        )
        report["training_cost_scope"] += (
            "; includes isolated probe attempts separately from completed probe updates; "
            "failed probe delegate without returned report has unknown attempts"
        )
        report["full_R21_complete"] = False
    return report
