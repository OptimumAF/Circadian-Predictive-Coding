"""Fixed shared-rehearsal demonstrations on the existing four-arm stream.

Owns native binding and bounded resource reporting. Application code owns replay
and scoring isolation. Optional disposable probes reuse the existing reserved
sixteen-step observer. No metric feedback, tuning or promotion occurs here.
"""

from dataclasses import asdict, fields
from copy import deepcopy
import pickle
from time import monotonic
from typing import Any

from src.adapters import learnability_pilot as learnability
from src.adapters.coupled_attribution_pilot import build_learner, build_stream
from src.app.coupled_attribution import run_attribution_stream
from src.app.coupled_attribution_summary import METHODS, PLANNED_SEEDS
from src.app.stream_rehearsal import RecentStreamReplayPolicy, StreamReplayPolicy, StreamRehearsal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.local_pilot_budget import PilotRequest, PilotResources, validate_local_pilot_request
from src.core.shared_replay_schedule import ReplayAllocation
from src.shared.process_memory import ProcessRssSampler

PROTOCOL_ID = "coherent_stream_shared_rehearsal_v1"
RESOURCES = PilotResources(training_updates=120, environment_steps=0, replay_bytes=384)
LEARNABILITY_PROTOCOL_ID = "coherent_rehearsal_reserved_sixteen_v1"
LEARNABILITY_RESOURCES = PilotResources(training_updates=376, environment_steps=0, replay_bytes=384)
RECENT_PROTOCOL_ID = "coherent_stream_recent_rehearsal_v1"
RECENT_LEARNABILITY_PROTOCOL_ID = "coherent_recent_rehearsal_reserved_sixteen_v1"
CHECKPOINTS = (0, 6, 12, 18)
TRACKED_PROTOCOL_ID = "coherent_tracked_rehearsal_reserved_sixteen_v1"
PAIRED_PROTOCOL_ID = "coherent_paired_retention_initializations_v1"
PAIRED_SEEDS = PLANNED_SEEDS[1:]
POLICIES = ("rotating", "recent_only")
PAIRED_RESOURCES = PilotResources(training_updates=480, environment_steps=0, replay_bytes=384)
NO_REPLAY_PROTOCOL_ID = "coherent_stream_no_rehearsal_v1"
NO_REPLAY_LEARNABILITY_PROTOCOL_ID = "coherent_no_rehearsal_reserved_sixteen_v1"
NO_REPLAY_RESOURCES = PilotResources(training_updates=72, environment_steps=0, replay_bytes=0)
NO_REPLAY_LEARNABILITY_RESOURCES = PilotResources(
    training_updates=328, environment_steps=0, replay_bytes=0
)
CONFIRMATION_TRAINING_RNG = 7302
CONFIRMATION_FINAL_RNG = 9102


def _validate_recent_only(recent_only: bool) -> None:
    if type(recent_only) is not bool:
        raise ValueError("recent_only must be an exact boolean")


def _validate_no_replay(
    no_replay: bool, recent_only: bool, track_unit_history: bool = False
) -> None:
    if type(no_replay) is not bool:
        raise ValueError("no_replay must be an exact boolean")
    if no_replay and (recent_only or track_unit_history):
        raise ValueError("no_replay cannot use a replay policy or the fixed tracked preset")


def _validate_tracking(
    track_unit_history: bool, diagnose_learnability: bool, recent_only: bool
) -> None:
    if type(track_unit_history) is not bool:
        raise ValueError("track_unit_history must be an exact boolean")
    if track_unit_history and (diagnose_learnability is not True or recent_only is not False):
        raise ValueError("tracked unit history requires the fixed rotating learnability scenario")


def replay_policy(*, recent_only: bool = False) -> StreamReplayPolicy | RecentStreamReplayPolicy:
    _validate_recent_only(recent_only)
    if recent_only:
        return RecentStreamReplayPolicy(
            every_updates=3, rows_per_event=2, retention=ReplayRetentionBudget(16, 384)
        )
    return StreamReplayPolicy(
        every_updates=3,
        rows_per_event=2,
        retention=ReplayRetentionBudget(16, 384),
        allocation=ReplayAllocation(ReplayRetentionBudget(8, 192), ReplayRetentionBudget(8, 192)),
        seed=53,
    )


def _resources(diagnose_learnability: bool, *, no_replay: bool = False) -> PilotResources:
    if type(diagnose_learnability) is not bool:
        raise ValueError("diagnose_learnability must be an exact boolean")
    _validate_no_replay(no_replay, False)
    if no_replay:
        return NO_REPLAY_LEARNABILITY_RESOURCES if diagnose_learnability else NO_REPLAY_RESOURCES
    return LEARNABILITY_RESOURCES if diagnose_learnability else RESOURCES


def describe_protocol(
    *, diagnose_learnability: bool = False, recent_only: bool = False, no_replay: bool = False
) -> dict[str, Any]:
    _validate_recent_only(recent_only)
    _validate_no_replay(no_replay, recent_only)
    resources = (
        _resources(diagnose_learnability, no_replay=True)
        if no_replay
        else _resources(diagnose_learnability)
    )
    protocol: dict[str, Any] = dict(
        protocol_id=PROTOCOL_ID,
        classification="exploratory_native_rehearsal_engineering_smoke",
        methods=list(METHODS),
        streams=["recurring"],
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
        replay=None
        if no_replay
        else asdict(replay_policy(recent_only=True) if recent_only else replay_policy()),
        replay_selection="arrived-only recent/reservoir rotating interleave; context-blind",
        score_release="after all wake and replay training for each arm; no feedback",
        metrics=dict(
            primary="BCE clipped1e-12 lower better", secondary="accuracy >=0.5 higher better"
        ),
        directions=dict(
            new_B_gain="B BCE at0 minus12",
            retention_loss="A BCE at12 minus6",
            return_gain="A BCE at12 minus18",
        ),
        charging="separate initially planned wake/replay sessions, one global deadline; replay charged before admission; failed charges never refunded",
        native_clocks="replay calls native train_batch and advances its native clocks/gates",
        planned_work=dict(
            wake_updates=72,
            replay_updates=48,
            optimizer_updates=120,
            wake_row_exposures=576,
            replay_row_exposures=48,
            total_row_exposures=624,
            consolidation_events=24,
            final_score_forwards=32,
            probes=0,
            svd=0,
        ),
        resources=asdict(RESOURCES),
        stop_policy="one invocation, all failures/unreached rows retained; no retry",
        limitations="one fixed seed/data, unequal compute versus retained no-replay evidence; no confirmatory or circadian advantage claim",
    )
    if diagnose_learnability:
        original = learnability.describe_protocol(extended=True)
        # Why reuse the descriptor: the original negative endpoints must retain
        # exactly the same reserved data, horizon, diagnostic and metric rules.
        keys = (
            "probe_target",
            "probe_training_rng",
            "probe_evaluation_rng",
            "diagnostic_features_rng",
            "probe_training_batches",
            "probe_training_batch_rows",
            "probe_evaluation_rows",
            "probe_checkpoint_updates",
            "diagnostic_feature_rows",
            "feature_health_definition",
            "effective_rank_definition",
            "effective_rank_limit",
            "unavailable_diagnostics",
            "reserved_data_array_bytes",
            "diagnostic_reference_array_bytes",
            "score_definitions",
            "comparison_directions",
        )
        protocol.update(
            protocol_id=LEARNABILITY_PROTOCOL_ID,
            classification="exploratory_rehearsal_with_isolated_learnability",
            resources=asdict(resources),
            learnability=dict(
                source_protocol_id=original["protocol_id"], **{key: original[key] for key in keys}
            ),
            probe_isolation="independent template and disposable fork for each copied post-rehearsal checkpoint; benchmark final data unavailable to probes",
            checkpoint_clocks="0/6/12/18 are wake counts; native states additionally include rehearsal updates",
            probe_feedback=False,
        )
        protocol["planned_work"].update(
            optimizer_updates=376,
            probe_updates=256,
            probe_row_exposures=2048,
            total_row_exposures=2672,
            probe_clones=16,
            probe_score_forwards=96,
            diagnostic_feedforward_passes=16,
            probes=16,
            svd=32,
        )
    if recent_only:
        protocol.update(
            protocol_id=RECENT_LEARNABILITY_PROTOCOL_ID
            if diagnose_learnability
            else RECENT_PROTOCOL_ID,
            control_for=LEARNABILITY_PROTOCOL_ID if diagnose_learnability else PROTOCOL_ID,
            replay_selection="arrived-only recent FIFO; latest n distinct retained rows in arrival order",
            replay_retention="recent_fifo",
            comparison_scope="matched planned work/exposure and array-memory caps per method; retention composition AND within-event order differ; no isolated reservoir effect",
            limitations="one fixed seed/data; matched exposure/native work per method versus rotating control; retention and order change together; actual memory/latency may differ; no confirmation or circadian advantage claim",
        )
    if no_replay:
        protocol.update(
            protocol_id=NO_REPLAY_LEARNABILITY_PROTOCOL_ID
            if diagnose_learnability
            else NO_REPLAY_PROTOCOL_ID,
            classification="exploratory_no_replay_component_removal",
            control_for=LEARNABILITY_PROTOCOL_ID if diagnose_learnability else PROTOCOL_ID,
            replay_disabled=True,
            replay_selection="disabled; no retention, replay session or rehearsal helper",
            resources=asdict(resources),
            charging="original wake sessions and optional separate probe sessions; no replay allocation or charges",
            native_clocks="wake updates only; optional isolated probe forks retain their separate original clocks",
            limitations="fixed replay-removal control; unequal native work and row exposure versus rehearsal; no equivalence, minimal-bundle or circadian advantage claim",
        )
        protocol["planned_work"].update(
            replay_updates=0,
            optimizer_updates=328 if diagnose_learnability else 72,
            replay_row_exposures=0,
            total_row_exposures=2624 if diagnose_learnability else 576,
            consolidation_events=0,
        )
        if diagnose_learnability:
            protocol["checkpoint_clocks"] = (
                "0/6/12/18 are wake counts; native benchmark states contain no replay updates"
            )
            protocol["probe_isolation"] = (
                "independent template and disposable fork for each copied wake checkpoint; benchmark final data unavailable to probes"
            )
    return protocol


def _outcomes(result: dict[str, Any]) -> dict[str, float]:
    points = {row["updates"]: row["scores"] for row in result["checkpoints"]}
    return dict(
        new_B_gain=points[0]["B"]["bce"] - points[12]["B"]["bce"],
        retention_loss=points[12]["A"]["bce"] - points[6]["A"]["bce"],
        return_gain=points[12]["A"]["bce"] - points[18]["A"]["bce"],
    )


def _run_one(
    row,
    initial,
    deadline,
    sampler,
    *,
    diagnose_learnability: bool = False,
    recent_only: bool = False,
    track_unit_history: bool = False,
    no_replay: bool = False,
    confirmation: bool = False,
):
    if type(confirmation) is not bool:
        raise ValueError("confirmation must be an exact boolean")
    if confirmation and (diagnose_learnability or recent_only or track_unit_history):
        raise ValueError("confirmation fixes rotating/no replay without probes or tracking")
    _validate_recent_only(recent_only)
    _validate_no_replay(no_replay, recent_only, track_unit_history)
    _validate_tracking(track_unit_history, diagnose_learnability, recent_only)
    resources = (
        _resources(diagnose_learnability, no_replay=True)
        if no_replay
        else _resources(diagnose_learnability)
    )
    wake = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=18,
            max_wall_seconds=max(0.0, deadline - monotonic()),
            max_process_rss_bytes=resources.process_memory_bytes,
        ),
        monotonic,
    )
    replay = (
        None
        if no_replay
        else ToyBudgetSession(
            ToyExecutionBudget(
                max_replay_examples=12,
                max_wall_seconds=max(0.0, deadline - monotonic()),
                max_process_rss_bytes=resources.process_memory_bytes,
            ),
            monotonic,
        )
    )
    rehearsal = None
    started = monotonic()
    try:
        wake.attach_memory(sampler)
        if replay is not None:
            replay.attach_memory(sampler)
        wake.before_final()
        if replay is not None:
            replay.before_final()
        training, final = (
            build_stream(
                "recurring",
                training_rng=CONFIRMATION_TRAINING_RNG,
                heldout_rng=CONFIRMATION_FINAL_RNG,
            )
            if confirmation
            else build_stream("recurring")
        )
        learner = (
            build_learner(row["method"], initial, track_unit_history=True)
            if track_unit_history
            else build_learner(row["method"], initial)
        )
        before = pickle.dumps(learner.snapshot_state())
        observer = None
        template = None
        template_before = None
        if diagnose_learnability:
            # This independent template never receives benchmark final data or
            # the benchmark learner. The existing observer owns disposable forks.
            template = (
                build_learner(row["method"], initial, track_unit_history=True)
                if track_unit_history
                else build_learner(row["method"], initial)
            )
            if template is learner:
                raise ValueError("probe template must be distinct from benchmark learner")
            template_before = pickle.dumps(template.snapshot_state())
            observer = learnability.create_sixteen_probe_observer(
                template, row["observations"], deadline, sampler
            )
        if not no_replay:
            if replay is None:
                raise RuntimeError("rehearsal requires its original replay session")
            rehearsal = StreamRehearsal(
                learner,
                wake,
                replay_policy(recent_only=True) if recent_only else replay_policy(),
                replay_budget=replay,
                total_updates=18,
                input_dim=2,
            )
        options: dict[str, Any] = dict(
            checkpoint_updates=CHECKPOINTS, label_delay=2, budget=wake, rehearsal=rehearsal
        )
        if diagnose_learnability:
            options["observe_checkpoint"] = observer
        result = run_attribution_stream(learner, training, final, **options)
        if pickle.dumps(learner.snapshot_state()) != before:
            raise RuntimeError("rehearsal changed the source learner")
        if diagnose_learnability:
            if template is None or pickle.dumps(template.snapshot_state()) != template_before:
                raise RuntimeError("probe changed its independent source template")
            row["probe_template_unchanged"] = True
        row.update(
            status="complete",
            reason=None,
            stream_result=result,
            source_learner_unchanged=True,
            outcomes=_outcomes(result),
        )
    except Exception as error:
        row.update(status="failed", reason=f"{type(error).__name__}: {error}")
    row.update(
        wake_updates=wake.updates_completed,
        replay_charged_examples=0 if replay is None else replay.replay_examples_completed,
        elapsed_seconds=monotonic() - started,
        rehearsal=None if rehearsal is None else rehearsal.report,
    )
    if diagnose_learnability:
        row["diagnostic_seconds"] = sum(
            o.get("diagnostic_seconds", 0.0) for o in row["observations"]
        )
        row["probe_seconds"] = sum(o.get("probe_seconds", 0.0) for o in row["observations"])
        row["benchmark_rehearsal_and_orchestration_seconds"] = (
            row["elapsed_seconds"] - row["diagnostic_seconds"] - row["probe_seconds"]
        )


def run_pilot(
    *,
    diagnose_learnability: bool = False,
    recent_only: bool = False,
    track_unit_history: bool = False,
    no_replay: bool = False,
) -> dict[str, Any]:
    """Run the sole fixed case, keeping failed and unreached method rows."""
    _validate_recent_only(recent_only)
    _validate_no_replay(no_replay, recent_only, track_unit_history)
    _validate_tracking(track_unit_history, diagnose_learnability, recent_only)
    resources = (
        _resources(diagnose_learnability, no_replay=True)
        if no_replay
        else _resources(diagnose_learnability)
    )
    validate_local_pilot_request(PilotRequest(resources))
    started = monotonic()
    rows: list[dict[str, Any]] = [
        dict(
            method=m,
            stream="recurring",
            seed=23,
            status="not_run",
            reason="pilot stopped before execution",
            wake_updates=0,
            replay_charged_examples=0,
            rehearsal=None,
            planned_event_updates=[] if no_replay else list(range(3, 19, 3)),
        )
        for m in METHODS
    ]
    if diagnose_learnability:
        for row in rows:
            row["observations"] = learnability.empty_learnability_observations()
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
                if no_replay:
                    _run_one(
                        row,
                        initial,
                        started + resources.wall_seconds,
                        sampler,
                        no_replay=True,
                        diagnose_learnability=diagnose_learnability,
                    )
                elif track_unit_history:
                    _run_one(
                        row,
                        initial,
                        started + resources.wall_seconds,
                        sampler,
                        diagnose_learnability=True,
                        track_unit_history=True,
                    )
                elif recent_only:
                    _run_one(
                        row,
                        initial,
                        started + resources.wall_seconds,
                        sampler,
                        recent_only=True,
                        diagnose_learnability=diagnose_learnability,
                    )
                elif diagnose_learnability:
                    _run_one(
                        row,
                        initial,
                        started + resources.wall_seconds,
                        sampler,
                        diagnose_learnability=True,
                    )
                else:
                    _run_one(row, initial, started + resources.wall_seconds, sampler)
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
    wake = sum(row["wake_updates"] for row in rows)
    replay = sum(row["replay_charged_examples"] for row in rows)
    complete_rows = all(row["status"] == "complete" for row in rows)
    parity = rows[1].get("stream_result") == rows[2].get("stream_result") if complete_rows else None
    reports = [row["rehearsal"] for row in rows if row["rehearsal"] is not None]
    returned = sum(report["returned_updates"] for report in reports)
    committed = sum(report["committed_updates"] for report in reports)
    plans = [
        [(event["updates"], event["selected_ids"]) for event in report["events"]]
        for report in reports
    ]
    shared_plans = len(plans) == 4 and all(plan == plans[0] for plan in plans)
    replay_complete = (
        replay == returned == committed == 0 and not reports
        if no_replay
        else replay == returned == committed == 48 and shared_plans
    )
    complete = (
        not errors
        and complete_rows
        and wake == 72
        and replay_complete
        and elapsed <= resources.wall_seconds
        and memory is not None
        and memory["peak_bytes"] <= resources.process_memory_bytes
        and parity is True
    )
    report: dict[str, Any] = dict(
        protocol=describe_protocol(no_replay=True, diagnose_learnability=diagnose_learnability)
        if no_replay
        else describe_protocol(),
        status="complete" if complete else "incomplete",
        initial_parameters=initial_data,
        runs=rows,
        global_errors=errors,
        total_seconds=elapsed,
        observed_process_rss=memory,
        wake_updates=wake,
        replay_charged_examples=replay,
        replay_returned_updates=returned,
        replay_committed_updates=committed,
        shared_replay_plans_exact=None if no_replay else shared_plans,
        row_exposures_charged=8 * wake + replay,
        pc_neutral_exact=parity,
        scientific_advantage_verified=False,
        full_R23_complete=False,
    )
    if diagnose_learnability:
        if no_replay:
            _add_learnability_result(report, no_replay=True)
        else:
            _add_learnability_result(report)
    if recent_only:
        report["protocol"] = describe_protocol(
            recent_only=True, diagnose_learnability=diagnose_learnability
        )
    if track_unit_history:
        _add_tracked_result(report)
    if no_replay:
        report.update(
            protocol=describe_protocol(no_replay=True, diagnose_learnability=diagnose_learnability),
            replay_disabled=True,
            no_replay_verified=report["status"] == "complete",
            full_R28_complete=False,
        )
    return report


def _add_learnability_result(report: dict[str, Any], *, no_replay: bool = False) -> None:
    observations = [o for row in report["runs"] for o in row["observations"]]
    probes = [o["probe"] for o in observations if o["probe"] is not None]
    completed = sum(p["completed_updates"] for p in probes)
    attempted = sum(p["attempted_updates"] for p in probes)
    complete = (
        all(o["status"] == "complete" for o in observations) and completed == attempted == 256
    )
    pc, neutral = report["runs"][1:3]
    parity = (
        (
            [o["probe"]["checkpoints"] for o in pc["observations"]]
            == [o["probe"]["checkpoints"] for o in neutral["observations"]]
        )
        if complete
        else None
    )
    report.update(
        protocol=describe_protocol(diagnose_learnability=True, no_replay=True)
        if no_replay
        else describe_protocol(diagnose_learnability=True),
        status="complete"
        if report["status"] == "complete" and complete and parity is True
        else "incomplete",
        probe_attempted_updates=attempted,
        probe_completed_updates=completed,
        probe_training_row_exposures=sum(p["completed_training_examples"] for p in probes),
        probe_score_forward_calls_attempted=sum(p["score_forward_calls_attempted"] for p in probes),
        probe_score_forward_calls_completed=sum(p["score_forward_calls_completed"] for p in probes),
        total_completed_updates=report["wake_updates"]
        + report["replay_committed_updates"]
        + completed,
        total_training_row_exposures=report["row_exposures_charged"]
        + sum(p["completed_training_examples"] for p in probes),
        diagnostic_feedforward_passes_completed=sum("feature_health" in o for o in observations),
        diagnostic_svd_calls_completed=sum(len(o.get("feature_health", ())) for o in observations),
        pc_neutral_probe_curves_exact=parity,
        full_R21_complete=False,
    )


def describe_paired_protocol() -> dict[str, Any]:
    """Describe fixed initialization pairs without constructing native models."""
    protocol = describe_protocol()
    protocol.update(
        protocol_id=PAIRED_PROTOCOL_ID,
        classification="exploratory_fixed_data_paired_initializations",
        seeds=list(PAIRED_SEEDS),
        policies=list(POLICIES),
        loop_order="seed, policy, method; each row admitted once",
        initial_parameters="one owned native initial bundle per seed, shared across both policies and methods",
        replay={
            name: asdict(replay_policy(recent_only=name == "recent_only")) for name in POLICIES
        },
        replay_selection={
            "rotating": "arrived-only recent/reservoir rotating interleave",
            "recent_only": "arrived-only recent FIFO; latest n distinct rows in arrival order",
        },
        planned_work=dict(
            runs=16,
            pairs=8,
            wake_updates=288,
            replay_updates=192,
            optimizer_updates=480,
            wake_row_exposures=2304,
            replay_row_exposures=192,
            total_row_exposures=2496,
            consolidation_events=96,
            final_score_forwards=128,
            probes=0,
            svd=0,
        ),
        resources=asdict(PAIRED_RESOURCES),
        pairing="recent_only minus rotating for each seed/method; retain both primary checkpoint BCE/accuracy levels and all failure reasons",
        comparison_scope="matched planned exposure/native work and memory caps per method; retention composition and within-event order change together; BP versus PC compute remains unequal",
        limitations="two prespecified remaining initialization seeds on the same fixed data; no pooling, equivalence margin, intervals, task generalization, universal winner or confirmation; seed23 remains a retained exploratory development reference",
    )
    return protocol


def _paired_rows() -> list[dict[str, Any]]:
    return [
        dict(
            seed=seed,
            policy=policy,
            method=method,
            stream="recurring",
            status="not_run",
            reason="pilot stopped before execution",
            wake_updates=0,
            replay_charged_examples=0,
            rehearsal=None,
            outcomes=None,
            planned_event_updates=list(range(3, 19, 3)),
        )
        for seed in PAIRED_SEEDS
        for policy in POLICIES
        for method in METHODS
    ]


def _execute_paired_rows(
    rows: list[dict[str, Any]],
    initial_data: dict[str, Any],
    started: float,
    sampler: ProcessRssSampler,
) -> None:
    deadline = started + PAIRED_RESOURCES.wall_seconds
    for seed in PAIRED_SEEDS:
        if monotonic() >= deadline:
            return
        initial = CoupledPredictiveCodingNetwork(2, (8, 8), seed).snapshot_parameters()
        initial_data[str(seed)] = {
            item.name: ([a.tolist() for a in value] if type(value) is tuple else value.tolist())
            for item in fields(initial)
            for value in (getattr(initial, item.name),)
        }
        for row in rows:
            if row["seed"] != seed:
                continue
            if monotonic() >= deadline:
                return
            try:
                if row["policy"] == "recent_only":
                    _run_one(row, initial, deadline, sampler, recent_only=True)
                else:
                    _run_one(row, initial, deadline, sampler)
            except Exception as error:
                # A failure outside _run_one's protected native body still owns
                # its original slot; it is never retried or silently omitted.
                row.update(status="failed", reason=f"{type(error).__name__}: {error}")


def _paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["seed"], row["policy"], row["method"]): row for row in rows}
    pairs = []
    for seed in PAIRED_SEEDS:
        for method in METHODS:
            both = {policy: indexed[seed, policy, method] for policy in POLICIES}
            complete = all(row["status"] == "complete" for row in both.values())
            levels = (
                {p: deepcopy(row["stream_result"]["checkpoints"]) for p, row in both.items()}
                if complete
                else None
            )
            outcomes = (
                {p: deepcopy(row["outcomes"]) for p, row in both.items()} if complete else None
            )
            differences = (
                {
                    key: both["recent_only"]["outcomes"][key] - both["rotating"]["outcomes"][key]
                    for key in ("new_B_gain", "retention_loss", "return_gain")
                }
                if complete
                else None
            )
            pairs.append(
                dict(
                    seed=seed,
                    method=method,
                    status="complete" if complete else "incomplete",
                    reasons={p: row["reason"] for p, row in both.items()},
                    primary_levels=levels,
                    outcomes=outcomes,
                    recent_minus_rotating=differences,
                    initial_primary_levels_exact=(
                        levels["rotating"][0]["scores"] == levels["recent_only"][0]["scores"]
                    )
                    if levels is not None
                    else None,
                )
            )
    return pairs


def _plan(row: dict[str, Any]) -> list[tuple[int, list[str]]]:
    return [(event["updates"], event["selected_ids"]) for event in row["rehearsal"]["events"]]


def _paired_facts(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = []
    for seed in PAIRED_SEEDS:
        for policy in POLICIES:
            group = [row for row in rows if row["seed"] == seed and row["policy"] == policy]
            complete = all(row["status"] == "complete" for row in group)
            groups.append(
                dict(
                    seed=seed,
                    policy=policy,
                    status="complete" if complete else "unavailable",
                    reasons={row["method"]: row["reason"] for row in group},
                    pc_neutral_exact=(group[1]["stream_result"] == group[2]["stream_result"])
                    if complete
                    else None,
                    shared_replay_plans_exact=all(_plan(row) == _plan(group[0]) for row in group)
                    if complete
                    else None,
                )
            )
    policies = []
    for policy in POLICIES:
        group = [row for row in rows if row["policy"] == policy]
        complete = all(row["status"] == "complete" for row in group)
        policies.append(
            dict(
                policy=policy,
                status="complete" if complete else "unavailable",
                replay_plans_exact_across_initializations=all(
                    _plan(row) == _plan(group[0]) for row in group
                )
                if complete
                else None,
            )
        )
    return groups, policies


def _paired_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [row["rehearsal"] for row in rows if row["rehearsal"] is not None]
    wake = sum(row["wake_updates"] for row in rows)
    charged = sum(row["replay_charged_examples"] for row in rows)
    attempted = sum(report["attempted_updates"] for report in reports)
    returned = sum(report["returned_updates"] for report in reports)
    committed = sum(report["committed_updates"] for report in reports)
    forwards = sum(
        row["stream_result"]["score_forward_calls"] for row in rows if row["status"] == "complete"
    )
    complete = all(row["status"] == "complete" for row in rows)
    return dict(
        wake_updates=wake,
        replay_charged_examples=charged,
        replay_attempted_updates=attempted,
        replay_returned_updates=returned,
        replay_committed_updates=committed,
        total_committed_updates=wake + committed,
        row_exposures_charged=8 * wake + charged,
        known_native_train_calls=wake + attempted,
        known_completed_score_forwards=forwards,
        native_train_calls=wake + attempted if complete else None,
        final_score_forwards=forwards if complete else None,
        partial_work_note="known counts retain reported wake/receipt and replay work; failed wake or scoring calls may contain additional unobserved work, never imputed as zero",
    )


def run_paired_pilot() -> dict[str, Any]:
    """Run each declared pair once under one global deadline and RSS sampler."""
    validate_local_pilot_request(PilotRequest(PAIRED_RESOURCES))
    rows = _paired_rows()  # Allocate every failure/unreached slot before setup.
    initial_data: dict[str, Any] = {}
    sampler = None
    errors = []
    started = monotonic()
    try:
        sampler = ProcessRssSampler()
        with sampler:
            _execute_paired_rows(rows, initial_data, started, sampler)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        if sampler is None:
            raise RuntimeError("RSS sampler setup unavailable")
        memory = asdict(sampler.snapshot())
    except Exception as error:
        memory = None
        errors.append(f"{type(error).__name__}: {error}")
    pairs = _paired_comparisons(rows)
    groups, policies = _paired_facts(rows)
    totals = _paired_totals(rows)
    elapsed = monotonic() - started
    complete = (
        not errors
        and all(
            pair["status"] == "complete" and pair["initial_primary_levels_exact"] is True
            for pair in pairs
        )
        and all(
            group["pc_neutral_exact"] is True and group["shared_replay_plans_exact"] is True
            for group in groups
        )
        and all(policy["replay_plans_exact_across_initializations"] is True for policy in policies)
        and totals["wake_updates"] == 288
        and totals["replay_charged_examples"]
        == totals["replay_returned_updates"]
        == totals["replay_committed_updates"]
        == 192
        and totals["native_train_calls"] == 480
        and totals["final_score_forwards"] == 128
        and elapsed <= PAIRED_RESOURCES.wall_seconds
        and memory is not None
        and memory["peak_bytes"] <= PAIRED_RESOURCES.process_memory_bytes
    )
    return dict(
        protocol=describe_paired_protocol(),
        status="complete" if complete else "incomplete",
        initial_parameters=initial_data,
        runs=rows,
        pairs=pairs,
        group_facts=groups,
        policy_plan_facts=policies,
        global_errors=errors,
        total_seconds=elapsed,
        observed_process_rss=memory,
        **totals,
        scientific_advantage_verified=False,
        full_R28_complete=False,
        full_R23_complete=False,
    )


def describe_tracked_protocol() -> dict[str, Any]:
    """Describe native history instrumentation of the retained fixed probe case."""
    protocol = describe_protocol(diagnose_learnability=True)
    protocol.update(
        protocol_id=TRACKED_PROTOCOL_ID,
        classification="native_history_observer_composition_verification",
        control_for=LEARNABILITY_PROTOCOL_ID,
        native_unit_history=dict(
            methods=["pc", "neutral_cpc", "gated_cpc"],
            bp="unavailable; ordinary matched BP stores no native unit history",
            ema_decay=0.95,
            definition="EMA of absolute batch-aggregate outgoing coupled-energy gradient before CPC gates; not causal importance",
            age="successful native updates since tracked construction, including replay",
            checkpoint_native_updates=[0, 10, 20, 30],
            persistent_array_bytes_per_model=192,
            expected_completed_observations=12,
            history_snapshot_format=2,
            persistence="existing exact format1 disk codecs/managed capture refuse tracked state",
        ),
        limitations="instrumentation verification on the retained fixed seed/data/horizon, not new independent learning evidence; no advantage, renewal, maturity differentiation, persistence, generalization or confirmation claim",
    )
    protocol["planned_work"].update(
        native_unit_history_reads=12, native_unit_history_transitions=282
    )
    protocol["learnability"]["unavailable_diagnostics"] = {
        "importance": "BP unavailable; exact tracked PC/CPC snapshots provide native gradient EMA summaries",
        "neuron_age": "BP unavailable; exact tracked PC/CPC snapshots provide successful-update ages including replay",
    }
    return protocol


def _add_tracked_result(report: dict[str, Any]) -> None:
    completed = [
        o["native_unit_history"]
        for row in report["runs"][1:]
        for o in row["observations"]
        if o["status"] == "complete" and "native_unit_history" in o
    ]
    complete = len(completed) == 12 and all(
        o["native_unit_history"]["successful_updates"] == point + 2 * (point // 3)
        and o["native_unit_history"]["array_bytes"] == 192
        for row in report["runs"][1:]
        for point, o in zip(CHECKPOINTS, row["observations"], strict=True)
        if o["status"] == "complete" and "native_unit_history" in o
    )
    pc, neutral = report["runs"][1:3]
    parity = (
        (
            [o["native_unit_history"] for o in pc["observations"]]
            == [o["native_unit_history"] for o in neutral["observations"]]
        )
        if complete
        else None
    )
    report.update(
        protocol=describe_tracked_protocol(),
        status="complete"
        if report["status"] == "complete" and complete and parity is True
        else "incomplete",
        native_unit_history_observations_completed=len(completed),
        native_unit_history_array_bytes_in_completed_observations=sum(
            o["array_bytes"] for o in completed
        ),
        pc_neutral_native_unit_histories_exact=parity,
    )


def run_tracked_pilot() -> dict[str, Any]:
    """Execute the fixed tracked scenario through the existing rehearsal runner."""
    return run_pilot(diagnose_learnability=True, track_unit_history=True)
