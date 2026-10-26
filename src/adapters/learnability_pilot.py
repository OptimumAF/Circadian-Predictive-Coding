"""Fixed fresh-task learning probes on the existing delayed ABA stream.

Checkpoint observers receive copied state and no benchmark evaluation data.
Probe templates are independent of benchmark learners. Outputs are exploratory
fixed-data observations, not a controller, checkpoint or plasticity theorem.
"""

from dataclasses import asdict, fields, is_dataclass, replace
from math import isfinite
import pickle
from time import monotonic
from typing import Any, Callable

import numpy as np

from src.adapters.coupled_attribution_pilot import build_learner, build_stream
from src.adapters.native_unit_diagnostics import (
    read_coupled_unit_diagnostics,
    read_renewable_coupled_unit_diagnostics,
)
from src.app.coupled_attribution import AttributionBatch, run_attribution_stream
from src.app.coupled_attribution_summary import CHECKPOINTS, METHODS
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_energy import Array, feedforward_latents
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.actor_ports import ForkableLearner
from src.core.learnability_diagnostics import diagnose_activations
from src.core.local_pilot_budget import PilotRequest, PilotResources, validate_local_pilot_request
from src.core.tracked_coupled_predictive_coding import TrackedCoupledSnapshot
from src.core.renewable_coupled_predictive_coding import RenewableCoupledSnapshot
from src.shared.process_memory import ProcessRssSampler

PROTOCOL_ID = "reserved_fresh_task_learnability_smoke_v1"
RESOURCES = PilotResources(training_updates=136, environment_steps=0, replay_bytes=0)
PROBE_CHECKPOINTS = (0, 1, 2, 4)
EXTENDED_PROTOCOL_ID = "reserved_fresh_task_learnability_sixteen_v1"
EXTENDED_PROBE_CHECKPOINTS = (0, 1, 2, 4, 8, 16)
EXTENDED_RESOURCES = PilotResources(training_updates=328, environment_steps=0, replay_bytes=0)


def _scenario(extended: bool) -> tuple[str, tuple[int, ...], PilotResources]:
    if type(extended) is not bool:
        raise ValueError("extended must be an exact boolean")
    if extended:
        return EXTENDED_PROTOCOL_ID, EXTENDED_PROBE_CHECKPOINTS, EXTENDED_RESOURCES
    return PROTOCOL_ID, PROBE_CHECKPOINTS, RESOURCES


def describe_protocol(*, extended: bool = False) -> dict[str, Any]:
    protocol_id, checkpoints, resources = _scenario(extended)
    probe_updates = 16 * checkpoints[-1]
    protocol = dict(
        protocol_id=protocol_id,
        classification="exploratory_fixed_data_engineering_smoke",
        seeds=[23],
        methods=list(METHODS),
        benchmark_stream="recurring_A_B_A",
        benchmark_updates_per_phase=6,
        benchmark_checkpoint_updates=list(CHECKPOINTS),
        benchmark_training_rng=7301,
        benchmark_final_rng=9101,
        hidden_dims=[8, 8],
        learning_rate=0.03,
        precisions=[1.0, 1.0],
        active_cpc_config=asdict(CoupledCircadianConfig()),
        neutral_cpc_config=asdict(CoupledCircadianConfig.matched_pc_control()),
        benchmark_batch_rows=8,
        benchmark_final_rows_per_context=64,
        benchmark_target_rules={"A": "x0 >= 0", "B": "x1 >= 0"},
        label_delay=2,
        inference_steps=4,
        inference_learning_rate=0.2,
        probe_target="x0*x1 >= 0; fixed nonlinear composition of A/B inputs",
        probe_training_rng=17031,
        probe_evaluation_rng=17033,
        diagnostic_features_rng=17035,
        probe_training_batches=checkpoints[-1],
        probe_training_batch_rows=8,
        probe_evaluation_rows=32,
        probe_checkpoint_updates=list(checkpoints),
        diagnostic_feature_rows=32,
        feature_health_definition="mean absolute tanh activation; dead <=1e-3; active complement; saturated entry abs>=.99",
        effective_rank_definition="centered SVD singular VALUES entropy; s>eps*max(shape)*max(1,max(s)); numerical zero rank0",
        effective_rank_limit="absolute numerical cutoff is not invariant at tiny scales; diagnostic, not causal proof",
        unavailable_diagnostics={
            "importance": "Coupled models do not store native per-neuron importance.",
            "neuron_age": "Coupled models do not store native per-neuron ages.",
        },
        planned_work=dict(
            benchmark_updates=72,
            probe_updates=probe_updates,
            total_updates=72 + probe_updates,
            benchmark_training_row_exposures=576,
            probe_training_row_exposures=8 * probe_updates,
            probe_clones=16,
            probe_score_forward_calls=16 * len(checkpoints),
            benchmark_score_forward_calls=32,
            diagnostic_feedforward_passes=16,
            diagnostic_svd_calls=32,
            replay_examples=0,
        ),
        score_definitions=dict(
            bce="probability BCE clip1e-12 lower better", accuracy="threshold >=.5 higher better"
        ),
        comparison_directions=dict(
            fresh_gain=f"probe BCE at0 minus BCE at{checkpoints[-1]}; retain initial level and entire curve",
            retention_loss="A BCE afterB minus A BCE afterA; lower better",
            return_gain="A BCE afterB minus A BCE afterReturn; higher better",
            new_B_gain="B BCE initial minus B BCE afterB; higher better",
        ),
        isolation="independent probe template; copied checkpoint and reserved probe data; probes finish before benchmark final scoring",
        work_limit="BP and PC have unequal native work; nominal counts and elapsed time are not matched compute",
        reserved_data_array_bytes=8 * 3 * 8 * checkpoints[-1] + 32 * 3 * 8,
        diagnostic_reference_array_bytes=512,
        stop_policy="One invocation; preserve failed or unreached rows; no retry or tuning.",
        resources=asdict(resources),
        limitations="One prespecified engineering seed, reused fixed ABA data, no intervals or confirmation; C is not independent task generalization. Native importance/age and full R2.1 remain open. No tuning, promotion, replay, sleep or CPC advantage claim.",
    )
    if extended:
        protocol.update(
            predecessor_protocol_id=PROTOCOL_ID,
            motivation=(
                "Retain the negative fifteen-of-sixteen four-update endpoints and observe "
                "one predeclared longer exposure on the same data; no favorable endpoint selection."
            ),
            extended_probe_limits=dict(
                training_batches=16,
                checkpoint_updates=list(checkpoints),
                no_endpoint_selection=True,
            ),
        )
    return protocol


def build_reserved_data(*, extended: bool = False):
    _, checkpoints, _ = _scenario(extended)
    training = []
    generator = np.random.default_rng(17031)
    for index in range(checkpoints[-1]):
        x = generator.uniform(-1, 1, (8, 2))
        y = (x[:, [0]] * x[:, [1]] >= 0).astype(np.float64)
        training.append(
            AttributionBatch(
                "probe_train", "C", tuple(f"probe-train-{index}-{j}" for j in range(8)), x, y
            )
        )
    x = np.random.default_rng(17033).uniform(-1, 1, (32, 2))
    evaluation = (
        AttributionBatch(
            "probe_test",
            "C",
            tuple(f"probe-test-{j}" for j in range(32)),
            x,
            (x[:, [0]] * x[:, [1]] >= 0).astype(np.float64),
        ),
    )
    diagnostic_features = np.random.default_rng(17035).uniform(-1, 1, (32, 2))
    return tuple(training), evaluation, diagnostic_features


def _array_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Count retained snapshot arrays once; excludes Python and temporary memory."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if type(value) is np.ndarray:
        return value.nbytes
    if is_dataclass(value) and not isinstance(value, type):
        return sum(_array_bytes(getattr(value, field.name), seen) for field in fields(value))
    if type(value) in (tuple, list):
        return sum(_array_bytes(item, seen) for item in value)
    return 0


def _empty_observations() -> list[dict[str, Any]]:
    return [
        dict(updates=n, status="not_run", reason="checkpoint not reached", probe=None)
        for n in CHECKPOINTS
    ]


def empty_learnability_observations() -> list[dict[str, Any]]:
    """Create the existing fixed benchmark-checkpoint observation slots."""
    return _empty_observations()


def create_sixteen_probe_observer(
    template: ForkableLearner[Array, Array, Array, Any],
    observations: list[dict[str, Any]],
    deadline: float,
    sampler: ProcessRssSampler | None = None,
    *,
    probe_budgets: tuple[ToyBudgetSession, ...] | None = None,
) -> Callable[[int, Any, AttributionBatch | None], None]:
    """Reuse the fixed reserved probe on independent post-rehearsal snapshots.

    The caller supplies an independent probe template, never its benchmark
    learner. This facade preserves the existing observer's copied-state checks,
    fixed data/horizon and deferred benchmark evaluation boundaries.
    Optional prepared budgets retain four setup clocks and quotas, spending each
    original session once at the ordered checkpoints without constructing one.
    """
    if not callable(getattr(template, "fork", None)) or not callable(
        getattr(template, "snapshot_state", None)
    ):
        raise ValueError("probe template must provide owned fork and snapshot_state methods")
    if (
        type(observations) is not list
        or len(observations) != len(CHECKPOINTS)
        or any(
            type(row) is not dict or type(row.get("updates")) is not int or row["updates"] != point
            for row, point in zip(observations, CHECKPOINTS)
        )
    ):
        raise ValueError("probe observations must be fixed slots at updates 0, 6, 12 and 18")
    try:
        valid_deadline = type(deadline) in {int, float} and isfinite(deadline)
    except OverflowError:
        valid_deadline = False
    if not valid_deadline:
        raise ValueError("probe deadline must be a finite native number")
    if probe_budgets is not None:
        _validate_prepared_probe_budgets(probe_budgets, deadline, sampler)
    training, evaluation, reference = build_reserved_data(extended=True)
    prepared: dict[str, Any] = {} if probe_budgets is None else {"probe_budgets": probe_budgets}
    observer: Callable[[int, Any, AttributionBatch | None], None] = _observer(
        template,
        training,
        evaluation,
        reference,
        observations,
        deadline,
        sampler,
        probe_checkpoints=EXTENDED_PROBE_CHECKPOINTS,
        **prepared,
    )
    return observer


def _validate_prepared_probe_budgets(budgets, deadline, sampler):
    """Admit original setup allowances before allocating reserved probe data."""
    if (
        type(budgets) is not tuple
        or len(budgets) != len(CHECKPOINTS)
        or len({id(budget) for budget in budgets}) != len(CHECKPOINTS)
    ):
        raise ValueError("prepared probes require four distinct original budget sessions")
    now = monotonic()
    if not isfinite(now) or now >= deadline:
        raise ValueError("prepared probe global deadline exhausted")
    for budget in budgets:
        if type(budget) is not ToyBudgetSession or type(budget.budget) is not ToyExecutionBudget:
            raise ValueError("prepared probes require exact ToyBudgetSession limits")
        replace(budget.budget)
        limits = budget.budget
        if (
            limits.max_training_updates != 16
            or type(limits.max_training_updates) is not int
            or limits.max_wall_seconds is None
            or budget.clock is not monotonic
            or type(budget.started_at) not in {int, float}
            or not isfinite(budget.started_at)
            or type(budget.last_clock) not in {int, float}
            or not isfinite(budget.last_clock)
            or not budget.started_at <= budget.last_clock <= now
            or limits.max_wall_seconds > RESOURCES.wall_seconds
            or now - budget.started_at >= limits.max_wall_seconds
            or limits.max_process_rss_bytes
            != (None if sampler is None else RESOURCES.process_memory_bytes)
            or budget.process_rss_sampler is not sampler
            or type(budget.updates_completed) is not int
            or budget.updates_completed != 0
            or type(budget.replay_examples_completed) is not int
            or budget.replay_examples_completed != 0
            or budget.checkpoint_position is not None
            or budget.hidden_width_observed is not None
            or budget.peak_hidden_width_observed is not None
            or budget.rejected_proposed_hidden_width is not None
            or not _prepared_progress_matches(budget)
        ):
            raise ValueError(
                "prepared probe budgets must be fresh original sixteen-update allowances"
            )


def _prepared_progress_matches(budget):
    progress = budget.progress
    return progress is None or (
        type(progress.updates_completed) is int
        and progress.updates_completed == budget.updates_completed
        and type(progress.replay_examples_completed) is int
        and progress.replay_examples_completed == 0
        and progress.checkpoint_position is None
        and progress.hidden_width_observed is None
        and progress.peak_hidden_width_observed is None
        and progress.rejected_proposed_hidden_width is None
    )


def _prepared_probe_admission(budgets, deadline, sampler):
    _validate_prepared_probe_budgets(budgets, deadline, sampler)
    # Why retain identities: checkpoint probes spend setup allowances rather than
    # starting new clocks, replacing samplers or granting another update quota.
    originals = tuple(
        (budget.budget, asdict(budget.budget), budget.clock, budget.started_at, budget.progress)
        for budget in budgets
    )
    counts = [0] * len(budgets)
    clocks = [budget.last_clock for budget in budgets]
    selected = -1

    def check():
        now = monotonic()
        if not isfinite(now) or now >= deadline:
            raise RuntimeError("prepared probe global deadline exhausted")
        for index, (budget, original) in enumerate(zip(budgets, originals, strict=True)):
            limits, values, clock, started, progress = original
            count = budget.updates_completed
            if (
                budget.budget is not limits
                or asdict(budget.budget) != values
                or budget.clock is not clock
                or budget.started_at != started
                or budget.progress is not progress
                or budget.process_rss_sampler is not sampler
                or type(count) is not int
                or not counts[index] <= count <= min(16, counts[index] + 1)
                or (index != selected and count != counts[index])
                or type(budget.replay_examples_completed) is not int
                or budget.replay_examples_completed != 0
                or budget.checkpoint_position is not None
                or budget.hidden_width_observed is not None
                or budget.peak_hidden_width_observed is not None
                or budget.rejected_proposed_hidden_width is not None
                or not _prepared_progress_matches(budget)
                or type(budget.last_clock) not in {int, float}
                or not isfinite(budget.last_clock)
                or budget.last_clock < clocks[index]
            ):
                raise RuntimeError("prepared probe original budget changed or was renewed")
            counts[index] = count
            clocks[index] = budget.last_clock

    def admit(index):
        nonlocal selected
        check()
        if index != selected + 1 or (selected >= 0 and counts[selected] != 16):
            raise RuntimeError("prepared probes require ordered complete checkpoints 0, 6, 12, 18")
        selected = index
        return budgets[index], check

    return admit


def _measure_features(state, reference, row):
    started = monotonic()
    try:
        # Reject corrupt native history before paying for feature inference/SVD.
        if type(state) is TrackedCoupledSnapshot:
            native = read_coupled_unit_diagnostics(state)
        elif type(state) is RenewableCoupledSnapshot:
            native = read_renewable_coupled_unit_diagnostics(state)
        else:
            native = None
        activations = feedforward_latents(state.parameters, reference)
        health = [asdict(item) for item in diagnose_activations(activations)]
        row.update(
            feature_health=health,
            diagnostic_activation_bytes=sum(a.nbytes for a in activations),
            checkpoint_snapshot_array_bytes=_array_bytes(state),
            importance=None,
            neuron_age=None,
            unavailable_reason="No native per-neuron importance/age in coupled models.",
        )
        if native is not None:
            row.update(
                native_unit_history=asdict(native),
                importance=asdict(native.history.importance),
                neuron_age=asdict(native.history.age),
                unavailable_reason=None,
            )
    finally:
        row["diagnostic_seconds"] = monotonic() - started


def _measure_probe(
    template,
    state,
    training,
    evaluation,
    row,
    deadline,
    sampler,
    *,
    probe_checkpoints=PROBE_CHECKPOINTS,
    probe_budget=None,
    check_boundary=None,
):
    if probe_budget is None:
        remaining = deadline - monotonic()
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=probe_checkpoints[-1],
                max_wall_seconds=max(0.0, remaining),
                max_process_rss_bytes=RESOURCES.process_memory_bytes
                if sampler is not None
                else None,
            ),
            monotonic,
        )
        if sampler is not None:
            budget.attach_memory(sampler)
    else:
        budget = probe_budget
    guarded: dict[str, Any] = {} if check_boundary is None else {"check_boundary": check_boundary}
    started = monotonic()
    try:
        row["probe"] = run_fresh_task_probe(
            template,
            state,
            training,
            evaluation,
            budget=budget,
            checkpoint_updates=probe_checkpoints,
            **guarded,
        )
    finally:
        # Includes validation and owned data/state detachment, unlike the inner
        # native-operation timer. Failed attempts keep their measured call time.
        row["probe_seconds"] = monotonic() - started
    return row["probe"]


def _observer(
    template,
    training,
    evaluation,
    reference,
    observations,
    deadline,
    sampler=None,
    *,
    probe_checkpoints=PROBE_CHECKPOINTS,
    probe_budgets=None,
):
    """No benchmark learner or final data is captured by this callback."""
    admit = (
        None
        if probe_budgets is None
        else _prepared_probe_admission(probe_budgets, deadline, sampler)
    )

    def observe(updates, state, arrived):
        if arrived is not None and arrived.role != "train":
            raise ValueError("checkpoint observer accepts only arrived training metadata")
        row = observations[CHECKPOINTS.index(updates)]
        prepared: dict[str, Any] = {}
        if admit is not None:
            try:
                budget, check = admit(CHECKPOINTS.index(updates))
            except Exception as error:
                if row["status"] != "complete":
                    row.update(status="failed", reason=f"{type(error).__name__}: {error}")
                raise
            prepared = {"probe_budget": budget, "check_boundary": check}
        row.update(status="failed", reason="checkpoint observation incomplete")
        state_before = pickle.dumps(state)
        template_before = pickle.dumps(template.snapshot_state())
        _measure_features(state, reference, row)
        if prepared:
            prepared["check_boundary"]()
        probe = _measure_probe(
            template,
            state,
            training,
            evaluation,
            row,
            deadline,
            sampler,
            probe_checkpoints=probe_checkpoints,
            **prepared,
        )
        if (
            pickle.dumps(state) != state_before
            or pickle.dumps(template.snapshot_state()) != template_before
        ):
            raise RuntimeError("fresh probe changed its input checkpoint or independent template")
        if probe["status"] != "complete":
            row["reason"] = probe["reason"]
            raise RuntimeError(f"fresh probe incomplete: {probe['reason']}")
        curve = probe["checkpoints"]
        row.update(
            status="complete",
            reason=None,
            fresh_initial_bce=curve[0]["scores"]["C"]["bce"],
            fresh_gain=curve[0]["scores"]["C"]["bce"] - curve[-1]["scores"]["C"]["bce"],
            probe_source_and_checkpoint_unchanged=True,
        )

    return observe


def _comparison(result):
    points = {row["updates"]: row["scores"] for row in result["checkpoints"]}
    return dict(
        retention_loss=points[12]["A"]["bce"] - points[6]["A"]["bce"],
        return_gain=points[12]["A"]["bce"] - points[18]["A"]["bce"],
        new_B_gain=points[0]["B"]["bce"] - points[12]["B"]["bce"],
    )


def _run_one(row, initial, deadline, sampler, *, extended: bool = False):
    _, probe_checkpoints, resources = _scenario(extended)
    started = monotonic()
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=18,
            max_wall_seconds=max(0.0, deadline - started),
            max_process_rss_bytes=resources.process_memory_bytes,
        ),
        monotonic,
    )
    try:
        budget.attach_memory(sampler)
        budget.before_final()
        training, heldout = build_stream("recurring")
        learner = build_learner(row["method"], initial)
        # Why separate templates: probes must never even receive the learner used
        # by benchmark orchestration or its final checkpoint scorer.
        template = build_learner(row["method"], initial)
        before = pickle.dumps(learner.snapshot_state())
        reserved_training, reserved_test, reference = (
            build_reserved_data(extended=True) if extended else build_reserved_data()
        )
        observer = _observer(
            template,
            reserved_training,
            reserved_test,
            reference,
            row["observations"],
            deadline,
            sampler,
            probe_checkpoints=probe_checkpoints,
        )
        result = run_attribution_stream(
            learner,
            training,
            heldout,
            checkpoint_updates=CHECKPOINTS,
            label_delay=2,
            budget=budget,
            observe_checkpoint=observer,
        )
        if pickle.dumps(learner.snapshot_state()) != before:
            raise RuntimeError("benchmark template changed during stream/probes")
        row.update(
            status="complete",
            reason=None,
            stream_result=result,
            benchmark_template_unchanged=True,
            outcomes=_comparison(result),
        )
    except Exception as error:
        row.update(status="failed", reason=f"{type(error).__name__}: {error}", outcomes=None)
    row["benchmark_completed_updates"] = budget.updates_completed
    row["elapsed_seconds"] = monotonic() - started
    row["diagnostic_seconds"] = sum(o.get("diagnostic_seconds", 0.0) for o in row["observations"])
    row["probe_seconds"] = sum(o.get("probe_seconds", 0.0) for o in row["observations"])
    row["benchmark_and_orchestration_seconds"] = (
        row["elapsed_seconds"] - row["diagnostic_seconds"] - row["probe_seconds"]
    )
    return row


def run_pilot(*, extended: bool = False) -> dict[str, Any]:
    _, probe_checkpoints, resources = _scenario(extended)
    validate_local_pilot_request(PilotRequest(resources))
    started = monotonic()
    rows: list[dict[str, Any]] = [
        dict(
            method=method,
            seed=23,
            status="not_run",
            reason="pilot not reached",
            observations=_empty_observations(),
            outcomes=None,
        )
        for method in METHODS
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
                if extended:
                    _run_one(row, initial, started + resources.wall_seconds, sampler, extended=True)
                else:
                    _run_one(row, initial, started + resources.wall_seconds, sampler)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        if sampler is None:
            raise RuntimeError("RSS sampler setup was unavailable")
        memory = asdict(sampler.snapshot())
    except Exception as error:
        memory = None
        errors.append(f"{type(error).__name__}: {error}")
    elapsed = monotonic() - started
    probes = [o["probe"] for row in rows for o in row["observations"] if o["probe"] is not None]
    committed = sum(row.get("benchmark_completed_updates", 0) for row in rows)
    probe_updates = sum(p["completed_updates"] for p in probes)
    complete = (
        not errors
        and all(row["status"] == "complete" for row in rows)
        and committed == 72
        and probe_updates == 16 * probe_checkpoints[-1]
        and elapsed <= resources.wall_seconds
        and memory is not None
        and memory["peak_bytes"] <= resources.process_memory_bytes
    )
    pc, neutral = rows[1], rows[2]
    parity = None
    if pc["status"] == neutral["status"] == "complete":
        parity = pc["stream_result"] == neutral["stream_result"] and [
            o["probe"]["checkpoints"] for o in pc["observations"]
        ] == [o["probe"]["checkpoints"] for o in neutral["observations"]]
    complete = complete and parity is True
    return dict(
        protocol=describe_protocol(extended=True) if extended else describe_protocol(),
        status="complete" if complete else "incomplete",
        initial_parameters=initial_data,
        runs=rows,
        global_errors=errors,
        total_seconds=elapsed,
        observed_process_rss=memory,
        benchmark_completed_updates=committed,
        probe_completed_updates=probe_updates,
        total_completed_updates=committed + probe_updates,
        probe_attempted_updates=sum(p["attempted_updates"] for p in probes),
        probe_training_row_exposures=sum(p["completed_training_examples"] for p in probes),
        benchmark_training_row_exposures=8 * committed,
        probe_score_forward_calls_attempted=sum(p["score_forward_calls_attempted"] for p in probes),
        probe_score_forward_calls_completed=sum(p["score_forward_calls_completed"] for p in probes),
        diagnostic_feedforward_passes=sum(
            "feature_health" in o for r in rows for o in r["observations"]
        ),
        diagnostic_svd_calls=sum(
            len(o.get("feature_health", ())) for r in rows for o in r["observations"]
        ),
        pc_neutral_curves_exact=parity,
        scientific_advantage_verified=False,
        full_R21_complete=False,
    )
