"""Disposable continuous probes using the existing delayed-label runtime.

Inputs are owned reserved-role episodes and a released native checkpoint. Outputs
are bounded physical scores, receipts and costs, never an adapted learner. No fit,
data selection, final evaluation, replay, promotion or classification coercion.
"""

from dataclasses import asdict, dataclass, replace
import math
import json
from time import monotonic
from typing import Any

import numpy as np

from src.adapters.continuous_learnability_modes import (
    RecordedContinuousPC,
    same_continuous_state,
)
from src.adapters.continuous_probe_control import (
    _native_state,
    _observe_state,
    run_zero_velocity_pair,
)
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner
from src.adapters.numpy_continuous_pc import NumpyContinuousPCLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.app.probe_endpoint_observer import _BudgetWitness
from src.app.probe_return_phase import budget_binding, paid_report_guard
from src.core import continuous_bc_diagnostics as diagnostics
from src.core import continuous_bc as task
from src.core.continuous_pc import CONTINUOUS_PC_ENERGY_ID
from src.core.continuous_step_diagnostics import _StepDirectionWork, diagnose_step_direction
from src.core.learner_ports import TrainingDiagnostic
from src.core.continuous_demonstration import (
    ContinuousEpisode,
    continuous_training_pairs,
    copy_continuous_episode,
)
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock


@dataclass(frozen=True)
class ContinuousProbePlan:
    training: ContinuousEpisode
    evaluation: ContinuousEpisode
    updates: int = 1
    diagnostics: bool = False
    zero_velocity_control: bool = False

    def __post_init__(self):
        if type(self.zero_velocity_control) is not bool:
            raise ValueError("continuous probe zero_velocity_control must be an exact bool")
        if type(self.diagnostics) is not bool:
            raise ValueError("continuous probe diagnostics must be an exact bool")
        if type(self.updates) is not int or not 1 <= self.updates <= 16:
            raise ValueError("continuous probe updates must be an integer in [1,16]")
        episodes = (self.training, self.evaluation)
        if any(type(item) is not ContinuousEpisode for item in episodes):
            raise ValueError("continuous probe requires exact reserved episodes")
        if any(not 1 <= len(item.transitions) <= 128 for item in episodes):
            raise ValueError("continuous probe episodes require1..128 rows")
        if (
            sum(
                len(item.transitions) * (item.spec.observation_dim + len(item.spec.action_units))
                for item in episodes
            )
            > 65536
        ):
            raise ValueError("continuous probe exceeds65536 feature/target entries")
        training, evaluation = (copy_continuous_episode(item) for item in episodes)
        if training.role != "probe_train" or evaluation.role != "probe_test":
            raise ValueError("continuous probe requires probe_train and probe_test roles")
        if (
            training.spec != evaluation.spec
            or training.context != evaluation.context
            or training.episode_id == evaluation.episode_id
        ):
            raise ValueError("probe episodes require matching spec/context and distinct identities")
        object.__setattr__(self, "training", training)
        object.__setattr__(self, "evaluation", evaluation)


def copy_probe_plan(value, benchmark_episodes=()):
    if value is None:
        return None
    if type(value) is not ContinuousProbePlan:
        raise ValueError("probe_plan must be an exact ContinuousProbePlan")
    owned = ContinuousProbePlan(
        value.training,
        value.evaluation,
        value.updates,
        value.diagnostics,
        value.zero_velocity_control,
    )
    for episode in benchmark_episodes:
        if (
            episode.spec != owned.training.spec
            or episode.context == owned.training.context
            or episode.episode_id in (owned.training.episode_id, owned.evaluation.episode_id)
        ):
            raise ValueError("probe spec must match benchmark; context/identities must be distinct")
    return owned


def _score_view(episode):
    # Deliberately NOT continuous_training_pairs: probe_test labels can enter
    # only this read-only score view, never a training-role conversion.
    return (
        np.asarray([row.observation for row in episode.transitions], dtype=np.float64),
        np.asarray([row.action for row in episode.transitions], dtype=np.float64),
    )


class _RecordedProbeBP:
    """Keep actual native BP returns before the app's later receipt checks."""

    def __init__(self, learner, reports, limit):
        self.learner, self.reports, self.limit = learner, reports, limit

    def fork(self):
        return _RecordedProbeBP(self.learner.fork(), self.reports, self.limit)

    def snapshot_state(self):
        return self.learner.snapshot_state()

    def restore_state(self, state):
        self.learner.restore_state(state)

    def predict(self, features):
        return self.learner.predict(features)

    def train_batch(self, features, targets):
        if len(self.reports) >= self.limit:
            raise ValueError("continuous BP probe report capacity exhausted")
        diagnostic = self.learner.train_batch(features, targets)
        self.reports.append(
            dict(diagnostic=asdict(diagnostic), training_rows=len(features), work=None)
        )
        return diagnostic


def _score(report, readout, state, episodes, boundary):
    readout.restore_state(state)
    point = dict(updates=state.completed_updates, complete=False, scores={})
    report["checkpoints"].append(point)
    for name, episode in episodes.items():
        boundary()
        features, targets = _score_view(episode)
        report["work"]["prediction_attempts"] += 1
        report["work"]["prediction_rows_attempted"] += len(features)
        report["work"]["score_rows_attempted_by_role"][episode.role] += len(features)
        prediction = readout.predict(features)
        report["work"]["prediction_returns"] += 1
        report["work"]["prediction_rows_returned"] += len(features)
        report["work"]["score_rows_returned_by_role"][episode.role] += len(features)
        if type(prediction) is np.ndarray:
            report["work"]["prediction_numeric_bytes_returned"] += prediction.nbytes
        else:
            report["work"]["prediction_bytes_unknown_returns"] += 1
        result = diagnostics.continuous_bc_residuals(
            prediction, targets, episode.spec, ("all",) * len(features)
        )
        point["scores"][name] = dict(
            episode_id=episode.episode_id, role=episode.role, residual=asdict(result)
        )
        if not same_continuous_state(state, readout.snapshot_state()):
            raise RuntimeError("continuous probe scoring changed readout")
        boundary()
    point["complete"] = True


def run_continuous_probe(
    source, plan, *, training_readouts=(), check_boundary=None, endpoint_observer=None
):
    """Default single probe, or an explicitly paired disposable velocity control."""
    if endpoint_observer is not None and not callable(endpoint_observer):
        raise ValueError("endpoint_observer must be callable or None")
    if type(plan) is ContinuousProbePlan and plan.zero_velocity_control is True:
        if endpoint_observer is not None:
            raise ValueError("endpoint observer requires a single disposable probe")
        owned = copy_probe_plan(plan)
        return run_zero_velocity_pair(
            source,
            replace(owned, zero_velocity_control=False),
            training_readouts=training_readouts,
            check_boundary=check_boundary,
            run_probe=_run_single_probe,
        )
    options = {} if endpoint_observer is None else dict(endpoint_observer=endpoint_observer)
    return _run_single_probe(
        source, plan, training_readouts=training_readouts, check_boundary=check_boundary, **options
    )


def _run_single_probe(
    source, plan, *, training_readouts=(), check_boundary=None, endpoint_observer=None
):
    """Use one fresh, prospectively authorized disposable probe scope.

    Native global counts continue the checkpoint; local runtime charges start0.
    The caller must supply released arrays/episodes, never final data. Returned
    partial work is retained even if training, scoring or an owner check fails.
    """
    started_probe = monotonic()
    if type(source) not in (NumpyContinuousBCLearner, NumpyContinuousPCLearner):
        raise ValueError("continuous probe requires an owned continuous learner")
    # Why: opaque score/snapshot callbacks cannot redefine the original source
    # before the first endpoint guard is constructed.
    native_before: Any = (
        _native_state(source, type(source)) if endpoint_observer is not None else None
    )
    source_owners = (source._state, source._policy) if endpoint_observer is not None else ()
    # Why: the acquired third TRAIN context must remain visible during a private
    # fresh-task probe; keep the readout count finite without changing its updates.
    if type(training_readouts) is not tuple or len(training_readouts) > 3:
        raise ValueError("continuous probe permits at most three TRAIN readouts")
    readouts = tuple(copy_continuous_episode(item) for item in training_readouts)
    if any(item.role != "train" for item in readouts):
        raise ValueError("old task readouts require TRAIN episodes")
    owned = copy_probe_plan(plan, readouts)
    if owned is None:
        raise ValueError("continuous probe requires a plan")
    if check_boundary is not None and not callable(check_boundary):
        raise ValueError("check_boundary must be callable or None")
    before = source.snapshot_state()
    if endpoint_observer is not None:
        _native_state(source, type(source), native_before)
        if not same_continuous_state(before, native_before):
            raise RuntimeError("probe source observation differs from original native state")
    base = before.policy.base if isinstance(source, NumpyContinuousPCLearner) else before.policy
    if base.spec != owned.training.spec:
        raise ValueError("continuous probe spec differs from checkpoint")
    features, targets = (
        np.asarray(value, dtype=np.float64)
        for value in continuous_training_pairs(owned.training, role="probe_train")
    )
    task._binary_gripper_targets(targets, base.spec, base.binary_gripper_axis)
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=owned.updates), monotonic)
    limits, timer, started = budget.budget, budget.clock, budget.started_at
    original_budget_binding = budget_binding(budget) if endpoint_observer is not None else None
    runtime = None
    readout = None
    runtime_owners: Any = None
    runtime_actor_state = runtime_candidate_state = None
    expected, paid, tick, last_clock = before, 0, 0, budget.last_clock
    report: dict[str, Any] = dict(
        complete=False,
        definition="continuous_disposable_probe_v1",
        source_updates=before.completed_updates,
        source_training_examples=before.training_examples,
        plan=dict(
            updates=owned.updates,
            training_id=owned.training.episode_id,
            evaluation_id=owned.evaluation.episode_id,
            context=owned.training.context,
        ),
        native_updates=[],
        events=[],
        checkpoints=[],
        applied_updates=[],
        work=dict(
            prediction_attempts=0,
            prediction_returns=0,
            prediction_rows_attempted=0,
            prediction_rows_returned=0,
            prediction_numeric_bytes_returned=0,
            prediction_bytes_unknown_returns=0,
            training_attempts=0,
            training_rows_attempted=0,
            score_rows_attempted_by_role=dict(probe_test=0, train=0),
            score_rows_returned_by_role=dict(probe_test=0, train=0),
        ),
    )

    if endpoint_observer is not None:
        report["endpoint_observations"] = []

    if owned.diagnostics:
        report["plan"]["diagnostics"] = True
        report["update_diagnostics"] = []
        for key in ("score_rows_attempted_by_role", "score_rows_returned_by_role"):
            report["work"][key]["probe_train"] = 0

    def check_originals():
        _native_state(source, type(source), native_before)
        if (
            any(
                actual is not original
                for actual, original in zip(
                    (source._state, source._policy), source_owners, strict=True
                )
            )
            or budget_binding(budget) != original_budget_binding
        ):
            raise RuntimeError("probe replaced original source or budget ownership")
        if runtime is not None:
            ports = _probe_runtime_owners(runtime, type(source))
            if runtime_owners is None or any(
                actual is not original
                for actual, original in zip(ports, runtime_owners, strict=True)
            ):
                raise RuntimeError("probe replaced original runtime ownership")
            actor_native, candidate_native = ports[-2:]
            _native_state(actor_native, type(source), native_before)
            _native_state(candidate_native, type(source), expected)
            if (
                actor_native._state is not runtime_actor_state
                or candidate_native._state is not runtime_candidate_state
            ):
                raise RuntimeError("probe replaced an unobserved native runtime state")

    def boundary():
        nonlocal last_clock
        if check_boundary is not None:
            check_boundary()
        if not same_continuous_state(before, source.snapshot_state()):
            raise RuntimeError("continuous probe changed checkpoint source")
        last_clock = _require_budget(
            budget, limits, timer, started, owned.updates, paid, clock, tick, last_clock, runtime
        )
        if endpoint_observer is not None:
            check_originals()
        if runtime is not None:
            actor, candidate = runtime.actor.snapshot_state(), runtime.candidate_snapshot()
            if (
                actor.actor_version != runtime.actor.version
                or actor.actor_version != "probe-actor"
                or candidate.actor_version != "probe-actor"
                or candidate.learner_version != "probe-candidate"
                or type(candidate.wake_updates) is not int
                or candidate.wake_updates != paid
                or candidate.consolidation_attempts != 0
                or candidate.consolidations_completed != 0
                or not same_continuous_state(before, actor.state)
                or not same_continuous_state(expected, candidate.state)
            ):
                raise RuntimeError("continuous probe actor/candidate changed")
        if not same_continuous_state(before, source.snapshot_state()):
            raise RuntimeError("continuous probe changed checkpoint source after owner observation")
        if check_boundary is not None:
            check_boundary()
        last_clock = _require_budget(
            budget, limits, timer, started, owned.updates, paid, clock, tick, last_clock, runtime
        )
        if endpoint_observer is not None:
            check_originals()

    try:
        boundary()
        native_source = (
            RecordedContinuousPC(source, report["native_updates"], owned.updates)
            if isinstance(source, NumpyContinuousPCLearner)
            else _RecordedProbeBP(source, report["native_updates"], owned.updates)
        )
        runtime = ActorShadowRuntime(
            native_source,
            actor_version="probe-actor",
            candidate_version="probe-candidate",
            clock=clock,
            budget=budget,
            max_experiences=owned.updates,
            max_consolidations=0,
            training_role="probe_train",
        )
        if endpoint_observer is not None:
            runtime_owners = _probe_runtime_owners(runtime, type(source))
            runtime_actor_state = runtime_owners[-2]._state
            runtime_candidate_state = runtime_owners[-1]._state
        readout = source.fork()
        if endpoint_observer is not None and (
            type(readout) is not type(source) or readout is source
        ):
            raise RuntimeError("probe endpoint requires a detached exact readout")
        episodes = dict(
            fresh=owned.evaluation, **{f"old_{index}": item for index, item in enumerate(readouts)}
        )
        if owned.diagnostics:
            episodes["fresh_training"] = owned.training
        boundary()
        _score(report, readout, expected, episodes, boundary)
        if endpoint_observer is not None:
            _observe_probe_endpoint(
                endpoint_observer,
                readout,
                "before_probe",
                report,
                source=source,
                state=expected,
                boundary=boundary,
                budget=budget,
                limits=limits,
                paid=paid,
                original_budget_binding=original_budget_binding,
                source_state=native_before,
                check_originals=check_originals,
            )
        for index in range(owned.updates):
            boundary()
            tick = 2 * index
            clock.advance_to(tick)
            identity = f"probe-batch-{index}"
            event: dict[str, Any] = dict(index=index, deferred_exact=False, success_validated=False)
            report["events"].append(event)
            runtime.record_experience(
                Experience(
                    identity,
                    owned.training.episode_id,
                    tick,
                    "probe-actor",
                    features,
                    "probe_train",
                    ExperiencePermissions(training=True),
                )
            )
            if runtime.train_ready(max_updates=1) != ():
                raise RuntimeError("continuous probe updated before label arrival")
            boundary()
            event["deferred_exact"] = True
            tick += 1
            clock.advance_to(tick)
            runtime.record_label(
                LabelArrival(
                    f"probe-label-{index}",
                    identity,
                    owned.training.episode_id,
                    tick,
                    "probe-actor",
                    targets,
                    role="probe_train",
                )
            )
            boundary()
            report["work"]["training_attempts"] += 1
            report["work"]["training_rows_attempted"] += len(features)
            before_update = expected
            receipts = runtime.train_ready(max_updates=1)
            event["returned_receipts"] = [asdict(item) for item in receipts]
            state = runtime.candidate_snapshot().state
            report["applied_updates"] = [asdict(item) for item in runtime.applied_updates]
            paid += 1
            if (
                len(receipts) != 1
                or state.policy != before.policy
                or state.completed_updates != before.completed_updates + paid
                or state.training_examples != before.training_examples + paid * len(features)
                or receipts[0].update_number != paid
                or receipts[0].sample_id != identity
                or receipts[0].episode_id != owned.training.episode_id
                or receipts[0].event_id != f"probe-label-{index}"
                or receipts[0].observed_at != tick - 1
                or receipts[0].arrived_at != tick
                or receipts[0].applied_at != tick
                or receipts[0].actor_version != "probe-actor"
                or receipts[0].learner_version != "probe-candidate"
            ):
                raise RuntimeError("continuous probe receipt/state differs from fixed sequence")
            if any(
                type(value) is not int
                for value in (
                    receipts[0].update_number,
                    receipts[0].observed_at,
                    receipts[0].arrived_at,
                    receipts[0].applied_at,
                )
            ):
                raise RuntimeError("continuous probe receipt counts must be native integers")
            diagnostic = receipts[0].diagnostic
            objective = (
                CONTINUOUS_PC_ENERGY_ID
                if isinstance(source, NumpyContinuousPCLearner)
                else task.OBJECTIVE
                if base.binary_gripper_axis is None
                else task.MIXED_OBJECTIVE
            )
            if (
                type(diagnostic) is not TrainingDiagnostic
                or diagnostic.definition != objective
                or type(diagnostic.value) not in (int, float)
                or not math.isfinite(diagnostic.value)
                or len(report["native_updates"]) != paid
                or report["native_updates"][-1]["diagnostic"] != asdict(diagnostic)
                or report["native_updates"][-1]["training_rows"] != len(features)
                or (
                    isinstance(source, NumpyContinuousPCLearner)
                    and report["native_updates"][-1]["work"]["published_updates"] != 1
                )
            ):
                raise RuntimeError("continuous probe native return differs from receipt")
            expected = state
            if endpoint_observer is not None:
                _native_state(runtime_owners[-1], type(source), expected)
                runtime_candidate_state = runtime_owners[-1]._state
            boundary()
            event["success_validated"] = True
            if owned.diagnostics:
                _diagnose_update(report, paid, features, targets, before_update, state, base)
                boundary()
        _score(report, readout, expected, episodes, boundary)
        if runtime.train_ready(max_updates=1) != ():
            raise RuntimeError("continuous probe has unexpected remaining work")
        boundary()
        if not same_continuous_state(expected, readout.snapshot_state()):
            raise RuntimeError("continuous probe final readout changed")
        first, final = (point["scores"] for point in report["checkpoints"])
        score = lambda point, name: point[name]["residual"]["overall"]["normalized_half_mse"]
        report["metrics"] = dict(
            fresh_acquisition_gain=score(first, "fresh") - score(final, "fresh"),
            old_retention_loss={
                name: score(final, name) - score(first, name)
                for name in first
                if name.startswith("old_")
            },
        )
        if owned.diagnostics:
            report["metrics"]["fresh_training_gain"] = score(first, "fresh_training") - score(
                final, "fresh_training"
            )
        report["complete"] = True
    except Exception as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}")
        if hasattr(error, "work"):
            report["failed_native_work"] = error.work
    report["receipt_observation_complete"] = True
    if runtime is not None:
        try:
            report["applied_updates"] = [asdict(item) for item in runtime.applied_updates]
        except Exception as error:
            report["complete"] = False
            report.pop("metrics", None)
            report["receipt_observation_complete"] = False
            report["receipt_read_error"] = f"{type(error).__name__}: {error}"
            report.setdefault("error", "continuous probe receipt observation failed")
    if report["complete"]:
        try:
            returned = [item for event in report["events"] for item in event["returned_receipts"]]
            if len(report["applied_updates"]) != paid or json.dumps(
                report["applied_updates"], sort_keys=True, allow_nan=False
            ) != json.dumps(returned, sort_keys=True, allow_nan=False):
                raise RuntimeError("continuous probe retained receipt sequence changed")
            boundary()
            if readout is None or not same_continuous_state(expected, readout.snapshot_state()):
                raise RuntimeError("continuous probe final receipt observation changed readout")
        except Exception as error:
            report.update(complete=False, error=f"{type(error).__name__}: {error}")
            report.pop("metrics", None)
    if report["complete"] and endpoint_observer is not None:
        try:
            _observe_probe_endpoint(
                endpoint_observer,
                readout,
                "after_probe",
                report,
                source=source,
                state=expected,
                boundary=boundary,
                budget=budget,
                limits=limits,
                paid=paid,
                original_budget_binding=original_budget_binding,
                source_state=native_before,
                check_originals=check_originals,
            )
        except Exception as error:
            report.update(complete=False, error=f"{type(error).__name__}: {error}")
    native_count = lambda value: value if type(value) is int and 0 <= value <= 2**53 - 1 else None
    if endpoint_observer is not None:
        _summarize_work(report, source, base, before, owned, readouts)
        final_guard, trusted = paid_report_guard(report)
        readout_owners: Any = None if readout is None else (readout._state, readout._policy)
        observed_tick = elapsed = None
        try:
            final_witness = _BudgetWitness(budget, original_budget_binding, limits, paid)
            observed_tick = native_count(clock.now())
            elapsed = monotonic() - started_probe
            if observed_tick != tick or type(clock._time) is not int or clock._time != tick:
                raise RuntimeError("probe final observation changed original logical time")
            final_guard()
            check_originals()
            if readout is not None:
                _native_state(readout, type(source), expected)
                if any(
                    actual is not original
                    for actual, original in zip(
                        (readout._state, readout._policy), readout_owners, strict=True
                    )
                ):
                    raise RuntimeError("probe final clock observation replaced readout")
            final_witness.check()
            final_guard()
        except Exception as error:
            report.clear()
            report.update(trusted)
            report.update(
                complete=False, final_observation_error=f"{type(error).__name__}: {error}"
            )
        report["progress"] = dict(
            updates_completed=native_count(budget.updates_completed),
            replay_examples_completed=native_count(budget.replay_examples_completed),
            next_tick=None if observed_tick is None else observed_tick + 1,
        )
        report["elapsed_seconds"] = elapsed
        return report
    observed_tick = native_count(clock.now())
    report["progress"] = dict(
        updates_completed=native_count(budget.updates_completed),
        replay_examples_completed=native_count(budget.replay_examples_completed),
        next_tick=None if observed_tick is None else observed_tick + 1,
    )
    _summarize_work(report, source, base, before, owned, readouts)
    report["elapsed_seconds"] = monotonic() - started_probe
    return report


def _observe_probe_endpoint(
    observer,
    readout,
    phase,
    report,
    *,
    source,
    state,
    boundary,
    budget,
    limits,
    paid,
    original_budget_binding,
    source_state,
    check_originals,
):
    """Borrow a private readout synchronously; preserve paid work on refusal.

    The observer owns its separate readouts/costs and must not retain the learner.
    Existing bounded report and budget guards supply the proof; no new runtime.
    """
    report["endpoint_observations"].append(dict(phase=phase, attempted=True, completed=False))
    evidence, trusted = paid_report_guard(report)
    witness = _BudgetWitness(budget, original_budget_binding, limits, paid)
    kind = type(source)
    owners = (source._state, source._policy, readout._state, readout._policy)
    refusal = None

    def check():
        evidence()
        witness.check()
        boundary()
        _observe_state(source, kind, source_state)
        _observe_state(readout, kind, state)
        # Why: public state and parent callbacks are opaque. Finish from original
        # native fields after their last invocation, then prove paid evidence.
        _native_state(source, kind, source_state)
        _native_state(readout, kind, state)
        check_originals()
        if any(
            actual is not original
            for actual, original in zip(
                (source._state, source._policy, readout._state, readout._policy),
                owners,
                strict=True,
            )
        ):
            raise RuntimeError("endpoint observer replaced an original native owner")
        evidence()
        witness.check()

    def guard(read_ports=None):
        nonlocal refusal
        if refusal is not None:
            raise refusal
        try:
            check()
            witness.observe()
            check()
            try:
                if read_ports is not None:
                    if not callable(read_ports) or read_ports() is not None:
                        raise ValueError("endpoint read ports must be callable and return None")
            finally:
                check()
                witness.observe()
                check()
        except Exception as error:
            refusal = error
            raise

    try:
        guard()
        if observer(readout, phase, guard) is not None:
            raise ValueError("endpoint observer must retain its readings and return None")
        guard()
    except Exception:
        report.clear()
        report.update(trusted)
        raise
    report["endpoint_observations"][-1]["completed"] = True


def _probe_runtime_owners(runtime, kind):
    """Bind existing ports and exact native owners, without public observations."""
    wrappers = (runtime._actor._learner, runtime._candidate, runtime._inbox._learner)
    wrapper_kind = _RecordedProbeBP if kind is NumpyContinuousBCLearner else RecordedContinuousPC
    if any(type(port) is not wrapper_kind for port in wrappers) or wrappers[1] is not wrappers[2]:
        raise RuntimeError("probe runtime lost its original recorded ports")
    native = tuple(
        port.learner if kind is NumpyContinuousBCLearner else port._learner for port in wrappers[:2]
    )
    if native[0] is native[1] or any(type(owner) is not kind for owner in native):
        raise RuntimeError("probe runtime requires detached exact native owners")
    reports = tuple(
        port.reports if kind is NumpyContinuousBCLearner else port._reports for port in wrappers
    )
    return (
        runtime,
        runtime._actor,
        runtime._inbox,
        *wrappers,
        *reports,
        *(owner._policy for owner in native),
        *native,
    )


def _diagnose_update(report, paid, features, targets, before, after, base):
    """Save paid-history-independent diagnostic attempts before any extra math."""
    progress = _StepDirectionWork()
    row = dict(update_number=paid, complete=False, work=asdict(progress))
    report["update_diagnostics"].append(row)
    try:
        result = diagnose_step_direction(
            features,
            targets,
            before.parameters,
            before.velocity,
            after.parameters,
            base.spec,
            standardization=base.observation_standardization,
            learning_rate=base.learning_rate,
            momentum=base.momentum,
            _work=progress,
        )
        row["result"] = asdict(result)
        row["complete"] = True
    finally:
        row["work"] = asdict(progress)


def _summarize_work(report, source, base, before, owned, readouts):
    """Keep report arithmetic separate from the stateful fixed sequence."""
    if owned.diagnostics:
        rows = report["update_diagnostics"]
        counts = {
            key: sum(row["work"][key] for row in rows) for key in asdict(_StepDirectionWork())
        }
        gradient_returns = counts["task_gradient_returns"]
        gradient_values = sum(array.size for array in before.parameters) * gradient_returns
        report["work"]["step_diagnostics"] = dict(
            **counts,
            gradient_rows_returned=gradient_returns * len(owned.training.transitions),
            gradient_values_returned=gradient_values,
            gradient_numeric_bytes_returned=gradient_values * 8,
            derived_task_dense_multiply_accumulates=gradient_returns
            * len(owned.training.transitions)
            * (
                2 * base.spec.observation_dim * base.hidden_dim
                + 3 * base.hidden_dim * len(base.spec.action_units)
            ),
        )
    native_work = [item for item in report["native_updates"] if item["work"] is not None]
    report["work"].update(
        native_update_counts={
            key: sum(item["work"][key] for item in native_work) for key in native_work[0]["work"]
        }
        if native_work
        else {},
        native_event_row_weights={
            key: sum(item["work"][key] * item["training_rows"] for item in native_work)
            for key in native_work[0]["work"]
        }
        if native_work
        else {},
        native_returns=len(report["native_updates"]),
        training_rows_native_returned=sum(
            item["training_rows"] for item in report["native_updates"]
        ),
        returned_prediction_normalizations=report["work"]["prediction_returns"],
        derived_returned_prediction_dense_multiply_accumulates=report["work"][
            "prediction_rows_returned"
        ]
        * (
            base.spec.observation_dim * base.hidden_dim
            + base.hidden_dim * len(base.spec.action_units)
        ),
        derived_bp_native_forward_backward_returns=len(report["native_updates"])
        if isinstance(source, NumpyContinuousBCLearner)
        else 0,
        training_rows_completed=len(report["applied_updates"]) * len(owned.training.transitions)
        if report["receipt_observation_complete"]
        else None,
        reserved_numeric_bytes=(len(owned.training.transitions) + len(owned.evaluation.transitions))
        * (base.spec.observation_dim + len(base.spec.action_units))
        * 8,
        old_readout_input_numeric_bytes=sum(len(item.transitions) for item in readouts)
        * (base.spec.observation_dim + len(base.spec.action_units))
        * 8,
        derived_bp_training_dense_multiply_accumulates=sum(
            item["training_rows"] for item in report["native_updates"]
        )
        * (
            2 * base.spec.observation_dim * base.hidden_dim
            + 3 * base.hidden_dim * len(base.spec.action_units)
        )
        if isinstance(source, NumpyContinuousBCLearner)
        else 0,
        extra_model_owners=3,
        parameter_momentum_numeric_bytes_per_owner=sum(
            array.nbytes for array in before.parameters + before.velocity
        ),
        wake_numeric_bytes_per_owner=0
        if not isinstance(source, NumpyContinuousPCLearner) or before.wake_state is None
        else before.wake_state.chemical_state.nbytes + before.wake_state.plasticity_state.nbytes,
        scope="actual returned native reports/receipts and prediction events; extra owners are transient per probe; excludes snapshot/gate validation, copies, elementwise, metadata, temporaries and process RSS; BP native arithmetic derived from receipt counts, not PC formulas",
    )


def _require_budget(
    budget, limits, timer, started, updates, paid, clock, tick, last_clock, runtime
):
    """Check the original local scope before and after native owner observations."""
    replace(limits)  # Do not let equal-looking bool/float limit values renew authority.
    if (
        budget.budget is not limits
        or budget.budget != ToyExecutionBudget(max_training_updates=updates)
        or budget.clock is not timer
        or budget.started_at != started
        or type(budget.updates_completed) is not int
        or budget.updates_completed != paid
        or type(budget.replay_examples_completed) is not int
        or budget.replay_examples_completed != 0
        or type(budget.started_at) not in (int, float)
        or not math.isfinite(budget.started_at)
        or type(budget.last_clock) not in (int, float)
        or not math.isfinite(budget.last_clock)
        or budget.last_clock < last_clock
        or type(clock.now()) is not int
        or clock.now() != tick
        or (
            runtime is not None
            and (
                runtime._budget is not budget
                or runtime._inbox._budget is not budget
                or runtime._inbox._clock is not clock
            )
        )
    ):
        raise RuntimeError("continuous probe original budget/clock changed")
    last_clock = budget.last_clock
    return budget.last_clock
