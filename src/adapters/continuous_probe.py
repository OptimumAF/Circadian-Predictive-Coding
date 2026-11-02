"""Disposable continuous probes using the existing delayed-label runtime.

Inputs are owned reserved-role episodes and a released native checkpoint. Outputs
are bounded physical scores, receipts and costs, never an adapted learner. No source fit,
data selection, final evaluation, promotion or classification coercion. Explicit
old-TRAIN joint rehearsal is confined to the opt-in private full-network route.
"""

from dataclasses import asdict, dataclass, replace
from copy import deepcopy
from collections.abc import Callable
import math
import json
from time import monotonic
from typing import Any

import numpy as np

from src.adapters.continuous_learnability_modes import (
    RecordedContinuousPC,
    same_continuous_state as _same_neural_state,
)
from src.adapters.continuous_probe_control import (
    _native_state as _native_neural_state,
    _observe_state as _observe_neural_state,
    run_zero_velocity_pair,
)
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner
from src.adapters.continuous_probe_fit import _FullProbeBinding
from src.adapters.continuous_probe_rehearsal import _ProbeRehearsalBinding
from src.adapters.numpy_continuous_pc import NumpyContinuousPCLearner
from src.adapters import continuous_nearest_probe as nearest_probe
from src.adapters import continuous_retained_training as retained_training
from src.adapters.numpy_nearest_continuous_bc import (
    NumpyNearestContinuousBCLearner,
    NearestContinuousBCSnapshot,
    OBJECTIVE as NEAREST_OBJECTIVE,
)
from src.adapters.numpy_retained_nearest_bc import (
    NumpyRetainedNearestBCLearner,
    RetainedNearestSnapshot,
    advance_retained,
    OBJECTIVE as RETAINED_OBJECTIVE,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.app.probe_endpoint_observer import _BudgetWitness
from src.app.probe_return_phase import budget_binding, paid_report_guard
from src.core import continuous_bc_diagnostics as diagnostics
from src.core import continuous_bc as task
from src.core.continuous_pc import CONTINUOUS_PC_ENERGY_ID
from src.core.continuous_step_diagnostics import _StepDirectionWork, diagnose_step_direction
from src.core.learner_ports import TrainingDiagnostic
from src.core.pick_cube_probe_corrections import (
    ProbeCorrectionBatch,
    PROVENANCE as CORRECTION_PROVENANCE,
)
from src.core.continuous_demonstration import (
    ContinuousEpisode,
    continuous_training_pairs,
    copy_continuous_episode,
)
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock


def same_continuous_state(first, second):
    if type(first) in (NearestContinuousBCSnapshot, RetainedNearestSnapshot) or type(second) in (
        NearestContinuousBCSnapshot,
        RetainedNearestSnapshot,
    ):
        return nearest_probe.same_state(first, second)
    return _same_neural_state(first, second)


def _native_state(owner, kind, expected=None):
    if kind in (NumpyNearestContinuousBCLearner, NumpyRetainedNearestBCLearner):
        return nearest_probe.native_state(owner, expected)
    return _native_neural_state(owner, kind, expected)


def _observe_state(owner, kind, expected=None):
    if kind not in (NumpyNearestContinuousBCLearner, NumpyRetainedNearestBCLearner):
        return _observe_neural_state(owner, kind, expected)
    observed, policy = owner.snapshot_state(), owner.policy
    actual = _native_state(owner, kind, expected)
    if observed.policy != policy or not same_continuous_state(observed, actual):
        raise RuntimeError("nearest probe public state differs from original owner")
    return actual


@dataclass(frozen=True)
class ContinuousProbePlan:
    training: ContinuousEpisode
    evaluation: ContinuousEpisode
    updates: int = 1
    diagnostics: bool = False
    zero_velocity_control: bool = False
    optimizer: str = "sgd"

    def __post_init__(self):
        if type(self.optimizer) is not str or self.optimizer not in ("sgd", "full-gauss-newton"):
            raise ValueError("continuous probe optimizer must be sgd or full-gauss-newton")
        if self.optimizer != "sgd" and (self.diagnostics or self.zero_velocity_control):
            raise ValueError("full-network probe excludes SGD-direction/velocity controls")
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
        value.optimizer,
    )
    for episode in benchmark_episodes:
        if (
            episode.spec != owned.training.spec
            or episode.context == owned.training.context
            or episode.episode_id in (owned.training.episode_id, owned.evaluation.episode_id)
        ):
            raise ValueError("probe spec must match benchmark; context/identities must be distinct")
    return owned


@dataclass(frozen=True)
class _NearestTrainingPlan:
    """Explicit single-packet authority; no evaluation object or role conversion."""

    training: ContinuousEpisode
    updates: int = 1
    diagnostics: bool = False
    evaluation: None = None

    def __post_init__(self):
        if (
            type(self.training) is not ContinuousEpisode
            or self.training.role != "probe_train"
            or not 1 <= len(self.training.transitions) <= 32
            or self.training.spec.observation_dim != 12
            or type(self.updates) is not int
            or self.updates != 1
            or self.diagnostics is not False
            or self.evaluation is not None
        ):
            raise ValueError("nearest training diagnostic requires1..32probe_train TS12 rows")
        object.__setattr__(self, "training", copy_continuous_episode(self.training))


def run_nearest_training_probe(source, training, *, check_boundary=None, endpoint_observer=None):
    """One actual delayed private append and TRAIN-only readouts, never transfer scores."""
    plan = _NearestTrainingPlan(training)
    if type(source) is not NumpyNearestContinuousBCLearner:
        raise ValueError("training-only probe requires exact fixed-input nearest")
    if endpoint_observer is not None and not callable(endpoint_observer):
        raise ValueError("endpoint_observer must be callable or None")
    options = {} if endpoint_observer is None else dict(endpoint_observer=endpoint_observer)
    return _run_single_probe(source, plan, check_boundary=check_boundary, **options)


def run_retained_nearest_training_probe(
    source, training, *, training_readouts=(), check_boundary=None, endpoint_observer=None
):
    """One isolated retained append and TRAIN fit/coverage, with no evaluation packet."""
    plan = _NearestTrainingPlan(training)
    if type(source) is not NumpyRetainedNearestBCLearner:
        raise ValueError("retained training-only probe requires exact bounded nearest")
    if endpoint_observer is not None and not callable(endpoint_observer):
        raise ValueError("endpoint_observer must be callable or None")
    return _run_single_probe(
        source,
        plan,
        training_readouts=training_readouts,
        check_boundary=check_boundary,
        endpoint_observer=endpoint_observer,
        retained_nearest=True,
    )


def run_retained_nearest_probe(
    source,
    plan,
    *,
    training_readouts=(),
    check_boundary=None,
    endpoint_observer=None,
    return_training=None,
):
    """One isolated bounded-memory append; fresh/old scores never train or promote.

    Why explicit: the old append-only probe's capacity and index-origin rules
    cannot admit retained state by subclassing. This route owns no data loading.
    """
    if (
        type(source) is not NumpyRetainedNearestBCLearner
        or type(plan) is not ContinuousProbePlan
        or plan.updates != 1
        or plan.diagnostics
        or plan.zero_velocity_control
        or plan.optimizer != "sgd"
    ):
        raise ValueError("retained probe requires exact bounded source and one plain append")
    if endpoint_observer is not None and not callable(endpoint_observer):
        raise ValueError("endpoint_observer must be callable or None")
    if return_training is not None:
        from src.adapters.continuous_nearest_return import run_return_probe

        return run_return_probe(
            source,
            plan,
            return_training,
            training_readouts=training_readouts,
            check_boundary=check_boundary,
            endpoint_observer=endpoint_observer,
        )
    return _run_single_probe(
        source,
        plan,
        training_readouts=training_readouts,
        check_boundary=check_boundary,
        endpoint_observer=endpoint_observer,
        retained_nearest=True,
    )


@dataclass(frozen=True)
class _RetainedReturnPlan:
    """One original old TRAIN return; fresh reserved data stays score-only."""

    training: ContinuousEpisode
    evaluation: ContinuousEpisode
    fresh_training: ContinuousEpisode
    updates: int = 1
    diagnostics: bool = False
    zero_velocity_control: bool = False
    optimizer: str = "sgd"

    def __post_init__(self):
        fresh = ContinuousProbePlan(self.fresh_training, self.evaluation)
        if (
            type(self.training) is not ContinuousEpisode
            or self.training.role != "train"
            or not 1 <= len(self.training.transitions) <= 50
            or self.training.spec != fresh.training.spec
            or self.training.context == fresh.training.context
            or self.training.episode_id in (fresh.training.episode_id, fresh.evaluation.episode_id)
            or type(self.updates) is not int
            or self.updates != 1
            or self.diagnostics is not False
            or self.zero_velocity_control is not False
            or self.optimizer != "sgd"
        ):
            raise ValueError("retained return requires one original old TRAIN packet")
        object.__setattr__(self, "training", copy_continuous_episode(self.training))
        object.__setattr__(self, "evaluation", fresh.evaluation)
        object.__setattr__(self, "fresh_training", fresh.training)


def _score_view(episode):
    # Deliberately NOT continuous_training_pairs: probe_test labels can enter
    # only this read-only score view, never a training-role conversion.
    return (
        np.asarray([row.observation for row in episode.transitions], dtype=np.float64),
        np.asarray([row.action for row in episode.transitions], dtype=np.float64),
    )


class _RecordedProbeBP:
    """Keep actual native BP returns before the app's later receipt checks."""

    def __init__(self, learner, reports, limit, fit=None, *, protect=False):
        self.learner, self.reports, self.limit = learner, reports, limit
        self.fit, self.failed = fit, False
        self.protected = fit is not None or protect
        self._full_reports, self._full_guards = reports, []

    def require_fit_work(self, *, writer=False, force=False):
        if not (self.protected or force):
            return
        failure: Exception | None = None
        if self.reports is not self._full_reports or (
            writer and len(self.reports) != len(self._full_guards)
        ):
            failure = RuntimeError("full-network probe replaced paid report history")
        for index, row, guard, trusted in self._full_guards:
            try:
                if len(self.reports) <= index or self.reports[index] is not row:
                    raise RuntimeError("full-network probe replaced paid update record")
                guard()
            except Exception as error:
                failure = error
        if failure is not None:
            # Retain EVERY paid return for capture, then permanently refuse;
            # repairing the displayed history grants no renewed authority.
            for index, row, guard, trusted in self._full_guards:
                row.clear()
                row.update(deepcopy(trusted))
            self._full_reports[:] = [row for _, row, _, _ in self._full_guards]
            self.failed = True
            raise failure

    def fork(self):
        if self.protected and self.failed:
            raise RuntimeError("full-network probe port permanently failed")
        self.require_fit_work()
        return _RecordedProbeBP(
            self.learner.fork(), self.reports, self.limit, self.fit, protect=self.protected
        )

    def snapshot_state(self):
        return self.learner.snapshot_state()

    def restore_state(self, state):
        self.learner.restore_state(state)

    def predict(self, features):
        return self.learner.predict(features)

    def train_batch(self, features, targets):
        if self.protected:
            if self.failed:
                raise RuntimeError("full-network probe port permanently failed")
            try:
                self.require_fit_work(writer=True)
                if len(self.reports) >= self.limit:
                    raise ValueError("continuous BP probe report capacity exhausted")
                if self.fit is not None:
                    training_rows = (
                        self.fit.training_rows
                        if type(self.fit) is _ProbeRehearsalBinding
                        else len(features)
                    )
                    result = self.fit.update(self.learner, features, targets)
                    diagnostic = result.diagnostic
                    row = dict(
                        diagnostic=asdict(result.diagnostic),
                        training_rows=training_rows,
                        work=dict(result.work),
                        loss_after=result.loss_after,
                        step_norm=result.step_norm,
                        momentum_reset=True,
                    )
                else:
                    diagnostic = self.learner.train_batch(features, targets)
                    row = dict(
                        diagnostic=asdict(diagnostic), training_rows=len(features), work=None
                    )
                self.reports.append(row)
                row = self.reports[-1]
                guard, trusted = paid_report_guard(row)
                self._full_guards.append((len(self.reports) - 1, row, guard, trusted))
                return diagnostic
            except Exception:
                self.failed = True
                raise
        if len(self.reports) >= self.limit:
            raise ValueError("continuous BP probe report capacity exhausted")
        nearest = nearest_probe.is_nearest(self.learner)
        before = nearest_probe.native_state(self.learner) if nearest else None
        diagnostic = self.learner.train_batch(features, targets)
        self.reports.append(
            dict(diagnostic=asdict(diagnostic), training_rows=len(features), work=None)
        )
        if nearest:
            self.reports[-1]["work"] = nearest_probe.append_work(
                before,
                len(features),
                **(
                    dict(after=nearest_probe.native_state(self.learner))
                    if type(self.learner) is NumpyRetainedNearestBCLearner
                    else {}
                ),
            )
        return diagnostic


def _score(
    report,
    readout,
    state,
    episodes,
    boundary,
    *,
    _work=None,
    _publish_work=None,
    _source_rows=None,
    _retained_content=None,
):
    readout.restore_state(state)
    identity_fn = (
        nearest_probe.owner_identity
        if _source_rows is None and _retained_content is None
        else nearest_probe.training_owner_identity
    )
    identity = identity_fn(readout)
    point = dict(updates=state.completed_updates, complete=False, scores={})
    report["checkpoints"].append(point)
    points, position = report["checkpoints"], len(report["checkpoints"]) - 1
    work = report["work"] if _work is None else _work

    def observe(callback):
        if _work is None:
            return callback()
        guard, trusted = paid_report_guard(point)
        try:
            result = callback()
            guard()
            if report["checkpoints"] is not points or points[position] is not point:
                raise RuntimeError("nearest probe replaced a paid score holder")
            return result
        except Exception:
            point.clear()
            point.update(trusted)
            report["checkpoints"] = points
            points[position] = point
            raise

    def publish_work():
        if _publish_work is not None:
            _publish_work()

    for name, episode in episodes.items():
        observe(boundary)
        features, targets = _score_view(episode)
        work["prediction_attempts"] += 1
        work["prediction_rows_attempted"] += len(features)
        work["score_rows_attempted_by_role"][episode.role] += len(features)
        publish_work()
        selection: list[int] | None = (
            [] if _source_rows is not None or _retained_content is not None else None
        )
        selection_options = {} if selection is None else dict(selection=selection)
        try:
            prediction = (
                observe(
                    lambda: nearest_probe.predict(
                        readout, state, features, work, **selection_options
                    )
                )
                if _work is not None and nearest_probe.is_nearest(readout)
                else observe(lambda: readout.predict(features))
                if _work is not None
                else readout.predict(features)
            )
        finally:
            publish_work()
        work["prediction_returns"] += 1
        work["prediction_rows_returned"] += len(features)
        work["score_rows_returned_by_role"][episode.role] += len(features)
        if type(prediction) is np.ndarray:
            work["prediction_numeric_bytes_returned"] += prediction.nbytes
        else:
            work["prediction_bytes_unknown_returns"] += 1
        publish_work()
        result = diagnostics.continuous_bc_residuals(
            prediction, targets, episode.spec, ("all",) * len(features)
        )
        point["scores"][name] = dict(
            episode_id=episode.episode_id, role=episode.role, residual=asdict(result)
        )
        if _retained_content is not None:
            point["scores"][name].update(
                observe(
                    lambda: retained_training.describe_selection(
                        state, _retained_content, features, targets, selection, work, publish_work
                    )
                )
            )
        elif selection is not None:
            point["scores"][name]["selected"] = [
                dict(
                    index=i,
                    origin="source" if i < _source_rows else "probe_train",
                    row=i if i < _source_rows else i - _source_rows,
                )
                for i in selection
            ]
        if not same_continuous_state(state, observe(readout.snapshot_state)):
            raise RuntimeError("continuous probe scoring changed readout")
        if nearest_probe.is_nearest(readout) and any(
            a is not b for a, b in zip(identity_fn(readout), identity, strict=True)
        ):
            raise RuntimeError("nearest probe scoring replaced original readout arrays")
        observe(boundary)
    point["complete"] = True


def run_continuous_probe(
    source,
    plan,
    *,
    training_readouts=(),
    check_boundary=None,
    endpoint_observer=None,
    correction_batch=None,
    correction_release_tick=None,
    replay_plan=None,
    replay_episodes=(),
    replay_budget=None,
):
    """Disposable updates with default SGD or an explicit BP fitting control."""
    if endpoint_observer is not None and not callable(endpoint_observer):
        raise ValueError("endpoint_observer must be callable or None")
    rehearsal = replay_plan is not None or replay_budget is not None or replay_episodes != ()
    if rehearsal and (
        type(source) is not NumpyContinuousBCLearner
        or type(plan) is not ContinuousProbePlan
        or plan.optimizer != "full-gauss-newton"
        or plan.diagnostics
        or plan.zero_velocity_control
        or correction_batch is not None
        or correction_release_tick is not None
    ):
        raise ValueError("joint rehearsal requires exact BP and a plain full-network probe")
    if correction_batch is not None or correction_release_tick is not None:
        if (
            type(correction_batch) is not ProbeCorrectionBatch
            or type(correction_release_tick) is not int
            or type(plan) is not ContinuousProbePlan
            or plan.diagnostics
            or plan.zero_velocity_control
            or type(source) is not NumpyContinuousBCLearner
        ):
            raise ValueError(
                "corrective probe requires exact BP, arrived batch and single non-diagnostic plan"
            )
    if (
        type(plan) is ContinuousProbePlan
        and plan.optimizer != "sgd"
        and type(source) is not NumpyContinuousBCLearner
    ):
        raise ValueError("full-network probe optimizer requires exact BP")
    if type(source) is NumpyNearestContinuousBCLearner and (
        type(plan) is not ContinuousProbePlan
        or plan.updates != 1
        or plan.diagnostics
        or plan.zero_velocity_control
    ):
        raise ValueError("nearest probe requires one append without gradient/velocity options")
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
    if correction_batch is not None:
        options.update(
            correction_batch=correction_batch, correction_release_tick=correction_release_tick
        )
    if rehearsal:
        options.update(
            replay_plan=replay_plan, replay_episodes=replay_episodes, replay_budget=replay_budget
        )
    return _run_single_probe(
        source, plan, training_readouts=training_readouts, check_boundary=check_boundary, **options
    )


def _run_single_probe(
    source,
    plan,
    *,
    training_readouts=(),
    check_boundary=None,
    endpoint_observer=None,
    correction_batch=None,
    correction_release_tick=None,
    retained_nearest=False,
    return_scope=None,
    replay_plan=None,
    replay_episodes=(),
    replay_budget=None,
):
    """Use one fresh, prospectively authorized disposable probe scope.

    Native global counts continue the checkpoint; local runtime charges start0.
    The caller must supply released arrays/episodes, never final data. Returned
    partial work is retained even if training, scoring or an owner check fails.
    """
    started_probe = monotonic()
    if type(retained_nearest) is not bool or (
        retained_nearest != (type(source) is NumpyRetainedNearestBCLearner)
    ):
        raise ValueError("retained nearest requires the explicit isolated probe route")
    nearest = type(source) is NumpyNearestContinuousBCLearner or retained_nearest
    return_stage = type(plan) is _RetainedReturnPlan
    if return_stage != (return_scope is not None) or return_stage and not retained_nearest:
        raise ValueError("old TRAIN return requires its separate admitted scope")
    if return_stage and (
        type(return_scope) is not tuple
        or len(return_scope) != 3
        or type(return_scope[0]) is not ToyBudgetSession
        or type(return_scope[1]) is not LogicalClock
        or type(return_scope[2]) is not _BudgetWitness
        or return_scope[2].budget is not return_scope[0]
    ):
        raise ValueError("retained return requires original budget and logical-clock owners")
    if return_stage:
        return_scope[2].check()
    guarded_work = nearest or correction_batch is not None
    full_fit = type(plan) is ContinuousProbePlan and plan.optimizer == "full-gauss-newton"
    if full_fit and type(source) is not NumpyContinuousBCLearner:
        raise ValueError("full-network probe optimizer requires exact BP")
    training_only = type(plan) is _NearestTrainingPlan
    if training_only and not nearest:
        raise ValueError("training-only plan requires exact nearest")
    identity_fn = (
        nearest_probe.training_owner_identity if training_only else nearest_probe.owner_identity
    )
    protect_owners = (
        endpoint_observer is not None or nearest or full_fit or correction_batch is not None
    )
    if not retained_nearest and type(source) not in (
        NumpyContinuousBCLearner,
        NumpyContinuousPCLearner,
        NumpyNearestContinuousBCLearner,
    ):
        raise ValueError("continuous probe requires an owned continuous learner")
    # Why: opaque score/snapshot callbacks cannot redefine the original source
    # before the first endpoint guard is constructed.
    native_before: Any = _native_state(source, type(source)) if protect_owners else None
    source_owners = identity_fn(source) if protect_owners else ()
    # Why: the acquired third TRAIN context must remain visible during a private
    # fresh-task probe; keep the readout count finite without changing its updates.
    maximum_readouts = 4 if retained_nearest else 3
    if type(training_readouts) is not tuple or len(training_readouts) > maximum_readouts:
        raise ValueError(f"continuous probe permits at most {maximum_readouts} TRAIN readouts")
    readouts = tuple(copy_continuous_episode(item) for item in training_readouts)
    if any(item.role != "train" for item in readouts):
        raise ValueError("old task readouts require TRAIN episodes")
    owned = (
        _RetainedReturnPlan(plan.training, plan.evaluation, plan.fresh_training)
        if return_stage
        else _NearestTrainingPlan(plan.training, plan.updates, plan.diagnostics, plan.evaluation)
        if training_only
        else copy_probe_plan(plan, readouts)
    )
    if owned is None:
        raise ValueError("continuous probe requires a plan")
    if training_only and retained_nearest:
        if not nearest_probe.stored_rows(native_before):
            raise ValueError("retained training-only probe requires nonempty source")
        if any(
            item.spec != owned.training.spec or item.episode_id == owned.training.episode_id
            for item in readouts
        ) or len({item.episode_id for item in readouts}) != len(readouts):
            raise ValueError("retained training readouts require matching specs and distinct IDs")
    if return_stage and sum(item == owned.training for item in readouts) != 1:
        raise ValueError("return packet must equal exactly one declared old TRAIN readout")
    corrected_pairs = None
    if correction_batch is not None:
        correction_batch._require_original()
        correction_batch = ProbeCorrectionBatch(
            correction_batch.consent,
            correction_batch.episode,
            correction_batch.labels,
            correction_batch.observed_grasp,
        )
        if owned.training != correction_batch.model_episode():
            raise ValueError(
                "probe training episode differs from actual corrective student history"
            )
        corrected_pairs = correction_batch.training_pairs(now=correction_release_tick)
    fit: Any = (
        (
            _FullProbeBinding(*corrected_pairs)
            if corrected_pairs is not None
            else _FullProbeBinding.from_training(owned.training)
        )
        if full_fit
        else None
    )
    rehearsal = None
    if replay_plan is not None or replay_budget is not None or replay_episodes != ():
        if not full_fit or retained_nearest or return_stage or correction_batch is not None:
            raise ValueError("joint rehearsal requires a plain full-network BP probe")
        rehearsal = _ProbeRehearsalBinding(
            owned.training, owned.evaluation, replay_plan, replay_episodes, replay_budget, readouts
        )
        fit = rehearsal
    if check_boundary is not None and not callable(check_boundary):
        raise ValueError("check_boundary must be callable or None")
    before = source.snapshot_state()
    if protect_owners:
        _native_state(source, type(source), native_before)
        if not same_continuous_state(before, native_before):
            raise RuntimeError("probe source observation differs from original native state")
    base = before.policy.base if isinstance(source, NumpyContinuousPCLearner) else before.policy
    if base.spec != owned.training.spec:
        raise ValueError("continuous probe spec differs from checkpoint")
    features, targets = (
        np.asarray(value, dtype=np.float64)
        for value in (
            corrected_pairs
            if corrected_pairs is not None
            else continuous_training_pairs(
                owned.training, role="train" if return_stage else "probe_train"
            )
        )
    )
    physical_rows = len(features) if rehearsal is None else rehearsal.training_rows
    if nearest:
        if not retained_nearest and before.training_examples + len(features) > base.max_prototypes:
            raise ValueError("nearest probe append exceeds original prototype capacity")
    else:
        task._binary_gripper_targets(targets, base.spec, base.binary_gripper_axis)
    clock = LogicalClock() if return_scope is None else return_scope[1]
    budget = (
        ToyBudgetSession(ToyExecutionBudget(max_training_updates=owned.updates), monotonic)
        if return_scope is None
        else return_scope[0]
    )
    if return_stage and (
        type(clock._time) is not int
        or clock._time != 0
        or type(budget.updates_completed) is not int
        or budget.updates_completed != 0
        or budget.replay_examples_completed != 0
        or budget.budget.max_training_updates != 1
    ):
        raise ValueError("return scope already spent or changed before return admission")
    limits, timer, started = budget.budget, budget.clock, budget.started_at
    if return_stage:
        # Source/plan observations above cannot silently supply a newer baseline.
        return_scope[2].check()
    return_authority_taken = False
    original_budget_binding = budget_binding(budget) if protect_owners else None
    runtime = None
    full_candidate: Any = None
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
            evaluation_id=None if owned.evaluation is None else owned.evaluation.episode_id,
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
    paid_records: list[tuple[str, list[Any], int, Any, Callable[[], None], Any]] = []
    if correction_batch is not None:
        report["plan"]["corrections"] = dict(
            provenance=CORRECTION_PROVENANCE,
            source_episode=correction_batch.episode.episode_id,
            labels_released_at=correction_release_tick,
            query_limit=correction_batch.consent.max_queries,
            student_actions_retained=True,
            teacher_targets_separate=True,
        )
        report["work"]["correction_projection_rows"] = 4 * len(correction_batch.labels)
    if full_fit:
        report["plan"]["optimizer"] = "full-gauss-newton"
        report["work"]["probe_binding_numeric_values"] = features.size + targets.size
    if rehearsal is not None:
        for key in ("score_rows_attempted_by_role", "score_rows_returned_by_role"):
            report["work"][key]["probe_train"] = 0

    def seal_paid(key, index):
        holder: list[Any] = report[key]
        original = holder[index]
        guard, trusted = paid_report_guard(original)
        paid_records.append((key, holder, index, original, guard, trusted))

    if nearest:
        report["definition"] = "continuous_disposable_nearest_probe_v1"
    if retained_nearest:
        report["definition"] = "continuous_disposable_retained_nearest_probe_v1"
        report["classification"] = (
            "isolated fit/interference; native success and return recovery unmeasured"
        )
        for key in ("score_rows_attempted_by_role", "score_rows_returned_by_role"):
            report["work"][key]["probe_train"] = 0
    if isinstance(owned, _RetainedReturnPlan):
        report["definition"] = "continuous_retained_old_train_return_v1"
        report["plan"].update(
            training_role="train", fresh_training_id=owned.fresh_training.episode_id
        )
    if training_only:
        report["definition"] = (
            "continuous_retained_nearest_training_probe_v1"
            if retained_nearest
            else "continuous_nearest_training_probe_v1"
        )
        if retained_nearest:
            report["classification"] = (
                "TRAIN-only fit/interference/content coverage; no evaluation or native outcome"
            )
        report["training_consistency"] = []
        for key in ("score_rows_attempted_by_role", "score_rows_returned_by_role"):
            report["work"][key]["probe_train"] = 0
    # Why: caller-visible reports cannot be the authority for paid counters.
    # Keep one private set, publish detached views, and reject altered views.
    private_work = deepcopy(report["work"]) if guarded_work else report["work"]
    work_guard = None

    def require_work():
        if work_guard is not None:
            work_guard()

    def publish_work():
        nonlocal work_guard
        require_work()
        report["work"] = deepcopy(private_work)
        work_guard, _ = paid_report_guard({"work": report["work"]})
        original_view = report["work"]
        original_guard = work_guard

        def check():
            if report.get("work") is not original_view:
                raise RuntimeError("nearest probe paid work holder replaced")
            original_guard()

        work_guard = check

    if guarded_work:
        publish_work()

    if endpoint_observer is not None:
        report["endpoint_observations"] = []

    if owned.diagnostics:
        report["plan"]["diagnostics"] = True
        report["update_diagnostics"] = []
        for key in ("score_rows_attempted_by_role", "score_rows_returned_by_role"):
            report["work"][key]["probe_train"] = 0

    def check_originals():
        if rehearsal is not None:
            rehearsal.check_boundary()
        if correction_batch is not None:
            correction_batch._require_original()
        _native_state(source, type(source), native_before)
        if (
            any(
                actual is not original
                for actual, original in zip(identity_fn(source), source_owners, strict=True)
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
            if any(
                a is not b
                for a, b in zip(identity_fn(actor_native), runtime_actor_state, strict=True)
            ) or any(
                a is not b
                for a, b in zip(
                    identity_fn(candidate_native),
                    runtime_candidate_state,
                    strict=True,
                )
            ):
                raise RuntimeError("probe replaced an unobserved native runtime state")

    def require_paid():
        for key, paid_holder, index, original, guard, trusted in paid_records:
            try:
                if (
                    type(paid_holder) is not list
                    or report[key] is not paid_holder
                    or len(paid_holder) <= index
                    or paid_holder[index] is not original
                ):
                    raise RuntimeError("nearest probe preceding paid evidence changed")
                guard()
            except Exception:
                original.clear()
                original.update(deepcopy(trusted))
                report[key] = paid_holder
                while len(paid_holder) <= index:
                    paid_holder.append(None)
                paid_holder[index] = original
                raise

    def boundary():
        nonlocal last_clock
        if check_boundary is not None:
            check_boundary()
        if guarded_work:
            require_work()
        if nearest or full_fit or correction_batch is not None:
            require_paid()
        if not same_continuous_state(before, source.snapshot_state()):
            raise RuntimeError("continuous probe changed checkpoint source")
        if return_stage and not return_authority_taken:
            return_scope[2].check()
        last_clock = _require_budget(
            budget, limits, timer, started, owned.updates, paid, clock, tick, last_clock, runtime
        )
        if return_stage and not return_authority_taken:
            return_scope[2].check()
        if protect_owners:
            check_originals()
        if guarded_work:
            require_work()
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
        if return_stage and not return_authority_taken:
            return_scope[2].check()
        if protect_owners:
            check_originals()

        if guarded_work:
            require_work()
        if nearest or full_fit or correction_batch is not None:
            require_paid()

    try:
        boundary()
        if training_only:
            nearest_probe.record_training_consistency(
                before, owned.training, report, private_work, boundary, publish_work, seal_paid
            )
        native_source = (
            RecordedContinuousPC(source, report["native_updates"], owned.updates)
            if isinstance(source, NumpyContinuousPCLearner)
            else _RecordedProbeBP(
                source,
                report["native_updates"],
                owned.updates,
                fit,
                protect=correction_batch is not None,
            )
        )
        if return_stage:
            return_scope[2].check()
            # The existing runtime's first operation observes this same budget.
            # Transfer only after all opaque admission/initial-boundary reads.
            return_authority_taken = True
        runtime = ActorShadowRuntime(
            native_source,
            actor_version="probe-actor",
            candidate_version="probe-candidate",
            clock=clock,
            budget=budget,
            max_experiences=owned.updates,
            max_consolidations=0,
            training_role="train" if return_stage else "probe_train",
        )
        if full_fit or correction_batch is not None:
            full_candidate = runtime._candidate
        if protect_owners:
            runtime_owners = _probe_runtime_owners(runtime, type(source))
            runtime_actor_state = identity_fn(runtime_owners[-2])
            runtime_candidate_state = identity_fn(runtime_owners[-1])
        readout = source.fork()
        if protect_owners and (type(readout) is not type(source) or readout is source):
            raise RuntimeError("probe endpoint requires a detached exact readout")
        episodes = (
            dict(
                training=owned.training,
                **{f"old_{index}": item for index, item in enumerate(readouts)},
            )
            if training_only
            else dict(
                fresh=owned.evaluation,
                **{f"old_{index}": item for index, item in enumerate(readouts)},
            )
        )
        if owned.diagnostics or rehearsal is not None:
            episodes["fresh_training"] = owned.training
        if retained_nearest and not training_only:
            episodes["fresh_training"] = (
                owned.fresh_training if isinstance(owned, _RetainedReturnPlan) else owned.training
            )
        boundary()
        score_options = dict(_work=private_work, _publish_work=publish_work) if guarded_work else {}
        if training_only:
            if retained_nearest:
                score_options["_retained_content"] = retained_training.bind_content(
                    before, features, targets, private_work, publish_work
                )
            else:
                score_options["_source_rows"] = before.training_examples
        _score(report, readout, expected, episodes, boundary, **score_options)
        if nearest or full_fit or correction_batch is not None:
            seal_paid("checkpoints", 0)
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
                identity_fn=identity_fn,
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
                    "train" if return_stage else "probe_train",
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
                    role="train" if return_stage else "probe_train",
                )
            )
            boundary()
            private_work["training_attempts"] += 1
            private_work["training_rows_attempted"] += physical_rows
            if guarded_work:
                publish_work()
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
                or state.training_examples != before.training_examples + paid * physical_rows
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
                RETAINED_OBJECTIVE
                if retained_nearest
                else NEAREST_OBJECTIVE
                if nearest
                else CONTINUOUS_PC_ENERGY_ID
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
                or report["native_updates"][-1]["training_rows"] != physical_rows
                or (nearest and diagnostic.value != len(features))
                or (
                    isinstance(source, NumpyContinuousPCLearner)
                    and report["native_updates"][-1]["work"]["published_updates"] != 1
                )
            ):
                raise RuntimeError("continuous probe native return differs from receipt")
            if retained_nearest and not same_continuous_state(
                advance_retained(before_update, features, targets), state
            ):
                raise RuntimeError("retained probe update differs from declared content selection")
            expected = state
            if protect_owners:
                _native_state(runtime_owners[-1], type(source), expected)
                runtime_candidate_state = identity_fn(runtime_owners[-1])
            boundary()
            event["success_validated"] = True
            if nearest or full_fit or correction_batch is not None:
                seal_paid("native_updates", index)
                seal_paid("events", index)
            if owned.diagnostics:
                _diagnose_update(report, paid, features, targets, before_update, state, base)
                boundary()
        _score(report, readout, expected, episodes, boundary, **score_options)
        if nearest or full_fit or correction_batch is not None:
            seal_paid("checkpoints", 1)
        if runtime.train_ready(max_updates=1) != ():
            raise RuntimeError("continuous probe has unexpected remaining work")
        boundary()
        if not same_continuous_state(expected, readout.snapshot_state()):
            raise RuntimeError("continuous probe final readout changed")
        first, final = (point["scores"] for point in report["checkpoints"])
        score = lambda point, name: point[name]["residual"]["overall"]["normalized_half_mse"]
        report["metrics"] = (
            dict(training_fit_gain=score(first, "training") - score(final, "training"))
            if training_only
            else dict(
                fresh_acquisition_gain=score(first, "fresh") - score(final, "fresh"),
                old_retention_loss={
                    name: score(final, name) - score(first, name)
                    for name in first
                    if name.startswith("old_")
                },
            )
        )
        if training_only and retained_nearest:
            report["metrics"]["old_retention_loss"] = {
                name: score(final, name) - score(first, name)
                for name in first
                if name.startswith("old_")
            }
        if owned.diagnostics or retained_nearest and not training_only or rehearsal is not None:
            report["metrics"]["fresh_training_gain"] = score(first, "fresh_training") - score(
                final, "fresh_training"
            )
        if return_stage:
            slot = next(f"old_{i}" for i, item in enumerate(readouts) if item == owned.training)
            report["metrics"]["return_training_gain"] = score(first, slot) - score(final, slot)
        report["complete"] = True
    except Exception as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}")
        if hasattr(error, "work"):
            report["failed_native_work"] = error.work
    if full_candidate is not None:
        try:
            # Admission, not a possibly corrupted live selector, owns capture.
            full_candidate.require_fit_work(writer=True, force=True)
        except Exception as error:
            report.update(complete=False, paid_history_error=f"{type(error).__name__}: {error}")
            report.pop("metrics", None)
        report["native_updates"] = full_candidate._full_reports
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
                identity_fn=identity_fn,
            )
        except Exception as error:
            report.update(complete=False, error=f"{type(error).__name__}: {error}")
    native_count = lambda value: value if type(value) is int and 0 <= value <= 2**53 - 1 else None
    if protect_owners:
        if guarded_work:
            try:
                require_work()
            except Exception as error:
                report.update(complete=False, error=f"{type(error).__name__}: {error}")
                report.pop("metrics", None)
            report["work"] = deepcopy(private_work)
        _summarize_work(report, source, base, before, owned, readouts)
        if rehearsal is not None:
            report["rehearsal"] = rehearsal.report()
            report["work"]["probe_training_rows_completed"] = report["work"][
                "training_rows_completed"
            ]
            report["work"]["training_rows_completed"] = (
                len(report["applied_updates"]) * physical_rows
                if report["receipt_observation_complete"]
                else None
            )
        if isinstance(owned, _RetainedReturnPlan):
            columns = base.spec.observation_dim + len(base.spec.action_units)
            report["work"]["reserved_numeric_bytes"] = (
                (len(owned.fresh_training.transitions) + len(owned.evaluation.transitions))
                * columns
                * 8
            )
            report["work"]["return_input_numeric_bytes"] = (
                len(owned.training.transitions) * columns * 8
            )
        final_guard, trusted = paid_report_guard(report)
        readout_owners: Any = None if readout is None else identity_fn(readout)
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
                    for actual, original in zip(identity_fn(readout), readout_owners, strict=True)
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
    identity_fn=nearest_probe.owner_identity,
):
    """Borrow a private readout synchronously; preserve paid work on refusal.

    The observer owns its separate readouts/costs and must not retain the learner.
    Existing bounded report and budget guards supply the proof; no new runtime.
    """
    report["endpoint_observations"].append(dict(phase=phase, attempted=True, completed=False))
    evidence, trusted = paid_report_guard(report)
    witness = _BudgetWitness(budget, original_budget_binding, limits, paid)
    kind = type(source)
    owners = identity_fn(source) + identity_fn(readout)
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
                identity_fn(source) + identity_fn(readout),
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
    common = kind in (
        NumpyContinuousBCLearner,
        NumpyNearestContinuousBCLearner,
        NumpyRetainedNearestBCLearner,
    )
    wrapper_kind = _RecordedProbeBP if common else RecordedContinuousPC
    if any(type(port) is not wrapper_kind for port in wrappers) or wrappers[1] is not wrappers[2]:
        raise RuntimeError("probe runtime lost its original recorded ports")
    if common:
        for port in wrappers:
            port.require_fit_work(writer=port is wrappers[1])
    native = tuple(port.learner if common else port._learner for port in wrappers[:2])
    if native[0] is native[1] or any(type(owner) is not kind for owner in native):
        raise RuntimeError("probe runtime requires detached exact native owners")
    reports = tuple(port.reports if common else port._reports for port in wrappers)
    fit_owners = tuple(
        owner
        for port in wrappers
        if common and port.fit is not None
        for owner in (port.fit, port.fit.features, port.fit.targets, port.limit)
    )
    protected_owners = tuple(
        owner
        for port in wrappers
        if common and port.protected
        for owner in (port.protected, port.limit)
    )
    return (
        runtime,
        runtime._actor,
        runtime._inbox,
        *wrappers,
        *reports,
        *fit_owners,
        *protected_owners,
        *(owner._retention for owner in native if kind is NumpyRetainedNearestBCLearner),
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
    if nearest_probe.is_nearest(source):
        nearest_probe.summarize(report, before, owned, readouts)
        return
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
    full_fit = getattr(owned, "optimizer", "sgd") == "full-gauss-newton"
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
        if isinstance(source, NumpyContinuousBCLearner) and not full_fit
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
        if isinstance(source, NumpyContinuousBCLearner) and not full_fit
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
    if full_fit:
        # Boolean unknown flags are disjunctions, never misleading integer sums.
        report["work"]["native_update_counts"] = {
            key: (
                all(item["work"][key] for item in native_work)
                if key == "solver_work_known"
                else any(item["work"][key] for item in native_work)
            )
            if type(value) is bool
            else sum(item["work"][key] for item in native_work)
            for key, value in (native_work[0]["work"].items() if native_work else ())
        }
        report["work"]["native_event_row_weights"] = {}
        report["work"]["bp_sgd_cost_applicable"] = False
        report["work"]["full_network_momentum_resets"] = len(native_work)


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
