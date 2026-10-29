"""Bind the existing sharing prediction port to one MiniGrid actor.

Owns finite control pins and start/latest observations, not a scheduler, training
permit, checkpoint proof, consent authority or history of private updates.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from math import isfinite
from typing import Any

from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import budget_binding
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.resource_sharing import SharingLimits, SharingSnapshot

REASONS = ("paused", "serving_active", "training_active", "work_quota", "resource_unavailable")


def _count(value):
    if type(value) is not int or not 0 <= value <= 2**53 - 1:
        raise ValueError("MiniGrid sharing counters require bounded native nonnegative integers")


def _raw_gate(gate):
    if type(gate) is not ServingPriorityGate or type(gate._limits) is not SharingLimits:
        raise ValueError("MiniGrid shared prediction requires an exact priority gate and limits")
    for field in fields(gate._limits):
        _count(getattr(gate._limits, field.name))
    gate._limits.__post_init__()
    _count(gate._serving)
    _count(gate._admitted)
    if (
        gate._serving > gate._limits.max_serving_requests
        or gate._admitted > gate._limits.max_admitted_updates
    ):
        raise ValueError("MiniGrid raw sharing state exceeds original limits")
    if type(gate._training) is not bool or type(gate._paused) is not bool:
        raise ValueError("MiniGrid sharing flags require native booleans")
    if type(gate._deferrals) is not dict or len(gate._deferrals) > 5:
        raise ValueError("MiniGrid sharing deferrals exceed five native reasons")
    for key, count in gate._deferrals.items():
        if type(key) is not str or key not in REASONS:
            raise ValueError("MiniGrid sharing has a foreign deferral reason")
        _count(count)
    if not callable(gate._resource):
        raise ValueError("MiniGrid sharing requires its resource probe")


def _budget(owner):
    if type(owner) is not ToyBudgetSession or type(owner.budget) is not ToyExecutionBudget:
        raise ValueError("MiniGrid sharing requires an exact original native budget")
    for field in fields(owner.budget):
        value = getattr(owner.budget, field.name)
        if value is not None and (
            type(value) not in (int, float) or abs(value) > 2**53 - 1 or not isfinite(value)
        ):
            raise ValueError("MiniGrid native budget has a foreign or unbounded limit")
    owner.budget.__post_init__()
    _count(owner.updates_completed)
    _count(owner.replay_examples_completed)
    for clock in (owner.started_at, owner.last_clock):
        if type(clock) is not float or not isfinite(clock):
            raise ValueError("MiniGrid native budget clock must be a finite native float")


def _snapshot(value):
    if type(value) is not SharingSnapshot or type(value.limits) is not SharingLimits:
        raise ValueError("MiniGrid sharing requires an exact snapshot")
    for field in fields(value.limits):
        _count(getattr(value.limits, field.name))
    value.limits.__post_init__()
    _count(value.active_serving_requests)
    _count(value.admitted_updates)
    if (
        value.active_serving_requests > value.limits.max_serving_requests
        or value.admitted_updates > value.limits.max_admitted_updates
        or type(value.training_active) is not bool
        or type(value.paused) is not bool
        or type(value.deferrals) is not tuple
        or len(value.deferrals) > 5
    ):
        raise ValueError("MiniGrid sharing snapshot exceeds original limits")
    seen = set()
    for pair in value.deferrals:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or pair[0] not in REASONS
            or pair[0] in seen
        ):
            raise ValueError("MiniGrid sharing snapshot has foreign or duplicate deferrals")
        _count(pair[1])
        seen.add(pair[0])


def _owners(shared, actor):
    if (
        type(shared) is not ResourceSharedRuntime
        or type(shared._runtime) is not ActorShadowRuntime
        or shared._runtime._actor is not actor
    ):
        raise ValueError("MiniGrid shared runtime must own this exact actor")
    runtime, gate = shared._runtime, shared._sharing
    if (
        runtime._training_role != "train"
        or runtime._original_training_role != "train"
        or runtime._inbox._training_role != "train"
        or runtime._inbox._original_training_role != "train"
    ):
        raise ValueError("MiniGrid shared runtime requires its original ordinary TRAIN role")
    _raw_gate(gate)
    _budget(runtime._budget)
    authority = shared._action_authority
    if (
        type(authority) is not tuple
        or len(authority) != 3
        or any(
            value is not expected
            for value, expected in zip(authority, (runtime, gate, runtime._budget))
        )
    ):
        raise ValueError("MiniGrid shared runtime original authority changed")
    if not _same_value(shared._action_limits, (gate._limits, runtime._budget.budget)):
        raise ValueError("MiniGrid shared runtime original limits changed")
    return runtime, gate, runtime._budget


@dataclass
class SharedMiniGridPrediction:
    shared: Any
    actor: Any
    runtime: Any
    gate: Any
    budget: Any
    limits: Any
    native_limits: Any
    controls: Any
    resource: Any
    initial: Any
    latest: Any
    first_progress: Any
    progress: Any
    error: str | None = None

    def _check_controls(self):
        runtime, gate, budget = _owners(self.shared, self.actor)
        if (
            runtime is not self.runtime
            or gate is not self.gate
            or budget is not self.budget
            or gate._limits is not self.limits
            or gate._resource is not self.resource
            or not _same_value(budget_binding(budget), self.controls)
            or not _same_value(budget.budget, self.native_limits)
        ):
            raise ValueError("pinned MiniGrid sharing controls were changed or replaced")
        return gate, budget

    def check(self):
        if self.error is not None:
            raise ValueError("MiniGrid shared prediction guard is stopped")
        try:
            gate, budget = self._check_controls()
            current = gate.snapshot()
            _snapshot(current)
            self._check_controls()
            if not _same_value(current.limits, self.initial.limits):
                raise ValueError("MiniGrid sharing observation changed original limits")
            old, new = dict(self.latest.deferrals), dict(current.deferrals)
            if current.admitted_updates < self.latest.admitted_updates or any(
                new.get(key, 0) < count for key, count in old.items()
            ):
                raise ValueError("MiniGrid sharing admission or deferral counter moved backwards")
            progress = (
                budget.updates_completed,
                budget.replay_examples_completed,
                budget.last_clock,
            )
            if any(after < before for after, before in zip(progress, self.progress)):
                raise ValueError("MiniGrid original native budget progress moved backwards")
            self.latest = deepcopy(current)
            self.progress = progress
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"[:512]
            raise

    def report(self):
        return dict(
            initial=asdict(self.initial),
            latest=asdict(self.latest),
            native_progress_before=list(self.first_progress),
            native_progress_latest=list(self.progress),
            last_readback_error=self.error,
            scope="observed original gate counters; prediction requests only; not checkpoint or private-update history authority",
        )


def bind_shared_prediction(shared, actor):
    runtime, gate, budget = _owners(shared, actor)
    # Pin before the first public observation; its callback cannot redefine
    # which controls count as the original ones for this binding.
    limits, native_limits = gate._limits, deepcopy(budget.budget)
    controls, resource = budget_binding(budget), gate._resource
    progress = (budget.updates_completed, budget.replay_examples_completed, budget.last_clock)
    initial = gate.snapshot()
    _snapshot(initial)
    result = SharedMiniGridPrediction(
        shared,
        actor,
        runtime,
        gate,
        budget,
        limits,
        native_limits,
        controls,
        resource,
        deepcopy(initial),
        deepcopy(initial),
        progress,
        progress,
    )
    result.check()
    return result
