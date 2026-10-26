"""One birth-bound observed-damage prune through the existing structural gate.

The runtime owns publication and consumed attempt IDs. This driver owns fixed
validation data/policy and bounded scalar evidence, never a model-state chain.
Accepted gate evidence is distinct from publication; only an exact rejected
decision plus unchanged live candidate permits subsequent training. No new
prune math, data selection, threshold tuning or budget renewal is provided.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import isfinite
import pickle
from time import perf_counter
from typing import Any

import numpy as np

from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_pruning_gate import CoupledPruneLimits
from src.app.coupled_scaling_prune_consolidation import (
    CoupledScalingPruneRefused,
    _owned_probes,
    _same_raw,
    _validate_evidence,
)
from src.app.coupled_scaling_pruning_gate import CoupledScalingPruneDecision
from src.app.coupled_structural_consolidation import CoupledScalingStructuralPruneConsolidator
from src.app.linear_scaling_measurement import ScalingBatch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.capacity_cost import CapacityUsage
from src.core.coupled_inference import _finite_number
from src.core.coupled_structural_state import CoupledStructuralState


@dataclass(frozen=True)
class _PrunePolicy:
    at_update: int
    removed_units: tuple[int, ...]
    limits: CoupledPruneLimits
    inference_steps: int
    inference_learning_rate: float


def _budget_stamp(budget: ToyBudgetSession) -> tuple[Any, ...]:
    if type(budget) is not ToyBudgetSession or type(budget.budget) is not ToyExecutionBudget:
        raise ValueError("capacity pruning requires exact original budget session and limits")
    replace(budget.budget)
    if not callable(budget.clock):
        raise ValueError("capacity pruning requires its original clock")
    for value in (budget.started_at, budget.last_clock):
        if type(value) not in {int, float} or not isfinite(value):
            raise ValueError("capacity pruning requires finite original clock metadata")
    if budget.last_clock < budget.started_at:
        raise ValueError("capacity pruning original clock metadata regressed")
    for count in (budget.updates_completed, budget.replay_examples_completed):
        if type(count) is not int or count < 0:
            raise ValueError("capacity pruning requires native nonnegative work counters")
    return (
        id(budget),
        id(budget.budget),
        asdict(budget.budget),
        id(budget.clock),
        budget.started_at,
        id(budget.progress),
        id(budget.process_rss_sampler),
    )


def _probe_stamp(probes: tuple[ScalingBatch, ...]) -> tuple[Any, ...]:
    if type(probes) is not tuple or not 1 <= len(probes) <= 8:
        raise ValueError("capacity pruning requires its bounded original probes")
    for probe in probes:
        if (
            type(probe) is not ScalingBatch
            or type(probe.features) is not np.ndarray
            or type(probe.targets) is not np.ndarray
            or probe.features.dtype != np.float64
            or probe.targets.dtype != np.float64
            or probe.features.ndim != 2
            or not 1 <= len(probe.features) <= 1024
            or probe.targets.shape != (len(probe.features), 1)
            or probe.features.size + probe.targets.size > 1_000_000
        ):
            raise ValueError("capacity pruning original probe layout changed")
    return (
        id(probes),
        tuple(
            (
                id(p),
                id(p.features),
                id(p.targets),
                p.features.shape,
                p.targets.shape,
                p.features.dtype.str,
                p.targets.dtype.str,
            )
            for p in probes
        ),
    )


def _same_state(left: CoupledStructuralState, right: CoupledStructuralState) -> bool:
    return (
        type(left) is CoupledStructuralState
        and type(right) is CoupledStructuralState
        and left.format_version == right.format_version
        and left.origin_config == right.origin_config
        and left.transitions == right.transitions
        and _same_raw(left.raw_state, right.raw_state)
    )


class StreamCapacityPrune:
    def __init__(
        self,
        *,
        at_update: int,
        removed_units: tuple[int, ...],
        probes: tuple[ScalingBatch, ...],
        limits: CoupledPruneLimits,
        budget: ToyBudgetSession,
        inference_steps: int,
        inference_learning_rate: float,
    ) -> None:
        if type(at_update) is not int or not 1 <= at_update <= 512:
            raise ValueError("capacity prune at_update must be a native integer in 1..512")
        if (
            type(removed_units) is not tuple
            or not 1 <= len(removed_units) <= 16
            or any(type(index) is not int or index < 0 for index in removed_units)
        ):
            raise ValueError(
                "capacity prune requires one native nonnegative index per hidden layer"
            )
        if type(limits) is not CoupledPruneLimits:
            raise ValueError("capacity pruning requires exact native damage limits")
        if type(inference_steps) is not int or not 1 <= inference_steps <= 64:
            raise ValueError("capacity prune inference_steps must lie in 1..64")
        rate = _finite_number(inference_learning_rate, "inference_learning_rate", positive=True)
        self._budget = budget
        self._policy = _PrunePolicy(
            at_update, removed_units, replace(limits), inference_steps, rate
        )
        self._probes = _owned_probes(probes)
        self._probe_metadata = _probe_stamp(self._probes)
        self._probe_content = pickle.dumps(self._probes)
        self._original_binding = (pickle.dumps(self._policy), _budget_stamp(budget))
        if budget.updates_completed != 0 or budget.replay_examples_completed != 0:
            raise ValueError("capacity prune budget must be fresh at birth")
        self._bound, self._stopped, self._spent = False, False, False
        self._total_updates: int | None = None
        self._runtime: ActorShadowRuntime[Any, Any, Any, Any] | None = None
        self._reason: str | None = None
        self._seconds = 0.0
        self._event: dict[str, Any] = dict(
            updates=at_update,
            event_id="capacity-prune",
            status="not_run",
            reason="event not reached",
            requested=False,
            attempted=False,
            committed=False,
            receipt=None,
            decision_status=None,
            decision_reason=None,
            diagnostics=[],
            planned_probe_count=len(self._probes),
            known_validation_score_calls=0,
            known_validation_row_forwards=0,
            known_validation_energy_evaluations=0,
            validation_work_complete=False,
            validation_score_calls=None,
            validation_row_forwards=None,
            validation_energy_evaluations=None,
            post_usage=None,
            measurement_error=None,
            evidence_error=None,
        )

    @property
    def at_update(self) -> int:
        return self._policy.at_update

    def _check_binding(self) -> None:
        if (pickle.dumps(self._policy), _budget_stamp(self._budget)) != self._original_binding:
            raise ValueError("capacity prune original policy or budget binding changed")
        if (
            _probe_stamp(self._probes) != self._probe_metadata
            or pickle.dumps(self._probes) != self._probe_content
        ):
            raise ValueError("capacity prune original probe data changed")
        if (
            type(self._spent) is not bool
            or self._spent != self._event["requested"]
            or self._event["updates"] != self.at_update
        ):
            raise ValueError("capacity prune spent schedule differs from original event")

    def bind(self, budget: ToyBudgetSession, total_updates: int) -> int:
        if self._bound:
            raise ValueError("capacity pruning is single-use and already bound")
        self._check_binding()
        if (
            budget is not self._budget
            or type(total_updates) is not int
            or not self.at_update <= total_updates <= 512
            or budget.updates_completed != 0
            or budget.replay_examples_completed != 0
        ):
            raise ValueError(
                "capacity pruning must bind its original budget and complete planned wakes"
            )
        self._total_updates, self._bound = total_updates, True
        return 1

    def _progress(self, runtime: ActorShadowRuntime[Any, Any, Any, Any], updates: int) -> None:
        self._check_binding()
        if (
            not self._bound
            or self._stopped
            or type(runtime) is not ActorShadowRuntime
            or runtime._budget is not self._budget
            or (self._runtime is not None and runtime is not self._runtime)
            or type(updates) is not int
            or updates != self.at_update
            or self._budget.updates_completed != updates
            or self._budget.replay_examples_completed != 0
            or len(runtime.applied_updates) != updates
        ):
            raise ValueError("capacity pruning requires its actual scheduled wake/runtime/budget")

    def _evidence(self, decision: CoupledScalingPruneDecision) -> None:
        _validate_evidence(
            decision, self._probes, self._policy.limits, self._policy.inference_steps
        )
        diagnostics = decision.diagnostics
        calls, rows = 2 * len(diagnostics), 2 * sum(len(row.row_ids) for row in diagnostics)
        energies = sum(
            row.source_inference.energy_evaluations + row.candidate_inference.energy_evaluations
            for row in diagnostics
        )
        complete = len(diagnostics) == len(self._probes)
        self._event.update(
            decision_status=decision.status,
            decision_reason=decision.reason,
            diagnostics=[asdict(row) for row in diagnostics],
            known_validation_score_calls=calls,
            known_validation_row_forwards=rows,
            known_validation_energy_evaluations=energies,
            validation_work_complete=complete,
            validation_score_calls=calls if complete else None,
            validation_row_forwards=rows if complete else None,
            validation_energy_evaluations=energies if complete else None,
        )

    def after_update(self, runtime: ActorShadowRuntime[Any, Any, Any, Any], updates: int) -> bool:
        self._progress(runtime, updates)
        if self._spent:
            raise ValueError("capacity pruning schedule is already spent")
        self._runtime, self._spent = runtime, True
        self._event.update(requested=True, status="failed", reason=None)
        started = perf_counter()
        try:
            return self._apply_event(runtime, updates)
        except Exception as error:
            self._stopped, self._reason = True, f"{type(error).__name__}: {error}"
            self._event.update(status="failed", reason=self._reason)
            raise
        finally:
            self._seconds += perf_counter() - started

    def _apply_event(self, runtime: ActorShadowRuntime[Any, Any, Any, Any], updates: int) -> bool:
        consolidator = None
        primary: Exception | None = None
        rejected = False
        source = None
        try:
            self._budget.before_sleep()
            source = runtime.candidate_snapshot().state
            if type(source) is not CoupledStructuralState:
                raise ValueError("capacity pruning requires exact structural raw state")
            consolidator = CoupledScalingStructuralPruneConsolidator(
                self._policy.removed_units,
                self._probes,
                limits=replace(self._policy.limits),
                budget=self._budget,
                inference_steps=self._policy.inference_steps,
                inference_learning_rate=self._policy.inference_learning_rate,
                retain_decision=True,
            )

            def operation(state):
                self._check_binding()
                self._event["attempted"] = True
                result = consolidator(state)
                self._check_binding()
                return result

            receipt = runtime.consolidate(self._event["event_id"], operation)
            self._event.update(committed=True, receipt=asdict(receipt))
        except Exception as error:
            primary = error
            for receipt in runtime.applied_consolidations:
                if receipt.event_id == self._event["event_id"]:
                    self._event.update(committed=True, receipt=asdict(receipt))
                    break
            if type(error) is CoupledScalingPruneRefused and not self._event["committed"]:
                decision = error.decision
                self._evidence(decision)
                if decision.status == "rejected" and source is not None:
                    self._progress(runtime, updates)
                    current = runtime.candidate_snapshot().state
                    if (
                        type(current) is CoupledStructuralState
                        and _same_state(source, current)
                        and _same_raw(decision.state, source.raw_state)
                    ):
                        self._budget.before_final()
                        rejected, primary = True, None
        finally:
            if consolidator is not None:
                try:
                    retained_decision = consolidator.last_decision
                    if retained_decision is not None:
                        self._evidence(retained_decision)
                except Exception as error:
                    self._event["evidence_error"] = f"{type(error).__name__}: {error}"
                    if primary is None:
                        primary = error
        if primary is not None:
            raise primary
        self._check_binding()
        self._budget.before_final()
        self._event.update(
            status="rejected" if rejected else "complete",
            reason=self._event["decision_reason"] if rejected else None,
        )
        return bool(self._event["committed"])

    def _record_observation(self, usage: dict[str, Any] | None, error: Exception | None) -> None:
        self._event["post_usage"] = None if usage is None else asdict(CapacityUsage(**usage))
        if error is not None:
            self._event["measurement_error"] = f"{type(error).__name__}: {error}"
            if not self._stopped:
                self._reason = self._event["measurement_error"]
            self._stopped = True
            self._event.update(status="failed", reason=self._reason)

    @property
    def report(self) -> dict[str, Any]:
        return deepcopy(
            dict(
                bound=self._bound,
                stopped=self._stopped,
                reason=self._reason,
                at_update=self.at_update,
                total_updates=self._total_updates,
                work_seconds=self._seconds,
                events=[self._event],
                partial_work_note="known completed probe pairs only; a failed partial score may contain additional unknown work; gate acceptance does not attest publication",
            )
        )
