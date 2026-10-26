"""Bounded candidate renewal through the existing stream consolidation boundary.

Inputs are an owned renewable learner, original wake budget and fixed schedule.
Outputs distinguish permanent replacement charges from returned/committed work.
No labels, evaluation, replay, durable budget restoration or policy tuning occur.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from time import perf_counter
from typing import Any

from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ConsolidatedState
from src.core.coupled_energy import Array
from src.core.coupled_unit_renewal import select_renewal_indices
from src.core.learner_ports import TrainingDiagnostic
from src.core.renewable_coupled_predictive_coding import RenewableCoupledSnapshot


def _count(value: int, name: str, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be a native integer in {minimum}..{maximum}")


def _budget(session: ToyBudgetSession, *, fresh: bool) -> None:
    if type(session) is not ToyBudgetSession or type(session.budget) is not ToyExecutionBudget:
        raise ValueError("unit renewal requires an exact wake budget session and limits")
    replace(session.budget)
    if (
        session.budget.max_replay_examples is not None
        or session.budget.max_hidden_width is not None
    ):
        raise ValueError("unit renewal does not support replay or hidden-width budget limits")
    for value in (session.updates_completed, session.replay_examples_completed):
        if type(value) is not int or value < 0 or (fresh and value != 0):
            raise ValueError("unit renewal requires native nonnegative fresh birth counters")


class StreamUnitRenewal:
    """Single-use, nonrefundable replacement allowance independent of model state."""

    def __init__(
        self,
        learner: Any,
        budget: ToyBudgetSession,
        *,
        every_updates: int,
        total_updates: int,
        max_replacements: int,
    ) -> None:
        _count(every_updates, "every_updates", 1, 512)
        _count(total_updates, "total_updates", 1, 512)
        _count(max_replacements, "max_replacements", 0, 128)
        _budget(budget, fresh=True)
        if (
            not callable(getattr(learner, "renew_units", None))
            or type(learner.snapshot_state()) is not RenewableCoupledSnapshot
        ):
            raise ValueError("unit renewal requires a renewable complete native state")
        self._learner, self._budget = learner, budget
        self._every, self._total, self._limit = every_updates, total_updates, max_replacements
        self._bound, self._stopped, self._observed, self._charged = False, False, 0, 0
        self._reason: str | None = None
        self._runtime: ActorShadowRuntime[Array, Array, Array, Any] | None = None
        self._events: list[dict[str, Any]] = [
            dict(
                updates=k,
                event_id=f"unit-renewal-{k}",
                status="not_run",
                reason="event not reached",
                selected_indices=[],
                old_unit_ids=[],
                new_unit_ids=[],
                charged_units=0,
                attempted_units=0,
                returned_units=0,
                committed_units=0,
                receipt=None,
                work_seconds=0.0,
            )
            for k in range(every_updates, total_updates + 1, every_updates)
        ]
        self._original_binding = self._binding()

    def _binding(self) -> tuple[Any, ...]:
        _budget(self._budget, fresh=False)
        _count(self._every, "every_updates", 1, 512)
        _count(self._total, "total_updates", 1, 512)
        _count(self._limit, "max_replacements", 0, 128)
        return (
            id(self._learner),
            id(self._budget),
            id(self._budget.budget),
            asdict(self._budget.budget),
            id(self._budget.clock),
            self._budget.started_at,
            id(self._budget.progress),
            id(self._budget.process_rss_sampler),
            self._every,
            self._total,
            self._limit,
        )

    def _check_binding(self) -> None:
        if self._binding() != self._original_binding:
            raise ValueError("unit renewal original schedule, learner or budget binding changed")
        if (
            type(self._charged) is not int
            or not 0 <= self._charged <= self._limit
            or len(self._events) != self._total // self._every
            or any(
                type(row["charged_units"]) is not int or row["charged_units"] < 0
                for row in self._events
            )
            or self._charged != sum(row["charged_units"] for row in self._events)
        ):
            raise ValueError("unit renewal permanent charges differ from original event work")

    def check_progress(
        self, runtime: ActorShadowRuntime[Array, Array, Array, Any], updates: int
    ) -> None:
        """Refuse changed birth authority/charges before another wake is admitted."""
        self._check_binding()
        if (
            not self._bound
            or self._stopped
            or type(updates) is not int
            or type(self._observed) is not int
            or updates != self._observed
            or not 0 <= updates <= self._total
            or type(runtime) is not ActorShadowRuntime
            or runtime._budget is not self._budget
            or (self._runtime is not None and runtime is not self._runtime)
            or self._budget.updates_completed != updates
            or len(runtime.applied_updates) != updates
        ):
            raise ValueError("unit renewal progress differs from actual completed wake receipts")

    def bind(self, learner: Any, budget: ToyBudgetSession, total_updates: int) -> int:
        if self._bound:
            raise ValueError("unit renewal is single-use and already bound")
        self._check_binding()
        if (
            learner is not self._learner
            or budget is not self._budget
            or type(total_updates) is not int
            or total_updates != self._total
        ):
            raise ValueError(
                "unit renewal binding must match its original learner, budget and count"
            )
        _budget(budget, fresh=True)
        self._bound = True
        return len(self._events)

    def _operation(self, state: Any, row: dict[str, Any]) -> ConsolidatedState[Any]:
        self._check_binding()
        self._budget.before_sleep()
        if type(state) is not RenewableCoupledSnapshot:
            raise ValueError("candidate lost its complete renewable state")
        history = state.unit_history
        policy = state.renewal_policy
        if history is None or policy is None:
            raise ValueError("candidate renewal state is incomplete")
        remaining = self._limit - self._charged
        indices = select_renewal_indices(history, policy, remaining)
        old_ids = tuple(int(history.unit_ids[i]) for i in indices)
        new_ids = tuple(range(history.next_unit_id, history.next_unit_id + len(indices)))
        # Why charge before even the private fork: failed preparation is permanent
        # attempted work, and restoring an older native snapshot cannot refund it.
        self._charged += len(indices)
        row.update(
            selected_indices=list(indices),
            old_unit_ids=list(old_ids),
            new_unit_ids=list(new_ids),
            charged_units=len(indices),
        )
        learner = self._learner.fork()
        self._check_binding()
        if learner is self._learner:
            raise ValueError("renewal fork must be distinct from source")
        self._budget.before_sleep()
        learner.restore_state(deepcopy(state))
        self._check_binding()
        self._budget.before_sleep()
        row["attempted_units"] = len(indices)
        result = learner.renew_units(remaining)
        row["returned_units"] = len(result.indices)
        self._check_binding()
        if (result.indices, result.old_unit_ids, result.new_unit_ids) != (
            indices,
            old_ids,
            new_ids,
        ):
            raise ValueError("native renewal differs from the precharged selection")
        self._budget.before_sleep()
        snapshot = learner.snapshot_state()
        self._budget.before_sleep()
        return ConsolidatedState(
            snapshot, TrainingDiagnostic("coupled_unit_renewal_count_v1", float(len(indices)))
        )

    def _run_event(
        self, runtime: ActorShadowRuntime[Array, Array, Array, Any], row: dict[str, Any]
    ) -> None:
        started = perf_counter()
        try:
            self._apply_event(runtime, row)
        finally:
            # Includes selection, private preparation, reset and publication,
            # even on failure. This is separate from native wake work and units.
            row["work_seconds"] += perf_counter() - started

    def _apply_event(
        self, runtime: ActorShadowRuntime[Array, Array, Array, Any], row: dict[str, Any]
    ) -> None:
        if self._charged == self._limit:
            row.update(
                status="allowance_exhausted", reason="permanent replacement allowance exhausted"
            )
            return
        try:
            receipt = runtime.consolidate(
                row["event_id"], lambda state: self._operation(state, row)
            )
            row.update(
                status="complete",
                reason=None,
                receipt=asdict(receipt),
                committed_units=row["returned_units"],
            )
        except Exception as error:
            for receipt in runtime.applied_consolidations:
                if receipt.event_id == row["event_id"]:
                    row.update(receipt=asdict(receipt), committed_units=row["returned_units"])
                    break
            row.update(status="failed", reason=f"{type(error).__name__}: {error}")
            raise

    def after_update(
        self, runtime: ActorShadowRuntime[Array, Array, Array, Any], updates: int
    ) -> None:
        self._check_binding()
        if not self._bound or self._stopped:
            raise ValueError("unit renewal must be bound and open")
        if type(runtime) is not ActorShadowRuntime or runtime._budget is not self._budget:
            raise ValueError("unit renewal requires its original wake-budget runtime")
        if self._runtime is not None and runtime is not self._runtime:
            raise ValueError("unit renewal runtime identity must remain unchanged")
        if (
            type(updates) is not int
            or type(self._observed) is not int
            or updates != self._observed + 1
            or updates > self._total
        ):
            raise ValueError("unit renewal requires consecutive newly completed wakes")
        _budget(self._budget, fresh=False)
        if self._budget.updates_completed != updates or len(runtime.applied_updates) != updates:
            raise ValueError("unit renewal requires actual completed wake budget and receipts")
        try:
            self._runtime, self._observed = runtime, updates
            if updates % self._every == 0:
                self._run_event(runtime, self._events[updates // self._every - 1])
            self.check_progress(runtime, updates)
        except Exception as error:
            self._stopped, self._reason = True, f"{type(error).__name__}: {error}"
            raise

    @property
    def report(self) -> dict[str, Any]:
        return deepcopy(
            dict(
                bound=self._bound,
                stopped=self._stopped,
                reason=self._reason,
                total_updates=self._total,
                every_updates=self._every,
                max_replacements=self._limit,
                wake_updates_observed=self._observed,
                planned_events=len(self._events),
                charged_units=self._charged,
                event_seconds=sum(row["work_seconds"] for row in self._events),
                **{
                    name: sum(row[name] for row in self._events)
                    for name in ("attempted_units", "returned_units", "committed_units")
                },
                events=self._events,
            )
        )
