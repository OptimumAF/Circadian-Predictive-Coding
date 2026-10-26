"""Committed capacity/work observations and one predeclared guarded raw split.

The existing runtime owns native training and publication. This helper stores
bounded scalar samples, charges no replacement budget and sees no labels or
evaluation data. Means weight the model used by each successful wake, whereas
initial/post-split measurements affect peaks and future wakes only. No RSS/FLOP
measurement, state chain, continuation or general growth policy lives here.
One optional birth-bound prune driver reuses the existing observed-damage gate.
Observation-only mode can also sample one fixed-size consolidation per wake,
without giving that observation an additional vote in the wake-weighted means.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from math import isfinite
from time import perf_counter
from typing import Any, Callable

from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_structural_consolidation import CoupledScalingSplitConsolidator
from src.app.stream_capacity_prune import StreamCapacityPrune
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.capacity_cost import CapacityUsage, count_dense_parameters
from src.core.coupled_scaling_structure import CoupledScalingSplitLineage
from src.core.coupled_structural_state import CoupledStructuralState


def _positive(value: int, name: str) -> None:
    if type(value) is not int or not 0 < value < 2**63:
        raise ValueError(f"capacity {name} must be a positive native integer below 2**63")


def _budget_binding(budget: ToyBudgetSession) -> tuple[Any, ...]:
    if type(budget) is not ToyBudgetSession or type(budget.budget) is not ToyExecutionBudget:
        raise ValueError("capacity requires an exact original budget session and limits")
    replace(budget.budget)
    if not callable(budget.clock):
        raise ValueError("capacity requires its original callable clock")
    for value in (budget.started_at, budget.last_clock):
        if type(value) not in {int, float} or not isfinite(value):
            raise ValueError("capacity original clock metadata must be finite")
    if budget.last_clock < budget.started_at:
        raise ValueError("capacity original clock metadata regressed")
    for count in (budget.updates_completed, budget.replay_examples_completed):
        if type(count) is not int or count < 0:
            raise ValueError("capacity work counters must be native nonnegative integers")
    return id(budget), id(budget.budget), asdict(budget.budget), id(budget.clock), budget.started_at


class StreamCapacity:
    def __init__(
        self,
        learner: Any,
        budget: ToyBudgetSession,
        *,
        total_updates: int,
        measure_state: Callable[[Any], CapacityUsage],
        max_parameters: int,
        max_state_array_bytes: int,
        split_at_update: int | None = None,
        max_width: int | None = None,
        pruning: StreamCapacityPrune | None = None,
    ) -> None:
        if type(total_updates) is not int or not 1 <= total_updates <= 512:
            raise ValueError("capacity total_updates must be a native integer in 1..512")
        if not callable(measure_state):
            raise ValueError("capacity measure_state must be callable")
        _positive(max_parameters, "max_parameters")
        _positive(max_state_array_bytes, "max_state_array_bytes")
        if split_at_update is not None and (
            type(split_at_update) is not int or not 1 <= split_at_update <= total_updates
        ):
            raise ValueError("capacity split_at_update must lie in the planned wake range")
        if max_width is not None:
            _positive(max_width, "max_width")
        if split_at_update is not None and max_width is None:
            raise ValueError("scheduled capacity split requires a finite max_width")
        if pruning is not None and type(pruning) is not StreamCapacityPrune:
            raise ValueError("capacity pruning must be an exact StreamCapacityPrune")
        if pruning is not None:
            pruning._check_binding()
            if pruning._budget is not budget or pruning.at_update > total_updates:
                raise ValueError(
                    "capacity pruning must share the original budget and planned wakes"
                )
        self._pruning = pruning
        self._original_budget = _budget_binding(budget)
        if (
            type(budget.updates_completed) is not int
            or budget.updates_completed != 0
            or type(budget.replay_examples_completed) is not int
            or budget.replay_examples_completed != 0
        ):
            raise ValueError("capacity budget counters must be fresh at birth")
        self._learner, self._budget, self._measure_state = learner, budget, measure_state
        self._total_updates, self._max_parameters = total_updates, max_parameters
        self._max_state_bytes, self._split_at, self._max_width = (
            max_state_array_bytes,
            split_at_update,
            max_width,
        )
        self._original_config = (
            total_updates,
            max_parameters,
            max_state_array_bytes,
            split_at_update,
            max_width,
            id(measure_state),
            id(learner),
            id(pruning),
        )
        self._bound, self._started, self._stopped = False, False, False
        self._runtime: ActorShadowRuntime[Any, Any, Any, Any] | None = None
        self._reason: str | None = None
        self._current: CapacityUsage | None = None
        self._calls, self._rows = 0, 0
        self._last_consolidation_observed = 0
        self._measurement_seconds = 0.0
        self._sums = dict(parameters=0, parameter_bytes=0, auxiliary_bytes=0, total_array_bytes=0)
        self._peaks = dict(self._sums)
        self._samples: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = (
            []
            if split_at_update is None
            else [
                dict(
                    updates=split_at_update,
                    status="not_run",
                    reason="event not reached",
                    requested=False,
                    attempted=False,
                    committed=False,
                    receipt=None,
                    predicted_usage=None,
                    source_width=None,
                    target_width=None,
                    post_usage=None,
                    measurement_error=None,
                )
            ]
        )

    def _check_binding(self) -> None:
        if _budget_binding(self._budget) != self._original_budget:
            raise ValueError("capacity original budget, limits or clock binding changed")
        current = (
            self._total_updates,
            self._max_parameters,
            self._max_state_bytes,
            self._split_at,
            self._max_width,
            id(self._measure_state),
            id(self._learner),
            id(self._pruning),
        )
        if any(
            type(value) is not type(original) or value != original
            for value, original in zip(current, self._original_config, strict=True)
        ):
            raise ValueError("capacity original schedule, ceilings or measurement binding changed")
        if self._pruning is not None:
            self._pruning._check_binding()

    def bind(self, learner: Any, budget: ToyBudgetSession, total_updates: int) -> int:
        if self._bound:
            raise ValueError("capacity helper is single-use and already bound")
        self._check_binding()
        if (
            learner is not self._learner
            or budget is not self._budget
            or type(total_updates) is not int
            or total_updates != self._total_updates
            or self._budget.updates_completed != 0
            or self._budget.replay_examples_completed != 0
        ):
            raise ValueError(
                "capacity binding must match original learner, budget and planned wakes"
            )
        slots = 0 if self._pruning is None else self._pruning.bind(budget, total_updates)
        self._bound = True
        return len(self._events) + slots

    @property
    def observation_only(self) -> bool:
        """Whether the original helper policy contains no scheduled transform."""
        return self._original_config[3] is None and self._original_config[7] == id(None)

    def _require_runtime(self, runtime: ActorShadowRuntime[Any, Any, Any, Any]) -> None:
        self._check_binding()
        if (
            not self._bound
            or self._stopped
            or type(runtime) is not ActorShadowRuntime
            or runtime._budget is not self._budget
            or self._budget.replay_examples_completed != 0
            or (self._runtime is not None and runtime is not self._runtime)
        ):
            raise ValueError("capacity requires its original bound open runtime and budget")

    def _observe(self, usage: CapacityUsage) -> None:
        self._current = usage
        for name in self._peaks:
            self._peaks[name] = max(self._peaks[name], getattr(usage, name))

    def _limits(self, usage: CapacityUsage) -> None:
        if (
            usage.parameters > self._max_parameters
            or usage.total_array_bytes > self._max_state_bytes
        ):
            raise ValueError("committed capacity exceeds original parameter or state-array ceiling")

    def _measure(
        self,
        runtime: ActorShadowRuntime[Any, Any, Any, Any],
        phase: str,
        *,
        check_budget: bool = True,
    ) -> CapacityUsage:
        started = perf_counter()
        row: dict[str, Any] = dict(
            updates=self._calls, phase=phase, status="failed", reason=None, usage=None
        )
        self._samples.append(row)
        try:
            if check_budget:
                self._budget.before_final()
            state = runtime.candidate_snapshot().state
            # A measurement callback cannot alter the source or a later split's
            # admission state: it receives only a private detached observation.
            measured = self._measure_state(deepcopy(state))
            if type(measured) is not CapacityUsage:
                raise ValueError("capacity measurement must return exact CapacityUsage")
            usage = replace(measured)
            row["usage"] = asdict(usage)
            self._observe(usage)
            self._check_binding()
            if (
                self._budget.updates_completed != self._calls
                or self._budget.replay_examples_completed != 0
            ):
                raise ValueError("capacity measurement changed original completed work counters")
            self._limits(usage)
            if check_budget:
                self._budget.before_final()
            row.update(status="complete", reason=None)
            return usage
        except Exception as error:
            row["reason"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            self._measurement_seconds += perf_counter() - started

    def start(self, runtime: ActorShadowRuntime[Any, Any, Any, Any]) -> None:
        self._require_runtime(runtime)
        if self._started or self._budget.updates_completed != 0:
            raise ValueError("capacity must start once before any wake work")
        self._runtime, self._started = runtime, True
        try:
            self._measure(runtime, "initial")
        except Exception as error:
            self._stopped, self._reason = True, f"{type(error).__name__}: {error}"
            raise

    def _predicted_split(self, state: Any) -> CapacityUsage:
        self._check_binding()
        if type(state) is not CoupledStructuralState:
            raise ValueError("scheduled capacity split requires exact CoupledStructuralState")
        lineage = CoupledScalingSplitLineage(1, state.raw_state.config)
        source, target = lineage.source_config, lineage.target_config
        parameters = count_dense_parameters(
            target.input_dim, (target.width,) * target.hidden_layers
        )
        auxiliary = 0 if target.circadian is None else 2 * target.width * 8
        usage = CapacityUsage(parameters, 8 * parameters, auxiliary)
        event = self._events[0]
        event.update(
            source_width=source.width, target_width=target.width, predicted_usage=asdict(usage)
        )
        if self._max_width is None or target.width > self._max_width:
            raise ValueError("scheduled split target exceeds original max_width")
        self._limits(usage)
        return usage

    def _split(self, runtime: ActorShadowRuntime[Any, Any, Any, Any]) -> None:
        event = self._events[0]
        event.update(requested=True, status="failed", reason=None)
        primary: Exception | None = None
        try:
            self._predicted_split(runtime.candidate_snapshot().state)

            def operation(state):
                event["attempted"] = True
                self._predicted_split(state)
                return CoupledScalingSplitConsolidator(max_width=self._max_width)(state)

            receipt = runtime.consolidate("capacity-split", operation)
            event.update(committed=True, receipt=asdict(receipt))
        except Exception as error:
            primary = error
            # The original runtime can publish and then fail its final budget
            # check. The receipt, rather than the helper return, defines commit.
            for receipt in runtime.applied_consolidations:
                if receipt.event_id == "capacity-split":
                    event.update(committed=True, receipt=asdict(receipt))
                    break
        if event["committed"]:
            try:
                usage = self._measure(runtime, "post_split", check_budget=primary is None)
                event["post_usage"] = asdict(usage)
            except Exception as error:
                event["post_usage"] = self._samples[-1]["usage"]
                event["measurement_error"] = f"{type(error).__name__}: {error}"
                if primary is None:
                    primary = error
        if primary is not None:
            event["reason"] = f"{type(primary).__name__}: {primary}"
            raise primary
        event.update(status="complete", reason=None)

    def _prune(self, runtime: ActorShadowRuntime[Any, Any, Any, Any]) -> None:
        pruning = self._pruning
        if pruning is None:
            raise ValueError("capacity pruning was not configured")
        primary: Exception | None = None
        try:
            pruning.after_update(runtime, self._calls)
        except Exception as error:
            primary = error
        # Only an actual publication changes future wake-use capacity. A late
        # budget error still permits bounded readback without new admission.
        if pruning.report["events"][0]["committed"]:
            try:
                usage = self._measure(runtime, "post_prune", check_budget=primary is None)
                pruning._record_observation(asdict(usage), None)
            except Exception as error:
                pruning._record_observation(self._samples[-1]["usage"], error)
                if primary is None:
                    primary = error
        if primary is not None:
            raise primary

    def after_update(
        self, runtime: ActorShadowRuntime[Any, Any, Any, Any], updates: int, rows: int
    ) -> None:
        self._require_runtime(runtime)
        if (
            not self._started
            or self._current is None
            or type(updates) is not int
            or updates != self._calls + 1
            or updates > self._total_updates
            or self._budget.updates_completed != updates
            or len(runtime.applied_updates) != updates
            or type(rows) is not int
            or not 1 <= rows <= 128
        ):
            raise ValueError(
                "capacity requires consecutive actual completed wakes and bounded rows"
            )
        used = self._current
        # Record known completed work before any observer/split can fail. The
        # prior measured model was used for this wake; a later split gets no vote.
        self._calls, self._rows = updates, self._rows + rows
        for name in self._sums:
            self._sums[name] += getattr(used, name)
        try:
            current = self._measure(runtime, "post_wake")
            if current != used:
                raise ValueError("native wake changed capacity outside the scheduled split")
            if updates == self._split_at:
                self._split(runtime)
            if self._pruning is not None and updates == self._pruning.at_update:
                self._prune(runtime)
        except Exception as error:
            self._stopped, self._reason = True, f"{type(error).__name__}: {error}"
            raise

    def after_consolidation(
        self,
        runtime: ActorShadowRuntime[Any, Any, Any, Any],
        updates: int,
        *,
        check_budget: bool = True,
    ) -> None:
        """Observe one fixed-size event without adding wake work or mean weight.

        The caller reserves check_budget=False for late-error readback. That path
        skips admission callbacks only; original bindings and completed receipts
        are still checked, and no budget telemetry or allowance is restored.
        """
        self._require_runtime(runtime)
        if (
            type(check_budget) is not bool
            or not self.observation_only
            or not self._started
            or self._current is None
            or type(updates) is not int
            or updates < 1
            or updates != self._calls
            or self._budget.updates_completed != updates
            or len(runtime.applied_updates) != updates
            or updates <= self._last_consolidation_observed
        ):
            raise ValueError("capacity consolidation observation requires its same unobserved wake")
        used = self._current
        self._last_consolidation_observed = updates
        try:
            current = self._measure(runtime, "post_consolidation", check_budget=check_budget)
            if current != used:
                raise ValueError("fixed-size consolidation changed capacity")
        except Exception as error:
            self._stopped, self._reason = True, f"{type(error).__name__}: {error}"
            raise

    @property
    def report(self) -> dict[str, Any]:
        return deepcopy(
            dict(
                bound=self._bound,
                started=self._started,
                stopped=self._stopped,
                reason=self._reason,
                total_updates=self._total_updates,
                completed_native_calls=self._calls,
                completed_training_rows=self._rows,
                measurement_seconds=self._measurement_seconds,
                current_usage=None if self._current is None else asdict(self._current),
                **{f"peak_{name}": value for name, value in self._peaks.items()},
                **{
                    f"mean_{name}": value / self._calls if self._calls else None
                    for name, value in self._sums.items()
                },
                samples=self._samples,
                events=self._events,
                **{
                    f"split_{name}": sum(int(event[name]) for event in self._events)
                    for name in ("requested", "attempted", "committed")
                },
                partial_work_note="committed wake calls/rows only; unknown failed native attempts are not zero-imputed; stored field bytes exclude temporary allocations, metadata and process overhead",
                **({} if self._pruning is None else {"pruning": self._pruning.report}),
            )
        )
