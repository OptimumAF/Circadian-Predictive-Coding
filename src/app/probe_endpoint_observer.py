"""Observe a paid probe endpoint without granting training or budget renewal.

Inputs are the existing disposable learner, paid report and original two budget
bindings. The observer owns native-state validation and its separately accounted
readouts. This boundary protects report containers and resources, observes clock
and RSS changes through its supplied guard, and restores rejected paid evidence.
Synchronous supplied read ports retain parent checks between external port groups;
original clocks are observed before and after them. No trust is cached across reads.
It supplies no evaluator, persistence, native-state interpretation or selection.
"""

from copy import deepcopy
from dataclasses import fields
from math import isfinite
from typing import Any, Callable, Protocol, cast

from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import budget_binding, paid_report_guard
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionProgress


class EndpointBoundary(Protocol):
    """Qualify synchronous read ports and the adjoining budget observations."""

    def __call__(
        self,
        read_ports: Callable[[], None] | None = None,
        *,
        after_clocks: Callable[[], None] | None = None,
    ) -> None: ...


EndpointObserver = Callable[[Any, str, EndpointBoundary], None]


class _BudgetWitness:
    """Admit only the original paid counters and explicitly observed reads."""

    def __init__(self, budget: ToyBudgetSession, binding, limits, updates: int) -> None:
        self.budget = budget
        self.binding = binding
        self.limits = limits
        self.updates = updates
        self.last_clock = budget.last_clock
        self.progress = deepcopy(budget.progress)
        self.session_work = deepcopy(self._session_work())
        self.segment = deepcopy(budget.process_rss_segment)
        self.segment_owner = budget.process_rss_segment
        self.check()

    def _session_work(self):
        budget = self.budget
        return (
            budget.checkpoint_position,
            budget.hidden_width_observed,
            budget.peak_hidden_width_observed,
            budget.rejected_proposed_hidden_width,
        )

    def _check_paid_work(self) -> None:
        budget = self.budget
        if (
            budget_binding(budget) != self.binding
            or not _same_value(budget.budget, self.limits)
            or type(budget.updates_completed) is not int
            or budget.updates_completed != self.updates
            or type(budget.replay_examples_completed) is not int
            or budget.replay_examples_completed != 0
            or type(budget.last_clock) is not float
            or not isfinite(budget.last_clock)
            or type(budget.started_at) is not float
            or not isfinite(budget.started_at)
            or budget.last_clock < budget.started_at
            or not _same_value(self._session_work(), self.session_work)
        ):
            raise ValueError("endpoint observer changed an original budget or paid counters")
        progress = budget.progress
        if progress is not None and (
            type(progress) is not ToyExecutionProgress
            or set(vars(progress)) != {field.name for field in fields(ToyExecutionProgress)}
            or type(progress.updates_completed) is not int
            or progress.updates_completed != self.updates
            or type(progress.replay_examples_completed) is not int
            or progress.replay_examples_completed != 0
            or not _same_value(
                (
                    progress.checkpoint_position,
                    progress.hidden_width_observed,
                    progress.peak_hidden_width_observed,
                    progress.rejected_proposed_hidden_width,
                ),
                self.session_work,
            )
            or not _same_value(progress.process_rss_segment, budget.process_rss_segment)
        ):
            raise ValueError("endpoint observer changed original budget progress")

    def check(self) -> None:
        self._check_paid_work()
        if (
            self.budget.last_clock != self.last_clock
            or not _same_value(self.budget.progress, self.progress)
            or self.budget.process_rss_segment is not self.segment_owner
            or not _same_value(self.budget.process_rss_segment, self.segment)
        ):
            raise ValueError("endpoint observer changed unobserved clock or resource progress")

    def observe(self) -> None:
        self.check()
        self.budget.before_final()
        self._check_paid_work()
        if self.budget.last_clock < self.last_clock:
            raise ValueError("endpoint observer budget clock moved backwards")
        # Why: only this boundary admits new clock/RSS readings. A delegate's
        # direct assignments must not become a new accepted observation.
        self.last_clock = self.budget.last_clock
        self.progress = deepcopy(self.budget.progress)
        self.segment = deepcopy(self.budget.process_rss_segment)
        self.segment_owner = self.budget.process_rss_segment


def observe_endpoint(
    observer: EndpointObserver,
    learner: Any,
    phase: str,
    report: dict[str, Any],
    *,
    budget: ToyBudgetSession,
    return_budget: ToyBudgetSession | None,
    fixed_budgets,
    return_limits,
    check_boundary: Callable[[], None] | None,
    original_probe_updates: int = 16,
) -> None:
    """Run one observer, retaining the trusted complete report on any refusal."""
    paid_guard, trusted = paid_report_guard(report)
    try:
        if type(original_probe_updates) is not int or original_probe_updates not in (16, 48):
            raise ValueError(
                "endpoint observer supports only explicit original F16 or F48 authority"
            )
        if return_budget is None:
            raise ValueError("endpoint observer requires its original return budget")
        original, original_binding, original_limits, return_binding = fixed_budgets
        if original is not budget:
            raise ValueError("endpoint observer lost its original probe budget")
        witnesses = (
            _BudgetWitness(budget, original_binding, original_limits, original_probe_updates),
            _BudgetWitness(
                return_budget, return_binding, return_limits, 0 if phase == "after_probe" else 16
            ),
        )

        def check_evidence() -> None:
            try:
                paid_guard()
                for witness in witnesses:
                    witness.check()
            finally:
                if check_boundary is not None:
                    check_boundary()
                paid_guard()
                for witness in witnesses:
                    witness.check()

        def observe_boundary(
            read_ports: Callable[[], None] | None = None,
            *,
            after_clocks: Callable[[], None] | None = None,
        ) -> None:
            check_evidence()
            failure: Exception | None = None
            try:
                for witness in witnesses:
                    witness.observe()
                if read_ports is not None:
                    # A clock can mutate parent data. Refuse that change before
                    # admitting the supplied reads, even if composition saves no work.
                    check_evidence()
                    result = cast(Callable[..., Any], read_ports)()
                    if result is not None:
                        raise ValueError("endpoint read ports must return None")
                    check_evidence()
                    # The supplied reads consume time too; qualify that elapsed
                    # time before the observer can admit another external port.
                    for witness in witnesses:
                        witness.observe()
                    if after_clocks is not None:
                        # Why: an original clock port can change a native endpoint
                        # after the first proof. Read it after the last clock,
                        # then reprove paid/parent evidence before more work.
                        paid_guard()
                        for witness in witnesses:
                            witness.check()
                        result = cast(Callable[..., Any], after_clocks)()
                        if result is not None:
                            raise ValueError("endpoint final read ports must return None")
            except Exception as error:
                failure = error
            try:
                # Why: failures still require post-port checks. The parent must
                # run even when the paid report or budget was corrupted.
                check_evidence()
            except Exception as error:
                if failure is None:
                    failure = error
            if failure is not None:
                raise failure

        refusal: Exception | None = None

        def guard(
            read_ports: Callable[[], None] | None = None,
            *,
            after_clocks: Callable[[], None] | None = None,
        ) -> None:
            nonlocal refusal
            if refusal is not None:
                raise refusal
            try:
                observe_boundary(read_ports, after_clocks=after_clocks)
            except Exception as error:
                # A callback cannot catch a refused reading, repair its values,
                # and thereby admit a later observation as a successful phase.
                refusal = error
                raise

        guard()
        result = cast(Callable[..., Any], observer)(learner, phase, guard)
        if result is not None:
            raise ValueError("endpoint observer must retain its readings and return None")
        guard()
    except Exception:
        report.clear()
        report.update(trusted)
        raise
