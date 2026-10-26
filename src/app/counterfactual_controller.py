"""Four disposable version-bound TRAIN trials followed by one frozen fit.

The application supplies native execution and accounting. This owner keeps one
complete detached base snapshot and at most four scalar rows. It grants no
promotion, final/probe access, budget renewal or checkpoint authority.
"""

from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

from src.core.actor_ports import VersionedState
from src.core.contextual_controller import (
    ARMS,
    ContextualEstimate,
    FittedController,
    fit_choices,
    training_context,
)
from src.core.controller_objective import (
    ControllerMeasures,
    ControllerObjective,
    controller_utility,
)
from src.core.experience import LogicalClock, require_identifier, require_tick
from src.core.reference_controller import ReferenceObservation


@dataclass(frozen=True)
class CounterfactualRequest:
    learner_version: str
    state_digest: str
    context: str
    action_id: str
    started_at: int
    role: str = "controller_train"


@dataclass(frozen=True)
class CounterfactualReturn:
    request: CounterfactualRequest
    labels_arrived_at: int
    completed_at: int
    measures: ControllerMeasures


class CounterfactualController:
    def __init__(
        self,
        source: Callable[[], VersionedState[Any]],
        state_digest: Callable[[Any], str],
        objective: ControllerObjective,
        clock: LogicalClock,
    ):
        if type(objective) is not ControllerObjective or type(clock) is not LogicalClock:
            raise ValueError("counterfactual training requires original objective and clock")
        objective.__post_init__()
        self._source, self._digest = source, state_digest
        self._base = deepcopy(source())
        if type(self._base) is not VersionedState:
            raise ValueError("source must return original versioned complete state")
        require_identifier(self._base.actor_version, "learner_version")
        self._base_digest = state_digest(self._base.state)
        require_identifier(self._base_digest, "state_digest")
        self._objective, self._objective_pin = deepcopy(objective), deepcopy(objective)
        self._clock, self._clock_pin = clock, clock
        self._rows: tuple[ContextualEstimate, ...] = ()
        self._attempts = 0
        self._stopped = False
        self._fitted: FittedController | None = None
        self._gate = Lock()

    def _require_current(self):
        if self._clock is not self._clock_pin or self._objective != self._objective_pin:
            raise ValueError("counterfactual objective/clock changed")
        current = self._source()
        if (
            type(current) is not VersionedState
            or current.actor_version != self._base.actor_version
            or self._digest(current.state) != self._base_digest
            or self._digest(self._base.state) != self._base_digest
        ):
            raise ValueError("stale learner version or complete source payload changed")
        if self._stopped:
            raise ValueError("counterfactual training stopped after paid failed work")

    def run(
        self,
        observation: ReferenceObservation,
        operation: Callable[[CounterfactualRequest, Any], CounterfactualReturn],
    ) -> ContextualEstimate:
        if not self._gate.acquire(blocking=False):
            raise ValueError("counterfactual owner busy; reentrant execution refused")
        try:
            self._require_current()
            if self._fitted is not None or self._attempts >= 4:
                raise ValueError("counterfactual training is closed")
            context = training_context(observation)
            if (
                context is None
                or observation.actor_version != self._base.actor_version
                or observation.tick > self._clock.now()
            ):
                raise ValueError("trial requires current arrived ordinary TRAIN context")
            prior = [e for e in self._rows if e.context == context]
            if len(prior) >= 2 or not callable(operation):
                raise ValueError("trial requires an untried context/arm and callable operation")
            request = CounterfactualRequest(
                self._base.actor_version,
                self._base_digest,
                context,
                ARMS[len(prior)],
                self._clock.now(),
            )
            pin = deepcopy(request)
            self._attempts += 1
            try:
                result = operation(request, deepcopy(self._base.state))
                self._require_current()
                if (
                    type(result) is not CounterfactualReturn
                    or result.request is not request
                    or request != pin
                    or request.role != "controller_train"
                ):
                    raise ValueError("counterfactual return requires original TRAIN trial metadata")
                require_tick(result.labels_arrived_at, "trial label arrival")
                require_tick(result.completed_at, "trial return")
                if (
                    not request.started_at
                    <= result.labels_arrived_at
                    <= result.completed_at
                    <= self._clock.now()
                ):
                    raise ValueError("counterfactual labels/returns have not arrived causally")
                utility = controller_utility(self._objective, result.measures)
                row = ContextualEstimate(context, request.action_id, utility.value)
                self._rows = (*self._rows, row)
                return deepcopy(row)
            except BaseException:
                self._stopped = True
                raise
        finally:
            self._gate.release()

    def freeze(self) -> FittedController:
        if not self._gate.acquire(blocking=False):
            raise ValueError("counterfactual owner busy")
        try:
            self._require_current()
            if self._fitted is not None:
                raise ValueError("controller already fitted; no repeated selection")
            choices = fit_choices(self._rows)
            self._fitted = FittedController(
                self._base.actor_version,
                self._base_digest,
                deepcopy(self._objective),
                self._clock.now(),
                deepcopy(self._rows),
                choices,
            )
            return deepcopy(self._fitted)
        finally:
            self._gate.release()

    def snapshot(self) -> dict:
        return dict(
            attempts=self._attempts,
            completed=len(self._rows),
            discarded_trials=self._attempts,
            stopped=self._stopped,
            rows=deepcopy(self._rows),
            fitted=deepcopy(self._fitted),
            clock=self._clock.now(),
        )
