"""Finite disposable version-bound TRAIN trials followed by one frozen fit.

The application supplies native execution and accounting. This owner keeps one
complete detached base snapshot and four legacy or six opt-in catalog rows. It grants no
promotion, final/probe access, budget renewal or checkpoint authority.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any, Callable, cast

from src.core.actor_ports import VersionedState
from src.core.contextual_controller import (
    ARMS,
    ContextualEstimate,
    FittedController,
    fit_choices,
    training_context,
)
from src.core.adaptation_action import AdaptationEnvelope
from src.core.catalog_contextual_controller import (
    ContextualArmCatalog,
    CatalogEstimate,
    CatalogTrialWork,
    FittedCatalogController,
    fit_catalog_choices,
    choose_catalog_contextual,
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


@dataclass(frozen=True, kw_only=True)
class CatalogCounterfactualRequest(CounterfactualRequest):
    catalog: ContextualArmCatalog


@dataclass(frozen=True)
class CatalogCounterfactualReturn:
    request: CatalogCounterfactualRequest
    labels_arrived_at: int
    completed_at: int
    measures: ControllerMeasures
    work: CatalogTrialWork


class CounterfactualController:
    def __init__(
        self,
        source: Callable[[], VersionedState[Any]],
        state_digest: Callable[[Any], str],
        objective: ControllerObjective,
        clock: LogicalClock,
        *,
        catalog: ContextualArmCatalog | None = None,
    ):
        if type(objective) is not ControllerObjective or type(clock) is not LogicalClock:
            raise ValueError("counterfactual training requires original objective and clock")
        objective.__post_init__()
        if catalog is not None:
            if type(catalog) is not ContextualArmCatalog:
                raise ValueError("counterfactual catalog must be an exact finite catalog")
            catalog.__post_init__()
            self._catalog = self._catalog_owner = deepcopy(catalog)
            self._catalog_pin = deepcopy(catalog)
            self._failed_catalog_work: CatalogTrialWork | None = None
            self._catalog_prefix_pin: Any = ((), ())
            self._catalog_prefix_owners: tuple[Any, ...] = ()
        self._source, self._digest = source, state_digest
        self._base = deepcopy(source())
        if type(self._base) is not VersionedState:
            raise ValueError("source must return original versioned complete state")
        require_identifier(self._base.actor_version, "learner_version")
        self._base_digest = state_digest(self._base.state)
        require_identifier(self._base_digest, "state_digest")
        self._objective, self._objective_pin = deepcopy(objective), deepcopy(objective)
        self._clock, self._clock_pin = clock, clock
        self._rows: tuple[ContextualEstimate | CatalogEstimate, ...] = ()
        self._returns: tuple[CounterfactualReturn | CatalogCounterfactualReturn, ...] = ()
        self._attempts = 0
        self._stopped = False
        self._fitted: FittedController | FittedCatalogController | None = None
        self._application_pin: Any = None
        self._application_owners: tuple[Any, ...] = ()
        self._gate = Lock()

    def _require_current(self):
        if self._clock is not self._clock_pin or self._objective != self._objective_pin:
            raise ValueError("counterfactual objective/clock changed")
        catalog = getattr(self, "_catalog", None)
        if catalog is not getattr(self, "_catalog_owner", None) or catalog != getattr(
            self, "_catalog_pin", None
        ):
            raise ValueError("original counterfactual arm catalog changed or was replaced")
        if catalog is not None:
            if type(catalog) is not ContextualArmCatalog:
                raise ValueError("original counterfactual catalog type changed")
            catalog.__post_init__()
        current = self._source()
        if (
            type(current) is not VersionedState
            or current.actor_version != self._base.actor_version
            or self._digest(current.state) != self._base_digest
            or self._digest(self._base.state) != self._base_digest
        ):
            raise ValueError("stale learner version or complete source payload changed")
        if catalog is not None:
            self._require_catalog_prefix()
        if self._stopped:
            raise ValueError("counterfactual training stopped after paid failed work")

    def run(
        self,
        observation: ReferenceObservation,
        operation: Callable[
            [CounterfactualRequest, Any], CounterfactualReturn | CatalogCounterfactualReturn
        ],
    ) -> ContextualEstimate | CatalogEstimate:
        if not self._gate.acquire(blocking=False):
            raise ValueError("counterfactual owner busy; reentrant execution refused")
        try:
            self._require_current()
            catalog = getattr(self, "_catalog", None)
            arms = ARMS if catalog is None else catalog.arms
            if self._fitted is not None or self._attempts >= 2 * len(arms):
                raise ValueError("counterfactual training is closed")
            context = training_context(observation)
            if (
                context is None
                or observation.actor_version != self._base.actor_version
                or observation.tick > self._clock.now()
            ):
                raise ValueError("trial requires current arrived ordinary TRAIN context")
            prior = [e for e in self._rows if e.context == context]
            if len(prior) >= len(arms) or not callable(operation):
                raise ValueError("trial requires an untried context/arm and callable operation")
            if catalog is None:
                request = CounterfactualRequest(
                    self._base.actor_version,
                    self._base_digest,
                    context,
                    arms[len(prior)],
                    self._clock.now(),
                )
            else:
                request = CatalogCounterfactualRequest(
                    self._base.actor_version,
                    self._base_digest,
                    context,
                    arms[len(prior)],
                    self._clock.now(),
                    catalog=deepcopy(catalog),
                )
            pin = deepcopy(request)
            self._attempts += 1
            try:
                result = operation(request, deepcopy(self._base.state))
                self._require_current()
                if (
                    type(result)
                    is not (
                        CounterfactualReturn if catalog is None else CatalogCounterfactualReturn
                    )
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
                if catalog is not None:
                    if (
                        type(result) is not CatalogCounterfactualReturn
                        or type(result.work) is not CatalogTrialWork
                    ):
                        raise ValueError("catalog trial requires exact returned work counters")
                    result.work.__post_init__()
                    # Keep known partial costs if scoring is later refused; the
                    # callback's native owner retains its own permanent charges.
                    self._failed_catalog_work = deepcopy(result.work)
                    result.work.require_complete(catalog, request.action_id)
                utility = controller_utility(self._objective, result.measures)
                row = (ContextualEstimate if catalog is None else CatalogEstimate)(
                    context, request.action_id, utility.value
                )
                self._rows = (*self._rows, row)
                self._returns = (*self._returns, deepcopy(result))
                if catalog is not None:
                    self._failed_catalog_work = None
                    self._catalog_prefix_pin = deepcopy((self._rows, self._returns))
                    self._catalog_prefix_owners = self._catalog_evidence_owners()
                return deepcopy(row)
            except BaseException:
                self._stopped = True
                raise
        finally:
            self._gate.release()

    def freeze(self) -> FittedController | FittedCatalogController:
        if not self._gate.acquire(blocking=False):
            raise ValueError("counterfactual owner busy")
        try:
            self._require_current()
            if self._fitted is not None:
                raise ValueError("controller already fitted; no repeated selection")
            catalog = getattr(self, "_catalog", None)
            if catalog is None:
                rows = cast(tuple[ContextualEstimate, ...], self._rows)
                self._fitted = FittedController(
                    self._base.actor_version,
                    self._base_digest,
                    deepcopy(self._objective),
                    self._clock.now(),
                    deepcopy(rows),
                    fit_choices(rows),
                )
            else:
                catalog_rows = cast(tuple[CatalogEstimate, ...], self._rows)
                self._fitted = FittedCatalogController(
                    self._base.actor_version,
                    self._base_digest,
                    deepcopy(self._objective),
                    self._clock.now(),
                    deepcopy(catalog_rows),
                    fit_catalog_choices(catalog, catalog_rows),
                    catalog=deepcopy(catalog),
                )
            # Why: application must preserve all original finite paid returns,
            # including cost proxies, without retaining a growing fit history.
            self._application_pin = deepcopy((self._fitted, self._rows, self._returns))
            self._application_owners = self._fit_owners()
            return deepcopy(self._fitted)
        finally:
            self._gate.release()

    def application_snapshot(self) -> dict:
        """Detached frozen fit and full source binding, never training authority.

        The caller must also compare both source payloads to its actual learner;
        a supplied digest alone is not complete-state equality. Fit cost measures
        retain the existing time/RSS proxies, not unrecorded native FLOP counts.
        """
        if not self._gate.acquire(blocking=False):
            raise ValueError("counterfactual owner busy")
        try:
            self._require_current()
            self._require_frozen_fit()
            current = deepcopy(self._source())
            self._require_frozen_fit()
            if type(current) is not VersionedState:
                raise ValueError("application source requires original versioned complete state")
            return dict(
                source=deepcopy(self._base),
                current_source=current,
                policy=deepcopy(self._fitted),
                fit=dict(
                    attempts=self._attempts,
                    completed=len(self._rows),
                    discarded_trials=self._attempts,
                    returns=[asdict(r) for r in self._returns],
                    cost_scope=(
                        "original complete ControllerMeasures time/RSS/violation proxies; native work ledger remains external"
                        if not hasattr(self, "_catalog")
                        else "original ControllerMeasures plus all catalog native/row/replay/prediction counters; detailed compute ledger remains external"
                    ),
                ),
            )
        finally:
            self._gate.release()

    def _require_frozen_fit(self) -> None:
        fitted = self._fitted
        catalog = getattr(self, "_catalog", None)
        expected_type = FittedController if catalog is None else FittedCatalogController
        count = 4 if catalog is None else 2 * len(catalog.arms)
        if (
            type(fitted) is not expected_type
            or self._attempts != count
            or len(self._rows) != count
            or len(self._returns) != count
        ):
            raise ValueError(
                "application requires four completed trials and an original frozen fit"
                if catalog is None
                else "application requires six completed catalog trials and an original frozen fit"
            )
        assert fitted is not None
        fitted.__post_init__()
        owners = self._fit_owners()
        same_owners = len(owners) == len(self._application_owners) and all(
            current is original for current, original in zip(owners, self._application_owners)
        )
        if not same_owners or (fitted, self._rows, self._returns) != self._application_pin:
            raise ValueError("original frozen counterfactual fit evidence changed or was replaced")
        if fitted.fitted_at > self._clock.now():
            raise ValueError("fitted controller has not arrived on its original clock")
        for returned in self._returns:
            require_tick(returned.request.started_at, "original trial start")
            require_tick(returned.labels_arrived_at, "original trial label arrival")
            require_tick(returned.completed_at, "original trial return")
            controller_utility(self._objective, returned.measures)
            if catalog is not None:
                if (
                    type(returned) is not CatalogCounterfactualReturn
                    or returned.request.catalog != catalog
                ):
                    raise ValueError("original trial catalog binding changed")
                returned.work.require_complete(catalog, returned.request.action_id)

    def _fit_owners(self) -> tuple[Any, ...]:
        fitted = self._fitted
        assert fitted is not None
        owners: list[Any] = [
            fitted,
            self._rows,
            self._returns,
            fitted.objective,
            fitted.estimates,
            fitted.choices,
        ]
        for returned in self._returns:
            owners.extend((returned, returned.request, returned.measures))
            if type(returned) is CatalogCounterfactualReturn:
                owners.extend(
                    (
                        returned.work,
                        returned.request.catalog,
                        returned.request.catalog.envelope,
                        returned.request.catalog.envelope.actions,
                        *returned.request.catalog.envelope.actions,
                    )
                )
        if type(fitted) is FittedCatalogController:
            owners.extend(
                (
                    fitted.catalog,
                    fitted.catalog.envelope,
                    fitted.catalog.envelope.actions,
                    *fitted.catalog.envelope.actions,
                )
            )
        return tuple(owners)

    def _catalog_evidence_owners(self) -> tuple[Any, ...]:
        owners: list[Any] = [self._rows, self._returns]
        owners.extend(self._rows)
        for returned in self._returns:
            if type(returned) is not CatalogCounterfactualReturn:
                raise ValueError("catalog trial return type changed")
            catalog = returned.request.catalog
            owners.extend(
                (
                    returned,
                    returned.request,
                    returned.measures,
                    returned.work,
                    catalog,
                    catalog.envelope,
                    catalog.envelope.actions,
                    *catalog.envelope.actions,
                )
            )
        return tuple(owners)

    def _require_catalog_prefix(self) -> None:
        owners = self._catalog_evidence_owners() if self._rows or self._returns else ()
        original = self._catalog_prefix_owners
        if (
            len(owners) != len(original)
            or any(current is not prior for current, prior in zip(owners, original))
            or (self._rows, self._returns) != self._catalog_prefix_pin
        ):
            raise ValueError("preceding catalog trial evidence changed or was replaced")

    def choose(
        self,
        observation: ReferenceObservation,
        *,
        envelope: AdaptationEnvelope,
        remaining_replay_examples: int,
    ):
        """Select an opt-in catalog ID; the original shared owner admits work."""
        if not self._gate.acquire(blocking=False):
            raise ValueError("counterfactual owner busy")
        try:
            self._require_current()
            self._require_frozen_fit()
            if type(self._fitted) is not FittedCatalogController:
                raise ValueError("catalog selection requires explicit catalog mode")
            if type(observation) is not ReferenceObservation:
                raise ValueError("catalog selection requires an exact causal observation")
            if observation.tick > self._clock.now():
                raise ValueError("catalog observation has not arrived on the original clock")
            return choose_catalog_contextual(
                self._fitted,
                observation,
                envelope=envelope,
                remaining_replay_examples=remaining_replay_examples,
            )
        finally:
            self._gate.release()

    def snapshot(self) -> dict:
        result = dict(
            attempts=self._attempts,
            completed=len(self._rows),
            discarded_trials=self._attempts,
            stopped=self._stopped,
            rows=deepcopy(self._rows),
            fitted=deepcopy(self._fitted),
            clock=self._clock.now(),
        )
        if hasattr(self, "_catalog"):
            result.update(
                catalog=deepcopy(self._catalog), failed_work=deepcopy(self._failed_catalog_work)
            )
        return result
