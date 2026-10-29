"""Opt-in finite replay/wake selection records, distinct from frozen v1 schemas.

Owns scalar catalog/coverage/cost validation and deterministic empirical choices.
Uses existing contexts and objective; no model, data, clock mutation, budget,
replay selection or execution authority belongs here.
"""

from dataclasses import dataclass, field
from math import isfinite

from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.contextual_controller import CONTEXTS, training_context
from src.core.controller_objective import ControllerObjective
from src.core.experience import require_identifier, require_tick
from src.core.reference_controller import ReferenceDecision, ReferenceObservation


@dataclass(frozen=True)
class ContextualArmCatalog:
    envelope: AdaptationEnvelope

    def __post_init__(self):
        if type(self.envelope) is not AdaptationEnvelope:
            raise ValueError("catalog requires an exact original adaptation envelope")
        self.envelope.__post_init__()
        actions = self.envelope.actions
        if len(actions) != 3 or actions[:2] != (
            AdaptationAction("wait"),
            AdaptationAction("wake1", 1),
        ):
            raise ValueError("catalog requires ordered wait, wake1 and one replay-wake choice")
        replay = actions[2]
        if (
            replay.wake_updates != 1
            or not 1 <= replay.replay_examples <= 128
            or replay.learning_rate_multiplier != 1
            or replay.gate_setting != "unchanged"
            or replay.sleep_updates != 0
        ):
            raise ValueError("catalog supports only one default replay-then-wake action")

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(a.action_id for a in self.envelope.actions)


def _catalog(catalog: ContextualArmCatalog) -> None:
    if type(catalog) is not ContextualArmCatalog:
        raise ValueError("an exact bounded contextual arm catalog is required")
    catalog.__post_init__()


@dataclass(frozen=True)
class CatalogEstimate:
    context: str
    action_id: str
    mean_utility: float
    samples: int = 1

    def __post_init__(self):
        if type(self.context) is not str or self.context not in CONTEXTS:
            raise ValueError("estimate requires an original TRAIN context")
        require_identifier(self.action_id, "catalog action_id")
        if type(self.samples) is not int or self.samples != 1:
            raise ValueError("catalog requires one fixed sample per context/arm")
        try:
            finite = type(self.mean_utility) in (int, float) and isfinite(self.mean_utility)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("catalog estimate requires finite utility")


def fit_catalog_choices(
    catalog: ContextualArmCatalog, estimates: tuple[CatalogEstimate, ...]
) -> tuple[str, str]:
    _catalog(catalog)
    if (
        type(estimates) is not tuple
        or len(estimates) != 6
        or any(type(e) is not CatalogEstimate for e in estimates)
    ):
        raise ValueError("catalog fit requires six complete context/arm estimates")
    for row in estimates:
        row.__post_init__()
    table = {(e.context, e.action_id): e.mean_utility for e in estimates}
    if set(table) != {(c, a) for c in CONTEXTS for a in catalog.arms}:
        raise ValueError("catalog fit requires every context/arm exactly once")
    # Why: max keeps the first declared arm on ties, with wait always first.
    selected = tuple(max(catalog.arms, key=lambda arm: table[context, arm]) for context in CONTEXTS)
    return selected[0], selected[1]


@dataclass(frozen=True)
class CatalogTrialWork:
    attempted_updates: int
    returned_updates: int
    attempted_rows: int
    returned_rows: int
    replay_examples_charged: int
    prediction_calls: int
    prediction_rows: int
    work_complete: bool = True

    def __post_init__(self):
        for name in (
            "attempted_updates",
            "returned_updates",
            "attempted_rows",
            "returned_rows",
            "replay_examples_charged",
            "prediction_calls",
            "prediction_rows",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 2**31 - 1:
                raise ValueError("catalog work requires bounded exact known counters")
        if type(self.work_complete) is not bool:
            raise ValueError("catalog work completeness must be an exact boolean")

    def require_complete(self, catalog: ContextualArmCatalog, action_id: str) -> None:
        self.__post_init__()
        _catalog(catalog)
        action = catalog.envelope.select(action_id)
        if (
            not self.work_complete
            or self.attempted_updates != self.returned_updates
            or self.attempted_rows != self.returned_rows
        ):
            raise ValueError("incomplete or unknown catalog work has no utility")
        bank_rows = catalog.envelope.actions[2].replay_examples
        events, remainder = divmod(self.replay_examples_charged, bank_rows)
        wakes = self.attempted_updates - events
        if (
            remainder
            or self.replay_examples_charged > catalog.envelope.max_replay_examples
            or not 0 <= wakes <= catalog.envelope.max_wake_updates
            or self.replay_examples_charged > self.attempted_rows
            or (action.wake_updates == 0 and self.attempted_updates != 0)
            or (action.replay_examples == 0 and self.replay_examples_charged != 0)
            or (action.replay_examples > 0 and wakes > events)
            or (self.attempted_updates == 0) != (self.attempted_rows == 0)
            or self.attempted_rows < self.attempted_updates
            or (self.prediction_calls == 0) != (self.prediction_rows == 0)
            or self.prediction_rows < self.prediction_calls
        ):
            raise ValueError("catalog native/replay work differs from the declared action envelope")


@dataclass(frozen=True)
class FittedCatalogController:
    catalog: ContextualArmCatalog = field(kw_only=True)
    learner_version: str
    state_digest: str
    objective: ControllerObjective
    fitted_at: int
    estimates: tuple[CatalogEstimate, ...]
    choices: tuple[str, str]

    def __post_init__(self):
        _catalog(self.catalog)
        require_identifier(self.learner_version, "learner_version")
        require_identifier(self.state_digest, "state_digest")
        require_tick(self.fitted_at, "fitted_at")
        if type(self.objective) is not ControllerObjective:
            raise ValueError("catalog fit requires original prospective objective")
        self.objective.__post_init__()
        if type(self.choices) is not tuple or self.choices != fit_catalog_choices(
            self.catalog, self.estimates
        ):
            raise ValueError("catalog choices differ from complete empirical estimates")


def choose_catalog_contextual(
    policy: FittedCatalogController,
    observation: ReferenceObservation,
    *,
    envelope: AdaptationEnvelope,
    remaining_replay_examples: int,
) -> ReferenceDecision:
    if type(policy) is not FittedCatalogController:
        raise ValueError("catalog selection requires an exact frozen catalog policy")
    policy.__post_init__()
    if type(envelope) is not AdaptationEnvelope or envelope != policy.catalog.envelope:
        raise ValueError("catalog selection requires its original matching envelope")
    envelope.__post_init__()
    require_tick(remaining_replay_examples, "remaining replay credit")
    if remaining_replay_examples > envelope.max_replay_examples:
        raise ValueError("remaining replay credit exceeds the original envelope")
    context = training_context(observation)
    if observation.actor_version != policy.learner_version:
        raise ValueError("catalog policy belongs to a stale learner version")
    if context is None or observation.ready_updates == 0 or observation.remaining_wake == 0:
        return ReferenceDecision("wait", "no_arrived_context_ready_work_or_credit")
    action_id = policy.choices[CONTEXTS.index(context)]
    if envelope.select(action_id).replay_examples > remaining_replay_examples:
        return ReferenceDecision("wait", "insufficient_declared_replay_credit")
    return ReferenceDecision(action_id, "catalog_contextual_" + context)
