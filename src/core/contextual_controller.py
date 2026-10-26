"""Small empirical contextual selector over arrived TRAIN feedback.

One full-horizon utility per context/arm is a deliberately weak estimate. It
does not identify individual update effects, hidden state or delayed credit.
No model, target arrays, permission, budget mutation or evaluation access here.
"""

from dataclasses import dataclass
from math import isfinite

from src.core.controller_objective import ControllerObjective
from src.core.experience import require_identifier, require_tick
from src.core.reference_controller import (
    ArrivedTrainingFeedback,
    ReferenceDecision,
    ReferenceObservation,
)

CONTEXTS = ("clean", "mistakes")
ARMS = ("wait", "wake1")


def training_context(observation: ReferenceObservation) -> str | None:
    if type(observation) is not ReferenceObservation:
        raise ValueError("context requires exact causal TRAIN observation")
    observation.__post_init__()
    feedback = observation.feedback
    if feedback is not None and type(feedback) is not ArrivedTrainingFeedback:
        raise ValueError("classifier contextual policies cannot consume task-return feedback")
    return None if feedback is None else ("clean" if feedback.mistakes == 0 else "mistakes")


@dataclass(frozen=True)
class ContextualChoice:
    clean: str
    mistakes: str

    def __post_init__(self):
        if any(type(a) is not str or a not in ARMS for a in (self.clean, self.mistakes)):
            raise ValueError("context choices require finite wait/wake1 IDs")


@dataclass(frozen=True)
class ContextualEstimate:
    context: str
    action_id: str
    mean_utility: float
    samples: int = 1

    def __post_init__(self):
        if self.context not in CONTEXTS or self.action_id not in ARMS:
            raise ValueError("estimate requires a supported context/arm")
        if type(self.samples) is not int or self.samples != 1:
            raise ValueError("initial selector requires one finite trial per context/arm")
        if type(self.mean_utility) not in (int, float) or not isfinite(self.mean_utility):
            raise ValueError("estimate requires finite utility")


def fit_choices(estimates: tuple[ContextualEstimate, ...]) -> ContextualChoice:
    if (
        type(estimates) is not tuple
        or len(estimates) != 4
        or any(type(e) is not ContextualEstimate for e in estimates)
    ):
        raise ValueError("fit requires four complete context/arm estimates")
    for estimate in estimates:
        estimate.__post_init__()
    table = {(e.context, e.action_id): e.mean_utility for e in estimates}
    if set(table) != {(c, a) for c in CONTEXTS for a in ARMS}:
        raise ValueError("fit requires each context/arm exactly once")
    # Why this: a fixed wait tie-break avoids inventing useful work from ties.
    selected = ["wake1" if table[c, "wake1"] > table[c, "wait"] else "wait" for c in CONTEXTS]
    return ContextualChoice(*selected)


@dataclass(frozen=True)
class FittedController:
    learner_version: str
    state_digest: str
    objective: ControllerObjective
    fitted_at: int
    estimates: tuple[ContextualEstimate, ...]
    choices: ContextualChoice

    def __post_init__(self):
        require_identifier(self.learner_version, "learner_version")
        require_identifier(self.state_digest, "state_digest")
        require_tick(self.fitted_at, "fitted_at")
        if type(self.objective) is not ControllerObjective:
            raise ValueError("fit requires original prospective objective")
        self.objective.__post_init__()
        if type(self.choices) is not ContextualChoice or self.choices != fit_choices(
            self.estimates
        ):
            raise ValueError("fitted choices differ from complete empirical estimates")


def choose_contextual(
    policy: ContextualChoice | FittedController, observation: ReferenceObservation
) -> ReferenceDecision:
    context = training_context(observation)
    if type(policy) is FittedController:
        policy.__post_init__()
        if observation.actor_version != policy.learner_version:
            raise ValueError("fitted controller belongs to a stale learner version")
        choices = policy.choices
    elif type(policy) is ContextualChoice:
        choices = policy
        choices.__post_init__()
    else:
        raise ValueError("contextual selection requires an exact frozen policy")
    if context is None or observation.ready_updates == 0 or observation.remaining_wake == 0:
        return ReferenceDecision("wait", "no_arrived_context_ready_work_or_credit")
    return ReferenceDecision(getattr(choices, context), "contextual_" + context)
