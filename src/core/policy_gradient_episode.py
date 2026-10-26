"""Owned reward trajectories and unnormalized policy-gradient return weights.

Inputs are physical transition records, behavior probabilities, and a discount.
Outputs preserve declared provenance and finite episode endings. No environment,
learner, objective reduction, sampling, replay, or IO is implemented here.
"""

from dataclasses import dataclass
import math

from src.core.categorical_probability_scores import validate_class_count
from src.core.demonstration_episode import DemonstrationTransition

_EPISODE_FIELDS = {
    "episode_id",
    "context",
    "role",
    "transitions",
    "behavior_probabilities",
    "termination",
    "num_classes",
}
_TRANSITION_FIELDS = {
    "step",
    "observation",
    "action",
    "reward",
    "next_observation",
    "terminal",
    "num_classes",
}


def _exact_fields(value: object, kind: type, names: set[str]) -> None:
    if type(value) is not kind or set(vars(value)) != names:
        raise ValueError(f"expected an exact {kind.__name__} with only its declared fields")


def _text(value: object, name: str, limit: int) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty trimmed text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be valid UTF-8 text") from error
    if len(encoded) > limit:
        raise ValueError(f"{name} exceeds {limit} UTF-8 bytes")


def _copy_transition(value: DemonstrationTransition) -> DemonstrationTransition:
    _exact_fields(value, DemonstrationTransition, _TRANSITION_FIELDS)
    return DemonstrationTransition(
        value.step,
        value.observation,
        value.action,
        value.reward,
        value.next_observation,
        value.terminal,
        value.num_classes,
    )


def _probability(value: object) -> float:
    if type(value) not in (int, float) or not 0 <= value <= 1:  # type: ignore[operator]
        raise ValueError("behavior probabilities must be native finite numbers in [0, 1]")
    return float(value)  # type: ignore[arg-type]


def _copy_probabilities(
    values: tuple[tuple[float, ...], ...],
    transitions: tuple[DemonstrationTransition, ...],
    num_classes: int,
) -> tuple[tuple[float, ...], ...]:
    if type(values) is not tuple or len(values) != len(transitions):
        raise ValueError("behavior probabilities must be a tuple with one vector per step")
    owned = []
    for vector, transition in zip(values, transitions):
        if type(vector) is not tuple or len(vector) != num_classes:
            raise ValueError(
                "each behavior probability vector must be a tuple of num_classes values"
            )
        row = tuple(_probability(value) for value in vector)
        if abs(math.fsum(row) - 1.0) > 1e-6:
            raise ValueError("behavior probability vector must sum to one within absolute 1e-6")
        if row[transition.action] <= 0:
            raise ValueError("the selected action must have positive behavior probability")
        owned.append(row)
    return tuple(owned)


@dataclass(frozen=True)
class PolicyGradientEpisode:
    episode_id: str
    context: str
    role: str
    transitions: tuple[DemonstrationTransition, ...]
    behavior_probabilities: tuple[tuple[float, ...], ...]
    termination: str
    num_classes: int = 5

    def __post_init__(self) -> None:
        _exact_fields(self, PolicyGradientEpisode, _EPISODE_FIELDS)
        _text(self.episode_id, "episode_id", 64)
        _text(self.context, "context", 32)
        count = validate_class_count(self.num_classes)
        if type(self.role) is not str or self.role not in (
            "train",
            "final_test",
            "probe_train",
            "probe_test",
        ):
            raise ValueError("episode role must be train, final_test, probe_train or probe_test")
        if type(self.termination) is not str or self.termination not in (
            "terminal",
            "truncated",
            "illegal_action",
        ):
            raise ValueError("termination must be terminal, truncated or illegal_action")
        transitions = self._owned_transitions(count)
        probabilities = _copy_probabilities(self.behavior_probabilities, transitions, count)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "behavior_probabilities", probabilities)

    def _owned_transitions(self, count: int) -> tuple[DemonstrationTransition, ...]:
        if type(self.transitions) is not tuple or not 1 <= len(self.transitions) <= 16:
            raise ValueError("transitions must be a tuple of 1..16 physical steps")
        transitions = tuple(_copy_transition(value) for value in self.transitions)
        dimension = len(transitions[0].observation)
        for index, transition in enumerate(transitions):
            if transition.num_classes != count:
                raise ValueError("episode and transition class counts must agree")
            if transition.step != index or len(transition.observation) != dimension:
                raise ValueError("steps must be consecutive with fixed observation dimensions")
            expected_terminal = self.termination == "terminal" and index == len(transitions) - 1
            if transition.terminal != expected_terminal:
                raise ValueError("transition terminal flags do not match the declared ending")
            if index and transitions[index - 1].next_observation != transition.observation:
                raise ValueError("adjacent transition observations must be continuous")
        last = transitions[-1]
        if self.termination == "illegal_action" and (
            last.reward != 0 or last.next_observation != last.observation
        ):
            raise ValueError("illegal_action must end with zero reward and unchanged observation")
        return transitions


def copy_policy_gradient_episode(episode: PolicyGradientEpisode) -> PolicyGradientEpisode:
    """Own and revalidate every field, including bypassed frozen-record changes."""
    _exact_fields(episode, PolicyGradientEpisode, _EPISODE_FIELDS)
    return PolicyGradientEpisode(
        episode.episode_id,
        episode.context,
        episode.role,
        episode.transitions,
        episode.behavior_probabilities,
        episode.termination,
        episode.num_classes,
    )


def policy_gradient_weights(
    episode: PolicyGradientEpisode, discount: float = 1.0
) -> tuple[float, ...]:
    """Return gamma^t times discounted reward-to-go, without mean or baseline.

    Why: each reward k contributes gamma^k to every preceding eligible score
    term. Using an unweighted reward-to-go would silently change this objective.
    """
    owned = copy_policy_gradient_episode(episode)
    if type(discount) not in (int, float) or not 0 <= discount <= 1:
        raise ValueError("discount must be a finite native number in [0, 1]")
    gamma = float(discount)
    rewards = tuple(transition.reward for transition in owned.transitions)
    return tuple(
        gamma**index
        * math.fsum(
            gamma ** (later - index) * rewards[later] for later in range(index, len(rewards))
        )
        for index in range(len(rewards))
    )
