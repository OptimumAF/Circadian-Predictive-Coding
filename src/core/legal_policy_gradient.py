"""Declared-support conditioning and owned reward-trajectory evidence.

Inputs are a raw categorical distribution, declared legal indices, and an
existing reward episode. Outputs normalize that support without fallback and
bind each recorded behavior vector to its raw distribution/support. This module
does not observe an environment, attest legality/provenance, train a learner,
sample actions, change returns, authorize data roles, or perform IO.
"""

from dataclasses import dataclass
from math import fsum, isfinite

from src.core.categorical_probability_scores import validate_class_count
from src.core.policy_gradient_episode import PolicyGradientEpisode, copy_policy_gradient_episode


_FIELDS = {"episode", "legal_actions", "raw_probabilities"}


def _raw(values: tuple[float, ...]) -> tuple[float, ...]:
    if type(values) is not tuple:
        raise ValueError("raw probabilities must be an exact tuple")
    validate_class_count(len(values))
    if any(type(value) not in (int, float) or not 0 <= value <= 1 for value in values):
        raise ValueError("raw probabilities must be native finite numbers in [0, 1]")
    owned = tuple(float(value) for value in values)
    if not all(isfinite(value) for value in owned) or abs(fsum(owned) - 1.0) > 1e-6:
        raise ValueError("raw probabilities must sum to one within absolute 1e-6")
    return owned


def _support(actions: tuple[int, ...], count: int) -> tuple[int, ...]:
    if (
        type(actions) is not tuple
        or not actions
        or len(actions) > count
        or any(type(action) is not int or not 0 <= action < count for action in actions)
        or actions != tuple(sorted(set(actions)))
    ):
        raise ValueError(
            "legal_actions must be a nonempty exact tuple of sorted unique native indices"
        )
    return tuple(action for action in actions)


def condition_action_probabilities(
    raw: tuple[float, ...], legal_actions: tuple[int, ...]
) -> tuple[float, ...]:
    """Condition on exactly the supplied support; refuse zero legal mass.

    Why this: adding an epsilon or uniform fallback would change the declared
    behavior distribution and conceal a policy/support inconsistency.
    """
    probabilities = _raw(raw)
    support = _support(legal_actions, len(probabilities))
    mass = fsum(probabilities[action] for action in support)
    if not isfinite(mass) or mass <= 0:
        raise ValueError("declared legal support must have positive finite probability mass")
    return tuple(
        value / mass if action in support else 0.0 for action, value in enumerate(probabilities)
    )


@dataclass(frozen=True)
class LegalPolicyGradientEpisode:
    """A base trajectory plus owned per-step raw probabilities and support.

    Role remains evidence. Consumers must separately require train authority;
    neither the declared support nor an owned copy proves physical provenance.
    """

    episode: PolicyGradientEpisode
    legal_actions: tuple[tuple[int, ...], ...]
    raw_probabilities: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        _exact(self)
        episode = copy_policy_gradient_episode(self.episode)
        if episode.termination == "illegal_action":
            raise ValueError("legal-conditioned episodes cannot end with illegal_action")
        steps = len(episode.transitions)
        if (
            type(self.legal_actions) is not tuple
            or len(self.legal_actions) != steps
            or type(self.raw_probabilities) is not tuple
            or len(self.raw_probabilities) != steps
        ):
            raise ValueError("legal support and raw probabilities require one exact tuple per step")
        supports, raw_rows = [], []
        for transition, raw, legal, behavior in zip(
            episode.transitions,
            self.raw_probabilities,
            self.legal_actions,
            episode.behavior_probabilities,
            strict=True,
        ):
            owned_raw = _raw(raw)
            if len(owned_raw) != episode.num_classes:
                raise ValueError("raw probabilities must match the episode class count")
            support = _support(legal, episode.num_classes)
            conditional = condition_action_probabilities(owned_raw, support)
            if transition.action not in support:
                raise ValueError("the selected action must occur in declared legal support")
            if conditional[transition.action] <= 0:
                raise ValueError("the selected action must have positive conditional probability")
            if any(saved != 0 for action, saved in enumerate(behavior) if action not in support):
                raise ValueError(
                    "behavior probabilities outside declared legal support must be zero"
                )
            if any(
                abs(saved - expected) > 1e-12
                for saved, expected in zip(behavior, conditional, strict=True)
            ):
                raise ValueError("behavior probabilities differ from declared legal conditioning")
            supports.append(support)
            raw_rows.append(owned_raw)
        object.__setattr__(self, "episode", episode)
        object.__setattr__(self, "legal_actions", tuple(supports))
        object.__setattr__(self, "raw_probabilities", tuple(raw_rows))


def _exact(value: object) -> None:
    if type(value) is not LegalPolicyGradientEpisode or set(vars(value)) != _FIELDS:
        raise ValueError("expected an exact LegalPolicyGradientEpisode with only its three fields")


def copy_legal_policy_gradient_episode(
    value: LegalPolicyGradientEpisode,
) -> LegalPolicyGradientEpisode:
    """Reconstruct the complete wrapper, including all bypassed frozen changes."""
    _exact(value)
    return LegalPolicyGradientEpisode(value.episode, value.legal_actions, value.raw_probabilities)
