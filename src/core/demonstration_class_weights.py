"""Role-specific per-episode weights for explicit demonstration objectives.

Inputs are complete released demonstration labels or an explicit bounded weight
tuple. Outputs are owned native floats. No learner, metric, evaluation access,
normalization, class selection, RNG, or IO lives here.
"""

from dataclasses import fields
import math

from src.core.categorical_probability_scores import validate_class_count
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    copy_episode,
)


def validate_class_weights(weights: tuple[float, ...], num_classes: int) -> tuple[float, ...]:
    """Own an exact K-tuple of finite positive native weights, without rescaling."""
    count = validate_class_count(num_classes)
    if type(weights) is not tuple or len(weights) != count:
        raise ValueError("class weights must be an exact tuple of num_classes values")
    owned = []
    for value in weights:
        if type(value) not in (int, float) or not 1e-6 <= value <= 1e6 or not math.isfinite(value):
            raise ValueError("class weights must be native finite numbers in [1e-6, 1e6]")
        owned.append(float(value))
    return tuple(owned)


def _exact_record(
    value: object, kind: type[DemonstrationEpisode] | type[DemonstrationTransition]
) -> None:
    if type(value) is not kind or set(vars(value)) != {field.name for field in fields(kind)}:
        raise ValueError(f"expected an exact complete {kind.__name__} record")


def _episode_class_weights(episode: DemonstrationEpisode, role: str) -> tuple[float, ...]:
    """Equalize target-class mass within this complete train episode only.

    Why: absent classes have no target term, so weight one preserves a neutral
    explicit value without inventing labels or consulting another data role.
    """
    _exact_record(episode, DemonstrationEpisode)
    # Inspect only a bounded complete tuple; existing copy handles other geometry.
    if type(episode.transitions) is tuple and 1 <= len(episode.transitions) <= 16:
        for transition in episode.transitions:
            _exact_record(transition, DemonstrationTransition)
    owned = copy_episode(episode)
    if owned.role != role:
        raise ValueError("class weights require the explicitly admitted demonstration role")
    counts = [0] * owned.num_classes
    for transition in owned.transitions:
        counts[transition.action] += 1
    present = sum(count > 0 for count in counts)
    rows = len(owned.transitions)
    return tuple(float(rows / (present * count)) if count else 1.0 for count in counts)


def episode_class_weights(episode: DemonstrationEpisode) -> tuple[float, ...]:
    """Weights from one complete released TRAIN episode only."""
    return _episode_class_weights(episode, "train")


def probe_episode_class_weights(episode: DemonstrationEpisode) -> tuple[float, ...]:
    """Weights from one complete reserved probe-training episode only.

    Why: the disposable probe has separate release authority; original source
    roles remain intact and no evaluation labels contribute to weights.
    """
    return _episode_class_weights(episode, "probe_train")
