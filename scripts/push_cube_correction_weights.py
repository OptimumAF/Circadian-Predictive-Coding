"""Fixed positive prefix/group weights for declared148/198-row TRAIN packets.

Inputs are fixed five/six complete, already authorized TRAIN episodes in declared order.
Outputs are raw positive weights and detached descriptive recipe metadata. This
helper does not qualify labels, relabel roles, select data, fit, train, normalize
weights, create learners, use RNG or perform IO.
"""

from typing import Any

from src.core.continuous_demonstration import ContinuousEpisode, copy_continuous_episode

GROUPS = ("corrections", "push", "a", "b", "train20")
GROUP_ROWS = (50, 18, 15, 50, 15)
DEFINITION = "positive_train_prefix_group_weights_v1"
AGGREGATED_GROUPS = ("current_corrections", "corrections", "push", "a", "b", "train20")
AGGREGATED_GROUP_ROWS = (50, 50, 18, 15, 50, 15)
AGGREGATED_DEFINITION = "positive_train_aggregated_prefix_group_weights_v1"


def _own_groups(groups) -> tuple[ContinuousEpisode, ...]:
    if type(groups) is not tuple or len(groups) != len(GROUPS):
        raise ValueError("balanced correction weights require an exact five-group tuple")
    originals = []
    for item, name, rows in zip(groups, GROUPS, GROUP_ROWS, strict=True):
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or item[0] != name
            or type(item[1]) is not ContinuousEpisode
        ):
            raise ValueError("balanced groups must be ordered corrections, push, a, b, train20")
        episode = item[1]
        if (
            type(episode.role) is not str
            or episode.role != "train"
            or type(episode.transitions) is not tuple
            or len(episode.transitions) != rows
        ):
            raise ValueError("balanced groups require TRAIN episodes with50/18/15/50/15 rows")
        originals.append(episode)
    if any(episode.spec != originals[0].spec for episode in originals):
        raise ValueError("balanced TRAIN groups must share exactly the same declared spec")
    owned = tuple(copy_continuous_episode(episode) for episode in originals)
    if len({episode.episode_id for episode in owned}) != len(owned):
        raise ValueError("balanced TRAIN groups must have distinct episode IDs")
    return owned


def build_balanced_correction_weights(groups) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Return the single declared recipe without inspecting outcomes or metrics."""
    episodes = _own_groups(groups)
    # Why: each group has ideal raw mass98. The first correction and its49
    # remainder rows split that group's mass equally without dropping any row.
    values: tuple[float, ...] = (49.0,) + (1.0,) * 49
    for rows in GROUP_ROWS[1:]:
        values += (98.0 / rows,) * rows
    protocol = {
        "definition": DEFINITION,
        "group_order": list(GROUPS),
        "groups": [
            {"group": name, "episode_id": episode.episode_id, "rows": rows}
            for name, episode, rows in zip(GROUPS, episodes, GROUP_ROWS, strict=True)
        ],
        "values": list(values),
        "rule": "equal five-group mass; correction initial/remainder split equally",
        "ideal_group_mass": 0.2,
        "ideal_correction_initial_mass": 0.1,
    }
    return values, protocol


def _own_aggregated_groups(groups) -> tuple[ContinuousEpisode, ...]:
    if type(groups) is not tuple or len(groups) != len(AGGREGATED_GROUPS):
        raise ValueError("aggregated correction weights require an exact six-group tuple")
    first = groups[0]
    if (
        type(first) is not tuple
        or len(first) != 2
        or type(first[0]) is not str
        or first[0] != "current_corrections"
        or type(first[1]) is not ContinuousEpisode
    ):
        raise ValueError("aggregated groups must start with current_corrections")
    current = first[1]
    if (
        type(current.role) is not str
        or current.role != "train"
        or type(current.transitions) is not tuple
        or len(current.transitions) != 50
    ):
        raise ValueError("current corrections require a complete50-row TRAIN episode")
    previous = _own_groups(groups[1:])
    current = copy_continuous_episode(current)
    if current.spec != previous[0].spec:
        raise ValueError("all aggregated TRAIN groups must share the declared spec")
    owned = (current, *previous)
    if len({episode.episode_id for episode in owned}) != len(owned):
        raise ValueError("all aggregated TRAIN groups require distinct episode IDs")
    return owned


def build_aggregated_correction_weights(groups) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Prepend the same positive prefix rule, preserving every original weight."""
    episodes = _own_aggregated_groups(groups)
    # Why: current and original correction packets have the same supervision
    # type; retain the declared rule rather than introduce another tunable ratio.
    values: tuple[float, ...] = (49.0,) + (1.0,) * 49 + (49.0,) + (1.0,) * 49
    for rows in GROUP_ROWS[1:]:
        values += (98.0 / rows,) * rows
    protocol = {
        "definition": AGGREGATED_DEFINITION,
        "group_order": list(AGGREGATED_GROUPS),
        "groups": [
            {"group": name, "episode_id": episode.episode_id, "rows": rows}
            for name, episode, rows in zip(
                AGGREGATED_GROUPS, episodes, AGGREGATED_GROUP_ROWS, strict=True
            )
        ],
        "values": list(values),
        "rule": "equal six-group mass; both correction initial/remainder split equally",
        "ideal_group_mass": 1 / 6,
        "ideal_correction_initial_mass": 1 / 12,
    }
    return values, protocol
