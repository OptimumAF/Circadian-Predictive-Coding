"""Exact TRAIN supervision consistency across bounded, distinct episode groups.

Own2..6 same-spec groups, compose the existing100-row kernel and retain every
intended global pair once. Outputs are descriptive scalar records and actual paid
kernel work. No labels, thresholds, model, RNG, fitting, callback or IO occurs.
"""

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from src.core import continuous_bc as bc
from src.core.continuous_bc_diagnostics import (
    ContinuousInputTargetConsistency,
    InputTargetPair,
    _InputTargetConsistencyWork,
    continuous_input_target_consistency,
)
from src.core.continuous_demonstration import (
    ContinuousEpisode,
    _text,
    continuous_training_pairs,
    copy_continuous_episode,
)

DEFINITION = "continuous_grouped_input_target_exact_consistency_v1"
_PAYLOAD_KEYS = (
    "owned_input_numeric_bytes",
    "owned_target_numeric_bytes",
    "logical_owned_array_bytes",
    "native_target_float32_numeric_bytes",
    "owned_native_target_numeric_bytes",
)


@dataclass(frozen=True)
class _Group:
    name: str
    episode: ContinuousEpisode
    features: bc.Array
    targets: bc.Array
    start: int


def _admit(groups) -> tuple[_Group, ...]:
    if type(groups) is not tuple or not 2 <= len(groups) <= 6:
        raise ValueError("grouped consistency requires an exact tuple of2..6 groups")
    originals = []
    names = []
    for item in groups:
        if type(item) is not tuple or len(item) != 2 or type(item[1]) is not ContinuousEpisode:
            raise ValueError("each group must be an exact(name,ContinuousEpisode) tuple")
        name, episode = item
        names.append(_text(name, "group name", 32))
        if (
            type(episode.role) is not str
            or episode.role != "train"
            or type(episode.transitions) is not tuple
            or not 1 <= len(episode.transitions) <= 50
        ):
            raise ValueError("groups require complete TRAIN episodes with1..50 rows each")
        originals.append(episode)
    if len(set(names)) != len(names):
        raise ValueError("group names must be distinct")
    episodes = tuple(copy_continuous_episode(value) for value in originals)
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError("group episode IDs must be distinct")
    spec = episodes[0].spec
    if (
        spec.observation_dim > 64
        or len(spec.action_units) != 4
        or spec.lower_bounds != (-1.0,) * 4
        or spec.upper_bounds != (1.0,) * 4
        or any(episode.spec != spec for episode in episodes)
        or sum(len(episode.transitions) for episode in episodes) > 198
    ):
        raise ValueError("groups require one normalized four-axis spec,D<=64,total<=198")
    bc._binary_gripper_axis(3, spec)
    owned = []
    start = 0
    for name, episode in zip(names, episodes, strict=True):
        features, targets = continuous_training_pairs(episode, role="train")
        values = bc._features(np.asarray(features, dtype=np.float64), spec)
        actions = bc._targets(np.asarray(targets, dtype=np.float64), len(values), spec)
        bc._binary_gripper_targets(actions, spec, 3)
        owned.append(_Group(name, episode, values, actions, start))
        start += len(values)
    return tuple(owned)


def _mapped_pairs(result, block) -> list[InputTargetPair]:
    count = sum(len(group.features) for group in block)
    if (
        type(result) is not ContinuousInputTargetConsistency
        or result.rows != count
        or result.input_dimension != block[0].episode.spec.observation_dim
        or result.action_dimension != 4
        or type(result.pairs) is not tuple
        or len(result.pairs) != count * (count - 1) // 2
    ):
        raise ValueError("returned consistency block shape differs from its admitted rows")
    global_rows = tuple(group.start + row for group in block for row in range(len(group.features)))
    split = len(block[0].features)
    selected = []
    ordinals = ((first, second) for first in range(count) for second in range(first + 1, count))
    for pair, (first, second) in zip(result.pairs, ordinals, strict=True):
        if (
            type(pair) is not InputTargetPair
            or type(pair.first_row) is not int
            or type(pair.second_row) is not int
            or (pair.first_row, pair.second_row) != (first, second)
        ):
            raise ValueError("returned consistency pair order differs from admitted rows")
        if len(block) == 1 or first < split <= second:
            selected.append(
                replace(pair, first_row=global_rows[first], second_row=global_rows[second])
            )
    return selected


def _publish(report, pairs, nearest, kernel_work, costs) -> None:
    pairs.sort(key=lambda pair: (pair.first_row, pair.second_row))
    report["pairs"] = [asdict(pair) for pair in pairs]
    report["nearest_other_rows"] = [None if value is None else value[1] for value in nearest]
    report["duplicate_input_pairs"] = sum(pair.duplicate_input for pair in pairs)
    report["raw_conflict_pairs"] = sum(pair.raw_target_conflict for pair in pairs)
    report["native_conflict_pairs"] = sum(pair.native_target_conflict for pair in pairs)
    report["opposite_gripper_pairs"] = sum(pair.opposite_gripper for pair in pairs)
    report["work"] = dict(asdict(kernel_work), **costs)
    report["work"].update(
        retained_pairs=len(pairs),
        serialized_pair_scalar_values=17 * len(pairs),
        serialized_nearest_row_values=len(nearest),
        scope="kernel counters include all repeated pairwork; payload bytes sum returned calls only; excludes caller arrays, validation, copies, Python/JSON and RSS",
    )


def diagnose_grouped_input_targets(groups) -> dict[str, Any]:
    """Admit all inputs first; failed kernel work returns an honest paid prefix."""
    owned = _admit(groups)
    rows = sum(len(group.features) for group in owned)
    kernel_work = _InputTargetConsistencyWork()
    costs = dict(
        kernel_calls_attempted=0,
        kernel_calls_returned=0,
        kernel_rows=0,
        discarded_repeated_pairs=0,
        global_nearest_comparisons=0,
        **dict.fromkeys(_PAYLOAD_KEYS, 0),
    )
    report: dict[str, Any] = dict(
        definition=DEFINITION,
        complete=False,
        error=None,
        unknown_unreturned_kernel_work=False,
        rows=rows,
        input_dimension=owned[0].episode.spec.observation_dim,
        action_dimension=4,
        groups=[
            dict(
                name=group.name,
                episode_id=group.episode.episode_id,
                context=group.episode.context,
                role=group.episode.role,
                rows=len(group.features),
                start=group.start,
            )
            for group in owned
        ],
        row_bindings=[
            dict(
                global_row=group.start + row,
                group=group.name,
                episode_id=group.episode.episode_id,
                context=group.episode.context,
                row=row,
                step=transition.step,
            )
            for group in owned
            for row, transition in enumerate(group.episode.transitions)
        ],
        blocks=[],
    )
    pairs: list[InputTargetPair] = []
    nearest: list[tuple[float, int] | None] = [None] * rows
    blocks: list[tuple[_Group, ...]] = [(group,) for group in owned]
    blocks.extend(
        (first, second) for index, first in enumerate(owned) for second in owned[index + 1 :]
    )
    try:
        for block in blocks:
            input_rows = sum(len(group.features) for group in block)
            entry = dict(
                groups=[group.name for group in block],
                input_rows=input_rows,
                attempted=True,
                returned=False,
                validated=False,
            )
            report["blocks"].append(entry)
            costs["kernel_calls_attempted"] += 1
            costs["kernel_rows"] += input_rows
            result = continuous_input_target_consistency(
                np.concatenate([group.features for group in block], axis=0),
                np.concatenate([group.targets for group in block], axis=0),
                owned[0].episode.spec,
                _work=kernel_work,
            )
            entry["returned"] = True
            costs["kernel_calls_returned"] += 1
            for key in _PAYLOAD_KEYS:
                value = result.work[key]
                if type(value) is not int or value < 0:
                    raise ValueError(
                        "returned kernel payload work must be nonnegative native counts"
                    )
                costs[key] += value
            selected = _mapped_pairs(result, block)
            entry.update(
                kernel_pairs=len(result.pairs),
                retained_pairs=len(selected),
                discarded_repeated_pairs=len(result.pairs) - len(selected),
            )
            costs["discarded_repeated_pairs"] += len(result.pairs) - len(selected)
            pairs.extend(selected)
            for pair in selected:
                for row, other in (
                    (pair.first_row, pair.second_row),
                    (pair.second_row, pair.first_row),
                ):
                    costs["global_nearest_comparisons"] += 1
                    choice = (pair.input_l2, other)
                    previous = nearest[row]
                    if previous is None or choice < previous:
                        nearest[row] = choice
            entry["validated"] = True
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["unknown_unreturned_kernel_work"] = not report["blocks"][-1]["returned"]
    finally:
        _publish(report, pairs, nearest, kernel_work, costs)
    return report
