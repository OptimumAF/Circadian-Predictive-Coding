"""Complete bounded TRAIN schedule proof without repeated observation serialization.

Construction validates exact records, primitive values and complete episodes.
Checks re-read every mutable record field, while retained exact native tuples of
immutable floats prove unchanged observations by identity. This is an in-process
ownership witness, not a learner, persisted format, digest or history chain.
"""

from dataclasses import dataclass, fields
from typing import Any, cast

from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    copy_episode,
)

_EPISODE_FIELDS = frozenset(field.name for field in fields(DemonstrationEpisode))
_TRANSITION_FIELDS = frozenset(field.name for field in fields(DemonstrationTransition))
_EPISODE_SCALARS = (("episode_id", str), ("context", str), ("role", str), ("num_classes", int))
_TRANSITION_SCALARS = (
    ("step", int),
    ("action", int),
    ("reward", float),
    ("terminal", bool),
    ("num_classes", int),
)


@dataclass(frozen=True, slots=True)
class _RecordWitness:
    owner: Any
    kind: type
    dictionary: dict[str, Any]
    order: tuple[str, ...]
    scalars: tuple[tuple[str, type, Any], ...]
    tuples: tuple[tuple[str, tuple[Any, ...]], ...]

    def check(self) -> None:
        if type(self.owner) is not self.kind:
            raise ValueError("owned demonstration record type changed")
        values = vars(self.owner)
        if (
            values is not self.dictionary
            or tuple(values) != self.order
            or any(type(name) is not str for name in values)
            or any(values[name] is not expected for name, expected in self.tuples)
        ):
            raise ValueError("owned demonstration record, fields or tuple identity changed")
        for name, kind, expected in self.scalars:
            actual = values[name]
            if (
                type(actual) is not kind
                or (cast(float, actual).hex() if kind is float else actual) != expected
                or (kind is str and actual is not expected)
            ):
                raise ValueError("owned demonstration primitive type or value changed")


def _capture(
    record: Any,
    kind: type,
    expected_fields: frozenset[str],
    scalar_fields: tuple[tuple[str, type], ...],
    tuple_fields: tuple[str, ...],
) -> _RecordWitness:
    if (
        type(record) is not kind
        or set(vars(record)) != expected_fields
        or any(type(name) is not str for name in vars(record))
    ):
        raise ValueError("schedule witness requires exact complete demonstration records")
    values = vars(record)
    scalars = []
    for name, scalar_type in scalar_fields:
        value = values[name]
        if type(value) is not scalar_type:
            raise ValueError("schedule witness requires exact native primitive types")
        scalars.append(
            (name, scalar_type, cast(float, value).hex() if scalar_type is float else value)
        )
    for name in tuple_fields:
        value = values[name]
        if type(value) is not tuple or (
            name != "transitions" and any(type(x) is not float for x in value)
        ):
            raise ValueError("schedule witness requires exact tuples of immutable native values")
    return _RecordWitness(
        record,
        kind,
        values,
        tuple(values),
        tuple(scalars),
        tuple((name, values[name]) for name in tuple_fields),
    )


@dataclass(frozen=True, slots=True, init=False)
class DemonstrationScheduleWitness:
    """Lifetime proof of one owned1..512 complete TRAIN-episode schedule.

    No TTL: immutable leaves remain valid for this owner's lifetime. Any changed
    record/type/order/identity refuses; only a new owner may create a new witness.
    Private witness records are trusted authority, like the previous byte seal.
    """

    _owner: tuple[DemonstrationEpisode, ...]
    _records: tuple[_RecordWitness, ...]

    def __init__(self, episodes: tuple[DemonstrationEpisode, ...]) -> None:
        if (
            type(self) is not DemonstrationScheduleWitness
            or type(episodes) is not tuple
            or not 1 <= len(episodes) <= 512
        ):
            raise ValueError("schedule witness requires an exact tuple of1..512 TRAIN episodes")
        records = []
        for episode in episodes:
            records.append(
                _capture(
                    episode,
                    DemonstrationEpisode,
                    _EPISODE_FIELDS,
                    _EPISODE_SCALARS,
                    ("transitions",),
                )
            )
            if episode.role != "train" or not 1 <= len(episode.transitions) <= 16:
                raise ValueError("schedule witness requires complete1..16-row TRAIN episodes")
            records.extend(
                _capture(
                    transition,
                    DemonstrationTransition,
                    _TRANSITION_FIELDS,
                    _TRANSITION_SCALARS,
                    ("observation", "next_observation"),
                )
                for transition in episode.transitions
            )
            # Why: native tuple identity proves only immutability, not semantic
            # validity. Establish all continuity, dimensions, bounds and clocks once.
            copy_episode(episode)
        object.__setattr__(self, "_owner", episodes)
        object.__setattr__(self, "_records", tuple(records))

    def check(self, episodes: tuple[DemonstrationEpisode, ...]) -> None:
        if type(self) is not DemonstrationScheduleWitness or episodes is not self._owner:
            raise ValueError("owned demonstration schedule identity changed")
        for record in self._records:
            record.check()

    def check_copy(self, other: "DemonstrationScheduleWitness") -> None:
        """Prove a newly admitted copy matches all original values and field order.

        This full immutable-leaf walk occurs only when creating another owner,
        never on ordinary checks. Float hex preserves signed zero like pickle.
        """
        self.check(self._owner)
        other.check(other._owner)
        if len(self._records) != len(other._records):
            raise ValueError("copied demonstration schedule length changed")
        for source, copied in zip(self._records, other._records, strict=True):
            if (
                source.kind is not copied.kind
                or source.order != copied.order
                or source.scalars != copied.scalars
                or any(
                    kind is str and original is not replacement
                    for (_, kind, original), (_, _, replacement) in zip(
                        source.scalars, copied.scalars, strict=True
                    )
                )
            ):
                raise ValueError("copied demonstration record values changed")
            for (name, values), (other_name, other_values) in zip(
                source.tuples, copied.tuples, strict=True
            ):
                if (
                    name != other_name
                    or len(values) != len(other_values)
                    or (
                        name != "transitions"
                        and any(
                            a.hex() != b.hex() for a, b in zip(values, other_values, strict=True)
                        )
                    )
                ):
                    raise ValueError("copied demonstration tuple values changed")
