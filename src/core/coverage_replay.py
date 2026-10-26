"""Bounded latest-episode retention with declared-context replay coverage.

Inputs are existing encoded train-episode rows, an exact episode schema, and
1..6 predeclared context/cue keys. Outputs are owned SharedReplaySelections and
ordinary encoded-row retention facts. No learner, clock, RNG, IO, objective or
physical provenance is added. The existing codec admits train episodes only;
its wire has no role field and cannot attest to a packet's physical origin.
"""

from dataclasses import replace

import numpy as np

from src.core.circadian_predictive_coding import ReplayRetentionBudget, replay_sample_id
from src.core.demonstration_episode import EpisodeReplaySchema, decode_episode
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.shared_replay_schedule import Array, SharedReplayBuffer, SharedReplaySelection


def _contexts(value: tuple[str, ...]) -> tuple[str, ...]:
    if type(value) is not tuple or not 1 <= len(value) <= 6:
        raise ValueError("coverage contexts must be an exact tuple containing 1..6 keys")
    for context in value:
        if type(context) is not str or not context or context != context.strip():
            raise ValueError("coverage context keys must be nonempty trimmed strings")
        try:
            encoded = context.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("coverage context keys must be valid UTF-8") from error
        if len(encoded) > 32:
            raise ValueError("coverage context keys must contain at most 32 UTF-8 bytes")
    if len(set(value)) != len(value):
        raise ValueError("coverage context keys must be unique")
    return value


class CoverageReplayBuffer(SharedReplayBuffer):
    """Keep one complete latest packet per key and rotate among eligible keys.

    Why this: replacement is keyed by declared training context, so repeated
    packets from a recent context cannot evict the last packet from another key.
    Storage counts encoded rows and array bytes, as in SharedReplayBuffer.
    """

    def __init__(
        self,
        input_dim: int,
        budget: ReplayRetentionBudget,
        *,
        schema: EpisodeReplaySchema,
        contexts: tuple[str, ...],
    ) -> None:
        declared = _contexts(contexts)
        if type(schema) is not EpisodeReplaySchema:
            raise ValueError("coverage replay requires an exact EpisodeReplaySchema")
        if type(budget) is not ReplayRetentionBudget:
            raise ValueError("coverage replay requires an exact ReplayRetentionBudget")
        geometry, limits = replace(schema), replace(budget)
        if type(input_dim) is not int or input_dim != geometry.encoded_dim:
            raise ValueError("coverage input_dim must match the encoded episode schema")
        if limits.max_examples < len(declared) or limits.max_bytes < len(declared) * 8 * (
            input_dim + 1
        ):
            raise ValueError("coverage budget must hold one whole encoded episode per declared key")
        super().__init__(input_dim, limits, ReplayRetentionPolicy("recent_fifo"))
        self._schema, self._contexts = geometry, declared
        self._coverage_ids: dict[str, str] = {}
        self._coverage_cursor: str | None = None

    @property
    def coverage_keys(self) -> tuple[str, ...]:
        """Currently retained context/cue keys in their predeclared order."""
        return tuple(context for context in self._contexts if context in self._coverage_ids)

    @property
    def coverage_cursor(self) -> str | None:
        """The last successfully selected key, independent of packet replacement."""
        return self._coverage_cursor

    def observe_train_batch(self, inputs: Array, targets: Array) -> None:
        """Atomically replace latest whole packets after validating every row."""
        if (
            type(inputs) is not np.ndarray
            or inputs.dtype != np.float64
            or inputs.ndim != 2
            or not 1 <= len(inputs) <= 512
            or inputs.shape[1] != self.input_dim
            or type(targets) is not np.ndarray
            or targets.dtype != np.float64
            or targets.shape != (len(inputs), 1)
        ):
            raise ValueError(
                "coverage replay requires 1..512 float64 encoded rows and dummy targets"
            )
        staged = []
        for index in range(len(inputs)):
            row, target = inputs[index : index + 1].copy(), targets[index : index + 1].copy()
            episode = decode_episode(row, target, self._schema)
            if episode.role != "train" or episode.context not in self._contexts:
                raise ValueError("coverage replay requires a declared train context/cue key")
            staged.append((episode.context, replay_sample_id(row, target), row, target))
        retained, membership = self._rows.copy(), self._coverage_ids.copy()
        for context, sample_id, row, target in staged:
            previous = membership.get(context)
            if previous is not None:
                retained.pop(previous)
            retained[sample_id] = (row, target)
            membership[context] = sample_id
        self._rows, self._coverage_ids = retained, membership

    def select_coverage(self, max_updates: int) -> SharedReplaySelection:
        """Select each eligible key at most once, starting after the last key."""
        if type(max_updates) is not int or max_updates <= 0:
            raise ValueError("coverage replay update budget must be a positive native integer")
        start = (
            0 if self._coverage_cursor is None else self._contexts.index(self._coverage_cursor) + 1
        )
        order = (*self._contexts[start:], *self._contexts[:start])
        keys = tuple(context for context in order if context in self._coverage_ids)[:max_updates]
        ids = tuple(self._coverage_ids[context] for context in keys)
        selection = SharedReplaySelection(
            ids, tuple((self._rows[key][0].copy(), self._rows[key][1].copy()) for key in ids)
        )
        if keys:
            self._coverage_cursor = keys[-1]
        return selection
