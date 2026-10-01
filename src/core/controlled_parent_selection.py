"""Select explicit-proposal parents without replacing existing topology rules.

Inputs are validated original candidates/counts and immutable selector
settings. Outputs are parent indices and detached stable-ID decision/state
views. This module performs no scheduling, guarding, dataset access, scoring
or IO. Built-in split-capable sleep requires an explicit policy here.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianNetworkSnapshot,
    CircadianPredictiveCodingNetwork,
)
from src.core.neuron_adaptation import NeuronChangeProposal, PruneOutcome


SelectionMode = Literal["usage", "scheduled", "random"]
_HASH = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ParentSelectionSettings:
    mode: SelectionMode
    seed: int
    initial_cursor_id: int = 0

    def __post_init__(self) -> None:
        _validate_settings(self)


def _validate_settings(settings: Any) -> ParentSelectionSettings:
    if (
        type(settings) is not ParentSelectionSettings
        or type(settings.mode) is not str
        or settings.mode not in ("usage", "scheduled", "random")
    ):
        raise ValueError("parent selection mode must be usage, scheduled or random")
    if type(settings.seed) is not int or settings.seed < 0:
        raise ValueError("parent selection seed must be a nonnegative Python integer")
    if type(settings.initial_cursor_id) is not int or settings.initial_cursor_id < 0:
        raise ValueError("parent selection cursor must be a nonnegative Python integer")
    return settings


@dataclass(frozen=True)
class ParentSelectionDecision:
    mode: SelectionMode
    eligible_parent_ids: tuple[int, ...]
    preferred_parent_ids: tuple[int, ...]
    selected_parent_ids: tuple[int, ...]
    cursor_before: int
    cursor_after: int
    rng_sha256_before: str
    rng_sha256_after: str


@dataclass(frozen=True)
class ParentSelectionState:
    settings: ParentSelectionSettings
    cursor_id: int
    selection_calls: int
    rng_sha256: str
    last_decision: ParentSelectionDecision | None


def _rng_hash(rng: np.random.Generator) -> str:
    state = json.dumps(rng.bit_generator.state, sort_keys=True, allow_nan=False)
    return sha256(state.encode("utf-8")).hexdigest()


class ParentControlledCircadianNetwork(CircadianPredictiveCodingNetwork):
    """Use original explicit-proposal eligibility with controlled parent ranking.

    Why this: an isolated subclass preserves earlier pinned core behavior
    and source identities while inheriting function-preserving split/prune.
    The selector RNG is separate from the original split-noise RNG.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seed: int,
        circadian_config: CircadianConfig | None = None,
        min_hidden_dim: int = 4,
        max_hidden_dim: int | None = None,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
        *,
        parent_selection: ParentSelectionSettings,
    ) -> None:
        _validate_settings(parent_selection)
        super().__init__(
            input_dim,
            hidden_dim,
            seed,
            circadian_config,
            min_hidden_dim,
            max_hidden_dim,
            hidden_dims,
        )
        self._parent_selection_settings = parent_selection
        self._parent_selection_rng = np.random.Generator(np.random.PCG64(parent_selection.seed))
        self._parent_selection_cursor = parent_selection.initial_cursor_id
        self._parent_selection_calls = 0
        self._last_parent_selection: ParentSelectionDecision | None = None

    def get_parent_selection_state(self) -> ParentSelectionState:
        return ParentSelectionState(
            self._parent_selection_settings,
            self._parent_selection_cursor,
            self._parent_selection_calls,
            _rng_hash(self._parent_selection_rng),
            self._last_parent_selection,
        )

    def _select_split_indices(
        self, max_split_limit: int | None = None, excluded_indices: tuple[int, ...] = ()
    ) -> tuple[int, ...]:
        if max_split_limit == 0:
            return ()
        raise ValueError("controlled parent selection requires an explicit adaptation policy")

    def _rank_proposal_split_sources(
        self, eligible: NDArray[np.int64], add_count: int
    ) -> tuple[int, ...]:
        if add_count == 0:
            return ()
        settings = self._parent_selection_settings
        cursor_before = self._parent_selection_cursor
        rng_before = _rng_hash(self._parent_selection_rng)
        threshold, _ = self._resolve_split_prune_thresholds()
        threshold += self.config.split_hysteresis_margin
        preferred = eligible[self._hidden_chemical[eligible] >= threshold]
        fallback = eligible[self._hidden_chemical[eligible] < threshold]
        if settings.mode == "usage":
            chosen = super()._rank_proposal_split_sources(eligible, add_count)
        else:
            ordered = np.concatenate((self._order_tier(preferred), self._order_tier(fallback)))
            chosen = tuple(int(index) for index in ordered[:add_count])
        if len(chosen) != add_count or len(set(chosen)) != add_count:
            raise ValueError("parent selection differs from the validated count")
        selected_ids = tuple(int(self._neuron_ids[index]) for index in chosen)
        if settings.mode == "scheduled":
            self._parent_selection_cursor = selected_ids[-1] + 1
        self._parent_selection_calls += 1
        self._last_parent_selection = ParentSelectionDecision(
            settings.mode,
            tuple(sorted(int(self._neuron_ids[index]) for index in eligible)),
            tuple(sorted(int(self._neuron_ids[index]) for index in preferred)),
            selected_ids,
            cursor_before,
            self._parent_selection_cursor,
            rng_before,
            _rng_hash(self._parent_selection_rng),
        )
        return chosen

    def _order_tier(self, tier: NDArray[np.int64]) -> NDArray[np.int64]:
        stable = tier[np.argsort(self._neuron_ids[tier], kind="stable")]
        if self._parent_selection_settings.mode == "random":
            return self._parent_selection_rng.permutation(stable)
        after = self._neuron_ids[stable] >= self._parent_selection_cursor
        return np.concatenate((stable[after], stable[~after]))

    def apply_neuron_proposals(
        self, proposals: list[NeuronChangeProposal], *, max_hidden_width: int | None = None
    ) -> PruneOutcome:
        # Why this: a later width check can follow mutable random/cyclic selection.
        saved = self.snapshot_state()
        try:
            outcome = super().apply_neuron_proposals(proposals, max_hidden_width=max_hidden_width)
            self._validate_parent_state(self.__dict__)
            return outcome
        except Exception:
            self.restore_state(saved)
            raise

    def restore_state(self, snapshot: CircadianNetworkSnapshot) -> None:
        if not isinstance(snapshot, CircadianNetworkSnapshot) or type(snapshot.state) is not dict:
            raise ValueError("parent selection requires a compatible circadian snapshot")
        settings = _validate_settings(snapshot.state.get("_parent_selection_settings"))
        if settings != self._parent_selection_settings:
            raise ValueError("parent selection snapshot settings are incompatible")
        self._validate_parent_state(snapshot.state)
        super().restore_state(snapshot)

    def _validate_sleep_post_state(self) -> None:
        super()._validate_sleep_post_state()
        self._validate_parent_state(self.__dict__)

    @staticmethod
    def _validate_parent_state(state: dict[str, Any]) -> None:
        settings = _validate_settings(state.get("_parent_selection_settings"))
        rng = state.get("_parent_selection_rng")
        cursor, calls = state.get("_parent_selection_cursor"), state.get("_parent_selection_calls")
        if (
            not isinstance(rng, np.random.Generator)
            or type(rng.bit_generator) is not np.random.PCG64
            or type(cursor) is not int
            or cursor < 0
            or type(calls) is not int
            or calls < 0
        ):
            raise ValueError("parent selection snapshot RNG/cursor/count is invalid")
        decision = state.get("_last_parent_selection")
        if calls == 0:
            if decision is not None or cursor != settings.initial_cursor_id:
                raise ValueError("parent selection initial snapshot is inconsistent")
        else:
            ParentControlledCircadianNetwork._validate_decision(decision, settings, cursor, rng)
        if settings.mode != "scheduled" and cursor != settings.initial_cursor_id:
            raise ValueError("parent selection nonscheduled cursor changed")
        if (calls == 0 or settings.mode != "random") and _rng_hash(rng) != _rng_hash(
            np.random.Generator(np.random.PCG64(settings.seed))
        ):
            raise ValueError("parent selection initial/nonrandom RNG changed")

    @staticmethod
    def _validate_decision(
        decision: Any,
        settings: ParentSelectionSettings,
        cursor: int,
        rng: np.random.Generator,
    ) -> None:
        if type(decision) is not ParentSelectionDecision or decision.mode != settings.mode:
            raise ValueError("parent selection snapshot decision is incompatible")
        groups = (
            decision.eligible_parent_ids,
            decision.preferred_parent_ids,
            decision.selected_parent_ids,
        )
        if (
            any(
                type(ids) is not tuple
                or any(type(value) is not int or value < 0 for value in ids)
                or len(ids) != len(set(ids))
                for ids in groups
            )
            or not decision.selected_parent_ids
        ):
            raise ValueError("parent selection snapshot decision IDs are invalid")
        if (
            not set(decision.preferred_parent_ids).issubset(decision.eligible_parent_ids)
            or not set(decision.selected_parent_ids).issubset(decision.eligible_parent_ids)
            or type(decision.cursor_before) is not int
            or decision.cursor_before < 0
            or type(decision.cursor_after) is not int
            or decision.cursor_after != cursor
            or type(decision.rng_sha256_before) is not str
            or not _HASH.fullmatch(decision.rng_sha256_before)
            or type(decision.rng_sha256_after) is not str
            or decision.rng_sha256_after != _rng_hash(rng)
            or (
                settings.mode != "random"
                and decision.rng_sha256_before != decision.rng_sha256_after
            )
            or (settings.mode == "scheduled" and cursor != decision.selected_parent_ids[-1] + 1)
            or (settings.mode != "scheduled" and decision.cursor_before != cursor)
        ):
            raise ValueError("parent selection snapshot decision state is inconsistent")
