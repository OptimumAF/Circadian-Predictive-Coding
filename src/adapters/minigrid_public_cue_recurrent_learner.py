"""Two native recurrent owners selected by the initial PUBLIC cue only.

Inputs are one ordered raw148 episode and K7 targets; outputs are the existing
CE diagnostic/probabilities and complete owned continuation. Live activation is
owned by one bounded session. No environment metadata, roles for raw arrays,
budgets, clock, replay, IO, optimizer or new recurrent mathematics lives here.
"""

from dataclasses import dataclass, fields, replace
from typing import Any, NoReturn

import numpy as np

from src.adapters import torch_recurrent_learner as native
from src.adapters.torch_recurrent_weighted import TORCH_RECURRENT_WEIGHTED_CE_ID
from src.app.coupled_attribution import _same_value
from src.core.action_memory import _feature_keys
from src.core.categorical_probability_scores import (
    validate_class_probabilities,
    validate_class_targets,
)
from src.core.demonstration_episode import DemonstrationEpisode, copy_episode, episode_arrays
from src.core.demonstration_class_weights import episode_class_weights
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from src.core.minigrid_context_encoding import context_public_history
from src.core.minigrid_invariant_encoding import _decode_coordinate, canonicalize_public_history

METHOD = "recurrent_public_cue"
ENCODING_ID = "minigrid_raw148_initial_public_cue_recurrent_heads_v1"
FAILURE_DEFINITION = "public_cue_recurrent_failed_work_v1"


def _record(value: Any, kind: type) -> None:
    if type(value) is not kind or set(vars(value)) != {f.name for f in fields(kind)}:
        raise ValueError(f"an exact complete {kind.__name__} is required")


@dataclass(frozen=True)
class PublicCueRecurrentPolicy:
    native_policy: native.TorchRecurrentPolicy = native.TorchRecurrentPolicy(148, 8, 7, 17)
    encoding_id: str = ENCODING_ID

    def __post_init__(self) -> None:
        policy = native._policy(self.native_policy)
        if policy.feature_dim != 148 or policy.num_classes != 7 or policy.max_episode_steps != 16:
            raise ValueError("public-cue recurrent heads require raw148/K7/max16")
        if type(self.encoding_id) is not str or self.encoding_id != ENCODING_ID:
            raise ValueError("public-cue recurrent encoding differs")
        object.__setattr__(self, "native_policy", policy)

    @property
    def method(self) -> str:
        return METHOD

    @property
    def feature_dim(self) -> int:
        return 148

    @property
    def num_classes(self) -> int:
        return 7


def copy_policy(value: PublicCueRecurrentPolicy) -> PublicCueRecurrentPolicy:
    _record(value, PublicCueRecurrentPolicy)
    return PublicCueRecurrentPolicy(value.native_policy, value.encoding_id)


@dataclass(frozen=True)
class FailureWork:
    operation: str
    head_index: int
    native_stages_attempted: int = 0
    native_stages_returned: int = 0
    session_starts_attempted: int = 0
    session_starts_returned: int = 0
    native_calls_attempted: int = 0
    native_calls_returned: int = 0
    native_rows_attempted: int = 0
    native_rows_returned: int = 0
    failure_work_may_be_unobserved: bool = True


def _failure(value: FailureWork | None) -> None:
    if value is None:
        return
    _record(value, FailureWork)
    if type(value.operation) is not str or value.operation not in ("train", "predict", "session"):
        raise ValueError("invalid recurrent failure operation")
    if type(value.head_index) is not int or value.head_index not in (0, 1):
        raise ValueError("invalid recurrent failed head")
    for f in fields(value)[2:-1]:
        if type(getattr(value, f.name)) is not int:
            raise ValueError("failure counters must be native integers")
    for stem in ("native_stages", "session_starts", "native_calls"):
        if not 0 <= getattr(value, stem + "_returned") <= getattr(value, stem + "_attempted") <= 1:
            raise ValueError("invalid recurrent failure counters")
    if not (0 <= value.native_rows_returned <= value.native_rows_attempted <= 16):
        raise ValueError("invalid recurrent failure rows")
    if (value.native_calls_attempted == 0) != (value.native_rows_attempted == 0):
        raise ValueError("failed native call and rows differ")
    if (value.native_calls_returned == 0) != (value.native_rows_returned == 0):
        raise ValueError("returned native call and rows differ")
    if (
        value.native_calls_returned == value.native_calls_attempted
        and value.native_rows_returned != value.native_rows_attempted
    ):
        raise ValueError("returned call must account for all returned rows")
    if value.operation != "session" and (
        value.session_starts_attempted or value.session_starts_returned
    ):
        raise ValueError("non-session work cannot start a native session")
    if value.failure_work_may_be_unobserved is not True:
        raise ValueError("failed work must preserve uncertainty")


@dataclass(frozen=True)
class PublicCueRecurrentSnapshot:
    policy: PublicCueRecurrentPolicy
    heads: tuple[native.TorchRecurrentSnapshot, native.TorchRecurrentSnapshot]
    failed_work: FailureWork | None = None

    @property
    def completed_updates(self) -> int:
        return sum(h.completed_updates for h in self.heads)

    @property
    def training_examples(self) -> int:
        return sum(h.training_examples for h in self.heads)


def copy_snapshot(value: PublicCueRecurrentSnapshot) -> PublicCueRecurrentSnapshot:
    _record(value, PublicCueRecurrentSnapshot)
    policy = copy_policy(value.policy)
    if type(value.heads) is not tuple or len(value.heads) != 2:
        raise ValueError("public-cue recurrent snapshot requires exactly two native heads")
    heads = tuple(native._snapshot(h) for h in value.heads)
    if any(h.policy != policy.native_policy for h in heads):
        raise ValueError("recurrent head policy differs from declared policy")
    if max(value.completed_updates, value.training_examples) > 2**53 - 1:
        raise ValueError("aggregate recurrent counters exceed exact scalar bound")
    _failure(value.failed_work)
    return PublicCueRecurrentSnapshot(policy, (heads[0], heads[1]), value.failed_work)


def _raw(features: np.ndarray, *, single: bool = False) -> np.ndarray:
    rows = _feature_keys(features, 148)
    if not 1 <= len(rows) <= (1 if single else 16):
        raise ValueError("one recurrent call requires one bounded complete episode or live row")
    for row in rows:
        canonicalize_public_history(row + row)
    # Why: canonicalization above validates public pixels only. Native recurrence
    # receives the ORIGINAL raw148 values, never the canonical spatial design.
    return features.copy()


def _cue(frame: np.ndarray) -> int:
    first = tuple(float(x) for x in frame)
    return int(context_public_history(first + first)[-1])


def _consistent_cues(rows: np.ndarray, selected: int) -> None:
    for row in rows:
        if any(_decode_coordinate(row[j], 10) == 6 for j in range(0, 147, 3)):
            if _cue(row) != selected:
                raise ValueError("one recurrent episode cannot switch initial public cue")


def _work(operation: str, selected: int) -> dict[str, Any]:
    return vars(FailureWork(operation, selected)).copy()


def _stage(head: native.TorchRecurrentSnapshot, work: dict[str, Any]) -> Any:
    work["native_stages_attempted"] += 1
    learner = native.TorchRecurrentLearner(head.policy)
    learner.restore_state(head)
    work["native_stages_returned"] += 1
    return learner


def _failed(before: PublicCueRecurrentSnapshot, work: dict[str, Any], error: Exception) -> NoReturn:
    evidence = FailureWork(**work)
    raise TrainingUpdateFailed(
        f"{type(error).__name__}: {error}"[:512],
        FAILURE_DEFINITION,
        _failure_report(before, evidence),
    ) from error


def _failure_report(before: PublicCueRecurrentSnapshot, evidence: FailureWork) -> dict[str, Any]:
    # The shared failure port accepts counters/bools, not text.
    result = vars(evidence).copy()
    result.pop("operation")
    result.update(
        operation_index=("train", "predict", "session").index(evidence.operation),
        committed_updates=before.completed_updates,
        committed_rows=before.training_examples,
    )
    return result


class PublicCueRecurrentLearner:
    """Bounded two-head continuation; stage one native owner per operation."""

    def __init__(self, policy: PublicCueRecurrentPolicy = PublicCueRecurrentPolicy()) -> None:
        self._policy = copy_policy(policy)
        heads = tuple(
            native.TorchRecurrentLearner(self._policy.native_policy).snapshot_state()
            for _ in range(2)
        )
        self._state = copy_snapshot(PublicCueRecurrentSnapshot(self._policy, (heads[0], heads[1])))

    @property
    def policy(self) -> PublicCueRecurrentPolicy:
        return copy_policy(self._policy)

    @property
    def completed_updates(self) -> int:
        return self._state.completed_updates

    @property
    def training_examples(self) -> int:
        return self._state.training_examples

    def snapshot_state(self) -> PublicCueRecurrentSnapshot:
        saved = copy_snapshot(self._state)
        if saved.policy != self.policy:
            raise ValueError("original recurrent policy changed")
        return saved

    def restore_state(self, state: PublicCueRecurrentSnapshot) -> None:
        if self.snapshot_state().failed_work is not None:
            raise ValueError("failed recurrent owner cannot erase permanent work through restore")
        saved = copy_snapshot(state)
        if saved.policy != self.policy:
            raise ValueError("recurrent restore requires original public-cue policy")
        self._state = saved

    def fork(self) -> "PublicCueRecurrentLearner":
        child = object.__new__(PublicCueRecurrentLearner)
        child._policy, child._state = self.policy, self.snapshot_state()
        return child

    def _ready(self) -> tuple[Any, PublicCueRecurrentSnapshot]:
        before = self.snapshot_state()
        if before.failed_work is not None:
            raise TrainingUpdateFailed(
                "public-cue recurrent owner already failed",
                FAILURE_DEFINITION,
                _failure_report(before, before.failed_work),
            )
        return self._state, before

    def _guard(
        self,
        owner: Any,
        before: PublicCueRecurrentSnapshot,
        features: np.ndarray,
        original: np.ndarray,
        targets: np.ndarray | None = None,
        labels: np.ndarray | None = None,
    ) -> None:
        if (
            self._state is not owner
            or not _same_value(self.snapshot_state(), before)
            or not _same_value(features, original)
            or not _same_value(targets, labels)
        ):
            raise ValueError("original recurrent owner or input changed during native work")

    def _run(
        self,
        features: np.ndarray,
        targets: np.ndarray | None,
        *,
        weighted_episode: DemonstrationEpisode | None = None,
    ) -> Any:
        owner, before = self._ready()
        original, raw = _raw(features), features.copy()
        selected = _cue(raw[0])
        _consistent_cues(raw, selected)
        expected = raw.copy()
        if targets is not None:
            validate_class_targets(targets, len(raw), 7)
        labels = None if targets is None else targets.copy()
        frozen_labels = None if labels is None else labels.copy()
        weights = None if weighted_episode is None else episode_class_weights(weighted_episode)
        packet = None if weighted_episode is None else copy_episode(weighted_episode)
        frozen_packet = None if packet is None else copy_episode(packet)
        if packet is not None and (
            targets is None
            or packet.num_classes != 7
            or not _same_value(episode_arrays(packet), (raw, labels))
        ):
            raise ValueError("weighted training requires the same complete K7 TRAIN episode")

        def check() -> None:
            self._guard(owner, before, features, original, targets, frozen_labels)
            if not _same_value(packet, frozen_packet) or not _same_value(
                weighted_episode, frozen_packet
            ):
                raise ValueError("original or owned weighted TRAIN episode changed")

        work = _work("predict" if targets is None else "train", selected)
        try:
            learner = _stage(before.heads[selected], work)
            check()
            work["native_calls_attempted"], work["native_rows_attempted"] = 1, len(raw)
            if packet is not None:
                result = learner.train_weighted_episode(packet, weights)
            else:
                result = (
                    learner.predict(raw) if labels is None else learner.train_batch(raw, labels)
                )
            work["native_calls_returned"], work["native_rows_returned"] = 1, len(raw)
            check()
            after = native._snapshot(learner.snapshot_state())
            if (
                not _same_value(raw, expected)
                or not _same_value(labels, frozen_labels)
                or not _same_value(targets, frozen_labels)
            ):
                raise ValueError("native recurrent input or labels changed")
            if targets is None:
                if not _same_value(after, before.heads[selected]):
                    raise ValueError("prediction changed native recurrent learned state")
                validate_class_probabilities(result, 7)
                if result.shape != (len(raw), 7):
                    raise ValueError("recurrent probabilities require Tx7")
                check()
                if not _same_value(raw, expected):
                    raise ValueError(
                        "native recurrent input changed during final prediction readback"
                    )
                return result.copy()
            if (
                after.policy != before.policy.native_policy
                or after.completed_updates != before.heads[selected].completed_updates + 1
                or after.training_examples != before.heads[selected].training_examples + len(raw)
                or type(result) is not TrainingDiagnostic
                or result.definition
                != (
                    native.TORCH_RECURRENT_CE_ID
                    if packet is None
                    else TORCH_RECURRENT_WEIGHTED_CE_ID
                )
            ):
                raise ValueError("native recurrent diagnostic or update/row clocks differ")
            heads = list(before.heads)
            heads[selected] = after
            candidate = copy_snapshot(
                PublicCueRecurrentSnapshot(before.policy, (heads[0], heads[1]))
            )
            check()
            if not _same_value(raw, expected) or not _same_value(labels, frozen_labels):
                raise ValueError("native recurrent inputs changed during final state capture")
            self._state = candidate
            return result
        except Exception as error:
            # Keep detached known paid prefix and block this owner permanently;
            # this is not a claim that a corrupted live native owner was repaired.
            self._state = replace(before, failed_work=FailureWork(**work))
            _failed(before, work, error)

    def train_batch(self, features: np.ndarray, targets: np.ndarray) -> TrainingDiagnostic:
        if targets is None:
            raise ValueError("training requires class targets")
        return self._run(features, targets)

    def train_episode(self, episode: DemonstrationEpisode) -> TrainingDiagnostic:
        saved = copy_episode(episode)
        if saved.role != "train" or saved.num_classes != 7:
            raise ValueError("public-cue recurrent training requires complete K7 TRAIN episode")
        return self.train_batch(*episode_arrays(saved))

    def train_class_balanced_episode(self, episode: DemonstrationEpisode) -> TrainingDiagnostic:
        """Explicit own-episode TRAIN weights; ordinary training stays mean CE.

        Why this: reuse the native weighted objective without changing defaults
        or deriving weights from another task, evaluation or reserved probe role.
        """
        return self._run(*episode_arrays(episode), weighted_episode=episode)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._run(features, None)

    def start_episode(self) -> "PublicCueRecurrentEpisode":
        self._ready()
        return PublicCueRecurrentEpisode(self.snapshot_state())


class PublicCueRecurrentEpisode:
    """One detached checkpoint; choose only after seeing the first actual frame."""

    def __init__(self, state: PublicCueRecurrentSnapshot) -> None:
        self._state = copy_snapshot(state)
        if self._state.failed_work is not None:
            raise ValueError("failed recurrent checkpoint cannot start a session")
        self._frozen = copy_snapshot(self._state)
        self._selected: int | None = None
        self._session: Any = None
        self._binding: tuple[int, Any] | None = None
        self._steps = 0

    @property
    def completed_steps(self) -> int:
        return self._steps

    def snapshot_state(self) -> PublicCueRecurrentSnapshot:
        saved = copy_snapshot(self._state)
        if saved.failed_work is not None:
            return saved
        if not _same_value(saved, self._frozen):
            raise ValueError("session checkpoint changed")
        if self._binding is None:
            if self._session is not None or self._selected is not None or self._steps != 0:
                raise ValueError("session was selected before its first public observation")
        elif self._session is not self._binding[1] or self._selected != self._binding[0]:
            raise ValueError("original selected native session was replaced")
        if self._session is not None:
            if self._selected is None:
                raise ValueError("native session lacks its public selection")
            if (
                not _same_value(self._session.snapshot_state(), saved.heads[self._selected])
                or self._session.completed_steps != self._steps
            ):
                raise ValueError("native session state or activation counter changed")
        return saved

    def predict(self, features: np.ndarray) -> np.ndarray:
        before = self.snapshot_state()
        if before.failed_work is not None:
            _failed(
                before,
                dict(vars(before.failed_work)),
                ValueError("recurrent session already failed"),
            )
        if type(self._steps) is not int or not 0 <= self._steps < 16:
            raise ValueError("bounded recurrent session is exhausted or counter invalid")
        original, raw = _raw(features, single=True), features.copy()
        selected = _cue(raw[0]) if self._selected is None else self._selected
        _consistent_cues(raw, selected)
        work = _work("session", selected)
        try:
            if self._session is None:
                learner = _stage(before.heads[selected], work)
                work["session_starts_attempted"] = 1
                session = learner.start_episode()
                work["session_starts_returned"] = 1
                if (
                    session is learner
                    or session.completed_steps != 0
                    or not _same_value(session.snapshot_state(), before.heads[selected])
                ):
                    raise ValueError("native recurrent session did not start independently at zero")
                self._session, self._selected = session, selected
                self._binding = (selected, session)
            self.snapshot_state()
            work["native_calls_attempted"], work["native_rows_attempted"] = 1, 1
            result = self._session.predict(raw)
            work["native_calls_returned"], work["native_rows_returned"] = 1, 1
            if not _same_value(features, original) or not _same_value(raw, original):
                raise ValueError("live recurrent input changed")
            validate_class_probabilities(result, 7)
            if result.shape != (1, 7) or self._session.completed_steps != self._steps + 1:
                raise ValueError("live recurrent return/counter differs")
            if not _same_value(self._session.snapshot_state(), before.heads[selected]):
                raise ValueError("live prediction changed native learned state")
            self._steps += 1
            self.snapshot_state()
            if not _same_value(features, original) or not _same_value(raw, original):
                raise ValueError("live input changed during final state readback")
            return result.copy()
        except Exception as error:
            self._state = replace(before, failed_work=FailureWork(**work))
            _failed(before, work, error)
