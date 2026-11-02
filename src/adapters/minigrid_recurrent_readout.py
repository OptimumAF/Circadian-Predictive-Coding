"""Owned TRAIN readouts on frozen recurrent features; no SGD or new math.

Array ports rely on their caller's TRAIN release; complete episode ports enforce
the role. Predictions use actual raw frames only. A live session recomputes its
bounded prefix through the existing zero-reset activation reader, paying every
forward rather than introducing another recurrent-state implementation.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Callable

import numpy as np

from src.adapters import minigrid_public_cue_recurrent_learner as public
from src.adapters.torch_activation_diagnostics import read_native_activations
from src.app.coupled_attribution import _same_value as _same_state
from src.core.categorical_linear_fit import (
    CategoricalLinearFit,
    fit_categorical_linear,
    predict_categorical_linear,
)
from src.core.categorical_probability_scores import (
    validate_class_targets,
    validate_class_probabilities,
)
from src.core.demonstration_episode import DemonstrationEpisode, copy_episode, episode_arrays
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed

METHOD = "recurrent_readout_fit"
DEFINITION = "frozen_recurrent_readout_squared_error_sum_v1"
FAILURE = "frozen_recurrent_readout_failed_work_v1"


@dataclass(frozen=True)
class RecurrentReadoutPolicy:
    source_policy: public.PublicCueRecurrentPolicy = public.PublicCueRecurrentPolicy()

    def __post_init__(self) -> None:
        source = public.copy_policy(self.source_policy)
        if source != public.PublicCueRecurrentPolicy():
            raise ValueError("readout control requires original frozen raw148/H8/K7/seed17 policy")
        object.__setattr__(self, "source_policy", source)

    @property
    def method(self) -> str:
        return METHOD

    @property
    def feature_dim(self) -> int:
        return 148

    @property
    def num_classes(self) -> int:
        return 7


@dataclass(frozen=True)
class ReadoutFailure:
    feature_calls_attempted: int = 0
    feature_calls_returned: int = 0
    feature_rows_attempted: int = 0
    feature_rows_returned: int = 0
    fit_calls_attempted: int = 0
    fit_calls_returned: int = 0
    fit_rows_attempted: int = 0
    fit_rows_returned: int = 0


@dataclass(frozen=True)
class RecurrentReadoutSnapshot:
    policy: RecurrentReadoutPolicy
    native: public.PublicCueRecurrentSnapshot
    readouts: tuple[CategoricalLinearFit | None, CategoricalLinearFit | None] = (None, None)
    fit_updates: tuple[int, int] = (0, 0)
    fit_rows: tuple[int, int] = (0, 0)
    failure: ReadoutFailure | None = None

    @property
    def completed_updates(self) -> int:
        return sum(self.fit_updates)

    @property
    def training_examples(self) -> int:
        return sum(self.fit_rows)


def copy_snapshot(state: RecurrentReadoutSnapshot) -> RecurrentReadoutSnapshot:
    public._record(state, RecurrentReadoutSnapshot)
    public._record(state.policy, RecurrentReadoutPolicy)
    policy = RecurrentReadoutPolicy(state.policy.source_policy)
    native = public.copy_snapshot(state.native)
    if (
        native.policy != policy.source_policy
        or native.failed_work is not None
        or native.completed_updates
        or native.training_examples
    ):
        raise ValueError("readout source must remain the original unfitted recurrent state")
    for values in (state.readouts, state.fit_updates, state.fit_rows):
        if type(values) is not tuple or len(values) != 2:
            raise ValueError("readout state requires two exact cue fields")
    for model, updates, rows in zip(state.readouts, state.fit_updates, state.fit_rows, strict=True):
        if (
            type(updates) is not int
            or type(rows) is not int
            or not 0 <= updates <= rows <= 16 * updates <= 2**53 - 1
        ):
            raise ValueError("readout fit clocks differ from complete bounded episodes")
        if not updates:
            if model is not None or rows:
                raise ValueError("unfitted readout must be empty")
        else:
            if type(model) is not CategoricalLinearFit:
                raise ValueError("fitted readout requires the exact immutable core fit")
            model.__post_init__()
            if (
                model.feature_dim != 8
                or model.num_classes != 7
                or not 1 <= model.fit_rows <= min(rows, 16)
            ):
                raise ValueError("readout geometry differs from frozen H8/K7 features")
    if max(state.completed_updates, state.training_examples) > 2**53 - 1:
        raise ValueError("aggregate readout counters overflow")
    if state.failure is not None:
        public._record(state.failure, ReadoutFailure)
        w = asdict(state.failure)
        for name in ("feature", "fit"):
            if not (
                type(w[name + "_calls_attempted"]) is int
                and type(w[name + "_calls_returned"]) is int
                and 0 <= w[name + "_calls_returned"] <= w[name + "_calls_attempted"] <= 1
            ):
                raise ValueError("invalid permanent failed readout calls")
            for suffix in ("attempted", "returned"):
                rows = w[name + "_rows_" + suffix]
                if (
                    type(rows) is not int
                    or not 0 <= rows <= 16
                    or bool(rows) != bool(w[name + "_calls_" + suffix])
                ):
                    raise ValueError("invalid permanent failed readout rows")
        if state.failure.fit_calls_attempted > state.failure.feature_calls_returned:
            raise ValueError("fit cannot precede returned features")
    return RecurrentReadoutSnapshot(
        policy, native, deepcopy(state.readouts), state.fit_updates, state.fit_rows, state.failure
    )


def _owners(state: RecurrentReadoutSnapshot) -> tuple[int, ...]:
    # Fixed class-specific ownership, including cold native tensors/containers.
    refs = [
        state,
        state.policy,
        state.policy.source_policy,
        state.policy.source_policy.native_policy,
        state.native,
        state.native.policy,
        state.native.policy.native_policy,
        state.native.heads,
        state.readouts,
        state.fit_updates,
        state.fit_rows,
        state.failure,
    ]
    for head in state.native.heads:
        refs.extend(
            (
                head,
                head.policy,
                head.parameters,
                head.gradients,
                head.optimizer_state,
                head.optimizer_state["state"],
                head.optimizer_state["param_groups"],
            )
        )
        refs.extend(head.parameters.values())
        for group in head.optimizer_state["param_groups"]:
            refs.extend((group, group["params"]))
    for model in state.readouts:
        if model is not None:
            refs.extend((model, model.coefficients, *model.coefficients))
    return tuple(id(value) for value in refs)


def _failed(state: RecurrentReadoutSnapshot, work: ReadoutFailure) -> TrainingUpdateFailed:
    return TrainingUpdateFailed(
        "readout owner has permanently failed paid work",
        FAILURE,
        dict(
            asdict(work),
            committed_updates=state.completed_updates,
            committed_rows=state.training_examples,
            failure_work_may_be_unobserved=True,
        ),
    )


class RecurrentReadoutLearner:
    """One frozen feature source and two independently owned fitted readouts."""

    def __init__(self, policy: RecurrentReadoutPolicy = RecurrentReadoutPolicy()) -> None:
        public._record(policy, RecurrentReadoutPolicy)
        policy = RecurrentReadoutPolicy(policy.source_policy)
        native = public.PublicCueRecurrentLearner(policy.source_policy).snapshot_state()
        self._terminal: tuple[ReadoutFailure, int, int] | None = None
        self._seal(RecurrentReadoutSnapshot(policy, native))

    def _seal(
        self, state: RecurrentReadoutSnapshot, check: Callable[[], None] | None = None
    ) -> None:
        owned = copy_snapshot(state)
        expected = deepcopy(owned)
        if check is not None:
            check()
        self._state, self._expected, self._pins = owned, expected, _owners(owned)

    def _check(self) -> None:
        if _owners(self._state) != self._pins or not _same_state(self._state, self._expected):
            raise ValueError("original readout state or owner was changed or replaced")

    def _ready(self) -> RecurrentReadoutSnapshot:
        if self._terminal is not None:
            work, updates, rows = self._terminal
            raise TrainingUpdateFailed(
                "readout owner permanently failed",
                FAILURE,
                dict(
                    asdict(work),
                    committed_updates=updates,
                    committed_rows=rows,
                    failure_work_may_be_unobserved=True,
                ),
            )
        state = self.snapshot_state()
        if state.failure is not None:
            raise _failed(state, state.failure)
        return state

    @property
    def policy(self) -> RecurrentReadoutPolicy:
        return RecurrentReadoutPolicy(self._state.policy.source_policy)

    @property
    def completed_updates(self) -> int:
        return self._state.completed_updates

    @property
    def training_examples(self) -> int:
        return self._state.training_examples

    def snapshot_state(self) -> RecurrentReadoutSnapshot:
        self._check()
        if self._terminal is not None:
            work, updates, rows = self._terminal
            if (
                self._state.failure != work
                or self._state.completed_updates != updates
                or self._state.training_examples != rows
            ):
                raise ValueError(
                    "failed readout capture is unavailable; paid work remains terminal"
                )
        saved = copy_snapshot(self._state)
        self._check()
        if not _same_state(saved, self._expected):
            raise ValueError("snapshot copy changed the readout state")
        return saved

    def restore_state(self, state: RecurrentReadoutSnapshot) -> None:
        before = self._ready()
        saved = copy_snapshot(state)
        if saved.policy != before.policy or not _same_state(saved.native, before.native):
            raise ValueError("restore cannot change the frozen recurrent representation")
        self._seal(saved, self._check)

    def fork(self) -> "RecurrentReadoutLearner":
        state = self._ready()
        child = object.__new__(RecurrentReadoutLearner)
        child._terminal = None
        child._seal(state)
        self._check()
        return child

    def train_batch(self, features: np.ndarray, targets: np.ndarray) -> TrainingDiagnostic:
        before = self._ready()
        failure_baseline = deepcopy(before)
        raw = public._raw(features)
        selected = public._cue(raw[0])
        public._consistent_cues(raw, selected)
        validate_class_targets(targets, len(raw), 7)
        labels = targets.copy()
        raw_pin, labels_pin = raw.copy(), labels.copy()
        owner = self._state
        work = ReadoutFailure()

        def guard():
            self._check()
            if (
                self._state is not owner
                or not _same_state(before, self._expected)
                or not _same_state(
                    (features, raw, targets, labels), (raw_pin, raw_pin, labels_pin, labels_pin)
                )
            ):
                raise ValueError("readout TRAIN input or original owner changed")

        try:
            work = replace(work, feature_calls_attempted=1, feature_rows_attempted=len(raw))
            hidden = read_native_activations(before.native.heads[selected], raw)
            work = replace(work, feature_calls_returned=1, feature_rows_returned=len(raw))
            guard()
            hidden_pin = hidden.copy()
            work = replace(work, fit_calls_attempted=1, fit_rows_attempted=len(raw))
            fit = fit_categorical_linear(hidden, labels, 7)
            work = replace(work, fit_calls_returned=1, fit_rows_returned=len(raw))
            guard()
            if not _same_state(hidden, hidden_pin):
                raise ValueError("readout fit changed its frozen hidden input")
            if type(fit) is not CategoricalLinearFit or fit.fit_rows != len(raw):
                raise ValueError("readout fit returned inconsistent geometry")
            models, updates, rows = (
                list(before.readouts),
                list(before.fit_updates),
                list(before.fit_rows),
            )
            models[selected], updates[selected], rows[selected] = (
                fit,
                updates[selected] + 1,
                rows[selected] + len(raw),
            )
            candidate = copy_snapshot(
                replace(
                    before,
                    readouts=(models[0], models[1]),
                    fit_updates=(updates[0], updates[1]),
                    fit_rows=(rows[0], rows[1]),
                )
            )
            guard()
            self._seal(candidate, guard)
            return TrainingDiagnostic(DEFINITION, fit.squared_error)
        except Exception as error:
            # Known paid prefix is retained as failed, never a repair attestation.
            self._terminal = (
                work,
                failure_baseline.completed_updates,
                failure_baseline.training_examples,
            )
            try:
                self._seal(replace(failure_baseline, failure=work))
            except Exception:
                # An unavailable capture cannot erase the permanent work latch.
                pass
            raise _failed(failure_baseline, work) from error

    def train_episode(self, episode: DemonstrationEpisode) -> TrainingDiagnostic:
        saved = copy_episode(episode)
        if saved.role != "train" or saved.num_classes != 7:
            raise ValueError("readout fitting requires a complete K7 TRAIN episode")
        return self.train_batch(*episode_arrays(saved))

    def predict(self, features: np.ndarray) -> np.ndarray:
        state = self._ready()
        raw = public._raw(features)
        selected = public._cue(raw[0])
        public._consistent_cues(raw, selected)
        model = state.readouts[selected]
        if model is None:
            raise ValueError("selected public-cue readout is not fitted")
        raw_pin = raw.copy()
        hidden = read_native_activations(state.native.heads[selected], raw)
        self._check()
        if not _same_state(state, self._expected) or not _same_state(
            (features, raw), (raw_pin, raw_pin)
        ):
            raise ValueError("readout prediction input changed")
        hidden_pin = hidden.copy()
        result = predict_categorical_linear(model, hidden)
        validate_class_probabilities(result, 7)
        if result.shape != (len(raw), 7):
            raise ValueError("readout probabilities must preserve all seven action columns")
        self._check()
        if not _same_state(state, self._expected) or not _same_state(
            (features, raw, hidden), (raw_pin, raw_pin, hidden_pin)
        ):
            raise ValueError("readout prediction input or frozen state changed")
        return result.copy()

    def start_episode(self) -> "RecurrentReadoutEpisode":
        state = self._ready()
        if any(model is None for model in state.readouts):
            raise ValueError("live readout requires both public cues fitted")
        return RecurrentReadoutEpisode(state)


class RecurrentReadoutEpisode:
    """At most16actual frames, no labels or learned updates; failure is terminal."""

    def __init__(self, state: RecurrentReadoutSnapshot) -> None:
        self._state = copy_snapshot(state)
        self._expected = deepcopy(self._state)
        self._pins = _owners(self._state)
        self._frames: tuple[tuple[float, ...], ...] = ()
        self._frames_pin: tuple[tuple[float, ...], ...] = self._frames
        self._selected: int | None = None
        self._selected_pin: int | None = None
        self._failed = False
        self._failed_pin = False
        self.feature_rows_attempted = self.feature_rows_returned = 0
        self._costs = (0, 0)

    def _check(self) -> None:
        if (
            self._failed
            or self._failed is not self._failed_pin
            or _owners(self._state) != self._pins
            or not _same_state(self._state, self._expected)
            or self._frames is not self._frames_pin
            or self._selected != self._selected_pin
            or (self.feature_rows_attempted, self.feature_rows_returned) != self._costs
        ):
            raise ValueError("live readout state/prefix changed or session already failed")

    @property
    def completed_steps(self) -> int:
        return len(self._frames)

    def snapshot_state(self) -> RecurrentReadoutSnapshot:
        self._check()
        saved = copy_snapshot(self._state)
        self._check()
        if not _same_state(saved, self._expected):
            raise ValueError("live snapshot copy changed frozen readout state")
        return saved

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._check()
        raw = public._raw(features, single=True)
        if self.completed_steps >= 16:
            raise ValueError("readout session exceeds16actual frames")
        selected = public._cue(raw[0]) if self._selected is None else self._selected
        frames = self._frames + (tuple(float(x) for x in raw[0]),)
        prefix = np.array(frames, dtype=np.float64)
        prefix_pin = prefix.copy()
        public._consistent_cues(prefix, selected)
        try:
            self.feature_rows_attempted += len(prefix)
            self._costs = (self.feature_rows_attempted, self.feature_rows_returned)
            admitted_costs = self._costs
            model = self._state.readouts[selected]
            if model is None:
                raise ValueError("live readout cue is not fitted")
            hidden = read_native_activations(self._state.native.heads[selected], prefix)
            counters_unchanged = (
                self.feature_rows_attempted,
                self.feature_rows_returned,
            ) == admitted_costs
            self._costs = (admitted_costs[0], admitted_costs[1] + len(prefix))
            if counters_unchanged:
                self.feature_rows_returned = self._costs[1]
            self._check()
            if not _same_state(prefix, prefix_pin):
                raise ValueError("live readout prefix input changed")
            hidden_pin = hidden.copy()
            result = predict_categorical_linear(model, hidden[-1:])
            validate_class_probabilities(result, 7)
            if result.shape != (1, 7):
                raise ValueError("live readout requires one seven-action probability row")
            self._check()
            if not _same_state((features, prefix, hidden), (raw, prefix_pin, hidden_pin)):
                raise ValueError("live readout input changed")
            self._frames = self._frames_pin = frames
            self._selected = self._selected_pin = selected
            return result.copy()
        except Exception:
            self._failed = self._failed_pin = True
            raise
