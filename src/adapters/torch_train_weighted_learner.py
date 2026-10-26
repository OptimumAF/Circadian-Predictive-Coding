"""Explicit weighted learning sources through the existing sequential array port.

Inputs are a recurrent or bound BP/PC/neutral policy and an owned TRAIN-episode
schedule, one bound feature catalogue, or a warm reserved probe schedule.
Outputs retain native diagnostics/
snapshots; recurrent owners also expose sessions. The native counter selects
the next admitted wake; this adapter
adds no objective math, release authority, budget, environment, checkpoint format
or persistent activation.
"""

from copy import deepcopy
from dataclasses import fields, replace
from collections.abc import Callable
from typing import Any, NoReturn, cast
import pickle

import numpy as np

from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters import torch_feature_learner as feature_native
from src.adapters import minigrid_public_cue_recurrent_learner as public_recurrent
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
    InvariantTorchFeatureSnapshot,
    copy_invariant_torch_feature_policy,
    copy_invariant_torch_feature_snapshot,
)
from src.adapters.torch_feature_weighted import (
    TORCH_INVARIANT_WEIGHTED_CE_ID,
    _autograd_marks,
    _rng,
    _record,
    train_weighted_episode as train_weighted_feature_episode,
)
from src.adapters.torch_recurrent_learner import (
    RecurrentEpisodePredictor,
    TorchRecurrentLearner,
    TorchRecurrentPolicy,
    TorchRecurrentSnapshot,
    _policy,
)
from src.adapters.torch_recurrent_weighted import TORCH_RECURRENT_WEIGHTED_CE_ID
from src.adapters.torch_pc_weighted import (
    TORCH_INVARIANT_WEIGHTED_PC_ID,
    train_weighted_episode as train_weighted_pc_episode,
)
from src.adapters.torch_catalogue_weighted import (
    TrainCatalogue,
    TORCH_CATALOGUE_WEIGHTED_CE_ID,
    TORCH_CATALOGUE_WEIGHTED_PC_ID,
    _catalogue as _copy_catalogue,
    catalogue_class_weights,
    train_weighted_catalogue,
)
from src.adapters.minigrid_train_catalogue import compose_catalogue_training
from src.core.coupled_energy import Array
from src.core.demonstration_class_weights import episode_class_weights, probe_episode_class_weights
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    copy_episode,
    episode_arrays,
)
from src.core.learner_ports import TrainingDiagnostic
from src.core.learner_ports import TrainingUpdateFailed
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.categorical_training_diagnostic import (
    CategoricalPCTrainingDiagnostic,
    ScheduledCategoricalUpdateFailed,
    STREAM_WORK_FIELDS,
    copy_categorical_native_work,
)
from src.adapters.torch_pc_inference import (
    BoundedPCTrainingUpdate,
    TORCH_BOUNDED_WEIGHTED_PC_ID,
    train_bounded_weighted_episode,
    train_bounded_probe_episode,
    _policy as _copy_inference_policy,
    _same_policy as _same_inference_policy,
)
from src.core.demonstration_schedule_witness import DemonstrationScheduleWitness
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from src.shared.torch_runtime import require_torch


Policy = (
    TorchRecurrentPolicy | InvariantTorchFeaturePolicy | public_recurrent.PublicCueRecurrentPolicy
)
Snapshot = (
    TorchRecurrentSnapshot
    | InvariantTorchFeatureSnapshot
    | public_recurrent.PublicCueRecurrentSnapshot
)


def _charge_stream(work: dict[str, int] | None, name: str) -> None:
    if work is not None:
        work[name] += 1


class _BoundedEpisodeFailure(RuntimeError):
    """Mark a failure received directly at the trusted bounded episode port."""

    def __init__(self, failure: TrainingUpdateFailed):
        self.failure = failure
        super().__init__(str(failure))


def _inference_signature(policy: CoupledInferencePolicy | None):
    if policy is None:
        return None
    return tuple(
        (name, type(value).__name__, value.hex() if type(value) is float else value)
        for name, value in vars(_copy_inference_policy(policy)).items()
    )


def _copy_policy(policy: Policy) -> Policy:
    if type(policy) is public_recurrent.PublicCueRecurrentPolicy:
        return public_recurrent.copy_policy(policy)
    if type(policy) is InvariantTorchFeaturePolicy:
        owned = copy_invariant_torch_feature_policy(policy)
        if owned.native_policy.method not in {"bp", "pc", "neutral_cpc"}:
            raise ValueError("weighted feature schedules require bound BP, PC or neutral_cpc")
        return owned
    return _policy(cast(TorchRecurrentPolicy, policy))


def _objective(policy: Policy, catalogue: bool = False, bounded: bool = False) -> str:
    if bounded:
        return TORCH_BOUNDED_WEIGHTED_PC_ID
    if type(policy) is InvariantTorchFeaturePolicy:
        if catalogue:
            return (
                TORCH_CATALOGUE_WEIGHTED_CE_ID
                if policy.native_policy.method == "bp"
                else TORCH_CATALOGUE_WEIGHTED_PC_ID
            )
        if policy.native_policy.method == "bp":
            return TORCH_INVARIANT_WEIGHTED_CE_ID
        if policy.native_policy.method in {"pc", "neutral_cpc"}:
            return TORCH_INVARIANT_WEIGHTED_PC_ID
        raise ValueError("weighted feature objective requires BP, PC or neutral_cpc")
    return TORCH_RECURRENT_WEIGHTED_CE_ID


def _match_arrays(features: Array, targets: Array, expected: tuple[Array, Array]) -> None:
    if any(
        type(actual) is not np.ndarray
        or actual.dtype != np.dtype(np.float64)
        or actual.shape != wanted.shape
        or not np.array_equal(actual, wanted)
        for actual, wanted in zip((features, targets), expected, strict=True)
    ):
        raise ValueError("weighted arrays must exactly match the next owned train episode")


def _complete_fields(episodes: tuple[DemonstrationEpisode, ...]) -> bool:
    return all(
        type(episode) is DemonstrationEpisode
        and set(vars(episode)) == {field.name for field in fields(DemonstrationEpisode)}
        and type(episode.transitions) is tuple
        and all(
            type(transition) is DemonstrationTransition
            and set(vars(transition)) == {field.name for field in fields(DemonstrationTransition)}
            for transition in episode.transitions
        )
        for episode in episodes
    )


def _schedule(
    episodes: tuple[DemonstrationEpisode, ...], policy: Policy, *, role: str = "train"
) -> tuple[DemonstrationEpisode, ...]:
    if (
        type(episodes) is not tuple
        or not 1 <= len(episodes) <= 512
        or not _complete_fields(episodes)
    ):
        raise ValueError(
            "weighted schedule requires an exact tuple of1..512 complete train episodes"
        )
    owned = tuple(copy_episode(episode) for episode in episodes)
    max_steps = (
        16
        if isinstance(
            policy, (InvariantTorchFeaturePolicy, public_recurrent.PublicCueRecurrentPolicy)
        )
        else policy.max_episode_steps
    )
    if any(
        episode.role != role
        or episode.num_classes != policy.num_classes
        or len(episode.transitions) > max_steps
        or len(episode.transitions[0].observation) != policy.feature_dim
        for episode in owned
    ):
        raise ValueError("weighted schedule requires train episodes matching the native policy")
    if type(policy) is InvariantTorchFeaturePolicy:
        # Whole-schedule public admission precedes native construction. Next views
        # remain evidence, never additional prediction features or labels.
        for episode in owned:
            for transition in episode.transitions:
                canonicalize_public_history(transition.observation)
                canonicalize_public_history(transition.next_observation)
    if type(policy) is public_recurrent.PublicCueRecurrentPolicy:
        for episode in owned:
            current = public_recurrent._raw(episode_arrays(episode)[0])
            selected = public_recurrent._cue(current[0])
            public_recurrent._consistent_cues(current, selected)
            next_rows = public_recurrent._raw(
                np.array(
                    [transition.next_observation for transition in episode.transitions],
                    dtype=np.float64,
                )
            )
            public_recurrent._consistent_cues(next_rows, selected)
    return owned


class TorchTrainWeightedLearner:
    """Bind updates to owned TRAIN sources or an explicit disposable probe."""

    def __init__(
        self,
        policy: Policy,
        episodes: tuple[DemonstrationEpisode, ...],
        *,
        inference_policy: CoupledInferencePolicy | None = None,
    ) -> None:
        if type(self) is not TorchTrainWeightedLearner:
            raise ValueError("weighted training requires an exact TorchTrainWeightedLearner")
        self._policy = _copy_policy(policy)
        self._policy_seal = replace(self._policy)
        self._origin = self._origin_owner = ("train", 0, 0, 0)
        self._configure_inference(inference_policy)
        self._catalogue: TrainCatalogue | None = None
        self._episodes = _schedule(episodes, self._policy)
        self._schedule_owner = self._episodes
        self._schedule_seal = tuple(copy_episode(episode) for episode in self._episodes)
        self._schedule_payload = pickle.dumps(self._episodes, protocol=5)
        if type(self._policy_seal) in (
            InvariantTorchFeaturePolicy,
            public_recurrent.PublicCueRecurrentPolicy,
        ):
            self._bind_schedule()
        self._objective_id = _objective(self._policy, bounded=self._inference_policy is not None)
        self._initialize_native()

    def _configure_inference(self, policy: CoupledInferencePolicy | None) -> None:
        owned = None if policy is None else _copy_inference_policy(policy)
        if owned is not None and (
            type(self._policy_seal) is not InvariantTorchFeaturePolicy
            or self._policy_seal.native_policy.method not in {"pc", "neutral_cpc"}
            or owned.min_steps > self._policy_seal.native_policy.inference_steps
        ):
            raise ValueError("bounded schedules require bound PC/neutral episodes and min<=max")
        self._inference_policy = self._inference_policy_owner = owned
        self._inference_policy_seal = None if owned is None else _copy_inference_policy(owned)
        self._inference_policy_seal_owner = self._inference_policy_seal
        self._inference_birth = self._inference_birth_owner = _inference_signature(owned)

    def _check_inference(self) -> None:
        actual, seal = self._inference_policy, self._inference_policy_seal
        if (
            actual is not self._inference_policy_owner
            or seal is not self._inference_policy_seal_owner
            or (actual is None) != (seal is None)
            or self._inference_birth is not self._inference_birth_owner
            or _inference_signature(actual) != self._inference_birth
        ):
            raise ValueError("bounded scheduled policy owner or seal changed")
        if actual is not None:
            if seal is None or not _same_inference_policy(actual, _copy_inference_policy(seal)):
                raise ValueError("bounded scheduled policy changed")

    def _initialize_native(
        self, initial_state: InvariantTorchFeatureSnapshot | None = None
    ) -> None:
        self._public_failure: tuple[str, str, tuple[tuple[str, int | bool], ...]] | None = None
        self._learner: Any = (
            public_recurrent.PublicCueRecurrentLearner(self._policy)
            if type(self._policy) is public_recurrent.PublicCueRecurrentPolicy
            else InvariantTorchFeatureLearner(self._policy)
            if isinstance(self._policy, InvariantTorchFeaturePolicy)
            else TorchRecurrentLearner(cast(TorchRecurrentPolicy, self._policy))
        )
        if type(self._policy) is InvariantTorchFeaturePolicy:
            self._feature_native_owner = cast(InvariantTorchFeatureLearner, self._learner)._native
        self._native_owner = self._learner
        if initial_state is not None:
            cast(InvariantTorchFeatureLearner, self._learner).restore_state(initial_state)
        self._expected_state = deepcopy(self._learner.snapshot_state())
        self._check()

    @classmethod
    def from_catalogue(
        cls,
        policy: InvariantTorchFeaturePolicy,
        catalogue: TrainCatalogue,
        updates: int,
    ) -> "TorchTrainWeightedLearner":
        """Bind one complete TRAIN catalogue to1..128 sequential array updates.

        Why: repeated wakes share a fixed owned source catalogue, not fabricated
        episodes or128 separate metadata copies. The existing stream still owns
        label release, paid budgets and source/actor/checkpoint boundaries.
        """
        if cls is not TorchTrainWeightedLearner or type(policy) is not InvariantTorchFeaturePolicy:
            raise ValueError("weighted catalogue requires the exact bound feature schedule")
        child = object.__new__(cls)
        child._policy = _copy_policy(policy)
        child._policy_seal = replace(child._policy)
        child._origin = child._origin_owner = ("train", 0, 0, 0)
        child._configure_inference(None)
        child._episodes = ()
        child._schedule_owner = child._episodes
        child._schedule_seal = ()
        child._schedule_payload = b""
        child._bind_catalogue(catalogue, updates)
        child._objective_id = _objective(child._policy, True)
        child._initialize_native()
        return child

    @classmethod
    def from_probe_snapshot(
        cls,
        snapshot: InvariantTorchFeatureSnapshot,
        episodes: tuple[DemonstrationEpisode, ...],
        *,
        inference_policy: CoupledInferencePolicy,
        return_episodes: tuple[DemonstrationEpisode, ...] = (),
    ) -> "TorchTrainWeightedLearner":
        """Own a warm disposable probe schedule without resetting native clocks.

        Why: reserved updates and the optional sixteen TRAIN returns belong to
        the existing probe runner, while full native history stays continuous.
        Roles and complete source fields are retained; no evaluation is admitted.
        """
        if cls is not TorchTrainWeightedLearner:
            raise ValueError("probe schedule requires the exact weighted learner")
        _record(snapshot, InvariantTorchFeatureSnapshot)
        child = object.__new__(cls)
        child._policy = _copy_policy(snapshot.policy)
        child._policy_seal = replace(child._policy)
        if inference_policy is None:
            raise ValueError("probe schedule requires explicit bounded inference")
        child._configure_inference(inference_policy)
        if type(episodes) is not tuple or not 1 <= len(episodes) <= 16:
            raise ValueError("probe schedule requires1..16 complete reserved episodes")
        fresh = _schedule(episodes, child._policy, role="probe_train")
        if type(return_episodes) is not tuple or len(return_episodes) not in {0, 16}:
            raise ValueError("probe return schedule must be empty or exactly16 TRAIN episodes")
        returning = () if not return_episodes else _schedule(return_episodes, child._policy)
        # All roles and raw observations precede any native tensor allocation.
        owned = copy_invariant_torch_feature_snapshot(snapshot)
        child._origin = child._origin_owner = (
            "probe_train",
            owned.completed_updates,
            owned.training_examples,
            len(fresh),
        )
        child._catalogue = None
        child._episodes = fresh + returning
        child._schedule_owner = child._episodes
        child._schedule_seal = tuple(copy_episode(item) for item in child._episodes)
        child._schedule_payload = pickle.dumps(child._episodes, protocol=5)
        child._objective_id = _objective(child._policy, bounded=True)
        child._initialize_native(owned)
        if not _same_state(owned, copy_invariant_torch_feature_snapshot(snapshot)):
            raise ValueError("probe snapshot changed during construction")
        return child

    @property
    def source_role(self) -> str:
        self._check()
        return self._origin[0]

    def _check_origin(self) -> None:
        origin = self._origin
        if (
            type(origin) is not tuple
            or origin is not self._origin_owner
            or len(origin) != 4
            or type(origin[0]) is not str
            or any(type(value) is not int or value < 0 for value in origin[1:])
        ):
            raise ValueError("weighted schedule origin changed")
        mode, updates, rows, fresh = origin
        if mode == "train":
            if (updates, rows, fresh) != (0, 0, 0):
                raise ValueError("TRAIN origin must start at zero")
        elif (
            mode != "probe_train"
            or not 1 <= fresh <= 16
            or len(self._episodes) - fresh not in {0, 16}
            or self._catalogue is not None
            or self._inference_policy is None
            or type(self._policy_seal) is not InvariantTorchFeaturePolicy
            or self._policy_seal.native_policy.method not in {"pc", "neutral_cpc"}
        ):
            raise ValueError("reserved schedule origin or phase bounds changed")

    def _bind_catalogue(self, catalogue: TrainCatalogue, updates: int) -> None:
        if type(updates) is not int or not 1 <= updates <= 128:
            raise ValueError("weighted catalogue updates require an exact integer in1..128")
        self._catalogue = _copy_catalogue(catalogue)
        self._catalogue_owner = self._catalogue
        self._catalogue_updates = self._catalogue_limit = updates
        self._catalogue_witness = DemonstrationScheduleWitness(self._catalogue.episodes)
        self._catalogue_witness_owner = self._catalogue_witness
        self._catalogue_spans = self._catalogue.source_spans
        self._catalogue_span_owners = tuple(self._catalogue_spans)
        self._catalogue_span_payload = pickle.dumps(self._catalogue_spans, protocol=5)
        self._catalogue_span_payload_owner = self._catalogue_span_payload
        self._catalogue_rows = sum(len(item.transitions) for item in self._catalogue.episodes)
        self._catalogue_row_limit = self._catalogue_rows

    def _catalogue_metadata(self) -> None:
        value = self._catalogue
        if (
            type(value) is not TrainCatalogue
            or value is not self._catalogue_owner
            or set(vars(value)) != {"episodes", "source_spans"}
            or type(self._catalogue_updates) is not int
            or type(self._catalogue_limit) is not int
            or not 1 <= self._catalogue_updates <= 128
            or self._catalogue_updates != self._catalogue_limit
            or type(self._catalogue_rows) is not int
            or type(self._catalogue_row_limit) is not int
            or not 1 <= self._catalogue_rows <= 54
            or self._catalogue_rows != self._catalogue_row_limit
            or self._catalogue_witness is not self._catalogue_witness_owner
            or value.source_spans is not self._catalogue_spans
            or type(value.source_spans) is not tuple
            or len(value.source_spans) != len(self._catalogue_span_owners)
            or any(a is not b for a, b in zip(value.source_spans, self._catalogue_span_owners))
            or self._catalogue_span_payload is not self._catalogue_span_payload_owner
            or pickle.dumps(value.source_spans, protocol=5) != self._catalogue_span_payload
        ):
            raise ValueError("weighted catalogue source spans, ownership or limits changed")
        self._catalogue_witness.check(value.episodes)

    def _bind_schedule(self) -> None:
        self._schedule_witness = DemonstrationScheduleWitness(self._episodes)
        self._schedule_seal_witness = DemonstrationScheduleWitness(self._schedule_seal)
        self._schedule_witness.check_copy(self._schedule_seal_witness)
        self._schedule_witness_owner = self._schedule_witness
        self._schedule_seal_witness_owner = self._schedule_seal_witness
        self._schedule_payload_owner = self._schedule_payload
        if type(self._policy_seal) is public_recurrent.PublicCueRecurrentPolicy:
            # Why: these immutable counts share the schedule witness's lifetime.
            # Re-decoding every preceding public frame on each read adds work,
            # while the witness still refuses every changed mutable source field.
            prefixes = [(0, 0, 0, 0)]
            for episode in self._episodes:
                selected = public_recurrent._cue(np.asarray(episode.transitions[0].observation))
                counts = list(prefixes[-1])
                counts[2 * selected] += 1
                counts[2 * selected + 1] += len(episode.transitions)
                prefixes.append((counts[0], counts[1], counts[2], counts[3]))
            self._public_prefixes = self._public_prefixes_owner = tuple(prefixes)

    def _check_public_prefixes(self) -> None:
        if (
            type(self._public_prefixes) is not tuple
            or self._public_prefixes is not self._public_prefixes_owner
            or len(self._public_prefixes) != len(self._episodes) + 1
        ):
            raise ValueError("weighted public-cue prefix evidence owner changed")

    def _schedule_metadata(self) -> None:
        if self._catalogue is not None:
            self._catalogue_metadata()
            return
        if (
            type(self._policy_seal)
            in (InvariantTorchFeaturePolicy, public_recurrent.PublicCueRecurrentPolicy)
            and self._origin[0] == "train"
        ):
            # Why: exact immutable leaves need no repeated serialization. Every
            # mutable field is still checked against this owner's lifetime proof.
            if (
                type(self._schedule_witness) is not DemonstrationScheduleWitness
                or self._schedule_witness is not self._schedule_witness_owner
                or type(self._schedule_seal_witness) is not DemonstrationScheduleWitness
                or self._schedule_seal_witness is not self._schedule_seal_witness_owner
                or type(self._schedule_payload) is not bytes
                or self._schedule_payload is not self._schedule_payload_owner
            ):
                raise ValueError("weighted schedule evidence owner changed")
            self._schedule_witness.check(self._episodes)
            self._schedule_seal_witness.check(self._schedule_seal)
            if type(self._policy_seal) is public_recurrent.PublicCueRecurrentPolicy:
                self._check_public_prefixes()
        elif (
            type(self._episodes) is not tuple
            or not _complete_fields(self._episodes)
            or type(self._schedule_seal) is not tuple
            or not _complete_fields(self._schedule_seal)
            or type(self._schedule_payload) is not bytes
            or pickle.dumps(self._episodes, protocol=5) != self._schedule_payload
            or pickle.dumps(self._schedule_seal, protocol=5) != self._schedule_payload
        ):
            raise ValueError("weighted recurrent schedule changed")

    def _metadata(self) -> None:
        self._check_inference()
        self._check_origin()
        if type(self._policy_seal) is public_recurrent.PublicCueRecurrentPolicy and (
            type(self._learner) is not public_recurrent.PublicCueRecurrentLearner
            or self._catalogue is not None
        ):
            raise ValueError("weighted public-cue schedule requires its original two-head owner")
        if type(self._policy_seal) is InvariantTorchFeaturePolicy and (
            type(self._learner) is not InvariantTorchFeatureLearner
            or self._learner._native is not self._feature_native_owner
        ):
            raise ValueError("weighted bound feature native owner changed")
        if (
            type(self) is not TorchTrainWeightedLearner
            or self._learner is not self._native_owner
            or self._episodes is not self._schedule_owner
            or type(self._objective_id) is not str
            or self._objective_id
            != _objective(
                self._policy_seal, self._catalogue is not None, self._inference_policy is not None
            )
            or not _same_state(self._policy, self._policy_seal)
            or self._learner.policy != self._policy_seal
        ):
            raise ValueError("weighted schedule, objective, policy or native owner changed")
        self._schedule_metadata()

    def _progress(self, snapshot: Snapshot) -> None:
        count = snapshot.completed_updates
        if type(count) is not int:
            raise ValueError("weighted native counters must match the owned schedule prefix")
        count -= self._origin[1]
        limit = self._catalogue_updates if self._catalogue is not None else len(self._episodes)
        if (
            snapshot.policy != self._policy_seal
            or not 0 <= count <= limit
            or type(snapshot.training_examples) is not int
        ):
            raise ValueError("weighted native counters must match the owned schedule prefix")
        if type(self._policy_seal) is public_recurrent.PublicCueRecurrentPolicy:
            self._check_public_prefixes()
            first_updates, first_rows, second_updates, second_rows = self._public_prefixes[count]
            if snapshot.training_examples != first_rows + second_rows:
                raise ValueError("weighted native counters must match the owned schedule prefix")
            owned = public_recurrent.copy_snapshot(
                cast(public_recurrent.PublicCueRecurrentSnapshot, snapshot)
            )
            head_updates, head_rows = (first_updates, second_updates), (first_rows, second_rows)
            if any(
                h.completed_updates != head_updates[i] or h.training_examples != head_rows[i]
                for i, h in enumerate(owned.heads)
            ):
                raise ValueError(
                    "weighted public-cue heads differ from the original schedule prefix"
                )
        else:
            rows = (
                count * self._catalogue_rows
                if self._catalogue is not None
                else self._origin[2]
                + sum(len(episode.transitions) for episode in self._episodes[:count])
            )
            if snapshot.training_examples != rows:
                raise ValueError("weighted native counters must match the owned schedule prefix")

    def _check(self) -> Snapshot:
        self._metadata()
        snapshot = self._learner.snapshot_state()
        if type(self._policy_seal) is InvariantTorchFeaturePolicy:
            snapshot = copy_invariant_torch_feature_snapshot(snapshot)
        self._progress(snapshot)
        if not _same_state(snapshot, self._expected_state):
            raise ValueError("weighted native owner state changed outside an admitted operation")
        if self._public_failure is not None and (
            type(snapshot) is not public_recurrent.PublicCueRecurrentSnapshot
            or snapshot.failed_work is None
        ):
            raise ValueError("failed public-cue schedule has no verified terminal state")
        return snapshot

    @property
    def policy(self) -> Policy:
        self._check()
        return replace(self._policy_seal)

    @property
    def inference_policy(self) -> CoupledInferencePolicy | None:
        self._check()
        return (
            None
            if self._inference_policy is None
            else _copy_inference_policy(self._inference_policy)
        )

    @property
    def completed_updates(self) -> int:
        return self._check().completed_updates

    @property
    def training_examples(self) -> int:
        return self._check().training_examples

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        self._refuse_public_failure()
        before = self._check()
        admitted_native = self._learner
        if self._catalogue is not None:
            if before.completed_updates == self._catalogue_updates:
                raise ValueError("weighted catalogue training schedule is exhausted")
            catalogue = _copy_catalogue(self._catalogue)
            expected = compose_catalogue_training(catalogue.episodes, 1).training[0]
            _match_arrays(features, targets, (expected.features, expected.targets))
            return self._train_feature(before, admitted_native, catalogue, features, targets)
        index = before.completed_updates - self._origin[1]
        if index == len(self._episodes):
            raise ValueError("weighted training schedule is exhausted")
        episode = copy_episode(self._episodes[index])
        expected_features, expected_targets = episode_arrays(episode)
        _match_arrays(features, targets, (expected_features, expected_targets))
        if type(self._policy_seal) is InvariantTorchFeaturePolicy:
            return self._train_feature(before, admitted_native, episode, features, targets)
        if type(self._policy_seal) is public_recurrent.PublicCueRecurrentPolicy:
            return self._train_public(
                cast(public_recurrent.PublicCueRecurrentSnapshot, before),
                admitted_native,
                episode,
                features,
                targets,
            )
        weights = episode_class_weights(episode)
        # Why: bind the episode metadata and weights without inventing missing
        # transitions from arrays. The caller retains release/provenance authority.
        try:
            diagnostic = admitted_native.train_weighted_episode(episode, weights)
            self._metadata()
            after = admitted_native.snapshot_state()
            self._progress(after)
            if (
                after.completed_updates != before.completed_updates + 1
                or after.training_examples != before.training_examples + len(features)
                or diagnostic.definition != TORCH_RECURRENT_WEIGHTED_CE_ID
            ):
                raise ValueError(
                    "weighted native update returned inconsistent progression or objective"
                )
        except Exception:
            # The accepted native update is atomic. Retain that contract even
            # if an injected boundary violates its post-update metadata.
            admitted_native.restore_state(before)
            raise
        self._expected_state = deepcopy(after)
        return diagnostic

    def _refuse_public_failure(self) -> None:
        if self._public_failure is not None:
            reason, definition, work = self._public_failure
            raise TrainingUpdateFailed(reason, definition, dict(work))

    def _fail_public(
        self,
        error: Exception,
        before: public_recurrent.PublicCueRecurrentSnapshot,
        admitted: public_recurrent.PublicCueRecurrentLearner,
        returned: bool,
        rows: int,
    ) -> NoReturn:
        # Why: the public owner permanently retains paid failure. Never apply
        # the single-RNN rollback contract or retain an exception/traceback chain.
        work: dict[str, int | bool] = (
            dict(error.work)
            if isinstance(error, TrainingUpdateFailed)
            else dict(
                adapter_calls_attempted=1,
                adapter_calls_returned=int(returned),
                adapter_rows_attempted=rows,
                adapter_rows_returned=rows if returned else 0,
                committed_updates=before.completed_updates,
                committed_rows=before.training_examples,
                failure_work_may_be_unobserved=True,
            )
        )
        # Preserve every allowed native counter even when all32slots are used.
        # An absent read-proof flag is unverified, never evidence of success.
        can_record_proof = "failed_state_read_verified" in work or len(work) < 32
        if can_record_proof:
            work["failed_state_read_verified"] = False
        try:
            actual = public_recurrent.copy_snapshot(admitted.snapshot_state())
            if actual.failed_work is not None and _same_state(actual.heads, before.heads):
                self._progress(actual)
                self._expected_state = deepcopy(actual)
                if can_record_proof:
                    work["failed_state_read_verified"] = True
        except Exception:
            # Preserve the primary paid counters; an unavailable/corrupt final
            # read is explicitly unverified and cannot authorize continuation.
            if can_record_proof:
                work["failed_state_read_verified"] = False
        failure = TrainingUpdateFailed(
            f"{type(error).__name__}: {error}"[:512],
            error.definition
            if isinstance(error, TrainingUpdateFailed)
            else "public_cue_weighted_schedule_failed_work_v1",
            work,
        )
        self._public_failure = (str(failure), failure.definition, tuple(failure.work.items()))
        raise failure from error

    def _train_public(
        self,
        before: public_recurrent.PublicCueRecurrentSnapshot,
        admitted: public_recurrent.PublicCueRecurrentLearner,
        episode: DemonstrationEpisode,
        features: Array,
        targets: Array,
    ) -> TrainingDiagnostic:
        expected = episode_arrays(episode)
        returned = False
        try:
            diagnostic = admitted.train_class_balanced_episode(episode)
            returned = True
            self._metadata()
            if self._learner is not admitted:
                raise ValueError("original public-cue owner was replaced")
            _match_arrays(features, targets, expected)
            after = admitted.snapshot_state()
            self._progress(after)
            if (
                after.failed_work is not None
                or after.completed_updates != before.completed_updates + 1
                or after.training_examples != before.training_examples + len(features)
                or type(diagnostic) is not TrainingDiagnostic
                or diagnostic.definition != TORCH_RECURRENT_WEIGHTED_CE_ID
            ):
                raise ValueError("public-cue weighted update or diagnostic differs")
        except Exception as error:
            self._fail_public(error, before, admitted, returned, len(features))
        self._expected_state = deepcopy(after)
        return diagnostic

    def _feature_guard(self, admitted: InvariantTorchFeatureLearner) -> Callable[[], None]:
        """Seal non-serialized donor identity and caller state for one operation."""
        torch = require_torch()
        native = admitted._native
        payload, policy, native_policy = native._payload, admitted._policy, native._policy
        marks, rng = _autograd_marks(native), deepcopy(_rng(torch))
        modes = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())

        def check() -> None:
            self._check()
            if (
                self._learner is not admitted
                or admitted._native is not native
                or admitted._policy is not policy
                or native._policy is not native_policy
                or native._payload is not payload
                or _autograd_marks(native) != marks
                or not _same_state(rng, _rng(torch))
                or modes != (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
            ):
                raise ValueError(
                    "weighted bound feature donor identity, gradients, RNG or mode changed"
                )

        return check

    def _publish_feature(
        self,
        state: InvariantTorchFeatureSnapshot,
        admitted: InvariantTorchFeatureLearner,
        check_inputs: Callable[[], None],
        check_source: Callable[[], None],
        *,
        stream_work: dict[str, int] | None = None,
    ) -> None:
        owned = copy_invariant_torch_feature_snapshot(state)
        self._progress(owned)
        _charge_stream(stream_work, "publication_restores_attempted")
        head, optimizer = feature_native._restore_components(owned.native_state)
        _charge_stream(stream_work, "publication_restores_completed")
        _charge_stream(stream_work, "publication_validations_attempted")
        actual = feature_native._state(
            head,
            optimizer,
            owned.policy.native_policy,
            owned.completed_updates,
            owned.training_examples,
        )
        if not _same_state(actual, owned.native_state):
            raise ValueError("weighted detached restore differs from requested native state")
        _charge_stream(stream_work, "publication_validations_completed")
        expected = deepcopy(owned)
        check_source()
        check_inputs()
        check_source()
        if self._learner is not admitted:
            raise ValueError("weighted bound feature admitted owner changed")
        # Fresh detached components prevent a retained candidate/restore input
        # from sharing live tensors. No rollback can overwrite a foreign change.
        admitted._native._payload = (
            head,
            optimizer,
            owned.completed_updates,
            owned.training_examples,
        )
        self._expected_state = expected
        _charge_stream(stream_work, "published_updates")

    def _train_feature(
        self,
        before: Snapshot,
        admitted: InvariantTorchFeatureLearner,
        episode: DemonstrationEpisode | TrainCatalogue,
        features: Array,
        targets: Array,
    ) -> TrainingDiagnostic:
        if self._inference_policy is None:
            return self._execute_feature(before, admitted, episode, features, targets)
        native_work: dict[str, int | bool] = {}
        stream_work = dict.fromkeys(STREAM_WORK_FIELDS, 0)
        try:
            return self._execute_feature(
                before,
                admitted,
                episode,
                features,
                targets,
                native_work=native_work,
                stream_work=stream_work,
            )
        except _BoundedEpisodeFailure as error:
            raise ScheduledCategoricalUpdateFailed(
                str(error), TORCH_BOUNDED_WEIGHTED_PC_ID, native_work, stream_work
            ) from error.failure
        except Exception as error:
            native_work["failure_work_may_be_unobserved"] = True
            raise ScheduledCategoricalUpdateFailed(
                str(error) or type(error).__name__,
                TORCH_BOUNDED_WEIGHTED_PC_ID,
                native_work,
                stream_work,
            ) from error

    def _execute_feature(
        self,
        before: Snapshot,
        admitted: InvariantTorchFeatureLearner,
        episode: DemonstrationEpisode | TrainCatalogue,
        features: Array,
        targets: Array,
        *,
        native_work: dict[str, int | bool] | None = None,
        stream_work: dict[str, int] | None = None,
    ) -> TrainingDiagnostic:
        check_source = self._feature_guard(admitted)
        catalogue = type(episode) is TrainCatalogue
        if catalogue:
            batch = compose_catalogue_training(cast(TrainCatalogue, episode).episodes, 1).training[
                0
            ]
            expected = batch.features, batch.targets
        else:
            expected = episode_arrays(cast(DemonstrationEpisode, episode))
        _charge_stream(stream_work, "candidate_forks_attempted")
        candidate = admitted.fork()
        _charge_stream(stream_work, "candidate_forks_completed")
        check_source()
        if (
            type(candidate) is not InvariantTorchFeatureLearner
            or candidate is admitted
            or candidate._native is admitted._native
            or not _same_state(before, candidate.snapshot_state())
        ):
            raise ValueError("weighted bound feature requires an independent candidate")
        # Why: retain the same guarded candidate/publication path for each
        # explicit objective; complete native updates own the differing math.
        result: BoundedPCTrainingUpdate | None = None
        if self._inference_policy is not None:
            try:
                packet = cast(DemonstrationEpisode, episode)
                reserved = packet.role == "probe_train"
                bounded_update = (
                    train_bounded_probe_episode if reserved else train_bounded_weighted_episode
                )
                weights = (
                    probe_episode_class_weights(packet)
                    if reserved
                    else episode_class_weights(packet)
                )
                result = bounded_update(
                    candidate,
                    packet,
                    weights,
                    policy=self._inference_policy,
                )
            except TrainingUpdateFailed as error:
                if (
                    type(error) is not TrainingUpdateFailed
                    or error.definition != TORCH_BOUNDED_WEIGHTED_PC_ID
                ):
                    raise
                try:
                    observed = dict(copy_categorical_native_work(tuple(error.work.items())))
                except ValueError:
                    raise error
                assert native_work is not None
                native_work.update(observed)
                raise _BoundedEpisodeFailure(error) from error
            assert native_work is not None
            _record(result, BoundedPCTrainingUpdate)
            native_work.update(dict(copy_categorical_native_work(result._work_items)))
            diagnostic = result.diagnostic
        elif catalogue:
            item = cast(TrainCatalogue, episode)
            diagnostic = train_weighted_catalogue(candidate, item, catalogue_class_weights(item))
        else:
            packet = cast(DemonstrationEpisode, episode)
            update = (
                train_weighted_feature_episode
                if cast(InvariantTorchFeaturePolicy, self._policy_seal).native_policy.method == "bp"
                else train_weighted_pc_episode
            )
            diagnostic = update(candidate, packet, episode_class_weights(packet))
        after = candidate.snapshot_state()
        self._metadata()
        self._progress(after)
        if (
            after.completed_updates != before.completed_updates + 1
            or after.training_examples != before.training_examples + len(features)
            or type(diagnostic) is not TrainingDiagnostic
            or diagnostic.definition != self._objective_id
        ):
            raise ValueError(
                "weighted native update returned inconsistent progression or objective"
            )

        diagnostic = TrainingDiagnostic(diagnostic.definition, diagnostic.value)

        def check_inputs() -> None:
            _match_arrays(features, targets, expected)
            if catalogue:
                if _copy_catalogue(cast(TrainCatalogue, episode)) != self._catalogue:
                    raise ValueError("weighted admitted catalogue changed")
            elif (
                not _complete_fields((cast(DemonstrationEpisode, episode),))
                or copy_episode(cast(DemonstrationEpisode, episode))
                != self._episodes[before.completed_updates - self._origin[1]]
            ):
                raise ValueError("weighted bound feature admitted episode changed")

        if result is None:
            self._publish_feature(after, admitted, check_inputs, check_source)
            return diagnostic
        assert native_work is not None and stream_work is not None
        assert self._inference_policy is not None
        # Prepare/validate the owned scalar receipt before the publication port.
        # The forecast is returned only after actual counters match the commit.
        expected_work = dict(stream_work)
        for name in STREAM_WORK_FIELDS:
            if not name.startswith("candidate_forks"):
                expected_work[name] += 1
        rich = CategoricalPCTrainingDiagnostic(
            diagnostic.definition,
            diagnostic.value,
            self._inference_policy,
            result.inference,
            tuple(native_work.items()),
            tuple(expected_work.items()),
        )
        self._publish_feature(after, admitted, check_inputs, check_source, stream_work=stream_work)
        if stream_work != expected_work:
            raise ValueError("bounded scheduled publication work differs from its receipt")
        return rich

    def predict(self, features: Array) -> Array:
        self._refuse_public_failure()
        before = self._check()
        admitted, returned = self._learner, False
        try:
            result = admitted.predict(features)
            returned = True
            self._check()
        except Exception as error:
            if type(self._policy_seal) is public_recurrent.PublicCueRecurrentPolicy and (
                returned or isinstance(error, TrainingUpdateFailed)
            ):
                self._fail_public(
                    error,
                    cast(public_recurrent.PublicCueRecurrentSnapshot, before),
                    admitted,
                    returned,
                    len(features),
                )
            raise
        return result

    def start_episode(
        self,
    ) -> RecurrentEpisodePredictor | public_recurrent.PublicCueRecurrentEpisode:
        self._refuse_public_failure()
        self._check()
        if type(self._policy_seal) is InvariantTorchFeaturePolicy:
            raise ValueError("bound feature policy has no recurrent session; use predict")
        session = self._learner.start_episode()
        self._check()
        return session

    def snapshot_state(self) -> Snapshot:
        return self._check()

    def restore_state(self, snapshot: Snapshot) -> None:
        self._refuse_public_failure()
        before = self._check()
        admitted_native = self._learner
        if type(self._policy_seal) is public_recurrent.PublicCueRecurrentPolicy:
            owned_public = public_recurrent.copy_snapshot(
                cast(public_recurrent.PublicCueRecurrentSnapshot, snapshot)
            )
            self._progress(owned_public)
            if owned_public.failed_work is not None:
                raise ValueError("failed public-cue snapshots cannot enter a weighted schedule")
            returned = False
            try:
                admitted_native.restore_state(owned_public)
                returned = True
                self._metadata()
                after_public = admitted_native.snapshot_state()
                if self._learner is not admitted_native or not _same_state(
                    after_public, owned_public
                ):
                    raise ValueError("public-cue restore differs from its owned input")
            except Exception as error:
                self._fail_public(
                    error,
                    cast(public_recurrent.PublicCueRecurrentSnapshot, before),
                    admitted_native,
                    returned,
                    0,
                )
            self._expected_state = deepcopy(after_public)
            return
        if type(self._policy_seal) is InvariantTorchFeaturePolicy:
            check_source = self._feature_guard(admitted_native)
            owned = copy_invariant_torch_feature_snapshot(
                cast(InvariantTorchFeatureSnapshot, snapshot)
            )
            self._progress(owned)

            def check_input() -> None:
                if not _same_state(
                    owned,
                    copy_invariant_torch_feature_snapshot(
                        cast(InvariantTorchFeatureSnapshot, snapshot)
                    ),
                ):
                    raise ValueError("weighted bound feature restore input changed")

            self._publish_feature(owned, admitted_native, check_input, check_source)
            return
        self._progress(snapshot)
        try:
            admitted_native.restore_state(deepcopy(snapshot))
            self._metadata()
            after = admitted_native.snapshot_state()
            if not _same_state(after, snapshot):
                raise ValueError("weighted restore differs from the requested native state")
        except Exception:
            admitted_native.restore_state(before)
            raise
        self._expected_state = deepcopy(after)

    def fork(self) -> "TorchTrainWeightedLearner":
        self._refuse_public_failure()
        before = self._check()
        check_source = (
            self._feature_guard(self._learner)
            if type(self._policy_seal) is InvariantTorchFeaturePolicy
            else self._check
        )
        native = self._learner.fork()
        check_source()
        if native is self._learner or not _same_state(before, native.snapshot_state()):
            raise ValueError("weighted fork requires a distinct identical native owner")
        if type(self._policy_seal) is InvariantTorchFeaturePolicy and (
            type(native) is not InvariantTorchFeatureLearner
            or native._native is self._learner._native
        ):
            raise ValueError("weighted bound feature fork requires an independent native owner")
        child = object.__new__(TorchTrainWeightedLearner)
        child._policy = replace(self._policy_seal)
        child._policy_seal = replace(self._policy_seal)
        child._origin = child._origin_owner = (
            self._origin[0],
            self._origin[1],
            self._origin[2],
            self._origin[3],
        )
        child._configure_inference(self._inference_policy)
        child._episodes = tuple(copy_episode(episode) for episode in self._episodes)
        child._schedule_owner = child._episodes
        child._schedule_seal = tuple(copy_episode(episode) for episode in self._episodes)
        child._schedule_payload = self._schedule_payload
        child._catalogue = None
        if self._catalogue is not None:
            child._bind_catalogue(self._catalogue, self._catalogue_updates)
            self._catalogue_witness.check_copy(child._catalogue_witness)
            check_source()
        elif (
            type(self._policy_seal)
            in (InvariantTorchFeaturePolicy, public_recurrent.PublicCueRecurrentPolicy)
            and self._origin[0] == "train"
        ):
            child._bind_schedule()
            self._schedule_witness.check_copy(child._schedule_witness)
        child._objective_id = _objective(
            self._policy_seal, self._catalogue is not None, self._inference_policy is not None
        )
        child._learner = child._native_owner = native
        child._public_failure = None
        if type(self._policy_seal) is InvariantTorchFeaturePolicy:
            child._feature_native_owner = native._native
        child._expected_state = deepcopy(before)
        child._check()
        return child

    def __deepcopy__(self, memo: dict[int, Any]) -> "TorchTrainWeightedLearner":
        # Native heads retain the Torch module, which generic deepcopy cannot copy.
        # Use the existing full-state fork contract for both supported owners.
        child = self.fork()
        memo[id(self)] = child
        return child
