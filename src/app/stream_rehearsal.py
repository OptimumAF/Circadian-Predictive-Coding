"""Fixed train-only shared rehearsal through existing candidate consolidation.

The wake budget remains the runtime's original budget. A separate, birth-bound
replay budget charges attempted row exposures before native calls, including
failed calls. Reports distinguish returned work from published candidate work.
No heldout selection, policy adaptation, durable state or native gate bypass is
provided. Source learners must satisfy the owned-fork contract.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import isfinite
import pickle
from typing import Any, Callable

import numpy as np

from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ConsolidatedState, ForkableLearner
from src.core.circadian_predictive_coding import (
    ReplayRetentionBudget,
    ReplayRetentionSnapshot,
    replay_sample_id,
)
from src.core.coupled_energy import Array
from src.core.categorical_probability_scores import validate_class_targets
from src.core.experience import require_training_role
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    EpisodeReplaySchema,
    copy_episode,
    decode_episode,
    encode_episode,
    episode_arrays,
)
from src.core.demonstration_schedule_witness import DemonstrationScheduleWitness
from src.core.coverage_replay import CoverageReplayBuffer
from src.core.learner_ports import TrainingDiagnostic
from src.core.learner_ports import TrainingUpdateFailed
from src.core.context_joint_replay import JointContextReplayPolicy
from src.core.context_linear_state import ContextLinearFitSnapshot
from src.app.context_joint_rehearsal import prepare_joint_rehearsal
from src.core.incident_replay import IncidentReplayGate
from src.core.shift_incidents import ShiftIncidentDecision
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.shared_replay_schedule import (
    ReplayAllocation,
    SharedReplayBuffer,
    SharedReplaySelection,
)
from src.core.training_validation import validate_binary_training_batch
from src.core.slow_teacher import (
    SlowTeacherPolicy,
    TeacherTargets,
    copy_slow_teacher_policy,
    copy_teacher_targets,
)


def _teacher_port(learner: Any, policy: SlowTeacherPolicy) -> Any:
    """An opt-in inner contract; no concrete adapter dependency or policy fallback."""
    declared = getattr(learner, "teacher_policy", None)
    if type(declared) is not SlowTeacherPolicy or copy_slow_teacher_policy(declared) != policy:
        raise ValueError("rehearsal teacher policy differs from its original declared policy")
    if any(
        not callable(getattr(learner, name, None))
        for name in ("make_teacher_targets", "train_distillation_batch")
    ):
        raise ValueError("rehearsal distillation requires the complete teacher learner port")
    return learner


@dataclass(frozen=True)
class StreamReplayPolicy:
    every_updates: int
    rows_per_event: int
    retention: ReplayRetentionBudget
    allocation: ReplayAllocation
    seed: int = 53

    def __post_init__(self) -> None:
        if type(self.every_updates) is not int or not 1 <= self.every_updates <= 512:
            raise ValueError("rehearsal every_updates must be a native integer in 1..512")
        if type(self.rows_per_event) is not int or not 1 <= self.rows_per_event <= 128:
            raise ValueError("rehearsal rows_per_event must be a native integer in 1..128")
        if (
            type(self.retention) is not ReplayRetentionBudget
            or type(self.allocation) is not ReplayAllocation
        ):
            raise ValueError("rehearsal requires exact retention and allocation records")
        retention = replace(self.retention)
        if retention.max_examples > 128 or retention.max_bytes > 1024**2:
            raise ValueError("rehearsal retention is bounded to 128 rows and 1 MiB of arrays")
        allocation = ReplayAllocation(self.allocation.recent, self.allocation.reservoir)
        if (
            allocation.recent.max_examples + allocation.reservoir.max_examples
            > retention.max_examples
            or allocation.recent.max_bytes + allocation.reservoir.max_bytes > retention.max_bytes
        ):
            raise ValueError("rehearsal allocation must fit the global retention caps")
        ReplayRetentionPolicy("seeded_reservoir", self.seed)
        object.__setattr__(self, "retention", retention)
        object.__setattr__(self, "allocation", allocation)


@dataclass(frozen=True)
class RecentStreamReplayPolicy:
    """Matched rehearsal quotas with no inapplicable reservoir controls."""

    every_updates: int
    rows_per_event: int
    retention: ReplayRetentionBudget

    def __post_init__(self) -> None:
        if type(self.every_updates) is not int or not 1 <= self.every_updates <= 512:
            raise ValueError("rehearsal every_updates must be a native integer in 1..512")
        if type(self.rows_per_event) is not int or not 1 <= self.rows_per_event <= 128:
            raise ValueError("rehearsal rows_per_event must be a native integer in 1..128")
        if type(self.retention) is not ReplayRetentionBudget:
            raise ValueError("recent rehearsal requires an exact retention budget")
        retention = replace(self.retention)
        if retention.max_examples > 128 or retention.max_bytes > 1024**2:
            raise ValueError("rehearsal retention is bounded to 128 rows and 1 MiB of arrays")
        object.__setattr__(self, "retention", retention)


@dataclass(frozen=True)
class CoverageStreamReplayPolicy:
    """Explicit training-context information with fixed bounded replay quotas."""

    every_updates: int
    rows_per_event: int
    retention: ReplayRetentionBudget
    contexts: tuple[str, ...]

    def __post_init__(self) -> None:
        validated = RecentStreamReplayPolicy(
            self.every_updates, self.rows_per_event, self.retention
        )
        if (
            type(self.contexts) is not tuple
            or not 1 <= len(self.contexts) <= 6
            or any(
                type(key) is not str
                or not key
                or key != key.strip()
                or len(key.encode("utf8")) > 32
                for key in self.contexts
            )
            or len(set(self.contexts)) != len(self.contexts)
        ):
            raise ValueError(
                "coverage requires 1..6 unique declared context strings of 1..32 UTF-8 bytes"
            )
        object.__setattr__(self, "retention", validated.retention)


def _sessions(wake: ToyBudgetSession, replay: ToyBudgetSession, *, fresh: bool) -> None:
    if type(wake) is not ToyBudgetSession or type(replay) is not ToyBudgetSession or wake is replay:
        raise ValueError("rehearsal requires distinct exact wake and replay budget sessions")
    for session in (wake, replay):
        if type(session.budget) is not ToyExecutionBudget:
            raise ValueError("rehearsal requires exact budget limit records")
        replace(session.budget)
        if session.budget.max_hidden_width is not None:
            raise ValueError("rehearsal does not support hidden-width budgets")
        for count in (session.updates_completed, session.replay_examples_completed):
            if type(count) is not int or count < 0 or (fresh and count != 0):
                raise ValueError(
                    "rehearsal budget counters must be native nonnegative and fresh at birth"
                )
    if wake.budget.max_replay_examples is not None:
        raise ValueError("wake runtime budget must not contain a replay limit")
    if replay.budget.max_replay_examples is None:
        raise ValueError("rehearsal replay budget requires a finite replay-example allowance")


@dataclass(frozen=True)
class _RehearsalContinuation:
    """Owned local data only; binding IDs convey no transferable authority."""

    format_version: int
    policy: StreamReplayPolicy | RecentStreamReplayPolicy | CoverageStreamReplayPolicy
    buffer: SharedReplayBuffer
    observed: int
    events: list[dict[str, Any]]
    stopped: bool
    reason: str | None
    binding: tuple[Any, ...]
    incident_gate: IncidentReplayGate | None = None


def _continuation_budget(session: ToyBudgetSession) -> tuple[Any, ...]:
    for value in (session.started_at, session.last_clock):
        if type(value) not in {int, float} or not isfinite(value):
            raise ValueError("rehearsal continuation requires finite original clock metadata")
    if session.last_clock < session.started_at:
        raise ValueError("rehearsal continuation clock metadata regressed")
    # Deadline/RSS admission belongs to the original stream owner. Its checks
    # may advance observational telemetry; none of it is restored or renewed.
    return (
        id(session),
        id(session.budget),
        asdict(session.budget),
        id(session.clock),
        session.started_at,
        session.updates_completed,
        session.replay_examples_completed,
        id(session.progress),
        id(session.process_rss_sampler),
        session.hidden_width_observed,
        session.peak_hidden_width_observed,
        session.rejected_proposed_hidden_width,
        deepcopy(session.checkpoint_position),
    )


def _episode_budget(
    budget: ReplayRetentionBudget, schema: EpisodeReplaySchema
) -> ReplayRetentionBudget:
    """Convert native-transition caps to whole encoded rows without extending them."""
    budget = replace(budget)
    count = budget.max_examples // schema.steps_per_episode
    if count < 1 or budget.max_bytes < 8 * (schema.encoded_dim + 1):
        raise ValueError("episode replay quotas must hold one whole encoded episode")
    return ReplayRetentionBudget(count, budget.max_bytes)


class StreamRehearsal:
    def __init__(
        self,
        learner: ForkableLearner[Array, Array, Array, Any],
        budget: ToyBudgetSession,
        policy: StreamReplayPolicy | RecentStreamReplayPolicy | CoverageStreamReplayPolicy,
        *,
        replay_budget: ToyBudgetSession,
        total_updates: int,
        input_dim: int,
        distillation: SlowTeacherPolicy | None = None,
        episode_schema: EpisodeReplaySchema | None = None,
        incident_channel: str | None = None,
        fixed_training_episodes: tuple[DemonstrationEpisode, ...] = (),
        wake_role: str = "train",
        check_boundary: Callable[[], None] | None = None,
        joint_context: JointContextReplayPolicy | None = None,
    ) -> None:
        if check_boundary is not None and not callable(check_boundary):
            raise ValueError("rehearsal boundary must be callable or None")
        self._check_boundary = self._original_check_boundary = check_boundary
        self._wake_role = self._original_wake_role = require_training_role(wake_role)
        if self._wake_role == "probe_train" and not fixed_training_episodes:
            raise ValueError("probe rehearsal requires a nonempty fixed TRAIN bank")
        if type(fixed_training_episodes) is not tuple:
            raise ValueError("fixed training episodes require an exact native tuple")
        if fixed_training_episodes and (
            type(policy) is not RecentStreamReplayPolicy
            or episode_schema is None
            or distillation is not None
            or incident_channel is not None
        ):
            raise ValueError(
                "fixed training episodes require recent episode replay without optional modes"
            )
        if type(policy) not in {
            StreamReplayPolicy,
            RecentStreamReplayPolicy,
            CoverageStreamReplayPolicy,
        }:
            raise ValueError(
                "rehearsal requires an exact partitioned, recent or coverage replay policy"
            )
        if type(total_updates) is not int or not 1 <= total_updates <= 512:
            raise ValueError("rehearsal total_updates must be a native integer in 1..512")
        if episode_schema is not None and type(episode_schema) is not EpisodeReplaySchema:
            raise ValueError("episode rehearsal requires an exact EpisodeReplaySchema")
        if type(policy) is CoverageStreamReplayPolicy and episode_schema is None:
            raise ValueError("coverage rehearsal requires a complete episode schema")
        self._episode_schema = None if episode_schema is None else replace(episode_schema)
        self._episode_schema_birth = (
            id(self._episode_schema),
            None
            if self._episode_schema is None
            else (
                self._episode_schema.observation_dim,
                self._episode_schema.steps_per_episode,
                self._episode_schema.num_classes,
            ),
        )
        if joint_context is not None:
            if type(joint_context) is not JointContextReplayPolicy:
                raise ValueError("joint context requires an exact JointContextReplayPolicy")
            joint_context = replace(joint_context)
            state = learner.snapshot_state()
            if (
                type(policy) is not RecentStreamReplayPolicy
                or episode_schema is None
                or distillation is not None
                or incident_channel is not None
                or type(state) is not ContextLinearFitSnapshot
                or (
                    state.policy.feature_dim,
                    state.policy.num_classes,
                    state.policy.context_columns,
                )
                != (
                    joint_context.feature_dim,
                    joint_context.num_classes,
                    joint_context.context_columns,
                )
                or episode_schema.num_classes != joint_context.num_classes
                or episode_schema.observation_dim != joint_context.feature_dim
            ):
                raise ValueError(
                    "joint context requires original categorical LS episode replay geometry alone"
                )
        self._joint_context = joint_context
        self._joint_context_owner = joint_context
        self._joint_context_pin = deepcopy(joint_context)
        self._joint_paid: list[tuple[Any, Any]] = []
        if self._episode_schema is not None:
            if distillation is not None:
                raise ValueError("episode rehearsal does not support teacher distillation")
            if type(input_dim) is not int or input_dim != self._episode_schema.observation_dim:
                raise ValueError("episode schema observation_dim must match rehearsal input_dim")
        _sessions(budget, replay_budget, fresh=True)
        self._distillation = (
            None if distillation is None else copy_slow_teacher_policy(distillation)
        )
        self._teacher_storage_bytes = 0
        if self._distillation is not None:
            size = _teacher_port(learner, self._distillation).teacher_storage_bytes
            if type(size) is not int or size < 0:
                raise ValueError(
                    "rehearsal teacher storage must be a nonnegative native byte count"
                )
            self._teacher_storage_bytes = size
        self._policy = replace(policy)
        retention = self._policy.retention
        allocation = (
            self._policy.allocation if isinstance(self._policy, StreamReplayPolicy) else None
        )
        buffer_dim = input_dim
        if self._episode_schema is not None:
            schema = self._episode_schema
            if self._policy.rows_per_event < schema.steps_per_episode:
                raise ValueError("episode replay event cap must hold one whole episode")
            retention = _episode_budget(retention, schema)
            allocation = (
                None
                if allocation is None
                else ReplayAllocation(
                    _episode_budget(allocation.recent, schema),
                    _episode_budget(allocation.reservoir, schema),
                )
            )
            buffer_dim = schema.encoded_dim
        self._buffer: SharedReplayBuffer
        if isinstance(self._policy, CoverageStreamReplayPolicy):
            assert self._episode_schema is not None
            self._buffer = CoverageReplayBuffer(
                buffer_dim, retention, schema=self._episode_schema, contexts=self._policy.contexts
            )
        elif isinstance(self._policy, RecentStreamReplayPolicy):
            self._buffer = SharedReplayBuffer(
                buffer_dim, retention, ReplayRetentionPolicy("recent_fifo")
            )
        else:
            self._buffer = SharedReplayBuffer(
                buffer_dim,
                retention,
                ReplayRetentionPolicy("seeded_reservoir", self._policy.seed),
                allocation=allocation,
            )
        self._fixed_bank_birth: tuple[Any, ...] | None = None
        if fixed_training_episodes:
            assert self._episode_schema is not None
            schema = self._episode_schema
            native_count = len(fixed_training_episodes) * schema.steps_per_episode
            encoded_bytes = len(fixed_training_episodes) * 8 * (schema.encoded_dim + 1)
            if (
                native_count > self._policy.retention.max_examples
                or native_count > self._policy.rows_per_event
                or encoded_bytes > self._policy.retention.max_bytes
            ):
                raise ValueError(
                    "the complete fixed training bank must fit retention and every event"
                )
            # Why: establish exact complete TRAIN records once, then keep only
            # encoded arrays and their existing IDs, rather than a second bank.
            DemonstrationScheduleWitness(fixed_training_episodes)
            episode_ids = tuple(episode.episode_id for episode in fixed_training_episodes)
            if len(set(episode_ids)) != len(episode_ids):
                raise ValueError("fixed training episode IDs must be unique")
            encoded = tuple(encode_episode(episode, schema) for episode in fixed_training_episodes)
            for features, targets in encoded:
                self._buffer.observe_train_batch(features, targets)
            self._fixed_bank_birth = (
                episode_ids,
                self._buffer.retained_order_ids,
                native_count,
                encoded_bytes,
                id(self._policy),
                pickle.dumps(self._policy),
                self._fixed_buffer_configuration(),
            )
            if self._wake_role != "train":
                self._fixed_bank_birth += (self._wake_role,)
        self._fixed_bank_owner = self._fixed_bank_birth
        self._remember_fixed_buffer_owner()
        self._learner, self._budget, self._replay_budget = learner, budget, replay_budget
        self._joint_budget_birth = (
            self._joint_budget_signature() if joint_context is not None else None
        )
        self._joint_replay_work = 0
        self._total_updates, self._input_dim = total_updates, input_dim
        self._incident_channel_birth = incident_channel
        self._incident_gate = (
            None
            if incident_channel is None
            else IncidentReplayGate(incident_channel, self._policy.every_updates, total_updates)
        )
        self._incident_gate_owner = self._incident_gate
        self._bound = False
        self._runtime: ActorShadowRuntime[Array, Array, Array, Any] | None = None
        self._observed = 0
        self._stopped = False
        self._reason: str | None = None
        self._events: list[dict[str, Any]] = [
            dict(
                updates=k,
                event_id=f"rehearsal-{k}",
                status="not_run",
                reason="event not reached",
                selected_ids=[],
                charged_exposures=0,
                attempted_updates=0,
                returned_updates=0,
                committed_updates=0,
                diagnostics=[],
                receipt=None,
                retention=None,
                partition_retention=None,
                rotation_cursor=None,
            )
            for k in range(
                self._policy.every_updates, total_updates + 1, self._policy.every_updates
            )
        ]
        if self._distillation is not None:
            for row in self._events:
                row.update(
                    teacher_forward_attempts=0,
                    teacher_forward_returns=0,
                    teacher_probability_bytes_generated=0,
                    mixed_target_bytes_returned=0,
                    teacher_peak_probability_bytes=0,
                    teacher_targets=[],
                )
        if self._episode_schema is not None:
            for row in self._events:
                row["selected_episodes"] = []
        self._joint_events_owner = self._events
        self._joint_event_rows = tuple(self._events) if joint_context is not None else ()

    def _fixed_buffer_configuration(self) -> bytes:
        buffer = self._buffer
        return pickle.dumps(
            (
                buffer.input_dim,
                buffer.budget,
                buffer.policy,
                buffer._allocation,
                buffer._recent,
                buffer._reservoir,
                buffer._rotation_cursor,
            )
        )

    def _remember_fixed_buffer_owner(self) -> None:
        if self._fixed_bank_birth is None:
            return
        self._fixed_buffer_owner = self._buffer
        self._fixed_rows_owner = self._buffer._rows
        self._fixed_array_owners = tuple(
            (id(rows), id(rows[0]), id(rows[1])) for rows in self._buffer._rows.values()
        )

    def _check_fixed_bank(self) -> None:
        if require_training_role(self._wake_role) != self._original_wake_role:
            raise ValueError("rehearsal requires its original wake role")
        birth = self._fixed_bank_birth
        if birth is not self._fixed_bank_owner:
            raise ValueError("fixed training bank authority changed")
        if birth is None:
            if self._wake_role != "train":
                raise ValueError("probe rehearsal lost its fixed TRAIN bank")
            return
        if (
            self._wake_role != (birth[7] if len(birth) > 7 else "train")
            or type(self._policy) is not RecentStreamReplayPolicy
            or id(self._policy) != birth[4]
            or pickle.dumps(self._policy) != birth[5]
            or self._distillation is not None
            or self._incident_channel_birth is not None
            or self._incident_gate is not None
            or type(self._buffer) is not SharedReplayBuffer
            or self._buffer is not self._fixed_buffer_owner
            or self._buffer._rows is not self._fixed_rows_owner
            or type(self._buffer._rows) is not dict
            or tuple(self._buffer._rows) != birth[1]
            or self._fixed_buffer_configuration() != birth[6]
        ):
            raise ValueError("fixed training bank or replay authority changed")
        schema = self._episode_schema
        assert schema is not None
        owners = []
        for sample_id, rows in self._buffer._rows.items():
            if type(rows) is not tuple or len(rows) != 2:
                raise ValueError("fixed training bank row layout changed")
            features, targets = rows
            if any(
                type(values) is not np.ndarray
                or values.dtype != np.float64
                or values.shape != shape
                or not values.flags.writeable
                for values, shape in ((features, (1, schema.encoded_dim)), (targets, (1, 1)))
            ):
                raise ValueError("fixed training bank array layout changed")
            if replay_sample_id(features, targets) != sample_id:
                raise ValueError("fixed training bank content changed")
            owners.append((id(rows), id(features), id(targets)))
        if tuple(owners) != self._fixed_array_owners:
            raise ValueError("fixed training bank arrays were replaced")

    def _joint_budget_signature(self):
        return tuple(
            (
                id(s),
                id(s.budget),
                deepcopy(s.budget),
                id(s.clock),
                s.started_at,
                id(s.process_rss_sampler),
            )
            for s in (self._budget, self._replay_budget)
        )

    def _check_schema(self) -> EpisodeReplaySchema | None:
        """Keep codec, optional joint configuration and paid evidence source-bound."""
        if self._joint_context is not None and (
            self._joint_budget_signature() != self._joint_budget_birth
            or self._replay_budget.replay_examples_completed != self._joint_replay_work
        ):
            self._stopped, self._reason = True, "original joint replay budget changed"
            raise ValueError(self._reason)
        if (
            self._joint_context is not self._joint_context_owner
            or self._joint_context != self._joint_context_pin
        ):
            raise ValueError("original joint replay policy changed")
        changed_container = self._joint_context is not None and (
            self._events is not self._joint_events_owner
            or len(self._events) != len(self._joint_event_rows)
        )
        changed_paid = not changed_container and any(
            self._events[value["updates"] // self._policy.every_updates - 1] is not row
            or pickle.dumps(row) != pickle.dumps(value)
            for row, value in self._joint_paid
        )
        if changed_container or changed_paid:
            self._events = self._joint_events_owner
            self._events[:] = self._joint_event_rows
            for row, value in self._joint_paid:
                index = value["updates"] // self._policy.every_updates - 1
                row.clear()
                row.update(deepcopy(value))
                self._events[index] = row
            self._stopped, self._reason = True, "preceding paid joint replay evidence changed"
            raise ValueError(self._reason)
        if self._check_boundary is not self._original_check_boundary:
            raise ValueError("original replay boundary was replaced")
        schema = self._episode_schema
        if schema is not None:
            if type(schema) is not EpisodeReplaySchema:
                raise ValueError("episode replay schema type changed")
            validated = replace(schema)
            content = (
                validated.observation_dim,
                validated.steps_per_episode,
                validated.num_classes,
            )
        else:
            content = None
        if (id(schema), content) != self._episode_schema_birth:
            raise ValueError("episode replay schema differs from its original identity or geometry")
        self._check_fixed_bank()
        return schema

    @property
    def episode_schema(self) -> EpisodeReplaySchema | None:
        """Expose detached geometry for stream-ingress preflight, without authority."""
        schema = self._check_schema()
        return None if schema is None else replace(schema)

    @property
    def packet_episode_replay(self) -> bool:
        """Only ordinary dynamic joint replay admits multi-episode paid packets."""
        self._check_schema()
        return (
            self._joint_context is not None
            and self._wake_role == "train"
            and self._fixed_bank_birth is None
        )

    def bind(
        self,
        learner: ForkableLearner[Array, Array, Array, Any],
        budget: ToyBudgetSession,
        total_updates: int,
        input_dim: int,
        *,
        wake_role: str = "train",
    ) -> int:
        self._check_schema()
        if self._bound:
            raise ValueError("rehearsal is single-use and already bound")
        if (
            require_training_role(wake_role) != self._wake_role
            or learner is not self._learner
            or budget is not self._budget
            or type(total_updates) is not int
            or total_updates != self._total_updates
            or type(input_dim) is not int
            or input_dim != self._input_dim
        ):
            raise ValueError(
                "rehearsal binding must match its original learner, budget and geometry"
            )
        replace(self._policy)
        _sessions(self._budget, self._replay_budget, fresh=True)
        if self._distillation is not None:
            _teacher_port(self._learner, self._distillation)
        self._bound = True
        return len(self._events)

    def _continuation_binding(self) -> tuple[Any, ...]:
        if self._original_check_boundary is not None:
            raise ValueError("externally guarded replay has no continuation contract")
        self._check_schema()
        self._check_incident_gate()
        if self._distillation is not None:
            raise ValueError("distillation rehearsal continuation is not yet supported")
        if self._bound is not True or self._stopped is not False or self._reason is not None:
            raise ValueError("rehearsal continuation requires its bound open helper")
        replace(self._policy)
        _sessions(self._budget, self._replay_budget, fresh=False)
        if (
            type(self._observed) is not int
            or not 0 <= self._observed <= self._total_updates
            or self._observed != self._budget.updates_completed
        ):
            raise ValueError("rehearsal continuation requires unchanged completed wake work")
        # The stream owner holds runtime._exclusive across capture/prepare and
        # native/inbox staging. Public runtime readers would reenter that gate.
        if self._runtime is None:
            if self._observed != 0:
                raise ValueError("rehearsal continuation lost its original runtime")
        elif (
            self._runtime._budget is not self._budget
            or len(self._runtime._inbox._applied) != self._observed
        ):
            raise ValueError("rehearsal continuation wake receipts differ")
        binding: tuple[Any, ...] = (
            id(self),
            id(self._learner),
            id(self._runtime),
            id(self._policy),
            id(self._buffer),
            id(self._events),
            self._bound,
            self._total_updates,
            self._input_dim,
            _continuation_budget(self._budget),
            _continuation_budget(self._replay_budget),
            self._episode_schema_birth,
        )
        binding = binding if self._wake_role == "train" else (*binding, self._wake_role)
        return (
            binding
            if self._joint_context is None
            else (*binding, id(self._joint_context), self._joint_context)
        )

    def _capture_continuation(self) -> _RehearsalContinuation:
        """Copy bounded NumPy replay data while the owner holds its runtime lease."""
        binding = self._continuation_binding()
        return _RehearsalContinuation(
            1,
            replace(self._policy),
            deepcopy(self._buffer),
            self._observed,
            deepcopy(self._events),
            self._stopped,
            self._reason,
            binding,
            deepcopy(self._incident_gate),
        )

    def _prepare_continuation(self, snapshot: _RehearsalContinuation) -> _RehearsalContinuation:
        """Validate same-owner unchanged data, then detach a complete candidate.

        This private boundary accepts only trusted local snapshots. Pickle is
        used for pure NumPy content comparison, never loading external bytes.
        Native state and original application authority remain with the owner.
        """
        if (
            type(snapshot) is not _RehearsalContinuation
            or vars(snapshot).keys() != _RehearsalContinuation.__dataclass_fields__.keys()
            or type(snapshot.format_version) is not int
            or snapshot.format_version != 1
        ):
            raise ValueError("rehearsal continuation requires its complete format-1 record")
        current = self._capture_continuation()
        schema = self._check_schema()
        steps = 1 if schema is None else schema.steps_per_episode
        buffer_dim = self._input_dim if schema is None else schema.encoded_dim
        if (
            type(snapshot.buffer) is not type(self._buffer)
            or type(snapshot.events) is not list
            or len(snapshot.events) != len(current.events)
            or vars(snapshot.buffer).keys() != vars(current.buffer).keys()
            or type(snapshot.buffer._rows) is not dict
            or len(snapshot.buffer._rows) > self._policy.retention.max_examples // steps
        ):
            raise ValueError("rehearsal continuation buffer or event layout differs")
        retained_bytes = 0
        for rows in snapshot.buffer._rows.values():
            if type(rows) is not tuple or len(rows) != 2:
                raise ValueError("rehearsal continuation retained row layout differs")
            for values, shape in zip(rows, ((1, buffer_dim), (1, 1)), strict=True):
                if (
                    type(values) is not np.ndarray
                    or values.dtype != np.float64
                    or values.shape != shape
                    or not values.flags.writeable
                ):
                    raise ValueError("rehearsal continuation retained array layout differs")
                retained_bytes += values.nbytes
            if schema is not None:
                decode_episode(rows[0], rows[1], schema)
        if retained_bytes > self._policy.retention.max_bytes:
            raise ValueError("rehearsal continuation exceeds its original byte allowance")
        if pickle.dumps(snapshot) != pickle.dumps(current):
            raise ValueError("rehearsal continuation differs from its unchanged current helper")
        return deepcopy(snapshot)

    def _commit_continuation(self, prepared: _RehearsalContinuation) -> None:
        """Publish already prepared data; no callbacks or authority replacement."""
        self._buffer = prepared.buffer
        self._events = prepared.events
        self._observed = prepared.observed
        self._stopped = prepared.stopped
        self._reason = prepared.reason
        self._incident_gate = prepared.incident_gate
        self._incident_gate_owner = self._incident_gate
        if self._joint_context is not None:
            # prepare validated exact same-owner content before this authorized
            # replacement; pins follow those detached rows without renewing work.
            self._joint_events_owner = self._events
            self._joint_event_rows = tuple(self._events)
            self._joint_paid = [
                (r, deepcopy(r)) for r in self._events if r["updates"] <= self._observed
            ]
        self._remember_fixed_buffer_owner()

    def _check_incident_gate(self) -> IncidentReplayGate | None:
        gate = self._incident_gate
        if gate is not self._incident_gate_owner:
            raise ValueError("rehearsal incident gate must remain its owned original state")
        if self._incident_channel_birth is None:
            if gate is not None:
                raise ValueError("periodic rehearsal cannot acquire incident authority")
            return None
        if type(gate) is not IncidentReplayGate:
            raise ValueError("incident rehearsal requires its complete original gate")
        config = gate.report()
        if (config["channel"], config["every_updates"], config["total_updates"]) != (
            self._incident_channel_birth,
            self._policy.every_updates,
            self._total_updates,
        ):
            raise ValueError("rehearsal incident configuration differs from its birth policy")
        return gate

    def observe_incident(self, decision: ShiftIncidentDecision, *, completed_updates: int) -> None:
        """Copy arrived-only incident state after this helper's actual wake receipt.

        Why: the stream invokes checkpoint observers after rehearsal. An onset
        seen here can authorize only a later declared boundary, never this wake.
        """
        self._check_schema()
        gate = self._check_incident_gate()
        if gate is None or not self._bound or self._stopped or self._runtime is None:
            raise ValueError("incident observation requires bound open incident rehearsal")
        if (
            type(completed_updates) is not int
            or completed_updates != self._observed
            or completed_updates != self._budget.updates_completed
            or len(self._runtime._inbox._applied) != completed_updates
            or self._runtime._budget is not self._budget
        ):
            raise ValueError("incident observation requires its actual completed wake receipt")
        _sessions(self._budget, self._replay_budget, fresh=False)
        try:
            gate.observe(deepcopy(decision), completed_updates=completed_updates)
        except BaseException as error:
            self._stopped, self._reason = True, f"{type(error).__name__}: {error}"
            raise

    def check_incident_progress(self, completed_updates: int) -> None:
        """Require the preceding arrived observation before admitting another wake."""
        self._check_schema()
        gate = self._check_incident_gate()
        if gate is not None and (
            gate.report()["observed_updates"] != completed_updates
            or self._observed != completed_updates
        ):
            raise ValueError("incident rehearsal is missing its preceding arrived observation")

    def _before_native(self) -> None:
        self._check_schema()
        if self._check_boundary is not None:
            self._check_boundary()
        self._budget.before_sleep()
        self._replay_budget.before_sleep()
        if self._check_boundary is not None:
            self._check_boundary()

    def _distillation_step(
        self, learner: Any, features: Array, targets: Array, row: dict[str, Any]
    ) -> TrainingDiagnostic:
        policy = self._distillation
        if policy is None:
            raise ValueError("distillation replay requires its predeclared teacher policy")
        port = _teacher_port(learner, policy)
        packet: TeacherTargets | None = None
        if policy.mixture_weight > 0:
            row["teacher_forward_attempts"] += 1
            packet = port.make_teacher_targets(
                features,
                targets,
                role="train",
                feature_schema=policy.feature_schema,
                output_schema=policy.output_schema,
            )
            row["teacher_forward_returns"] += 1
            packet = copy_teacher_targets(packet)
            if packet.policy != policy:
                raise ValueError("generated teacher targets differ from original rehearsal policy")
            size = packet.probabilities.nbytes
            row["teacher_probability_bytes_generated"] += size
            row["teacher_peak_probability_bytes"] = max(row["teacher_peak_probability_bytes"], size)
            row["teacher_targets"].append(
                dict(
                    teacher_revision=packet.teacher_revision,
                    student_updates=packet.student_updates,
                    sample_ids=list(packet.sample_ids),
                    probability_array_bytes=size,
                )
            )
            self._before_native()
        # Teacher admission and native replay work are different permanent costs.
        self._replay_budget.record_replay(len(features))
        row["attempted_updates"] += 1
        diagnostic = port.train_distillation_batch(features, targets, packet)
        if policy.mixture_weight > 0:
            row["mixed_target_bytes_returned"] += targets.nbytes
        return diagnostic

    def _operation(
        self,
        state: Any,
        selection: SharedReplaySelection,
        row: dict[str, Any],
        joint_packet: tuple[Array, Array] | None = None,
        packet_guard: Callable[[], None] | None = None,
    ) -> ConsolidatedState[Any]:
        before = self._before_native if packet_guard is None else packet_guard
        self._check_schema()
        before()
        learner = self._learner.fork()
        if learner is self._learner:
            raise ValueError("rehearsal fork must be distinct from its source learner")
        before()
        learner.restore_state(deepcopy(state))
        before()
        diagnostic = None
        for features, targets in (
            (joint_packet,) if joint_packet is not None else selection.training_batches()
        ):
            before()
            if self._distillation is None:
                if joint_packet is None:
                    self._replay_budget.record_replay(len(features))
                else:
                    charged_before, returned = self._joint_replay_work, False
                    try:
                        self._replay_budget.record_replay(len(features))
                        returned = True
                    finally:
                        delta = self._replay_budget.replay_examples_completed - charged_before
                        self._joint_replay_work += delta
                        if delta not in (0, len(features)) or returned and delta != len(features):
                            row["replay_charge_unknown"] = True
                            raise ValueError("joint replay observed an invalid charge transition")
                row["attempted_updates"] += 1
                try:
                    diagnostic = learner.train_batch(features, targets)
                except TrainingUpdateFailed as error:
                    if joint_packet is not None:
                        row["native_failure_work"] = dict(
                            definition=error.definition, work=error.work
                        )
                    raise
            else:
                diagnostic = self._distillation_step(learner, features, targets, row)
            row["returned_updates"] += 1
            if type(diagnostic) is not TrainingDiagnostic:
                raise ValueError(
                    "rehearsal native calls must return exact TrainingDiagnostic records"
                )
            diagnostic = replace(diagnostic)
            row["diagnostics"].append(asdict(diagnostic))
            before()
        if diagnostic is None:
            raise ValueError("rehearsal selection must contain at least one row")
        result = learner.snapshot_state()
        before()
        # The receipt retains the last native pre-update diagnostic; all native
        # diagnostics remain in the event report, with their original objectives.
        return ConsolidatedState(deepcopy(result), diagnostic)

    def _run_event(
        self,
        runtime: ActorShadowRuntime[Array, Array, Array, Any],
        row: dict[str, Any],
        current: tuple[Array, Array] | None = None,
        packet_guard: Callable[[], None] | None = None,
    ) -> None:
        initial_charge = self._replay_budget.replay_examples_completed
        try:
            schema = self._check_schema()
            steps = 1 if schema is None else schema.steps_per_episode
            maximum = self._policy.rows_per_event // steps
            if type(self._policy) is CoverageStreamReplayPolicy:
                if type(self._buffer) is not CoverageReplayBuffer:
                    raise ValueError("coverage rehearsal lost its configured buffer")
                selection = self._buffer.select_coverage(maximum)
                partitions = None
                cursor = None
                row.update(
                    coverage_keys=list(self._buffer.coverage_keys),
                    coverage_cursor=self._buffer.coverage_cursor,
                )
            elif type(self._policy) is RecentStreamReplayPolicy:
                # Preserve the existing selector's latest-n arrival chronology.
                selection = self._buffer.select_recent(
                    self._policy.retention.max_examples // steps
                    if self._joint_context is not None
                    else maximum
                )
                partitions = None
                cursor = None
            else:
                selection = self._buffer.select_partitioned_rotating(maximum)
                partitions = self._buffer.partition_retention
                if partitions is None:
                    raise ValueError("rehearsal buffer lost its configured allocation")
                cursor = asdict(self._buffer.rotation_cursor)
            joint_packet = None
            if self._joint_context is not None:
                assert schema is not None and current is not None and packet_guard is not None
                packet_guard()
                selection, joint_packet, detail = prepare_joint_rehearsal(
                    self._joint_context,
                    schema,
                    selection,
                    current,
                    self._policy.rows_per_event,
                    self._wake_role,
                )
                row["joint_context"] = detail
                if joint_packet is None:
                    row.update(
                        status="skipped", reason="no prior distinct same-context TRAIN episode"
                    )
                    return
            row.update(
                selected_ids=list(selection.sample_ids),
                retention=asdict(self.retention),
                partition_retention=self._partition_report(partitions),
                rotation_cursor=cursor,
            )
            if schema is not None:
                episodes = tuple(
                    decode_episode(features, targets, schema)
                    for features, targets in selection.training_batches()
                )
                row["selected_episodes"] = [asdict(episode) for episode in episodes]
                selection = SharedReplaySelection(
                    selection.sample_ids, tuple(episode_arrays(episode) for episode in episodes)
                )
            remaining = self._replay_budget.remaining_replay_examples
            exposures = (
                len(joint_packet[0])
                if joint_packet is not None
                else sum(len(features) for features, _ in selection._rows)
            )
            if remaining is None or remaining < exposures:
                self._replay_budget.stop_replay()
            operation_row = row
            before_operation = packet_guard
            if joint_packet is not None:
                pending = deepcopy(row)
                operation_row = deepcopy(row)
                index = row["updates"] // self._policy.every_updates - 1

                def before_operation():
                    try:
                        assert packet_guard is not None
                        packet_guard()
                    finally:
                        if self._events[index] is not row or pickle.dumps(row) != pickle.dumps(
                            pending
                        ):
                            row.clear()
                            row.update(deepcopy(pending))
                            self._events[index] = row
                            raise ValueError("pending joint replay metadata changed")

            (self._before_native if before_operation is None else before_operation)()
            operation = (
                (lambda state: self._operation(state, selection, row))
                if joint_packet is None
                else (
                    lambda state: self._operation(
                        state, selection, operation_row, joint_packet, before_operation
                    )
                )
            )
            try:
                receipt = runtime.consolidate(row["event_id"], operation)
            finally:
                if joint_packet is not None:
                    row.clear()
                    row.update(operation_row)
            row.update(
                status="complete",
                reason=None,
                receipt=asdict(receipt),
                committed_updates=1 if joint_packet is not None else len(selection.sample_ids),
            )
        except Exception as error:
            # consolidate can publish a receipt then fail its final budget check.
            for receipt in runtime.applied_consolidations:
                if receipt.event_id == row["event_id"]:
                    row.update(
                        receipt=asdict(receipt),
                        committed_updates=1
                        if self._joint_context is not None
                        else len(row["selected_ids"]),
                    )
                    break
            row.update(status="failed", reason=f"{type(error).__name__}: {error}")
            raise
        finally:
            row["charged_exposures"] = (
                self._replay_budget.replay_examples_completed - initial_charge
            )

    def after_update(
        self,
        runtime: ActorShadowRuntime[Array, Array, Array, Any],
        updates: int,
        features: Array,
        targets: Array,
        *,
        episode: DemonstrationEpisode | None = None,
        packet_episodes: tuple[DemonstrationEpisode, ...] = (),
    ) -> None:
        self._check_schema()
        if not self._bound or self._stopped:
            raise ValueError("rehearsal must be bound and open")
        if (
            type(runtime) is not ActorShadowRuntime
            or runtime._budget is not self._budget
            or runtime._training_role != self._wake_role
        ):
            raise ValueError("rehearsal requires its original wake-budget runtime")
        if self._runtime is not None and runtime is not self._runtime:
            raise ValueError("rehearsal runtime identity must remain unchanged")
        if (
            type(updates) is not int
            or updates != self._observed + 1
            or updates > self._total_updates
        ):
            raise ValueError("rehearsal requires consecutive newly completed wake counts")
        if self._budget.updates_completed != updates or len(runtime.applied_updates) != updates:
            raise ValueError("rehearsal requires the actual completed wake budget and receipts")
        if (
            type(features) is not np.ndarray
            or type(targets) is not np.ndarray
            or features.dtype != np.float64
            or targets.dtype != np.float64
        ):
            raise ValueError("rehearsal requires exact float64 training arrays")
        schema = self._check_schema()
        if schema is None or schema.num_classes == 2:
            validate_binary_training_batch(features, targets, self._input_dim)
        else:
            # Whole-packet validation below supplies finite bounded observations;
            # do not coerce categorical actions through the binary label contract.
            validate_class_targets(targets, len(features), schema.num_classes)
        if type(packet_episodes) is not tuple or (
            packet_episodes
            and (episode is not None or len(packet_episodes) > 2 or not self.packet_episode_replay)
        ):
            raise ValueError(
                "packet episodes require one or two ordinary joint TRAIN episodes alone"
            )
        if packet_episodes:
            owned = tuple(copy_episode(ep) for ep in packet_episodes)
            if len({ep.episode_id for ep in owned}) != len(owned):
                raise ValueError("packet episode IDs must be unique")
            arrays = tuple(episode_arrays(ep) for ep in owned)
            packet = (
                np.concatenate([x for x, _ in arrays]),
                np.concatenate([y for _, y in arrays]),
            )
            if not np.array_equal(features, packet[0]) or not np.array_equal(targets, packet[1]):
                raise ValueError("packet episodes must exactly match completed training arrays")
            encoded = tuple(
                self._retained_arrays(x, y, ep) for ep, (x, y) in zip(owned, arrays, strict=True)
            )
            retained_features = np.concatenate([x for x, _ in encoded])
            retained_targets = np.concatenate([y for _, y in encoded])
        else:
            retained_features, retained_targets = self._retained_arrays(features, targets, episode)
        replace(self._policy)
        _sessions(self._budget, self._replay_budget, fresh=False)
        try:
            self._runtime = runtime
            if self._fixed_bank_birth is None:
                self._buffer.observe_train_batch(retained_features, retained_targets)
            self._observed = updates
            if updates % self._policy.every_updates == 0:
                row = self._events[updates // self._policy.every_updates - 1]
                gate = self._check_incident_gate()
                if gate is not None:
                    admission = gate.admit_boundary(updates)
                    row["incident_admission"] = deepcopy(admission)
                    if not admission["admitted"]:
                        row.update(status="skipped", reason=admission["reason"])
                        return
                if self._joint_context is None:
                    self._run_event(runtime, row)
                else:
                    current_episodes = packet_episodes
                    if not current_episodes:
                        assert episode is not None
                        current_episodes = (episode,)
                        packet = episode_arrays(copy_episode(episode))
                    originals = tuple(copy_episode(ep) for ep in current_episodes)

                    def packet_guard():
                        self._before_native()
                        self._check_schema()
                        if (
                            not np.array_equal(features, packet[0])
                            or not np.array_equal(targets, packet[1])
                            or tuple(copy_episode(ep) for ep in current_episodes) != originals
                        ):
                            raise ValueError("joint replay current paid wake packet changed")

                    try:
                        self._run_event(runtime, row, packet, packet_guard)
                    finally:
                        self._joint_paid.append((row, deepcopy(row)))
        except Exception as error:
            self._stopped, self._reason = True, f"{type(error).__name__}: {error}"
            raise

    @property
    def retention(self) -> ReplayRetentionSnapshot:
        """Read detached bounded membership, including after a final non-event wake."""
        schema = self._check_schema()
        retained = self._buffer.retention
        if schema is None:
            return retained
        return replace(retained, example_count=retained.example_count * schema.steps_per_episode)

    def _retained_arrays(
        self, features: Array, targets: Array, episode: DemonstrationEpisode | None
    ) -> tuple[Array, Array]:
        schema = self._check_schema()
        if schema is None:
            if episode is not None:
                raise ValueError("episode metadata requires configured episode rehearsal")
            return features, targets
        if episode is None:
            raise ValueError("episode rehearsal requires complete training episode metadata")
        owned = copy_episode(episode)
        if (
            owned.role != self._wake_role
            or owned.num_classes != schema.num_classes
            or len(owned.transitions) != schema.steps_per_episode
            or len(owned.transitions[0].observation) != schema.observation_dim
        ):
            raise ValueError("completed episode must retain the original wake role and schema")
        expected_features, expected_targets = episode_arrays(owned)
        if not np.array_equal(features, expected_features) or not np.array_equal(
            targets, expected_targets
        ):
            raise ValueError("episode metadata must exactly match the completed training batch")
        if self._wake_role == "probe_train":
            # Why: probe episodes authorize only this disposable wake update;
            # the separate fixed bank retains original TRAIN records unchanged.
            return features, targets
        return encode_episode(owned, schema)

    def _partition_report(self, partitions: Any) -> dict[str, Any] | None:
        if partitions is None:
            return None
        report = asdict(partitions)
        schema = self._check_schema()
        if schema is not None:
            for name in ("recent", "reservoir", "unique"):
                report[name]["example_count"] *= schema.steps_per_episode
            report["overlap_examples"] *= schema.steps_per_episode
        return report

    @property
    def report(self) -> dict[str, Any]:
        schema = self._check_schema()
        report = dict(
            bound=self._bound,
            stopped=self._stopped,
            reason=self._reason,
            total_updates=self._total_updates,
            input_dim=self._input_dim,
            wake_updates_observed=self._observed,
            planned_events=len(self._events),
            policy=asdict(self._policy),
            **{
                name: sum(row[name] for row in self._events)
                for name in (
                    "charged_exposures",
                    "attempted_updates",
                    "returned_updates",
                    "committed_updates",
                )
            },
            replay_budget_charged_exposures=self._replay_budget.replay_examples_completed,
            events=self._events,
        )
        if type(self._policy) is RecentStreamReplayPolicy:
            report["control"] = "recent_fifo_v1"
        if self._joint_context is not None:
            report["joint_context"] = dict(
                policy=asdict(self._joint_context),
                scope="one additional joint fit per eligible event; charge all native matrix rows including repeated current; no probe labels enter retention",
            )
        if self._fixed_bank_birth is not None:
            birth = self._fixed_bank_birth
            report.update(
                control="fixed_train_episodes_v1",
                fixed_bank=dict(
                    episode_ids=list(birth[0]),
                    episode_count=len(birth[0]),
                    native_transition_count=birth[2],
                    encoded_array_bytes=birth[3],
                    construction_updates=0,
                ),
            )
        gate = self._check_incident_gate()
        if gate is not None:
            report["incident_admission"] = gate.report()
        if type(self._policy) is CoverageStreamReplayPolicy:
            if type(self._buffer) is not CoverageReplayBuffer:
                raise ValueError("coverage rehearsal lost its configured buffer")
            report.update(
                control="context_coverage_v1",
                coverage_keys=list(self._buffer.coverage_keys),
                coverage_cursor=self._buffer.coverage_cursor,
                selector_information="declared released training context/cue keys; stronger metadata than ordinary replay",
            )
        if schema is not None:
            report["episode_schema"] = asdict(schema)
            if schema.num_classes == 2:
                # Preserve the original two-field binary report alongside its wire bytes.
                report["episode_schema"].pop("num_classes")
            report["episode_storage_scope"] = (
                "retention counts native transitions; content IDs identify whole encoded episodes; "
                "retained_bytes includes fixed UTF-8 ID/context header, all observations, actions, "
                "rewards, next observations, terminal flags and dummy target float64 arrays; "
                "excludes Python metadata, copies, selected-episode event records and process RSS"
            )
        if self._distillation is not None:
            report["distillation"] = dict(
                policy=asdict(self._distillation),
                teacher_storage_array_bytes=self._teacher_storage_bytes,
                retained_target_array_bytes=0,
                **{
                    name: sum(row[name] for row in self._events)
                    for name in (
                        "teacher_forward_attempts",
                        "teacher_forward_returns",
                        "teacher_probability_bytes_generated",
                        "mixed_target_bytes_returned",
                    )
                },
                peak_probability_array_bytes=max(
                    (row["teacher_peak_probability_bytes"] for row in self._events), default=0
                ),
                partial_work_note="attempts charged before teacher forward; completed probability and mixed-target payload bytes reported separately; failed work may include additional unobserved computation/allocation, never zero-imputed",
                storage_scope="logical teacher weights/probability/mixed-target array payloads only; excludes metadata, copies, event records and process RSS",
            )
        if self._wake_role != "train":
            report["wake_role"] = self._wake_role
        return deepcopy(report)
