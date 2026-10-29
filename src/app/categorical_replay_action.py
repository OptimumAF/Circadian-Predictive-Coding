"""One fixed categorical TRAIN bank batch per admitted controller replay action.

Uses existing episodes, exposure budgets, native forks and consolidation. Keeps
one current expected candidate model, original source/actor snapshots and finite
metadata prefixes (1024wake receipts/32consolidations), never a model-state chain.
No policy search, new targets, promotion, probe/final access or persistence.
"""

from copy import deepcopy
from dataclasses import replace
from math import isfinite

import numpy as np

from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_attribution import _same_value
from src.app.payload_ownership import PayloadOwnershipBusy
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.actor_ports import AppliedConsolidation, ConsolidatedState
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.demonstration_episode import (
    copy_episode,
    encode_episode,
    episode_arrays,
    EpisodeReplaySchema,
)
from src.core.experience import AppliedExperience, require_identifier
from src.core.learner_ports import TrainingDiagnostic


class _BudgetPin:
    def __init__(self, session):
        if type(session) is not ToyBudgetSession:
            raise ValueError("replay requires original exact budget sessions")
        self.session, self.limits = session, session.budget
        self.values, self.clock = replace(session.budget), session.clock
        self.start, self.last = session.started_at, session.last_clock
        self.sampler, self.progress = session.process_rss_sampler, session.progress

    def check(self):
        s = self.session
        replace(s.budget)  # Revalidate exact primitive types even when values compare equal.
        if (
            s.budget is not self.limits
            or s.budget != self.values
            or s.clock is not self.clock
            or s.started_at != self.start
            or s.last_clock < self.last
            or s.process_rss_sampler is not self.sampler
            or s.progress is not self.progress
            or type(s.last_clock) not in (int, float)
            or not isfinite(s.last_clock)
            or type(s.started_at) is not type(self.start)
        ):
            raise ValueError("original replay/wake limits, clocks, start or sampler changed")
        self.last = s.last_clock


class CategoricalReplayAction:
    def __init__(
        self, runtime, learner, replay_budget, episodes, retention, *, states_equal, definition
    ):
        if (
            not isinstance(runtime, ActorShadowRuntime)
            or runtime._training_role != "train"
            or runtime._original_training_role != "train"
            or runtime.actor._payload_registry._lifecycle is not None
            or not 1 <= runtime._consolidation_limit <= 32
            or runtime._inbox._capacity > 1024
        ):
            raise ValueError(
                "replay requires bounded ordinary TRAIN owner without managed lifecycle"
            )
        if (
            type(retention) is not ReplayRetentionBudget
            or type(episodes) is not tuple
            or not 1 <= len(episodes) <= 4
        ):
            raise ValueError("replay requires original retention and1..4complete TRAIN episodes")
        limits = replace(retention)
        if (
            type(limits.max_examples) is not int
            or not 1 <= limits.max_examples <= 128
            or type(limits.max_bytes) is not int
            or not 1 <= limits.max_bytes <= 1024**2
        ):
            raise ValueError("categorical replay retention exceeds128rows/1MiB numeric bytes")
        bank = tuple(copy_episode(e) for e in episodes)
        if any(e.role != "train" for e in bank) or len({e.episode_id for e in bank}) != len(bank):
            raise ValueError("replay bank requires distinct complete TRAIN episodes")
        encoded = [
            encode_episode(
                e,
                EpisodeReplaySchema(
                    len(e.transitions[0].observation), len(e.transitions), e.num_classes
                ),
            )
            for e in bank
        ]
        numeric_bytes = sum(x.nbytes + y.nbytes for x, y in encoded)
        if len(bank) > limits.max_examples or numeric_bytes > limits.max_bytes:
            raise ValueError("whole replay bank exceeds original retained episode/byte limits")
        if not callable(states_equal) or any(
            not callable(getattr(learner, n, None))
            for n in ("fork", "snapshot_state", "restore_state", "train_batch")
        ):
            raise ValueError("replay requires complete native fork/state/equality ports")
        require_identifier(definition, "native replay diagnostic")
        if len(definition) > 128:
            raise ValueError("native diagnostic definition exceeds128characters")
        self._wake, self._replay = _BudgetPin(runtime._budget), _BudgetPin(replay_budget)
        if (
            replay_budget is runtime._budget
            or replay_budget.updates_completed != 0
            or replay_budget.replay_examples_completed != 0
            or replay_budget.budget.max_training_updates not in (None, 0)
            or replay_budget.budget.max_replay_examples is None
            or replay_budget.budget.max_hidden_width is not None
        ):
            raise ValueError("replay requires separate fresh finite exposure authority")
        self._runtime, self._learner = runtime, learner
        self._owners = (runtime.actor, runtime._candidate, runtime._inbox)
        self._states_equal = self._equality_pin = states_equal
        self._learner_pin = learner
        self._source = deepcopy(learner.snapshot_state())
        self._bank = self._bank_owner = bank
        self._bank_pin, self._retention = deepcopy(bank), limits
        self._rows, self._bytes = sum(len(e.transitions) for e in bank), numeric_bytes
        self._bank_geometry = (self._rows, self._bytes, deepcopy(limits))
        self._definition = definition
        self._charged = self._native_attempts = self._native_returns = 0
        self._stopped = self._publication_unknown = False
        self._error = None
        self._token = self._gate = None
        self._prepared = None
        with runtime._exclusive():
            runtime._require_open()
            self._actor_birth = runtime.actor.snapshot_state_if_idle()
            with runtime.actor._payload_exclusive():
                self._actor_model_ids = tuple(
                    id(m) for m in runtime.actor._payload_references().models
                )
            if id(learner) in self._actor_model_ids or learner is runtime._candidate:
                raise ValueError("replay source must be separate from actor and candidate")
            self._expected = deepcopy(runtime._candidate.snapshot_state())
            self._versions = (
                runtime._base_actor_version,
                runtime._candidate_version,
                runtime._inbox._learner_version,
            )
            self._wake_owners = runtime._inbox.applied_updates
            self._wakes = deepcopy(self._wake_owners)
            self._consolidations: tuple[AppliedConsolidation, ...] = ()
            self._consolidation_owners: tuple[AppliedConsolidation, ...] = ()
            self._attempt_owner = runtime._attempted_ids
            self._attempts: set[str] = set()
            if runtime._attempted_ids or runtime._consolidations:
                raise ValueError("bind replay before any candidate consolidation attempts")
        policy = self._source.policy
        if (
            any(
                e.num_classes != policy.num_classes
                or len(e.transitions[0].observation) != policy.feature_dim
                for e in bank
            )
            or policy.num_classes < 3
        ):
            raise ValueError("categorical bank geometry differs from native policy")
        if self._expected.policy != policy:
            raise ValueError("replay source and candidate native policies differ")
        if self.rows > self.max_replay_examples:
            raise ValueError("whole replay bank exceeds original exposure allowance")
        self._counts(self._expected)
        self._counts(self._source)
        self._counts(self._actor_birth.state)
        self._budgets(replay=True)

    @property
    def rows(self):
        return self._rows

    @property
    def max_replay_examples(self):
        maximum = self._replay.values.max_replay_examples
        if type(maximum) is not int:
            raise ValueError("replay requires an exact finite exposure limit")
        return min(maximum, self._runtime._consolidation_limit * self._rows)

    @property
    def stopped(self):
        return self._stopped

    def bind(self, runtime, gate, token):
        if self._token is not None or runtime is not self._runtime:
            raise ValueError("replay action is already bound or belongs to another owner")
        self._gate, self._token = gate, token

    def _same(self, expected, actual):
        if (
            self._states_equal is not self._equality_pin
            or self._states_equal(expected, actual) is not True
        ):
            raise ValueError("replay full native state changed without observed successful work")

    @staticmethod
    def _counts(state):
        values = (state.completed_updates, state.training_examples)
        if any(type(v) is not int or not 0 <= v < 2**63 for v in values):
            raise ValueError("native replay clocks must be bounded exact integers")
        return values

    def _budgets(self, *, replay=False):
        self._wake.check()
        self._replay.check()
        if (
            self._runtime._budget is not self._wake.session
            or any(
                type(v) is not int
                for s in (self._wake.session, self._replay.session)
                for v in (s.updates_completed, s.replay_examples_completed)
            )
            or self._wake.session.updates_completed != len(self._wakes)
            or self._wake.session.replay_examples_completed != 0
            or self._replay.session.updates_completed != 0
            or self._replay.session.replay_examples_completed != self._charged
        ):
            raise ValueError("original wake/replay counters changed without observed work")
        self._wake.session.before_final()
        if replay:
            self._replay.session.before_final()

    def _inside(self, state, *, active_event=None):
        r = self._runtime
        if self._stopped:
            raise ValueError("replay owner stopped after failed or unknown work")
        if (
            r.actor,
            r._candidate,
            r._inbox,
        ) != self._owners or self._learner is not self._learner_pin:
            raise ValueError("original replay actor/candidate/source owner replaced")
        if (
            r._base_actor_version,
            r._candidate_version,
            r._inbox._learner_version,
        ) != self._versions:
            raise ValueError("original replay native versions changed")
        if (
            self._bank is not self._bank_owner
            or self._bank != self._bank_pin
            or (self._rows, self._bytes, self._retention) != self._bank_geometry
        ):
            raise ValueError("original complete TRAIN replay bank or retention changed")
        r._require_open()
        self._budgets(replay=active_event is not None)
        allowed = self._attempts | ({active_event} if active_event is not None else set())
        if (
            r._attempted_ids is not self._attempt_owner
            or r._attempted_ids != allowed
            or not self._same_receipts(r._inbox.applied_updates, self._wake_owners, self._wakes)
            or not self._same_receipts(
                tuple(r._consolidations), self._consolidation_owners, self._consolidations
            )
        ):
            raise ValueError("original wake/replay receipt history changed without observed work")
        self._same(self._source, self._learner.snapshot_state())
        self._same(self._expected, state)

    @staticmethod
    def _same_receipts(current, owners, values):
        # Values alone accept equal replacement and Python bool/int coercion.
        # Keep only the existing bounded metadata prefix, never native-state history.
        return (
            len(current) == len(owners)
            and all(a is b for a, b in zip(current, owners))
            and _same_value(current, values)
        )

    def before_wake_admission(self):
        # Called under the existing runtime lease AFTER sharing admission, before
        # the native-start notification. Never wait on the actor's other lock.
        try:
            self._inside(self._runtime._candidate.snapshot_state())
            actor = self._runtime.actor.snapshot_state_if_idle()
            if actor.actor_version != self._actor_birth.actor_version:
                raise ValueError("original replay actor version changed")
            self._same(self._actor_birth.state, actor.state)
        except PayloadOwnershipBusy:
            raise
        except BaseException:
            self._stopped = True
            raise

    def observe_wake(self, stage, read_origin):
        try:
            self._observe_wake(stage, read_origin)
        except BaseException:
            self._stopped = True
            self._publication_unknown = self._publication_unknown or stage == "completed"
            raise

    def _observe_wake(self, stage, read_origin):
        if stage in ("uncertain", "committed_failure"):
            self._stopped = True
            self._publication_unknown = self._publication_unknown or stage == "uncertain"
            return
        if stage not in ("started", "completed"):
            return
        origin = read_origin()
        if (
            origin.learner is not self._owners[1]
            or origin.source.role != "train"
            or origin.label.role != "train"
        ):
            raise ValueError("wake observation requires original native TRAIN owner")
        if (
            origin.learner_version != self._versions[1]
            or origin.source.model_version != self._versions[0]
            or origin.label.model_version != self._versions[0]
        ):
            raise ValueError("wake observation native versions changed")
        state = origin.learner.snapshot_state()
        if stage == "started":
            self._inside(state)
            return
        receipt = origin.receipt
        if (
            type(receipt) is not AppliedExperience
            or receipt.update_number != len(self._wakes) + 1
            or origin.completed_updates != receipt.update_number
            or (receipt.actor_version, receipt.learner_version) != self._versions[:2]
        ):
            raise ValueError("wake success requires exact next committed native receipt")
        u, n = self._counts(self._expected)
        if self._counts(state) != (u + 1, n + len(origin.features)):
            raise ValueError("observed wake native clock/row delta changed")
        self._wakes = (*self._wakes, deepcopy(receipt))
        self._wake_owners = (*self._wake_owners, receipt)
        self._expected = deepcopy(state)

    def finish_wake(self):
        self._runtime.with_candidate_snapshot(lambda view: self._inside(view.state))

    def _operation(self, state, event, observer):
        self._inside(state, active_event=event)
        if self._bank is not self._bank_owner or self._bank != self._bank_pin:
            raise ValueError("original complete TRAIN replay bank changed")
        pairs = [episode_arrays(e) for e in self._bank]
        features, targets = (np.vstack([p[i] for p in pairs]) for i in (0, 1))
        learner = self._learner.fork()
        if (
            learner is self._learner
            or learner is self._owners[1]
            or id(learner) in self._actor_model_ids
        ):
            raise ValueError("native replay fork must be independently owned")
        learner.restore_state(deepcopy(state))
        self._same(state, learner.snapshot_state())
        self._inside(state, active_event=event)
        if self._replay.session.remaining_replay_examples < self.rows:
            self._replay.session.stop_replay()
        previous = self._charged
        try:
            self._replay.session.record_replay(self.rows)
        finally:
            self._charged = self._replay.session.replay_examples_completed
        if self._charged != previous + self.rows:
            raise ValueError("replay exposure charge changed")
        self._native_attempts += 1
        observer("started")
        diagnostic = learner.train_batch(features, targets)
        self._native_returns += 1
        observer("completed")
        if type(diagnostic) is not TrainingDiagnostic or diagnostic.definition != self._definition:
            raise ValueError("replay native diagnostic changed")
        diagnostic = replace(diagnostic)
        result = learner.snapshot_state()
        u, n = self._counts(state)
        if self._counts(result) != (u + 1, n + self.rows):
            raise ValueError("native replay clock/row delta differs from one bank batch")
        self._inside(state, active_event=event)
        self._prepared = deepcopy(result)
        return ConsolidatedState(deepcopy(result), diagnostic)

    def execute(self, token, event, observer):
        if (
            token is not self._token
            or self._token is None
            or self._gate is None
            or not self._gate.snapshot().training_active
        ):
            raise ValueError("replay requires its original live shared admission")
        require_identifier(event, "replay event")
        self._prepared = None
        try:
            self._runtime.with_candidate_snapshot(lambda view: self.before_wake_admission())
            self._budgets(replay=True)
            if self._replay.session.remaining_replay_examples < self.rows:
                self._replay.session.stop_replay()
            receipt = self._runtime.consolidate(
                event, lambda state: self._operation(state, event, observer)
            )
            self._accept_commit(event)
            return receipt
        except BaseException as error:
            self._error = f"{type(error).__name__}: {error}"[:512]
            try:
                self._accept_commit(event)
            except BaseException as observation:
                self._error = (
                    self._error + f"; reconcile: {type(observation).__name__}: {observation}"
                )[:512]
            finally:
                self._stopped = not (
                    isinstance(error, PayloadOwnershipBusy)
                    and event not in self._runtime._attempted_ids
                )
                self._publication_unknown = self._publication_unknown or (
                    self._runtime._stopped
                    and not any(r.event_id == event for r in self._runtime._consolidations)
                )
            raise
        finally:
            if event in self._runtime._attempted_ids:
                self._attempts.add(event)
            self._prepared = None

    def _accept_commit(self, event):
        if (
            self._runtime._attempted_ids is not self._attempt_owner
            or self._runtime._attempted_ids not in (self._attempts, self._attempts | {event})
        ):
            raise ValueError("replay event attempt history changed")
        if not self._same_receipts(self._runtime.applied_updates, self._wake_owners, self._wakes):
            raise ValueError("wake receipt history changed during replay")
        receipts = self._runtime.applied_consolidations
        if len(receipts) == len(self._consolidations):
            if not self._same_receipts(receipts, self._consolidation_owners, self._consolidations):
                raise ValueError("prior replay receipt replaced")
            return
        if (
            len(receipts) != len(self._consolidations) + 1
            or not self._same_receipts(
                receipts[:-1], self._consolidation_owners, self._consolidations
            )
            or receipts[-1].event_id != event
            or self._prepared is None
        ):
            raise ValueError("replay publication receipt history changed")
        self._consolidations = deepcopy(receipts)
        self._consolidation_owners = receipts
        try:
            state = self._runtime.candidate_snapshot().state
            self._same(self._prepared, state)
        except BaseException:
            self._publication_unknown = True
            raise
        self._expected = deepcopy(state)
        self._budgets(replay=True)

    def report(self):
        return dict(
            rows_per_event=self.rows,
            retained_episodes=len(self._bank),
            retained_numeric_bytes=self._bytes,
            replay_examples_charged=self._charged,
            native_attempts=self._native_attempts,
            native_completions=self._native_returns,
            committed_events=len(self._consolidations),
            publication_unknown=self._publication_unknown,
            stopped=self._stopped,
            error=self._error,
        )
