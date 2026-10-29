"""Cooperative serving priority over new candidate native updates.

The gate bounds requests and admitted attempts. It never preempts an in-flight
native call, creates background work, measures p50/p95 or claims a hard RSS cap.
All supported shared predict/serve/train calls must pass through the wrapper.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from threading import Lock
from typing import Callable, Generic, Iterator, TypeVar

from src.app.actor_shadow import ActorShadowRuntime
from src.app.categorical_replay_action import CategoricalReplayAction
from src.app.serving_promotion import PromotableActor
from src.app.payload_ownership import PayloadOwnershipBusy
from src.core.actor_ports import VersionedPrediction, ConsolidatedState
from src.core.learner_ports import validate_learning_rate_multiplier, validate_gate_setting
from src.core.adaptation_action import (
    AdaptationEnvelope,
    AdaptationPoll,
    AdaptationProgress,
    AdaptationSnapshot,
)
from src.core.native_update_origin import NativeUpdateObserver
from src.core.resource_sharing import (
    DeferReason,
    SharingLimits,
    SharingSnapshot,
    TrainingAdmission,
    TrainingPoll,
)
from src.core.serving_ports import ServingPrediction

Features = TypeVar("Features")
Targets = TypeVar("Targets")
Prediction = TypeVar("Prediction")
State = TypeVar("State")


class ServingPriorityGate:
    def __init__(self, limits: SharingLimits, *, resource_available: Callable[[], bool]) -> None:
        if type(limits) is not SharingLimits or not callable(resource_available):
            raise ValueError("sharing gate requires declared limits and callable resource probe")
        self._limits, self._resource = deepcopy(limits), resource_available
        self._serving, self._admitted = 0, 0
        self._training, self._paused = False, False
        self._checkpointing = False
        self._retention_hold: object | None = None
        self._deferrals: dict[DeferReason, int] = {}
        self._gate = Lock()

    def snapshot(self) -> SharingSnapshot:
        with self._gate:
            return SharingSnapshot(
                deepcopy(self._limits),
                self._serving,
                self._training,
                self._paused or self._retention_hold is not None,
                self._admitted,
                tuple(sorted(self._deferrals.items())),
            )

    def pause(self) -> None:
        with self._gate:
            self._paused = True

    def resume(self) -> None:
        with self._gate:
            if self._checkpointing:
                raise ValueError("cannot resume during checkpoint preparation")
            self._paused = False

    def _hold_retention(self, token: object) -> None:
        with self._gate:
            if self._retention_hold is not None and self._retention_hold is not token:
                raise PayloadOwnershipBusy("sharing retention hold belongs to another owner")
            self._retention_hold = token

    def _release_retention(self, token: object) -> None:
        with self._gate:
            if self._retention_hold is not token:
                raise ValueError("foreign retention hold")
            self._retention_hold = None

    @contextmanager
    def _checkpoint_lease(self) -> Iterator[None]:
        """Require paused quiescence; keep new serving available during restore."""
        acquired_gate = self._gate
        with acquired_gate:
            if self._checkpointing or self._training or self._serving:
                raise PayloadOwnershipBusy("sharing gate is busy during checkpoint handoff")
            if not self._paused and self._retention_hold is None:
                raise ValueError("checkpoint handoff requires paused training")
            self._checkpointing = True
        try:
            yield
        finally:
            with acquired_gate:
                self._checkpointing = False

    @contextmanager
    def serving(self) -> Iterator[None]:
        with self._gate:
            if self._serving >= self._limits.max_serving_requests:
                raise ValueError("serving request capacity exceeded; defer the request")
            self._serving += 1
        try:
            yield
        finally:
            with self._gate:
                self._serving -= 1

    def _reason(self) -> DeferReason | None:
        if self._paused or self._retention_hold is not None:
            return "paused"
        if self._serving:
            return "serving_active"
        if self._training:
            return "training_active"
        if self._admitted >= self._limits.max_admitted_updates:
            return "work_quota"
        return None

    def _deny(self, reason: DeferReason) -> TrainingAdmission:
        self._deferrals[reason] = self._deferrals.get(reason, 0) + 1
        return TrainingAdmission(False, reason)

    def _admit(self) -> TrainingAdmission:
        with self._gate:
            reason = self._reason()
            if reason is not None:
                return self._deny(reason)
        # Why: never hold the gate while an external resource probe runs. A
        # serving arrival/pause during that probe wins the second check below.
        available = self._resource()
        if type(available) is not bool:
            raise ValueError("resource probe must return an exact boolean")
        with self._gate:
            reason = self._reason()
            if reason is None and not available:
                reason = "resource_unavailable"
            if reason is not None:
                return self._deny(reason)
            self._training = True
            self._admitted += 1  # charge attempts before callbacks/native work
            return TrainingAdmission(True, None)

    @contextmanager
    def training(self) -> Iterator[TrainingAdmission]:
        decision = self._admit()
        try:
            yield decision
        finally:
            if decision.allowed:
                with self._gate:
                    self._training = False


class ResourceSharedRuntime(Generic[Features, Targets, Prediction, State]):
    def __init__(
        self,
        runtime: ActorShadowRuntime[Features, Targets, Prediction, State],
        gate: ServingPriorityGate,
        *,
        action_envelope: AdaptationEnvelope | None = None,
        replay_action: CategoricalReplayAction | None = None,
    ) -> None:
        if not isinstance(runtime, ActorShadowRuntime) or type(gate) is not ServingPriorityGate:
            raise ValueError("resource sharing requires an actor/shadow runtime and priority gate")
        self._runtime, self._sharing = runtime, gate
        if action_envelope is not None:
            if type(action_envelope) is not AdaptationEnvelope:
                raise ValueError("adaptation requires a declared finite action envelope")
            action_envelope.__post_init__()
            if runtime._training_role != "train" or runtime._original_training_role != "train":
                raise ValueError("adaptation actions require an ordinary TRAIN runtime")
        self._action_envelope = deepcopy(action_envelope)
        self._action_pin = deepcopy(action_envelope)
        self._action_authority = (runtime, gate, runtime._budget)
        self._action_limits = (deepcopy(gate.snapshot().limits), deepcopy(runtime._budget.budget))
        self._action_progress = AdaptationProgress()
        self._action_gate = Lock()
        if action_envelope is not None and any(a.sleep_updates for a in action_envelope.actions):
            self._action_sleep_quota = runtime._consolidation_limit
        # Omit these optional fields from ordinary owners/capture layouts.
        if replay_action is not None:
            if type(replay_action) is not CategoricalReplayAction or action_envelope is None:
                raise ValueError("replay action requires a finite original adaptation envelope")
            if action_envelope.max_replay_examples > replay_action.max_replay_examples:
                raise ValueError(
                    "replay envelope exceeds original replay budget/consolidation quota"
                )
            if any(
                a.replay_examples not in (0, replay_action.rows) for a in action_envelope.actions
            ):
                raise ValueError("replay amount must equal one whole fixed TRAIN bank batch")
            token = object()
            replay_action.bind(runtime, gate, token)
            self._action_replay = self._action_replay_pin = replay_action
            self._action_replay_token = token

    def predict(self, features: Features) -> VersionedPrediction[Prediction]:
        with self._sharing.serving():
            return self._runtime.actor.predict(features)

    def serve(self, features: Features, *, now: int) -> ServingPrediction[Prediction]:
        actor = self._runtime.actor
        if not isinstance(actor, PromotableActor):
            raise ValueError("cached shared serving requires a PromotableActor")
        with self._sharing.serving():
            return actor.serve(features, now=now)

    def train_ready(
        self, *, native_observer: NativeUpdateObserver[Features, Targets] | None = None
    ) -> TrainingPoll:
        if self._action_envelope is not None or self._action_pin is not None:
            raise ValueError("explicit adaptation choice required; use train_action")
        return self._train_ready(
            self._sharing.snapshot().limits.max_updates_per_poll, native_observer=native_observer
        )

    def _train_ready(
        self,
        max_updates: int,
        *,
        native_observer: NativeUpdateObserver[Features, Targets] | None = None,
        count_action: bool = False,
        learning_rate_multiplier: float = 1.0,
        gate_setting: str = "unchanged",
    ) -> TrainingPoll:
        deferred: list[DeferReason] = []

        @contextmanager
        def admission() -> Iterator[bool]:
            with self._sharing.training() as decision:
                if decision.reason is not None:
                    deferred.append(decision.reason)
                if decision.allowed and count_action:
                    self._action_progress = replace(
                        self._action_progress,
                        wake_admissions=self._action_progress.wake_admissions + 1,
                    )
                    replay = getattr(self, "_action_replay", None)
                    if replay is not None:
                        replay.before_wake_admission()
                yield decision.allowed

        updates = self._runtime.train_ready(
            max_updates=max_updates,
            before_each_update=admission,
            native_observer=native_observer,
            learning_rate_multiplier=learning_rate_multiplier,
            gate_setting=gate_setting,
        )
        return TrainingPoll(updates, deferred[-1] if deferred else None)

    def _require_action_owner(self) -> AdaptationEnvelope:
        runtime, gate, budget = self._action_authority
        if (
            runtime is not self._runtime
            or gate is not self._sharing
            or budget is not runtime._budget
            or self._action_envelope != self._action_pin
            or (gate.snapshot().limits, budget.budget) != self._action_limits
        ):
            raise ValueError("adaptation requires original envelope, runtime, gate and budget")
        if self._action_envelope is None:
            raise ValueError("no adaptation action envelope configured")
        if getattr(self, "_action_replay", None) is not getattr(self, "_action_replay_pin", None):
            raise ValueError("original categorical replay executor replaced")
        self._action_envelope.__post_init__()
        if hasattr(self, "_action_sleep_quota"):
            if self._runtime._consolidation_limit != self._action_sleep_quota:
                raise ValueError("original sleep consolidation quota replaced")
        expected = getattr(self, "_action_sleep_history", None)
        if expected is not None and self._sleep_history() != expected:
            raise ValueError("preceding sleep consolidation evidence changed")
        return self._action_envelope

    def _sleep_history(self):
        cursor = self._runtime.capture_consolidation_cursor()
        # Only consolidation provenance is pinned here. Wake/arrival revisions
        # have their original runtime checks and are not consolidation events.
        return (
            cursor.actor_version,
            cursor.learner_version,
            cursor.consolidation_limit,
            cursor.attempted_ids,
            cursor.consolidations,
        )

    def _sleep_once(self):
        before = self._sleep_history()
        self._action_sleep_history = before
        event_id = f"adaptation-sleep-{self._action_progress.sleep_admissions}"
        started = False

        def operation(state):
            nonlocal started
            started = True
            p = self._action_progress
            self._action_progress = replace(
                p,
                native_attempts=p.native_attempts + 1,
                sleep_native_attempts=p.sleep_native_attempts + 1,
            )
            prepare = getattr(self._runtime._candidate, "prepare_component_sleep")
            result = prepare(state)
            if type(result) is not ConsolidatedState:
                raise ValueError("component sleep returned invalid consolidation evidence")
            p = self._action_progress
            self._action_progress = replace(
                p,
                native_completions=p.native_completions + 1,
                sleep_native_completions=p.sleep_native_completions + 1,
            )
            return result

        try:
            return self._runtime.consolidate(event_id, operation)
        except BaseException:
            if started:
                self._action_sleep_stopped = True
            raise
        finally:
            try:
                after = self._sleep_history()
            except BaseException:
                self._action_sleep_stopped = True
                self._action_progress = replace(
                    self._action_progress,
                    sleep_publication_unknown=True,
                    unreturned_work_unknown=True,
                )
                raise
            previous, current = before[4], after[4]
            committed = (
                len(current) == len(previous) + 1
                and current[:-1] == previous
                and current[-1].event_id == event_id
            )
            allowed_ids = (set(before[3]), set(before[3]) | {event_id})
            if (
                after[:3] != before[:3]
                or set(after[3]) not in allowed_ids
                or current[: len(previous)] != previous
                or len(current) > len(previous) + 1
                or (len(current) > len(previous) and not committed)
            ):
                self._action_sleep_stopped = True
                self._action_progress = replace(
                    self._action_progress,
                    sleep_publication_unknown=True,
                    unreturned_work_unknown=True,
                )
                raise ValueError("component sleep consolidation provenance changed")
            self._action_sleep_history = after
            p = self._action_progress
            self._action_progress = replace(
                p,
                sleep_commits=p.sleep_commits + int(committed),
                sleep_publication_unknown=p.sleep_publication_unknown
                or (started and not committed),
            )

    def adaptation_snapshot(self) -> AdaptationSnapshot:
        with self._action_exclusive():
            envelope = self._require_action_owner()
            return AdaptationSnapshot(deepcopy(envelope), replace(self._action_progress))

    @contextmanager
    def _action_exclusive(self) -> Iterator[None]:
        if not self._action_gate.acquire(blocking=False):
            raise ValueError("adaptation choice is busy; nested/concurrent selection refused")
        try:
            yield
        finally:
            self._action_gate.release()

    def _wake_action(self, action, replay) -> TrainingPoll:
        def observe(stage, read_origin):
            # Never acquire payload references; this internal counter observer
            # is not controller-supplied executable code.
            p = self._action_progress
            if stage == "started":
                self._action_progress = replace(p, native_attempts=p.native_attempts + 1)
            elif stage == "completed":
                self._action_progress = replace(p, native_completions=p.native_completions + 1)
            if replay is not None:
                replay.observe_wake(stage, read_origin)

        training = self._train_ready(
            min(action.wake_updates, self._sharing.snapshot().limits.max_updates_per_poll),
            native_observer=observe,
            count_action=True,
            learning_rate_multiplier=action.learning_rate_multiplier,
            gate_setting=action.gate_setting,
        )
        if replay is not None and training.updates:
            replay.finish_wake()
        return training

    def train_action(self, action_id: str) -> AdaptationPoll:
        """Select a declared scalar ID; never accept executable controller data.

        Known choices spend one decision before refusal/work. Unknown IDs are
        invalid input. Native admissions stay paid even if budget/training fails;
        completed receipts remain owned by the original inbox. No-op retains
        pending labels and spends no sharing admission or native invocation.
        """
        with self._action_exclusive():
            envelope = self._require_action_owner()
            action = envelope.select(action_id)
            if self._action_progress.decisions_attempted >= envelope.max_decisions:
                raise ValueError("adaptation decision quota exhausted")
            self._action_progress = replace(
                self._action_progress,
                decisions_attempted=self._action_progress.decisions_attempted + 1,
            )
            try:
                replay = getattr(self, "_action_replay", None)
                replay_step = (
                    action.replay_examples > 0
                    and action.learning_rate_multiplier == 1.0
                    and action.gate_setting == "unchanged"
                    and action.sleep_updates == 0
                )
                rate_wake = (
                    action.wake_updates > 0
                    and action.replay_examples == 0
                    and action.learning_rate_multiplier != 1
                    and action.sleep_updates == 0
                )
                gate_wake = (
                    action.wake_updates > 0
                    and action.replay_examples == 0
                    and action.gate_setting != "unchanged"
                    and action.sleep_updates == 0
                )
                sleep_only = (
                    action.sleep_updates == 1
                    and action.wake_updates == 0
                    and action.replay_examples == 0
                    and action.learning_rate_multiplier == 1
                    and action.gate_setting == "unchanged"
                )
                if action.needs_extended_executor and not (
                    (replay_step and replay is not None) or rate_wake or gate_wake or sleep_only
                ):
                    raise ValueError("replay/rate/gate/sleep adaptation executor is unsupported")
                if getattr(self, "_action_sleep_stopped", False) and (
                    action.wake_updates or replay_step or sleep_only
                ):
                    raise ValueError("component sleep owner stopped; further native work refused")
                if sleep_only:
                    candidate = self._runtime._candidate
                    validator = getattr(candidate, "validate_component_sleep", None)
                    if (
                        replay is not None
                        or not callable(validator)
                        or not callable(getattr(candidate, "prepare_component_sleep", None))
                    ):
                        raise ValueError("native component sleep executor is unsupported")
                    validator()
                    if self._action_progress.sleep_admissions >= envelope.max_sleep_updates:
                        raise ValueError("adaptation cumulative sleep envelope exhausted")
                if gate_wake:
                    validate_gate_setting(action.gate_setting)
                    candidate = self._runtime._candidate
                    validator = getattr(candidate, "validate_gate_setting", None)
                    if not callable(validator) or not callable(
                        getattr(
                            candidate,
                            "train_batch_with_rate_and_gate"
                            if rate_wake
                            else "train_batch_with_gate_setting",
                            None,
                        )
                    ):
                        raise ValueError("native gate executor is unsupported")
                    validator(action.gate_setting)
                if rate_wake:
                    validate_learning_rate_multiplier(action.learning_rate_multiplier)
                    if not callable(
                        getattr(
                            self._runtime._candidate,
                            "train_batch_with_rate_and_gate"
                            if gate_wake
                            else "train_batch_with_rate_multiplier",
                            None,
                        )
                    ):
                        raise ValueError("native learning-rate executor is unsupported")
                if (
                    replay_step
                    and action.replay_examples
                    > envelope.max_replay_examples - self._action_progress.replay_examples_charged
                ):
                    raise ValueError("adaptation cumulative replay envelope exhausted")
                if replay is not None and replay.stopped and (action.wake_updates or replay_step):
                    raise ValueError("replay owner stopped; further native work refused")
                if (
                    action.wake_updates
                    > envelope.max_wake_updates - self._action_progress.wake_admissions
                ):
                    raise ValueError("adaptation cumulative wake envelope exhausted")
                # Check availability/role/budget without reading native state or
                # queued payloads, including no-update opportunities.
                with self._runtime._exclusive():
                    self._runtime._require_open()
                    if self._runtime._training_role != "train":
                        raise ValueError("adaptation requires its original TRAIN role")
                    self._runtime._budget.before_final()
                replay_receipt = None
                sleep_receipt = None
                if sleep_only:
                    with self._sharing.training() as admission:
                        training = TrainingPoll((), admission.reason)
                        if admission.allowed:
                            self._action_progress = replace(
                                self._action_progress,
                                sleep_admissions=self._action_progress.sleep_admissions + 1,
                            )
                            sleep_receipt = self._sleep_once()
                elif replay_step:
                    if replay is None:
                        raise ValueError("replay executor is unsupported")
                    with self._sharing.training() as admission:
                        training = TrainingPoll((), admission.reason)
                        if admission.allowed:
                            self._action_progress = replace(
                                self._action_progress,
                                replay_admissions=self._action_progress.replay_admissions + 1,
                            )

                            def observe_replay(stage):
                                p = self._action_progress
                                self._action_progress = replace(
                                    p,
                                    native_attempts=p.native_attempts + int(stage == "started"),
                                    native_completions=p.native_completions
                                    + int(stage == "completed"),
                                )

                            try:
                                replay_receipt = replay.execute(
                                    self._action_replay_token,
                                    f"adaptation-replay-{self._action_progress.replay_admissions}",
                                    observe_replay,
                                )
                            finally:
                                self._sync_replay_progress()
                    if replay_receipt is not None and action.wake_updates:
                        # Why: the phases are separately paid, not atomic. Prove
                        # the preceding replay before admitting fresh work; a
                        # deferred wake leaves that commit in the returned poll.
                        self._runtime.with_candidate_snapshot(
                            lambda view: replay.before_wake_admission()
                        )
                        training = self._wake_action(action, replay)
                elif action.wake_updates == 0:
                    training = TrainingPoll((), None)
                else:
                    training = self._wake_action(action, replay)
                self._require_action_owner()
                self._action_progress = replace(
                    self._action_progress,
                    decisions_returned=self._action_progress.decisions_returned + 1,
                    poll_updates_returned=self._action_progress.poll_updates_returned
                    + len(training.updates),
                )
                return AdaptationPoll(
                    action_id,
                    action.wake_updates,
                    training,
                    replace(self._action_progress),
                    action.replay_examples,
                    replay_receipt,
                    action.learning_rate_multiplier,
                    action.gate_setting,
                    action.sleep_updates,
                    sleep_receipt,
                )
            except BaseException:
                self._sync_replay_progress()
                p = self._action_progress
                self._action_progress = replace(
                    p,
                    decisions_refused=p.decisions_refused + 1,
                    unreturned_work_unknown=p.unreturned_work_unknown
                    or p.native_attempts > p.native_completions
                    or p.sleep_publication_unknown,
                )
                raise

    def _sync_replay_progress(self):
        replay = getattr(self, "_action_replay", None)
        if replay is None:
            return
        state = replay.report()
        self._action_progress = replace(
            self._action_progress,
            replay_examples_charged=state["replay_examples_charged"],
            replay_native_attempts=state["native_attempts"],
            replay_native_completions=state["native_completions"],
            replay_commits=state["committed_events"],
            replay_publication_unknown=state["publication_unknown"],
            unreturned_work_unknown=self._action_progress.unreturned_work_unknown
            or state["publication_unknown"]
            or state["native_attempts"] > state["native_completions"],
        )


from src.app.expiry_release_proof import pin_release_methods as _pin_release_methods

_EXPIRY_RELEASE_PINS = _pin_release_methods(ServingPriorityGate, ("_checkpoint_lease",))
