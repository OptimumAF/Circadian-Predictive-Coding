"""One bounded remaining wake from an observed, separately paid replay action.

No learner/data/state history, scheduling, new credit or policy fit. The existing
replay owner validates full native/source/actor/bank state; this adapter binds
returned polls, receipt identities and exact sharing/progress deltas.
"""

from copy import deepcopy
from dataclasses import asdict, replace

from src.app.coupled_attribution import _same_value
from src.core.adaptation_action import AdaptationPoll, AdaptationProgress
from src.core.reference_controller import ReferenceDecision, ReferenceObservation
from src.core.resource_sharing import TrainingPoll

RETRYABLE = ("serving_active", "paused", "resource_unavailable")


class ContextWakeContinuation:
    def __init__(self, shared, runtime, gate, check):
        self._shared = self._shared_owner = shared
        self._runtime, self._gate, self._check = runtime, gate, check
        self._anchor = shared.adaptation_snapshot().progress
        snapshot = gate.snapshot()
        if (
            type(snapshot.admitted_updates) is not int
            or snapshot.admitted_updates != 0
            or snapshot.deferrals
        ):
            raise ValueError("continuation requires original empty sharing admission history")
        self._deferrals = snapshot.deferrals
        self._issued = self._issued_owner = None
        self._returned = self._returned_pin = None
        self._pending = self._pending_owner = None
        self._last_tick = -1
        self._stopped = False
        self._armed = self._completed = self._deferred = 0

    def _current(self):
        if self._stopped or self._shared is not self._shared_owner:
            raise ValueError("original continuation stopped or shared owner replaced")
        if self._shared._runtime is not self._runtime or self._shared._sharing is not self._gate:
            raise ValueError("original continuation runtime or sharing owner replaced")
        if self._pending is not self._pending_owner or self._issued is not self._issued_owner:
            raise ValueError("original continuation metadata owner replaced")
        self._check()
        progress = self._shared.adaptation_snapshot().progress
        sharing = self._gate.snapshot()
        sharing.limits.__post_init__()
        if (
            type(sharing.admitted_updates) is not int
            or sharing.admitted_updates
            != progress.wake_admissions + progress.replay_admissions + progress.sleep_admissions
        ):
            raise ValueError("sharing admissions changed without observed original action work")
        return progress, sharing

    def _require_pending(self):
        if self._pending is None:
            return
        poll, pin, index, _ = self._pending
        receipts = self._runtime.applied_consolidations
        if (
            not _same_value(poll, pin)
            or not receipts
            or receipts[-1] is not poll.replay_receipt
            or len(self._runtime.applied_updates) != index
        ):
            raise ValueError("preceding original replay poll/receipt/FIFO prefix changed")

    def begin(self, observation, progress, select):
        try:
            current, sharing = self._current()
            if self._issued is not None:
                raise ValueError("preceding action poll was not observed")
            if (
                type(progress) is not AdaptationProgress
                or not _same_value(progress, current)
                or not _same_value(current, self._anchor)
                or not _same_value(sharing.deferrals, self._deferrals)
            ):
                raise ValueError("unobserved action or sharing counter change")
            if type(observation) is not ReferenceObservation:
                raise ValueError("continuation requires original causal observation")
            observation.__post_init__()
            if (
                observation.tick <= self._last_tick
                or observation.tick != self._runtime._inbox._clock.now()
                or observation.actor_version != self._runtime._base_actor_version
            ):
                raise ValueError("continuation tick or actor differs from original runtime")
            self._require_pending()
            if self._pending is not None:
                if observation.ready_updates < 1 or observation.remaining_wake < 1:
                    raise ValueError("original remaining wake/FIFO credit unavailable")
                decision = ReferenceDecision(
                    "wake1", "continue_original_wake_after_verified_replay"
                )
            else:
                decision = select()
            if type(decision) is not ReferenceDecision:
                raise ValueError("continuation requires an exact scalar decision")
            self._shared._action_envelope.select(decision.action_id)
            self._issued = (
                decision.action_id,
                observation.tick,
                current,
                len(self._runtime.applied_updates),
                len(self._runtime.applied_consolidations),
            )
            self._issued_owner = self._issued
            self._last_tick = observation.tick
            return decision
        except Exception:
            self._stopped = True
            raise

    def observe(self, action_id, poll):
        try:
            current, sharing = self._current()
            if self._issued is None:
                raise ValueError("poll has no original issued action")
            if poll is not self._returned or not _same_value(poll, self._returned_pin):
                raise ValueError("original returned poll replaced or changed")
            selected, tick, before, wake_count, replay_count = self._issued
            action = self._shared._action_envelope.select(selected)
            if (
                action_id != selected
                or type(poll) is not AdaptationPoll
                or poll.action_id != selected
                or type(poll.training) is not TrainingPoll
            ):
                raise ValueError("returned poll differs from original selected action")
            if not _same_value(
                (
                    poll.requested_wake_updates,
                    poll.requested_replay_examples,
                    poll.requested_learning_rate_multiplier,
                    poll.requested_gate_setting,
                    poll.requested_sleep_updates,
                ),
                (
                    action.wake_updates,
                    action.replay_examples,
                    action.learning_rate_multiplier,
                    action.gate_setting,
                    action.sleep_updates,
                ),
            ):
                raise ValueError("returned poll changed original declared effects")
            wakes, replays = self._runtime.applied_updates, self._runtime.applied_consolidations
            nw, nr = len(wakes) - wake_count, len(replays) - replay_count
            if (
                nw not in (0, 1)
                or nr not in (0, 1)
                or nw > action.wake_updates
                or nr
                and action.replay_examples != 6
            ):
                raise ValueError("unexpected native work before continuation observation")
            if nr:
                if poll.replay_receipt is not replays[-1]:
                    raise ValueError("original replay receipt replaced or unobserved")
            elif poll.replay_receipt is not None:
                raise ValueError("poll claims an unobserved replay")
            if (
                type(poll.training.updates) is not tuple
                or len(poll.training.updates) != nw
                or nw
                and poll.training.updates[0] is not wakes[-1]
            ):
                raise ValueError("poll changed original wake receipt")
            if nw:
                receipt = wakes[-1]
                if (
                    type(receipt.update_number) is not int
                    or receipt.update_number != wake_count + 1
                    or receipt.sample_id != f"batch-{wake_count}"
                    or receipt.event_id != f"label-{wake_count}"
                    or receipt.observed_at != wake_count
                    or receipt.arrived_at != wake_count + 1
                    or receipt.applied_at != tick
                ):
                    raise ValueError("wake did not complete original causal FIFO target")
            expected = replace(
                before,
                decisions_attempted=before.decisions_attempted + 1,
                decisions_returned=before.decisions_returned + 1,
                wake_admissions=before.wake_admissions + nw,
                native_attempts=before.native_attempts + nw + nr,
                native_completions=before.native_completions + nw + nr,
                poll_updates_returned=before.poll_updates_returned + nw,
                replay_admissions=before.replay_admissions + nr,
                replay_examples_charged=before.replay_examples_charged + 6 * nr,
                replay_native_attempts=before.replay_native_attempts + nr,
                replay_native_completions=before.replay_native_completions + nr,
                replay_commits=before.replay_commits + nr,
            )
            if not _same_value(current, expected) or not _same_value(poll.progress, expected):
                raise ValueError("poll/progress contains unknown or unobserved work")
            reason = poll.training.deferred_reason
            deferrals = dict(self._deferrals)
            if reason is not None:
                if nw or reason not in (*RETRYABLE, "training_active", "work_quota"):
                    raise ValueError("poll deferral conflicts with returned native work")
                deferrals[reason] = deferrals.get(reason, 0) + 1
            if not _same_value(sharing.deferrals, tuple(sorted(deferrals.items()))):
                raise ValueError("sharing deferral counter changed without this observed poll")
            if self._pending is not None:
                self._require_pending_before_completion(wake_count, nr)
                if nw:
                    self._pending = self._pending_owner = None
                    self._completed += 1
                elif reason not in RETRYABLE:
                    raise ValueError("remaining original wake was not completed or safely deferred")
                else:
                    self._deferred += 1
            elif nr and not nw:
                if reason not in RETRYABLE:
                    raise ValueError("replay succeeded without a qualified deferred wake")
                self._pending = (poll, deepcopy(poll), wake_count, tick)
                self._pending_owner = self._pending
                self._armed += 1
                self._deferred += 1
            self._anchor, self._deferrals = current, sharing.deferrals
            self._issued = self._issued_owner = None
            self._returned = self._returned_pin = None
        except Exception:
            self._stopped = True
            raise

    def execute(self, action_id):
        """Keep the exact ordinary returned poll until its observation, once."""
        try:
            progress, sharing = self._current()
            if self._issued is None or action_id != self._issued[0] or self._returned is not None:
                raise ValueError("execution differs from original issued action")
            if self._runtime._inbox._clock.now() != self._issued[1]:
                raise ValueError("original selected opportunity clock changed before execution")
            if not _same_value(progress, self._anchor) or not _same_value(
                sharing.deferrals, self._deferrals
            ):
                raise ValueError("unobserved work before original action execution")
            self._require_pending()
            poll = self._shared.train_action(action_id)
            self._returned, self._returned_pin = poll, deepcopy(poll)
            return poll
        except Exception:
            self._stopped = True
            raise

    def _require_pending_before_completion(self, wake_count, nr):
        if self._pending is None:
            raise ValueError("remaining wake requires original pending replay")
        poll, pin, index, _ = self._pending
        if (
            not _same_value(poll, pin)
            or nr
            or wake_count != index
            or self._runtime.applied_consolidations[-1] is not poll.replay_receipt
        ):
            raise ValueError("pending wake lost its original replay consent/receipt")

    def snapshot(self):
        pending = self._pending_owner
        return dict(
            mode="finish_selected_replay_wake",
            stopped=self._stopped,
            awaiting_poll=self._issued_owner is not None,
            armed=self._armed,
            completed=self._completed,
            deferred_polls=self._deferred,
            pending=None
            if pending is None
            else dict(
                selected_action=pending[1].action_id,
                selected_at=pending[3],
                sample_id=f"batch-{pending[2]}",
                replay_receipt=asdict(pending[1].replay_receipt),
            ),
        )
