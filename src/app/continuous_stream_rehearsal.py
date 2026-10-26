"""Fixed continuous TRAIN rehearsal through existing candidate consolidation.

Complete immutable episodes supply one bounded bank batch per scheduled event.
Wake authority remains with ActorShadowRuntime; a separate original session pays
replay exposure before training. No selection, fitting, promotion or IO belongs
here. Native state validation is supplied by the caller's trusted equality port.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import isfinite
import pickle
from typing import Any, Callable

import numpy as np

from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionProgress
from src.core.actor_ports import AppliedConsolidation, ConsolidatedState
from src.core.continuous_demonstration import (
    continuous_training_pairs,
    copy_continuous_episode,
)
from src.core.experience import AppliedExperience
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed


@dataclass(frozen=True)
class ContinuousReplayPlan:
    every_updates: int
    max_replay_examples: int
    max_retained_rows: int = 128
    max_retained_bytes: int = 1048576

    def __post_init__(self) -> None:
        for name, maximum in (
            ("every_updates", 384),
            ("max_replay_examples", 65536),
            ("max_retained_rows", 128),
            ("max_retained_bytes", 1048576),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(
                    f"continuous replay {name} must be a native integer in1..{maximum}"
                )


def prepare_replay_bank(plan, episodes):
    """Own a finite complete TRAIN bank before an outer fit or model allocation.

    Numeric bytes count both observations, action and reward on every transition;
    Python containers, validation copies, native arrays and process RSS add cost.
    """
    if type(plan) is not ContinuousReplayPlan:
        raise ValueError("continuous rehearsal requires an exact ContinuousReplayPlan")
    owned_plan = replace(plan)
    if type(episodes) is not tuple or not 1 <= len(episodes) <= 4:
        raise ValueError("continuous replay bank requires an exact tuple of1..4 TRAIN episodes")
    owned = tuple(copy_continuous_episode(value) for value in episodes)
    if any(value.role != "train" for value in owned):
        raise ValueError("continuous replay bank admits TRAIN episodes only")
    if len({value.episode_id for value in owned}) != len(owned):
        raise ValueError("continuous replay bank requires distinct episode IDs")
    if any(value.spec != owned[0].spec for value in owned):
        raise ValueError("continuous replay bank episodes must share their complete action spec")
    rows = sum(len(value.transitions) for value in owned)
    numeric_bytes = 8 * sum(
        len(row.observation) + len(row.next_observation) + len(row.action) + 1
        for value in owned
        for row in value.transitions
    )
    if rows > owned_plan.max_retained_rows or numeric_bytes > owned_plan.max_retained_bytes:
        raise ValueError("whole continuous replay bank exceeds its retained row/byte capacity")
    if rows > owned_plan.max_replay_examples:
        raise ValueError("whole continuous replay bank must fit one replay exposure allowance")
    return owned_plan, owned, rows, numeric_bytes


def _signature(value):
    return pickle.dumps(value, protocol=5)


def _diagnostic(value):
    if type(value) is not TrainingDiagnostic:
        raise ValueError("continuous replay requires an exact native TrainingDiagnostic")
    owned = replace(value)
    if len(owned.definition) > 128:
        raise ValueError("continuous replay native diagnostic definition exceeds128 characters")
    return owned


def _error(error):
    return f"{type(error).__name__}: {error}"[:1024]


class _BudgetBinding:
    def __init__(self, session):
        if type(session) is not ToyBudgetSession or type(session.budget) is not ToyExecutionBudget:
            raise ValueError("continuous replay requires exact original budget sessions and limits")
        replace(session.budget)
        if session.budget.max_hidden_width is not None:
            raise ValueError("continuous replay does not support hidden-width budgets")
        if session.progress is not None and type(session.progress) is not ToyExecutionProgress:
            raise ValueError("continuous replay requires exact original progress metadata")
        self.session, self.limits, self.timer = session, session.budget, session.clock
        self.owner = session
        self.progress, self.sampler = session.progress, session.process_rss_sampler
        self.limit_values = _signature(session.budget)
        self.limit_report = asdict(session.budget)
        self.started = (type(session.started_at), session.started_at)
        self.last_clock = session.last_clock
        self.check(0, 0)

    def check(self, updates, replay):
        value = self.session
        if (
            value is not self.owner
            or type(value) is not ToyBudgetSession
            or value.budget is not self.limits
            or _signature(value.budget) != self.limit_values
            or value.clock is not self.timer
            or value.progress is not self.progress
            or value.process_rss_sampler is not self.sampler
            or (type(value.started_at), value.started_at) != self.started
            or not callable(value.clock)
        ):
            raise ValueError(
                "continuous replay budget identity, limits, timer, start or progress changed"
            )
        for clock in (value.started_at, value.last_clock):
            if type(clock) not in (int, float) or not isfinite(clock):
                raise ValueError("continuous replay requires finite original clock metadata")
        if value.last_clock < value.started_at or value.last_clock < self.last_clock:
            raise ValueError("continuous replay budget clock regressed")
        if (
            type(value.updates_completed) is not int
            or value.updates_completed != updates
            or type(value.replay_examples_completed) is not int
            or value.replay_examples_completed != replay
        ):
            raise ValueError(
                "continuous replay budget counters changed or lost their original charges"
            )
        if self.progress is not None and (
            type(self.progress.updates_completed) is not int
            or self.progress.updates_completed != updates
            or type(self.progress.replay_examples_completed) is not int
            or self.progress.replay_examples_completed != replay
        ):
            raise ValueError("continuous replay progress differs from original session counters")
        self.last_clock = value.last_clock


class ContinuousStreamRehearsal:
    def __init__(
        self,
        learner,
        wake_budget,
        plan,
        *,
        replay_budget,
        total_updates,
        fixed_training_episodes,
        states_equal: Callable[[Any, Any], bool],
        expected_definition,
        native_return_observer=None,
    ):
        self._plan, self._bank, self._rows, self._bytes = prepare_replay_bank(
            plan, fixed_training_episodes
        )
        if type(total_updates) is not int or not 1 <= total_updates <= 384:
            raise ValueError("continuous replay total_updates must be a native integer in1..384")
        if not callable(states_equal) or any(
            not callable(getattr(learner, name, None))
            for name in ("fork", "snapshot_state", "restore_state", "train_batch")
        ):
            raise ValueError(
                "continuous replay requires an owned forkable native learner/equality port"
            )
        if (
            type(expected_definition) is not str
            or not expected_definition
            or len(expected_definition) > 128
        ):
            raise ValueError("continuous replay requires a fixed native diagnostic definition")
        if wake_budget is replay_budget:
            raise ValueError("continuous replay requires distinct wake and replay budgets")
        self._wake = _BudgetBinding(wake_budget)
        self._replay = _BudgetBinding(replay_budget)
        if (
            wake_budget.budget.max_replay_examples is not None
            or replay_budget.budget.max_replay_examples != self._plan.max_replay_examples
            or (wake_budget.progress is not None and wake_budget.progress is replay_budget.progress)
        ):
            raise ValueError("continuous replay requires separate original wake/replay authority")
        if native_return_observer is not None and not callable(native_return_observer):
            raise ValueError("continuous replay native return observer must be callable")
        self._native_return_observer = self._native_return_observer_owner = native_return_observer
        self._learner, self._states_equal = learner, states_equal
        self._learner_owner, self._equality_owner = learner, states_equal
        self._definition, self._total = expected_definition, total_updates
        self._bank_birth = (
            id(self._bank),
            _signature(self._bank),
            id(self._plan),
            _signature(self._plan),
        )
        self._plan_report = asdict(self._plan)
        self._bank_ids = tuple(value.episode_id for value in self._bank)
        self._bank_contexts = tuple(value.context for value in self._bank)
        self._source = deepcopy(learner.snapshot_state())
        self._policy = _signature(self._source.policy)
        self._state_counts(self._source)
        self._expected = deepcopy(self._source)
        self._runtime: Any = None
        self._runtime_birth: Any = None
        self._logical_clock: Any = None
        self._last_tick: int | None = None
        self._observed = self._charged = 0
        self._receipts: tuple[Any, ...] = ()
        self._consolidations: tuple[Any, ...] = ()
        self._attempts: set[str] = set()
        self._attempt_owner: Any = None
        self._active_event: str | None = None
        self._event_result: Any = None
        self._bound = self._stopped = False
        self._reason: str | None = None
        self._events: list[dict[str, Any]] = [
            dict(
                event_id=f"continuous-replay-{update}",
                wake_updates=update,
                status="pending",
                attempted_updates=0,
                returned_updates=0,
                committed_updates=0,
                charged_exposures=0,
                diagnostics=[],
                receipt=None,
                error=None,
            )
            for update in range(
                self._plan.every_updates, total_updates + 1, self._plan.every_updates
            )
        ]
        self._check_source()
        self._budgets(0)

    def _same(self, expected, actual):
        if self._states_equal(expected, actual) is not True:
            raise ValueError("continuous replay native full state differs from its expected owner")

    def _state_counts(self, state):
        self._same(state, state)
        for name in ("completed_updates", "training_examples"):
            value = getattr(state, name, None)
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError("continuous replay native counters must be finite native integers")
        if hasattr(self, "_policy") and _signature(state.policy) != self._policy:
            raise ValueError("continuous replay native policy changed")
        wake = getattr(state, "wake_state", None)
        if wake is not None and (
            type(wake.wake_updates) is not int or wake.wake_updates != state.completed_updates
        ):
            raise ValueError("continuous replay CPC wake counter differs from native updates")
        return state.completed_updates, state.training_examples

    def _bank_check(self):
        if (
            self._learner is not self._learner_owner
            or self._states_equal is not self._equality_owner
            or self._native_return_observer is not self._native_return_observer_owner
        ):
            raise ValueError("continuous replay original learner or equality port changed")
        if (
            id(self._bank),
            _signature(self._bank),
            id(self._plan),
            _signature(self._plan),
        ) != self._bank_birth:
            raise ValueError("continuous replay original fixed bank or plan changed")

    def _budgets(self, updates):
        self._wake.check(updates, 0)
        self._replay.check(0, self._charged)

    def _runtime_check(self, runtime, *, observe_clock=True, check_wakes=True, admitted_tick=None):
        if type(runtime) is not ActorShadowRuntime or runtime is not self._runtime:
            raise ValueError("continuous replay requires its single original runtime")
        if (
            runtime._budget is not self._wake.session
            or runtime._training_role != "train"
            or runtime._original_training_role != "train"
            or (
                id(runtime._actor),
                id(runtime._candidate),
                runtime._base_actor_version,
                runtime._candidate_version,
                runtime._consolidation_limit,
                id(runtime._inbox),
                id(runtime._inbox._clock),
            )
            != self._runtime_birth
            or runtime._inbox._budget is not self._wake.session
            or runtime._inbox._clock is not self._logical_clock
            or runtime._inbox._learner is not runtime._candidate
        ):
            raise ValueError("continuous replay runtime authority or owned learner binding changed")
        if type(runtime._consolidations) is not list or _signature(
            tuple(runtime._consolidations)
        ) != _signature(self._consolidations):
            raise ValueError("continuous replay preceding consolidation receipts changed")
        actual_attempts = runtime._attempted_ids
        allowed = self._attempts | (
            {self._active_event} if self._active_event is not None else set()
        )
        if (
            actual_attempts is not self._attempt_owner
            or type(actual_attempts) is not set
            or (actual_attempts != self._attempts and actual_attempts != allowed)
        ):
            raise ValueError(
                "continuous replay original consolidation attempt identity/history changed"
            )
        if check_wakes and _signature(tuple(runtime._inbox._applied.values())) != _signature(
            self._receipts
        ):
            raise ValueError("continuous replay retained wake receipts changed after observation")
        if observe_clock:
            tick = self._logical_clock.now()
            expected_tick = self._last_tick if admitted_tick is None else admitted_tick
            if (
                type(tick) is not int
                or tick < 0
                or (expected_tick is not None and tick != expected_tick)
                or (self._last_tick is not None and tick < self._last_tick)
            ):
                raise ValueError(
                    "continuous replay original logical clock changed outside its admitted wake"
                )
            self._last_tick = tick

    def _check_source(self):
        self._same(self._source, self._learner.snapshot_state())
        if self._runtime is not None:
            actor = self._runtime.actor.snapshot_state()
            if actor.actor_version != self._runtime._base_actor_version:
                raise ValueError("continuous replay serving actor version changed")
            self._same(self._source, actor.state)
        self._same(self._source, self._learner.snapshot_state())

    def _before_native(self):
        self._bank_check()
        self._runtime_check(self._runtime)
        self._wake.session.before_sleep()
        self._replay.session.before_sleep()
        self._check_source()
        self._budgets(self._observed)
        self._bank_check()
        self._runtime_check(self._runtime)

    def bind(self, runtime):
        if self._bound or self._stopped:
            raise ValueError("continuous replay is single-use and already bound or stopped")
        try:
            if type(runtime) is not ActorShadowRuntime or runtime._budget is not self._wake.session:
                raise ValueError(
                    "continuous replay binding requires its exact original wake runtime"
                )
            if (
                runtime._training_role != "train"
                or runtime._original_training_role != "train"
                or runtime._consolidation_limit != len(self._events)
                or runtime.applied_updates
                or runtime.applied_consolidations
                or runtime._attempted_ids
            ):
                raise ValueError(
                    "continuous replay binding requires a fresh TRAIN runtime and exact event quota"
                )
            if (
                runtime._candidate is self._learner
                or runtime.actor._owns(self._learner)
                or runtime.actor._owns(runtime._candidate)
            ):
                raise ValueError(
                    "continuous replay binding requires independently owned actor and candidate"
                )
            self._runtime = runtime
            self._attempt_owner = runtime._attempted_ids
            self._logical_clock = runtime._inbox._clock
            self._runtime_birth = (
                id(runtime._actor),
                id(runtime._candidate),
                runtime._base_actor_version,
                runtime._candidate_version,
                runtime._consolidation_limit,
                id(runtime._inbox),
                id(runtime._inbox._clock),
            )
            self._bank_check()
            self._check_source()
            self._same(self._source, runtime.candidate_snapshot().state)
            self._budgets(0)
            self._check_source()
            self._bank_check()
            self._runtime_check(runtime)
            self._budgets(0)
            self._bound = True
            return len(self._events)
        except BaseException as error:
            self._stopped, self._reason = True, _error(error)
            raise

    def _receipt_history(self, runtime, updates):
        receipts = runtime.applied_updates
        if type(receipts) is not tuple or len(receipts) != updates:
            raise ValueError("continuous replay requires actual completed wake receipts")
        if _signature(receipts[:-1]) != _signature(self._receipts):
            raise ValueError("continuous replay preceding wake receipts changed")
        for index, receipt in enumerate(receipts, 1):
            if (
                type(receipt) is not AppliedExperience
                or type(receipt.update_number) is not int
                or receipt.update_number != index
            ):
                raise ValueError("continuous replay wake receipt order or type changed")
            diagnostic = _diagnostic(receipt.diagnostic)
            if diagnostic.definition != self._definition:
                raise ValueError("continuous replay wake diagnostic definition changed")
            if (
                receipt.actor_version != runtime._base_actor_version
                or receipt.learner_version != runtime._candidate_version
            ):
                raise ValueError("continuous replay wake receipt owner changed")
            if (
                any(
                    type(getattr(receipt, key)) is not int or getattr(receipt, key) < 0
                    for key in ("observed_at", "arrived_at", "applied_at")
                )
                or not receipt.observed_at <= receipt.arrived_at <= receipt.applied_at
            ):
                raise ValueError("continuous replay wake receipt clocks are invalid")
        return deepcopy(receipts)

    def _charge(self, row):
        remaining = self._replay.session.remaining_replay_examples
        if type(remaining) is not int or remaining < self._rows:
            self._replay.session.stop_replay()
        previous = self._charged
        try:
            self._replay.session.record_replay(self._rows)
        finally:
            observed = self._replay.session.replay_examples_completed
            if type(observed) is int and observed >= previous:
                self._charged = observed
                row["charged_exposures"] = observed - previous
        if self._charged != previous + self._rows:
            raise ValueError("continuous replay original exposure charge was not recorded exactly")
        self._budgets(self._observed)

    def _admitted_candidate(self, expected):
        # Consolidation already holds this runtime's nonreentrant lease. Read
        # the owned native port directly rather than reacquiring that lease.
        self._budgets(self._observed)
        self._bank_check()
        self._runtime_check(self._runtime)
        if (
            self._active_event is not None
            and self._active_event not in self._runtime._attempted_ids
        ):
            raise ValueError("continuous replay native operation lacks its consumed event ID")
        self._same(expected, self._runtime._candidate.snapshot_state())

    def _operation(self, state, expected, row):
        self._same(expected, state)
        self._before_native()
        learner = self._learner.fork()
        if (
            learner is self._learner
            or self._runtime._owns_candidate(learner)
            or self._runtime.actor._owns(learner)
        ):
            raise ValueError("continuous replay fork must be independently owned")
        self._before_native()
        learner.restore_state(deepcopy(expected))
        self._same(expected, learner.snapshot_state())
        self._before_native()
        pairs = [continuous_training_pairs(value, role="train") for value in self._bank]
        features = np.asarray([value for inputs, _ in pairs for value in inputs], dtype=np.float64)
        targets = np.asarray([value for _, outputs in pairs for value in outputs], dtype=np.float64)
        self._before_native()
        self._admitted_candidate(expected)
        self._charge(row)
        row["attempted_updates"] = 1
        diagnostic = learner.train_batch(features, targets)
        row["returned_updates"] = 1
        # Preserve richer adapter-owned native work before later source/clock
        # observers can rewrite a shared sink. This hook grants no model access.
        if self._native_return_observer is not None:
            self._native_return_observer()
        if type(diagnostic) is TrainingDiagnostic:
            value = diagnostic.value
            valid_value = type(value) in (int, float) and isfinite(value)
            row["diagnostics"].append(
                dict(
                    definition=diagnostic.definition[:256]
                    if type(diagnostic.definition) is str
                    else None,
                    value=value if valid_value else None,
                )
            )
            if not valid_value:
                row["returned_diagnostic_value_error"] = "nonfinite or nonnative scalar diagnostic"
        else:
            row["returned_diagnostic_type"] = type(diagnostic).__name__
        diagnostic = _diagnostic(diagnostic)
        if diagnostic.definition != self._definition:
            raise ValueError("continuous replay native diagnostic definition changed")
        self._before_native()
        self._admitted_candidate(expected)
        result = learner.snapshot_state()
        before_updates, before_rows = self._state_counts(expected)
        if self._state_counts(result) != (before_updates + 1, before_rows + self._rows):
            raise ValueError(
                "continuous replay native update/row counters differ from one bank batch"
            )
        row["native_updates_after"] = result.completed_updates
        row["native_training_examples_after"] = result.training_examples
        self._before_native()
        self._admitted_candidate(expected)
        self._event_result = deepcopy(result)
        return ConsolidatedState(deepcopy(result), diagnostic)

    def _run_event(self, runtime, expected, row):
        row["status"] = "attempted"
        try:
            self._before_native()
            remaining = self._replay.session.remaining_replay_examples
            if type(remaining) is not int or remaining < self._rows:
                self._replay.session.stop_replay()
            self._active_event = row["event_id"]
            receipt = runtime.consolidate(
                row["event_id"], lambda state: self._operation(state, expected, row)
            )
            self._observe_attempt(runtime, row)
            self._active_event = None
            self._record_receipt(receipt, row)
            self._consolidations = deepcopy(runtime.applied_consolidations)
            actual = runtime.candidate_snapshot().state
            if self._state_counts(actual) != (
                expected.completed_updates + 1,
                expected.training_examples + self._rows,
            ):
                raise ValueError(
                    "continuous replay published native counters differ from returned replay"
                )
            self._same(self._event_result, actual)
            self._expected = deepcopy(actual)
            self._before_native()
            row["status"] = "complete"
        except BaseException as error:
            row.update(status="failed", error=_error(error))
            if isinstance(error, TrainingUpdateFailed):
                row["native_failure_work"] = deepcopy(error.work)
                row["native_failure_definition"] = error.definition
            try:
                for receipt in runtime.applied_consolidations:
                    if receipt.event_id == row["event_id"]:
                        self._record_receipt(receipt, row)
            except BaseException as observation:
                row["receipt_observation_error"] = _error(observation)
            raise
        finally:
            try:
                self._observe_attempt(runtime, row)
            except BaseException as observation:
                row["attempt_observation_error"] = _error(observation)
            self._active_event = None

    def _observe_attempt(self, runtime, row):
        actual = runtime._attempted_ids
        if actual is not self._attempt_owner or type(actual) is not set:
            raise ValueError("continuous replay consumed event history was replaced")
        expected = self._attempts | {row["event_id"]}
        if actual == expected:
            self._attempts = expected
            row["consolidation_attempt_consumed"] = True
        elif actual != self._attempts:
            raise ValueError("continuous replay consumed event history changed")

    def _record_receipt(self, receipt, row):
        row["receipt_observed"] = True
        if type(receipt) is not AppliedConsolidation:
            row["publication_observation_unknown"] = True
            raise ValueError("continuous replay requires an exact consolidation receipt")
        row["receipt"] = {
            name: getattr(receipt, name)[:256] if type(getattr(receipt, name)) is str else None
            for name in ("event_id", "actor_version", "learner_version")
        }
        row["receipt"]["attempt_number"] = (
            receipt.attempt_number if type(receipt.attempt_number) is int else None
        )
        row["publication_observation_unknown"] = True
        diagnostic = _diagnostic(receipt.diagnostic)
        row["receipt"]["diagnostic"] = asdict(diagnostic)
        retained = tuple(self._runtime._consolidations)
        prior = self._consolidations
        if prior and prior[-1].event_id == row["event_id"]:
            expected_retained = prior
        else:
            expected_retained = prior + (receipt,)
        if (
            receipt.event_id != row["event_id"]
            or receipt.actor_version != self._runtime._base_actor_version
            or receipt.learner_version != self._runtime._candidate_version
            or type(receipt.attempt_number) is not int
            or receipt.attempt_number != self._events.index(row) + 1
            or not row["diagnostics"]
            or asdict(diagnostic) != row["diagnostics"][-1]
            or _signature(retained) != _signature(expected_retained)
        ):
            raise ValueError("continuous replay retained consolidation receipt changed")
        row["committed_updates"] = 1
        row["publication_observation_unknown"] = False

    def after_update(self, runtime, updates, *, expected_state=None):
        """Observe a completed wake, optionally bound to the caller's admitted state.

        Why: source/clock observers below must not redefine an already validated
        native wake before the helper first takes its candidate snapshot.
        """
        if not self._bound or self._stopped:
            raise ValueError("continuous replay requires its bound open helper")
        try:
            admitted = deepcopy(expected_state)
            self._runtime_check(runtime, observe_clock=False, check_wakes=False)
            self._bank_check()
            if type(updates) is not int or updates != self._observed + 1 or updates > self._total:
                raise ValueError(
                    "continuous replay requires consecutive newly completed wake counts"
                )
            self._budgets(updates)
            receipts = self._receipt_history(runtime, updates)
            self._runtime_check(runtime, check_wakes=False, admitted_tick=receipts[-1].applied_at)
            self._check_source()
            current = runtime.candidate_snapshot().state
            if admitted is not None:
                self._same(admitted, current)
            old_updates, old_rows = self._state_counts(self._expected)
            new_updates, new_rows = self._state_counts(current)
            if new_updates != old_updates + 1 or new_rows <= old_rows:
                raise ValueError("continuous replay requires the newly completed native wake state")
            self._check_source()
            self._same(current, runtime.candidate_snapshot().state)
            self._budgets(updates)
            self._bank_check()
            self._receipts = receipts
            self._runtime_check(runtime)
            self._observed, self._receipts = updates, receipts
            self._expected = deepcopy(current)
            if updates % self._plan.every_updates == 0:
                self._run_event(
                    runtime,
                    deepcopy(current),
                    self._events[updates // self._plan.every_updates - 1],
                )
        except BaseException as error:
            self._stopped, self._reason = True, _error(error)
            raise

    def before_update(self, runtime, updates):
        """Validate the preceding success before another wake or final release."""
        if not self._bound or self._stopped:
            raise ValueError("continuous replay requires its bound open helper")
        try:
            if type(updates) is not int or updates != self._observed:
                raise ValueError("continuous replay preflight requires its observed wake count")
            self._runtime_check(runtime)
            self._bank_check()
            self._budgets(updates)
            self._check_source()
            self._same(self._expected, runtime.candidate_snapshot().state)
            self._runtime_check(runtime)
            self._bank_check()
            self._budgets(updates)
        except BaseException as error:
            self._stopped, self._reason = True, _error(error)
            raise

    @property
    def report(self):
        result = dict(
            plan=deepcopy(self._plan_report),
            planned_events=len(self._events),
            wake_updates_observed=self._observed,
            total_updates=self._total,
            bank_episode_ids=list(self._bank_ids),
            bank_contexts=list(self._bank_contexts),
            bank_rows=self._rows,
            retained_numeric_bytes=self._bytes,
            bank_signature_bytes=len(self._bank_birth[1]),
            retained_numeric_scope="obs+next_obs+actions+reward float64 payload; containers/copies/native arrays/RSS additional",
            charged_exposures=self._charged,
            stopped=self._stopped,
            reason=self._reason,
            charge_count_unknown=False,
            last_logical_tick=self._last_tick,
            events=deepcopy(self._events),
            original_limits=dict(
                wake=deepcopy(self._wake.limit_report), replay=deepcopy(self._replay.limit_report)
            ),
        )
        for name in ("attempted_updates", "returned_updates", "committed_updates"):
            result[name] = sum(row[name] for row in self._events)
        try:
            observed = self._replay.session.replay_examples_completed
            result["observed_budget_exposures"] = observed if type(observed) is int else None
            if type(observed) is not int or observed != self._charged:
                result["observation_error"] = (
                    "current budget exposure counter differs from retained paid work"
                )
                result["charge_count_unknown"] = True
        except BaseException as error:
            result["observation_error"] = f"{type(error).__name__}: {error}"
            result["charge_count_unknown"] = True
        return deepcopy(result)
