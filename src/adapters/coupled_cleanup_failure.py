"""Trusted one-shot coupled erasure fault for the fixed local failure control.

The callback is installed once at lifecycle birth. It validates the exact supported
learner before counting, raises one fixed error, then delegates to the captured
normal zero-replay erasure port. It never snapshots, copies, learns, resets state
or changes the supported domain. No cleanup algorithm, recovery or generic fault
injection framework is provided. Driver/candidate terminal semantics stay in app.
"""

from dataclasses import asdict
from typing import Any

from src.adapters.coupled_candidate_handoff import _plain, _refused, _require
from src.adapters.coupled_data_lifecycle import erase_coupled_replay_payloads
from src.adapters.coupled_elapsed_retention import _make_coupled_elapsed_lifecycle
from src.adapters.coupled_learners import CoupledLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_experience import ManagedExperienceOwner
from src.app.retention_expiry import RetentionExpiryDriver
from src.core.data_erasure import ReplayPayloadErasure
from src.core.data_retention import DataRetentionPolicy


class OneShotCoupledEraseFailure:
    """One genuine boundary failure; readonly counters record actual invocations."""

    def __init__(self) -> None:
        self._learner_type = CoupledLearner
        self._delegate = erase_coupled_replay_payloads
        self._calls = self._failures = self._delegated_calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def failures(self) -> int:
        return self._failures

    @property
    def delegated_calls(self) -> int:
        return self._delegated_calls

    def __call__(self, learner: object) -> ReplayPayloadErasure:
        if type(learner) is not self._learner_type:
            raise ValueError("coupled erase failure requires an exact CoupledLearner")
        self._calls += 1
        if self._failures == 0:
            self._failures = 1
            raise RuntimeError("injected coupled replay erasure failure")
        self._delegated_calls += 1
        return self._delegate(learner)


def make_failing_coupled_elapsed_lifecycle(
    owner: ManagedExperienceOwner,
    *,
    policy: DataRetentionPolicy,
    erase_failure: OneShotCoupledEraseFailure,
) -> ManagedDataLifecycle:
    if type(erase_failure) is not OneShotCoupledEraseFailure:
        raise ValueError("failing coupled lifecycle requires the exact one-shot erasure port")
    return _make_coupled_elapsed_lifecycle(owner, policy=policy, native_erase=erase_failure)


def _poll_cleanup_failure(
    driver: RetentionExpiryDriver, erase_failure: OneShotCoupledEraseFailure
) -> dict[str, Any]:
    result = driver.poll_once()
    observed = driver.snapshot()
    _require(
        result.outcome == "failed"
        and observed.state == "failed"
        and observed.error_type == "RuntimeError"
        and observed.pending_cleanup
        and not observed.alive
        and observed.polls == observed.cleanup_attempts == 1
        and observed.purges == 0
        and erase_failure.calls == erase_failure.failures == 1,
        "original manual cleanup did not retain its actual one-shot failure",
    )
    return dict(poll=_plain(asdict(result)), driver=_plain(asdict(observed)))


def _candidate_fences(
    old: ActorShadowRuntime[Any, Any, Any, Any],
    new: ActorShadowRuntime[Any, Any, Any, Any],
) -> list[dict[str, Any]]:
    records = []
    for name, runtime, message in (
        ("old", old, "candidate owner is retired after checkpoint handoff"),
        ("new", new, "candidate is stopped after uncertain native mutation"),
    ):
        for operation, action in (
            ("train", runtime.train_ready),
            ("snapshot", runtime.candidate_snapshot),
        ):
            revision, completed = runtime._revision, runtime._budget.updates_completed
            result = _refused(action, message)
            _require(
                runtime._revision == revision and runtime._budget.updates_completed == completed,
                "failed candidate fence consumed work or changed its revision",
            )
            records.append(dict(owner=name, operation=operation, message=result, native_work=False))
    return records
