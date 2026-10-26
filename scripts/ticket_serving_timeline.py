"""Finite real typed requests alongside the existing native schedule.

Owns one service worker and bounded request/latency records, not a training
runner. Live outcomes never become labels, policy feedback or promotion guards.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from math import ceil, isfinite
from threading import Event
from time import monotonic

from src.app.ticket_workflow import run_workflow_episode
from src.core.ticket_workflow import ACTION_IDS, WorkflowCapabilities, WorkflowState

CONCEPTS = ("invoice", "legacy", "parcel", "password")


@dataclass(frozen=True)
class FollowupTrainingPlan:
    """Two predeclared TRAIN windows on one lifetime owner; no outcome selection."""

    first_batches: int = 8
    start_tick: int = 24

    def __post_init__(self):
        if type(self.first_batches) is not int or not 1 <= self.first_batches <= 8:
            raise ValueError("followup first window requires1..8batches")
        if type(self.start_tick) is not int or not self.first_batches + 2 <= self.start_tick < 64:
            raise ValueError("followup start must be a fixed later tick below64")

    def index_at(self, tick, count):
        if 0 <= tick < self.first_batches:
            return tick
        index = self.first_batches + tick - self.start_tick
        return index if self.start_tick <= tick and index < count else None

    def label_index_at(self, tick, count, first_delay, second_delay):
        """Each window retains its own prospectively declared label clock."""
        self.__post_init__()
        if (
            type(tick) is not int
            or tick < 0
            or type(count) is not int
            or not self.first_batches < count <= self.first_batches + 8
            or any(type(d) is not int or d not in (1, 2) for d in (first_delay, second_delay))
        ):
            raise ValueError("followup label clocks require exact bounded counts and delays")
        first = self.index_at(tick - first_delay, count)
        if first is not None and first < self.first_batches:
            return first
        second = self.index_at(tick - second_delay, count)
        return second if second is not None and second >= self.first_batches else None


@dataclass(frozen=True)
class LiveServingPlan:
    opportunities: int = 24
    contention_ticks: tuple[int, ...] = (2, 6)
    wait_seconds: float = 10.0
    context_tasks: tuple[tuple[str, str], ...] | None = None

    def __post_init__(self):
        if type(self.opportunities) is not int or not 1 <= self.opportunities <= 64:
            raise ValueError("live serving requires1..64opportunities")
        if (
            type(self.contention_ticks) is not tuple
            or any(
                type(t) is not int or not 0 <= t < self.opportunities for t in self.contention_ticks
            )
            or tuple(sorted(set(self.contention_ticks))) != self.contention_ticks
        ):
            raise ValueError("contention ticks must be distinct sorted original opportunities")
        try:
            valid_wait = (
                type(self.wait_seconds) in (int, float)
                and isfinite(self.wait_seconds)
                and 0 < self.wait_seconds <= 10
            )
        except OverflowError:
            valid_wait = False
        if not valid_wait:
            raise ValueError("live worker wait must be finite in(0,10]seconds")
        if self.context_tasks is not None and (
            type(self.context_tasks) is not tuple
            or len(self.context_tasks) != self.opportunities
            or any(
                type(task) is not tuple
                or len(task) != 2
                or type(task[0]) is not str
                or task[0] not in ("v1", "v2")
                or type(task[1]) is not str
                or task[1] not in CONCEPTS
                for task in self.context_tasks
            )
        ):
            raise ValueError(
                "public context tasks require one immutable known version/concept per opportunity"
            )


def serving_opportunity(plan, tick):
    """Pure execution/preview mapping; fixtures are explicit finite tick lists."""
    if type(plan) is not LiveServingPlan:
        raise ValueError("serving preview requires exact original plan")
    plan.__post_init__()
    if type(tick) is not int or not 0 <= tick < plan.opportunities:
        raise ValueError("serving preview tick outside original plan")
    version, concept = (
        ("v2", CONCEPTS[tick % len(CONCEPTS)])
        if plan.context_tasks is None
        else plan.context_tasks[tick]
    )
    scenario = (
        "stale_once"
        if tick in (4, 12, 20)
        else "revoke_commit"
        if tick in (6, 14, 22)
        else "ordinary"
    )
    return dict(
        tick=tick,
        policy_version=version,
        concept=concept,
        scenario=scenario,
        public_context=plan.context_tasks is not None,
        contention=tick in plan.contention_ticks,
    )


def serving_preview(plan):
    """Owned finite declarations only: no workers, learner/data handles or IO."""
    if type(plan) is not LiveServingPlan:
        raise ValueError("serving preview requires exact original plan")
    plan.__post_init__()
    return tuple(serving_opportunity(plan, tick) for tick in range(plan.opportunities))


class TicketServingTimeline:
    def __init__(self, plan, gate, predictor, require_source):
        if type(plan) is not LiveServingPlan:
            raise ValueError("live serving requires an exact finite plan")
        plan.__post_init__()
        self.plan, self.gate, self.predictor = plan, gate, predictor
        self._owners = (plan, gate, predictor)
        self._plan_pin = deepcopy(plan)
        self._require_source = require_source
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ticket-serving")
        self._future = self._release = None
        self._tick = None
        self._stopped = self._closed = self._unknown = False
        self.records = []

    def _require_original(self):
        if (
            any(a is not b for a, b in zip((self.plan, self.gate, self.predictor), self._owners))
            or self.plan != self._plan_pin
        ):
            raise ValueError("original live plan/gate/predictor changed")
        self.plan.__post_init__()

    def _episode(self, row, ready, release):
        try:
            self._require_original()
            self._require_source()  # No shared ToyBudgetSession mutation in the worker.
            with self.gate.serving():
                row["lease_acquired_at"] = monotonic()
                row["training_active_at_lease"] = self.gate.snapshot().training_active
                ready.set()
                if release is not None and not release.wait(self.plan.wait_seconds):
                    raise TimeoutError("reserved serving request was not released")

                def predict(request):
                    self._require_original()
                    self._require_source()
                    paid = dict(
                        complete=False,
                        started_at=monotonic(),
                        request_id=request.request_id,
                        training_active_at_start=self.gate.snapshot().training_active,
                    )
                    row["requests"].append(paid)
                    try:
                        result = self.predictor.serve(request)
                        paid.update(
                            complete=True,
                            generation=result.generation,
                            actor_version=result.actor_version,
                            native_updates=result.native_updates,
                        )
                        return result.result
                    finally:
                        paid["seconds"] = monotonic() - paid["started_at"]

                task = serving_opportunity(self.plan, row["tick"])
                version, concept, scenario = (
                    task["policy_version"],
                    task["concept"],
                    task["scenario"],
                )
                contextual = task["public_context"]
                if contextual:
                    row.update(policy_version=version, public_context=True)
                row.update(concept=concept, scenario=scenario)
                row["episode"] = run_workflow_episode(
                    WorkflowState(f"live-{row['tick']}", concept, policy_version=version),
                    predict,
                    capabilities=WorkflowCapabilities(ACTION_IDS),
                    scenario=scenario,
                    max_steps=6,
                    **({"include_policy_context": True} if contextual else {}),
                )
                row["complete"] = row["episode"]["complete"]
                if not row["complete"]:
                    row["error"] = row["episode"].get("error", "incomplete live workflow")
        except Exception as error:
            row.update(complete=False, error=f"{type(error).__name__}: {error}"[:512])
        finally:
            row["finished_at"] = monotonic()
            row["elapsed_seconds"] = row["finished_at"] - row["queued_at"]
            ready.set()
        return row

    def _start(self, *, reserved):
        row = dict(
            tick=self._tick,
            role="live_service",
            reserved_before_admission=reserved,
            complete=False,
            requests=[],
            queued_at=monotonic(),
        )
        self.records.append(row)  # Preserve paid prefixes even if the worker fails.
        ready = Event()
        self._release = Event() if reserved else None
        self._future = self._pool.submit(self._episode, row, ready, self._release)
        if not ready.wait(self.plan.wait_seconds):
            self._unknown = self._stopped = True
            raise TimeoutError("live request admission was not observed")
        if row.get("error"):
            raise RuntimeError(row["error"])

    def native_started(self):
        """Called by original counted native port after shared TRAIN admission."""
        if self._stopped or self._closed or self._tick is None:
            raise ValueError("native live request requires its active original opportunity")
        if self._future is None:
            self._start(reserved=False)

    def _finish(self):
        if self._release is not None:
            self._release.set()
        if self._future is not None:
            try:
                row = self._future.result(timeout=self.plan.wait_seconds)
            except TimeoutError:
                self._unknown = self._stopped = True
                raise
            if not row["complete"]:
                self._stopped = True
                raise RuntimeError(row.get("error", "incomplete live serving"))

    @contextmanager
    def opportunity(self, tick):
        self._require_original()
        if self._stopped or self._closed or self._tick is not None or tick != len(self.records):
            raise ValueError("live opportunity is stopped, nested or out of sequence")
        if type(tick) is not int or not 0 <= tick < self.plan.opportunities:
            raise ValueError("live opportunity exceeds original finite plan")
        self._tick, self._future, self._release = tick, None, None
        try:
            if tick in self.plan.contention_ticks:
                self._start(reserved=True)
            try:
                yield
            except BaseException:
                self._stopped = True
                try:
                    self._finish()
                except Exception as error:
                    if self.records:
                        self.records[-1]["join_error"] = str(error)[:512]
                raise
            else:
                if self._future is None:
                    self._start(reserved=False)
                self._finish()
        except BaseException:
            self._stopped = True
            if self._release is not None:
                self._release.set()
            raise
        finally:
            self._tick = None

    def close(self):
        if not self._closed:
            self._pool.shutdown(wait=not self._unknown, cancel_futures=True)
            self._closed = True

    def snapshot(self):
        latencies = sorted(
            p["seconds"] for r in self.records for p in r["requests"] if "seconds" in p
        )
        return dict(
            records=deepcopy(self.records),
            closed=self._closed,
            stopped=self._stopped,
            unreturned_work_unknown=self._unknown
            or any(not p["complete"] for r in self.records for p in r["requests"]),
            latency=dict(
                definition="nearest_rank_observed_request_seconds",
                scope="predictor calls only; episode elapsed includes queue/reservation/workflow",
                attempts=len(latencies),
                p50=None if not latencies else latencies[ceil(0.5 * len(latencies)) - 1],
                p95=None if not latencies else latencies[ceil(0.95 * len(latencies)) - 1],
                maximum=None if not latencies else latencies[-1],
            ),
        )
