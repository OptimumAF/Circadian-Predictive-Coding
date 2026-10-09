"""Bounded actual shared actor requests during observed native candidate calls.

The native facade times only the original train_batch body, excluding event/wait/
join instrumentation. Request timing covers the real shared prediction path.
Inputs are an exclusively owned shared runtime, fresh observer and finite limits.
Outputs are complete raw timing/poll records or an explicit incomplete exception.
No model configuration, IO, scientific metric or automatic retries live here.
"""

from dataclasses import dataclass
from math import isfinite
from threading import Event, Lock, Thread
from time import perf_counter_ns
from typing import Callable, Generic, TypeVar

from src.app.resource_sharing import ResourceSharedRuntime
from src.core.experience import require_tick
from src.core.learner_ports import NativeLearner, TrainingDiagnostic
from src.core.resource_sharing import TrainingPoll
from src.core.serving_latency import NativeCallTiming, ServingRequestTiming

Features = TypeVar("Features")
Targets = TypeVar("Targets")
Prediction = TypeVar("Prediction")
State = TypeVar("State")
Result = TypeVar("Result")


@dataclass(frozen=True)
class MeasurementLimits:
    idle_requests: int
    training_blocks: int
    requests_per_block: int
    worker_timeout_seconds: float
    max_wall_seconds: float

    def __post_init__(self):
        for value in (self.idle_requests, self.training_blocks, self.requests_per_block):
            require_tick(value, "measurement request/block count")
            if value == 0:
                raise ValueError("measurement counts must be positive")
        for seconds in (self.worker_timeout_seconds, self.max_wall_seconds):
            if type(seconds) not in (int, float) or not isfinite(seconds) or seconds <= 0:
                raise ValueError("measurement time limits must be positive finite seconds")


@dataclass(frozen=True)
class ServingMeasurement:
    requests: tuple[ServingRequestTiming, ...]
    native_calls: tuple[NativeCallTiming, ...]
    polls: tuple[TrainingPoll, ...]


class MeasurementIncomplete(RuntimeError):
    def __init__(self, reason, *, requests, observer, polls, live_worker):
        super().__init__(reason)
        self.requests, self.native_calls, self.polls = tuple(requests), observer.calls, tuple(polls)
        self.live_worker, self.active_native = live_worker, observer.active_native


class NativeTimingObserver:
    def __init__(self, *, clock: Callable[[], int] = perf_counter_ns):
        if not callable(clock):
            raise ValueError("native timing requires a monotonic nanosecond clock")
        self.clock, self.started = clock, Event()
        self._gate, self._block = Lock(), 0
        self._calls: list[NativeCallTiming] = []
        self._active: tuple[int, int | None] | None = None

    def now(self) -> int:
        tick = self.clock()
        require_tick(tick, "measurement clock")
        return tick

    @property
    def calls(self) -> tuple[NativeCallTiming, ...]:
        with self._gate:
            return tuple(self._calls)

    @property
    def active_native(self):
        with self._gate:
            return self._active

    def begin_block(self, block):
        require_tick(block, "measurement block")
        with self._gate:
            if self._active is not None:
                raise ValueError("previous owned native worker remains active")
            self._block = block
            self.started.clear()

    def observe(self, operation: Callable[[], Result]) -> Result:
        # Signal outside the interval; no synchronization wait enters native timing.
        self.started.set()
        with self._gate:
            if self._active is not None:
                raise ValueError("concurrent native observation is unsupported")
            self._active = (self._block, None)
        completed = False
        start = self.now()
        self._active = (self._block, start)
        try:
            result = operation()
            completed = True
            return result
        finally:
            end = self.now()
            with self._gate:
                self._calls.append(NativeCallTiming(self._block, start, end, completed))
                self._active = None


class ObservedLearner(Generic[Features, Targets, Prediction, State]):
    """Compose an owned native learner without changing its equations or policy."""

    def __init__(
        self,
        learner: NativeLearner[Features, Targets, Prediction, State],
        observer: NativeTimingObserver,
    ):
        self._learner, self._observer = learner, observer

    def fork(self) -> NativeLearner[Features, Targets, Prediction, State]:
        fork = getattr(self._learner, "fork", None)
        if not callable(fork):
            raise ValueError("observed actor/candidate source requires an owned native fork")
        owned = fork()
        if owned is self._learner:
            raise ValueError("observed native fork must own a distinct learner")
        return ObservedLearner(owned, self._observer)

    def train_batch(self, features: Features, targets: Targets) -> TrainingDiagnostic:
        return self._observer.observe(lambda: self._learner.train_batch(features, targets))

    def predict(self, features: Features) -> Prediction:
        return self._learner.predict(features)

    def snapshot_state(self) -> State:
        return self._learner.snapshot_state()

    def restore_state(self, state: State) -> None:
        self._learner.restore_state(state)


def _request(shared, features, observer, requests, phase, block, index):
    completed = False
    version = shared._runtime.actor.version
    start = observer.now()
    try:
        prediction = shared.predict(features)
        completed, version = True, prediction.actor_version
    finally:
        end = observer.now()
        requests.append(ServingRequestTiming(phase, block, index, start, end, version, completed))


def measure_shared_serving(
    shared: ResourceSharedRuntime[Features, Targets, Prediction, State],
    features: Features,
    observer: NativeTimingObserver,
    limits: MeasurementLimits,
) -> ServingMeasurement:
    if (
        type(shared) is not ResourceSharedRuntime
        or type(observer) is not NativeTimingObserver
        or type(limits) is not MeasurementLimits
    ):
        raise ValueError("measurement requires actual shared runtime, observer and finite limits")
    if observer.calls or observer.active_native is not None:
        raise ValueError("measurement requires a fresh native observer; no work reset/retry")
    requests: list[ServingRequestTiming] = []
    polls: list[TrainingPoll] = []
    worker = None
    started = observer.now()

    def check_wall():
        if observer.now() - started >= limits.max_wall_seconds * 1e9:
            raise TimeoutError("measurement wall limit reached")

    try:
        for index in range(limits.idle_requests):
            check_wall()
            _request(shared, features, observer, requests, "idle", None, index)
        for block in range(limits.training_blocks):
            check_wall()
            observer.begin_block(block)
            done = Event()
            errors: list[BaseException] = []
            block_polls: list[TrainingPoll] = []

            def train():
                try:
                    block_polls.append(shared.train_ready())
                except BaseException as error:
                    errors.append(error)
                finally:
                    done.set()

            worker = Thread(target=train, name=f"serving-measurement-{block}", daemon=True)
            worker.start()
            wait_start = observer.now()
            while not observer.started.wait(0.01):
                check_wall()
                if done.is_set():
                    if errors:
                        raise errors[0]
                    raise ValueError("training poll did not enter an actual native call")
                if observer.now() - wait_start >= limits.worker_timeout_seconds * 1e9:
                    raise TimeoutError("owned native worker did not start before timeout")
            for index in range(limits.requests_per_block):
                check_wall()
                _request(shared, features, observer, requests, "shared", block, index)
            worker.join(timeout=limits.worker_timeout_seconds)
            if worker.is_alive():
                raise TimeoutError("owned native worker remains live after join timeout")
            polls.extend(block_polls)
            if errors:
                raise errors[0]
            if len(block_polls) != 1 or len(block_polls[0].updates) != 1:
                raise ValueError("each fixed block requires exactly one arrived native update")
        check_wall()
        return ServingMeasurement(tuple(requests), observer.calls, tuple(polls))
    except BaseException as error:
        raise MeasurementIncomplete(
            str(error),
            requests=requests,
            observer=observer,
            polls=polls,
            live_worker=worker is not None and worker.is_alive(),
        ) from error
