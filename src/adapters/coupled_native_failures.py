"""Observe real native mutation on either side of the receipt boundary.

Inputs are fixed local training fixtures; outputs retain complete stopped-state
forensics and actual observer origins. Why use an owned wrapper: the uncertain
fault occurs after genuine learning, without changing the native algorithm or
inventing a receipt. Private candidate reads are synchronous diagnostics only;
this example provides no retry, restore, promotion or recovery authority.
"""

from dataclasses import asdict
import os
from time import monotonic
from typing import Any, Callable, cast

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.adapters.coupled_checkpoint_interruption import FEATURES, FAMILIES, _plain
from src.adapters.coupled_growth_limit import Actor, Runtime, _source, _same, _feature_digest
from src.adapters.coupled_cache_transactions import _rss, _boundary
from src.adapters.structural_coupled_scaling_learner import StructuralCoupledScalingLearner
from src.app.coupled_structural_consolidation import CoupledScalingSplitConsolidator
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ForkableLearner
from src.core.coupled_energy import Array
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.native_update_origin import NativeUpdateOrigin, NativeUpdateStage
from src.core.serving_ports import ServingConfiguration
from src.shared.process_memory import ProcessRssSampler

_MODES = ("uncertain", "committed")
_NATIVE_FAULT = "injected native failure after update"
_OBSERVER_FAULT = "injected observer failure after commit"
_STOPPED = "candidate is stopped after uncertain native mutation"


class _FaultLearner:
    """Same-policy owned delegate; deliberately fail after its one genuine step."""

    def __init__(self, delegate: ForkableLearner[Array, Array, Array, Any], mode: str) -> None:
        if type(mode) is not str or mode not in _MODES:
            raise ValueError("mode must be uncertain or committed")
        self._delegate, self.mode, self.train_calls = delegate, mode, 0

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        self.train_calls += 1
        diagnostic = self._delegate.train_batch(features, targets)
        if self.mode == "uncertain":
            raise RuntimeError(_NATIVE_FAULT)
        return diagnostic

    def predict(self, features: Array) -> Array:
        return self._delegate.predict(features)

    def snapshot_state(self) -> Any:
        return self._delegate.snapshot_state()

    def restore_state(self, state: Any) -> None:
        self._delegate.restore_state(state)

    def fork(self) -> "_FaultLearner":
        delegate = self._delegate.fork()
        if delegate is self._delegate:
            raise ValueError("fault wrapper requires an independently owned delegate fork")
        return _FaultLearner(cast(ForkableLearner[Array, Array, Array, Any], delegate), self.mode)


class _PoisonTargets:
    def __init__(self) -> None:
        self.copy_attempts = 0

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        self.copy_attempts += 1
        raise AssertionError("stopped label targets must not be copied")


def describe_example() -> dict[str, Any]:
    """Describe the fixed engineering control without creating any native model."""
    return dict(
        protocol_id="coupled_native_failures_engineering_v1",
        families=list(FAMILIES),
        modes=list(_MODES),
        seed=23,
        scheme="mean_field",
        gamma0=1.5,
        geometry=dict(input_dim=2, width=2, hidden_layers=2, precisions=[1.0, 1.0]),
        features=[list(row) for row in FEATURES],
        targets=[[1.0], [0.0]],
        external_training_settings=dict(
            learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
        ),
        serving=dict(cache_ttl_ticks=10, max_cache_entries=2, metadata={"route": ["original"]}),
        runtime_limits=dict(max_experiences=1, max_consolidations=2),
        budget_limits=dict(
            max_training_updates=1,
            max_wall_seconds=60.0,
            max_process_rss_bytes=512 * 1024**2,
        ),
        native_training_calls=18,
        native_energy_evaluations=72,
        native_prediction_calls=6,
        split_kernel_calls=6,
        serving_requests=33,
        cache_misses=6,
        cache_hits=27,
        guarded_inference_calls=0,
        guard_model_measurements=0,
        owned_child_launches=0,
        limitations="Stopped native forensics are not resumable authority. RSS is boundary-sampled, not continuously enforced. No scientific benefit or recovery claim.",
    )


def _case(family: str, mode: str, sampler: ProcessRssSampler, started: float) -> dict[str, Any]:
    boundaries = [_boundary(sampler, started, "before_source")]
    features, targets = np.asarray(FEATURES, dtype=np.float64), np.array([[1.0], [0.0]])
    source = _source(family, features, targets)
    source_record = asdict(source)
    learner = _FaultLearner(
        StructuralCoupledScalingLearner(
            source, learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
        ),
        mode,
    )
    actor = Actor(
        learner,
        version="actor-0",
        configuration=ServingConfiguration(10, 2),
        feature_digest=_feature_digest,
        metadata={"route": ["original"]},
    )
    clock = LogicalClock(0)
    limits = ToyExecutionBudget(
        max_training_updates=1, max_wall_seconds=60.0, max_process_rss_bytes=512 * 1024**2
    )
    limits_record = asdict(limits)
    budget = ToyBudgetSession(limits, monotonic)
    budget.attach_memory(sampler)
    runtime = Runtime(
        learner,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=1,
        max_consolidations=2,
        actor=actor,
    )
    runtime.record_experience(
        Experience(
            "s0",
            "native-failures",
            0,
            "actor-0",
            features.copy(),
            "train",
            ExperiencePermissions(training=True),
        )
    )
    runtime.consolidate("split-1", CoupledScalingSplitConsolidator())
    clock.advance_to(2)
    runtime.record_label(
        LabelArrival("label-0", "s0", "native-failures", 2, "actor-0", targets.copy())
    )
    candidate = cast(_FaultLearner, runtime._candidate)
    actor_learner = cast(_FaultLearner, actor._slot.bundle.learner)
    events: list[dict[str, Any]] = []
    serves: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    poison = _PoisonTargets()
    callback_calls = 0

    def budget_identity() -> dict[str, int]:
        if (
            runtime._budget is not budget
            or runtime._inbox._budget is not budget
            or budget.budget is not limits
        ):
            raise AssertionError("native failure replaced original budget identity")
        _same(asdict(budget.budget), limits_record, "native failure changed original limits")
        return dict(
            original=id(budget),
            runtime=id(runtime._budget),
            inbox=id(runtime._inbox._budget),
            original_limits=id(limits),
            current_limits=id(budget.budget),
        )

    def serve(phase: str) -> None:
        result = actor.serve(features.copy(), now=2)
        if result.cache_hit != (phase != "prime"):
            raise AssertionError("native failure actor cache hit/miss differs")
        serving = asdict(actor.serving_snapshot())
        if serves:
            _same(serving, serves[0]["serving"], "native failure changed primed actor/cache")
        serves.append(dict(phase=phase, result=asdict(result), serving=serving))

    serve("prime")
    native_before = asdict(candidate.snapshot_state())
    actor_before = asdict(actor.serving_snapshot())
    _same(
        actor_before["model_state"]["raw_state"],
        source_record,
        "native failure actor differs from complete warm source",
    )
    inbox_before = asdict(runtime._inbox.capture_cursor())
    cursor_before = asdict(runtime.capture_consolidation_cursor())

    def observer(
        stage: NativeUpdateStage, read_origin: Callable[[], NativeUpdateOrigin[Array, Array]]
    ) -> None:
        origin = read_origin()
        if origin.learner is not candidate:
            raise AssertionError("observer did not expose original candidate")
        events.append(
            dict(
                stage=stage,
                native_state=asdict(candidate.snapshot_state()),
                completed_updates=origin.completed_updates,
                receipt=None if origin.receipt is None else asdict(origin.receipt),
                source=asdict(origin.source),
                label=asdict(origin.label),
                learner_version=origin.learner_version,
            )
        )
        serve("event_" + stage)
        if mode == "committed" and stage == "completed":
            raise RuntimeError(_OBSERVER_FAULT)

    expected_fault = _NATIVE_FAULT if mode == "uncertain" else _OBSERVER_FAULT
    try:
        runtime.train_ready(max_updates=1, native_observer=observer)
    except RuntimeError as error:
        if type(error) is not RuntimeError or str(error) != expected_fault:
            raise AssertionError(
                "native failure did not preserve the primary injected error"
            ) from error
        exception = dict(type=type(error).__name__, message=str(error))
    else:
        raise AssertionError("native failure unexpectedly returned a training receipt")
    serve("after_failure")
    boundaries.append(_boundary(sampler, started, "after_failure"))

    def capture() -> dict[str, Any]:
        # Trusted diagnostic only: the stopped public snapshot must remain closed.
        return dict(
            native=asdict(candidate.snapshot_state()),
            actor=asdict(actor.serving_snapshot()),
            inbox=asdict(runtime._inbox.capture_cursor()),
            cursor=asdict(runtime.capture_consolidation_cursor()),
            budget_identity=budget_identity(),
            budget_limits=asdict(budget.budget),
            train_calls=dict(candidate=candidate.train_calls, actor=actor_learner.train_calls),
            callback_calls=callback_calls,
            poison_copy_attempts=poison.copy_attempts,
        )

    def forbidden_transform(state: Any) -> Any:
        nonlocal callback_calls
        callback_calls += 1
        raise AssertionError("stopped candidate ran consolidation callback")

    operations: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("retry_train", lambda: runtime.train_ready(max_updates=1, native_observer=observer)),
        ("consolidate", lambda: runtime.consolidate("split-2", forbidden_transform)),
        ("public_snapshot", runtime.candidate_snapshot),
        (
            "record_label",
            lambda: runtime.record_label(
                LabelArrival(
                    "retry-label", "s0", "native-failures", 2, "actor-0", cast(Array, poison)
                )
            ),
        ),
    )
    for phase, operation in operations:
        before, budget_before = capture(), dict(vars(budget))
        try:
            operation()
        except ValueError as error:
            if str(error) != _STOPPED:
                raise AssertionError(
                    "stopped operation refused for an unexpected reason"
                ) from error
            refusals.append(dict(phase=phase, message=str(error)))
        else:
            raise AssertionError("stopped native operation was admitted")
        _same(capture(), before, "stopped refusal changed complete native/owner state")
        if vars(budget) != budget_before:
            raise AssertionError("stopped refusal changed budget counters/clock/telemetry")
    serve("after_refusals")
    final = capture()
    _validate_case(mode, native_before, final, events, budget.updates_completed)
    _same(asdict(source), source_record, "native failure changed source")
    boundaries.append(_boundary(sampler, started, "case_end"))
    return dict(
        family=family,
        mode=mode,
        source_state=source_record,
        native_before=native_before,
        native_after=final["native"],
        actor_before=actor_before,
        actor_after=final["actor"],
        inbox_before=inbox_before,
        inbox_after=final["inbox"],
        cursor_before=cursor_before,
        cursor_after=final["cursor"],
        budget_limits=limits_record,
        budget_identity=budget_identity(),
        updates_completed=budget.updates_completed,
        applied_updates=final["inbox"]["applied"],
        events=events,
        serves=serves,
        refusals=refusals,
        exception=exception,
        train_calls=final["train_calls"],
        consolidation_callback_calls=callback_calls,
        poison_copy_attempts=poison.copy_attempts,
        rss_boundaries=boundaries,
    )


def _validate_case(
    mode: str,
    before: dict[str, Any],
    final: dict[str, Any],
    events: list[dict[str, Any]],
    completed: int,
) -> None:
    expected = 0 if mode == "uncertain" else 1
    stages = (
        ["started", "uncertain"]
        if mode == "uncertain"
        else ["started", "completed", "committed_failure"]
    )
    cursor, inbox, native = final["cursor"], final["inbox"], final["native"]
    if (
        [event["stage"] for event in events] != stages
        or completed != expected
        or len(inbox["applied"]) != expected
    ):
        raise AssertionError("native failure observer/accounting boundary differs")
    _same(events[0]["native_state"], before, "started observer lost original native state")
    for event in events:
        event_completed = 0 if event["stage"] == "started" else expected
        if (
            event["completed_updates"] != event_completed
            or event["learner_version"] != "candidate-0"
        ):
            raise AssertionError("native observer did not retain actual completion boundary")
        _same(event["source"], inbox["experiences"][0], "native observer lost original experience")
        _same(event["label"], inbox["labels"][0], "native observer lost original label")
        _same(
            event["receipt"],
            None if event_completed == 0 else inbox["applied"][0],
            "native observer receipt differs from actual inbox",
        )
        if event["stage"] != "started":
            _same(event["native_state"], native, "terminal observer lost physical native update")
    if (
        cursor["revision"] != 4
        or cursor["attempted_ids"] != ("split-1",)
        or len(cursor["consolidations"]) != 1
        or cursor["consolidation_limit"] != 2
        or not cursor["stopped"]
        or cursor["retired"]
        or not cursor["payload_ready"]
        or not inbox["stopped"]
        or inbox["completed_updates"] != expected
        or len(inbox["experiences"]) != 1
        or len(inbox["labels"]) != 1
        or final["train_calls"] != dict(candidate=1, actor=0)
        or final["callback_calls"]
        or final["poison_copy_attempts"]
        or native["origin_config"]["width"] != 2
        or native["raw_state"]["config"]["width"] != 4
        or len(native["transitions"]) != 1
    ):
        raise AssertionError("native failure changed original stopped ownership/history")
    _same(native["origin_config"], before["origin_config"], "native failure lost full origin")
    _same(native["transitions"], before["transitions"], "native failure lost split history")
    _same(
        native["raw_state"]["config"],
        before["raw_state"]["config"],
        "native failure changed raw config",
    )
    wake = native["raw_state"]["wake_state"]
    if wake is not None and (
        wake["wake_updates"] != 3 or before["raw_state"]["wake_state"]["wake_updates"] != 2
    ):
        raise AssertionError("native failure did not retain genuine CPC update")
    if expected:
        receipt = inbox["applied"][0]
        if any(
            receipt[key] != value
            for key, value in dict(
                actor_version="actor-0",
                learner_version="candidate-0",
                observed_at=0,
                arrived_at=2,
                applied_at=2,
                update_number=1,
            ).items()
        ):
            raise AssertionError("committed observer failure lost original receipt")


def run_example() -> dict[str, Any]:
    started = monotonic()
    sampler = ProcessRssSampler(read_rss_bytes=_rss)
    _boundary(sampler, started, "invocation_start")
    cases = [_case(family, mode, sampler, started) for family in FAMILIES for mode in _MODES]
    for index in range(0, len(cases), 2):
        _same(
            cases[index]["native_after"],
            cases[index + 1]["native_after"],
            "failure accounting mode changed genuine native result",
        )
    for mode_index in (0, 1):
        for key in ("source_state", "native_before", "native_after"):
            left, right = cases[mode_index][key], cases[mode_index + 2][key]
            if key != "source_state":
                left, right = left["raw_state"], right["raw_state"]
            _same(
                left["raw_parameters"],
                right["raw_parameters"],
                "PC/neutral native failure raw parameters differ",
            )
    end = _boundary(sampler, started, "invocation_end")
    return _plain(
        dict(
            protocol_id="coupled_native_failures_engineering_v1",
            description=describe_example(),
            cases=cases,
            elapsed_seconds=end["elapsed_seconds"],
            rss=end["rss"],
        )
    )
