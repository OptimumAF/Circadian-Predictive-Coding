"""Observe fixed sensor checkpoints using existing isolated probe and health tools.

Inputs are copied native checkpoints and arrived training metadata. Outputs are
four bounded, detached observation slots, including paid partial failures. An
independent template owns probe forks; this adapter has no benchmark learner,
final evaluation, promotion, tuning, replay or training-policy selection port.
"""

from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from math import isfinite
import pickle
from time import monotonic
from typing import Any, Callable

import numpy as np

from src.app.coupled_attribution import AttributionBatch, _validate_batch
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ForkableLearner
from src.core.coupled_energy import Array, feedforward_latents
from src.core.coupled_predictive_coding import CoupledSnapshot
from src.core.learnability_diagnostics import diagnose_activations
from src.core.matched_backprop import MatchedBackpropSnapshot
from src.shared.process_memory import ProcessRssSampler

CHECKPOINTS = (0, 4, 8, 12)
PROBE_CHECKPOINTS = (0, 1, 2, 4)
RSS_LIMIT = 256 * 1024**2


def describe_learnability() -> dict[str, Any]:
    """Describe the frozen reserved task without allocating data or native models."""
    return dict(
        protocol="sensor_reserved_c_learnability_v1",
        benchmark_checkpoint_updates=list(CHECKPOINTS),
        probe_checkpoint_updates=list(PROBE_CHECKPOINTS),
        reserved_task="C: x0*x1 >= 0; uniform(-1,1), sixteen float64 features",
        training_rng=17031,
        evaluation_rng=17033,
        reference_rng=17035,
        training_batches=4,
        training_rows_per_batch=8,
        evaluation_rows=32,
        reference_rows=32,
        reserved_data_array_bytes=12800,
        probe_updates=16,
        probe_training_rows=128,
        probe_score_forward_calls=16,
        probe_score_forward_rows=512,
        health_forward_calls=4,
        health_forward_rows=128,
        health_svd_calls=4,
        health_svd_shape=[32, 8],
        gain="initial BCE minus BCE after exactly four probe updates; retain every point",
        feature_health="existing dead mean<=1e-3, active complement, saturation abs>=.99; centered singular-value entropy effective rank with existing absolute cutoff",
        unavailable_diagnostics=["importance", "neuron_age"],
        isolation="independent template and disposable full-state forks; no final data or feedback to benchmark",
        budgets="four distinct four-update sessions allocated at setup, shared original deadline and RSS sampler; no renewal",
        limitations="fixed reserved C is not task generalization, retention, recovery, causal plasticity loss or circadian advantage",
    )


def build_reserved_data() -> tuple[
    tuple[AttributionBatch, ...], tuple[AttributionBatch, ...], Array
]:
    """Build fresh disjoint C training/test arrays and a label-free reference."""
    generator = np.random.default_rng(17031)
    training = []
    for index in range(4):
        x = generator.uniform(-1, 1, (8, 16))
        training.append(_reserved_batch("probe_train", f"train-{index}", x))
    evaluation = (
        _reserved_batch(
            "probe_test", "test", np.random.default_rng(17033).uniform(-1, 1, (32, 16))
        ),
    )
    reference = np.random.default_rng(17035).uniform(-1, 1, (32, 16))
    return tuple(training), evaluation, reference


def _reserved_batch(role: str, prefix: str, x: Array) -> AttributionBatch:
    return AttributionBatch(
        role,
        "C",
        tuple(f"sensor-reserved-c-{prefix}-{i}" for i in range(len(x))),
        x,
        (x[:, [0]] * x[:, [1]] >= 0).astype(np.float64),
    )


def _array_bytes(value: Any) -> int:
    if type(value) is np.ndarray:
        return value.nbytes
    if is_dataclass(value) and not isinstance(value, type):
        return sum(_array_bytes(getattr(value, field.name)) for field in fields(value))
    if type(value) in (tuple, list):
        return sum(_array_bytes(item) for item in value)
    return 0


class SensorLearnability:
    """Fixed checkpoint observer; failed paid work permanently stops later probes."""

    def __init__(
        self,
        template: ForkableLearner[Array, Array, Array, Any],
        *,
        deadline: float,
        memory: ProcessRssSampler,
        check_boundary: Callable[[], None],
    ) -> None:
        if type(deadline) not in (int, float) or not isfinite(deadline):
            raise ValueError("sensor probe deadline must be finite")
        if type(memory) is not ProcessRssSampler or not callable(check_boundary):
            raise ValueError("sensor probe requires its original sampler and boundary guard")
        if not callable(getattr(template, "fork", None)) or not callable(
            getattr(template, "snapshot_state", None)
        ):
            raise ValueError("sensor probe requires an independent forkable template")
        self._template, self._deadline = template, float(deadline)
        self._check_external = check_boundary
        self._training, self._evaluation, self._reference = build_reserved_data()
        self._template_before = self._template_signature()
        self._data_before = self._data_signature()
        self._sessions = tuple(self._session(memory) for _ in CHECKPOINTS)
        self._rows: list[dict[str, Any]] = [
            dict(updates=n, status="not_run", reason="checkpoint not reached", probe=None)
            for n in CHECKPOINTS
        ]
        self._next = 0
        self._failed = False

    def _session(self, memory: ProcessRssSampler) -> ToyBudgetSession:
        session = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=4,
                max_wall_seconds=max(0.0, self._deadline - monotonic()),
                max_process_rss_bytes=RSS_LIMIT,
            ),
            monotonic,
        )
        session.attach_memory(memory)
        return session

    def _template_signature(self) -> bytes:
        policy = getattr(self._template, "checkpoint_policy", None)
        # BP's public native snapshot contains parameters only; the adapter owns
        # its fixed learning rate separately. Include the fixed adapter settings.
        settings = tuple(
            getattr(self._template, name, None)
            for name in (
                "_learning_rate",
                "_inference_steps",
                "_inference_learning_rate",
                "_inference_policy",
            )
        )
        return pickle.dumps(
            (self._template.snapshot_state(), policy() if callable(policy) else None, settings)
        )

    def _data_signature(self) -> bytes:
        return pickle.dumps((self._training, self._evaluation, self._reference))

    def _boundary(self, state: Any, arrived: AttributionBatch | None, before: bytes) -> None:
        self._check_external()
        if monotonic() >= self._deadline:
            raise RuntimeError("original sensor learnability deadline exceeded")
        if pickle.dumps((state, arrived)) != before:
            raise RuntimeError("sensor probe changed its input checkpoint or arrived metadata")
        if self._template_signature() != self._template_before:
            raise RuntimeError("sensor probe changed its independent template or policy")
        if self._data_signature() != self._data_before:
            raise RuntimeError("sensor probe changed reserved data")

    def _validate_observation(
        self, updates: int, state: Any, arrived: AttributionBatch | None
    ) -> None:
        if self._failed:
            raise RuntimeError("sensor learnability stopped after failed observation")
        if type(updates) is not int or self._next >= 4 or updates != CHECKPOINTS[self._next]:
            raise ValueError("sensor observations require ordered once-only checkpoints 0,4,8,12")
        if updates == 0:
            if arrived is not None:
                raise ValueError("initial sensor checkpoint requires no arrived metadata")
        else:
            if type(arrived) is not AttributionBatch:
                raise ValueError("sensor checkpoints require exact arrived training metadata")
            _validate_batch(arrived, "train", set())
        if type(state) not in (MatchedBackpropSnapshot, CoupledSnapshot):
            raise ValueError("sensor observation requires a complete native BP or PC/CPC snapshot")
        parameters = state.parameters
        if (
            len(parameters.hidden_weights) != 1
            or parameters.hidden_weights[0].shape != (16, 8)
            or parameters.hidden_biases[0].shape != (1, 8)
            or parameters.output_weight.shape != (8, 1)
            or parameters.output_bias.shape != (1, 1)
        ):
            raise ValueError("sensor observation requires fixed 16→8→1 geometry")

    def observe(self, updates: int, state: Any, arrived: AttributionBatch | None) -> None:
        """Measure health and a fresh reserved probe; raise on incomplete paid work."""
        started = monotonic()  # Includes validation, guards and detachment below.
        self._validate_observation(updates, state, arrived)
        row = self._rows[self._next]
        row.update(status="failed", reason="observation incomplete")
        session = self._sessions[self._next]
        self._next += 1
        before = pickle.dumps((state, arrived))
        guard = lambda: self._boundary(state, arrived, before)
        try:
            guard()
            self._measure_health(state, row, guard)
            probe_started = monotonic()
            try:
                row["probe"] = run_fresh_task_probe(
                    self._template,
                    state,
                    self._training,
                    self._evaluation,
                    budget=session,
                    checkpoint_updates=PROBE_CHECKPOINTS,
                    check_boundary=guard,
                )
            finally:
                row["probe_invocation_seconds"] = monotonic() - probe_started
            guard()
            probe = row["probe"]
            if probe["status"] != "complete":
                raise RuntimeError(f"sensor fresh probe incomplete: {probe['reason']}")
            curve = probe["checkpoints"]
            initial = curve[0]["scores"]["C"]["bce"]
            row.update(
                status="complete",
                reason=None,
                fresh_initial_bce=initial,
                fresh_gain=initial - curve[-1]["scores"]["C"]["bce"],
                probe_source_checkpoint_and_data_unchanged=True,
            )
        except Exception as error:
            self._failed = True
            row.update(status="failed", reason=f"{type(error).__name__}: {error}")
            raise
        finally:
            row["invocation_seconds"] = monotonic() - started
            row["charged_probe_updates"] = session.updates_completed

    def _measure_health(self, state: Any, row: dict[str, Any], guard: Callable[[], None]) -> None:
        started = monotonic()
        row.update(
            checkpoint_snapshot_array_bytes=_array_bytes(state),
            importance=None,
            neuron_age=None,
            unavailable_reason="ordinary native sensor models do not store per-neuron importance/age",
            health_forward_calls_attempted=0,
            health_forward_calls_completed=0,
            health_svd_calls_attempted=0,
            health_svd_calls_completed=0,
        )
        try:
            guard()
            row["health_forward_calls_attempted"] += 1
            activations = feedforward_latents(state.parameters, self._reference)
            row["health_forward_calls_completed"] += 1
            row["diagnostic_activation_bytes"] = _array_bytes(activations)
            guard()
            row["health_svd_calls_attempted"] += 1
            health = diagnose_activations(activations)
            row["health_svd_calls_completed"] += 1
            row["feature_health"] = [asdict(item) for item in health]
            guard()
        finally:
            row["diagnostic_seconds"] = monotonic() - started

    def report(self) -> dict[str, Any]:
        """Detach bounded observations; no native predictions or updates occur here."""
        costs = {
            name: sum(row.get(name, 0) for row in self._rows)
            for name in (
                "health_forward_calls_attempted",
                "health_forward_calls_completed",
                "health_svd_calls_attempted",
                "health_svd_calls_completed",
                "invocation_seconds",
                "diagnostic_seconds",
                "probe_invocation_seconds",
            )
        }
        costs.update(
            {
                name: sum((row.get("probe") or {}).get(name, 0) for row in self._rows)
                for name in (
                    "attempted_updates",
                    "completed_updates",
                    "attempted_training_examples",
                    "completed_training_examples",
                    "score_forward_calls_attempted",
                    "score_forward_calls_completed",
                )
            }
        )
        return deepcopy(
            dict(
                status="failed" if self._failed else "complete" if self._next == 4 else "partial",
                observations=self._rows,
                reserved_data_array_bytes=12800,
                reference_array_bytes=self._reference.nbytes,
                charged_probe_updates=sum(session.updates_completed for session in self._sessions),
                probe_session_origins=[session.started_at for session in self._sessions],
                original_deadline=self._deadline,
                costs=costs,
                costs_scope="array bytes exclude temporary copies, Python/native objects and RSS; invocation includes preflight; returned probe retains failed work and attempted/returned calls",
            )
        )
