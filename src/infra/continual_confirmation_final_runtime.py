"""Observe real final source fields and model calls outside producer accounting.

Inputs are held training and an outer budget-check port. Outputs bind the
supplied inventory's source/release/prediction events and exact outcomes.
This observer never authorizes final release, proves training/provenance or
owns resource policy. The full fixed app/worker gates remain mandatory.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
from functools import wraps
from typing import Any, Callable, Iterator
from unittest.mock import patch

import numpy as np

from src.app.continual_confirmation_execution import json_value
from src.app.continual_confirmation_final_observation import _expected_observation
from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_scoring import EndpointEvaluation, ScoredConfirmation
from src.app.continual_confirmation_state import HeldSeed
from src.app.continual_confirmation_training import TrainedConfirmation
from src.app.continual_replay_factor_pilot import Model
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.controlled_parent_selection import ParentControlledCircadianNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.confirmation_final_roles import (
    EndpointFailure,
    EndpointResult,
    FinalRole,
    FinalRoleFacts,
    capture_final_role,
    validate_endpoint_result,
)
from src.infra.continual_confirmation_final import (
    evaluate_confirmation_final,
    release_confirmation_final,
)
from src.infra.continual_roles import PhaseDecisionRoles, PhaseSource


_MODEL_KINDS = {
    BackpropMLP: "backprop",
    PredictiveCodingNetwork: "pc",
    CircadianPredictiveCodingNetwork: "circadian",
    ParentControlledCircadianNetwork: "circadian",
}


@dataclass(frozen=True)
class _Key:
    family: str
    seed: int
    phase: str


class _UnavailableOuter:
    def __init__(self, owner: FinalExecutionObserver) -> None:
        self.owner = owner

    @property
    def input(self) -> np.ndarray:
        return self.owner._reject("outer input access during final evaluation")

    @property
    def target(self) -> np.ndarray:
        return self.owner._reject("outer label access during final evaluation")


class _ObservedSource:
    def __init__(self, owner: FinalExecutionObserver, key: _Key, source: PhaseSource) -> None:
        self.owner, self.key, self.source = owner, key, source
        self.values: dict[str, np.ndarray] = {}
        self.started: set[str] = set()

    @property
    def train_input(self) -> np.ndarray:
        return self.owner._reject("training source input access during final evaluation")

    @property
    def train_target(self) -> np.ndarray:
        return self.owner._reject("training source label access during final evaluation")

    @property
    def test_input(self) -> np.ndarray:
        return self.owner._read_source(self, "test_input")

    @property
    def test_target(self) -> np.ndarray:
        return self.owner._read_source(self, "test_target")


@dataclass(frozen=True)
class _RoleBinding:
    item: HeldSeed
    attribute: str
    key: _Key
    original: PhaseDecisionRoles
    source: PhaseSource
    observed: _ObservedSource
    outer: _UnavailableOuter
    guarded: PhaseDecisionRoles


@dataclass(frozen=True)
class _Prediction:
    item: HeldSeed
    arm: str
    endpoint: str
    checkpoint: str
    phase: str
    model: Model
    kind: str


class FinalExecutionObserver:
    """Observe the supplied held inventory; complete scientific gates are external."""

    def __init__(self, trained: TrainedConfirmation, budget_check: Callable[[], None]) -> None:
        require(type(trained) is TrainedConfirmation, "final observer requires held training")
        self.trained, self.budget_check = trained, budget_check
        self.active = False
        self.used = False
        self.blocked_calls = 0
        self.release_attempts = 0
        self.input_attempts = self.target_attempts = 0
        self.prediction_attempts = self.prediction_returns = self.prediction_numerical_errors = 0
        self.prediction_examples = self.unexpected_prediction_errors = 0
        self.by_model_kind = {"backprop": 0, "pc": 0, "circadian": 0}
        self.release_events: list[dict[str, Any]] = []
        self.source_events: list[dict[str, Any]] = []
        self.prediction_events: list[dict[str, Any]] = []
        self.views: dict[_Key, tuple[FinalRole, FinalRoleFacts]] = {}
        self._active_release: _Key | None = None
        self._active_prediction: tuple[_Prediction, FinalRole] | None = None
        self._started_releases: set[_Key] = set()
        self._started_predictions: set[int] = set()
        self._next_release = self._next_prediction = 0
        self._last_result: ScoredConfirmation | None = None
        self.bindings, self.predictions = self._inventory()

    def _reject(self, message: str) -> Any:
        self.blocked_calls += 1
        raise ValueError(f"confirmation final observer {message}")

    def _require_active(self) -> None:
        if not self.active:
            self._reject("is not active")

    def _inventory(self) -> tuple[tuple[_RoleBinding, ...], tuple[_Prediction, ...]]:
        families = self.trained.facts.manifest.families
        expected = tuple((f.name, seed) for f in families for seed in f.seeds)
        require(
            tuple((item.facts.family, item.facts.seed) for item in self.trained.held) == expected,
            "final observer held family/seed inventory differs",
        )
        require(
            bool(expected) and len(expected) <= 60,
            "final observer inventory exceeds complete bound",
        )
        bindings, predictions = [], []
        by_family = {family.name: family for family in families}
        for item in self.trained.held:
            arms = by_family[item.facts.family].arms
            require(
                set(item.models_after_a) == set(item.models_after_b) == set(arms),
                "final observer held arm inventory differs",
            )
            for phase, attribute in (("a", "roles_a"), ("b", "roles_b")):
                bindings.append(self._binding(item, phase, attribute))
            for arm in arms:
                for endpoint, held_phase, phase in (
                    ("a_after_a", "a", "a"),
                    ("a_after_b", "b", "a"),
                    ("b_after_b", "b", "b"),
                ):
                    model = (item.models_after_a if held_phase == "a" else item.models_after_b)[arm]
                    require(type(model) in _MODEL_KINDS, "final observer unknown model type")
                    predictions.append(
                        _Prediction(
                            item, arm, endpoint, held_phase, phase, model, _MODEL_KINDS[type(model)]
                        )
                    )
        require(
            len(predictions) <= 1680, "final observer endpoint inventory exceeds complete bound"
        )
        return tuple(bindings), tuple(predictions)

    def _binding(self, item: HeldSeed, phase: str, attribute: str) -> _RoleBinding:
        original = getattr(item, attribute)
        require(
            type(original) is PhaseDecisionRoles
            and original.phase == phase
            and type(original.seed) is int
            and original.seed == item.facts.seed
            and original._source is not None
            and original.final_test is None
            and original.final_released is False
            and original.expected_final_count == 40,
            "final observer original role identity/seal differs",
        )
        source = original._source
        assert source is not None
        key = _Key(item.facts.family, item.facts.seed, phase)
        observed = _ObservedSource(self, key, source)
        outer = _UnavailableOuter(self)
        guarded = replace(original, _source=observed, outer_selection=outer)
        return _RoleBinding(item, attribute, key, original, source, observed, outer, guarded)

    def _require_bindings(self) -> None:
        for binding in self.bindings:
            current = getattr(binding.item, binding.attribute)
            require(
                current is binding.guarded
                and current._source is binding.observed
                and current.outer_selection is binding.outer
                and binding.observed.source is binding.source
                and (binding.item.facts.family, binding.item.facts.seed)
                == (binding.key.family, binding.key.seed),
                "final observer original role/source binding changed",
            )
        for request in self.predictions:
            models = (
                request.item.models_after_a
                if request.checkpoint == "a"
                else request.item.models_after_b
            )
            require(
                models.get(request.arm) is request.model
                and type(request.model) in _MODEL_KINDS
                and _MODEL_KINDS[type(request.model)] == request.kind,
                "final observer held model binding changed",
            )

    def _restore_guards(self) -> None:
        for binding in self.bindings:
            current = getattr(binding.item, binding.attribute)
            if current is binding.guarded:
                setattr(binding.item, binding.attribute, binding.original)
            elif type(current) is PhaseDecisionRoles:
                # Remove only owned guards; preserve other corrupted metadata
                # for the enclosing live-state gate to reject.
                changes: dict[str, Any] = {}
                if current._source is binding.observed:
                    changes["_source"] = binding.source
                if current.outer_selection is binding.outer:
                    changes["outer_selection"] = binding.original.outer_selection
                if changes:
                    setattr(binding.item, binding.attribute, replace(current, **changes))

    @contextmanager
    def observe(self) -> Iterator[FinalExecutionObserver]:
        if self.active or self.used:
            self._reject("context cannot be nested or reused")
        self.budget_check()
        self.used = self.active = True
        try:
            with ExitStack() as stack:
                for binding in self.bindings:
                    require(
                        getattr(binding.item, binding.attribute) is binding.original,
                        "final observer role changed before installation",
                    )
                    setattr(binding.item, binding.attribute, binding.guarded)
                for model, kind in (
                    (BackpropMLP, "backprop"),
                    (PredictiveCodingNetwork, "pc"),
                    (CircadianPredictiveCodingNetwork, "circadian"),
                ):
                    stack.enter_context(
                        patch.object(
                            model, "predict_proba", self._wrap_prediction(model.predict_proba, kind)
                        )
                    )
                for model, name in (
                    (BackpropMLP, "train_epoch"),
                    (PredictiveCodingNetwork, "train_epoch"),
                    (CircadianPredictiveCodingNetwork, "_run_training_step"),
                ):
                    stack.enter_context(
                        patch.object(
                            model,
                            name,
                            lambda *args, **kwargs: self._reject(
                                "optimizer call during final evaluation"
                            ),
                        )
                    )
                self._require_bindings()
                self.budget_check()
                yield self
                require(
                    self._last_result is not None, "final observer incomplete result verification"
                )
                assert self._last_result is not None
                self.verify_result(self._last_result)
        finally:
            self.active = False
            self._active_release = self._active_prediction = None
            self._restore_guards()

    def _read_source(self, source: _ObservedSource, field: str) -> np.ndarray:
        self._require_active()
        expected_field = "test_input" if not source.started else "test_target"
        if self._active_release != source.key or field in source.started or field != expected_field:
            self._reject("source field outside its single scheduled release")
        self.budget_check()
        source.started.add(field)
        if field == "test_input":
            self.input_attempts += 1
        else:
            self.target_attempts += 1
        value = getattr(source.source, field)
        source.values[field] = value
        self.source_events.append(
            {
                "family": source.key.family,
                "seed": source.key.seed,
                "phase": source.key.phase,
                "field": field,
                "example_count": int(value.shape[0])
                if isinstance(value, np.ndarray) and value.ndim
                else None,
            }
        )
        self.budget_check()
        return value

    def release(self, item: HeldSeed, phase: str) -> FinalRole:
        self._require_active()
        if self._next_release >= len(self.bindings):
            return self._reject("extra final release")
        binding = self.bindings[self._next_release]
        if (
            item is not binding.item
            or phase != binding.key.phase
            or binding.key in self._started_releases
        ):
            return self._reject("release order/inventory differs")
        self._require_bindings()
        self.budget_check()
        self._started_releases.add(binding.key)
        self.release_attempts += 1
        self._active_release = binding.key
        try:
            role = release_confirmation_final(item, phase)
        finally:
            self._active_release = None
        facts = self._capture_view(binding, role)
        self.views[binding.key] = (role, facts)
        self.release_events.append(
            {
                "family": binding.key.family,
                "seed": binding.key.seed,
                "phase": phase,
                "role_sha256": facts.sha256,
                "example_count": facts.count,
                "sample_ids": list(facts.sample_ids),
            }
        )
        self._next_release += 1
        self.budget_check()
        return role

    def _capture_view(self, binding: _RoleBinding, role: FinalRole) -> FinalRoleFacts:
        facts = capture_final_role(role, binding.original.expected_final_count)
        require(
            set(binding.observed.values) == {"test_input", "test_target"}
            and role.input is binding.observed.values["test_input"]
            and role.target is binding.observed.values["test_target"],
            "final observer released arrays differ from actual source fields",
        )
        require(
            (facts.phase, facts.seed, facts.sample_ids)
            == (binding.key.phase, binding.key.seed, binding.original.sample_ids["final_test"]),
            "final observer released identity differs from original role",
        )
        return facts

    def verify_views(self) -> None:
        self._require_active()
        self.budget_check()
        self._require_bindings()
        for binding in self.bindings:
            if binding.key in self.views:
                role, before = self.views[binding.key]
                require(
                    self._capture_view(binding, role) == before,
                    "final observer released content changed after capture",
                )

    def checkpoint(self, stage: str) -> None:
        self._require_active()
        self.budget_check()
        if stage in {"before_final_release", "before_final_evaluation", "after_final_evaluation"}:
            self.verify_views()

    def _observed_result(self, probability: Any, role: FinalRole) -> EndpointResult:
        require(
            isinstance(probability, np.ndarray)
            and probability.dtype == np.float64
            and probability.shape == (len(role.sample_ids), 1),
            "final observer prediction shape/dtype differs",
        )
        if not np.all(np.isfinite(probability)):
            return EndpointResult(None, EndpointFailure("nonfinite_predictions"))
        require(
            bool(np.all((probability >= 0.0) & (probability <= 1.0))),
            "final observer probability range differs",
        )
        return EndpointResult(int(np.count_nonzero((probability >= 0.5) == role.target)))

    def _record_prediction(
        self, request: _Prediction, role: FinalRole, result: EndpointResult, returned: bool
    ) -> None:
        record = EndpointEvaluation(
            request.item.facts.family,
            request.item.facts.seed,
            request.arm,
            request.endpoint,
            request.checkpoint,
            role.phase,
            role.sha256,
            len(role.sample_ids),
            result,
        )
        self.prediction_events.append(
            {**json_value(asdict(record)), "model_kind": request.kind, "returned": returned}
        )

    def _wrap_prediction(self, original: Callable[..., Any], kind: str) -> Callable[..., Any]:
        @wraps(original)
        def observed(model: Any, input_batch: Any) -> Any:
            self._require_active()
            if self._active_prediction is None:
                return self._reject("prediction outside scheduled endpoint")
            request, role = self._active_prediction
            if (
                model is not request.model
                or kind != request.kind
                or input_batch is not role.input
                or self.prediction_attempts != self._next_prediction
            ):
                return self._reject("prediction model/input/single-call identity differs")
            self.budget_check()
            self.prediction_attempts += 1
            self.prediction_examples += len(role.sample_ids)
            self.by_model_kind[kind] += 1
            try:
                probability = original(model, input_batch)
            except FloatingPointError:
                self.prediction_numerical_errors += 1
                self._record_prediction(
                    request,
                    role,
                    EndpointResult(
                        None, EndpointFailure("numerical_prediction_error", "FloatingPointError")
                    ),
                    False,
                )
                self.budget_check()
                raise
            except BaseException:
                self.unexpected_prediction_errors += 1
                raise
            self.prediction_returns += 1
            try:
                result = self._observed_result(probability, role)
            except ValueError:
                self.unexpected_prediction_errors += 1
                raise
            self._record_prediction(request, role, result, True)
            self.budget_check()
            return probability

        return observed

    def evaluate(self, model: Model, role: FinalRole) -> EndpointResult:
        self._require_active()
        if self._next_release != len(self.bindings) or self._next_prediction >= len(
            self.predictions
        ):
            return self._reject("prediction before all releases or extra endpoint")
        request = self.predictions[self._next_prediction]
        key = _Key(request.item.facts.family, request.item.facts.seed, request.phase)
        if (
            model is not request.model
            or key not in self.views
            or role is not self.views[key][0]
            or self._next_prediction in self._started_predictions
        ):
            return self._reject("endpoint model/role/order identity differs")
        self._require_bindings()
        require(
            capture_final_role(role, len(role.sample_ids)) == self.views[key][1],
            "final observer released content changed before prediction",
        )
        self.budget_check()
        self._started_predictions.add(self._next_prediction)
        self._active_prediction = (request, role)
        try:
            result = evaluate_confirmation_final(model, role)
        finally:
            self._active_prediction = None
        validate_endpoint_result(result, len(role.sample_ids))
        require(
            len(self.prediction_events) == self._next_prediction + 1,
            "final observer adapter did not make one actual prediction",
        )
        same_json(
            json_value(asdict(result)),
            self.prediction_events[-1]["result"],
            "final observer actual prediction/adapter result",
        )
        self._next_prediction += 1
        self.budget_check()
        return result

    def observations(self) -> dict[str, Any]:
        return json_value(
            {
                "schema_id": "p67_confirmation_final_observation_v1",
                "validation_scope": "actual_final_calls_for_supplied_held_inventory_only",
                "source_provenance_verified": False,
                "release_attempts": self.release_attempts,
                "release_successes": len(self.release_events),
                "input_attempts": self.input_attempts,
                "input_reads": sum(row["field"] == "test_input" for row in self.source_events),
                "target_attempts": self.target_attempts,
                "target_reads": sum(row["field"] == "test_target" for row in self.source_events),
                "prediction_attempts": self.prediction_attempts,
                "prediction_returns": self.prediction_returns,
                "prediction_numerical_errors": self.prediction_numerical_errors,
                "prediction_examples": self.prediction_examples,
                "unexpected_prediction_errors": self.unexpected_prediction_errors,
                "blocked_calls": self.blocked_calls,
                "by_model_kind": self.by_model_kind,
                "release_events": self.release_events,
                "source_events": self.source_events,
                "prediction_events": self.prediction_events,
            }
        )

    def verify_result(self, scored: ScoredConfirmation) -> None:
        self._require_active()
        self.budget_check()
        require(
            type(scored) is ScoredConfirmation
            and self._next_release == len(self.bindings)
            and self._next_prediction == len(self.predictions),
            "final observer incomplete supplied execution",
        )
        self.verify_views()
        same_json(
            self.observations(),
            _expected_observation(scored, self.trained.facts.manifest),
            "actual final execution/app endpoint links",
        )
        self._last_result = scored
