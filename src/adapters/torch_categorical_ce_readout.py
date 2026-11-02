"""Bounded disposable TRAIN CE learning and typed reads on fixed BP features.

The source remains unchanged; a separate float32 output fit owns its optimizer
and costs. Callers own label arrival, actor, clocks and promotion authority.
No model-state loading from files, final/probe selection, replay or IO.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
from math import isfinite
from typing import Callable

import numpy as np

from src.adapters import torch_feature_learner as native
from src.adapters.perceptron_decision import build_decision_training_batch
from src.adapters.torch_categorical_readout_fit import (
    _hidden,
    _unchanged_owner,
    read_hidden_features,
)
from src.adapters.torch_decision import _same_values, decision_from_logits
from src.adapters.torch_array_weighted import _ownership
from src.app.decision_reference_evaluation import DecisionCase
from src.core.decision_features import DecisionFeatureSpec, encode_decision
from src.shared.torch_runtime import require_torch

DEFINITION = "frozen_native_bp_output_mean_ce_lbfgs_float32_v1"
MAX_CLOSURES = 64


def _output_logits(values, weight, bias):
    """Same float32 affine arithmetic for fitting and frozen typed inference."""
    return values @ weight + bias


class CategoricalCEFitFailed(RuntimeError):
    """Attempted detached fitting remains paid even when no fit returns."""

    def __init__(self, reason: str, work: dict[str, int]):
        super().__init__(reason)
        self.work = deepcopy(work)


@dataclass(frozen=True)
class CategoricalCEFit:
    source: native.TorchFeatureSnapshot
    spec: DecisionFeatureSpec
    coefficients: tuple[tuple[float, ...], ...]
    training_logits: tuple[tuple[float, ...], ...]
    closure_losses: tuple[float, ...]
    final_loss: float
    work: dict[str, int]


def _fit_copy(fit: CategoricalCEFit) -> CategoricalCEFit:
    if type(fit) is not CategoricalCEFit:
        raise ValueError("an exact categorical CE fit is required")
    state = native._snapshot(fit.source)
    policy = state.policy
    if type(fit.spec) is not DecisionFeatureSpec:
        raise ValueError("fit requires an exact feature schema")
    fit.spec.__post_init__()
    if (
        policy.method != "bp"
        or policy.feature_dim != len(fit.spec.vocabulary)
        or policy.num_classes != len(fit.spec.catalog)
        or type(fit.coefficients) is not tuple
        or len(fit.coefficients) != policy.hidden_dim + 1
    ):
        raise ValueError("CE fit source/schema/coefficient geometry differs")
    for row in fit.coefficients:
        if (
            type(row) is not tuple
            or len(row) != policy.num_classes
            or any(
                type(v) is not float
                or not isfinite(v)
                or abs(v) > np.finfo(np.float32).max
                or float(np.float32(v)) != v
                for v in row
            )
        ):
            raise ValueError("CE fit coefficients must be finite immutable float32 values")
    work = fit.work
    if type(work) is not dict:
        raise ValueError("CE fit work must be an exact dictionary")
    rows = work.get("hidden_rows", 0)
    calls = work.get("closure_attempts", 0)
    if (
        any(type(v) is not int or v < 0 for v in work.values())
        or not 1 <= rows <= 128
        or not 1 <= calls <= 64
        or any(
            work.get(key) != 1
            for key in (
                "hidden_attempts",
                "hidden_returns",
                "optimizer_constructions",
                "step_attempts",
                "step_returns",
                "final_read_attempts",
                "final_read_returns",
            )
        )
        or any(
            work.get(key) != calls
            for key in (
                "forward_attempts",
                "forward_returns",
                "backward_attempts",
                "backward_returns",
            )
        )
        or work.get("forward_rows") != rows * calls
        or work.get("backward_rows") != rows * calls
        or work.get("final_read_rows") != rows
        or type(fit.closure_losses) is not tuple
        or len(fit.closure_losses) != calls
        or any(type(v) is not float or not isfinite(v) or v < 0 for v in fit.closure_losses)
        or type(fit.final_loss) is not float
        or not isfinite(fit.final_loss)
        or fit.final_loss < 0
        or type(fit.training_logits) is not tuple
        or len(fit.training_logits) != rows
        or any(
            type(row) is not tuple
            or len(row) != policy.num_classes
            or any(type(v) is not float or not isfinite(v) for v in row)
            for row in fit.training_logits
        )
    ):
        raise ValueError("CE fit requires complete finite bounded result and work")
    return deepcopy(replace(fit, source=state))


def _new_ce_work():
    return dict(
        hidden_attempts=0,
        hidden_returns=0,
        hidden_rows=0,
        optimizer_constructions=0,
        step_attempts=0,
        step_returns=0,
        closure_attempts=0,
        forward_attempts=0,
        forward_returns=0,
        forward_rows=0,
        backward_attempts=0,
        backward_returns=0,
        backward_rows=0,
        final_read_attempts=0,
        final_read_returns=0,
        final_read_rows=0,
    )


def _fit_output_arrays(state, features, targets, work, boundary):
    """Shared detached arithmetic; public wrappers retain distinct role authority."""
    torch = require_torch()
    closure_limit = MAX_CLOSURES
    if closure_limit != 64:
        raise ValueError("fixed CE readout closure limit changed")
    work.update(hidden_attempts=1, hidden_rows=len(features))
    hidden = _hidden(state, features)
    work["hidden_returns"] = 1
    boundary()
    values = torch.from_numpy(hidden.astype(np.float32))
    labels = torch.from_numpy(targets[:, 0].astype(np.int64))
    values_pin, labels_pin = values.clone(), labels.clone()
    weight = state.head_state["weight_hidden_output"].detach().clone().requires_grad_(True)
    bias = state.head_state["bias_output"].detach().clone().requires_grad_(True)
    losses = []

    def require_data():
        boundary()
        if not torch.equal(values, values_pin) or not torch.equal(labels, labels_pin):
            raise ValueError("CE readout detached fit arrays changed")

    require_data()
    work["optimizer_constructions"] += 1
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=32,
        max_eval=48,
        history_size=10,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    def closure():
        require_data()
        work["closure_attempts"] += 1
        if work["closure_attempts"] > closure_limit:
            raise RuntimeError("CE readout closure allowance spent")
        optimizer.zero_grad(set_to_none=True)
        work["forward_attempts"] += 1
        work["forward_rows"] += len(features)
        logits = _output_logits(values, weight, bias)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("nonfinite CE readout loss")
        work["forward_returns"] += 1
        require_data()
        work["backward_attempts"] += 1
        work["backward_rows"] += len(features)
        loss.backward()
        work["backward_returns"] += 1
        require_data()
        losses.append(float(loss.detach().item()))
        return loss

    work["step_attempts"] += 1
    optimizer.step(closure)
    work["step_returns"] += 1
    require_data()
    native._tensor(
        weight,
        (state.policy.hidden_dim, state.policy.num_classes),
        torch.float32,
        "CE output weight",
    )
    native._tensor(bias, (1, state.policy.num_classes), torch.float32, "CE output bias")
    work["final_read_attempts"] += 1
    work["final_read_rows"] += len(features)
    with torch.no_grad():
        logits = _output_logits(values, weight, bias)
        native._tensor(
            logits, (len(features), state.policy.num_classes), torch.float32, "CE fit logits"
        )
        final_loss = float(torch.nn.functional.cross_entropy(logits, labels).item())
        if not isfinite(final_loss):
            raise FloatingPointError("nonfinite final CE readout loss")
        coefficients = torch.cat((weight, bias), dim=0).tolist()
    work["final_read_returns"] += 1
    require_data()
    return (
        tuple(tuple(float(v) for v in row) for row in coefficients),
        tuple(tuple(float(v) for v in row) for row in logits.tolist()),
        tuple(losses),
        final_loss,
    )


def fit_categorical_ce_readout(
    learner: native.TorchFeatureLearner,
    cases: tuple[DecisionCase, ...],
    spec: DecisionFeatureSpec,
    *,
    check_boundary: Callable[[], None] | None = None,
) -> CategoricalCEFit:
    """One fixed LBFGS step on detached output parameters; never install them.

    Why: frozen affine mean CE is convex; source feature learning and original
    SGD state must not obscure whether this small readout can fit its TRAIN data.
    The hard closure cap is additional to LBFGS's soft evaluation stopping rule.
    """
    if (
        type(cases) is not tuple
        or not 1 <= len(cases) <= 128
        or any(
            type(c) is not DecisionCase or type(c.role) is not str or c.role != "train"
            for c in cases
        )
    ):
        raise ValueError("CE readout requires only exact released TRAIN cases")
    work = _new_ce_work()
    closure_limit = MAX_CLOSURES
    if closure_limit != 64:
        raise ValueError("fixed CE readout closure limit changed")
    batch = None
    batch_pin = None
    try:
        with _unchanged_owner(learner) as (state, unchanged):
            source_input = deepcopy((cases, spec))
            torch = require_torch()
            if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
                raise ValueError("CE fitting requires grad-enabled non-inference mode")

            def boundary():
                unchanged()
                if check_boundary is not None:
                    check_boundary()
                unchanged()
                if not _same_values((cases, spec), source_input):
                    raise ValueError("CE readout TRAIN input changed")
                if (
                    batch is not None
                    and batch_pin is not None
                    and (
                        not np.array_equal(batch.features, batch_pin[0])
                        or not np.array_equal(batch.targets, batch_pin[1])
                        or batch.features.dtype != batch_pin[0].dtype
                        or batch.targets.dtype != batch_pin[1].dtype
                    )
                ):
                    raise ValueError("CE readout TRAIN batch changed")

            boundary()
            batch = build_decision_training_batch(
                cases, spec, "frozen-ce-readout-train", allow_subsets=True
            )
            batch_pin = (batch.features.copy(), batch.targets.copy())
            if (
                len(spec.vocabulary) != state.policy.feature_dim
                or len(spec.catalog) != state.policy.num_classes
            ):
                raise ValueError("CE readout schema differs from native policy")
            boundary()
            coefficients, logits, losses, final_loss = _fit_output_arrays(
                state, batch.features, batch.targets, work, boundary
            )
            return CategoricalCEFit(
                deepcopy(state),
                deepcopy(spec),
                coefficients,
                logits,
                losses,
                final_loss,
                deepcopy(work),
            )
    except Exception as error:
        raise CategoricalCEFitFailed(f"{type(error).__name__}: {error}", work) from error


@dataclass(frozen=True)
class CEReadoutWork:
    decision_attempts: int = 0
    decision_returns: int = 0
    hidden_attempts: int = 0
    hidden_returns: int = 0
    hidden_rows: int = 0
    affine_attempts: int = 0
    affine_returns: int = 0
    affine_rows: int = 0


class CategoricalCEReadoutPredictor:
    """Constant-size owned readout; stop after faults, never modify a donor."""

    def __init__(self, fit: CategoricalCEFit):
        self._fit = _fit_copy(fit)
        self._pin = deepcopy(self._fit)
        self._fit_owner = self._fit
        self._learner = native.TorchFeatureLearner(self._fit.source.policy)
        self._learner.restore_state(self._fit.source)
        self._payload, self._policy = self._learner._payload, self._learner._policy
        self._binding = _ownership(self._learner)
        self._coefficients = require_torch().tensor(
            self._fit.coefficients, dtype=require_torch().float32, device="cpu"
        )
        self._coefficient_pin = self._coefficients.clone()
        self._coefficient_owner = self._coefficients
        self._work = CEReadoutWork()
        self._work_owner = self._work
        self._work_pin = deepcopy(self._work)
        self._stopped = False
        self.assert_unchanged()

    @property
    def work(self):
        return replace(self._work_pin)

    def _charge(self, **increments):
        self.assert_unchanged()
        self._work = replace(
            self._work,
            **{key: getattr(self._work, key) + value for key, value in increments.items()},
        )
        self._work_owner = self._work
        self._work_pin = deepcopy(self._work)

    def assert_unchanged(self):
        try:
            source = self._learner
            actual = native._state(
                source._head,
                source._optimizer,
                native._policy(source._policy),
                source.completed_updates,
                source.training_examples,
            )
            if (
                self._stopped
                or source._payload is not self._payload
                or source._policy is not self._policy
                or _ownership(source) != self._binding
                or not _same_values(actual, self._pin.source)
                or self._fit is not self._fit_owner
                or not _same_values(self._fit, self._pin)
                or self._coefficients is not self._coefficient_owner
                or not _same_values(self._coefficients, self._coefficient_pin)
                or self._work is not self._work_owner
                or self._work != self._work_pin
            ):
                raise ValueError("frozen CE readout state changed or previously stopped")
        except Exception:
            self._stopped = True
            raise

    def __call__(self, request):
        try:
            self._charge(decision_attempts=1)
            self.assert_unchanged()
            with _unchanged_owner(self._learner):
                features, visits = encode_decision(request, self._fit.spec)
                self._charge(hidden_attempts=1, hidden_rows=1)
                hidden = read_hidden_features(self._learner, features)
                self._charge(hidden_returns=1)
                self.assert_unchanged()
                self._charge(affine_attempts=1, affine_rows=1)
                torch = require_torch()
                with torch.no_grad():
                    logits = _output_logits(
                        torch.from_numpy(hidden.astype(np.float32)),
                        self._coefficients[:-1],
                        self._coefficients[-1:],
                    )
                    native._tensor(
                        logits, (1, len(self._fit.spec.catalog)), torch.float32, "CE readout logits"
                    )
                self._charge(affine_returns=1)
                self.assert_unchanged()
                result = decision_from_logits(request, self._fit.spec, logits[0].tolist(), visits)
                self.assert_unchanged()
            self._charge(decision_returns=1)
            return result
        except Exception:
            self._stopped = True
            raise
