"""Disposable categorical TRAIN readout on an unchanged native BP representation.

Inputs are an existing BP owner and explicitly released TRAIN routing cases.
Outputs are detached hidden features, a fixed least-squares readout and its TRAIN
logits. No model construction, updates, restore, promotion, selection or IO.
Caller owns data release, clocks and accounting, including failed work.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from src.adapters import torch_feature_learner as native
from src.adapters.perceptron_decision import build_decision_training_batch
from src.adapters.torch_array_weighted import _ownership, _rng
from src.adapters.torch_decision import _same_values
from src.app.decision_reference_evaluation import DecisionCase
from src.core.categorical_linear_fit import (
    CategoricalLinearFit,
    fit_categorical_linear,
    predict_categorical_linear_logits,
)
from src.core.decision_features import DecisionFeatureSpec
from src.shared.torch_runtime import require_torch


@dataclass(frozen=True)
class CategoricalReadoutDiagnostic:
    hidden_features: np.ndarray
    fit: CategoricalLinearFit
    training_logits: np.ndarray
    feature_calls: int
    feature_rows: int
    solve_calls: int
    solve_rows: int
    readout_calls: int
    readout_rows: int


@contextmanager
def _unchanged_owner(learner: native.TorchFeatureLearner):
    if not isinstance(learner, native.TorchFeatureLearner):
        raise ValueError("categorical readout requires a native BP owner")
    policy_owner, payload_owner = learner._policy, learner._payload
    policy = native._policy(policy_owner)
    if policy.method != "bp":
        raise ValueError("categorical readout currently requires BP")

    def actual_state():
        return native._state(
            learner._head,
            learner._optimizer,
            policy,
            learner.completed_updates,
            learner.training_examples,
        )

    # Why: a replaceable public snapshot callback cannot attest to its owner.
    saved = actual_state()
    baseline = deepcopy(saved)
    ownership = _ownership(learner)
    torch = require_torch()
    rng = _rng(torch)

    def require_unchanged():
        if (
            learner._policy is not policy_owner
            or learner._payload is not payload_owner
            or native._policy(learner._policy) != policy
            or _ownership(learner) != ownership
            or not _same_values(saved, baseline)
            or not _same_values(baseline, actual_state())
            or not _same_values(rng, _rng(torch))
        ):
            raise ValueError("categorical readout source owner or RNG changed")

    try:
        yield saved, require_unchanged
    finally:
        # Never restore: refuse foreign changes while preserving their evidence.
        require_unchanged()


def _hidden(state: native.TorchFeatureSnapshot, features: np.ndarray) -> np.ndarray:
    values = native._features(features, state.policy.feature_dim)
    torch = require_torch()
    with torch.no_grad():
        hidden = torch.tanh(
            values @ state.head_state["weight_feature_hidden"] + state.head_state["bias_hidden"]
        )
        native._tensor(hidden, (len(features), state.policy.hidden_dim), torch.float32, "hidden")
        return hidden.cpu().numpy().astype(np.float64, copy=True)


def read_hidden_features(learner: native.TorchFeatureLearner, features: np.ndarray) -> np.ndarray:
    """Read native prior tanh features; no labels, gradients or optimizer work."""
    with _unchanged_owner(learner) as (state, _):
        # Validate before copying, preserving the native exact-array contract.
        native._features(features, state.policy.feature_dim)
        pin = features.copy()
        try:
            return _hidden(state, features)
        finally:
            if (
                type(features) is not np.ndarray
                or features.dtype != pin.dtype
                or features.shape != pin.shape
                or not np.array_equal(features, pin)
            ):
                raise ValueError("categorical readout input features changed")


def diagnose_categorical_readout_fit(
    learner: native.TorchFeatureLearner,
    cases: tuple[DecisionCase, ...],
    spec: DecisionFeatureSpec,
) -> CategoricalReadoutDiagnostic:
    """Fit only supplied TRAIN cases; known-row success is not task generalization.

    A failed LS readout does not rule out other CE optima. The hidden parameters
    stay fixed; the disposable (H+1)*K coefficients match the original output
    layer's capacity, but float64 LS is not the original float32 CE optimizer.
    """
    if (
        type(cases) is not tuple
        or not 1 <= len(cases) <= 128
        or any(type(case) is not DecisionCase or case.role != "train" for case in cases)
    ):
        raise ValueError("categorical readout requires only exact released TRAIN cases")
    source_pin = deepcopy((cases, spec))
    with _unchanged_owner(learner) as (state, unchanged):

        def require_inputs():
            unchanged()
            if not _same_values((cases, spec), source_pin):
                raise ValueError("categorical readout TRAIN cases or schema changed")

        try:
            batch = build_decision_training_batch(
                cases, spec, "frozen-categorical-train-readout", allow_subsets=True
            )
            if (
                len(spec.vocabulary) != state.policy.feature_dim
                or len(spec.catalog) != state.policy.num_classes
            ):
                raise ValueError("categorical readout schema differs from native policy")
            require_inputs()
            hidden = _hidden(state, batch.features)
            require_inputs()
            fit = fit_categorical_linear(hidden, batch.targets, state.policy.num_classes)
            require_inputs()
            logits = predict_categorical_linear_logits(fit, hidden)
            return CategoricalReadoutDiagnostic(
                hidden, fit, logits, 1, len(hidden), 1, len(hidden), 1, len(hidden)
            )
        finally:
            require_inputs()
