"""Explicit CE-output initialization for disposable native probes.

Derives owned in-memory state from an unchanged actual BP owner and its TRAIN
fit. No models, fitting, IO, source restore, budgets or promotion. The receiving
probe must use its own factory/callbacks and account for construction/training.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Callable

from src.adapters import torch_feature_learner as native
from src.adapters.torch_categorical_ce_readout import CategoricalCEFit, DEFINITION, _fit_copy
from src.adapters.torch_categorical_readout_fit import _unchanged_owner
from src.adapters.torch_decision import _same_values
from src.shared.torch_runtime import require_torch

INITIALIZATION = "native_bp_ce_output_probe_initialization_v1"
CONTINUATION = "preserve_source_sgd_momentum_gradients_policy_and_native_counters"


@dataclass(frozen=True)
class CategoricalCEInitialization:
    definition: str
    continuation: str
    fit: CategoricalCEFit
    state: native.TorchFeatureSnapshot
    copied_output_values: int


def _derived_state(fit: CategoricalCEFit) -> native.TorchFeatureSnapshot:
    torch = require_torch()
    coefficients = torch.tensor(fit.coefficients, dtype=torch.float32, device="cpu")
    head = deepcopy(fit.source.head_state)
    head["weight_hidden_output"] = coefficients[:-1].clone()
    head["bias_output"] = coefficients[-1:].clone()
    # Why: CE is additional auxiliary work, not another original SGD update.
    # Preserve even stale gradient/momentum buffers for explicit continuation.
    return replace(fit.source, head_state=head)


def copy_ce_initialization(
    initialization: CategoricalCEInitialization,
) -> CategoricalCEInitialization:
    """Revalidate the complete derived state and return a detached receipt.

    This is an ordinary Python ownership contract, not cryptographic evidence of
    TRAIN arrival or a permission to reconstruct a source from scalar reports.
    """
    if type(initialization) is not CategoricalCEInitialization:
        raise ValueError("an exact CE initialization is required")
    fit = _fit_copy(initialization.fit)
    expected = _derived_state(fit)
    values = (fit.source.policy.hidden_dim + 1) * fit.source.policy.num_classes
    if (
        type(initialization.definition) is not str
        or initialization.definition != INITIALIZATION
        or type(initialization.continuation) is not str
        or initialization.continuation != CONTINUATION
        or type(initialization.copied_output_values) is not int
        or initialization.copied_output_values != values
        or not _same_values(native._snapshot(initialization.state), expected)
    ):
        raise ValueError("CE initialization state or provenance changed")
    return CategoricalCEInitialization(INITIALIZATION, CONTINUATION, fit, expected, values)


def initialize_categorical_ce_probe(
    source: native.TorchFeatureLearner,
    fit: CategoricalCEFit,
    *,
    check_boundary: Callable[[], None] | None = None,
) -> CategoricalCEInitialization:
    """Bind a fit to the actual source, changing only detached output parameters."""
    with _unchanged_owner(source) as (state, unchanged):
        if type(fit) is not CategoricalCEFit:
            raise ValueError("an exact categorical CE fit is required")
        binding = (fit.source, fit.spec, fit.coefficients, fit.work)
        fit_pin = deepcopy(fit)

        def boundary():
            unchanged()
            if check_boundary is not None:
                check_boundary()
            unchanged()
            if not _same_values(fit, fit_pin) or any(
                current is not original
                for current, original in zip(
                    (fit.source, fit.spec, fit.coefficients, fit.work), binding, strict=True
                )
            ):
                raise ValueError("CE initialization fit changed")

        boundary()
        owned = _fit_copy(fit)
        if not _same_values(owned.source, state):
            raise ValueError("CE initialization fit differs from actual source")
        boundary()
        result = CategoricalCEInitialization(
            INITIALIZATION,
            CONTINUATION,
            owned,
            _derived_state(owned),
            (state.policy.hidden_dim + 1) * state.policy.num_classes,
        )
        boundary()
        return result


def ce_initialization_record(initialization: CategoricalCEInitialization) -> dict:
    """Small cost/provenance summary; full source/derived states stay separate."""
    owned = copy_ce_initialization(initialization)
    return dict(
        definition=owned.definition,
        continuation=owned.continuation,
        fit_definition=DEFINITION,
        fit_work=deepcopy(owned.fit.work),
        fit_closure_losses=owned.fit.closure_losses,
        fit_final_loss=owned.fit.final_loss,
        coefficients=owned.fit.coefficients,
        source_native_updates=owned.fit.source.completed_updates,
        source_native_rows=owned.fit.source.training_examples,
        copied_output_values=owned.copied_output_values,
        additional_native_updates=0,
        additional_native_rows=0,
        additional_model_constructions=0,
    )
