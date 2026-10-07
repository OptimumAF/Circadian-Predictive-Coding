"""Adapt pinned final release and model prediction to the inner contracts.

Inputs are already verified held roles/models. Outputs are final views or
exact endpoint counts/declared numerical failures. Global authorization,
scope/source/request/resource proof, orchestration and IO belong elsewhere.
"""

from __future__ import annotations

import numpy as np

from src.app.continual_confirmation_state import HeldSeed
from src.app.continual_replay_factor_pilot import Model
from src.core.confirmation_final_roles import (
    EndpointFailure,
    EndpointResult,
    FinalRole,
    capture_final_role,
)
from src.infra.continual_roles import PhaseDecisionRoles, release_final_test


def _require_release_only(original: PhaseDecisionRoles, released: PhaseDecisionRoles) -> None:
    if (
        type(released) is not PhaseDecisionRoles
        or released is original
        or released.final_released is not True
        or released._source is not None
        or released.phase != original.phase
        or type(released.seed) is not int
        or released.seed != original.seed
        or type(released.expected_final_count) is not int
        or released.expected_final_count != original.expected_final_count
        or released.final_test is None
        or any(
            getattr(released, name) is not getattr(original, name)
            for name in ("train", "inner_guard", "outer_selection", "sample_ids", "release_policy")
        )
        or set(released.split_hashes) != {*original.split_hashes, "final_test"}
        or any(
            released.split_hashes[name] != digest for name, digest in original.split_hashes.items()
        )
    ):
        raise ValueError("confirmation final release changed original role metadata")


def release_confirmation_final(item: HeldSeed, phase: str) -> FinalRole:
    if type(phase) is not str or phase not in {"a", "b"}:
        raise ValueError("confirmation final release requires phase a/b")
    original = item.roles_a if phase == "a" else item.roles_b
    released = release_final_test(original)
    _require_release_only(original, released)
    final = released.final_test
    assert final is not None
    role = FinalRole(
        released.phase,
        released.seed,
        released.sample_ids["final_test"],
        final.input,
        final.target,
        released.split_hashes["final_test"],
    )
    capture_final_role(role, original.expected_final_count)
    return role


def evaluate_confirmation_final(model: Model, role: FinalRole) -> EndpointResult:
    """Keep the original binary threshold; preserve declared numerical failures."""
    if type(role) is not FinalRole or type(role.sample_ids) is not tuple:
        raise ValueError("confirmation final role type differs before prediction")
    count = len(role.sample_ids)
    capture_final_role(role, count)
    try:
        probability = model.predict_proba(role.input)
    except FloatingPointError:
        return EndpointResult(
            None, EndpointFailure("numerical_prediction_error", "FloatingPointError")
        )
    if (
        not isinstance(probability, np.ndarray)
        or probability.dtype != np.float64
        or probability.shape != (count, 1)
    ):
        raise ValueError("confirmation final prediction shape/dtype differs")
    if not np.all(np.isfinite(probability)):
        return EndpointResult(None, EndpointFailure("nonfinite_predictions"))
    if not np.all((probability >= 0.0) & (probability <= 1.0)):
        raise ValueError("confirmation final prediction probability range differs")
    correct = int(np.count_nonzero((probability >= 0.5) == role.target))
    return EndpointResult(correct)
