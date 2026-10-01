"""Pure released final-data fingerprints and exact endpoint count contracts.

Inputs are already released arrays/IDs or declared correct counts/failures.
Outputs bind the original role-byte contract and optional accuracy. No source,
model, release, training, score computation, app/infra or IO belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np


@dataclass(frozen=True)
class FinalRole:
    phase: str
    seed: int
    sample_ids: tuple[str, ...]
    input: np.ndarray
    target: np.ndarray
    sha256: str


@dataclass(frozen=True)
class FinalRoleFacts:
    phase: str
    seed: int
    sample_ids: tuple[str, ...]
    sha256: str
    count: int


@dataclass(frozen=True)
class EndpointFailure:
    code: str
    error_type: str | None = None


@dataclass(frozen=True)
class EndpointResult:
    correct_count: int | None
    failure: EndpointFailure | None = None


def _require_fields(role: FinalRole, count: int) -> None:
    if (
        type(role) is not FinalRole
        or type(count) is not int
        or count <= 0
        or type(role.phase) is not str
        or role.phase not in {"a", "b"}
        or type(role.seed) is not int
        or type(role.sample_ids) is not tuple
        or len(role.sample_ids) != count
        or any(
            type(item) is not str or not item or not item.isascii() or "\0" in item
            for item in role.sample_ids
        )
        or len(set(role.sample_ids)) != count
    ):
        raise ValueError("confirmation final role identity/count/IDs differ")
    if (
        not isinstance(role.input, np.ndarray)
        or not isinstance(role.target, np.ndarray)
        or role.input.dtype != np.float64
        or role.target.dtype != np.float64
        or role.input.shape != (count, 2)
        or role.target.shape != (count, 1)
        or not np.all(np.isfinite(role.input))
        or not np.all(np.isin(role.target, (0.0, 1.0)))
    ):
        raise ValueError("confirmation final role arrays/count/binary fields differ")


def final_content_digest(role: FinalRole) -> str:
    """Independently rederive the unchanged original final-role hash contract."""
    if type(role) is not FinalRole or type(role.sample_ids) is not tuple:
        raise ValueError("confirmation final role identity differs")
    _require_fields(role, len(role.sample_ids))
    digest = sha256(
        f"continual_decision_role_v1/{role.phase}/{role.seed}/final_test".encode("ascii")
    )
    for sample_id in role.sample_ids:
        digest.update(sample_id.encode("ascii"))
        digest.update(b"\0")
    for values in (role.input, role.target):
        canonical = np.ascontiguousarray(values, dtype="<f8")
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def capture_final_role(role: FinalRole, expected_count: int) -> FinalRoleFacts:
    _require_fields(role, expected_count)
    if (
        type(role.sha256) is not str
        or len(role.sha256) != 64
        or any(item not in "0123456789abcdef" for item in role.sha256)
        or final_content_digest(role) != role.sha256
    ):
        raise ValueError("confirmation final role content digest differs")
    return FinalRoleFacts(role.phase, role.seed, role.sample_ids, role.sha256, expected_count)


def validate_endpoint_result(result: EndpointResult, expected_count: int) -> float | None:
    if type(result) is not EndpointResult or type(expected_count) is not int or expected_count <= 0:
        raise ValueError("confirmation endpoint result/count type differs")
    if result.failure is None:
        if type(result.correct_count) is not int or not 0 <= result.correct_count <= expected_count:
            raise ValueError("confirmation endpoint result correct count differs")
        return result.correct_count / expected_count
    failure = result.failure
    if (
        result.correct_count is not None
        or type(failure) is not EndpointFailure
        or type(failure.code) is not str
        or (failure.error_type is not None and type(failure.error_type) is not str)
        or (failure.code, failure.error_type)
        not in {
            ("nonfinite_predictions", None),
            ("numerical_prediction_error", "FloatingPointError"),
        }
    ):
        raise ValueError("confirmation endpoint result failure policy differs")
    return None
