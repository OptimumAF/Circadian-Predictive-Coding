"""Bounded sums and optional rank projection of same-base low-rank updates.

Inputs borrow quiescent A[r,in], B[out,r] arrays; each term represents
coefficient * scaling * B @ A. Outputs own float64 factors with scaling one.
The caller establishes common base identity and orientation. This module has
no learner, labels, optimizer, IO, native binding or dense base/operator path.
Workspace reserves cover explicit arrays, not hidden LAPACK workspace or RSS.
"""

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LowRankTerm:
    a: np.ndarray
    b: np.ndarray
    coefficient: float = 1.0
    scaling: float = 1.0


@dataclass(frozen=True)
class LowRankComposition:
    a: np.ndarray
    b: np.ndarray
    input_rank: int
    retained_rank: int
    singular_values: np.ndarray | None
    numerical_rank: int | None
    discarded_frobenius: float
    work: dict[str, Any]


def _finite_weight(term):
    values = (term.coefficient, term.scaling)
    if any(type(v) not in (int, float) for v in values):
        raise ValueError("coefficient and scaling must be native finite numbers")
    try:
        coefficient, scaling = (float(v) for v in values)
    except OverflowError:
        raise ValueError("coefficient or scaling exceeds finite float64 range") from None
    weight = coefficient * scaling
    if not all(isfinite(v) for v in (coefficient, scaling, weight)):
        raise ValueError("coefficient, scaling and their product must be finite")
    return weight


def _dimensions(term):
    if type(term) is not LowRankTerm:
        raise ValueError("expected exact LowRankTerm records")
    a, b = term.a, term.b
    if any(type(v) is not np.ndarray for v in (a, b)):
        raise ValueError("factors must be exact NumPy ndarrays")
    if any(
        v.ndim != 2 or v.dtype not in (np.dtype("float32"), np.dtype("float64")) for v in (a, b)
    ):
        raise ValueError("factors must be two-dimensional float32 or float64 arrays")
    r, inputs = a.shape
    outputs, other_rank = b.shape
    if r != other_rank or not 1 <= inputs <= 131072 or not 1 <= outputs <= 131072:
        raise ValueError("incompatible factor ranks or unsupported input/output dimension")
    return outputs, inputs, r


def _work_plan(outputs, inputs, total_rank, terms, rank):
    elements = total_rank * (inputs + outputs)
    projected = rank is not None and total_rank > 0
    retained = total_rank if rank is None else rank
    left, right = min(outputs, total_rank), min(inputs, total_rank)
    # Why this: reserve explicit factors, QR/core/SVD outputs and finite masks
    # before touching values. LAPACK's internal workspace remains unmeasured.
    reserve = (
        3 * elements
        if not projected
        else (
            8 * elements + 24 * total_rank**2 + 4 * (inputs + outputs) * retained + 16 * total_rank
        )
    )
    return dict(
        term_count=len(terms),
        input_rank=total_rank,
        input_elements=elements,
        output_factor_elements=(inputs + outputs) * retained,
        workspace_elements_reserved=reserve,
        qr_calls=2 if projected else 0,
        qr_shapes=((outputs, total_rank), (inputs, total_rank)) if projected else (),
        svd_calls=1 if projected else 0,
        core_shape=(left, right) if projected else None,
        matmul_scalar_products=(
            left * total_rank * right + outputs * left * retained + retained * right * inputs
        )
        if projected
        else 0,
    )


def _admit(terms, rank, max_elements):
    if type(terms) is not tuple or not 1 <= len(terms) <= 8:
        raise ValueError("expected one to eight terms in an exact tuple")
    if type(max_elements) is not int or not 1 <= max_elements <= 20_000_000:
        raise ValueError("max_elements must be an integer in 1..20000000")
    shapes = tuple(_dimensions(term) for term in terms)
    outputs, inputs, _ = shapes[0]
    total_rank = sum(s[2] for s in shapes)
    if any(s[:2] != (outputs, inputs) for s in shapes):
        raise ValueError(
            "all terms must share input/output dimensions and a caller-established base"
        )
    if total_rank > 64 or total_rank * (inputs + outputs) > 5_000_000:
        raise ValueError("input factor rank or element allowance exceeded")
    if rank is not None and (
        type(rank) is not int or not 0 <= rank <= min(total_rank, inputs, outputs)
    ):
        raise ValueError("rank must be None or an integer within the available rank/dimensions")
    weights = tuple(_finite_weight(term) for term in terms)
    work = _work_plan(outputs, inputs, total_rank, terms, rank)
    if work["workspace_elements_reserved"] > max_elements:
        raise ValueError("explicit workspace element allowance exceeded")
    return outputs, inputs, total_rank, weights, work


def _require_finite(*arrays):
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("low-rank factors or intermediate results are nonfinite")


def _stack_factors(terms, weights, outputs, inputs, total_rank):
    a = np.empty((total_rank, inputs), dtype=np.float64)
    b = np.empty((outputs, total_rank), dtype=np.float64)
    offset = 0
    for term, weight in zip(terms, weights):
        stop = offset + term.a.shape[0]
        a[offset:stop] = term.a
        np.multiply(term.b, weight, out=b[:, offset:stop], dtype=np.float64)
        offset = stop
    _require_finite(a, b)
    return a, b


def _project(a, b, rank):
    qb, rb = np.linalg.qr(b, mode="reduced")
    qa, ra = np.linalg.qr(a.T, mode="reduced")
    _require_finite(qb, rb, qa, ra)
    core = rb @ ra.T
    _require_finite(core)
    u, singular, vh = np.linalg.svd(core, full_matrices=False)
    _require_finite(u, singular, vh)
    # Balance factors without forming the dense out x in operator.
    root = np.sqrt(singular[:rank])
    projected_b = qb @ u[:, :rank]
    projected_a = vh[:rank] @ qa.T
    projected_b *= root
    projected_a *= root[:, None]
    _require_finite(projected_a, projected_b)
    return projected_a.copy(order="C"), projected_b.copy(order="C"), singular.copy()


def compose_low_rank(terms, *, rank=None, max_elements=20_000_000) -> LowRankComposition:
    """Compose weighted products; optionally return their best rank-k projection.

    Exact mode retains concatenated rank and skips decompositions. Projected
    discarded_frobenius is the small-core singular tail, not a certificate of
    rounded output error. Numerical rank uses eps64 * max(in,out) * largest s.
    Validation and resource admission precede finite scans and allocations.
    Failed work has no retained ledger here: an integrating owner must charge
    attempts before calling this pure operation. Source arrays are never written.
    """
    outputs, inputs, total_rank, weights, work = _admit(terms, rank, max_elements)
    for term in terms:
        _require_finite(term.a, term.b)
    try:
        with np.errstate(over="raise", invalid="raise"):
            a, b = _stack_factors(terms, weights, outputs, inputs, total_rank)
            singular = None
            numerical_rank = None
            discarded = 0.0
            if rank is not None:
                if total_rank:
                    a, b, singular = _project(a, b, rank)
                    threshold = np.finfo(np.float64).eps * max(outputs, inputs) * float(singular[0])
                    numerical_rank = int(np.count_nonzero(singular > threshold))
                    discarded = hypot(*(float(v) for v in singular[rank:]))
                else:
                    singular = np.empty(0, dtype=np.float64)
                    numerical_rank = 0
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError(f"low-rank composition failed: {error}") from error
    if not isfinite(discarded):
        raise ValueError("discarded Frobenius norm exceeds finite float64 range")
    return LowRankComposition(
        a=a,
        b=b,
        input_rank=total_rank,
        retained_rank=total_rank if rank is None else rank,
        singular_values=singular,
        numerical_rank=numerical_rank,
        discarded_frobenius=discarded,
        work=work,
    )
