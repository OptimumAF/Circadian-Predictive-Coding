# Bounded low-rank composition

`src/core/low_rank_composition.py` adds pure algebra needed for future native
adapter consolidation. It does not change the native learner, loader, scalar
head, optimizer, data roles or default training. No task-retention, calibration
or circadian advantage follows from algebra correctness.

## Contract

`LowRankTerm(a,b,coefficient=1.,scaling=1.)` borrows quiescent real float32/64
arrays A[r,in],B[out,r]. The term is coefficient × scaling × B A in [out,in]
orientation. Callers establish common frozen-base identity and orientation;
matching shapes do not authenticate a shared base. Terms may have signed or
zero coefficients, different inner ranks, readonly storage or noncontiguous
views. Full metadata admission precedes value scans and allocation.

`compose_low_rank(tuple_of_terms,rank=None,max_elements=20_000_000)` returns
independently owned float64 factors A/B with scaling1. Exact mode vertically
stacks A and horizontally concatenates weighted B. It retains all factor
columns, including zero terms, and introduces no factorwise-average cross terms.
Exact retained_rank is stored factor width, not measured numerical rank;
singular_values/numerical_rank are None and discarded_frobenius is algebraic0.
Floating arithmetic still has roundoff, underflow and range limitations.

```python
import numpy as np
from src.core.low_rank_composition import LowRankTerm, compose_low_rank

term = LowRankTerm(np.eye(2), np.diag([3., 1.]), scaling=2.)
exact = compose_low_rank((term,))
projected = compose_low_rank((term,), rank=1)
# Use factors directly; forming B @ A is only suitable for tiny test oracles.
```

## Optional rank projection

Reduced QR gives B=Qb Rb and A.T=Qa Ra. SVD acts on Rb Ra.T, whose dimensions
are min(out,total_rank) × min(in,total_rank), at most64×64. Output factors are
Qb U_k sqrt(s_k) and sqrt(s_k) Vh_k Qa.T. This avoids constructing the original
full B A operator or frozen base. For tiny dimensions the core can have the same
shape as the operator; it remains expressed in the orthonormal QR bases.

`discarded_frobenius` is stable hypot of the discarded core singular values,
the optimal truncation residual in exact arithmetic. It is not a certificate of
rounded output error. `numerical_rank` counts s > eps64 × max(in,out) × s0;
small cancellation residuals can have positive relative numerical rank. Never
compare singular-vector signs across platforms. Requested rank0 measures the
spectrum and returns empty factors; total input rank0 skips all decompositions.
No automatic rank, mixture-weight, dose or outcome selection occurs.

## Bounds, failures and work

Input: exact tuple1..8/exact term/ndarray; positive in/out≤131072; totalrank≤64;
inputfactor elements E=R(in+out)≤5m. Native finite int/float coefficient/scaling
and their product only (bool/NumPy scalars rejected). max_elements integer1..20m
can tighten the default. Rank is None or integer0..min(R,in,out).

Explicit coordinate reserve is3E in exact mode, or8E+24R²+4(in+out)k+16R in
projection. It conservatively covers Python-visible factors/masks/decomposition
outputs and is admitted before scans/allocation. Hidden NumPy/LAPACK workspace,
allocator overhead, traffic, actual FLOPs and total RSS are not bounded by this
number. `work` records input/output sizes, reserve, QR shapes/calls, small core,
SVD calls and only the three explicit products' scalar multiplication counts.
No model work or permanent ledger is implied. An integrating owner must charge
attempts before calling and retain failed costs under its prospective budget.

Nonfinite inputs/intermediates, arithmetic overflow and failed decomposition
raise ValueError; no clipping, fallback or source mutation. All sources must be
quiescent. Returned arrays are owned and writable; dataclass freezing does not
make their contents immutable. Casting factors back to native float32 is a
future integration boundary with additional error; it is not certified here.

## Structure and checks

- `src/core/low_rank_composition.py`: pure factor algebra and admission/work data.
- `tests/test_low_rank_composition.py`: tiny literal operators, independent dense
  SVD oracles, refusal/ownership/failure behavior; no model or random fixtures.
- Native binding, same-base proof, fan-in-fan-out handling, head policy,
  installation and optimizer-state intervention belong outside this core.

```powershell
python -m pytest -q tests/test_low_rank_composition.py
python -m ruff check src/core/low_rank_composition.py tests/test_low_rank_composition.py
python -m ruff format --check src/core/low_rank_composition.py tests/test_low_rank_composition.py
python -m mypy --explicit-package-bases --follow-imports=silent src/core/low_rank_composition.py tests/test_low_rank_composition.py
```

[Prospective local qualification](../artifacts/runs/native-factor-composition-20261022/PROTOCOL.md).
No new dependencies/configuration/environment variables. Extend through an
explicit native-owner operation with prospective work/reset authority; never
silently mutate frozen base/head, retain stale moments after factor rebasing,
renew native allowances or use final/probe labels for consolidation selection.
Native integration and comparisons require separate frozen protocols, preserving
all accepted A/B/C baselines and their negative/zero-shot results.
