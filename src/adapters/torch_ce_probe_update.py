"""Detached CE update proposals for the existing disposable-probe boundary.

Only the current role-checked batch and retained TRAIN anchors reach fitting.
No input owner is updated, no probe-label replay bank is retained, and no SGD
credit or publication authority is granted. Callers own arrival, budgets and
application to an already private learner after their original guards pass.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
from math import isfinite

import numpy as np

from src.adapters import torch_categorical_ce_readout as ce
from src.adapters import torch_feature_learner as native
from src.adapters.torch_array_weighted import _ownership, _rng
from src.adapters.torch_categorical_readout_fit import _unchanged_owner
from src.adapters.torch_decision import _same_values
from src.app.coupled_attribution import AttributionBatch, _same_value, _validate_batch
from src.core.categorical_probability_scores import validate_class_targets
from src.core.probe_update import ProbeUpdateFailed, ProbeUpdateReceipt
from src.shared.torch_runtime import require_torch

DEFINITION = "disposable_probe_ce_rehearsal_update_v1"


class CEProbeUpdateFailed(RuntimeError):
    """No state proposal returned; all attempted auxiliary work remains paid."""

    def __init__(self, reason, work):
        super().__init__(reason)
        self.work = deepcopy(work)


@dataclass(frozen=True)
class CEProbeUpdate:
    definition: str
    phase: str
    before: native.TorchFeatureSnapshot
    state: native.TorchFeatureSnapshot
    coefficients: tuple[tuple[float, ...], ...]
    current_rows: int
    retained_rows: int
    closure_losses: tuple[float, ...]
    final_loss: float
    work: dict[str, int]


def _inputs(current, retained, phase):
    if type(phase) is not str or phase not in ("fresh", "return"):
        raise ValueError("CE probe phase must be fresh or return")
    if type(retained) is not tuple or not 1 <= len(retained) <= 127:
        raise ValueError("CE probe requires a bounded tuple of retained TRAIN batches")
    expected = "probe_train" if phase == "fresh" else "train"
    # Check ALL roles before reading any matrix, including a foreign held-out one.
    for batch, role in ((current, expected), *((b, "train") for b in retained)):
        if type(batch) is not AttributionBatch or type(batch.role) is not str or batch.role != role:
            raise ValueError("CE probe requires current " + expected + " and retained train roles")
    _validate_batch(current, expected, set())
    seen: set[str] = set()
    for batch in retained:
        _validate_batch(batch, "train", seen)
    batches = (current, *retained)
    if sum(len(b.features) for b in batches) > 128:
        raise ValueError("CE probe combined rows exceed128")
    for batch in batches:
        validate_class_targets(batch.targets, len(batch.features), batch.num_classes)
        if (
            batch.num_classes != current.num_classes
            or batch.features.shape[1] != current.features.shape[1]
        ):
            raise ValueError("CE probe batch geometries differ")
    anchor_rows = {name: (b, i) for b in retained for i, name in enumerate(b.row_ids)}
    for i, name in enumerate(current.row_ids):
        if name not in anchor_rows:
            continue
        anchor, j = anchor_rows[name]
        if (
            phase != "return"
            or anchor.context != current.context
            or not np.array_equal(current.features[i], anchor.features[j])
            or not np.array_equal(current.targets[i], anchor.targets[j])
        ):
            raise ValueError("CE probe row IDs collide across distinct roles or facts")
    return len(current.features), sum(len(b.features) for b in retained)


def _bindings(current, retained):
    return tuple(
        (id(b), id(b.features), id(b.targets), id(b.row_ids), id(b.episode), id(b.packet_episodes))
        for b in (current, *retained)
    )


def _role_work(work, phase, current_rows, retained_rows):
    calls = work["backward_attempts"]
    work.update(
        native_sgd_updates=0,
        probe_gradient_rows=current_rows * calls if phase == "fresh" else 0,
        current_train_gradient_rows=current_rows * calls if phase == "return" else 0,
        retained_train_gradient_rows=retained_rows * calls,
    )


def _derived(before, coefficients):
    policy = before.policy
    if (
        type(coefficients) is not tuple
        or len(coefficients) != policy.hidden_dim + 1
        or any(
            type(row) is not tuple
            or len(row) != policy.num_classes
            or any(
                type(v) is not float
                or not isfinite(v)
                or abs(v) > np.finfo(np.float32).max
                or float(np.float32(v)) != v
                for v in row
            )
            for row in coefficients
        )
    ):
        raise ValueError("CE probe coefficients must be immutable finite float32 values")
    torch = require_torch()
    values = torch.tensor(coefficients, dtype=torch.float32, device="cpu")
    head = deepcopy(before.head_state)
    head["weight_hidden_output"] = values[:-1].clone()
    head["bias_output"] = values[-1:].clone()
    # Auxiliary fitting changes no original native SGD, gradients or momentum.
    return native._snapshot(replace(before, head_state=head))


def copy_ce_probe_update(update):
    """Revalidate a proposal, not arrival, publication or source-restore authority."""
    if (
        type(update) is not CEProbeUpdate
        or type(update.definition) is not str
        or update.definition != DEFINITION
    ):
        raise ValueError("an exact CE probe update is required")
    if (
        type(update.phase) is not str
        or update.phase not in ("fresh", "return")
        or any(type(n) is not int or n < 1 for n in (update.current_rows, update.retained_rows))
        or update.current_rows + update.retained_rows > 128
    ):
        raise ValueError("CE probe update role counts or phase differ")
    before = native._snapshot(update.before)
    if before.policy.method != "bp":
        raise ValueError("CE probe updates require BP")
    expected = _derived(before, update.coefficients)
    if not _same_values(native._snapshot(update.state), expected):
        raise ValueError("CE probe derived state or native SGD credit changed")
    work = update.work
    if (
        type(work) is not dict
        or any(type(k) is not str for k in work)
        or any(type(v) is not int or v < 0 for v in work.values())
    ):
        raise ValueError("CE probe work must be exact nonnegative counters")
    calls = work.get("closure_attempts", 0)
    rows = update.current_rows + update.retained_rows
    expected_work = ce._new_ce_work()
    for key in (
        "hidden_attempts",
        "hidden_returns",
        "optimizer_constructions",
        "step_attempts",
        "step_returns",
        "final_read_attempts",
        "final_read_returns",
    ):
        expected_work[key] = 1
    for key in (
        "closure_attempts",
        "forward_attempts",
        "forward_returns",
        "backward_attempts",
        "backward_returns",
    ):
        expected_work[key] = calls
    expected_work.update(
        hidden_rows=rows,
        final_read_rows=rows,
        forward_rows=rows * calls,
        backward_rows=rows * calls,
    )
    _role_work(expected_work, update.phase, update.current_rows, update.retained_rows)
    if (
        not 1 <= calls <= 64
        or work != expected_work
        or type(update.closure_losses) is not tuple
        or len(update.closure_losses) != calls
        or any(type(v) is not float or not isfinite(v) or v < 0 for v in update.closure_losses)
        or type(update.final_loss) is not float
        or not isfinite(update.final_loss)
        or update.final_loss < 0
    ):
        raise ValueError("CE probe fit or role work is incomplete or changed")
    return deepcopy(replace(update, before=before, state=expected))


def fit_ce_probe_update(learner, current, retained_train, *, phase="fresh", check_boundary=None):
    """Fit one owned proposal; the supplied learner and batches never change.

    Fresh current rows are probe_train; return current rows are TRAIN. Anchors
    are always TRAIN. Same-fact TRAIN return duplicates remain explicitly paid.
    No optimizer or input list is stored in a successful proposal; no probe-label
    replay bank is maintained.
    """
    current_rows, retained_rows = _inputs(current, retained_train, phase)
    if check_boundary is not None and not callable(check_boundary):
        raise ValueError("CE probe boundary must be callable")
    if ce.MAX_CLOSURES != 64:
        raise ValueError("fixed CE readout closure limit changed")
    work = ce._new_ce_work()
    _role_work(work, phase, current_rows, retained_rows)
    try:
        with _unchanged_owner(learner) as (state, unchanged):
            if (
                state.policy.feature_dim != current.features.shape[1]
                or state.policy.num_classes != current.num_classes
            ):
                raise ValueError("CE probe data differs from native policy")
            binding = _bindings(current, retained_train)
            pin = deepcopy((current, retained_train))
            features = targets = arrays_pin = None

            def boundary():
                unchanged()
                if check_boundary is not None:
                    check_boundary()
                unchanged()
                _inputs(current, retained_train, phase)
                if _bindings(current, retained_train) != binding or not _same_value(
                    (current, retained_train), pin
                ):
                    raise ValueError("CE probe input ownership or values changed")
                if arrays_pin is not None and (
                    not np.array_equal(features, arrays_pin[0])
                    or not np.array_equal(targets, arrays_pin[1])
                ):
                    raise ValueError("CE probe detached input arrays changed")

            boundary()
            features = np.concatenate([b.features for b in (*retained_train, current)])
            targets = np.concatenate([b.targets for b in (*retained_train, current)])
            arrays_pin = (features.copy(), targets.copy())
            boundary()
            coefficients, _, losses, final_loss = ce._fit_output_arrays(
                state, features, targets, work, boundary
            )
            _role_work(work, phase, current_rows, retained_rows)
            derived = _derived(state, coefficients)
            boundary()
            result = copy_ce_probe_update(
                CEProbeUpdate(
                    DEFINITION,
                    phase,
                    deepcopy(state),
                    derived,
                    coefficients,
                    current_rows,
                    retained_rows,
                    losses,
                    final_loss,
                    deepcopy(work),
                )
            )
            boundary()
            return result
    except Exception as error:
        _role_work(work, phase, current_rows, retained_rows)
        raise CEProbeUpdateFailed(f"{type(error).__name__}: {error}", work) from error


def apply_ce_probe_update(learner, current, retained_train, *, phase, check_boundary):
    """Existing-runner callback: fit and apply only to its already private owner.

    The caller owns private-fork/arrival authority and external clocks. The fit
    keeps the owner unchanged; application rechecks that exact before-state and
    ownership after the caller's guard. No rollback hides a foreign mutation.
    """
    work = dict(
        native_sgd_updates=0, application_attempts=0, application_returns=0, application_verified=0
    )
    work.update(fit_attempts=0, fit_returns=0, fit_work_available=0)
    applied = False
    loss = None

    def actual():
        return native._state(
            learner._head,
            learner._optimizer,
            native._policy(learner._policy),
            learner.completed_updates,
            learner.training_examples,
        )

    try:
        with _unchanged_owner(learner) as (before, unchanged):
            binding = _ownership(learner)
            rng = _rng(require_torch())
            work["fit_attempts"] += 1
            try:
                result = fit_ce_probe_update(
                    learner, current, retained_train, phase=phase, check_boundary=check_boundary
                )
            except CEProbeUpdateFailed as error:
                work.update(error.work)
                work["fit_work_available"] = 1
                raise
            work["fit_returns"] += 1
            work.update(result.work)
            work["fit_work_available"] = 1
            update = copy_ce_probe_update(result)
            pin = deepcopy(update)
            loss = update.final_loss
            check_boundary()
            unchanged()
            if not _same_values(before, update.before) or not _same_values(update, pin):
                raise ValueError("CE probe proposal differs from private owner before-state")
        # No external call between this final owner check and private application.
        if _ownership(learner) != binding or not _same_values(actual(), before):
            raise ValueError("CE probe private owner changed before application")
        work["application_attempts"] += 1
        learner.restore_state(update.state)
        work["application_returns"] += 1
        if not _same_values(actual(), update.state):
            raise ValueError("CE probe private application differs from derived state")
        applied = True
        work["application_verified"] += 1
        after_binding = (_ownership(learner), id(learner._policy), id(learner._payload))
        check_boundary()
        if (
            not _same_values(actual(), update.state)
            or after_binding != (_ownership(learner), id(learner._policy), id(learner._payload))
            or not _same_values(_rng(require_torch()), rng)
        ):
            raise ValueError("CE probe private owner or RNG changed after application")
        return ProbeUpdateReceipt(DEFINITION, work, applied, loss)
    except Exception as error:
        if type(error) is CEProbeUpdateFailed:
            work.update(error.work)
        raise ProbeUpdateFailed(
            f"{type(error).__name__}: {error}",
            ProbeUpdateReceipt(DEFINITION, work, applied, loss),
        ) from error
