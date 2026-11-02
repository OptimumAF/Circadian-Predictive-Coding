"""Scalar BP TRAIN-update observations from consecutive owned snapshots.

No learner construction, restore, forward/backward, fitting or IO. Snapshot
validation/copies and scalar reductions are additional diagnostic work. These
norms locate large updates; they do not establish the cause of task failure.
"""

from math import hypot, isfinite

from src.adapters import torch_feature_learner as native


def summarize_bp_update(before, after, *, rows):
    """Read actual gradients/momentum/deltas without applying another update."""
    if type(rows) is not int or not 1 <= rows <= 128:
        raise ValueError("BP update diagnostics require 1..128 TRAIN rows")
    before, after = native._snapshot(before), native._snapshot(after)
    if (
        before.policy.method != "bp"
        or before.policy != after.policy
        or after.completed_updates != before.completed_updates + 1
        or after.training_examples != before.training_examples + rows
        or before.optimizer_state is None
        or after.optimizer_state is None
        or after.gradients is None
    ):
        raise ValueError("BP update diagnostics require consecutive identical-policy snapshots")
    groups = {}
    parameter_count = 0
    for group, indices in (("hidden", (0, 1)), ("output", (2, 3))):
        old, new, gradients = [], [], []
        old_momentum: list[float] = []
        new_momentum: list[float] = []
        for index in indices:
            name = native._PARAMETERS[index]
            old.extend(before.head_state[name].reshape(-1).tolist())
            new.extend(after.head_state[name].reshape(-1).tolist())
            gradients.extend(after.gradients[name].reshape(-1).tolist())
            for state, destination in ((before, old_momentum), (after, new_momentum)):
                assert state.optimizer_state is not None
                buffer = state.optimizer_state["state"].get(index)
                destination.extend(
                    [0.0] * state.head_state[name].numel()
                    if buffer is None
                    else buffer["momentum_buffer"].reshape(-1).tolist()
                )
        delta = [b - a for a, b in zip(old, new, strict=True)]
        direction = new_momentum if after.policy.momentum > 0 else gradients
        rate = after.policy.learning_rate
        values = dict(
            parameter_values=len(old),
            parameter_before_l2=hypot(*old),
            parameter_after_l2=hypot(*new),
            gradient_l2=hypot(*gradients),
            momentum_before_l2=hypot(*old_momentum),
            momentum_after_l2=hypot(*new_momentum),
            parameter_delta_l2=hypot(*delta),
            nominal_sgd_delta_l2=rate * hypot(*direction),
            sgd_max_abs_residual=max(
                abs(d + rate * m) for d, m in zip(delta, direction, strict=True)
            ),
        )
        if not all(isfinite(v) for v in values.values()):
            raise ValueError("nonfinite BP update diagnostics")
        groups[group] = values
        parameter_count += len(old)
    return dict(
        definition="bp_train_update_norms_v1",
        before_updates=before.completed_updates,
        after_updates=after.completed_updates,
        rows=rows,
        groups=groups,
        work=dict(snapshot_validations=2, scalar_values_read=5 * parameter_count),
    )
