"""Read arrived TRAIN gradients and realized AdamW steps on native parameters.

Inputs are the learner's already validated CPU float32 trainable map and owned
permanent cost ledger. Outputs are temporary parameter copies and scalar group
reports. No model execution, randomness, data selection, frozen-base copy,
optimizer mutation, filesystem access or checkpoint ownership lives here.
"""

from math import fsum, isfinite, sqrt

DEFINITION = "open_jev_train_group_update_health_v1"
GROUPS = ("head", "lora_A", "lora_B")


def _group(name):
    if name.startswith("head."):
        return "head"
    for group in GROUPS[1:]:
        if f".{group}." in name:
            return group
    raise ValueError("update health received a non-native trainable parameter")


def _squared_norm(value):
    # Why: accumulate float32 coordinates in float64 so diagnostic squaring
    # does not overflow for finite large gradients or lose tiny realized steps.
    norm = float(value.detach().double().norm().item())
    if not isfinite(norm):
        raise ValueError("nonfinite native update-health norm")
    return norm * norm


def capture_update_health(parameters, work):
    """Own only pre-optimizer trainables; charge attempts before each operation."""
    work["update_health_attempts"] += 1
    before = {}
    groups: dict[str, dict[str, int | float]] = {
        group: dict(
            parameter_tensors=0,
            parameter_elements=0,
            gradient_tensors=0,
            missing_gradient_tensors=0,
        )
        for group in GROUPS
    }
    squares: dict[str, list[float]] = {group: [] for group in GROUPS}
    for name, parameter in parameters.items():
        group = _group(name)
        row = groups[group]
        elements = parameter.numel()
        row["parameter_tensors"] += 1
        row["parameter_elements"] += elements
        work["parameter_copy_elements_attempted"] += elements
        before[name] = parameter.detach().clone()
        if parameter.grad is None:
            row["missing_gradient_tensors"] += 1
        else:
            row["gradient_tensors"] += 1
            work["gradient_reduction_elements_attempted"] += elements
            squares[group].append(_squared_norm(parameter.grad))
    for group, row in groups.items():
        row["gradient_l2_before_clip"] = sqrt(fsum(squares[group]))
    return before, groups


def finish_update_health(parameters, before, groups, work):
    """Measure actual post-AdamW deltas, including moments and weight decay."""
    squares: dict[str, list[float]] = {group: [] for group in GROUPS}
    for name, parameter in parameters.items():
        work["step_reduction_elements_attempted"] += parameter.numel()
        delta = parameter.detach().double() - before[name].double()
        squares[_group(name)].append(_squared_norm(delta))
    for group, row in groups.items():
        row["parameter_step_l2"] = sqrt(fsum(squares[group]))
    work["update_health_reports_completed"] += 1
    return dict(
        definition=DEFINITION,
        groups=groups,
        work=dict(work),
        scope="arrived_TRAIN_update_only",
        arithmetic="float64_reductions_of_float32_gradients_and_steps",
    )
