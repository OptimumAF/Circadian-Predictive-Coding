"""One owned factor reference for fixed two-state native consolidation.

Captures only vanilla LoRA factors and returns detached metadata. The reference
is private, bounded and version guarded; no labels, selection, loader, portable
checkpoint or independently trained task-delta assumption lives here.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

REFERENCE_DEFINITION = "open_jev_owned_adapter_reference_v1"
DEFINITION = "open_jev_owned_adapter_consolidation_v1"


@dataclass(frozen=True)
class _Layer:
    name: str
    metadata: tuple
    a: Any
    b: Any
    signatures: tuple


@dataclass(frozen=True)
class _Reference:
    layers: tuple[_Layer, ...]
    native_work: tuple


def new_consolidation_work():
    return dict.fromkeys(
        (
            "reference_attempts",
            "reference_copy_elements_attempted",
            "reference_rng_read_elements_attempted",
            "references_completed",
            "consolidation_attempts",
            "consolidations_completed",
        ),
        0,
    )


def _signature(value):
    return id(value), value._version


def _reference_rng(owner):
    owner._adapter_consolidation_work["reference_rng_read_elements_attempted"] += owner._rng.numel()
    return owner._torch.get_rng_state()


def _copy_factor(owner, parameter):
    owner._adapter_consolidation_work["reference_copy_elements_attempted"] += parameter.numel()
    copied = parameter.detach().clone()
    if not owner._torch.isfinite(copied).all().item():
        raise ValueError("cannot capture nonfinite native adapter factors")
    return copied


def _copy_layer(owner, binding):
    a, b = _copy_factor(owner, binding.a), _copy_factor(owner, binding.b)
    return _Layer(binding.name, binding.metadata, a, b, (_signature(a), _signature(b)))


def _require_authority(owner):
    if not owner.policy.max_adapter_consolidations:
        raise ValueError("adapter consolidation is disabled")
    if owner.adapter_projection_work["projection_attempts"] >= owner.policy.max_adapter_projections:
        raise ValueError("consolidation requires unspent projection authority")
    if owner.optimizer_reset_work["optimizer_reset_attempts"] >= owner.policy.max_optimizer_resets:
        raise ValueError("consolidation requires unspent reset authority")


def admit_reference_capture(owner):
    owner._guard()
    _require_authority(owner)
    if (
        owner._adapter_consolidation_work["reference_attempts"]
        or owner._adapter_reference is not None
    ):
        raise ValueError("adapter reference capture allowance exhausted")


def build_adapter_reference(owner):
    return _Reference(
        tuple(_copy_layer(owner, binding) for binding in owner._projection_bindings),
        tuple(owner.work.items()),
    )


def publish_adapter_reference(owner, reference):
    work = owner._adapter_consolidation_work
    elements = sum(layer.a.numel() + layer.b.numel() for layer in reference.layers)
    report = dict(
        definition=REFERENCE_DEFINITION,
        modules=[
            dict(module=b.name, allocated_rank=b.metadata[0], scaling=b.metadata[2])
            for b in reference.layers
        ],
        source_native_work=dict(reference.native_work),
        numeric_elements=elements,
        numeric_bytes=4 * elements,
        storage="single_owned_float32_factor_slot",
        adapter_consolidation_work={**work, "references_completed": 1},
    )
    saved = deepcopy(report)
    owner._adapter_reference = owner._adapter_reference_origin = reference
    owner._last_adapter_reference = saved
    work["references_completed"] += 1
    return report


def admit_adapter_reference(owner, reference):
    """Normal Tensor mutations/replacement refuse; hostile .data writes are outside the port."""
    if (
        type(reference) is not _Reference
        or reference is not owner._adapter_reference
        or reference is not owner._adapter_reference_origin
        or owner._adapter_consolidation_work["references_completed"] != 1
        or len(reference.layers) != len(owner._projection_bindings)
    ):
        raise ValueError("missing or replaced owned adapter reference")
    for layer, binding in zip(reference.layers, owner._projection_bindings):
        if (
            type(layer) is not _Layer
            or (layer.name, layer.metadata) != (binding.name, binding.metadata)
            or layer.signatures != (_signature(layer.a), _signature(layer.b))
        ):
            raise ValueError("owned adapter reference metadata or factor changed")
        for value, template in ((layer.a, binding.a), (layer.b, binding.b)):
            if (
                type(value) is not owner._torch.Tensor
                or value.requires_grad
                or value.device.type != "cpu"
                or value.dtype != owner._torch.float32
                or value.shape != template.shape
            ):
                raise ValueError("owned adapter reference tensor contract changed")


def admit_consolidation(owner):

    owner._guard()
    _require_authority(owner)
    if owner._adapter_consolidation_work["consolidation_attempts"]:
        raise ValueError("adapter consolidation allowance exhausted")
    reference = owner._adapter_reference
    admit_adapter_reference(owner, reference)
    return reference
