"""Copy a trusted native CPU learner while sharing only frozen parameters.

Inputs are an admitted live native owner and its permanent copy ledger. Output
is one disposable owner with inherited limits/history and separate mutable
state. No constructor, model execution, loader, data selection, budget renewal
or portable checkpoint lives here. Arbitrary hostile Python is outside this port.
"""

from copy import deepcopy
from typing import Any

from src.core.learner_ports import TrainingUpdateFailed

DEFINITION = "open_jev_owned_fork_v1"


def _tensor_signature(tensor):
    return None if tensor is None else (id(tensor), tensor._version)


def _source_signature(owner):
    return (
        tuple(
            (n, _tensor_signature(p), _tensor_signature(p.grad)) for n, p in owner._named.items()
        ),
        tuple((n, _tensor_signature(b)) for n, b in owner.model.named_buffers()),
        tuple((n, m.training) for n, m in owner.model.named_modules()),
        tuple(_tensor_signature(v) for s in owner._optimizer.state.values() for v in s.values()),
        _tensor_signature(owner._rng),
        owner.work,
        owner.optimizer_reset_work,
        owner.update_health_work,
        deepcopy(owner.last_update),
        owner.adapter_projection_work,
        deepcopy(owner.last_adapter_projection),
        owner.adapter_consolidation_work,
        deepcopy(owner.last_adapter_reference),
        owner._adapter_reference is None,
    )


def _admit_buffers(owner):
    buffers = dict(owner.model.named_buffers())
    if (
        sum(b.numel() for b in buffers.values()) > 5_000_000
        or sum(b.numel() * b.element_size() for b in buffers.values()) > 40 * 1024**2
        or any(b.is_meta or b.device.type != "cpu" for b in buffers.values())
    ):
        raise ValueError("native fork requires bounded materialized CPU buffers")
    if any(p.grad is not None for n, p in owner._named.items() if n not in owner._trainable):
        raise ValueError("shared frozen native parameters must have no gradients")
    parameter_ids = {id(p) for p in owner._named.values()}
    if any(id(b) in parameter_ids for b in buffers.values()):
        raise ValueError("native fork refuses parameter/buffer aliases")
    storage: dict[int, Any] = {}
    for tensor in (*owner._named.values(), *buffers.values()):
        if tensor.numel():
            key = tensor.untyped_storage().data_ptr()
            if key in storage and storage[key] is not tensor:
                raise ValueError("native fork refuses distinct tensors sharing registered storage")
            storage[key] = tensor
    return buffers


def _copy_parameters(owner, memo):
    torch, work = owner._torch, owner._fork_work
    for name, parameter in owner._named.items():
        if name not in owner._trainable:
            memo[id(parameter)] = parameter
            continue
        work["parameter_copy_elements_attempted"] += parameter.numel()
        copied = torch.nn.Parameter(parameter.detach().clone(), requires_grad=True)
        memo[id(parameter)] = copied
        if parameter.grad is not None:
            work["gradient_copy_elements_attempted"] += parameter.grad.numel()
            copied.grad = parameter.grad.detach().clone()
            memo[id(parameter.grad)] = copied.grad


def _validate_fork(owner, copied, buffers):
    if copied is owner or copied.model is owner.model or copied._optimizer is owner._optimizer:
        raise ValueError("native fork reused a mutable owner")
    modules = dict(copied.model.named_modules())
    if any(modules[n] is m for n, m in owner.model.named_modules()):
        raise ValueError("native fork reused a mutable module")
    for name, parameter in copied.model.named_parameters():
        origin = owner._named[name]
        if name in owner._trainable:
            if (
                parameter is not copied._trainable[name]
                or parameter.data_ptr() == origin.data_ptr()
            ):
                raise ValueError("native fork reused mutable parameter storage")
        elif parameter is not origin:
            raise ValueError("native fork copied the frozen base")
    cloned_buffers = dict(copied.model.named_buffers())
    if cloned_buffers.keys() != buffers.keys() or any(
        cloned_buffers[n] is b or (b.numel() and cloned_buffers[n].data_ptr() == b.data_ptr())
        for n, b in buffers.items()
    ):
        raise ValueError("native fork reused mutable buffer storage")
    copied._guard()


def fork_native_learner(owner):
    """Charge a single attempt, preserve full native history, never reinitialize."""
    buffers = _admit_buffers(owner)
    work = owner._fork_work
    work["fork_attempts"] += 1
    torch = owner._torch
    try:
        signature = _source_signature(owner)
        work["rng_copy_elements_attempted"] += owner._rng.numel()
        caller_rng = torch.get_rng_state()
        memo = {id(torch): torch}
        _copy_parameters(owner, memo)
        for buffer in buffers.values():
            work["buffer_copy_elements_attempted"] += buffer.numel()
            memo[id(buffer)] = buffer.detach().clone()
        work["rng_copy_elements_attempted"] += owner._rng.numel()
        memo[id(owner._rng)] = owner._rng.clone()
        # Why: deepcopy's parameter memo binds optimizer groups/state to the
        # copied trainables without constructors or resetting native moments.
        work["optimizer_copy_elements_attempted"] += sum(
            v.numel() for state in owner._optimizer.state.values() for v in state.values()
        )
        copied = deepcopy(owner, memo)
        owner._guard()
        work["rng_copy_elements_attempted"] += owner._rng.numel()
        if signature != _source_signature(owner) or not torch.equal(
            caller_rng, torch.get_rng_state()
        ):
            raise ValueError("native fork changed source state or caller RNG")
        _validate_fork(owner, copied, buffers)
        work["forks_completed"] += 1
        copied._fork_work = dict(work)
        return copied
    except Exception as error:
        owner._poisoned = True
        raise TrainingUpdateFailed(str(error), DEFINITION, {**owner.work, **work}) from error
