"""Default-off projection of one native owner's vanilla Linear LoRA updates.

Admits live bindings, stages all factors, consumes existing reset authority and
publishes into existing Parameters. Owns no model/data loader or training rule.
Paid failures poison the owner; diagnostics/counters describe history, not a
claim that a subsequently trained/restored model still has projected factors.
"""

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from src.adapters import open_jev_consolidation as consolidation
from src.adapters.open_jev_owned_fork import _admit_buffers
from src.core.learner_ports import TrainingUpdateFailed
from src.core.low_rank_composition import LowRankTerm, _admit, compose_low_rank

DEFINITION = "open_jev_owned_adapter_projection_v1"


@dataclass(frozen=True)
class _Binding:
    name: str
    module: Any
    a: Any
    b: Any
    base: Any
    metadata: tuple


def new_projection_work():
    return dict.fromkeys(
        (
            "projection_attempts",
            "input_copy_elements_attempted",
            "core_calls_attempted",
            "qr_calls_reserved",
            "svd_calls_reserved",
            "matmul_scalar_products_reserved",
            "workspace_elements_reserved",
            "staging_elements_attempted",
            "cast_elements_attempted",
            "parameter_write_elements_attempted",
            "gradient_buffers_cleared",
            "rng_copy_elements_attempted",
            "reset_rng_copy_elements_reserved",
            "projections_completed",
        ),
        0,
    )


def _module_metadata(owner, module):
    torch = owner._torch
    for field in ("lora_A", "lora_B", "lora_dropout"):
        mapping = getattr(module, field)
        if type(mapping) is not torch.nn.ModuleDict or tuple(mapping) != ("default",):
            raise ValueError("projection requires one default Linear adapter")
    for field in ("r", "lora_alpha", "scaling", "use_dora", "use_rslora", "lora_bias"):
        mapping = getattr(module, field)
        if type(mapping) is not dict or tuple(mapping) != ("default",):
            raise ValueError("projection adapter metadata differs from vanilla LoRA")
    if (
        module.fan_in_fan_out is not False
        or module._disable_adapters is not False
        or module.active_adapters != ["default"]
        or module.merged_adapters != []
        or module.lora_variant != {}
        or any(
            getattr(module, n)["default"] is not False
            for n in ("use_dora", "use_rslora", "lora_bias")
        )
        or type(module.lora_dropout["default"]) is not torch.nn.Identity
        or len(module.lora_embedding_A)
        or len(module.lora_embedding_B)
        or len(module.lora_magnitude_vector)
    ):
        raise ValueError("projection refuses transposed, merged, disabled or variant adapters")
    a, b, base = module.lora_A["default"], module.lora_B["default"], module.base_layer
    if (
        any(type(v) is not torch.nn.Linear for v in (a, b, base))
        or a.bias is not None
        or b.bias is not None
    ):
        raise ValueError("projection requires unquantized Linear factors without adapter bias")
    rank, alpha, scaling = (getattr(module, n)["default"] for n in ("r", "lora_alpha", "scaling"))
    if type(rank) is not int or rank != owner.policy.expected_lora_rank:
        raise ValueError("projection allocated rank differs from native policy")
    if any(type(v) not in (int, float) for v in (alpha, scaling)):
        raise ValueError("projection requires finite positive alpha and scaling")
    try:
        alpha, scaling = float(alpha), float(scaling)
    except OverflowError:
        raise ValueError("projection alpha/scaling exceeds finite float64 range") from None
    if any(not isfinite(v) or v <= 0 for v in (alpha, scaling)):
        raise ValueError("projection requires finite positive alpha and scaling")
    if scaling != alpha / rank:
        raise ValueError("projection scaling differs from vanilla alpha/r")
    outputs, inputs = base.weight.shape
    if (
        a.weight.shape != (rank, inputs)
        or b.weight.shape != (outputs, rank)
        or rank > min(inputs, outputs)
    ):
        raise ValueError("projection factor orientation or shape changed")
    return rank, float(alpha), float(scaling), outputs, inputs


def bind_projection_modules(owner):
    """Metadata only; binds live objects so deepcopy can rebind a fork correctly."""
    result = []
    covered: set[str] = set()
    try:
        for name, module in owner.model.named_modules():
            if not hasattr(module, "lora_A") and not hasattr(module, "lora_B"):
                continue
            metadata = _module_metadata(owner, module)
            a, b, base = (
                module.lora_A["default"].weight,
                module.lora_B["default"].weight,
                module.base_layer.weight,
            )
            a_name, b_name, base_name = (
                f"{name}.{n}"
                for n in ("lora_A.default.weight", "lora_B.default.weight", "base_layer.weight")
            )
            if (
                owner._trainable.get(a_name) is not a
                or owner._trainable.get(b_name) is not b
                or owner._named.get(base_name) is not base
                or base_name not in owner._base
            ):
                raise ValueError("projection factors/base differ from registered native owners")
            covered.update((a_name, b_name))
            result.append(_Binding(name, module, a, b, base, metadata))
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("unsupported native projection module metadata") from error
    expected = {n for n in owner._trainable if not n.startswith("head.")}
    if (
        not result
        or len(result) > 256
        or covered != expected
        or sum(v.a.numel() + v.b.numel() for v in result) > 5_000_000
    ):
        raise ValueError("projection must cover every bounded native LoRA factor")
    return tuple(result)


def check_projection_bindings(owner):
    current = bind_projection_modules(owner)
    saved = owner._projection_bindings
    if len(current) != len(saved) or any(
        a.name != b.name
        or a.metadata != b.metadata
        or any(getattr(a, n) is not getattr(b, n) for n in ("module", "a", "b", "base"))
        for a, b in zip(current, saved)
    ):
        raise ValueError("native projection module binding or metadata changed")


def _signature(tensor):
    return None if tensor is None else (id(tensor), tensor._version)


def _optimizer_signature(optimizer):
    return tuple(_signature(v) for state in optimizer.state.values() for v in state.values())


def _caller_rng(owner):
    owner._adapter_projection_work["rng_copy_elements_attempted"] += owner._rng.numel()
    return owner._torch.get_rng_state()


def _context(owner, *, rng_reader=_caller_rng):
    return dict(
        versions=owner._trainable_versions(),
        head=tuple(
            (n, _signature(p), _signature(p.grad))
            for n, p in owner._trainable.items()
            if n.startswith("head.")
        ),
        buffers=tuple((n, _signature(b)) for n, b in owner.model.named_buffers()),
        modes=tuple((n, m.training) for n, m in owner.model.named_modules()),
        private_rng=_signature(owner._rng),
        caller_rng=rng_reader(owner),
        native=owner.work,
        reset=owner.optimizer_reset_work,
        fork=owner.fork_work,
        health=owner.update_health_work,
        last_update=deepcopy(owner.last_update),
        optimizer=owner._optimizer,
        optimizer_versions=_optimizer_signature(owner._optimizer),
    )


def _assert_context(owner, before, *, published, rng_reader=_caller_rng):
    owner._guard()
    for key, value in dict(
        head=tuple(
            (n, _signature(p), _signature(p.grad))
            for n, p in owner._trainable.items()
            if n.startswith("head.")
        ),
        buffers=tuple((n, _signature(b)) for n, b in owner.model.named_buffers()),
        modes=tuple((n, m.training) for n, m in owner.model.named_modules()),
        private_rng=_signature(owner._rng),
        native=owner.work,
        fork=owner.fork_work,
        health=owner.update_health_work,
        last_update=owner.last_update,
    ).items():
        if value != before[key]:
            raise ValueError(f"projection changed protected native {key}")
    if _optimizer_signature(before["optimizer"]) != before[
        "optimizer_versions"
    ] or not owner._torch.equal(rng_reader(owner), before["caller_rng"]):
        raise ValueError("projection changed prior moments or caller RNG")
    expected_reset = {k: v + int(published) for k, v in before["reset"].items()}
    if owner.optimizer_reset_work != expected_reset:
        raise ValueError("projection changed reset work outside its exact allowance")
    if published:
        if (
            owner._optimizer is before["optimizer"]
            or owner._optimizer.state
            or any(
                p.grad is not None for n, p in owner._trainable.items() if not n.startswith("head.")
            )
        ):
            raise ValueError("projected coordinates retained stale moments or gradients")
    elif (
        owner._optimizer is not before["optimizer"]
        or owner._trainable_versions() != before["versions"]
    ):
        raise ValueError("projection staging changed live weights, gradients or optimizer")


def _terms(owner, binding, reference_layer=None, *, copy=False):
    pairs: tuple[tuple[Any, Any], ...] = ((binding.a, binding.b),)
    coefficient = 1.0
    if reference_layer is not None:
        pairs = ((reference_layer.a, reference_layer.b), *pairs)
        coefficient = 0.5
    terms = []
    for pair in pairs:
        arrays = []
        for parameter in pair:
            array = parameter.detach().numpy()
            if copy:
                owner._adapter_projection_work["input_copy_elements_attempted"] += parameter.numel()
                array = array.astype(np.float64, copy=True)
            arrays.append(array)
        terms.append(
            LowRankTerm(arrays[0], arrays[1], coefficient=coefficient, scaling=binding.metadata[2])
        )
    return tuple(terms)


def _plan(owner, effective_rank, reference_layers=()):
    if (
        type(effective_rank) is not int
        or not 0 <= effective_rank <= owner.policy.expected_lora_rank
    ):
        raise ValueError("effective_rank must be an integer within allocated native rank")
    _admit_buffers(owner)  # Refuse aliases before staging or paid work.
    plans = []
    for index, binding in enumerate(owner._projection_bindings):
        # Reuse core admission, avoiding a second workspace-formula implementation.
        layer = reference_layers[index] if reference_layers else None
        plans.append(_admit(_terms(owner, binding, layer), effective_rank, 20_000_000)[-1])
    if sum(p["workspace_elements_reserved"] for p in plans) > 100_000_000:
        raise ValueError("aggregate projection workspace allowance exceeded")
    return plans


def _cast_factor(owner, values):
    owner._adapter_projection_work["cast_elements_attempted"] += values.size
    with np.errstate(over="raise", invalid="raise"):
        copied = np.array(values, dtype=np.float32, copy=True)
    if not np.isfinite(copied).all():
        raise ValueError("projection factors cannot be represented in float32")
    return owner._torch.from_numpy(copied)


def _stage_module(owner, binding, effective_rank, plan, reference_layer=None):
    work = owner._adapter_projection_work
    terms = _terms(owner, binding, reference_layer, copy=True)
    work["core_calls_attempted"] += 1
    for key in ("qr_calls", "svd_calls", "matmul_scalar_products"):
        work[key + "_reserved"] += plan[key]
    result = compose_low_rank(terms, rank=effective_rank)
    work["staging_elements_attempted"] += binding.a.numel()
    a = binding.a.detach().clone()
    work["staging_elements_attempted"] += binding.b.numel()
    b = owner._torch.zeros_like(binding.b)
    # Retain original inactive A rows, zero inactive B: later B gradients can
    # reuse these directions, without a random initializer or forced rank cap.
    a[:effective_rank].copy_(_cast_factor(owner, result.a))
    b[:, :effective_rank].copy_(_cast_factor(owner, result.b / binding.metadata[2]))
    assert result.singular_values is not None
    diagnostic = dict(
        module=binding.name,
        scaling=binding.metadata[2],
        allocated_rank=binding.metadata[0],
        effective_rank=effective_rank,
        numerical_rank=result.numerical_rank,
        discarded_frobenius=result.discarded_frobenius,
        singular_values=result.singular_values.tolist(),
        core_work=result.work,
    )
    return binding, a, b, diagnostic


def _publish(owner, staged):
    work = owner._adapter_projection_work
    with owner._torch.no_grad():
        for binding, a, b, _ in staged:
            for parameter, value in ((binding.a, a), (binding.b, b)):
                work["parameter_write_elements_attempted"] += parameter.numel()
                parameter.copy_(value)
                if parameter.grad is not None:
                    work["gradient_buffers_cleared"] += 1
                parameter.grad = None


def capture_native_adapter_reference(owner):
    consolidation.admit_reference_capture(owner)
    _admit_buffers(owner)
    # Rank0 metadata admission ensures some two-term result fits before retaining
    # a reference. It refuses impossible total rank/core bounds without copies.
    _plan(owner, 0, owner._projection_bindings)
    work = owner._adapter_consolidation_work
    work["reference_attempts"] += 1
    try:
        before = _context(owner, rng_reader=consolidation._reference_rng)
        reference = consolidation.build_adapter_reference(owner)
        _assert_context(owner, before, published=False, rng_reader=consolidation._reference_rng)
        return consolidation.publish_adapter_reference(owner, reference)
    except Exception as error:
        owner._poisoned = True
        raise TrainingUpdateFailed(
            str(error),
            consolidation.REFERENCE_DEFINITION,
            {**owner.work, **owner.optimizer_reset_work, **owner.adapter_projection_work, **work},
        ) from error


def project_native_adapters(owner, effective_rank, *, reference=None):
    owner._guard()
    if reference is None and owner._adapter_reference is not None:
        raise ValueError("pending adapter reference reserves projection for consolidation")
    if reference is not None:
        consolidation.admit_adapter_reference(owner, reference)
    work = owner._adapter_projection_work
    if work["projection_attempts"] >= owner.policy.max_adapter_projections:
        raise ValueError("adapter projection allowance disabled or exhausted")
    if owner.optimizer_reset_work["optimizer_reset_attempts"] >= owner.policy.max_optimizer_resets:
        raise ValueError("adapter projection requires unspent explicit optimizer reset authority")
    plans = _plan(owner, effective_rank, () if reference is None else reference.layers)
    definition = DEFINITION if reference is None else consolidation.DEFINITION
    if reference is not None:
        owner._adapter_consolidation_work["consolidation_attempts"] += 1
    work["projection_attempts"] += 1
    work["workspace_elements_reserved"] += sum(p["workspace_elements_reserved"] for p in plans)
    try:
        before = _context(owner)
        staged = []
        for index, (binding, plan) in enumerate(zip(owner._projection_bindings, plans)):
            extra = {} if reference is None else {"reference_layer": reference.layers[index]}
            staged.append(_stage_module(owner, binding, effective_rank, plan, **extra))
        if reference is not None:
            consolidation.admit_adapter_reference(owner, reference)
        _assert_context(owner, before, published=False)
        work["reset_rng_copy_elements_reserved"] += 4 * owner._rng.numel()
        owner.reset_optimizer_state(
            **({} if reference is None else {"_adapter_reference": reference})
        )
        if reference is not None:
            consolidation.admit_adapter_reference(owner, reference)
        _publish(owner, staged)
        _assert_context(owner, before, published=True)
        if reference is not None:
            consolidation.admit_adapter_reference(owner, reference)
        report = dict(
            definition=definition,
            effective_rank=effective_rank,
            inactive_rows="original_A_zero_B_regrowth_allowed",
            modules=[s[3] for s in staged],
            native_work=owner.work,
            optimizer_reset_work=owner.optimizer_reset_work,
            adapter_projection_work={
                **work,
                "projections_completed": work["projections_completed"] + 1,
            },
        )
        if reference is not None:
            report.update(
                coefficients={"reference": 0.5, "current": 0.5},
                composition="complete_operators_current_already_contains_prior_history",
                reference_native_work=dict(reference.native_work),
                head_policy="preserve_current",
                adapter_consolidation_work={
                    **owner.adapter_consolidation_work,
                    "consolidations_completed": 1,
                },
            )
        saved = deepcopy(report)
        work["projections_completed"] += 1
        owner.last_adapter_projection = saved
        if reference is not None:
            owner._adapter_consolidation_work["consolidations_completed"] += 1
            owner._adapter_reference = owner._adapter_reference_origin = None
        return report
    except Exception as error:
        owner._poisoned = True
        raise TrainingUpdateFailed(
            str(error),
            definition,
            {
                **owner.work,
                **owner.optimizer_reset_work,
                **work,
                **({} if reference is None else owner.adapter_consolidation_work),
            },
        ) from error
