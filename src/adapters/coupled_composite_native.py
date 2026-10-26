"""Borrow complete coupled fields for managed capture, without native operations.

Exact learner/model schemas retain stored geometry and policy as well as arrays.
Metadata bounds precede finite/wake scans. These projections neither copy payloads
nor encode, construct models, restore ownership or grant recovery authority.
"""

from dataclasses import fields
from types import MappingProxyType
from typing import Any

import numpy as np

from src.adapters.coupled_checkpoint_codec import (
    _native_array_metadata,
    _native_structure,
    _validate_wake,
)
from src.adapters.coupled_learners import CoupledLearner
from src.adapters.numpy_composite_capture import _native_limits
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_learner_state import CoupledUpdatePolicy
from src.core.coupled_predictive_coding import (
    CoupledPredictiveCodingNetwork,
    CoupledSnapshot,
    CoupledWakeState,
)
from src.core.managed_composite_state import CompositeCaptureLimits, NativeProjection, SourceRecord
from src.core.managed_lifecycle_state import AuthorityReference


def _kind(kind: type[Any]) -> str:
    return kind.__module__ + "." + kind.__name__


LEARNER_FIELDS = frozenset(
    {
        "_model",
        "_learning_rate",
        "_inference_steps",
        "_inference_learning_rate",
        "_inference_policy",
    }
)
_PC_FIELDS = frozenset({"_input_dim", "_hidden_dims", "_precisions", "_parameters"})
_CPC_FIELDS = _PC_FIELDS | {"_config", "_chemical_state", "_plasticity_state", "_wake_updates"}
MODEL_SCHEMAS = MappingProxyType(
    {
        _kind(CoupledPredictiveCodingNetwork): _PC_FIELDS,
        _kind(CoupledCircadianPredictiveCodingNetwork): frozenset(_CPC_FIELDS),
    }
)
LEARNER_KIND = _kind(CoupledLearner)


def _exact_record(value: Any, kind: type[Any]) -> None:
    if type(value) is not kind or vars(value).keys() != {field.name for field in fields(kind)}:
        raise ValueError(f"complete coupled capture requires exact {kind.__name__} fields")


def validate_coupled_snapshot(
    snapshot: CoupledSnapshot,
    limits: CompositeCaptureLimits,
    *,
    topology: tuple[int, tuple[int, ...]] | None = None,
) -> int:
    """Validate borrowed full native state, including CPC policy/gate consistency."""
    input_dim, hidden_dims, _, wake, shapes, values, policy = _native_structure(
        snapshot, _native_limits(limits)
    )
    if topology is not None and (input_dim, hidden_dims) != topology:
        raise ValueError("stored coupled model topology differs from native arrays")
    _native_array_metadata(shapes, values)
    for group in values.values():
        if any(not np.isfinite(value).all() for value in group):
            raise ValueError("nonfinite complete coupled native array")
    _validate_wake(values, wake, policy)
    return input_dim


def validate_coupled_model(record: SourceRecord, limits: CompositeCaptureLimits) -> int:
    """Validate every stored model field and bind explicit geometry to its arrays."""
    _exact_record(record, SourceRecord)
    if (
        type(record.kind) is not str
        or type(record.fields) is not tuple
        or any(
            type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str
            for pair in record.fields
        )
    ):
        raise ValueError("coupled native source requires exact named field tuples")
    spec = MODEL_SCHEMAS.get(record.kind)
    if spec is None or tuple(name for name, _ in record.fields) != tuple(sorted(spec)):
        raise ValueError("unsupported complete coupled model source schema")
    # Projected values remain dynamically typed until the structural validator
    # checks the complete native container and every array below.
    data: dict[str, Any] = dict(record.fields)
    input_dim, hidden_dims = data["_input_dim"], data["_hidden_dims"]
    if (
        type(input_dim) is not int
        or not 0 < input_dim <= limits.max_dimension
        or type(hidden_dims) is not tuple
        or not 1 <= len(hidden_dims) <= limits.max_depth
        or any(
            type(width) is not int or not 0 < width <= limits.max_dimension for width in hidden_dims
        )
    ):
        raise ValueError("stored coupled model topology exceeds exact native bounds")
    wake = None
    if "_config" in data:
        config = data["_config"]
        _exact_record(config, CoupledCircadianConfig)
        CoupledCircadianConfig.__post_init__(config)
        wake = CoupledWakeState(
            config.chemical_decay,
            config.chemical_buildup_rate,
            config.plasticity_sensitivity,
            config.min_plasticity,
            data["_chemical_state"],
            data["_plasticity_state"],
            data["_wake_updates"],
        )
    snapshot = CoupledSnapshot(1, data["_parameters"], data["_precisions"], wake)
    validate_coupled_snapshot(snapshot, limits, topology=(input_dim, hidden_dims))
    return input_dim


def validate_coupled_policy(data: dict[str, Any]) -> None:
    CoupledUpdatePolicy(
        data["_learning_rate"],
        data["_inference_steps"],
        data["_inference_learning_rate"],
        data["_inference_policy"],
    )


def project_coupled_native(
    learner: object, path: str, limits: CompositeCaptureLimits
) -> NativeProjection:
    if type(learner) is not CoupledLearner or vars(learner).keys() != LEARNER_FIELDS:
        raise ValueError("unsupported complete original CoupledLearner schema")
    data = vars(learner)
    validate_coupled_policy(data)
    model = data["_model"]
    if type(model) not in (CoupledPredictiveCodingNetwork, CoupledCircadianPredictiveCodingNetwork):
        raise ValueError("complete coupled learner requires an exact native PC/CPC model")
    spec = MODEL_SCHEMAS[_kind(type(model))]
    if vars(model).keys() != spec:
        raise ValueError("complete coupled native model fields changed")
    borrowed = SourceRecord(_kind(type(model)), tuple(sorted(vars(model).items())))
    validate_coupled_model(borrowed, limits)
    projected = SourceRecord(
        LEARNER_KIND,
        tuple(
            (name, borrowed if name == "_model" else data[name]) for name in sorted(LEARNER_FIELDS)
        ),
    )
    return NativeProjection(
        projected,
        (
            AuthorityReference(path + ".original", learner),
            AuthorityReference(path + "._model.original", model),
        ),
    )
