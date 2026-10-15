"""Independent complete-snapshot wire oracle and bounded continuation controls.

Encoding is a representation boundary, not recovery authority. Explicit NumPy
fixtures test byte preservation without constructing or training native models.
"""

from base64 import b64decode, b64encode
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
import sys
from typing import Any, Literal

import numpy as np
import pytest

import src.adapters.coupled_checkpoint_codec as codec_module
import src.adapters.numpy_checkpoint_frames as native_frames
from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
    _plasticity_gates,
)
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_predictive_coding import (
    CoupledPredictiveCodingNetwork,
    CoupledSnapshot,
    CoupledWakeState,
)

BINDING = CodecBinding("a" * 64, "b" * 64)
LIMITS = CodecLimits(32768, 8192, 8, 64)


def snapshot(kind="active", order="C"):
    parameters = CoupledParameters(
        (
            np.array(np.arange(12).reshape(3, 4) / 20 - 0.2, dtype=np.float64, order=order),
            np.array(np.arange(8).reshape(4, 2) / 20 - 0.1, dtype=np.float64, order=order),
        ),
        (np.array([[-0.0, 0.1, -0.2, 0.3]]), np.array([[0.2, -0.0]])),
        np.array([[0.3], [-0.2]]),
        np.array([[-0.0]]),
    )
    if kind == "pc":
        wake = None
    else:
        config = CoupledCircadianConfig(min_plasticity=1.0 if kind == "neutral" else 0.2)
        chemicals = np.array([0.1, 0.4])
        count = 7
        if kind == "zero":
            chemicals = np.zeros(2)
            count = 0
        elif kind == "saturated":
            config = CoupledCircadianConfig(plasticity_sensitivity=1e6, min_plasticity=1e-6)
            chemicals = np.full(2, 1e308)
        wake = CoupledWakeState(
            config.chemical_decay,
            config.chemical_buildup_rate,
            config.plasticity_sensitivity,
            config.min_plasticity,
            chemicals,
            _plasticity_gates(chemicals, config),
            count,
        )
    return CoupledSnapshot(1, parameters, (1e-6, 1e6), wake)


def arrays(value):
    p = value.parameters
    result = (*p.hidden_weights, *p.hidden_biases, p.output_weight, p.output_bias)
    return (
        result
        if value.wake_state is None
        else (*result, value.wake_state.chemical_state, value.wake_state.plasticity_state)
    )


def frame(value):
    order = "C" if value.flags.c_contiguous else "F"
    return dict(
        dtype="<f8",
        shape=list(value.shape),
        order=order,
        data=b64encode(value.tobytes(order=order)).decode("ascii"),
    )


def wire(value):
    p = value.parameters
    wake = value.wake_state
    return dict(
        codec_version=1,
        kind="coupled_pc" if wake is None else "coupled_cpc",
        binding=asdict(BINDING),
        snapshot_version=value.format_version,
        input_dim=p.hidden_weights[0].shape[0],
        hidden_dims=[w.shape[1] for w in p.hidden_weights],
        precisions=list(value.precisions),
        arrays=dict(
            hidden_weights=[frame(w) for w in p.hidden_weights],
            hidden_biases=[frame(b) for b in p.hidden_biases],
            output_weight=frame(p.output_weight),
            output_bias=frame(p.output_bias),
        ),
        wake_state=None
        if wake is None
        else dict(
            chemical_decay=wake.chemical_decay,
            chemical_buildup_rate=wake.chemical_buildup_rate,
            plasticity_sensitivity=wake.plasticity_sensitivity,
            min_plasticity=wake.min_plasticity,
            wake_updates=wake.wake_updates,
            chemical_state=frame(wake.chemical_state),
            plasticity_state=frame(wake.plasticity_state),
        ),
    )


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def encode(value=None, *, binding=BINDING, limits=LIMITS):
    return CoupledCheckpointCodec().encode(
        snapshot() if value is None else value, binding=binding, limits=limits
    )


def decode(raw, *, binding=BINDING, limits=LIMITS, digest=None):
    return CoupledCheckpointCodec().decode(
        raw,
        binding=binding,
        expected_sha256=sha256(raw).hexdigest() if digest is None else digest,
        limits=limits,
    )


def assert_same(actual, expected):
    assert type(actual) is CoupledSnapshot and type(actual.parameters) is CoupledParameters
    assert type(actual.format_version) is int and actual.format_version == expected.format_version
    assert type(actual.precisions) is tuple and actual.precisions == expected.precisions
    assert (actual.wake_state is None) == (expected.wake_state is None)
    if expected.wake_state is not None:
        assert actual.wake_state is not None
        for name in (
            "chemical_decay",
            "chemical_buildup_rate",
            "plasticity_sensitivity",
            "min_plasticity",
            "wake_updates",
        ):
            assert getattr(actual.wake_state, name) == getattr(expected.wake_state, name)
        assert type(actual.wake_state.wake_updates) is int
    for left, right in zip(arrays(actual), arrays(expected), strict=True):
        assert type(left) is np.ndarray and left.dtype == np.float64 and left.shape == right.shape
        assert left.flags.owndata and not np.shares_memory(left, right)
        order: Literal["C", "F"] = "C" if right.flags.c_contiguous else "F"
        assert left.tobytes(order=order) == right.tobytes(order=order)
        assert bool(left.flags.c_contiguous) == bool(right.flags.c_contiguous)
        assert bool(left.flags.f_contiguous) == bool(right.flags.f_contiguous)


@pytest.mark.parametrize("kind", ("pc", "zero", "active", "neutral", "saturated"))
@pytest.mark.parametrize("order", ("C", "F"))
def test_should_match_independent_complete_wire_bytes_and_own_all_storage(kind, order):
    source = snapshot(kind, order)
    before = deepcopy(source)
    expected = canonical(wire(source))
    raw = encode(source)
    assert type(raw) is bytes and raw == expected
    restored = decode(expected)
    assert_same(restored, source)
    assert encode(restored) == raw
    np.testing.assert_equal(asdict(source), asdict(before))
    assert np.signbit(restored.parameters.output_bias).all()
    for array in arrays(restored):
        array[:] = 999
    assert encode(source) == raw and encode(decode(raw)) == raw


def test_should_independently_decode_column_major_frames_and_canonical_singleton_tags():
    body = wire(snapshot(order="F"))
    raw = encode(snapshot(order="F"))
    assert json.loads(raw) == body
    for group in ("hidden_weights", "hidden_biases"):
        for descriptor in body["arrays"][group]:
            decoded = np.frombuffer(
                b64decode(descriptor["data"], validate=True), dtype="<f8"
            ).reshape(descriptor["shape"], order=descriptor["order"])
            assert descriptor["data"] == b64encode(
                decoded.tobytes(order=descriptor["order"])
            ).decode("ascii")
    assert all(f["order"] == "F" for f in body["arrays"]["hidden_weights"])
    assert all(f["order"] == "C" for f in body["arrays"]["hidden_biases"])
    assert body["arrays"]["output_weight"]["order"] == "C"
    assert body["wake_state"]["chemical_state"]["order"] == "C"


@pytest.mark.parametrize("kind", ("pc", "zero", "active", "neutral", "saturated"))
def test_should_restore_native_semantics_without_any_codec_model_construction(monkeypatch, kind):
    source = snapshot(kind)
    native_pc = CoupledPredictiveCodingNetwork.__init__
    native_cpc = CoupledCircadianPredictiveCodingNetwork.__init__
    with monkeypatch.context() as local:

        def forbidden(*args, **kwargs):
            pytest.fail("codec constructed a native learner")

        local.setattr(CoupledPredictiveCodingNetwork, "__init__", forbidden)
        local.setattr(CoupledCircadianPredictiveCodingNetwork, "__init__", forbidden)
        restored = decode(encode(source))
    assert CoupledPredictiveCodingNetwork.__init__ is native_pc
    assert CoupledCircadianPredictiveCodingNetwork.__init__ is native_cpc
    model = make_model(restored)
    model.restore_state(restored)
    assert_same(model.snapshot_state(), restored)


def make_model(state):
    p = state.parameters
    dims = tuple(w.shape[1] for w in p.hidden_weights)
    if state.wake_state is None:
        return CoupledPredictiveCodingNetwork(
            p.hidden_weights[0].shape[0], dims, 23, state.precisions
        )
    wake = state.wake_state
    policy = CoupledCircadianConfig(
        wake.chemical_decay,
        wake.chemical_buildup_rate,
        wake.plasticity_sensitivity,
        wake.min_plasticity,
    )
    return CoupledCircadianPredictiveCodingNetwork(
        p.hidden_weights[0].shape[0], dims, 23, state.precisions, config=policy
    )


@pytest.mark.parametrize(
    "fault",
    (
        "family",
        "version_bool",
        "version",
        "parameters",
        "weights_list",
        "empty",
        "bias_count",
        "topology",
        "dtype",
        "shape",
        "nan",
        "inf",
        "strided",
        "precisions_list",
        "precision_bool",
        "precision_low",
        "precision_high",
        "wake_family",
        "wake_count_bool",
        "wake_count_negative",
        "policy_bool",
        "policy_range",
        "chemical_negative",
        "chemical_shape",
        "gates",
        "zero_wake",
    ),
)
def test_should_refuse_incomplete_or_invalid_native_snapshot_without_mutation(fault):
    value: Any = snapshot()
    p = value.parameters
    if fault == "family":
        value = asdict(value)
    elif fault == "version_bool":
        value = replace(value, format_version=True)
    elif fault == "version":
        value = replace(value, format_version=2)
    elif fault == "parameters":
        value = replace(value, parameters=asdict(p))
    elif fault == "weights_list":
        value = replace(value, parameters=replace(p, hidden_weights=list(p.hidden_weights)))
    elif fault == "empty":
        value = replace(value, parameters=replace(p, hidden_weights=()))
    elif fault == "bias_count":
        value = replace(value, parameters=replace(p, hidden_biases=p.hidden_biases[:1]))
    elif fault == "topology":
        value = replace(
            value, parameters=replace(p, hidden_weights=(p.hidden_weights[0], np.ones((3, 2))))
        )
    elif fault in ("dtype", "shape", "nan", "inf"):
        bad = {
            "dtype": np.zeros((1, 1), dtype=np.float32),
            "shape": np.zeros((1, 2)),
            "nan": np.full((1, 1), np.nan),
            "inf": np.full((1, 1), np.inf),
        }[fault]
        value = replace(value, parameters=replace(p, output_bias=bad))
    elif fault == "strided":
        value = replace(
            value,
            parameters=replace(p, hidden_weights=(np.ones((3, 8))[:, ::2], p.hidden_weights[1])),
        )
    elif fault == "precisions_list":
        value = replace(value, precisions=list(value.precisions))
    elif fault.startswith("precision_"):
        beta = {"precision_bool": True, "precision_low": 1e-7, "precision_high": 1e6 + 1}[fault]
        value = replace(value, precisions=(beta, 1.0))
    elif fault == "wake_family":
        value = replace(value, wake_state=asdict(value.wake_state))
    else:
        wake = value.wake_state
        name, bad = {
            "wake_count_bool": ("wake_updates", True),
            "wake_count_negative": ("wake_updates", -1),
            "policy_bool": ("chemical_decay", True),
            "policy_range": ("min_plasticity", 0),
            "chemical_negative": ("chemical_state", np.array([-1.0, 0.0])),
            "chemical_shape": ("chemical_state", np.zeros((1, 2))),
            "gates": ("plasticity_state", np.ones(2)),
            "zero_wake": ("wake_updates", 0),
        }[fault]
        value = replace(value, wake_state=replace(wake, **{name: bad}))
    with pytest.raises(ValueError):
        encode(value)


@pytest.mark.parametrize(
    "group,field",
    (
        ("snapshot", "format_version"),
        ("snapshot", "parameters"),
        ("snapshot", "precisions"),
        ("snapshot", "wake_state"),
        ("parameters", "hidden_weights"),
        ("parameters", "hidden_biases"),
        ("parameters", "output_weight"),
        ("parameters", "output_bias"),
        ("wake", "chemical_decay"),
        ("wake", "chemical_buildup_rate"),
        ("wake", "plasticity_sensitivity"),
        ("wake", "min_plasticity"),
        ("wake", "chemical_state"),
        ("wake", "plasticity_state"),
        ("wake", "wake_updates"),
    ),
)
def test_should_refuse_missing_native_record_fields_with_value_error(group, field):
    value = snapshot()
    record = (
        value
        if group == "snapshot"
        else value.parameters
        if group == "parameters"
        else value.wake_state
    )
    object.__delattr__(record, field)
    with pytest.raises(ValueError):
        encode(value)


@pytest.mark.parametrize("group", ("snapshot", "parameters", "wake"))
def test_should_refuse_uninitialized_native_records_without_attribute_error(group):
    value = snapshot()
    if group == "snapshot":
        value = object.__new__(CoupledSnapshot)
    elif group == "parameters":
        value = replace(value, parameters=object.__new__(CoupledParameters))
    else:
        value = replace(value, wake_state=object.__new__(CoupledWakeState))
    with pytest.raises(ValueError):
        encode(value)


@pytest.mark.parametrize(
    "fault",
    (
        "extra",
        "missing",
        "codec_bool",
        "snapshot_bool",
        "kind",
        "kind_wake",
        "input_bool",
        "hidden_bool",
        "hidden_empty",
        "hidden_shape",
        "precision_bool",
        "precision_count",
        "precision_range",
        "array_extra",
        "array_missing",
        "weight_count",
        "dtype",
        "shape_bool",
        "shape",
        "order",
        "singleton_f",
        "base64",
        "base64_noncanonical",
        "bytes",
        "nan",
        "wake_extra",
        "wake_missing",
        "wake_count_bool",
        "wake_negative",
        "policy_bool",
        "policy_range",
        "chemical_negative",
        "gates",
        "zero_wake",
    ),
)
def test_should_refuse_malformed_wire_with_independently_matching_digest(fault):
    body = wire(snapshot())
    descriptor = body["arrays"]["output_bias"]
    if fault == "extra":
        body["future"] = None
    elif fault == "missing":
        del body["precisions"]
    elif fault in ("codec_bool", "snapshot_bool"):
        body["codec_version" if fault == "codec_bool" else "snapshot_version"] = True
    elif fault == "kind":
        body["kind"] = "future"
    elif fault == "kind_wake":
        body["kind"] = "coupled_pc"
    elif fault == "input_bool":
        body["input_dim"] = True
    elif fault == "hidden_bool":
        body["hidden_dims"][0] = True
    elif fault == "hidden_empty":
        body["hidden_dims"] = []
    elif fault == "hidden_shape":
        body["hidden_dims"][0] = 5
    elif fault == "precision_bool":
        body["precisions"][0] = True
    elif fault == "precision_count":
        body["precisions"] = [1]
    elif fault == "precision_range":
        body["precisions"][0] = 0
    elif fault == "array_extra":
        body["arrays"]["future"] = descriptor
    elif fault == "array_missing":
        del body["arrays"]["output_bias"]
    elif fault == "weight_count":
        body["arrays"]["hidden_weights"].pop()
    elif fault == "dtype":
        descriptor["dtype"] = "<f4"
    elif fault in ("shape_bool", "shape"):
        descriptor["shape"] = [True, 1] if fault == "shape_bool" else [1, 2]
    elif fault == "order":
        descriptor["order"] = "A"
    elif fault == "singleton_f":
        descriptor["order"] = "F"
    elif fault == "base64":
        descriptor["data"] = "!" * len(descriptor["data"])
    elif fault == "base64_noncanonical":
        descriptor["data"] = descriptor["data"][:-2] + "B="
    elif fault == "bytes":
        descriptor["data"] = "AA=="
    elif fault == "nan":
        descriptor["data"] = b64encode(np.array([[np.nan]]).tobytes()).decode("ascii")
    elif fault == "wake_extra":
        body["wake_state"]["future"] = None
    elif fault == "wake_missing":
        del body["wake_state"]["chemical_decay"]
    else:
        name, bad = {
            "wake_count_bool": ("wake_updates", True),
            "wake_negative": ("wake_updates", -1),
            "policy_bool": ("chemical_decay", True),
            "policy_range": ("min_plasticity", 0),
            "chemical_negative": ("chemical_state", frame(np.array([-1.0, 0.0]))),
            "gates": ("plasticity_state", frame(np.ones(2))),
            "zero_wake": ("wake_updates", 0),
        }[fault]
        body["wake_state"][name] = bad
    with pytest.raises(ValueError):
        decode(canonical(body))


@pytest.mark.parametrize(
    "raw", (b"{}", b"\xff", b"{", b'{"codec_version":1,"codec_version":1}', b"[" * 1500)
)
def test_should_refuse_invalid_utf8_parser_depth_and_duplicate_keys(raw):
    with pytest.raises(ValueError):
        decode(raw)


@pytest.mark.parametrize("representation", ("space", "indent", "duplicate", "binding_extra"))
def test_should_refuse_noncanonical_or_duplicate_complete_envelopes(representation):
    body = wire(snapshot())
    raw = canonical(body)
    if representation == "space":
        raw += b" "
    elif representation == "indent":
        raw = json.dumps(body, indent=2).encode()
    elif representation == "duplicate":
        raw = b'{"codec_version":1,' + raw[1:]
    else:
        body["binding"]["future"] = "c" * 64
        raw = canonical(body)
    with pytest.raises(ValueError):
        decode(raw)


@pytest.mark.parametrize("field", ("source_sha256", "policy_sha256"))
def test_should_refuse_wrong_independently_supplied_binding(field):
    raw = canonical(wire(snapshot()))
    with pytest.raises(ValueError):
        decode(raw, binding=replace(BINDING, **{field: "c" * 64}))


def test_should_refuse_corruption_against_independently_expected_digest():
    raw = canonical(wire(snapshot()))
    with pytest.raises(ValueError):
        decode(raw + b" ", digest=sha256(raw).hexdigest())


@pytest.mark.parametrize(
    "fault",
    (
        "raw_bytearray",
        "raw_string",
        "binding",
        "limits",
        "digest",
        "mutated_binding",
        "mutated_limits",
    ),
)
def test_should_revalidate_exact_public_controls_on_every_call(fault):
    raw: Any = canonical(wire(snapshot()))
    binding: Any = deepcopy(BINDING)
    limits: Any = deepcopy(LIMITS)
    digest = sha256(raw).hexdigest()
    if fault == "raw_bytearray":
        raw = bytearray(raw)
    elif fault == "raw_string":
        raw = raw.decode()
    elif fault == "binding":
        binding = asdict(binding)
    elif fault == "limits":
        limits = asdict(limits)
    elif fault == "digest":
        digest = "A" * 64
    elif fault == "mutated_binding":
        object.__setattr__(binding, "source_sha256", True)
    else:
        object.__setattr__(limits, "max_layers", True)
    with pytest.raises(ValueError):
        CoupledCheckpointCodec().decode(raw, binding=binding, expected_sha256=digest, limits=limits)
    if fault not in ("raw_bytearray", "raw_string", "digest"):
        with pytest.raises(ValueError):
            encode(snapshot(), binding=binding, limits=limits)


def test_should_accept_exact_inclusive_complete_resource_caps():
    source = snapshot(order="F")
    raw = canonical(wire(source))
    limits = CodecLimits(len(raw), sum(a.nbytes for a in arrays(source)), 2, 4)
    assert encode(source, limits=limits) == raw
    assert_same(decode(raw, limits=limits), source)


@pytest.mark.parametrize(
    "field", ("max_encoded_bytes", "max_array_bytes", "max_layers", "max_dimension")
)
def test_should_refuse_all_resource_excess_before_any_encoding_or_array_decode(monkeypatch, field):
    source = snapshot()
    before = asdict(source)
    raw = canonical(wire(source))
    exact = CodecLimits(len(raw), sum(a.nbytes for a in arrays(source)), 2, 4)
    small = replace(exact, **{field: getattr(exact, field) - 1})

    def forbidden(*args, **kwargs):
        pytest.fail("expensive array operation before resource admission")

    with monkeypatch.context() as local:
        local.setattr(codec_module, "encode_frame", forbidden)
        local.setattr(codec_module, "decode_frame", forbidden)
        local.setattr(native_frames, "b64decode", forbidden)
        local.setattr(np, "isfinite", forbidden)
        local.setattr(np, "frombuffer", forbidden)
        with pytest.raises(ValueError):
            encode(source, limits=small)
        with pytest.raises(ValueError):
            decode(raw, limits=small)
    np.testing.assert_equal(asdict(source), before)


@pytest.mark.parametrize("fault", ("last_shape", "late_alphabet", "late_pad_bits"))
def test_should_preflight_every_frame_before_decoding_even_the_first_array(monkeypatch, fault):
    body = wire(snapshot())
    if fault == "last_shape":
        body["wake_state"]["plasticity_state"]["shape"] = [3]
    elif fault == "late_alphabet":
        descriptor = body["wake_state"]["plasticity_state"]
        descriptor["data"] = "!" * len(descriptor["data"])
    else:
        descriptor = body["arrays"]["output_bias"]
        descriptor["data"] = descriptor["data"][:-2] + "B="
    raw = canonical(body)

    def forbidden(*args, **kwargs):
        pytest.fail("earlier array decoded before late frame admission")

    monkeypatch.setattr(codec_module, "decode_frame", forbidden)
    monkeypatch.setattr(native_frames, "b64decode", forbidden)
    monkeypatch.setattr(np, "frombuffer", forbidden)
    with pytest.raises(ValueError):
        decode(raw)


def test_should_refuse_huge_native_count_without_changing_global_integer_policy():
    source = snapshot()
    source = replace(source, wake_state=replace(source.wake_state, wake_updates=1 << 100000))
    configured = sys.get_int_max_str_digits()
    with pytest.raises(ValueError):
        encode(source, limits=replace(LIMITS, max_encoded_bytes=1024))
    assert sys.get_int_max_str_digits() == configured
    # A native count above int64 remains supported when its decimal representation
    # fits the caller's limits; no unrelated counter cap may truncate state.
    ordinary = replace(snapshot(), wake_state=replace(snapshot().wake_state, wake_updates=2**80))
    assert decode(encode(ordinary)).wake_state.wake_updates == 2**80


@pytest.mark.parametrize("family", ("pc", "neutral", "active"))
@pytest.mark.parametrize("guarded", (False, True))
def test_should_continue_complete_native_state_exactly_after_restore(family, guarded):
    policy = CoupledInferencePolicy() if guarded else None
    config = (
        CoupledCircadianConfig.matched_pc_control()
        if family == "neutral"
        else CoupledCircadianConfig()
    )
    source = (
        CoupledPredictiveCodingNetwork(2, (3, 2), 23, (0.8, 1.2))
        if family == "pc"
        else CoupledCircadianPredictiveCodingNetwork(2, (3, 2), 23, (0.8, 1.2), config=config)
    )
    x, y = np.array([[0.2, -0.3], [0.7, 0.1]]), np.array([[0.0], [1.0]])
    options = {} if policy is None else {"inference_policy": policy}
    for _ in range(2):
        source.train_epoch(x, y, 0.02, 2, 0.05, **options)
    saved = source.snapshot_state()
    raw = encode(saved)
    restored_state = decode(raw)
    restored = make_model(restored_state)
    restored.restore_state(restored_state)
    assert_same(restored.snapshot_state(), saved)
    np.testing.assert_array_equal(source.predict_proba(x), restored.predict_proba(x))
    for array in arrays(restored_state):
        array[:] = 999
    assert encode(saved) == raw
    left = source.train_epoch(x, y, 0.02, 2, 0.05, **options)
    right = restored.train_epoch(x, y, 0.02, 2, 0.05, **options)
    assert left == right
    assert_same(restored.snapshot_state(), source.snapshot_state())
    if family != "pc":
        assert restored.wake_updates == 3
        assert np.any(restored.get_chemical_state() > 0)
        if family == "neutral":
            np.testing.assert_array_equal(restored.get_plasticity_state(), np.ones(2))
