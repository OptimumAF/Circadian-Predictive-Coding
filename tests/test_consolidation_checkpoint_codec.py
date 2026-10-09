"""Complete consolidation observations; synthetic history performs no native work."""

from copy import deepcopy
from dataclasses import fields, replace
from hashlib import sha256
import json
import math
from typing import Any

import pytest

from src.adapters.consolidation_checkpoint_codec import ConsolidationCheckpointCodec
from src.core.actor_ports import AppliedConsolidation
from src.core.checkpoint_codec import CheckpointCodec, CodecBinding, CodecLimits
from src.core.consolidation_codec_policy import ConsolidationCodecPolicy
from src.core.consolidation_cursor import ConsolidationCursor
from src.core.learner_ports import TrainingDiagnostic

BINDING = CodecBinding(sha256(b"source").hexdigest(), sha256(b"policy").hexdigest())
AUTHORITY = sha256(b"original authority reference").hexdigest()
POLICY = ConsolidationCodecPolicy(4, 128)
LIMITS = CodecLimits(16384, 1, 1, 1)


@pytest.fixture(scope="session", autouse=True)
def record_zero_native_work(tmp_path_factory):
    ledger = tmp_path_factory.mktemp("native-work") / "native-work.json"
    ledger.write_text(
        json.dumps(dict(updates=0, restores=0, sleeps=0, structural=0, predictions=0, workers=0))
    )


def saved(*, value=-0.0, alias=True, stopped=False, retired=False, ready=True):
    diagnostic = TrainingDiagnostic("native-definition", value)
    second = diagnostic if alias else replace(diagnostic)
    return ConsolidationCursor(
        1,
        "actor",
        "learner",
        4,
        ("a-applied", "b-applied", "c-failed", "d-failed"),
        (
            AppliedConsolidation("b-applied", "actor", "learner", 2, diagnostic),
            AppliedConsolidation("a-applied", "actor", "learner", 4, second),
        ),
        stopped,
        retired,
        13,
        ready,
    )


def codec(policy=POLICY, authority=AUTHORITY) -> CheckpointCodec[ConsolidationCursor]:
    return ConsolidationCheckpointCodec(policy, authority_sha256=authority)


def encode(state=None, *, limits=LIMITS):
    return codec().encode(saved() if state is None else state, binding=BINDING, limits=limits)


def decode(raw, *, limits=LIMITS, binding=BINDING, decoder=None):
    return (codec() if decoder is None else decoder).decode(
        raw, binding=binding, expected_sha256=sha256(raw).hexdigest(), limits=limits
    )


def raw_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@pytest.mark.parametrize("value", [-0.0, 0.0, 0, -7, 1e100, 2**100, -(2**100)])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize(
    "stopped,retired,ready",
    [(a, b, c) for a in (False, True) for b in (False, True) for c in (False, True)],
)
def test_should_preserve_every_record_consumed_attempt_flag_and_diagnostic(
    value, alias, stopped, retired, ready
):
    original = saved(value=value, alias=alias, stopped=stopped, retired=retired, ready=ready)
    before = deepcopy(original)
    raw = encode(original)
    detached = decode(raw)
    assert detached == original == before and detached is not original
    assert encode(detached) == raw
    assert detached.attempted_ids == ("a-applied", "b-applied", "c-failed", "d-failed")
    for left, right in zip(original.consolidations, detached.consolidations, strict=True):
        assert left is not right and left.diagnostic is not right.diagnostic
        assert type(left.diagnostic.value) is type(right.diagnostic.value)
        if type(value) is float:
            assert math.copysign(1, right.diagnostic.value) == math.copysign(1, value)
    assert (detached.consolidations[0].diagnostic is detached.consolidations[1].diagnostic) is alias


@pytest.mark.parametrize("limit", [0, 4])
def test_should_preserve_empty_original_zero_or_nonzero_quota(limit):
    empty = ConsolidationCursor(1, "actor", "learner", limit, (), (), False, False, 0, True)
    component = codec(ConsolidationCodecPolicy(limit, 128))
    raw = component.encode(empty, binding=BINDING, limits=LIMITS)
    assert decode(raw, decoder=component) == empty


NATIVE_FIELDS = [
    (kind, field.name)
    for kind in (ConsolidationCursor, AppliedConsolidation, TrainingDiagnostic)
    for field in fields(kind)
]


@pytest.mark.parametrize("kind,name", NATIVE_FIELDS)
def test_should_refuse_every_missing_native_record_field(kind, name):
    state = saved()
    record = state if kind is ConsolidationCursor else state.consolidations[0]
    if kind is TrainingDiagnostic:
        record = record.diagnostic
    object.__delattr__(record, name)
    with pytest.raises(ValueError):
        encode(state)


@pytest.mark.parametrize("kind", [ConsolidationCursor, AppliedConsolidation, TrainingDiagnostic])
def test_should_refuse_unknown_native_fields(kind):
    state = saved()
    record = state if kind is ConsolidationCursor else state.consolidations[0]
    if kind is TrainingDiagnostic:
        record = record.diagnostic
    object.__setattr__(record, "future", 1)
    with pytest.raises(ValueError):
        encode(state)


def at_path(value, path):
    for key in path:
        value = value[key]
    return value


PATHS = [
    (),
    ("binding",),
    ("policy",),
    ("cursor",),
    ("cursor", "consolidations", 0),
    ("cursor", "consolidations", 0, "diagnostic"),
]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_should_refuse_incomplete_or_unknown_nested_wire_schemas(path, mutation):
    value = json.loads(encode())
    record = at_path(value, path)
    if mutation == "missing":
        del record[next(iter(record))]
    else:
        record["future"] = 1
    with pytest.raises(ValueError):
        decode(raw_json(value))


CORRUPTIONS = [
    (("codec_version",), True),
    (("kind",), "future"),
    (("authority_sha256",), "0" * 64),
    (("cursor", "format_version"), 2),
    (("cursor", "actor_version"), "learner"),
    (("cursor", "learner_version"), "other"),
    (("cursor", "consolidation_limit"), 5),
    (("cursor", "attempted_ids"), ["b", "a"]),
    (("cursor", "attempted_ids"), ["a", "a"]),
    (("cursor", "stopped"), 1),
    (("cursor", "retired"), 0),
    (("cursor", "payload_ready"), 1),
    (("cursor", "revision"), -1),
    (("cursor", "revision"), True),
    (("cursor", "consolidations", 0, "event_id"), "missing"),
    (("cursor", "consolidations", 0, "actor_version"), "other"),
    (("cursor", "consolidations", 0, "learner_version"), "other"),
    (("cursor", "consolidations", 0, "attempt_number"), 0),
    (("cursor", "consolidations", 0, "attempt_number"), 5),
    (("cursor", "consolidations", 1, "attempt_number"), 2),
    (("cursor", "consolidations", 1, "event_id"), "b-applied"),
    (("cursor", "consolidations", 0, "diagnostic", "definition"), ""),
    (("cursor", "consolidations", 0, "diagnostic", "value"), True),
    (("diagnostic_aliases",), [1, 1]),
    (("diagnostic_aliases",), [0]),
    (("diagnostic_aliases",), [0, True]),
    (("diagnostic_aliases",), [0, -1]),
    (("policy", "max_identifier_bytes"), 129),
]


@pytest.mark.parametrize("path,replacement", CORRUPTIONS)
def test_should_refuse_matching_digest_corruption_or_original_binding_mismatch(path, replacement):
    value = json.loads(encode())
    at_path(value, path[:-1])[path[-1]] = replacement
    with pytest.raises(ValueError):
        decode(raw_json(value))


@pytest.mark.parametrize("replacement", [0.0, 0, 9])
def test_should_refuse_alias_values_with_changed_bits_or_numeric_type(replacement):
    value = json.loads(encode())
    value["cursor"]["consolidations"][1]["diagnostic"]["value"] = replacement
    with pytest.raises(ValueError):
        decode(raw_json(value))


@pytest.mark.parametrize("field,value", [("source_sha256", "0" * 64), ("policy_sha256", "0" * 64)])
def test_should_refuse_independently_expected_source_or_policy(field, value):
    with pytest.raises(ValueError):
        decode(encode(), binding=replace(BINDING, **{field: value}))


def test_should_refuse_content_mismatch_before_json_parse(monkeypatch):
    monkeypatch.setattr(json, "loads", lambda *a, **k: pytest.fail("untrusted JSON parsed"))
    with pytest.raises(ValueError):
        codec().decode(b"{}", binding=BINDING, expected_sha256="0" * 64, limits=LIMITS)


def test_should_enforce_exact_original_wire_boundary():
    raw = encode()
    exact = replace(LIMITS, max_encoded_bytes=len(raw))
    assert encode(limits=exact) == raw and decode(raw, limits=exact) == saved()
    with pytest.raises(ValueError):
        encode(limits=replace(exact, max_encoded_bytes=len(raw) - 1))
    with pytest.raises(ValueError):
        decode(raw, limits=replace(exact, max_encoded_bytes=len(raw) - 1))


@pytest.mark.parametrize("text", ["é" * 65, "x" * 129, "\ud800"])
def test_should_refuse_identifier_utf8_bound_or_invalid_encoding(text):
    state = saved()
    object.__setattr__(state, "actor_version", text)
    with pytest.raises(ValueError):
        encode(state)


@pytest.mark.parametrize(
    "raw", [b"{}", b'{"x":1,"x":2}', b"\xff", b"NaN", b"[" * 2000 + b"]" * 2000]
)
def test_should_refuse_malformed_noncanonical_or_unsupported_json(raw):
    with pytest.raises(ValueError):
        decode(raw)


def test_should_refuse_noncanonical_json_even_with_matching_content_digest():
    with pytest.raises(ValueError):
        decode(json.dumps(json.loads(encode()), indent=2).encode())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 2**1024, -(2**1024), object()])
def test_should_refuse_unsupported_diagnostic_scalar_before_serialization(value, monkeypatch):
    state = saved()
    object.__setattr__(state.consolidations[0].diagnostic, "value", value)
    monkeypatch.setattr(
        json.JSONEncoder, "iterencode", lambda *a, **k: pytest.fail("invalid history serialized")
    )
    with pytest.raises(ValueError):
        encode(state)


@pytest.mark.parametrize(
    "field,value",
    [
        ("consolidation_limit", True),
        ("consolidation_limit", -1),
        ("consolidation_limit", 2**63),
        ("max_identifier_bytes", 0),
        ("max_identifier_bytes", True),
        ("max_identifier_bytes", 2**63),
    ],
)
def test_should_refuse_invalid_original_policy(field, value):
    with pytest.raises(ValueError):
        replace(POLICY, **{field: value})


def test_should_capture_actual_runtime_under_original_lease_without_any_model_operation(
    monkeypatch,
):
    from src.adapters.numpy_learners import BackpropLearner
    from src.app.actor_shadow import ActorShadowRuntime
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.core.backprop_mlp import BackpropMLP
    from src.core.experience import LogicalClock

    learner = BackpropLearner(BackpropMLP(3, 2, seed=23), learning_rate=0.03)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 7.0)
    clock = LogicalClock(9)
    runtime: Any = ActorShadowRuntime(
        learner,
        actor_version="actor",
        candidate_version="learner",
        clock=clock,
        budget=budget,
        max_consolidations=4,
    )
    for method in ("train_batch", "predict", "snapshot_state", "restore_state"):
        monkeypatch.setattr(BackpropLearner, method, lambda *a, **k: pytest.fail("model operation"))
    original = vars(runtime).copy()
    original_budget = vars(budget).copy()
    cursor = runtime.capture_consolidation_cursor()
    assert cursor == ConsolidationCursor(1, "actor", "learner", 4, (), (), False, False, 0, True)
    assert vars(runtime) == original and vars(budget) == original_budget and clock.now() == 9
    with runtime._exclusive(), pytest.raises(ValueError, match="busy"):
        runtime.capture_consolidation_cursor()
    # Synthetic histories do not execute transforms,restores or model updates.
    state = saved(stopped=True, retired=True)
    runtime._attempted_ids = set(state.attempted_ids)
    runtime._consolidations = list(state.consolidations)
    runtime._stopped, runtime._retired, runtime._revision = True, True, 13
    captured = runtime.capture_consolidation_cursor()
    assert captured == state
    assert captured.consolidations[0].diagnostic is captured.consolidations[1].diagnostic
    assert captured.consolidations[0].diagnostic is not state.consolidations[0].diagnostic
    assert decode(encode(captured)) == state
    before = vars(runtime).copy()
    runtime._consolidations.append(state.consolidations[0])
    invalid_records = runtime._consolidations.copy()
    with pytest.raises(ValueError):
        runtime.capture_consolidation_cursor()
    assert vars(runtime) == before and vars(budget) == original_budget
    assert runtime._consolidations == invalid_records
    assert not runtime._write_gate.locked()


WIRE_FIELDS = [
    (path, field.name)
    for path, kind in [
        (("cursor",), ConsolidationCursor),
        (("cursor", "consolidations", 0), AppliedConsolidation),
        (("cursor", "consolidations", 0, "diagnostic"), TrainingDiagnostic),
        (("policy",), ConsolidationCodecPolicy),
        (("binding",), CodecBinding),
    ]
    for field in fields(kind)
]


@pytest.mark.parametrize("path,name", WIRE_FIELDS)
def test_should_refuse_each_missing_wire_field_with_matching_digest(path, name):
    value = json.loads(encode())
    del at_path(value, path)[name]
    with pytest.raises(ValueError):
        decode(raw_json(value))


@pytest.mark.parametrize(
    "field,value",
    [
        ("attempted_ids", ["a"]),
        ("attempted_ids", ("a",) * 5),
        ("consolidations", []),
        ("revision", 2**63),
        ("consolidation_limit", True),
        ("actor_version", object()),
        ("payload_ready", 1),
    ],
)
def test_should_refuse_unsupported_native_shape_type_or_count_before_serialization(
    field, value, monkeypatch
):
    state = saved()
    object.__setattr__(state, field, value)
    monkeypatch.setattr(
        json.JSONEncoder, "iterencode", lambda *a, **k: pytest.fail("invalid metadata serialized")
    )
    with pytest.raises(ValueError):
        encode(state)


@pytest.mark.parametrize("kind", [ConsolidationCodecPolicy, CodecBinding, CodecLimits])
def test_should_refuse_unknown_original_configuration_fields(kind):
    config = deepcopy(
        {ConsolidationCodecPolicy: POLICY, CodecBinding: BINDING, CodecLimits: LIMITS}[kind]
    )
    object.__setattr__(config, "future", True)
    with pytest.raises(ValueError):
        if isinstance(config, ConsolidationCodecPolicy):
            codec(config)
        elif isinstance(config, CodecBinding):
            codec().encode(saved(), binding=config, limits=LIMITS)
        else:
            assert isinstance(config, CodecLimits)
            codec().encode(saved(), binding=BINDING, limits=config)


def test_should_preserve_exact_utf8_boundary_and_native_definition_whitespace():
    state = saved(alias=False)
    object.__setattr__(state.consolidations[0].diagnostic, "definition", "é" * 64)
    object.__setattr__(state.consolidations[1].diagnostic, "definition", " native definition ")
    assert decode(encode(state)) == state


def test_should_refuse_independently_expected_authority_reference():
    with pytest.raises(ValueError):
        decode(encode(), decoder=codec(authority="0" * 64))


def test_should_preserve_failed_transform_attempts_without_completed_receipts():
    failed = replace(saved(), consolidations=())
    assert decode(encode(failed)) == failed
