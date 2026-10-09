"""Complete paired bytes, independent original bindings and zero-native controls."""

from copy import deepcopy
from dataclasses import fields, replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

import src.adapters.managed_record_checkpoint_codec as module
from src.adapters.managed_record_checkpoint_codec import ManagedRecordCheckpointCodec
from src.adapters.managed_record_checkpoint_schema import PAIR_SCHEMAS
from src.adapters.lifecycle_checkpoint_schema import metadata_aliases, project_metadata
from src.adapters.numpy_learners import BackpropLearner
from src.app.managed_record_capture import capture_managed_records
from src.core.actor_ports import AppliedConsolidation
from src.core.checkpoint_codec import CheckpointCodec, CodecBinding, CodecLimits
from src.core.consolidation_codec_policy import ConsolidationCodecPolicy
from src.core.experience import LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.managed_record_codec_policy import ManagedRecordCodecPolicy
from src.core.managed_record_state import (
    ManagedRecordCapture,
    ManagedRecordMetadata,
    RECORD_AUTHORITY_PATHS,
)
from src.core.managed_lifecycle_state import AuthorityReference, LifecycleCaptureLimits
from src.core.managed_record_state import validate_managed_record_capture
from test_lifecycle_checkpoint_codec import policy as life_policy, dictionaries, at
from test_managed_record_capture import synthetic, make_owner  # Pure factories under NEW scope.

BINDING = CodecBinding("a" * 64, "b" * 64)
AUTHORITY = "c" * 64
LIMITS = CodecLimits(65536, 1, 1, 1)
CAPTURE_LIMITS = LifecycleCaptureLimits(64, 128)


def policy(state, limits=CAPTURE_LIMITS):
    life = state.metadata.lifecycle
    from src.core.managed_lifecycle_state import ManagedLifecycleCapture, AUTHORITY_PATHS

    return ManagedRecordCodecPolicy(
        life_policy(
            ManagedLifecycleCapture(
                life, tuple(item for item in state.authority if item.path in AUTHORITY_PATHS)
            ),
            limits,
        ),
        ConsolidationCodecPolicy(
            state.metadata.consolidation.consolidation_limit, limits.max_identifier_bytes
        ),
    )


def codec(state, limits=CAPTURE_LIMITS):
    return ManagedRecordCheckpointCodec(
        policy(state, limits), original=state, authority_sha256=AUTHORITY
    )


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def project(metadata):
    return project_metadata(metadata, ManagedRecordMetadata, schemas=PAIR_SCHEMAS)


def roundtrip(state, binding=BINDING):
    port: CheckpointCodec[ManagedRecordCapture] = codec(state)
    raw = port.encode(state, binding=binding, limits=LIMITS)
    result = port.decode(
        raw, binding=binding, expected_sha256=sha256(raw).hexdigest(), limits=LIMITS
    )
    assert result.metadata == state.metadata
    assert result.metadata is not state.metadata and result.authority is state.authority
    assert port.encode(result, binding=binding, limits=LIMITS) == raw
    assert metadata_aliases(
        result.metadata, ManagedRecordMetadata, schemas=PAIR_SCHEMAS
    ) == metadata_aliases(state.metadata, ManagedRecordMetadata, schemas=PAIR_SCHEMAS)
    validate_managed_record_capture(result, CAPTURE_LIMITS)
    return result, raw


@pytest.fixture(scope="module")
def encoded():
    state = synthetic()
    port = codec(state)
    raw = port.encode(state, binding=BINDING, limits=LIMITS)
    return state, port, raw


def refuse(encoded, monkeypatch, change):
    _, port, raw = encoded
    body = json.loads(raw)
    change(body)
    damaged = canonical(body)
    monkeypatch.setattr(
        module, "materialize_metadata", lambda *a, **k: pytest.fail("wire materialization")
    )
    with pytest.raises(ValueError):
        port.decode(
            damaged, binding=BINDING, expected_sha256=sha256(damaged).hexdigest(), limits=LIMITS
        )


def test_should_roundtrip_every_complete_paired_field_reference_and_alias(encoded):
    original, _, _ = encoded
    result, raw = roundtrip(original)
    assert len(json.loads(raw)["reference_manifest"]) == 59
    assert json.loads(raw)["kind"] == "managed_record_pair_v1"
    a, b = result.metadata.consolidation.consolidations
    assert a.diagnostic is b.diagnostic
    assert (
        result.metadata.lifecycle.copy.limits
        is result.metadata.lifecycle.lifecycle.policy.owned_payload_copies
    )
    assert (
        result.metadata.lifecycle.registry.limits
        is result.metadata.lifecycle.lifecycle.policy.holders
    )
    assert result.metadata.owner.inbox_last_tick > result.metadata.lifecycle.lifecycle.last_tick
    assert result.metadata.owner.serving_actor_version != result.metadata.owner.base_actor_version
    assert result.metadata.consolidation.attempted_ids == ("a", "b", "c")
    assert tuple(item.attempt_number for item in result.metadata.consolidation.consolidations) == (
        1,
        3,
    )


def test_should_cover_all_native_fields_with24_explicit_trusted_schemas():
    assert len(PAIR_SCHEMAS) == 24
    assert all(
        set(names) == {field.name for field in fields(kind)} for kind, names in PAIR_SCHEMAS.items()
    )


TREE = dict(
    metadata=project(synthetic().metadata),
    policy=project_metadata(policy(synthetic()), ManagedRecordCodecPolicy, schemas=PAIR_SCHEMAS),
)
RECORDS = dictionaries(TREE)
FIELD_CASES = [(path, name) for path, names in RECORDS for name in names]


@pytest.mark.parametrize("path,name", FIELD_CASES)
def test_should_refuse_every_missing_nested_wire_field_before_materialization(
    encoded, monkeypatch, path, name
):
    refuse(encoded, monkeypatch, lambda body: at(body, path).pop(name))


@pytest.mark.parametrize("path,names", RECORDS)
def test_should_refuse_every_unknown_nested_wire_field_before_materialization(
    encoded, monkeypatch, path, names
):
    refuse(encoded, monkeypatch, lambda body: at(body, path).update(future=True))


@pytest.mark.parametrize("field", sorted(module.ENVELOPE))
def test_should_refuse_every_missing_envelope_field(encoded, monkeypatch, field):
    refuse(encoded, monkeypatch, lambda body: body.pop(field))


@pytest.mark.parametrize("path", sorted(RECORD_AUTHORITY_PATHS))
@pytest.mark.parametrize("field", ("path", "present", "first"))
def test_should_refuse_every_original_reference_manifest_change(encoded, monkeypatch, path, field):
    def change(body):
        item = next(item for item in body["reference_manifest"] if item["path"] == path)
        if field == "path":
            item[field] = "unknown"
        elif field == "present":
            item[field] = not item[field]
        else:
            item[field] = -1

    refuse(encoded, monkeypatch, change)


@pytest.mark.parametrize("path", sorted(RECORD_AUTHORITY_PATHS))
def test_should_require_original_reference_identity_during_encoding(encoded, path):
    state, port, _ = encoded
    altered = replace(
        state,
        authority=tuple(
            AuthorityReference(item.path, object()) if item.path == path else item
            for item in state.authority
        ),
    )
    with pytest.raises(ValueError):
        port.encode(altered, binding=BINDING, limits=LIMITS)


@pytest.mark.parametrize(
    "owner_field,cursor_field,value",
    (
        ("revision", "revision", 12),
        ("base_actor_version", "actor_version", "other"),
        ("learner_version", "learner_version", "other"),
        ("retired", "retired", False),
    ),
)
def test_should_refuse_structurally_coherent_different_original_owner_observation(
    encoded, monkeypatch, owner_field, cursor_field, value
):
    def change(body):
        body["metadata"]["owner"][owner_field] = value
        body["metadata"]["consolidation"][cursor_field] = value
        if cursor_field.endswith("version"):
            for row in body["metadata"]["consolidation"]["consolidations"]:
                row[cursor_field] = value

    refuse(encoded, monkeypatch, change)


@pytest.mark.parametrize(
    "path,name,value",
    (
        (("owner",), "budget_updates", 8),
        (("owner",), "inbox_completed_updates", 6),
        (("owner",), "inbox_last_tick", 13),
        (("owner",), "serving_actor_version", "later"),
        (("lifecycle", "driver"), "created_at", 1.0),
        (("lifecycle", "lifecycle"), "last_seconds", 9.0),
        (("lifecycle", "lifecycle"), "auxiliary_started_at", 4.0),
        (("lifecycle", "copy"), "charged_bytes", 129),
        (("lifecycle", "lifecycle"), "admitted_bytes", 65),
    ),
)
def test_should_refuse_spliced_complete_clock_epoch_consumed_charge_and_budget_observations(
    encoded, monkeypatch, path, name, value
):
    refuse(encoded, monkeypatch, lambda body: at(body["metadata"], path).update({name: value}))


@pytest.mark.parametrize("value", (-0.0, -2.5, 0, 0.125, 2**100))
def test_should_preserve_native_diagnostic_types_signed_zero_and_finite_large_integer(value):
    state = synthetic()
    diagnostic = TrainingDiagnostic("native_loss", value)
    cursor = replace(
        state.metadata.consolidation,
        consolidations=tuple(
            replace(row, diagnostic=diagnostic)
            for row in state.metadata.consolidation.consolidations
        ),
    )
    state = replace(state, metadata=replace(state.metadata, consolidation=cursor))
    result, _ = roundtrip(state)
    a, b = result.metadata.consolidation.consolidations
    assert a.diagnostic is b.diagnostic
    assert type(a.diagnostic.value) is type(value) and canonical(a.diagnostic.value) == canonical(
        value
    )


def test_should_preserve_cross_component_shared_tuple_and_shared_entire_clock_anchor_graph():
    state = synthetic()
    life = state.metadata.lifecycle
    owner = replace(life.owner, declaration_seconds=life.owner.declaration_ticks)
    diagnostic = state.metadata.consolidation.consolidations[0].diagnostic
    cursor = replace(
        state.metadata.consolidation,
        attempted_ids=owner.opted_out,
        consolidations=(AppliedConsolidation("subject-b", "base", "candidate", 1, diagnostic),),
    )
    state = replace(
        state,
        metadata=replace(
            state.metadata, consolidation=cursor, lifecycle=replace(life, owner=owner)
        ),
    )
    result, _ = roundtrip(state)
    assert result.metadata.consolidation.attempted_ids is result.metadata.lifecycle.owner.opted_out
    assert (
        result.metadata.lifecycle.owner.declaration_ticks
        is result.metadata.lifecycle.owner.declaration_seconds
    )


def test_should_preserve_distinct_equal_native_diagnostics():
    state = synthetic()
    a, b = state.metadata.consolidation.consolidations
    b = replace(b, diagnostic=replace(a.diagnostic))
    state = replace(
        state,
        metadata=replace(
            state.metadata,
            consolidation=replace(state.metadata.consolidation, consolidations=(a, b)),
        ),
    )
    result, _ = roundtrip(state)
    a, b = result.metadata.consolidation.consolidations
    assert a.diagnostic == b.diagnostic and a.diagnostic is not b.diagnostic


@pytest.mark.parametrize(
    "state", ("ready", "running", "stopping", "stopped", "exhausted", "failed")
)
def test_should_preserve_all_complete_supported_driver_states(state):
    original = synthetic()
    life = original.metadata.lifecycle
    original = replace(
        original,
        metadata=replace(
            original.metadata, lifecycle=replace(life, driver=replace(life.driver, state=state))
        ),
    )
    result, _ = roundtrip(original)
    assert result.metadata.lifecycle.driver.state == state


def test_should_preserve_distinct_equal_aliases_and_reject_rebinding_alias_graph_to_original(
    encoded, monkeypatch
):
    state, _, _ = encoded
    diagnostic = state.metadata.consolidation.consolidations[0].diagnostic
    assert state.metadata.consolidation.consolidations[1].diagnostic is diagnostic

    def change(body):
        nodes = module.preflight_pair(body["metadata"], policy(state), canonical)
        index = next(
            i
            for i, node in enumerate(nodes)
            if node[0] == "metadata.consolidation.consolidations[1].diagnostic"
        )
        body["metadata_aliases"][index] = (
            index  # Structurally valid distinct equal object, different original alias graph.
        )

    refuse(encoded, monkeypatch, change)


@pytest.mark.parametrize("kind", ("type", "bits", "parent_child", "required_policy"))
def test_should_refuse_invalid_complete_alias_relationships_before_materialization(
    encoded, monkeypatch, kind
):
    state, _, _ = encoded

    def change(body):
        nodes = module.preflight_pair(body["metadata"], policy(state), canonical)
        paths = {node[0]: i for i, node in enumerate(nodes)}
        aliases = body["metadata_aliases"]
        if kind == "type":
            aliases[paths["metadata.consolidation.attempted_ids"]] = 0
        elif kind == "bits":
            body["metadata"]["consolidation"]["consolidations"][1]["diagnostic"]["value"] = 0.0
        elif kind == "parent_child":
            # Equal whole anchors can share a parent only when their nested
            # pair aliases agree too. Keep the old distinct child pair aliases.
            owner = body["metadata"]["lifecycle"]["owner"]
            owner["declaration_seconds"] = deepcopy(owner["declaration_ticks"])
            aliases[paths["metadata.lifecycle.owner.declaration_seconds"]] = paths[
                "metadata.lifecycle.owner.declaration_ticks"
            ]
        else:
            index = paths["metadata.lifecycle.registry.limits"]
            aliases[index] = index

    refuse(encoded, monkeypatch, change)


@pytest.mark.parametrize(
    "which", ("binding", "limits", "policy", "original", "metadata", "nested_policy")
)
def test_should_refuse_unknown_native_configuration_fields(encoded, which):
    state, port, _ = encoded
    if which == "binding":
        binding_value = replace(BINDING)
        object.__setattr__(binding_value, "future", True)
        with pytest.raises(ValueError):
            port.encode(state, binding=binding_value, limits=LIMITS)
    elif which == "limits":
        limit_value = replace(LIMITS)
        object.__setattr__(limit_value, "future", True)
        with pytest.raises(ValueError):
            port.encode(state, binding=BINDING, limits=limit_value)
    else:
        config, original = policy(state), replace(state, metadata=deepcopy(state.metadata))
        target = {
            "policy": config,
            "nested_policy": config.consolidation,
            "original": original,
            "metadata": original.metadata,
        }[which]
        object.__setattr__(target, "future", True)
        with pytest.raises(ValueError):
            ManagedRecordCheckpointCodec(config, original=original, authority_sha256=AUTHORITY)


@pytest.mark.parametrize(
    "field,value",
    (
        ("codec_version", True),
        ("codec_version", 2),
        ("kind", "other"),
        ("authority_sha256", "d" * 64),
        ("future", True),
    ),
)
def test_should_refuse_unsupported_top_level_binding_or_schema(encoded, monkeypatch, field, value):
    refuse(encoded, monkeypatch, lambda body: body.update({field: value}))


@pytest.mark.parametrize("name", ("source_sha256", "policy_sha256"))
def test_should_refuse_independently_different_source_or_policy_binding(encoded, monkeypatch, name):
    refuse(encoded, monkeypatch, lambda body: body["binding"].update({name: "d" * 64}))


@pytest.mark.parametrize(
    "which", ("content", "duplicate", "noncanonical", "truncated", "empty", "wrong_type")
)
def test_should_refuse_corrupt_or_noncanonical_bytes_before_materialization(
    encoded, monkeypatch, which
):
    _, port, raw = encoded
    monkeypatch.setattr(
        module, "materialize_metadata", lambda *a, **k: pytest.fail("materialization")
    )
    damaged = raw
    expected = sha256(raw).hexdigest()
    if which == "content":
        expected = "d" * 64
    elif which == "duplicate":
        damaged = b'{"codec_version":1,' + raw[1:]
    elif which == "noncanonical":
        damaged = b" " + raw
    elif which == "truncated":
        damaged = raw[:-1]
    elif which == "empty":
        damaged = b""
    else:
        damaged = bytearray(raw)
    if which != "content":
        expected = sha256(damaged).hexdigest()
    with pytest.raises(ValueError):
        port.decode(damaged, binding=BINDING, expected_sha256=expected, limits=LIMITS)


@pytest.mark.parametrize(
    "field,value",
    (
        ("revision", True),
        ("revision", -1),
        ("revision", 2**63),
        ("stopped", 1),
        ("inbox_last_tick", -2),
        ("inbox_last_tick", True),
        ("enrollment", 0),
        ("base_actor_version", "x" * 129),
        ("base_actor_version", "é" * 65),
    ),
)
def test_should_refuse_invalid_exact_runtime_scalar_and_utf8_before_materialization(
    encoded, monkeypatch, field, value
):
    refuse(encoded, monkeypatch, lambda body: body["metadata"]["owner"].update({field: value}))


@pytest.mark.parametrize("value", (True, "number", 2**1025, float("nan"), float("inf")))
def test_should_refuse_invalid_diagnostic_scalar_before_materialization(
    encoded, monkeypatch, value
):
    _, port, raw = encoded
    body = json.loads(raw)
    body["metadata"]["consolidation"]["consolidations"][0]["diagnostic"]["value"] = value
    damaged = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()
    monkeypatch.setattr(
        module, "materialize_metadata", lambda *a, **k: pytest.fail("materialization")
    )
    with pytest.raises(ValueError):
        port.decode(
            damaged, binding=BINDING, expected_sha256=sha256(damaged).hexdigest(), limits=LIMITS
        )


@pytest.mark.parametrize("value", (None, [], [-1], [True], [9999]))
def test_should_refuse_incomplete_or_invalid_immutable_alias_graph(encoded, monkeypatch, value):
    refuse(encoded, monkeypatch, lambda body: body.update(metadata_aliases=value))


def test_should_refuse_wire_capacity_before_json_parse(encoded, monkeypatch):
    _, port, raw = encoded
    monkeypatch.setattr(module.json, "loads", lambda *a, **k: pytest.fail("JSON parse"))
    with pytest.raises(ValueError):
        port.decode(
            raw,
            binding=BINDING,
            expected_sha256=sha256(raw).hexdigest(),
            limits=replace(LIMITS, max_encoded_bytes=1),
        )


@pytest.mark.parametrize("which", ("aggregate", "string", "consolidation", "shared_string"))
def test_should_refuse_independently_different_or_insufficient_component_policy(encoded, which):
    state, _, _ = encoded
    config = policy(state)
    if which == "aggregate":
        config = replace(
            config,
            lifecycle=replace(
                config.lifecycle, capture_limits=replace(CAPTURE_LIMITS, max_records=15)
            ),
        )
    elif which == "string":
        tiny = replace(CAPTURE_LIMITS, max_identifier_bytes=2)
        config = replace(
            config,
            lifecycle=replace(config.lifecycle, capture_limits=tiny),
            consolidation=replace(config.consolidation, max_identifier_bytes=2),
        )
    elif which == "consolidation":
        config = replace(config, consolidation=replace(config.consolidation, consolidation_limit=5))
    else:
        with pytest.raises(ValueError):
            replace(config, consolidation=replace(config.consolidation, max_identifier_bytes=127))
        return
    with pytest.raises(ValueError):
        port = ManagedRecordCheckpointCodec(config, original=state, authority_sha256=AUTHORITY)
        port.encode(state, binding=BINDING, limits=LIMITS)


@pytest.fixture(scope="module")
def actual(request):
    with pytest.MonkeyPatch.context() as patch:
        for name in ("train_batch", "predict", "snapshot_state", "restore_state"):
            patch.setattr(BackpropLearner, name, lambda *a, **k: pytest.fail("native operation"))
        owners = [make_owner(index) for index in range(3)]
        before = [[port.calls for port in row[-1]] for row in owners]
        patch.setattr(LogicalClock, "now", lambda *a: pytest.fail("clock read"))
        captures = [capture_managed_records(row[0], limits=CAPTURE_LIMITS) for row in owners]
        yield captures, owners
        for row, calls in zip(owners, before):
            _, life, runtime, budget, driver, ports = row
            assert budget.updates_completed == 0 and runtime._revision == 5
            assert life._admitted_bytes == 24
            assert life._copy_budget is None or life._copy_budget._charged == 24
            assert driver is None or (driver._thread is None and driver._polls == 0)
            assert [port.calls for port in ports] == calls
    folder = Path(request.config.option.basetemp)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "paired-byte-native-work.json").write_text(
        json.dumps(
            dict(
                graphs=3,
                managed_refusals=3,
                updates=0,
                predictions=0,
                snapshots=0,
                restores=0,
                sleeps=0,
                structural=0,
                workers=0,
                threads=0,
            ),
            indent=2,
        ),
        encoding="utf8",
    )


@pytest.mark.parametrize("index", range(3))
def test_should_compose_actual_common_interval_capture_with_complete_bytes_without_native_or_port_work(
    actual, index
):
    captures, owners = actual
    state = captures[index]
    paths = [
        "src/core/managed_record_state.py",
        "src/core/managed_record_codec_policy.py",
        "src/app/managed_record_capture.py",
        "src/app/managed_lifecycle_capture.py",
        "src/core/consolidation_cursor.py",
        "src/core/managed_lifecycle_validation.py",
        "src/adapters/managed_record_checkpoint_codec.py",
        "src/adapters/managed_record_checkpoint_schema.py",
        "src/adapters/lifecycle_checkpoint_schema.py",
    ]
    binding = CodecBinding(
        sha256(
            canonical({name: sha256(Path(name).read_bytes()).hexdigest() for name in paths})
        ).hexdigest(),
        sha256(
            canonical(
                project_metadata(policy(state), ManagedRecordCodecPolicy, schemas=PAIR_SCHEMAS)
            )
        ).hexdigest(),
    )
    result, raw = roundtrip(state, binding)
    assert json.loads(raw)["binding"] == dict(
        source_sha256=binding.source_sha256, policy_sha256=binding.policy_sha256
    )
    assert result.metadata.owner.revision == 5
    assert result.metadata.lifecycle.lifecycle.admitted_bytes == 24
    assert result.authority is state.authority
    assert {item.path: item.value for item in result.authority}["root.owner"] is owners[index][0]
    assert b"managed_record_pair_v1" in raw


def test_should_refuse_other_actual_owner_reference_graph_even_when_scalar_histories_match(actual):
    captures, _ = actual
    original, other = captures[:2]
    with pytest.raises(ValueError):
        codec(original).encode(other, binding=BINDING, limits=LIMITS)


def test_should_preserve_one_byte_native_identifiers_without_narrowing_fixed_schema_enums():
    from src.core.data_lifecycle import DataProvenance

    state = synthetic()
    life = state.metadata.lifecycle
    catalog = tuple(
        replace(item, key=("e", key), provenance=DataProvenance("q", "p", True, False))
        for item, key in zip(life.owner.catalog, ("x", "y"))
    )
    owner = replace(
        life.owner,
        catalog=catalog,
        opted_out=("p",),
        revoked_keys=(catalog[0].key,),
        declaration_ticks=tuple((item.key, i + 2) for i, item in enumerate(catalog)),
        declaration_seconds=tuple((item.key, float(i + 2)) for i, item in enumerate(catalog)),
    )
    diagnostic = TrainingDiagnostic("z", -0.0)
    cursor = replace(
        state.metadata.consolidation,
        actor_version="a",
        learner_version="b",
        attempted_ids=("d", "e", "f"),
        consolidations=(
            AppliedConsolidation("d", "a", "b", 1, diagnostic),
            AppliedConsolidation("f", "a", "b", 3, diagnostic),
        ),
    )
    state = replace(
        state,
        metadata=replace(
            state.metadata,
            owner=replace(
                state.metadata.owner,
                base_actor_version="a",
                serving_actor_version="c",
                learner_version="b",
            ),
            consolidation=cursor,
            lifecycle=replace(life, owner=owner, driver=replace(life.driver, error_type=None)),
        ),
    )
    port = codec(state, replace(CAPTURE_LIMITS, max_identifier_bytes=1))
    raw = port.encode(state, binding=BINDING, limits=LIMITS)
    result = port.decode(
        raw, binding=BINDING, expected_sha256=sha256(raw).hexdigest(), limits=LIMITS
    )
    assert result.metadata.lifecycle.driver.state == "failed"
    assert result.metadata.lifecycle.registry.holders[0].kind == "actor"


@pytest.mark.parametrize(
    "kind",
    (
        "duplicate",
        "order",
        "unknown_receipt",
        "duplicate_receipt",
        "attempt_number",
        "quota",
        "versions",
    ),
)
def test_should_refuse_corrupt_complete_consolidation_relationships_before_construction(
    encoded, monkeypatch, kind
):
    def change(body):
        cursor = body["metadata"]["consolidation"]
        if kind == "duplicate":
            cursor["attempted_ids"] = ["a", "a", "c"]
        elif kind == "order":
            cursor["attempted_ids"] = ["b", "a", "c"]
        elif kind == "unknown_receipt":
            cursor["consolidations"][0]["event_id"] = "unknown"
        elif kind == "duplicate_receipt":
            cursor["consolidations"][1]["event_id"] = "a"
        elif kind == "attempt_number":
            cursor["consolidations"][1]["attempt_number"] = 1
        elif kind == "quota":
            cursor["consolidation_limit"] = 3
        else:
            cursor["actor_version"] = cursor["learner_version"]

    refuse(encoded, monkeypatch, change)


def test_should_refuse_joint_wire_capacity_before_alias_checks_or_materialization(monkeypatch):
    state = synthetic()
    port = codec(state, replace(CAPTURE_LIMITS, max_records=17))
    body = json.loads(port.encode(state, binding=BINDING, limits=LIMITS))
    body["metadata"]["lifecycle"]["owner"]["catalog"].append(
        deepcopy(body["metadata"]["lifecycle"]["owner"]["catalog"][0])
    )
    raw = canonical(body)
    original_aliases = module.preflight_aliases
    calls: list[str] = []

    def original_configuration_only(*args, **kwargs):
        assert not calls, "wire aliases ran before aggregate refusal"
        calls.append("independent original configuration")
        return original_aliases(*args, **kwargs)

    monkeypatch.setattr(module, "preflight_aliases", original_configuration_only)
    monkeypatch.setattr(
        module, "materialize_metadata", lambda *a, **k: pytest.fail("materialization")
    )
    with pytest.raises(ValueError, match="aggregate"):
        port.decode(raw, binding=BINDING, expected_sha256=sha256(raw).hexdigest(), limits=LIMITS)
    assert calls == ["independent original configuration"]
