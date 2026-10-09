"""Complete metadata bytes and original references; fresh zero-native fixtures."""

from copy import deepcopy
from dataclasses import fields, replace
from hashlib import sha256
from pathlib import Path
import json
import pytest
import numpy as np

import src.adapters.lifecycle_checkpoint_codec as module
from src.adapters.lifecycle_checkpoint_codec import LifecycleCheckpointCodec
from src.adapters.lifecycle_checkpoint_schema import SCHEMAS, project_metadata
from src.core.checkpoint_codec import CheckpointCodec, CodecBinding, CodecLimits
from src.core.lifecycle_codec_policy import LifecycleCodecPolicy
from src.core.managed_lifecycle_state import (
    AuthorityReference,
    LifecycleCaptureLimits,
    ManagedLifecycleCapture,
)
from src.core.managed_lifecycle_validation import validate_lifecycle_capture
from test_managed_lifecycle_state import sample  # Pure synthetic factory only; no old fixture run.

BINDING = CodecBinding("a" * 64, "b" * 64)
AUTHORITY = "c" * 64
LIMITS = CodecLimits(32768, 1, 1, 1)
CAPTURE_LIMITS = LifecycleCaptureLimits(64, 128)


def policy(state, limits=CAPTURE_LIMITS):
    data = state.metadata
    return LifecycleCodecPolicy(
        data.owner.limits,
        data.lifecycle.policy,
        None if data.driver is None else data.driver.limits,
        limits,
    )


def codec(state, *, capture_limits=CAPTURE_LIMITS):
    return LifecycleCheckpointCodec(
        policy(state, capture_limits), original=state, authority_sha256=AUTHORITY
    )


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def roundtrip(state, *, capture_limits=CAPTURE_LIMITS):
    port: CheckpointCodec[ManagedLifecycleCapture] = codec(state, capture_limits=capture_limits)
    raw = port.encode(state, binding=BINDING, limits=LIMITS)
    result = port.decode(
        raw, binding=BINDING, expected_sha256=sha256(raw).hexdigest(), limits=LIMITS
    )
    assert result.metadata == state.metadata
    assert result.authority is state.authority
    assert result.metadata is not state.metadata
    assert port.encode(result, binding=BINDING, limits=LIMITS) == raw
    validate_lifecycle_capture(result, capture_limits)
    return result, raw


@pytest.fixture(scope="module")
def encoded():
    state = sample()
    port = codec(state)
    raw = port.encode(state, binding=BINDING, limits=LIMITS)
    return state, port, raw


@pytest.mark.parametrize(
    "elapsed,copies,driver",
    [(a, b, c) for a in (False, True) for b in (False, True) for c in (False, True) if a or not c],
)
def test_should_roundtrip_complete_optional_original_policy_and_every_reference(
    elapsed, copies, driver
):
    state = sample(elapsed=elapsed, copies=copies, driver=driver)
    result, raw = roundtrip(state)
    assert result.metadata.registry.limits is result.metadata.lifecycle.policy.holders
    if copies:
        assert result.metadata.copy.limits is result.metadata.lifecycle.policy.owned_payload_copies
    assert len(json.loads(raw)["reference_manifest"]) == 48


@pytest.mark.parametrize(
    "state", ["ready", "running", "stopping", "stopped", "exhausted", "failed"]
)
def test_should_preserve_all_supported_original_driver_states_and_history(state):
    original = sample(state=state)
    result, _ = roundtrip(original)
    assert result.metadata.driver == original.metadata.driver
    assert result.metadata.owner.catalog[0].key == ("episode", "b")  # Insertion order, not sorted.
    assert result.metadata.registry.holders[-1].ready is None
    assert result.metadata.registry.holders[1].ready is False


@pytest.mark.parametrize("value", [0, 0.0, -0.0, 2**100, 1.25])
def test_should_preserve_supported_epoch_numeric_types_signed_zero_and_large_native_integer(value):
    state = sample()
    state = replace(
        state,
        metadata=replace(state.metadata, driver=replace(state.metadata.driver, created_at=value)),
    )
    result, raw = roundtrip(state)
    assert type(result.metadata.driver.created_at) is type(value)
    assert canonical(result.metadata.driver.created_at) == canonical(value)
    assert b"created_at" in raw


def test_should_preserve_shared_and_distinct_equal_consent_provenance_and_key_aliases():
    state = sample()
    a, b = state.metadata.owner.catalog
    b = replace(b, provenance=a.provenance, consent=a.consent)
    owner = replace(state.metadata.owner, catalog=(a, b))
    state = replace(state, metadata=replace(state.metadata, owner=owner))
    result, _ = roundtrip(state)
    x, y = result.metadata.owner.catalog
    assert x.provenance is y.provenance and x.consent is y.consent
    assert x.key is result.metadata.owner.declaration_ticks[0][0]
    assert x.key is result.metadata.owner.declaration_seconds[0][0]
    distinct = replace(b, provenance=replace(a.provenance), consent=replace(a.consent))
    state = replace(
        state, metadata=replace(state.metadata, owner=replace(owner, catalog=(a, distinct)))
    )
    result, _ = roundtrip(state)
    x, y = result.metadata.owner.catalog
    assert x.provenance == y.provenance and x.provenance is not y.provenance
    assert x.consent == y.consent and x.consent is not y.consent


def test_should_preserve_shared_entire_anchor_tuple_graph_with_native_integer_seconds():
    state = sample()
    owner = replace(
        state.metadata.owner, declaration_seconds=state.metadata.owner.declaration_ticks
    )
    state = replace(state, metadata=replace(state.metadata, owner=owner))
    result, _ = roundtrip(state)
    assert result.metadata.owner.declaration_seconds is result.metadata.owner.declaration_ticks
    assert type(result.metadata.owner.declaration_seconds[0][1]) is int


def test_should_accept_native_one_byte_identifiers_without_narrowing_fixed_schema_enums():
    from src.core.data_lifecycle import DataConsent, DataProvenance, LifecycleDeclaration

    state = sample(state="ready", held=False, thread=False)
    item = LifecycleDeclaration(
        ("e", "s"), DataProvenance("q", "p", True, False), DataConsent(True, True), "replay"
    )
    owner = replace(
        state.metadata.owner,
        catalog=(item,),
        opted_out=(),
        revoked_keys=(),
        declaration_ticks=((item.key, 2),),
        declaration_seconds=((item.key, 2.0),),
    )
    state = replace(state, metadata=replace(state.metadata, owner=owner))
    result, _ = roundtrip(state, capture_limits=replace(CAPTURE_LIMITS, max_identifier_bytes=1))
    assert result.metadata.driver.state == "ready"
    assert result.metadata.registry.holders[0].kind == "actor"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_should_refuse_nonfinite_wire_epochs_before_construction(encoded, monkeypatch, value):
    _, port, raw = encoded
    body = json.loads(raw)
    body["metadata"]["driver"]["created_at"] = value
    damaged = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()
    monkeypatch.setattr(module, "materialize_metadata", lambda *a: pytest.fail("construction"))
    with pytest.raises(ValueError):
        port.decode(
            damaged, binding=BINDING, expected_sha256=sha256(damaged).hexdigest(), limits=LIMITS
        )


@pytest.mark.parametrize("kind", ["binding", "limits", "original", "metadata", "policy"])
def test_should_refuse_unknown_native_configuration_and_record_fields(encoded, kind):
    state, port, _ = encoded
    if kind == "binding":
        value = replace(BINDING)
        object.__setattr__(value, "future", True)
        with pytest.raises(ValueError):
            port.encode(state, binding=value, limits=LIMITS)
    elif kind == "limits":
        limit_value = replace(LIMITS)
        object.__setattr__(limit_value, "future", True)
        with pytest.raises(ValueError):
            port.encode(state, binding=BINDING, limits=limit_value)
    elif kind == "policy":
        value = policy(state)
        object.__setattr__(value, "future", True)
        with pytest.raises(ValueError):
            LifecycleCheckpointCodec(value, original=state, authority_sha256=AUTHORITY)
    else:
        value = replace(state, metadata=deepcopy(state.metadata))
        object.__setattr__(value if kind == "original" else value.metadata, "future", True)
        with pytest.raises(ValueError):
            port.encode(value, binding=BINDING, limits=LIMITS)


def test_should_cover_every_complete_current_native_field_with_explicit_schema():
    assert all(
        set(spec) == {field.name for field in fields(kind)} for kind, spec in SCHEMAS.items()
    )
    assert len(SCHEMAS) == 17


def dictionaries(value, path=()):
    result = []
    if type(value) is dict:
        result.append((path, tuple(value)))
        for key, item in value.items():
            result.extend(dictionaries(item, path + (key,)))
    elif type(value) is list:
        for index, item in enumerate(value):
            result.extend(dictionaries(item, path + (index,)))
    return result


SCHEMA_TREE = dict(
    metadata=project_metadata(sample().metadata),
    policy=project_metadata(policy(sample()), LifecycleCodecPolicy),
)
RECORD_CASES = dictionaries(SCHEMA_TREE)
FIELD_CASES = [(path, name) for path, names in RECORD_CASES for name in names]


def at(body, path):
    for item in path:
        body = body[item]
    return body


def refuse_mutation(encoded, monkeypatch, change):
    _, port, raw = encoded
    body = json.loads(raw)
    change(body)
    damaged = canonical(body)
    monkeypatch.setattr(
        module, "materialize_metadata", lambda *a: pytest.fail("materialized refused wire")
    )
    with pytest.raises(ValueError):
        port.decode(
            damaged, binding=BINDING, expected_sha256=sha256(damaged).hexdigest(), limits=LIMITS
        )


@pytest.mark.parametrize("path,name", FIELD_CASES)
def test_should_refuse_every_missing_full_record_field_before_materialization(
    encoded, monkeypatch, path, name
):
    refuse_mutation(encoded, monkeypatch, lambda body: at(body, path).pop(name))


@pytest.mark.parametrize("path,names", RECORD_CASES)
def test_should_refuse_unknown_fields_at_every_nested_record_before_materialization(
    encoded, monkeypatch, path, names
):
    refuse_mutation(encoded, monkeypatch, lambda body: at(body, path).__setitem__("future", True))


@pytest.mark.parametrize(
    "name",
    [
        "codec_version",
        "kind",
        "binding",
        "policy",
        "authority_sha256",
        "reference_manifest",
        "metadata",
        "metadata_aliases",
    ],
)
def test_should_refuse_missing_root_envelope_field(encoded, monkeypatch, name):
    refuse_mutation(encoded, monkeypatch, lambda body: body.pop(name))


@pytest.mark.parametrize("index", range(48))
def test_should_refuse_every_omitted_original_reference_slot_before_materialization(
    encoded, monkeypatch, index
):
    refuse_mutation(encoded, monkeypatch, lambda body: body["reference_manifest"].pop(index))


@pytest.mark.parametrize(
    "change",
    [
        ("codec_version", True),
        ("codec_version", 2),
        ("kind", "lifecycle_scalar_v1"),
        ("authority_sha256", "d" * 64),
        ("future", True),
    ],
)
def test_should_refuse_unknown_version_authority_or_scalar_substitute(encoded, monkeypatch, change):
    refuse_mutation(encoded, monkeypatch, lambda body: body.__setitem__(*change))


BAD_VALUES = [
    (("owner", "catalog", 0, "consent", "training"), False),
    (("owner", "catalog", 0, "consent", "replay"), False),
    (("owner", "catalog", 0, "provenance", "verified"), False),
    (("owner", "catalog", 0, "provenance", "synthetic"), True),
    (("owner", "catalog", 0, "retention"), "transient"),
    (("owner", "catalog", 0, "key"), ["episode"]),
    (("owner", "catalog", 0, "provenance", "source_id"), "é" * 65),
    (("owner", "catalog", 0, "provenance", "source_id"), "\ud800"),
    (("owner", "opted_out"), ["unknown"]),
    (("owner", "revoked_keys"), [["episode", "unknown"]]),
    (("owner", "declaration_ticks"), []),
    (("owner", "declaration_seconds"), []),
    (("owner", "declaration_ticks", 0, 1), 9),
    (("owner", "declaration_seconds", 0, 1), 9.0),
    (("lifecycle", "admitted_bytes"), True),
    (("lifecycle", "admitted_bytes"), 2049),
    (("lifecycle", "last_tick"), -1),
    (("lifecycle", "last_tick"), 2**63),
    (("lifecycle", "last_seconds"), None),
    (("lifecycle", "last_seconds"), -1.0),
    (("lifecycle", "auxiliary_started_at"), 9),
    (("lifecycle", "failed"), 0),
    (("lifecycle", "retention_fault"), 1),
    (("registry", "holders", 0, "enrollment"), 0),
    (("registry", "holders", 1, "enrollment"), 1),
    (("registry", "holders", 2, "enrollment"), 5),
    (("registry", "holders", 0, "kind"), "foreign"),
    (("registry", "holders", 1, "ready"), "False"),
    (("registry", "total_enrollments"), 17),
    (("copy", "charged_bytes"), 63),
    (("copy", "charged_bytes"), 2049),
    (("copy",), None),
    (("driver",), None),
    (("driver", "polls"), 5),
    (("driver", "cleanup_attempts"), 7),
    (("driver", "purges"), 4),
    (("driver", "created_at"), -1),
    (("driver", "state"), "future"),
    (("driver", "pending"), False),
    (("driver", "thread_present"), False),
    (("driver", "stop_set"), 1),
    (("driver", "wake_set"), None),
    (("driver", "purged_set"), 0),
    (("driver", "error_type"), "RuntimeError" * 20),
]


@pytest.mark.parametrize("path,value", BAD_VALUES)
def test_should_refuse_corrupt_native_scalars_policy_relationships_and_histories_before_construction(
    encoded, monkeypatch, path, value
):
    refuse_mutation(
        encoded,
        monkeypatch,
        lambda body: at(body["metadata"], path[:-1]).__setitem__(path[-1], value),
    )


@pytest.mark.parametrize("kind", ["catalog", "ticks", "seconds", "optouts", "revoked", "holders"])
def test_should_refuse_duplicate_ordered_histories_before_construction(encoded, monkeypatch, kind):
    names = {
        "catalog": ("owner", "catalog"),
        "ticks": ("owner", "declaration_ticks"),
        "seconds": ("owner", "declaration_seconds"),
        "optouts": ("owner", "opted_out"),
        "revoked": ("owner", "revoked_keys"),
        "holders": ("registry", "holders"),
    }

    def duplicate(body):
        value = at(body["metadata"], names[kind])
        value.append(deepcopy(value[0]))

    refuse_mutation(encoded, monkeypatch, duplicate)


@pytest.mark.parametrize(
    "change",
    ["missing", "extra", "forward", "boolean", "wrong_type", "holder_policy", "copy_policy"],
)
def test_should_refuse_corrupt_or_missing_complete_immutable_aliases(encoded, monkeypatch, change):
    def corrupt(body):
        aliases = body["metadata_aliases"]
        if change == "missing":
            aliases.pop()
        elif change == "extra":
            aliases.append(0)
        elif change == "forward":
            aliases[0] = 1
        elif change == "boolean":
            aliases[0] = False
        elif change == "wrong_type":
            aliases[1] = 0
        else:
            # Find native alias indices independently from the wire table.
            from src.adapters.lifecycle_checkpoint_schema import preflight_metadata

            nodes = preflight_metadata(body["metadata"], policy(encoded[0]), canonical)
            paths = {node[0]: index for index, node in enumerate(nodes)}
            name = (
                "metadata.registry.limits" if change == "holder_policy" else "metadata.copy.limits"
            )
            index = paths[name]
            aliases[index] = index

    refuse_mutation(encoded, monkeypatch, corrupt)


def test_should_refuse_conflicting_shared_parent_and_child_aliases_before_construction(monkeypatch):
    state = sample()
    owner = replace(
        state.metadata.owner, declaration_seconds=state.metadata.owner.declaration_ticks
    )
    state = replace(state, metadata=replace(state.metadata, owner=owner))
    port = codec(state)
    raw = port.encode(state, binding=BINDING, limits=LIMITS)

    def corrupt(body):
        from src.adapters.lifecycle_checkpoint_schema import preflight_metadata

        nodes = preflight_metadata(body["metadata"], policy(state), canonical)
        index = next(
            i for i, node in enumerate(nodes) if node[0] == "metadata.owner.declaration_seconds[0]"
        )
        body["metadata_aliases"][index] = index

    refuse_mutation((state, port, raw), monkeypatch, corrupt)


@pytest.mark.parametrize(
    "which",
    ["source", "policy", "authority", "reference_present", "reference_alias", "reference_path"],
)
def test_should_refuse_independently_expected_binding_and_reference_manifest_mismatch(
    encoded, monkeypatch, which
):
    def corrupt(body):
        if which in ("source", "policy"):
            body["binding"][which + "_sha256"] = "d" * 64
        elif which == "authority":
            body["authority_sha256"] = "d" * 64
        else:
            row = body["reference_manifest"][0]
            row[
                {
                    "reference_present": "present",
                    "reference_alias": "first",
                    "reference_path": "path",
                }[which]
            ] = (
                False
                if which == "reference_present"
                else ("future" if which == "reference_path" else 1)
            )

    refuse_mutation(encoded, monkeypatch, corrupt)


def test_should_refuse_duplicate_json_keys_noncanonical_bytes_and_changed_digest_before_construction(
    encoded, monkeypatch
):
    _, port, raw = encoded
    monkeypatch.setattr(module, "materialize_metadata", lambda *a: pytest.fail("construction"))
    for bad in (
        b'{"codec_version":1,' + raw[1:],
        json.dumps(json.loads(raw), indent=2).encode(),
        raw + b" ",
    ):
        with pytest.raises(ValueError):
            port.decode(
                bad, binding=BINDING, expected_sha256=sha256(bad).hexdigest(), limits=LIMITS
            )
    with pytest.raises(ValueError):
        port.decode(raw, binding=BINDING, expected_sha256="0" * 64, limits=LIMITS)


def test_should_enforce_exact_wire_bound_and_aggregate_record_bound(encoded, monkeypatch):
    state, port, raw = encoded
    exact = replace(LIMITS, max_encoded_bytes=len(raw))
    assert port.encode(state, binding=BINDING, limits=exact) == raw
    assert (
        port.decode(
            raw, binding=BINDING, expected_sha256=sha256(raw).hexdigest(), limits=exact
        ).metadata
        == state.metadata
    )
    for operation in ("encode", "decode"):
        with pytest.raises(ValueError):
            if operation == "encode":
                port.encode(
                    state, binding=BINDING, limits=replace(exact, max_encoded_bytes=len(raw) - 1)
                )
            else:
                port.decode(
                    raw,
                    binding=BINDING,
                    expected_sha256=sha256(raw).hexdigest(),
                    limits=replace(exact, max_encoded_bytes=len(raw) - 1),
                )
    total = sum(
        len(getattr(state.metadata.owner, name))
        for name in (
            "catalog",
            "opted_out",
            "revoked_keys",
            "declaration_ticks",
            "declaration_seconds",
        )
    ) + len(state.metadata.registry.holders)
    roundtrip(state, capture_limits=replace(CAPTURE_LIMITS, max_records=total))
    with pytest.raises(ValueError):
        codec(state, capture_limits=replace(CAPTURE_LIMITS, max_records=total - 1))


@pytest.mark.parametrize("path", sorted({item.path for item in sample().authority}))
def test_should_refuse_replaced_original_live_reference_without_foreign_equality(encoded, path):
    state, port, _ = encoded
    changed = replace(
        state,
        authority=tuple(
            AuthorityReference(r.path, object()) if r.path == path else r for r in state.authority
        ),
    )
    with pytest.raises(ValueError):
        port.encode(changed, binding=BINDING, limits=LIMITS)


@pytest.mark.parametrize(
    "field,value", [("max_records", True), ("max_records", 2**63), ("max_identifier_bytes", 0)]
)
def test_should_refuse_invalid_independent_capture_policy(field, value):
    state = sample()
    with pytest.raises(ValueError):
        policy(state, replace(CAPTURE_LIMITS, **{field: value}))


class Port:
    def __init__(self, result):
        self.result, self.reads, self.blocked = result, 0, False

    def __call__(self, *args):
        if self.blocked:
            pytest.fail("codec/capture invoked original clock or measurement port")
        self.reads += 1
        return self.result(*args) if callable(self.result) else self.result


@pytest.fixture(scope="module")
def actual(request):
    from src.adapters.numpy_learners import BackpropLearner
    from src.app.actor_shadow import ActorShadowRuntime
    from src.app.managed_data_lifecycle import ManagedDataLifecycle
    from src.app.managed_experience import ManagedExperienceOwner
    from src.app.managed_lifecycle_capture import capture_managed_lifecycle
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
    from src.app.retention_expiry import RetentionExpiryDriver
    from src.app.serving_promotion import PromotableActor
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.core.backprop_mlp import BackpropMLP
    from src.core.data_erasure import ReplayPayloadErasure
    from src.core.data_lifecycle import (
        DataConsent,
        DataProvenance,
        LifecycleDeclaration,
        LifecycleLimits,
    )
    from src.core.data_retention import DataRetentionPolicy
    from src.core.experience import Experience, ExperiencePermissions, LogicalClock
    from src.core.payload_bytes import PayloadCopyLimits
    from src.core.payload_ownership import PayloadOwnershipLimits
    from src.core.resource_sharing import SharingLimits
    from src.core.retention_driver import RetentionDriverLimits
    from src.core.serving_ports import ServingConfiguration

    results = []
    with pytest.MonkeyPatch.context() as patch:
        for name in ("train_batch", "predict", "snapshot_state", "restore_state"):
            patch.setattr(BackpropLearner, name, lambda *a, **k: pytest.fail("native operation"))
        for promotable in (False, True):
            wall = Port(8.0)
            budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), clock=wall)
            learner = BackpropLearner(BackpropMLP(3, 2, seed=23), learning_rate=0.01)
            actor = (
                PromotableActor(
                    learner,
                    version="actor",
                    configuration=ServingConfiguration(2, 2),
                    feature_digest=lambda x: "unused",
                    metadata={"origin": "owned"},
                )
                if promotable
                else None
            )
            runtime = ActorShadowRuntime(
                learner,
                actor_version="actor",
                candidate_version="candidate",
                clock=LogicalClock(0),
                budget=budget,
                max_experiences=4,
                actor=actor,
            )
            owner = ManagedExperienceOwner(
                ResourceSharedRuntime(
                    runtime,
                    ServingPriorityGate(SharingLimits(2, 0, 1), resource_available=lambda: True),
                ),
                limits=LifecycleLimits(4, 4),
            )
            ports = [
                Port(lambda value: int(value.nbytes)),
                Port(ReplayPayloadErasure(0, 0, 0)),
                Port(0),
                Port(0),
                Port(0),
                Port(0),
                Port(0),
            ]
            life = ManagedDataLifecycle(
                owner,
                policy=DataRetentionPolicy(
                    4096,
                    20,
                    PayloadOwnershipLimits(8, 12),
                    owned_payload_copies=PayloadCopyLimits(4096),
                    max_retention_seconds=2.0,
                ),
                measure_payload_bytes=ports[0],
                native_footprint=ports[1],
                native_erase=lambda x: pytest.fail("native erase"),
                measure_auxiliary_bytes=ports[2],
                measure_checkpoint_bytes=ports[3],
                native_growth_bytes=ports[4],
                prepare_model_bytes=ports[5],
                prediction_cache_bytes=ports[6],
            )
            driver = RetentionExpiryDriver(life, limits=RetentionDriverLimits(0.01, 64, 0.5))
            provenance, consent = (
                DataProvenance("local", "person", True, False),
                DataConsent(True, True),
            )
            for name in ("one", "two"):
                owner.declare(
                    LifecycleDeclaration(("episode", name), provenance, consent, "replay")
                )
            owner.record_experience(
                Experience(
                    "one",
                    "episode",
                    0,
                    "actor",
                    np.ones((1, 3)),
                    "train",
                    ExperiencePermissions(True, True),
                )
            )
            for port in [wall, *ports]:
                port.blocked = True
            before = [port.reads for port in [wall, *ports]]
            state = capture_managed_lifecycle(owner, limits=CAPTURE_LIMITS)
            results.append((state, owner, life, driver, budget, [wall, *ports], before))
        yield results
        for _, _, life, driver, budget, ports, before in results:
            assert budget.updates_completed == 0 and driver._thread is None
            assert life._copy_budget is not None
            assert life._copy_budget._charged == life._admitted_bytes == 24
            assert [port.reads for port in ports] == before
    ledger = dict(
        graphs=2,
        updates=0,
        predictions=0,
        snapshots=0,
        restores=0,
        sleeps=0,
        structural=0,
        workers=0,
        threads=0,
    )
    folder = Path(request.config.option.basetemp)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "native-work.json").write_text(json.dumps(ledger, indent=2), encoding="utf8")


@pytest.mark.parametrize("index", [0, 1])
def test_should_compose_actual_coherent_capture_with_complete_bytes_and_original_unchanged_authority(
    actual, index
):
    state, owner, life, driver, budget, ports, before = actual[index]
    known_source = [
        "src/core/managed_lifecycle_state.py",
        "src/core/managed_lifecycle_validation.py",
        "src/app/managed_lifecycle_schema.py",
        "src/app/managed_lifecycle_capture.py",
        "src/app/managed_data_lifecycle.py",
        "src/app/retention_expiry.py",
    ]
    binding = CodecBinding(
        sha256(
            canonical({name: sha256(Path(name).read_bytes()).hexdigest() for name in known_source})
        ).hexdigest(),
        sha256(canonical(project_metadata(policy(state), LifecycleCodecPolicy))).hexdigest(),
    )
    port = codec(state)
    raw = port.encode(state, binding=binding, limits=LIMITS)
    result = port.decode(
        raw, binding=binding, expected_sha256=sha256(raw).hexdigest(), limits=LIMITS
    )
    assert result.metadata == state.metadata
    assert port.encode(result, binding=binding, limits=LIMITS) == raw
    assert (
        result.metadata.copy.charged_bytes == 24 and result.metadata.lifecycle.admitted_bytes == 24
    )
    assert result.metadata.driver.created_at == driver._created_at
    assert (
        result.metadata.owner.catalog[0].provenance is result.metadata.owner.catalog[1].provenance
    )
    assert result.metadata.owner.catalog[0].consent is result.metadata.owner.catalog[1].consent
    assert {r.path: r.value for r in result.authority}["root.owner"] is owner
    assert result.authority is state.authority
    assert life._copy_budget._charged == 24 and budget.updates_completed == 0
    assert [port.reads for port in ports] == before and driver._thread is None
    assert b"root.owner" in raw and b"lifecycle_full_v1" in raw
