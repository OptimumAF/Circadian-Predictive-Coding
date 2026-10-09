"""Explicit lifecycle wire fields, scalar relationships and immutable aliases.

Preflight operates on bounded JSON containers before typed record construction.
Projection/materialization touch metadata only. No live authority, IO or arrays.
"""

from math import isfinite
from typing import Any
from src.core.data_lifecycle import (
    DataConsent,
    DataProvenance,
    LifecycleDeclaration,
    LifecycleLimits,
)
from src.core.data_retention import DataRetentionPolicy
from src.core.retention_driver import RetentionDriverLimits
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.experience import require_identifier
from src.core.lifecycle_codec_policy import LifecycleCodecPolicy
from src.core.managed_lifecycle_state import (
    CopyBudgetState,
    LifecycleAccountingState,
    LifecycleCaptureLimits,
    LifecycleMetadata,
    ManagedOwnerState,
    OwnershipEnrollment,
    OwnershipRegistryState,
    RetentionDriverRecord,
)

KEY = ("tuple", "string", "string")
SCHEMAS: dict[type, dict[str, Any]] = {
    LifecycleMetadata: dict(
        format_version="version",
        owner=ManagedOwnerState,
        lifecycle=LifecycleAccountingState,
        registry=OwnershipRegistryState,
        copy=("optional", CopyBudgetState),
        driver=("optional", RetentionDriverRecord),
    ),
    ManagedOwnerState: dict(
        limits=LifecycleLimits,
        catalog=("sequence", LifecycleDeclaration),
        opted_out=("sequence", "string"),
        revoked_keys=("sequence", KEY),
        declaration_ticks=("sequence", ("tuple", KEY, "counter")),
        declaration_seconds=("sequence", ("tuple", KEY, "seconds")),
    ),
    LifecycleAccountingState: dict(
        policy=DataRetentionPolicy,
        admitted_bytes="counter",
        last_tick="counter",
        failed="bool",
        last_seconds=("optional", "seconds"),
        auxiliary_started_at=("optional", "seconds"),
        retention_fault="bool",
    ),
    OwnershipRegistryState: dict(
        limits=PayloadOwnershipLimits,
        total_enrollments="counter",
        holders=("sequence", OwnershipEnrollment),
    ),
    OwnershipEnrollment: dict(
        enrollment="positive_counter", kind="kind", ready=("optional", "bool")
    ),
    CopyBudgetState: dict(limits=PayloadCopyLimits, charged_bytes="counter"),
    RetentionDriverRecord: dict(
        limits=RetentionDriverLimits,
        state="state",
        polls="counter",
        purges="counter",
        cleanup_attempts="counter",
        created_at="seconds",
        held="bool",
        pending="bool",
        error_type=("optional", "string"),
        stop_set="bool",
        wake_set="bool",
        purged_set="bool",
        thread_present="bool",
        thread_alive="bool",
    ),
    LifecycleDeclaration: dict(
        key=KEY, provenance=DataProvenance, consent=DataConsent, retention="retention"
    ),
    DataProvenance: dict(
        source_id="string", subject_id="string", verified="bool", synthetic="bool"
    ),
    DataConsent: dict(training="bool", replay="bool"),
    LifecycleLimits: dict(
        max_approved_records="counter",
        max_replay_records="counter",
        allow_synthetic="bool",
        allow_unverified="bool",
    ),
    DataRetentionPolicy: dict(
        max_lifetime_ingress_bytes="counter",
        max_retention_ticks="positive_counter",
        holders=PayloadOwnershipLimits,
        max_checkpoint_pending="positive_counter",
        max_checkpoint_preparations="positive_counter",
        max_promotion_pending="positive_counter",
        owned_payload_copies=("optional", PayloadCopyLimits),
        max_retention_seconds=("optional", "positive_seconds"),
    ),
    PayloadOwnershipLimits: dict(max_live_holders="counter", max_lifetime_enrollments="counter"),
    PayloadCopyLimits: dict(max_lifetime_owned_bytes="counter"),
    RetentionDriverLimits: dict(
        poll_interval_seconds="positive_seconds",
        max_polls="positive_counter",
        max_run_seconds="positive_seconds",
        join_timeout_seconds="positive_seconds",
    ),
    LifecycleCaptureLimits: dict(max_records="counter", max_identifier_bytes="positive_counter"),
    LifecycleCodecPolicy: dict(
        owner_limits=LifecycleLimits,
        lifecycle_policy=DataRetentionPolicy,
        driver_limits=("optional", RetentionDriverLimits),
        capture_limits=LifecycleCaptureLimits,
    ),
}


def project_metadata(value, spec=LifecycleMetadata, *, schemas=SCHEMAS):
    if isinstance(spec, type):
        return {
            name: project_metadata(getattr(value, name), field, schemas=schemas)
            for name, field in schemas[spec].items()
        }
    if type(spec) is tuple:
        if spec[0] == "optional":
            return None if value is None else project_metadata(value, spec[1], schemas=schemas)
        if spec[0] == "sequence":
            return [project_metadata(item, spec[1], schemas=schemas) for item in value]
        return [
            project_metadata(item, field, schemas=schemas) for item, field in zip(value, spec[1:])
        ]
    return value


def _scalar(value, spec, limits) -> None:
    if spec in ("counter", "positive_counter", "version"):
        if (
            type(value) is not int
            or not 0 <= value < 2**63
            or (spec == "positive_counter" and value == 0)
            or (spec == "version" and value != 1)
        ):
            raise ValueError("lifecycle wire counter/version differs from supported bound")
    elif spec in ("seconds", "positive_seconds"):
        if (
            type(value) not in (int, float)
            or not isfinite(value)
            or value < 0
            or (spec == "positive_seconds" and value == 0)
        ):
            raise ValueError("lifecycle wire requires finite original elapsed seconds")
    elif spec == "bool":
        if type(value) is not bool:
            raise ValueError("lifecycle wire flag requires exact boolean")
    else:
        choices = {
            "kind": ("actor", "candidate", "checkpoint", "promotion"),
            "state": ("ready", "running", "stopping", "stopped", "exhausted", "failed"),
            "retention": ("replay",),
        }
        if spec in choices:
            if type(value) is not str or value not in choices[spec]:
                raise ValueError("unsupported lifecycle wire kind/state/retention")
            return  # Fixed enum spelling is schema, not a caller identifier capacity.
        if (
            type(value) is not str
            or not value
            or len(value) > limits.max_identifier_bytes
            or len(value.encode("utf8")) > limits.max_identifier_bytes
        ):
            raise ValueError("lifecycle wire string exceeds original UTF8 bound")
        require_identifier(value, "lifecycle wire identity")


def _walk(value, spec, limits, nodes, path, *, schemas=SCHEMAS):
    if isinstance(spec, type):
        fields = schemas[spec]
        if type(value) is not dict or value.keys() != fields.keys():
            raise ValueError("lifecycle wire differs from complete explicit record fields")
        nodes.append((path, spec, value))
        for name, field in fields.items():
            _walk(value[name], field, limits, nodes, path + "." + name, schemas=schemas)
    elif type(spec) is tuple:
        if spec[0] == "optional":
            if value is not None:
                _walk(value, spec[1], limits, nodes, path, schemas=schemas)
            return
        if type(value) is not list or (spec[0] == "tuple" and len(value) != len(spec) - 1):
            raise ValueError("lifecycle wire requires complete ordered sequence/key/pair")
        if spec[0] == "sequence" and len(value) > limits.max_records:
            raise ValueError("lifecycle wire sequence exceeds original aggregate capacity")
        nodes.append((path, tuple, value))
        for index, item in enumerate(value):
            _walk(
                item,
                spec[1] if spec[0] == "sequence" else spec[index + 1],
                limits,
                nodes,
                f"{path}[{index}]",
                schemas=schemas,
            )
    elif callable(spec):
        spec(value, limits)  # Trusted internal scalar validator, never selected by wire data.
    else:
        _scalar(value, spec, limits)
    if len(nodes) > 32 * limits.max_records + 128:
        raise ValueError("lifecycle alias records exceed independent capacity")


def _owner_relationships(data, policy):
    owner, life = data["owner"], data["lifecycle"]
    if len(owner["catalog"]) > min(
        policy.owner_limits.max_approved_records, policy.owner_limits.max_replay_records
    ):
        raise ValueError("lifecycle catalog exceeds original lifetime quota")
    keys, subjects = set(), set()
    for item in owner["catalog"]:
        key = tuple(item["key"])
        provenance, consent = item["provenance"], item["consent"]
        if (
            key in keys
            or not consent["training"]
            or not consent["replay"]
            or (provenance["synthetic"] and not policy.owner_limits.allow_synthetic)
            or (not provenance["verified"] and not policy.owner_limits.allow_unverified)
        ):
            raise ValueError("duplicate or forbidden lifecycle catalog consent/provenance")
        keys.add(key)
        subjects.add(provenance["subject_id"])
    opted, revoked = owner["opted_out"], [tuple(key) for key in owner["revoked_keys"]]
    if (
        opted != sorted(set(opted))
        or not set(opted) <= subjects
        or revoked != sorted(set(revoked))
        or not set(revoked) <= keys
    ):
        raise ValueError("lifecycle optout/revocation differs from original catalog")
    _anchor_relationships(owner["declaration_ticks"], keys, life["last_tick"])
    if policy.lifecycle_policy.max_retention_seconds is None:
        if (
            owner["declaration_seconds"]
            or life["last_seconds"] is not None
            or life["auxiliary_started_at"] is not None
        ):
            raise ValueError("unconfigured lifecycle elapsed policy cannot invent epochs")
    else:
        if life["last_seconds"] is None:
            raise ValueError("configured lifecycle requires original seconds observation")
        _anchor_relationships(owner["declaration_seconds"], keys, life["last_seconds"])
        if (
            life["auxiliary_started_at"] is not None
            and life["auxiliary_started_at"] > life["last_seconds"]
        ):
            raise ValueError("lifecycle auxiliary epoch is ahead of original observation")


def _anchor_relationships(entries, keys, last):
    seen = set()
    for key, value in entries:
        key = tuple(key)
        if key in seen or value > last:
            raise ValueError("duplicate or future lifecycle declaration clock")
        seen.add(key)
    if seen != keys:
        raise ValueError("lifecycle declaration clocks omit or invent original identities")


def _accounting_relationships(data, policy, canonical):
    life, registry, copy = data["lifecycle"], data["registry"], data["copy"]
    if life["admitted_bytes"] > policy.lifecycle_policy.max_lifetime_ingress_bytes:
        raise ValueError("lifecycle consumed ingress exceeds original quota")
    if canonical(registry["limits"]) != canonical(
        project_metadata(policy.lifecycle_policy.holders, PayloadOwnershipLimits)
    ):
        raise ValueError("lifecycle registry differs from original holder policy")
    previous, live = 0, 0
    for item in registry["holders"]:
        if not previous < item["enrollment"] <= registry["total_enrollments"]:
            raise ValueError("lifecycle retained enrollment numbering differs")
        previous = item["enrollment"]
        live += item["ready"] is not None
    if (
        live > policy.lifecycle_policy.holders.max_live_holders
        or registry["total_enrollments"] > policy.lifecycle_policy.holders.max_lifetime_enrollments
    ):
        raise ValueError("lifecycle holder quota/enrollment allowance exceeded")
    original = policy.lifecycle_policy.owned_payload_copies
    if (original is None) != (copy is None):
        raise ValueError("lifecycle copy policy presence differs")
    if original is not None and (
        canonical(copy["limits"]) != canonical(project_metadata(original, PayloadCopyLimits))
        or not life["admitted_bytes"] <= copy["charged_bytes"] <= original.max_lifetime_owned_bytes
    ):
        raise ValueError("lifecycle original copy quota/consumed charges differ")


def _driver_relationships(data, policy):
    driver = data["driver"]
    if (driver is None) != (policy.driver_limits is None):
        raise ValueError("lifecycle original driver presence differs")
    if driver is not None and (
        driver["polls"] > policy.driver_limits.max_polls
        or driver["cleanup_attempts"] > policy.driver_limits.max_polls + 2
        or driver["purges"] > driver["cleanup_attempts"]
        or (driver["held"] and not driver["pending"])
        or (driver["thread_alive"] and not driver["thread_present"])
    ):
        raise ValueError("lifecycle driver counters/hold/thread observations differ")


def preflight_metadata(data, policy, canonical):
    nodes: list[tuple[str, type, Any]] = []
    _walk(data, LifecycleMetadata, policy.capture_limits, nodes, "metadata")
    owner = data["owner"]
    if (
        sum(
            len(owner[name])
            for name in (
                "catalog",
                "opted_out",
                "revoked_keys",
                "declaration_ticks",
                "declaration_seconds",
            )
        )
        + len(data["registry"]["holders"])
        > policy.capture_limits.max_records
    ):
        raise ValueError("lifecycle aggregate records exceed original capture bound")
    for value, original, kind in (
        (owner["limits"], policy.owner_limits, LifecycleLimits),
        (data["lifecycle"]["policy"], policy.lifecycle_policy, DataRetentionPolicy),
        (
            None if data["driver"] is None else data["driver"]["limits"],
            policy.driver_limits,
            RetentionDriverLimits,
        ),
    ):
        expected = None if original is None else project_metadata(original, kind)
        if canonical(value) != canonical(expected):
            raise ValueError("lifecycle independently expected original policy differs")
    _owner_relationships(data, policy)
    _accounting_relationships(data, policy, canonical)
    _driver_relationships(data, policy)
    return nodes


def metadata_aliases(metadata, spec=LifecycleMetadata, *, schemas=SCHEMAS):
    nodes = []

    def visit(value, spec):
        if isinstance(spec, type):
            nodes.append(value)
            for name, field in schemas[spec].items():
                visit(getattr(value, name), field)
        elif type(spec) is tuple:
            if spec[0] == "optional":
                if value is not None:
                    visit(value, spec[1])
            else:
                nodes.append(value)
                for index, item in enumerate(value):
                    visit(item, spec[1] if spec[0] == "sequence" else spec[index + 1])

    visit(metadata, spec)
    first: dict[int, int] = {}
    return [first.setdefault(id(value), index) for index, value in enumerate(nodes)]


def preflight_aliases(nodes, aliases, canonical, *, copy_present, prefix="metadata"):
    if type(aliases) is not list or len(aliases) != len(nodes):
        raise ValueError("complete lifecycle immutable alias records are required")
    paths = {node[0]: index for index, node in enumerate(nodes)}
    for index, target in enumerate(aliases):
        if type(target) is not int or not 0 <= target <= index or aliases[target] != target:
            raise ValueError("lifecycle immutable alias must name its original first node")
        if nodes[index][1] is not nodes[target][1] or canonical(nodes[index][2]) != canonical(
            nodes[target][2]
        ):
            raise ValueError("lifecycle shared immutable alias types/bits differ")
        if target != index:

            def descendants(at):
                path = nodes[at][0]
                return [i for i, n in enumerate(nodes) if n[0].startswith((path + ".", path + "["))]

            left, right = descendants(index), descendants(target)
            if len(left) != len(right) or any(
                aliases[a] != aliases[b] for a, b in zip(left, right)
            ):
                raise ValueError("lifecycle shared parent/child alias records conflict")
    required = [(prefix + ".registry.limits", prefix + ".lifecycle.policy.holders")]
    if copy_present:
        required.append(
            (prefix + ".copy.limits", prefix + ".lifecycle.policy.owned_payload_copies")
        )
    if any(aliases[paths[left]] != aliases[paths[right]] for left, right in required):
        raise ValueError("lifecycle original shared holder/copy policy alias is required")


def materialize_metadata(data, aliases, spec=LifecycleMetadata, *, schemas=SCHEMAS):
    """Internal construction only after complete raw preflight and alias validation."""
    position = 0
    memo: dict[int, Any] = {}

    def build(value, spec):
        nonlocal position
        if type(spec) is tuple and spec[0] == "optional":
            return None if value is None else build(value, spec[1])
        if not isinstance(spec, type) and type(spec) is not tuple:
            return value
        index = position
        position += 1
        if isinstance(spec, type):
            result = spec(
                **{name: build(value[name], field) for name, field in schemas[spec].items()}
            )
        else:
            result = tuple(
                build(item, spec[1] if spec[0] == "sequence" else spec[i + 1])
                for i, item in enumerate(value)
            )
        memo[index] = result if aliases[index] == index else memo[aliases[index]]
        return memo[index]

    return build(data, spec)
