"""Whole real V2/native records; historical mutations never confer live authority.

The payload/continuity matrix uses complete original bodies, with recomputed whole
identities so malformed records cannot pass solely by matching a hash. Lifecycle
controls inject malformed records while real native/runtime holders are active.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import zlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prospective_runtime_fixtures as actual
from src.app.prospective_generation_ownership import claim_prospective_generation_bundle
from src.app.prospective_runtime_closure import (
    observe_prospective_generation_runtime,
    validate_runtime_observation,
)
from src.core.runtime_payload_schema import validate_runtime_payload

# Actual collisions and detached/primitive state are created before observations.
# Type names are descriptive in V1; an opaque class may share a builtin name.
COLLISION_TYPE = type("list", (), {"__call__": actual.forbidden})
COLLISION_INSTANCE = COLLISION_TYPE()
COLLISION_KEY = type("str", (), {})()
COLLISION_MAPPING = {COLLISION_KEY: actual.forbidden}
DETACHED = types.FunctionType(actual.forbidden.__code__, {}, "detached_schema_fixture")
setattr(DETACHED, "__module__", None)
DETACHED.__defaults__ = (
    None,
    True,
    -7,
    float("nan"),
    float("inf"),
    -0.0,
    b"",
    ("a", [1, {"x": 2}]),
    object(),
    COLLISION_INSTANCE,
    "Ã©â‚¬æ¼¢ðŸ˜€\ud800\udfff",
    "quotes'\"\\u0061\x00\t\r\n",
)
DETACHED.__kwdefaults__ = {"state": 1}
CLOSED = actual.closure_callable()
# Preloaded fixture failure text avoids a marshal reference-flag change when
# an exception holds the message while the real observer performs final checks.
CONSUMER_FAILURE = "schema fixture consumer failure"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def identity(raw):
    return {"byte_count": len(raw), "sha256": sha256(raw).hexdigest()}


def publish(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
    return identity(raw)


def record_observation(out, name, snapshot, observation):
    raw = observation.runtime_json.encode()
    encoded = zlib.compress(raw, 6)
    assert zlib.decompress(encoded) == raw
    compressed_id = publish(out / (name + ".json.zlib"), encoded)
    header = asdict(observation)
    header.pop("runtime_json")
    return {
        "name": name,
        "complete_runtime_identity": identity(raw),
        "complete_compressed_runtime_identity": compressed_id,
        "header": header,
        "whole_snapshot_identity": identity(canonical(asdict(snapshot))),
        "whole_snapshot_compressed_identity": publish(
            out / (name + "-snapshot.json.zlib"), zlib.compress(canonical(asdict(snapshot)), 6)
        ),
        "historical_record_not_live_authority": True,
    }


def corrupted(observation, body):
    raw = canonical(body)
    return replace(
        observation, runtime_json=raw.decode(), runtime_identity=actual.raw_identity(raw)
    )


def require_denial(name, operation, outcomes, error_type: type[Exception] = ValueError):
    try:
        operation()
    except error_type as error:
        outcomes.append({"name": name, "passed": True, "denial": str(error)})
    else:
        raise AssertionError("missing runtime-record denial: " + name)


@contextmanager
def complete_bundle():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        bundle_root, registry = root / "bundle", root / "registry"
        bundle_root.mkdir()
        registry.mkdir()
        bundle = actual.make_generation_bundle(bundle_root)
        reader = actual.FileProspectiveGenerationBundleReader(bundle_root)
        owner = actual.FileGenerationRequestOwner(bundle_root, registry)
        yield bundle, reader, owner


def capture_complete(out, name):
    observer = actual.ProcessGenerationRuntimeObserver()
    rows, records = [], []
    with complete_bundle() as (bundle, reader, owner):
        design = actual.fixed_prospective_design()
        with claim_prospective_generation_bundle(design, bundle.spec, reader, owner) as held:
            snapshot = held.bundle.snapshot
            with observer.freeze(snapshot, actual.ENTRYPOINTS) as lease:
                for _ in range(3):
                    rows.append(lease.observe(snapshot))
                assert not actual.CALLS
            for index, observation in enumerate(rows):
                records.append(
                    record_observation(out, name + "-" + str(index), snapshot, observation)
                )
        assert not observer._active
    return snapshot, rows, records


def payload_controls(body):
    # Each mutation retains the full body and rebuilds the enclosing byte hash.
    function_index = next(
        index
        for index, row in enumerate(body["python"]["functions"])
        if row["object_id"] == id(DETACHED)
    )
    closed_index = next(
        index for index, row in enumerate(body["python"]["functions"]) if row["closure"]
    )
    module_index = next(
        index for index, row in enumerate(body["python"]["modules"]) if row["file"] is not None
    )
    return [
        ("foreign_python_record", lambda value: value.__setitem__("python", [])),
        (
            "foreign_module_record",
            lambda value: value["python"]["modules"].__setitem__(0, "foreign"),
        ),
        ("missing_module_field", lambda value: value["python"]["modules"][0].pop("members")),
        ("foreign_module_field", lambda value: value["python"]["modules"][0].update(foreign=True)),
        ("boolean_module_id", lambda value: value["python"]["modules"][0].update(object_id=True)),
        (
            "float_namespace_id",
            lambda value: value["python"]["modules"][0].update(namespace_id=1.0),
        ),
        (
            "duplicate_module_name",
            lambda value: value["python"]["modules"].append(value["python"]["modules"][0]),
        ),
        ("module_order", lambda value: value["python"]["modules"].reverse()),
        (
            "foreign_function_record",
            lambda value: value["python"]["functions"].__setitem__(0, False),
        ),
        ("missing_function_field", lambda value: value["python"]["functions"][0].pop("closure")),
        (
            "foreign_function_field",
            lambda value: value["python"]["functions"][0].update(foreign=[]),
        ),
        (
            "boolean_function_id",
            lambda value: value["python"]["functions"][0].update(object_id=True),
        ),
        ("float_globals_id", lambda value: value["python"]["functions"][0].update(globals_id=1.0)),
        (
            "duplicate_function_id",
            lambda value: value["python"]["functions"].append(value["python"]["functions"][0]),
        ),
        ("function_order", lambda value: value["python"]["functions"].reverse()),
        (
            "foreign_code_member",
            lambda value: value["python"]["functions"][0]["code"].update(foreign=1),
        ),
        (
            "missing_code_member",
            lambda value: value["python"]["functions"][0]["code"].pop("filename"),
        ),
        (
            "boolean_code_size",
            lambda value: value["python"]["functions"][0]["code"].update(byte_count=True),
        ),
        (
            "invalid_code_hash",
            lambda value: value["python"]["functions"][0]["code"].update(sha256="foreign"),
        ),
        (
            "float_code_line",
            lambda value: value["python"]["functions"][0]["code"].update(first_line=1.0),
        ),
        (
            "wrong_native_defaults_primitive",
            lambda value: value["python"]["functions"][function_index].update(
                defaults=["int", "1"]
            ),
        ),
        (
            "wrong_native_keyword_defaults_container",
            lambda value: value["python"]["functions"][function_index].update(
                keyword_defaults=["tuple", 1, []]
            ),
        ),
        (
            "foreign_default_tag",
            lambda value: value["python"]["functions"][function_index].update(
                defaults=["foreign", "1"]
            ),
        ),
        (
            "foreign_default_size",
            lambda value: value["python"]["functions"][function_index].update(
                defaults=["bytes", True, "0" * 64]
            ),
        ),
        (
            "noncanonical_default_integer",
            lambda value: value["python"]["functions"][function_index].update(
                defaults=["int", "01"]
            ),
        ),
        (
            "coerced_default_boolean",
            lambda value: value["python"]["functions"][function_index].update(
                defaults=["bool", "1"]
            ),
        ),
        (
            "foreign_keyword_defaults",
            lambda value: value["python"]["functions"][function_index].update(keyword_defaults={}),
        ),
        (
            "foreign_closure_cell",
            lambda value: value["python"]["functions"][closed_index]["closure"].__setitem__(0, []),
        ),
        (
            "boolean_closure_cell_id",
            lambda value: value["python"]["functions"][closed_index]["closure"][0].__setitem__(
                0, True
            ),
        ),
        ("foreign_namespace_record", lambda value: value["python"]["namespaces"].__setitem__(0, 1)),
        ("missing_namespace_field", lambda value: value["python"]["namespaces"][0].pop("type_id")),
        ("boolean_owner_id", lambda value: value["python"]["namespaces"][0].update(owner_id=True)),
        (
            "duplicate_namespace_id",
            lambda value: value["python"]["namespaces"].append(value["python"]["namespaces"][0]),
        ),
        ("namespace_order", lambda value: value["python"]["namespaces"].reverse()),
        (
            "foreign_namespace_edge",
            lambda value: value["python"]["namespaces"][0]["members"].__setitem__(0, [[], False]),
        ),
        (
            "foreign_import_member",
            lambda value: value["python"]["import_state"].update(foreign=True),
        ),
        ("missing_import_member", lambda value: value["python"]["import_state"].pop("path_hooks")),
        ("foreign_import_path", lambda value: value["python"]["import_state"]["path"].append(1)),
        (
            "boolean_import_identity",
            lambda value: value["python"]["import_state"].update(modules_id=True),
        ),
        ("foreign_entrypoint_record", lambda value: value["entrypoints"].__setitem__(0, False)),
        ("entrypoint_field_omission", lambda value: value["entrypoints"][0].pop()),
        ("boolean_entrypoint_id", lambda value: value["entrypoints"][0].__setitem__(1, True)),
        ("entrypoint_globals_join", lambda value: value["entrypoints"][0].__setitem__(3, 1)),
        ("entrypoint_code_join", lambda value: value["entrypoints"][0][2].update(sha256="0" * 64)),
        (
            "entrypoint_function_omission",
            lambda value: value["python"]["functions"].__setitem__(
                slice(None),
                [
                    row
                    for row in value["python"]["functions"]
                    if row["object_id"] != value["entrypoints"][0][1]
                ],
            ),
        ),
        (
            "foreign_physical_file_field",
            lambda value: value["python"]["modules"][module_index]["file"].update(foreign=1),
        ),
        (
            "coerced_physical_inode",
            lambda value: value["python"]["modules"][module_index]["file"].update(inode=1.0),
        ),
        (
            "boolean_physical_link_count",
            lambda value: value["python"]["modules"][module_index]["file"].update(link_count=True),
        ),
        (
            "noncanonical_file_path",
            lambda value: value["python"]["modules"][module_index]["file"].update(
                path="relative.py"
            ),
        ),
        ("foreign_native_platform", lambda value: value["native"].update(platform=True)),
        ("foreign_image_record", lambda value: value["native"]["images"].__setitem__(0, [])),
        ("boolean_image_base", lambda value: value["native"]["images"][0].update(base=True)),
        (
            "duplicate_image_base",
            lambda value: value["native"]["images"].append(value["native"]["images"][0]),
        ),
        ("image_order", lambda value: value["native"]["images"].reverse()),
        (
            "foreign_memory_record",
            lambda value: value["native"]["executable_regions"].__setitem__(0, "foreign"),
        ),
        (
            "missing_memory_field",
            lambda value: value["native"]["executable_regions"][0].pop("allocation_base"),
        ),
        (
            "float_memory_size",
            lambda value: value["native"]["executable_regions"][0].update(size=1.0),
        ),
        (
            "memory_size_identity_join",
            lambda value: value["native"]["executable_regions"][0]["identity"].update(byte_count=1),
        ),
        (
            "invalid_memory_hash",
            lambda value: value["native"]["executable_regions"][0]["identity"].update(
                sha256="foreign"
            ),
        ),
        (
            "nonexecutable_memory_protection",
            lambda value: value["native"]["executable_regions"][0].update(protect=4),
        ),
        (
            "guarded_memory_protection",
            lambda value: value["native"]["executable_regions"][0].update(protect=0x140),
        ),
        (
            "overlapping_memory",
            lambda value: value["native"]["executable_regions"][1].update(
                base=value["native"]["executable_regions"][0]["base"]
            ),
        ),
        ("memory_order", lambda value: value["native"]["executable_regions"].reverse()),
        ("foreign_top_member", lambda value: value.update(foreign=1)),
        ("false_source_attestation", lambda value: value.update(source_attestation=True)),
    ]


def run_payload(out, outcomes):
    snapshot, observations, records = capture_complete(out, "payload")
    first = observations[0]
    body = json.loads(first.runtime_json)
    validate_runtime_payload(body, actual.ENTRYPOINTS)
    validate_runtime_observation(snapshot, first)
    assert id(DETACHED) in {row["object_id"] for row in body["python"]["functions"]}
    for name, change in payload_controls(body):
        malformed = deepcopy(body)
        change(malformed)
        # The digest correctly covers all the malformed content. Schema must
        # reject the content itself rather than merely notice a stale hash.
        observation = corrupted(first, malformed)
        require_denial(name, lambda: validate_runtime_observation(snapshot, observation), outcomes)
    return records


def run_continuity(out, outcomes):
    snapshot, rows, records = capture_complete(out, "continuity")
    first, entry, final = rows
    validate_runtime_observation(snapshot, first)
    validate_runtime_observation(snapshot, entry, first)
    validate_runtime_observation(snapshot, final, entry)
    changed_scope = replace(first.scope, owner_id="7" * 32)
    future = (datetime.fromisoformat(final.observed_utc) + timedelta(seconds=1)).isoformat()
    variants = [
        ("foreign_previous", object()),
        ("boolean_previous_pid", replace(first, process_id=True)),
        ("float_previous_sequence", replace(first, sequence=0.0)),
        ("invalid_previous_nonce", replace(first, lease_nonce="foreign")),
        ("foreign_previous_scope", replace(first, scope=object())),
        ("changed_previous_request_owner", replace(first, scope=changed_scope)),
        ("false_previous_completeness", replace(first, complete_process_membership_observed=False)),
        ("false_previous_source_attestation", replace(first, runtime_source_version_attested=True)),
        ("previous_body_type", replace(first, runtime_json=b"foreign")),
        ("previous_identity_type", replace(first, runtime_identity=object())),
        ("previous_clock_future", replace(first, observed_utc=future)),
        ("previous_clock_type", replace(first, observed_utc=True)),
        ("previous_nonce_drift", replace(first, lease_nonce="6" * 32)),
        ("previous_pid_drift", replace(first, process_id=first.process_id + 1)),
        ("previous_body_drift", replace(first, runtime_json="{}")),
        ("previous_entrypoint_type", replace(first, entrypoints=list(first.entrypoints))),
        ("previous_sequence_gap", replace(first, sequence=5)),
    ]
    for name, previous in variants:
        require_denial(
            name, lambda: validate_runtime_observation(snapshot, entry, previous), outcomes
        )
    require_denial(
        "current_initial_sequence", lambda: validate_runtime_observation(snapshot, entry), outcomes
    )
    require_denial(
        "current_sequence_gap",
        lambda: validate_runtime_observation(snapshot, replace(entry, sequence=3), first),
        outcomes,
    )
    require_denial(
        "current_scope_foreign",
        lambda: validate_runtime_observation(snapshot, replace(first, scope=object())),
        outcomes,
    )
    require_denial(
        "current_utf8_foreign",
        lambda: validate_runtime_observation(snapshot, replace(first, runtime_json="\ud800")),
        outcomes,
    )
    return records


class CorruptLease:
    def __init__(self, delegate, wrapper):
        self.delegate, self.wrapper = delegate, wrapper

    def observe(self, snapshot):
        result = self.delegate.observe(snapshot)
        self.wrapper.observations.append(result)
        self.wrapper.snapshot = snapshot
        if result.sequence == self.wrapper.corrupt_sequence:
            body = json.loads(result.runtime_json)
            body["python"]["modules"][0]["foreign"] = True
            return corrupted(result, body)
        return result


class CorruptObserver:
    def __init__(self, corrupt_sequence):
        self.runtime = actual.ProcessGenerationRuntimeObserver()
        self.corrupt_sequence = corrupt_sequence
        self.observations = []
        self.snapshot = None
        self.last_lease = None

    @contextmanager
    def freeze(self, snapshot, entrypoints):
        with self.runtime.freeze(snapshot, entrypoints) as delegate:
            self.last_lease = delegate
            yield CorruptLease(delegate, self)


def error_chain(error):
    chain = []
    while error is not None:
        chain.append(type(error).__name__ + ": " + str(error))
        error = error.__context__
    return chain


def run_lifecycle(out, outcomes):
    records = []
    for name, sequence in (
        ("first", 0),
        ("entry", 1),
        ("final", 2),
        ("consumer", -1),
        ("request", -1),
    ):
        wrapper = CorruptObserver(sequence)
        with complete_bundle() as (bundle, reader, owner):
            saved = (bundle.root / "request.json").read_bytes()
            yielded = False
            try:
                with observe_prospective_generation_runtime(
                    actual.fixed_prospective_design(),
                    bundle.spec,
                    reader,
                    owner,
                    wrapper,
                    actual.ENTRYPOINTS,
                ) as result:
                    yielded = True
                    assert result.runtime_process_observed and not result.execution_authorized
                    if name == "consumer":
                        raise RuntimeError(CONSUMER_FAILURE)
                    if name == "request":
                        (bundle.root / "request.json").write_bytes(saved + b" ")
            except (ValueError, RuntimeError) as error:
                chain = error_chain(error)
                assert chain
                assert not yielded if name in {"first", "entry"} else yielded
                if name in {"first", "entry", "final"}:
                    assert any("runtime" in message for message in chain), chain
                if name == "consumer":
                    assert any("consumer failure" in message for message in chain)
                if name == "request":
                    assert any("whole byte count" in message for message in chain)
            else:
                raise AssertionError("missing lifecycle failure " + name)
            finally:
                (bundle.root / "request.json").write_bytes(saved)
            expected = 2 if name == "first" else 3
            assert len(wrapper.observations) == expected, (name, len(wrapper.observations))
            lease = wrapper.last_lease
            assert lease is not None
            assert not wrapper.runtime._active and not lease._active
            require_denial(
                "inactive_" + name,
                lambda: lease.observe(wrapper.snapshot),
                outcomes,
                RuntimeError,
            )
            # Reacquire the real native request owner after every failure.
            with claim_prospective_generation_bundle(
                actual.fixed_prospective_design(), bundle.spec, reader, owner
            ):
                pass
            for index, row in enumerate(wrapper.observations):
                records.append(
                    record_observation(out, name + "-" + str(index), wrapper.snapshot, row)
                )
            outcomes.append(
                {
                    "name": "finally_release_" + name,
                    "passed": True,
                    "all_actual_observation_sequences": [
                        row.sequence for row in wrapper.observations
                    ],
                    "earliest_denial_chain": chain,
                }
            )
    return records


def main():
    group, destination = sys.argv[1:]
    out = Path(destination)
    assert not out.exists()
    out.mkdir()
    actual.install_guards()
    outcomes: list[dict[str, object]] = []
    records = {"payload": run_payload, "continuity": run_continuity, "lifecycle": run_lifecycle}[
        group
    ](out, outcomes)
    assert not actual.CALLS and outcomes and records
    report = {
        "schema_id": "p67_actual_runtime_payload_control_group_v1",
        "passed": True,
        "group": group,
        "science_calls": actual.CALLS,
        "actual_complete_observations": len(records),
        "whole_observation_records": records,
        "all_control_names_and_outcomes": outcomes,
        "runtime_source_version_attested": False,
        "execution_authorized": False,
    }
    publish(out / "validation.json", canonical(report) + b"\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "whole_observation_records"}
        )
    )


if __name__ == "__main__":
    main()
