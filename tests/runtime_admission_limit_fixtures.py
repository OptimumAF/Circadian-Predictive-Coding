"""Real process counterexamples; complete historical edits grant no live authority.

Inputs are invented guarded V2/native ownership and precompiled diagnostic code.
Outputs save complete actual records, edited negative records, source/code bytes,
audit event names and observed return/lifetime values. Production is unchanged.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import dis
from hashlib import sha256
import json
import marshal
from pathlib import Path
import platform
import sys
import sysconfig
import types
import weakref

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prospective_runtime_fixtures as actual
import runtime_payload_fixtures as payload
from runtime_marshal_value_fixtures import describe_native_code
from src.app.prospective_generation_ownership import claim_prospective_generation_bundle
from src.app.prospective_runtime_closure import (
    observe_prospective_generation_runtime,
    validate_runtime_observation,
)
from src.core.prospective_generation_bundles import GenerationBundleSnapshot
from src.core.prospective_runtime_closure import RuntimeCodeObservation
from src.infra.runtime_python_objects import bind_runtime_entrypoints

MODULE_NAME = "p67_admission_fixture"
SOURCE = "def operation():\n    unused = None\n    return 100003\n"


def transient_operation():
    def created():
        return 23

    reference = weakref.ref(created)
    identity = id(created)
    result = created()
    del created
    assert reference() is None
    return identity, result


def default_operation(value=5):
    return value


class EventRecorder:
    def __init__(self):
        self.active = False
        self.names: list[str] = []

    def __call__(self, event, arguments):
        if self.active:
            # Primitive names only: holding argument code/function objects here
            # would change the observed lifetime/graph and invalidate the control.
            self.names.append(event)


class RecordingLease:
    def __init__(self, delegate, recorder):
        self.delegate, self.recorder = delegate, recorder

    def observe(self, snapshot):
        observation = self.delegate.observe(snapshot)
        self.recorder.rows.append((snapshot, observation))
        return observation


class RecordingObserver:
    def __init__(self):
        self.runtime = actual.ProcessGenerationRuntimeObserver()
        self.rows: list[tuple[GenerationBundleSnapshot, RuntimeCodeObservation]] = []
        self.lease = None

    @contextmanager
    def freeze(self, snapshot, entrypoints):
        with self.runtime.freeze(snapshot, entrypoints) as delegate:
            self.lease = delegate
            yield RecordingLease(delegate, self)


def pin(raw):
    return {"byte_count": len(raw), "sha256": sha256(raw).hexdigest()}


def write(out, name, raw):
    assert Path(name).name == name
    with (out / name).open("xb") as stream:
        stream.write(raw)
    return pin(raw)


def code_record(out, label, stage, code):
    raw = marshal.dumps(code)
    name = label + "-" + stage + ".marshal"
    return {
        "name": name,
        "identity": write(out, name, raw),
        "code_object_id": id(code),
        "complete_public_native_contents": describe_native_code(code),
    }


def edited_record(out, label, snapshot, observation, body, names):
    raw = payload.canonical(body)
    edited = replace(
        observation,
        entrypoints=names,
        runtime_json=raw.decode(),
        runtime_identity=actual.raw_identity(raw),
    )
    validate_runtime_observation(snapshot, edited)
    row = payload.record_observation(out, label + "-0", snapshot, edited)
    row["edited_historical_record_not_actual_or_live_proof"] = True
    return row


def main(out):
    out.mkdir()
    actual.install_guards()
    source_path = out / "claimed-source.py"
    source_identity = write(out, source_path.name, SOURCE.encode())
    module_code = compile(SOURCE, str(source_path), "exec")
    expected_code = module_code.co_consts[0]
    assert type(expected_code) is types.CodeType and expected_code.co_consts == (None, 100003)
    loaded_code = expected_code.replace(co_consts=(None, 100013))
    module = types.ModuleType(MODULE_NAME)
    module.__file__ = str(source_path)
    matched = types.FunctionType(expected_code, module.__dict__, "matched")
    module.__dict__["operation"] = types.FunctionType(loaded_code, module.__dict__, "operation")
    sys.modules[MODULE_NAME] = module
    omitted = types.FunctionType(loaded_code, {}, "unreferenced")
    assert omitted.__module__ is None
    entrypoints = actual.ENTRYPOINTS + (MODULE_NAME + ":operation",)
    recorder = EventRecorder()
    sys.addaudithook(recorder)
    observer = RecordingObserver()
    nested_code = next(
        value for value in transient_operation.__code__.co_consts if type(value) is types.CodeType
    )
    codes = {
        "source-matched": expected_code,
        "source-loaded": loaded_code,
        "transient-factory": transient_operation.__code__,
        "transient-nested": nested_code,
        "default-operation": default_operation.__code__,
    }
    outside_before = [
        code_record(out, label, "pre-retention", code) for label, code in codes.items()
    ]
    instructions = [
        {"offset": row.offset, "opname": row.opname, "arg": row.arg, "argrepr": row.argrepr}
        for row in dis.get_instructions(transient_operation)
    ]
    assert any(row["opname"] == "MAKE_FUNCTION" for row in instructions)
    interpreter = {
        "version": sys.version,
        "hexversion": sys.hexversion,
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "executable": sys.executable,
        "build": platform.python_build(),
        "compiler": platform.python_compiler(),
        "configuration": {
            name: sysconfig.get_config_var(name)
            for name in ("Py_GIL_DISABLED", "Py_DEBUG", "ABIFLAGS", "SOABI", "EXT_SUFFIX")
        },
        "trusted_build_source_correspondence": False,
    }
    controls, intervals = [], []
    with payload.complete_bundle() as (bundle, reader, owner):
        design = actual.fixed_prospective_design()
        with observe_prospective_generation_runtime(
            design, bundle.spec, reader, owner, observer, entrypoints
        ) as observed:
            assert observed.runtime_process_observed and not observed.execution_authorized
            assert (
                not observed.runtime_source_version_attested and not observed.fresh_roles_authorized
            )
            native_before = [
                code_record(out, label, "before", code) for label, code in codes.items()
            ]
            start = len(recorder.names)
            recorder.active = True
            transient_id, transient_value = transient_operation()
            recorder.active = False
            events = recorder.names[start:]
            assert transient_value == 23 and "function.__new__" not in events
            intervals.append(
                {
                    "name": "ordinary-make-function",
                    "every_audit_event_name": events,
                    "actual_function_object_id": transient_id,
                    "actual_return": transient_value,
                    "weak_reference_dead_before_next_observation": True,
                }
            )
            defaults = default_operation.__defaults__
            assert defaults == (5,)
            start = len(recorder.names)
            recorder.active = True
            default_operation.__defaults__ = (9,)
            changed_value = default_operation()
            default_operation.__defaults__ = defaults
            restored_value = default_operation()
            recorder.active = False
            assert (
                changed_value == 9
                and restored_value == 5
                and default_operation.__defaults__ is defaults
            )
            intervals.append(
                {
                    "name": "temporary-default-binding",
                    "every_audit_event_name": recorder.names[start:],
                    "actual_changed_return": changed_value,
                    "actual_restored_return": restored_value,
                    "original_defaults_identity_restored": True,
                }
            )
            code = default_operation.__code__
            start = len(recorder.names)
            recorder.active = True
            try:
                types.FunctionType(code, {})
            except RuntimeError as error:
                denial = str(error)
            else:
                raise AssertionError("direct FunctionType must still be denied")
            finally:
                recorder.active = False
            assert "function.__new__" in denial and "function.__new__" in recorder.names[start:]
            intervals.append(
                {
                    "name": "direct-function-construction",
                    "every_audit_event_name": recorder.names[start:],
                    "denial": denial,
                }
            )
            assert (
                module.operation.__code__.co_filename
                == matched.__code__.co_filename
                == str(source_path)
            )
            assert matched() == 100003 and module.operation() == 100013
            native_used = [code_record(out, label, "used", code) for label, code in codes.items()]
            assert (source_path.read_bytes(), pin(source_path.read_bytes())) == (
                SOURCE.encode(),
                source_identity,
            )
            assert not actual.CALLS
            native_after = [
                code_record(out, label, "pre-final", code) for label, code in codes.items()
            ]
        assert len(observer.rows) == 3 and not observer.runtime._active
        lease = observer.lease
        assert lease is not None and not lease._active
        with claim_prospective_generation_bundle(design, bundle.spec, reader, owner):
            pass
        records = []
        for index, (snapshot, observation) in enumerate(observer.rows):
            previous = observer.rows[index - 1][1] if index else None
            validate_runtime_observation(snapshot, observation, previous)
            records.append(
                payload.record_observation(out, "actual-" + str(index), snapshot, observation)
            )
        assert len({row["complete_runtime_identity"]["sha256"] for row in records}) == 1
        snapshot, first = observer.rows[0]
        body = json.loads(first.runtime_json)
        assert not any(row["object_id"] == transient_id for row in body["python"]["functions"])
        target = next(row for row in body["python"]["functions"] if row["object_id"] == id(omitted))
        omitted_body = deepcopy(body)
        omitted_body["python"]["functions"].remove(target)
        omitted_record = edited_record(out, "omitted", snapshot, first, omitted_body, entrypoints)
        missing_names = entrypoints[:-1] + (MODULE_NAME + ":missing.operation",)
        nested_body = deepcopy(body)
        nested_body["entrypoints"][-1][0] = missing_names[-1]
        nested_record = edited_record(out, "nested", snapshot, first, nested_body, missing_names)
        try:
            bind_runtime_entrypoints(missing_names)
        except ValueError as error:
            route_denial = str(error)
        else:
            raise AssertionError("real binder must reject the missing route")
        controls.extend(
            [
                {
                    "name": "actual-transient-function-execution-missed-between-observations",
                    "passed": True,
                    "admission_gap_demonstrated": True,
                },
                {
                    "name": "actual-temporary-default-execution-missed-between-observations",
                    "passed": True,
                    "admission_gap_demonstrated": True,
                },
                {"name": "actual-direct-function-construction-denied", "passed": True},
                {
                    "name": "actual-source-file-name-is-not-loaded-code-correspondence",
                    "passed": True,
                    "admission_gap_demonstrated": True,
                },
                {
                    "name": "edited-omitted-unreferenced-function-structure-accepted",
                    "passed": True,
                    "admission_gap_demonstrated": True,
                },
                {
                    "name": "edited-missing-nested-route-structure-accepted-real-binding-denied",
                    "passed": True,
                    "admission_gap_demonstrated": True,
                },
                {"name": "actual-finally-release-and-native-owner-reacquisition", "passed": True},
            ]
        )
    outside_after = [code_record(out, label, "post-release", code) for label, code in codes.items()]
    for before, used, after in zip(native_before, native_used, native_after):
        assert before["identity"] == used["identity"] == after["identity"]
        assert (
            before["complete_public_native_contents"]
            == used["complete_public_native_contents"]
            == after["complete_public_native_contents"]
        )
    assert native_before[0]["identity"] != native_before[1]["identity"]
    outside_comparisons = []
    for original, held, released in zip(outside_before, native_before, outside_after):
        assert (
            original["complete_public_native_contents"]
            == held["complete_public_native_contents"]
            == released["complete_public_native_contents"]
        )
        raw = (out / original["name"]).read_bytes()
        differences = []
        for row in (held, released):
            current = (out / row["name"]).read_bytes()
            differences.append(
                {
                    "name": row["name"],
                    "every_byte_difference": [
                        [
                            offset,
                            raw[offset] if offset < len(raw) else None,
                            current[offset] if offset < len(current) else None,
                        ]
                        for offset in range(max(len(raw), len(current)))
                        if (raw[offset] if offset < len(raw) else None)
                        != (current[offset] if offset < len(current) else None)
                    ],
                }
            )
        outside_comparisons.append(
            {
                "original": original,
                "held": held,
                "released": released,
                "all_byte_differences_from_pre_retention": differences,
                "same_complete_public_contents": True,
            }
        )
    del sys.modules[MODULE_NAME]
    result = {
        "schema_id": "p67_actual_runtime_admission_limits_v1",
        "passed": True,
        "group": "admission-limits",
        "science_calls": actual.CALLS,
        "actual_complete_observations": len(records),
        "whole_observation_records": records,
        "all_control_names_and_outcomes": controls,
        "all_actual_audit_intervals": intervals,
        "complete_actual_factory_instructions": instructions,
        "installed_interpreter_description": interpreter,
        "whole_raw_code_artifacts": outside_before
        + native_before
        + native_used
        + native_after
        + outside_after,
        "all_outside_lifetime_raw_comparisons": outside_comparisons,
        "complete_source_file_artifact": {"name": source_path.name, "identity": source_identity},
        "source_loaded_function_id": id(module.operation),
        "source_matched_function_id": id(matched),
        "source_loaded_code_id": id(loaded_code),
        "source_matched_code_id": id(expected_code),
        "source_actual_returns": {"matched": 100003, "loaded": 100013},
        "edited_historical_records": [omitted_record, nested_record],
        "removed_actual_function_row": target,
        "all_nested_entrypoint_changes": {
            "original": entrypoints[-1],
            "edited": missing_names[-1],
            "real_binder_denial": route_denial,
        },
        "native_owner_reacquired": True,
        "lease_inactive": True,
        "source_version_attested": False,
        "private_fields_attested": False,
        "continuous_execution_proved": False,
        "historical_edited_records_are_actual_proofs": False,
        "execution_authorized": False,
    }
    write(out, "validation.json", payload.canonical(result) + b"\n")
    print(
        json.dumps(
            {
                "passed": True,
                "group": "admission-limits",
                "science_calls": actual.CALLS,
                "actual_complete_observations": len(records),
                "all_control_names_and_outcomes": controls,
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
