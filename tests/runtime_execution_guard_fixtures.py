"""Actual global mutation denials; complete guard/source admission remains open."""

import ctypes
import dis
from hashlib import sha256
import json
import marshal
from pathlib import Path
import sys
import threading
import types
from typing import Any
import zlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prospective_runtime_fixtures as actual
import runtime_payload_fixtures as payload
from runtime_admission_limit_fixtures import (
    RecordingObserver,
    code_record,
    default_operation,
    transient_operation,
    write,
)
from runtime_marshal_value_fixtures import CODE_FIELDS, describe_native_value
from src.app.prospective_generation_ownership import claim_prospective_generation_bundle
from src.app.prospective_runtime_closure import (
    observe_prospective_generation_runtime,
    validate_runtime_observation,
)
from src.infra.runtime_execution_guard import (
    RuntimeExecutionGuard,
    capture_runtime_execution_catalog,
)

BOUND: Any = default_operation
BINDINGS: dict[str, Any] = {"operation": default_operation}
SET_DEFAULTS: Any = None
CURRENT: Any = None
MONITOR: Any = getattr(sys, "monitoring", None)
PREFLIGHT_BODIES = []
CONSUMER_ERROR = RuntimeError("actual preconstructed consumer failure")


class Binding:
    def __init__(self):
        self.operation: Any = default_operation


BINDING = Binding()


def alternative():
    return 39


def positive():
    return default_operation() + default_operation()


def defaults():
    default_operation.__defaults__ = (9,)


def global_binding():
    global BOUND
    BOUND = alternative


def attribute_binding():
    BINDING.operation = alternative


def subscript_binding():
    BINDINGS["operation"] = alternative


def native_setter():
    SET_DEFAULTS(default_operation, (9,))


def native_dictionary_update():
    BINDINGS.update({"operation": alternative})


def monitor_replace():
    MONITOR.set_events(3, 0)


def foreign_tool():
    MONITOR.use_tool_id(4, "p67-forbidden-foreign")


def register_callback():
    MONITOR.register_callback(3, MONITOR.events.CALL, foreign_callback)


def replace_trace():
    sys.settrace(None)


def replace_profile():
    sys.setprofile(None)


def direct_function():
    types.FunctionType(default_operation.__code__, {})


def code_binding():
    default_operation.__code__ = alternative.__code__


def premature_release():
    CURRENT.__exit__(None, None, None)


def make_closure():
    operation: Any = default_operation

    def setter():
        nonlocal operation
        operation = alternative

    def getter():
        return operation

    return setter, getter


CLOSURE_SETTER, CLOSURE_GETTER = make_closure()


def foreign_callback(code, offset, operation, argument):
    return None


def noop_trace(frame, event, argument):
    return None


def noop_profile(frame, event, argument):
    return None


def wait_for_release(event):
    event.wait(3)


def run_case(guard, operation):
    first, second, value = None, None, None
    with guard:
        try:
            value = operation()
        except RuntimeError as error:
            first = error
        if first is not None:
            try:
                alternative()
            except RuntimeError as error:
                second = error
    return value, first, second


def run_consumer(guard):
    with guard:
        raise CONSUMER_ERROR


def run_preflight(guard):
    with guard:
        PREFLIGHT_BODIES.append(17)


def describe_complete(value):
    if type(value) is types.CodeType:
        return {
            "type": "code",
            "fields": {name: describe_complete(getattr(value, name)) for name in CODE_FIELDS},
        }
    if type(value) is tuple:
        return {"type": "tuple", "items": [describe_complete(item) for item in value]}
    if type(value) is frozenset:
        return {
            "type": "frozenset",
            "items": sorted((describe_complete(item) for item in value), key=repr),
        }
    if type(value) is complex:
        return {"type": "complex", "real_hex": value.real.hex(), "imag_hex": value.imag.hex()}
    if type(value) is slice:
        return {
            "type": "slice",
            "start": describe_complete(value.start),
            "stop": describe_complete(value.stop),
            "step": describe_complete(value.step),
        }
    if value is Ellipsis:
        return {"type": "ellipsis"}
    return describe_native_value(value)


def instruction_value(value):
    if type(value) in (types.BuiltinFunctionType, type):
        # Disassembly resolves common constants to native callables/classes.
        # Their primitive identity/repr is descriptive, never a code constant
        # or native/source implementation attestation.
        return {
            "type": type(value).__name__,
            "object_id": id(value),
            "repr": repr(value),
            "native_disassembly_argument_not_code_constant": True,
        }
    return describe_complete(value)


def instructions(code):
    return [
        {
            "offset": row.offset,
            "opname": row.opname,
            "arg": row.arg,
            "argrepr": row.argrepr,
            "complete_argvalue": instruction_value(row.argval),
            "argvalue_code_id": id(row.argval) if type(row.argval) is types.CodeType else None,
        }
        for row in dis.get_instructions(code)
    ]


def preflight_controls():
    controls = []
    catalog = capture_runtime_execution_catalog()
    for name in ("foreign-tool", "trace", "profile", "thread"):
        guard = RuntimeExecutionGuard(catalog, run_preflight.__code__)
        guard.install_audit()
        worker, release = None, None
        if name == "foreign-tool":
            MONITOR.use_tool_id(4, "p67-preexisting-foreign")
            MONITOR.register_callback(4, MONITOR.events.CALL, foreign_callback)
        elif name == "trace":
            sys.settrace(noop_trace)
        elif name == "profile":
            sys.setprofile(noop_profile)
        else:
            release = threading.Event()
            worker = threading.Thread(target=wait_for_release, args=(release,))
            worker.start()
        before = {
            "tool_state": guard.tool_state(),
            "trace_id": id(sys.gettrace()),
            "profile_id": id(sys.getprofile()),
            "thread_count": threading.active_count(),
        }
        try:
            run_preflight(guard)
        except RuntimeError as error:
            denial = str(error)
        else:
            raise AssertionError("guard admitted unsupported preflight")
        assert not guard.active and not guard.used and not guard.events and not guard.denials
        assert PREFLIGHT_BODIES == []
        after = {
            "tool_state": guard.tool_state(),
            "trace_id": id(sys.gettrace()),
            "profile_id": id(sys.getprofile()),
            "thread_count": threading.active_count(),
        }
        assert before == after
        if name == "foreign-tool":
            MONITOR.clear_tool_id(4)
            MONITOR.free_tool_id(4)
        elif name == "trace":
            sys.settrace(None)
        elif name == "profile":
            sys.setprofile(None)
        else:
            assert release is not None and worker is not None
            release.set()
            worker.join(2)
            assert not worker.is_alive() and threading.active_count() == 1
        controls.append(
            {
                "name": "preflight-" + name,
                "passed": True,
                "denial": denial,
                "complete_before": before,
                "complete_after": after,
                "guard_armed": False,
                "consumer_body_entered": False,
            }
        )
    return controls


def save_catalog(out, catalog):
    codes = []
    for code in catalog.codes_by_id.values():
        raw = marshal.dumps(code)
        name = "catalog-code-" + str(id(code)) + ".marshal"
        codes.append(
            {
                "code_id": id(code),
                "name": name,
                "identity": write(out, name, raw),
                "complete_native_public_fields": describe_complete(code)["fields"],
                "complete_disassembly": instructions(code),
            }
        )
    contents = {
        "every_actual_gc_function": [
            {
                "function_id": id(value),
                "code_id": id(value.__code__),
                "globals_id": id(value.__globals__),
            }
            for value in catalog.functions_by_id.values()
        ],
        "every_recursive_code": codes,
    }
    raw = payload.canonical(contents)
    packed = zlib.compress(raw)
    return {
        "name": "complete-catalog.json.zlib",
        "identity": write(out, "complete-catalog.json.zlib", packed),
        "complete_contents_identity": {"byte_count": len(raw), "sha256": sha256(raw).hexdigest()},
        "every_raw_code_pin": {row["name"]: row["identity"] for row in codes},
        "actual_gc_function_count": len(catalog.functions_by_id),
        "actual_recursive_code_count": len(codes),
    }


def main(out):
    global SET_DEFAULTS, CURRENT
    out.mkdir()
    actual.install_guards()
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 14):
        catalog = capture_runtime_execution_catalog()
        guard = RuntimeExecutionGuard(catalog, run_preflight.__code__)
        guard.install_audit()
        try:
            run_preflight(guard)
        except RuntimeError as error:
            assert "measured CPython3.14" in str(error)
        else:
            raise AssertionError("unsupported interpreter accepted")
        report = {
            "passed": True,
            "group": "execution-guard",
            "supported_interpreter": False,
            "actual_complete_observations": 0,
            "science_calls": actual.CALLS,
            "execution_authorized": False,
            "all_control_names_and_outcomes": [],
        }
        write(out, "validation.json", payload.canonical(report) + b"\n")
        print(json.dumps(report))
        return
    controls = preflight_controls()
    SET_DEFAULTS = ctypes.pythonapi.PyFunction_SetDefaults
    SET_DEFAULTS.argtypes = [ctypes.py_object, ctypes.py_object]
    SET_DEFAULTS.restype = ctypes.c_int
    original_defaults, original_code = default_operation.__defaults__, default_operation.__code__
    operations = [
        ("positive", positive),
        ("make-function", transient_operation),
        ("defaults", defaults),
        ("global", global_binding),
        ("closure", CLOSURE_SETTER),
        ("attribute", attribute_binding),
        ("subscript", subscript_binding),
        ("native-c-api", native_setter),
        ("native-dict-update", native_dictionary_update),
        ("monitor-replacement", monitor_replace),
        ("foreign-tool-install", foreign_tool),
        ("callback-replacement", register_callback),
        ("trace-replacement", replace_trace),
        ("profile-replacement", replace_profile),
        ("direct-function", direct_function),
        ("code-binding", code_binding),
        ("premature-release", premature_release),
    ]
    codes = {name: operation.__code__ for name, operation in operations}
    codes.update(
        {
            "owner": run_case.__code__,
            "consumer-owner": run_consumer.__code__,
            "later-target": alternative.__code__,
            "default-target": default_operation.__code__,
        }
    )
    before_raws = [code_record(out, name, "pre-retention", code) for name, code in codes.items()]
    observer = RecordingObserver()
    with payload.complete_bundle() as (bundle, reader, owner):
        design = actual.fixed_prospective_design()
        catalog = capture_runtime_execution_catalog()
        guards = [
            (name, operation, RuntimeExecutionGuard(catalog, run_case.__code__))
            for name, operation in operations
        ]
        consumer = RuntimeExecutionGuard(catalog, run_consumer.__code__)
        for _, _, guard in guards:
            guard.install_audit()
        consumer.install_audit()
        for _, _, guard in guards:
            try:
                guard.require_complete_admission()
            except RuntimeError as error:
                assert "correspondence remains unproved" in str(error)
            else:
                raise AssertionError("partial mechanism granted complete admission")
        with observe_prospective_generation_runtime(
            design, bundle.spec, reader, owner, observer, actual.ENTRYPOINTS
        ) as observed:
            assert observed.runtime_process_observed and not observed.execution_authorized
            catalog_record = save_catalog(out, catalog)
            held_before = [code_record(out, name, "before", code) for name, code in codes.items()]
            for name, operation, guard in guards:
                CURRENT = guard
                value, first, second = run_case(guard, operation)
                if name == "positive":
                    assert value == 10 and first is None and second is None and guard.poison is None
                else:
                    assert first is not None and second is not None and guard.poison is not None
                    assert len(guard.denials) >= 2 and "prior denial" in guard.denials[-1]["reason"]
                    assert not any(
                        row[1] == id(alternative.__code__)
                        for row in guard.events
                        if row[0] in {"PY_START", "INSTRUCTION"}
                    )
                assert guard.used and not guard.active and len(guard.cleanup) == 3
                assert all(row["prior_callback_was_none"] for row in guard.cleanup)
                assert (
                    default_operation.__defaults__ is original_defaults
                    and default_operation.__code__ is original_code
                )
                assert BOUND is default_operation and BINDINGS == {"operation": default_operation}
                assert (
                    BINDING.operation is default_operation and CLOSURE_GETTER() is default_operation
                )
                controls.append(
                    {
                        "name": name,
                        "passed": True,
                        "actual_value": value,
                        "first_error": str(first) if first is not None else None,
                        "second_error": str(second) if second is not None else None,
                        "poison": guard.poison,
                        "every_delivered_event": guard.events,
                        "all_denials": guard.denials,
                        "all_actual_callback_cleanup": guard.cleanup,
                        "active_after_release": guard.active,
                        "complete_tool_state_after_release": guard.tool_state(),
                        "later_target_code_id": id(alternative.__code__),
                        "actual_operation_code_id": id(operation.__code__),
                        "actual_operation_id": id(operation),
                    }
                )
            try:
                run_consumer(consumer)
            except RuntimeError as error:
                assert error is CONSUMER_ERROR
            else:
                raise AssertionError("consumer failure did not propagate")
            assert (
                not consumer.active
                and consumer.used
                and consumer.poison is None
                and len(consumer.cleanup) == 3
            )
            controls.append(
                {
                    "name": "actual-consumer-failure-release",
                    "passed": True,
                    "actual_error": str(CONSUMER_ERROR),
                    "every_delivered_event": consumer.events,
                    "all_actual_callback_cleanup": consumer.cleanup,
                    "active_after_release": consumer.active,
                }
            )
            held_after = [code_record(out, name, "pre-final", code) for name, code in codes.items()]
            assert not actual.CALLS
        assert (
            len(observer.rows) == 3 and not observer.runtime._active and not observer.lease._active
        )
        with claim_prospective_generation_bundle(design, bundle.spec, reader, owner):
            pass
        records = []
        for index, (snapshot, observation) in enumerate(observer.rows):
            validate_runtime_observation(
                snapshot, observation, observer.rows[index - 1][1] if index else None
            )
            records.append(
                payload.record_observation(out, "actual-" + str(index), snapshot, observation)
            )
        assert len({row["complete_runtime_identity"]["sha256"] for row in records}) == 1
    after_raws = [code_record(out, name, "post-release", code) for name, code in codes.items()]
    for before, after in zip(held_before, held_after):
        assert (
            before["identity"] == after["identity"]
            and before["complete_public_native_contents"]
            == after["complete_public_native_contents"]
        )
    controls.append(
        {
            "name": "complete-admission-refused",
            "passed": True,
            "execution_authorized": False,
            "all_five_prior_admission_gaps_remain_required": True,
            "whole_observer_lifetime_guard_integrated": False,
        }
    )
    report = {
        "schema_id": "p67_runtime_execution_guard_actual_v1",
        "passed": True,
        "group": "execution-guard",
        "supported_interpreter": True,
        "installed_interpreter_description": {
            "version": sys.version,
            "implementation": sys.implementation.name,
            "executable": sys.executable,
            "trusted_build_source_correspondence": False,
        },
        "science_calls": actual.CALLS,
        "actual_complete_observations": len(records),
        "whole_observation_records": records,
        "all_control_names_and_outcomes": controls,
        "complete_execution_catalog": catalog_record,
        "all_raw_code_artifacts": before_raws + held_before + held_after + after_raws,
        "all_actual_code_ids": {name: id(code) for name, code in codes.items()},
        "complete_disassembly": {name: instructions(code) for name, code in codes.items()},
        "final_tool_state": consumer.tool_state(),
        "original_trace_profile_restored": sys.gettrace() is None and sys.getprofile() is None,
        "native_owner_reacquired": True,
        "lease_inactive": True,
        "new_mutation_enforcement_implemented": True,
        "complete_continuous_guard_implemented": False,
        "whole_observer_lifetime_guard_integrated": False,
        "source_version_attested": False,
        "private_fields_attested": False,
        "continuous_execution_proved": False,
        "execution_authorized": False,
    }
    write(out, "validation.json", payload.canonical(report) + b"\n")
    print(
        json.dumps(
            {
                "passed": True,
                "group": "execution-guard",
                "science_calls": actual.CALLS,
                "supported_interpreter": True,
                "actual_complete_observations": len(records),
                "all_control_names_and_outcomes": [
                    {"name": row["name"], "passed": row["passed"]} for row in controls
                ],
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
