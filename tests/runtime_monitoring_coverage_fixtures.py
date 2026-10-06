"""Measure actual VM/callback/native coverage; no continuous admission authority.

Inputs are precompiled diagnostic operations and real guarded V2/native ownership.
Outputs retain every event in declared intervals, complete records/code values,
and actual trace/monitor cleanup. Production observers and scientific work are untouched.
"""

import ctypes
import dis
import json
from pathlib import Path
import platform
import sys
import types
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prospective_runtime_fixtures as actual
import runtime_payload_fixtures as payload
from runtime_admission_limit_fixtures import (
    EventRecorder,
    RecordingObserver,
    code_record,
    default_operation,
    transient_operation,
    write,
)
from src.app.prospective_generation_ownership import claim_prospective_generation_bundle
from src.app.prospective_runtime_closure import (
    observe_prospective_generation_runtime,
    validate_runtime_observation,
)

TOOLS = (3, 4)
MONITOR: Any = None
SET_DEFAULTS: Any = None
GET_DEFAULTS: Any = None
TRACE_BODIES: list[int] = []
MILESTONES: list[str] = []
GETTER_POINTER_JOINS: list[list[int]] = []


def mark(phase):
    MILESTONES.append(phase)
    print(phase, file=sys.stderr, flush=True)


def native_operation():
    original = default_operation.__defaults__
    mark("before-native-default-setter")
    assert SET_DEFAULTS(default_operation, (9,)) == 0
    mark("after-native-default-setter")
    changed = default_operation()
    pointer = GET_DEFAULTS(default_operation)
    assert pointer == id(default_operation.__defaults__)
    GETTER_POINTER_JOINS.append([pointer, id(default_operation.__defaults__), changed])
    mark("after-native-default-getter")
    assert SET_DEFAULTS(default_operation, original) == 0
    restored = default_operation()
    pointer = GET_DEFAULTS(default_operation)
    assert pointer == id(original) and default_operation.__defaults__ is original
    GETTER_POINTER_JOINS.append([pointer, id(original), restored])
    return changed, restored


def foreign_trigger():
    return 31


def trace_target():
    TRACE_BODIES.append(17)
    return 17


class MonitoringRecorder:
    def __init__(self):
        self.active = False
        self.foreign_active = False
        self.rows: list[list[Any]] = []
        self.foreign_rows: list[list[Any]] = []
        self.foreign_returns = None

    def call(self, code, offset, operation, argument):
        if self.active:
            # Never retain callback arguments: that would change weak lifetime,
            # function membership and the native marshal reference experiment.
            self.rows.append(
                [
                    "CALL",
                    id(code),
                    offset,
                    id(operation),
                    type(operation).__name__,
                    id(argument),
                    type(argument).__name__,
                ]
            )

    def start(self, code, offset):
        if self.active:
            self.rows.append(["PY_START", id(code), offset])

    def instruction(self, code, offset):
        if self.active:
            self.rows.append(["INSTRUCTION", id(code), offset])

    def c_return(self, code, offset, operation, argument):
        if self.active:
            self.rows.append(
                [
                    "C_RETURN",
                    id(code),
                    offset,
                    id(operation),
                    type(operation).__name__,
                    id(argument),
                    type(argument).__name__,
                ]
            )

    def foreign_callback(self, code, offset, operation, argument):
        if self.active:
            self.foreign_rows.append(
                [
                    "CALL",
                    id(code),
                    offset,
                    id(operation),
                    type(operation).__name__,
                    id(argument),
                    type(argument).__name__,
                ]
            )
        if self.foreign_active and operation is foreign_trigger:
            self.foreign_active = False
            self.foreign_returns = (transient_operation(), native_operation())


class TraceRecorder:
    def __init__(self):
        self.rows: list[list[Any]] = []

    def hook(self, frame, event, argument):
        self.rows.append([event, id(frame.f_code), frame.f_lasti, frame.f_lineno])
        if frame.f_code is trace_target.__code__ and event == "call":
            raise RuntimeError("diagnostic trace guard denied before target body")
        return None


def tool_state(monitoring):
    return [
        {
            "tool_id": tool,
            "name": monitoring.get_tool(tool),
            "global_events": monitoring.get_events(tool),
        }
        for tool in range(6)
    ]


def instructions(code):
    return [
        {"offset": row.offset, "opname": row.opname, "arg": row.arg, "argrepr": row.argrepr}
        for row in dis.get_instructions(code)
    ]


def main(out):
    global MONITOR, SET_DEFAULTS, GET_DEFAULTS
    out.mkdir()
    actual.install_guards()
    monitoring = getattr(sys, "monitoring", None)
    assert monitoring is not None, "installed interpreter has no monitoring API"
    initial_tools = tool_state(monitoring)
    assert all(row["name"] is None and row["global_events"] == 0 for row in initial_tools)
    original_trace, original_profile = sys.gettrace(), sys.getprofile()
    assert original_trace is None and original_profile is None
    # Public C API only. Prebind before the real hold so dlsym audit denial stays intact.
    SET_DEFAULTS = ctypes.pythonapi.PyFunction_SetDefaults
    SET_DEFAULTS.argtypes = [ctypes.py_object, ctypes.py_object]
    SET_DEFAULTS.restype = ctypes.c_int
    GET_DEFAULTS = ctypes.pythonapi.PyFunction_GetDefaults
    GET_DEFAULTS.argtypes = [ctypes.py_object]
    # The public API returns a borrowed pointer. py_object would assume ownership
    # and decref it; compare the raw pointer to the actual CPython object identity.
    GET_DEFAULTS.restype = ctypes.c_void_p
    MONITOR = MonitoringRecorder()
    tracer = TraceRecorder()
    audit = EventRecorder()
    sys.addaudithook(audit)
    observer = RecordingObserver()
    nested = next(
        value for value in transient_operation.__code__.co_consts if type(value) is types.CodeType
    )
    codes = {
        "ordinary-factory": transient_operation.__code__,
        "ordinary-nested": nested,
        "default-operation": default_operation.__code__,
        "native-operation": native_operation.__code__,
        "foreign-trigger": foreign_trigger.__code__,
        "foreign-callback": MonitoringRecorder.foreign_callback.__code__,
        "trace-target": trace_target.__code__,
        "trace-hook": TraceRecorder.hook.__code__,
    }
    disassembly = {name: instructions(code) for name, code in codes.items()}
    interpreter = {
        "version": sys.version,
        "hexversion": sys.hexversion,
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "executable": sys.executable,
        "build": platform.python_build(),
        "compiler": platform.python_compiler(),
        "trusted_build_source_correspondence": False,
        "every_monitoring_event": {
            name: value for name, value in vars(monitoring.events).items() if type(value) is int
        },
    }
    mask = monitoring.events.CALL | monitoring.events.PY_START | monitoring.events.INSTRUCTION
    for tool in TOOLS:
        monitoring.use_tool_id(tool, "p67-diagnostic-monitor-" + str(tool))
    monitoring.register_callback(3, monitoring.events.CALL, MONITOR.call)
    monitoring.register_callback(3, monitoring.events.PY_START, MONITOR.start)
    monitoring.register_callback(3, monitoring.events.INSTRUCTION, MONITOR.instruction)
    monitoring.register_callback(3, monitoring.events.C_RETURN, MONITOR.c_return)
    monitoring.register_callback(4, monitoring.events.CALL, MONITOR.foreign_callback)
    pre_raws = [code_record(out, name, "pre-retention", code) for name, code in codes.items()]
    intervals, controls = [], []
    try:
        with payload.complete_bundle() as (bundle, reader, owner):
            design = actual.fixed_prospective_design()
            with observe_prospective_generation_runtime(
                design, bundle.spec, reader, owner, observer, actual.ENTRYPOINTS
            ) as observed:
                assert observed.runtime_process_observed and not observed.execution_authorized
                held_before = [
                    code_record(out, name, "before", code) for name, code in codes.items()
                ]
                for mode in ("ordinary", "native", "foreign"):
                    mark("before-monitoring-interval-" + mode)
                    MONITOR.rows.clear()
                    MONITOR.foreign_rows.clear()
                    MONITOR.foreign_returns = None
                    monitoring.set_events(3, mask)
                    monitoring.set_events(4, monitoring.events.CALL if mode == "foreign" else 0)
                    state = tool_state(monitoring)
                    audit.names.clear()
                    audit.active = True
                    MONITOR.active = True
                    mark("monitoring-active-" + mode)
                    if mode == "ordinary":
                        value = transient_operation()
                        defaults = default_operation.__defaults__
                        default_operation.__defaults__ = (9,)
                        changed = default_operation()
                        default_operation.__defaults__ = defaults
                        restored = default_operation()
                        result = {
                            "transient_id": value[0],
                            "transient_return": value[1],
                            "weak_reference_dead": True,
                            "changed": changed,
                            "restored": restored,
                        }
                        assert value[1] == 23 and (changed, restored) == (9, 5)
                    elif mode == "native":
                        value = native_operation()
                        result = {
                            "changed": value[0],
                            "restored": value[1],
                            "setter_id": id(SET_DEFAULTS),
                            "getter_id": id(GET_DEFAULTS),
                        }
                        assert value == (9, 5)
                    else:
                        MONITOR.foreign_active = True
                        value = foreign_trigger()
                        assert value == 31 and MONITOR.foreign_returns is not None
                        result = {
                            "trigger_return": value,
                            "foreign_returns": MONITOR.foreign_returns,
                        }
                        assert MONITOR.foreign_returns[0][1] == 23 and MONITOR.foreign_returns[
                            1
                        ] == (9, 5)
                    MONITOR.active = False
                    audit.active = False
                    monitoring.set_events(3, 0)
                    monitoring.set_events(4, 0)
                    mark("monitoring-inactive-" + mode)
                    pointer = GET_DEFAULTS(default_operation)
                    assert pointer == id(default_operation.__defaults__)
                    GETTER_POINTER_JOINS.append(
                        [pointer, id(default_operation.__defaults__), default_operation()]
                    )
                    intervals.append(
                        {
                            "name": mode,
                            "all_tool_state": state,
                            "every_delivered_primary_event": list(MONITOR.rows),
                            "every_delivered_secondary_event": list(MONITOR.foreign_rows),
                            "every_audit_event_name": list(audit.names),
                            "actual_values": result,
                        }
                    )
                denials = []
                for operation in (types.FunctionType, sys.settrace, sys.setprofile):
                    audit.names.clear()
                    audit.active = True
                    try:
                        if operation is types.FunctionType:
                            operation(default_operation.__code__, {})
                        else:
                            operation(None)
                    except RuntimeError as error:
                        denials.append(
                            {
                                "operation_id": id(operation),
                                "denial": str(error),
                                "every_audit_event_name": list(audit.names),
                            }
                        )
                    else:
                        raise AssertionError("original direct/trace/profile denial did not run")
                    finally:
                        audit.active = False
                held_after = [
                    code_record(out, name, "pre-final", code) for name, code in codes.items()
                ]
                assert not actual.CALLS
            assert len(observer.rows) == 3 and not observer.runtime._active
            assert observer.lease is not None and not observer.lease._active
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
        for tool in TOOLS:
            monitoring.set_events(tool, 0)
            monitoring.clear_tool_id(tool)
            monitoring.free_tool_id(tool)
        clean_tools = tool_state(monitoring)
        assert clean_tools == initial_tools
        callback_cleanup = []
        for tool in TOOLS:
            monitoring.use_tool_id(tool, "p67-cleanup-check-" + str(tool))
            events = (
                (
                    monitoring.events.CALL,
                    monitoring.events.PY_START,
                    monitoring.events.INSTRUCTION,
                    monitoring.events.C_RETURN,
                )
                if tool == 3
                else (monitoring.events.CALL,)
            )
            for event in events:
                previous = monitoring.register_callback(tool, event, None)
                assert previous is None, "released diagnostic tool retained a callback"
                callback_cleanup.append(
                    {"tool_id": tool, "event": event, "prior_callback_was_none": True}
                )
            monitoring.free_tool_id(tool)
        # Trace failures are measured outside the real hold. Replacing trace/profile
        # during that hold remains forbidden by the original observer above.
        sys.settrace(tracer.hook)
        try:
            trace_target()
        except RuntimeError as error:
            trace_denial = str(error)
        else:
            raise AssertionError("trace callback failed to deny before target body")
        assert sys.gettrace() is None and TRACE_BODIES == []
        later = trace_target()
        assert later == 17 and TRACE_BODIES == [17]
        trace_result = {
            "denial": trace_denial,
            "automatic_trace_inactive_after_caught_denial": True,
            "later_actual_return": later,
            "actual_target_bodies": list(TRACE_BODIES),
            "every_trace_event": tracer.rows,
        }
        post_raws = [code_record(out, name, "post-release", code) for name, code in codes.items()]
    finally:
        MONITOR.active = False
        MONITOR.foreign_active = False
        audit.active = False
        for tool in TOOLS:
            if monitoring.get_tool(tool) is not None:
                monitoring.set_events(tool, 0)
                monitoring.clear_tool_id(tool)
                monitoring.free_tool_id(tool)
        if sys.gettrace() is not original_trace:
            sys.settrace(original_trace)
        if sys.getprofile() is not original_profile:
            sys.setprofile(original_profile)
    for before, after in zip(held_before, held_after):
        assert before["identity"] == after["identity"]
        assert before["complete_public_native_contents"] == after["complete_public_native_contents"]
    assert tool_state(monitoring) == initial_tools and sys.gettrace() is original_trace
    assert sys.getprofile() is original_profile and not actual.CALLS
    ordinary, native, foreign = intervals
    assert any(
        row[0] == "CALL" and row[3] == ordinary["actual_values"]["transient_id"]
        for row in ordinary["every_delivered_primary_event"]
    )
    assert any(
        row[0] == "INSTRUCTION"
        and row[1] == id(codes["ordinary-factory"])
        and row[2]
        in [
            item["offset"]
            for item in disassembly["ordinary-factory"]
            if item["opname"] == "MAKE_FUNCTION"
        ]
        for row in ordinary["every_delivered_primary_event"]
    )
    foreign_visible = {
        label: any(row[1] == id(codes[label]) for row in foreign["every_delivered_primary_event"])
        for label in ("ordinary-factory", "ordinary-nested", "native-operation", "foreign-callback")
    }
    controls = [
        {"name": "ordinary-make-function-and-call-visible-with-actual-death", "passed": True},
        {"name": "actual-python-default-change-and-restore-with-complete-events", "passed": True},
        {"name": "actual-public-native-default-setter-getter-and-call-boundaries", "passed": True},
        {"name": "second-tool-callback-actual-execution-and-coverage-measured", "passed": True},
        {"name": "trace-denial-caught-then-actual-execution-with-trace-inactive", "passed": True},
        {"name": "original-direct-function-trace-profile-audit-denials", "passed": True},
        {"name": "actual-lease-release-owner-reacquisition-and-monitor-cleanup", "passed": True},
    ]
    result = {
        "schema_id": "p67_installed_runtime_monitoring_coverage_v1",
        "passed": True,
        "group": "monitoring-coverage",
        "science_calls": actual.CALLS,
        "all_control_names_and_outcomes": controls,
        "whole_observation_records": records,
        "actual_complete_observations": len(records),
        "all_actual_monitoring_intervals": intervals,
        "all_raw_code_artifacts": pre_raws + held_before + held_after + post_raws,
        "complete_disassembly": disassembly,
        "all_actual_code_ids": {name: id(code) for name, code in codes.items()},
        "installed_interpreter_description": interpreter,
        "all_original_audit_denials": denials,
        "secondary_callback_primary_visibility": foreign_visible,
        "actual_trace_failure_control": trace_result,
        "all_progress_milestones": MILESTONES,
        "all_public_borrowed_getter_pointer_joins": GETTER_POINTER_JOINS,
        "initial_tool_state": initial_tools,
        "final_tool_state": tool_state(monitoring),
        "all_local_masks_after_cleanup": {
            name: [monitoring.get_local_events(tool, code) for tool in TOOLS]
            for name, code in codes.items()
        },
        "all5_actual_reacquired_empty_callback_slots": callback_cleanup,
        "original_trace_profile_restored": True,
        "native_owner_reacquired": True,
        "lease_inactive": True,
        "complete_continuous_guard_implemented": False,
        "source_version_attested": False,
        "private_fields_attested": False,
        "continuous_execution_proved": False,
        "execution_authorized": False,
    }
    assert all(value == [0, 0] for value in result["all_local_masks_after_cleanup"].values())
    write(out, "validation.json", payload.canonical(result) + b"\n")
    print(
        json.dumps(
            {
                "passed": True,
                "group": "monitoring-coverage",
                "science_calls": actual.CALLS,
                "actual_complete_observations": len(records),
                "all_control_names_and_outcomes": controls,
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
