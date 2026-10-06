"""Complete real V2/native runtime controls across actual retained-code use.

Only diagnostic callables execute. Scientific source/array/model/training/final
ports raise. Preserve full observations, raw code bytes, and the actual freeze
frame's retained identities; saved evidence grants no live/source authority.
"""

from contextlib import contextmanager
from hashlib import sha256
import json
import marshal
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import runtime_payload_fixtures as payload
import prospective_runtime_fixtures as actual
from src.app.prospective_generation_ownership import claim_prospective_generation_bundle
from src.app.prospective_runtime_closure import (
    observe_prospective_generation_runtime,
    validate_runtime_observation,
)
from runtime_marshal_value_fixtures import CONSTANT_KINDS, describe_native_code, make_constant_code
from src.core.prospective_generation_bundles import GenerationBundleSnapshot
from src.core.prospective_runtime_closure import RuntimeCodeObservation


class RecordingLease:
    def __init__(self, delegate, recorder):
        self.delegate = delegate
        self.recorder = recorder

    def observe(self, snapshot):
        observation = self.delegate.observe(snapshot)
        self.recorder.rows.append((snapshot, observation))
        return observation


class RecordingObserver:
    def __init__(self):
        self.runtime = actual.ProcessGenerationRuntimeObserver()
        self.rows: list[tuple[GenerationBundleSnapshot, RuntimeCodeObservation]] = []
        self.retained_ids: tuple[int, ...] = ()
        self.retained_code_ids: tuple[int, ...] = ()
        self.function_codes: dict[int, int] = {}
        self.frame_released = False

    @contextmanager
    def freeze(self, snapshot, entrypoints):
        context = self.runtime.freeze(snapshot, entrypoints)
        with context as delegate:
            # Native context frame provides the real adapter's tuple, rather
            # than a caller-created manifest or a second retention invocation.
            generator = context.gen
            assert isinstance(generator, types.GeneratorType)
            frame = generator.gi_frame
            assert frame is not None
            retained = frame.f_locals["retained"]
            del frame
            identities, codes = [], []
            for value in retained:
                identities.append(id(value))
                if type(value) is types.CodeType:
                    codes.append(id(value))
                elif type(value) is types.FunctionType:
                    self.function_codes[id(value)] = id(value.__code__)
            self.retained_ids = tuple(identities)
            self.retained_code_ids = tuple(codes)
            yield RecordingLease(delegate, self)
        del retained
        self.frame_released = generator.gi_frame is None


def identity(raw):
    return {"byte_count": len(raw), "sha256": sha256(raw).hexdigest()}


def write(out, name, raw):
    with (out / name).open("xb") as stream:
        stream.write(raw)
    return identity(raw)


def observe_code(out, name, stage, function):
    raw = marshal.dumps(function.__code__)
    filename = name + "-" + stage + ".marshal"
    return {
        "name": filename,
        "identity": write(out, filename, raw),
        "code_object_id": id(function.__code__),
        "constant_id": id(function.__code__.co_consts[1]),
        "complete_public_native_contents": describe_native_code(function.__code__),
    }


def all_namespace_code_ids(body):
    found, pending = set(), [body]
    while pending:
        value = pending.pop()
        if type(value) is dict:
            pending.extend(value.values())
        elif type(value) is list:
            if (
                len(value) == 3
                and value[0] == "code"
                and type(value[1]) is int
                and type(value[2]) is dict
            ):
                if set(value[2]) == {"byte_count", "sha256", "filename", "qualname", "first_line"}:
                    found.add(value[1])
            pending.extend(value)
    return found


def main(out):
    out.mkdir()
    actual.install_guards()
    functions = {kind: types.FunctionType(make_constant_code(kind), {}) for kind in CONSTANT_KINDS}
    module = compile(
        "def controlled():\n    unused = None\n    raise ValueError('placeholder')\n",
        "<marshal-retention-exception>",
        "exec",
    )
    error_code = module.co_consts[0].replace(co_consts=make_constant_code("str").co_consts)
    functions["exception"] = types.FunctionType(error_code, {"ValueError": ValueError})
    observer = RecordingObserver()
    raw_artifacts, control_outcomes = [], []
    codes_before, codes_during, codes_after = {}, {}, {}
    with payload.complete_bundle() as (bundle, reader, owner):
        design = actual.fixed_prospective_design()
        with observe_prospective_generation_runtime(
            design, bundle.spec, reader, owner, observer, actual.ENTRYPOINTS
        ):
            assert len(observer.rows) == 2
            retained_ids = set(observer.retained_ids)
            for name, function in functions.items():
                code = function.__code__
                assert id(code) in retained_ids and id(code.co_consts[1]) in retained_ids
                codes_before[name] = observe_code(out, name, "before", function)
            returned = [functions[kind]() for kind in CONSTANT_KINDS]
            error = None
            try:
                functions["exception"]()
            except ValueError as caught:
                error = caught
            assert error is not None and error.__traceback__ is not None
            assert error.__traceback__.tb_next is not None
            assert error.__traceback__.tb_next.tb_frame.f_code is error_code
            assert error.args[0] is error_code.co_consts[1]
            for name, function in functions.items():
                codes_during[name] = observe_code(out, name, "used", function)
                assert codes_before[name]["identity"] == codes_during[name]["identity"]
                assert (
                    codes_before[name]["complete_public_native_contents"]
                    == codes_during[name]["complete_public_native_contents"]
                )
            assert all(
                value is functions[kind].__code__.co_consts[1]
                for kind, value in zip(CONSTANT_KINDS, returned)
            )
            del returned, error
            for name, function in functions.items():
                codes_after[name] = observe_code(out, name, "released", function)
                assert codes_after[name]["identity"] == codes_before[name]["identity"]
            assert not actual.CALLS
        assert len(observer.rows) == 3 and observer.frame_released
        assert not observer.runtime._active
        # Reacquire the real native owner after the full app/runtime finally path.
        with claim_prospective_generation_bundle(design, bundle.spec, reader, owner):
            pass
        assert not actual.CALLS
        records = []
        retained_code_ids = set(observer.retained_code_ids)
        for index, (snapshot, observation) in enumerate(observer.rows):
            previous = observer.rows[index - 1][1] if index else None
            validate_runtime_observation(snapshot, observation, previous)
            body = json.loads(observation.runtime_json)
            functions_in_body = {row["object_id"] for row in body["python"]["functions"]}
            assert functions_in_body <= observer.function_codes.keys()
            assert {observer.function_codes[key] for key in functions_in_body} <= retained_code_ids
            assert all_namespace_code_ids(body) <= retained_code_ids
            records.append(
                payload.record_observation(out, "retention-" + str(index), snapshot, observation)
            )
        assert len({row["complete_runtime_identity"]["sha256"] for row in records}) == 1
    for name in functions:
        rows = [codes_before[name], codes_during[name], codes_after[name]]
        raw_artifacts.extend(rows)
        control_outcomes.append(
            {
                "name": name + "-raw-first-use-release",
                "passed": True,
                "complete_before_during_after": rows,
            }
        )
    control_outcomes.extend(
        [
            {
                "name": "every-observed-function-and-namespace-code-in-actual-retention",
                "passed": True,
            },
            {"name": "actual-controlled-code-exception-and-constant-identity", "passed": True},
            {"name": "whole-runtime-entry-final-identical", "passed": True},
            {"name": "freeze-frame-release-and-real-native-owner-reacquisition", "passed": True},
        ]
    )
    result = {
        "schema_id": "p67_actual_runtime_retention_controls_v1",
        "passed": True,
        "group": "retention",
        "science_calls": actual.CALLS,
        "actual_complete_observations": len(records),
        "whole_observation_records": records,
        "whole_raw_code_artifacts": raw_artifacts,
        "all_control_names_and_outcomes": control_outcomes,
        "actual_retained_object_ids": observer.retained_ids,
        "actual_retained_code_ids": observer.retained_code_ids,
        "all_actual_function_code_joins": observer.function_codes,
        "frame_released": observer.frame_released,
        "native_owner_reacquired": True,
        "production_private_fields_attested": False,
        "source_version_attested": False,
        "execution_authorized": False,
    }
    write(out, "validation.json", payload.canonical(result))
    print(
        json.dumps(
            {
                "passed": True,
                "group": "retention",
                "science_calls": actual.CALLS,
                "actual_complete_observations": len(records),
                "all_control_names_and_outcomes": control_outcomes,
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
