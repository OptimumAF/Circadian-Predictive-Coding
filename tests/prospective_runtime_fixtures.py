"""Full V2/real process controls. Temporary Python/native code is never scientific.

All positive assertions use the real native owner and whole current-process
observer. The small pure-validator descriptor is only an invalid-record fixture.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from prospective_generation_bundle_fixtures import make_generation_bundle
from prospective_request_bundle_fixtures import raw_identity
from src.app import continual_arrived_benchmark as arrived
from src.app.prospective_confirmation_design import fixed_prospective_design
from src.app.prospective_runtime_closure import (
    observe_prospective_generation_runtime,
)
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.controlled_parent_selection import ParentControlledCircadianNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.prospective_generation_ownership import generation_ownership_scope
from src.core.prospective_runtime_closure import RuntimeCodeObservation
from src.infra import datasets
from src.infra import continual_confirmation_final as final
from src.app import continual_confirmation_training as training
from src.infra.prospective_generation_bundles import FileProspectiveGenerationBundleReader
from src.infra.prospective_generation_ownership import FileGenerationRequestOwner
from src.infra.prospective_runtime_closure import ProcessGenerationRuntimeObserver

CALLS: dict[str, int] = {}
ENTRYPOINTS = (
    "src.infra.datasets:generate_two_cluster_dataset",
    "src.app.continual_confirmation_training:train_confirmation",
)


def forbidden(*args: Any, **kwargs: Any) -> Any:
    CALLS["science"] = CALLS.get("science", 0) + 1
    raise AssertionError(
        "runtime proof touched scientific source/RNG/array/model/train/final/scoring"
    )


def other(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("detached test callable must never be executed")


def install_guards() -> None:
    for target, names in (
        (arrived, ("_build_phase_a_roles", "_build_phase_b_roles", "release_final_test")),
        (datasets, ("generate_two_cluster_dataset", "generate_two_cluster_dataset_with_transform")),
        (training, ("train_confirmation", "_train_families")),
        (final, ("release_confirmation_final", "evaluate_confirmation_final")),
        (np.random, ("default_rng",)),
        (np, ("array", "asarray", "ascontiguousarray")),
    ):
        for name in names:
            setattr(target, name, forbidden)
    for model in (
        BackpropMLP,
        PredictiveCodingNetwork,
        CircadianPredictiveCodingNetwork,
        ParentControlledCircadianNetwork,
    ):
        for name in ("__init__", "train_epoch", "compute_accuracy", "predict_proba"):
            setattr(model, name, forbidden)


@lru_cache(maxsize=1)
def invalid_observation_fixture():
    with TemporaryDirectory() as directory:
        bundle = make_generation_bundle(Path(directory))
        snapshot = FileProspectiveGenerationBundleReader(bundle.root).read_bundle(bundle.spec)
    body = {
        "schema_id": "p67_observed_process_runtime_v1",
        "python": {
            "modules": ["invalid-record-unit-fixture"],
            "functions": ["not-actual-runtime-proof"],
            "namespaces": [],
            "import_state": {},
        },
        "native": {
            "platform": "windows",
            "images": ["not-native-proof"],
            "executable_regions": ["not-memory-proof"],
        },
        "entrypoints": [[ENTRYPOINTS[0], 1, {}, 1]],
        "source_attestation": False,
    }
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    observation = RuntimeCodeObservation(
        generation_ownership_scope(snapshot),
        1,
        "2" * 32,
        0,
        snapshot.observed_utc,
        ENTRYPOINTS[:1],
        raw.decode(),
        raw_identity(raw),
    )
    # This deliberately invalid descriptor is never a positive runtime proof.
    # Complete real-process controls above/below provide the positive records.
    return snapshot, observation


class Dispatcher:
    operation = staticmethod(forbidden)


def closure_callable():
    value = 1

    def operation():
        return value

    return operation


@contextmanager
def expect_failure(text: str):
    try:
        yield
    except (ValueError, RuntimeError) as error:
        # All late checks still run. Their failures retain the earlier denial
        # in Python's exception chain rather than erase that evidence.
        chain = []
        current: BaseException | None = error
        while current is not None:
            chain.append(str(current))
            current = current.__context__
        assert any(text in message for message in chain), chain
    else:
        raise AssertionError(f"missing expected failure: {text}")


def run_group(group: str) -> None:
    install_guards()
    # Load every dependency and create all control callables/FFI types before
    # taking the baseline. No function/model/source in this fixture is called.
    detached = types.FunctionType(forbidden.__code__, forbidden.__globals__, "detached")
    closed = closure_callable()
    dispatch = Dispatcher()
    dispatch.operation = forbidden
    with TemporaryDirectory() as directory:
        root = Path(directory)
        bundle_root, registry = root / "bundle", root / "registry"
        bundle_root.mkdir()
        registry.mkdir()
        bundle = make_generation_bundle(bundle_root)
        reader = FileProspectiveGenerationBundleReader(bundle_root)
        owner = FileGenerationRequestOwner(bundle_root, registry)
        observer = ProcessGenerationRuntimeObserver()
        module_path = root / "p67_loaded_runtime_fixture.py"
        module_path.write_text("def operation():\n    return 3\n", encoding="utf-8")
        module_spec = importlib.util.spec_from_file_location(
            "p67_loaded_runtime_fixture", module_path
        )
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        design = fixed_prospective_design()

        def context():
            return observe_prospective_generation_runtime(
                design, bundle.spec, reader, owner, observer, ENTRYPOINTS
            )

        if group == "positive":
            with context() as result:
                assert (
                    result.runtime_process_observed and not result.runtime_source_version_attested
                )
                assert not result.execution_authorized and not result.fresh_roles_authorized
                assert len(result.required_actual_bindings) == 12
                assert len(result.ownership.bundle.snapshot.generation_request.role_recipes) == 480
                body = json.loads(result.runtime_at_entry.runtime_json)
                names = {row["name"] for row in body["python"]["modules"]}
                assert {"numpy", "src.infra.datasets", "p67_loaded_runtime_fixture"} <= names
                assert id(detached) in {row["object_id"] for row in body["python"]["functions"]}
                assert body["native"]["images"] and body["native"]["executable_regions"]
                first_nonce = result.runtime_at_entry.lease_nonce
            with context() as second:
                assert second.runtime_at_entry.lease_nonce != first_nonce
        elif group in ("python_a", "python_b"):
            controls: tuple[str, ...] = (
                "module",
                "callable",
                "detached",
                "closure",
                "instance",
                "file",
                "import",
                "code",
            )
            controls = controls[:4] if group == "python_a" else controls[4:]
            for case in controls:
                saved = (
                    detached.__defaults__,
                    closed.__closure__[0].cell_contents,
                    module_path.read_bytes(),
                )
                try:
                    with expect_failure(
                        "runtime" if case != "import" and case != "code" else "before execution"
                    ):
                        with context():
                            if case == "module":
                                sys.modules["p67_late_alias"] = module
                            elif case == "callable":
                                datasets.generate_two_cluster_dataset = other
                            elif case == "detached":
                                detached.__defaults__ = (7,)
                            elif case == "closure":
                                closed.__closure__[0].cell_contents = 2
                            elif case == "instance":
                                dispatch.operation = other
                            elif case == "file":
                                module_path.write_bytes(saved[2].replace(b"3", b"4"))
                            elif case == "import":
                                __import__("p67_never_import_this_runtime_fixture")
                            else:
                                detached.__code__ = other.__code__
                finally:
                    sys.modules.pop("p67_late_alias", None)
                    datasets.generate_two_cluster_dataset = forbidden
                    detached.__defaults__ = saved[0]
                    closed.__closure__[0].cell_contents = saved[1]
                    dispatch.operation = forbidden
                    module_path.write_bytes(saved[2])
        elif group == "native":
            if sys.platform != "win32":
                raise ValueError("native runtime fixture requires Windows")
            import ctypes
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            allocate, release = kernel.VirtualAlloc, kernel.VirtualFree
            allocate.argtypes, allocate.restype = (
                [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD],
                ctypes.c_void_p,
            )
            release.argtypes, release.restype = (
                [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD],
                wintypes.BOOL,
            )
            page = allocate(None, 4096, 0x3000, 0x40)
            assert page
            ctypes.memset(page, 0, 4096)
            try:
                with expect_failure("runtime"):
                    with context() as result:
                        regions = json.loads(result.runtime_at_entry.runtime_json)["native"][
                            "executable_regions"
                        ]
                        assert any(
                            row["base"] <= page < row["base"] + row["size"] for row in regions
                        )
                        ctypes.memset(page, 1, 1)
            finally:
                assert release(page, 0, 0x8000)
            with expect_failure("before execution"):
                with context():
                    ctypes.WinDLL("p67_forbidden_late_image.dll")
        elif group == "failure":
            with expect_failure("consumer"):
                with context():
                    raise RuntimeError("consumer exception")
            with context():
                pass
            snapshot = reader.read_bundle(bundle.spec)
            with observer.freeze(snapshot, ENTRYPOINTS) as lease:
                lease.observe(snapshot)
            with expect_failure("inactive"):
                lease.observe(snapshot)
            # Complete file failures must still invoke runtime/owner final checks.
            saved_request = (bundle.root / "request.json").read_bytes()
            try:
                with expect_failure("whole byte count"):
                    with context():
                        (bundle.root / "request.json").write_bytes(saved_request + b" ")
            finally:
                (bundle.root / "request.json").write_bytes(saved_request)
            with context():
                pass
        else:
            raise AssertionError("unknown fixture group")
        assert not CALLS, CALLS
        sys.modules.pop(module_spec.name)
        print(
            json.dumps(
                {
                    "group": group,
                    "complete_capture_timings": observer.phase_timings,
                    "science_calls": CALLS,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print(f"PASS {group}")


if __name__ == "__main__":
    run_group(sys.argv[1])
