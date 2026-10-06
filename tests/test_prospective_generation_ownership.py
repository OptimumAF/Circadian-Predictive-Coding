"""Full V2 physical/native-owner controls; no scientific execution is permitted."""

from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from typing import Any, Iterator

import numpy as np
import pytest

from prospective_generation_bundle_fixtures import (
    GenerationBundleFixture,
    make_generation_bundle,
    reseal_generation_request,
)
from src.app import continual_arrived_benchmark as arrived
from src.app.prospective_confirmation_design import fixed_prospective_design
from src.app.prospective_generation_ownership import claim_prospective_generation_bundle
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.prospective_generation_ownership import (
    GenerationOwnershipObservation,
    generation_ownership_scope,
)
from src.infra.prospective_generation_bundles import FileProspectiveGenerationBundleReader
from src.infra.prospective_generation_ownership import FileGenerationRequestOwner


@pytest.fixture(autouse=True)
def no_science(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("generation ownership touched scientific source/array/RNG/model/final")

    for name in ("_build_phase_a_roles", "_build_phase_b_roles", "release_final_test"):
        monkeypatch.setattr(arrived, name, forbidden)
    for model in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        monkeypatch.setattr(model, "__init__", forbidden)
        monkeypatch.setattr(model, "compute_accuracy", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    for name in ("array", "asarray", "ascontiguousarray"):
        monkeypatch.setattr(np, name, forbidden)


@pytest.fixture
def bundle(tmp_path: Path) -> GenerationBundleFixture:
    root = tmp_path / "bundle"
    root.mkdir()
    return make_generation_bundle(root)


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    root = tmp_path / "registry"
    root.mkdir()
    return root


def reader(bundle: GenerationBundleFixture) -> FileProspectiveGenerationBundleReader:
    return FileProspectiveGenerationBundleReader(
        bundle.root, clock=lambda: datetime(2026, 10, 5, 13, tzinfo=timezone.utc)
    )


def owned(bundle: GenerationBundleFixture, registry: Path) -> Any:
    return claim_prospective_generation_bundle(
        fixed_prospective_design(),
        bundle.spec,
        reader(bundle),
        FileGenerationRequestOwner(bundle.root, registry),
    )


def test_should_observe_real_live_ownership_of_the_entire_v2_scope_without_admission(
    bundle: GenerationBundleFixture,
    registry: Path,
) -> None:
    with owned(bundle, registry) as result:
        request = result.bundle.snapshot.generation_request
        observation = result.ownership_at_entry
        assert observation.scope == generation_ownership_scope(result.bundle.snapshot)
        assert observation.native_lock_observed is True and observation.sequence == 1
        assert observation.scope.owner_id == bundle.body["owner_id"]
        assert observation.registry_root == registry.resolve().as_posix()
        assert observation.physical_inode > 0 and observation.physical_device >= 0
        assert len(observation.lease_nonce) == 32
        assert len(request.role_recipes) == 480 and len(request.sources) == 100
        assert len(request.streams) == 400 and len(request.bindings) == 60
        assert sum(row.declared_sample_ids is None for row in request.role_recipes) == 360
        assert sum(row.role == "final_test" for row in request.role_recipes) == 120
        assert result.required_actual_bindings == result.bundle.required_actual_bindings
        assert len(result.required_actual_bindings) == 12
        assert result.independent_source_replications is None
        assert not any(
            (
                result.runtime_code_closure_verified,
                result.actual_arrival_verified,
                result.chronology_verified,
                result.fresh_roles_authorized,
                result.execution_authorized,
                result.precision_certified,
            )
        )
    # The immutable observation describes its recorded time, not a live handle.
    assert observation.native_lock_observed is True
    with owned(bundle, registry):
        pass


def test_should_reject_nested_same_process_owners_and_allow_reacquisition(
    bundle: GenerationBundleFixture,
    registry: Path,
) -> None:
    with owned(bundle, registry):
        with pytest.raises(RuntimeError, match="already active"):
            with owned(bundle, registry):
                pytest.fail("concurrent owner acquired the same complete request")
    with owned(bundle, registry):
        pass


def test_should_contend_on_same_inner_generation_request_with_different_outer_owners(
    bundle: GenerationBundleFixture,
    registry: Path,
    tmp_path: Path,
) -> None:
    other_root = tmp_path / "other"
    other_root.mkdir()
    other = make_generation_bundle(other_root)
    other.body["owner_id"] = "2" * 32
    other.save()
    assert other.spec.request_identity != bundle.spec.request_identity
    assert (
        other.body["generation_request"]["request_identity"]
        == bundle.body["generation_request"]["request_identity"]
    )
    with owned(bundle, registry):
        with pytest.raises(RuntimeError, match="already active"):
            with owned(other, registry):
                pytest.fail("changing outer owner evaded the generation request lock")
    with owned(other, registry):
        pass


def test_should_release_native_owner_after_a_consumer_exception(
    bundle: GenerationBundleFixture,
    registry: Path,
) -> None:
    with pytest.raises(RuntimeError, match="consumer failed"):
        with owned(bundle, registry):
            raise RuntimeError("consumer failed")
    with owned(bundle, registry):
        pass


def test_should_deny_released_handles_and_detached_live_scope(
    bundle: GenerationBundleFixture,
    registry: Path,
) -> None:
    snapshot = reader(bundle).read_bundle(bundle.spec)
    port = FileGenerationRequestOwner(bundle.root, registry)
    with port.claim(snapshot) as lease:
        first = lease.observe(snapshot)
        second = lease.observe(snapshot)
        assert (first.sequence, second.sequence) == (0, 1)
        assert first.lease_nonce == second.lease_nonce and first.acquired_utc == second.acquired_utc
        with pytest.raises(ValueError, match="scope"):
            lease.observe(replace(snapshot, owner_id="2" * 32))
        assert lease.observe(snapshot).sequence == 2
    with pytest.raises(RuntimeError, match="inactive"):
        lease.observe(snapshot)


CHILD = r"""
import json, os, sys
from pathlib import Path
import numpy as np
from src.app import continual_arrived_benchmark as arrived
from src.app.prospective_confirmation_design import fixed_prospective_design
from src.app.prospective_generation_ownership import claim_prospective_generation_bundle
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.prospective_request_bundles import ProspectiveBundleSpec
from src.core.seed_stream_screening import EvidenceIdentity
from src.infra.prospective_generation_bundles import FileProspectiveGenerationBundleReader
from src.infra.prospective_generation_ownership import FileGenerationRequestOwner
def forbidden(*args, **kwargs): raise AssertionError("child executed scientific work")
for name in ("_build_phase_a_roles", "_build_phase_b_roles", "release_final_test"): setattr(arrived,name,forbidden)
for model in (BackpropMLP,PredictiveCodingNetwork,CircadianPredictiveCodingNetwork):
    model.__init__=forbidden; model.compute_accuracy=forbidden
np.random.default_rng=forbidden
for name in ("array","asarray","ascontiguousarray"): setattr(np,name,forbidden)
body=json.loads(Path(sys.argv[3]).read_bytes())
for name in ("request_identity","source_map_identity","code_manifest_identity"): body[name]=EvidenceIdentity(**body[name])
body["code_paths"]=tuple(body["code_paths"])
spec=ProspectiveBundleSpec(**body)
with claim_prospective_generation_bundle(fixed_prospective_design(),spec,FileProspectiveGenerationBundleReader(Path(sys.argv[1])),FileGenerationRequestOwner(Path(sys.argv[1]),Path(sys.argv[2]))) as result:
    assert len(result.bundle.snapshot.generation_request.role_recipes)==480
    assert not result.execution_authorized
    print("HELD",flush=True)
    command=sys.stdin.readline().strip()
    if command=="die": os._exit(73)
print("RELEASED",flush=True)
"""


@contextmanager
def child_owner(bundle: GenerationBundleFixture, registry: Path, tmp_path: Path) -> Iterator[Any]:
    spec_path = tmp_path / "child-spec.json"
    spec_path.write_text(json.dumps(asdict(bundle.spec)), encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-X",
            "utf8",
            "-c",
            CHILD,
            str(bundle.root),
            str(registry),
            str(spec_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        cwd=Path(__file__).resolve().parents[1],
    )
    input_stream, output_stream, error_stream = process.stdin, process.stdout, process.stderr
    assert input_stream is not None and output_stream is not None and error_stream is not None
    ready: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: ready.put(output_stream.readline()), daemon=True).start()
    try:
        assert ready.get(timeout=10).strip() == "HELD", (
            "native child did not acquire its full request"
        )
        yield process
    finally:
        input_stream.close()
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        output_stream.close()
        error_stream.close()


@pytest.mark.parametrize("termination", ("normal", "death"))
def test_should_enforce_cross_process_contention_and_release_after_process_death(
    bundle: GenerationBundleFixture,
    registry: Path,
    tmp_path: Path,
    termination: str,
) -> None:
    with child_owner(bundle, registry, tmp_path) as process:
        with pytest.raises(RuntimeError, match="already active"):
            with owned(bundle, registry):
                pytest.fail("second process bypassed the native request lease")
        process.stdin.write("die\n" if termination == "death" else "release\n")
        process.stdin.flush()
        assert process.wait(timeout=10) == (73 if termination == "death" else 0)
    with owned(bundle, registry):
        pass


@pytest.mark.parametrize("kind", ("request", "sources", "manifest", "last_code"))
def test_should_recheck_every_full_file_on_failure_exit_and_release_the_owner(
    bundle: GenerationBundleFixture,
    registry: Path,
    kind: str,
) -> None:
    name = {
        "request": bundle.spec.request_path,
        "sources": bundle.spec.source_map_path,
        "manifest": bundle.spec.code_manifest_path,
        "last_code": bundle.spec.code_paths[-1],
    }[kind]
    path = bundle.root / name
    original = path.read_bytes()
    with pytest.raises(ValueError):
        with owned(bundle, registry):
            path.write_bytes(original[:-2] + bytes((original[-2] ^ 1,)) + original[-1:])
    path.write_bytes(original)
    with owned(bundle, registry):
        pass


@pytest.mark.parametrize("kind", ("bad_bytes", "directory", "hardlink"))
def test_should_reject_noncanonical_or_aliased_permanent_registry_files_without_rewriting(
    bundle: GenerationBundleFixture,
    registry: Path,
    tmp_path: Path,
    kind: str,
) -> None:
    with owned(bundle, registry) as result:
        path = registry / result.ownership_at_entry.lock_file_name
    if kind == "bad_bytes":
        path.write_bytes(b"x")
    elif kind == "directory":
        path.unlink()
        path.mkdir()
    else:
        os.link(path, tmp_path / "lock-alias")
    with pytest.raises(ValueError):
        with owned(bundle, registry):
            pytest.fail("invalid native lease namespace was accepted")
    if kind == "bad_bytes":
        assert path.read_bytes() == b"x"


@pytest.mark.parametrize("kind", ("append", "hardlink", "replacement"))
def test_should_prevent_or_detect_live_registry_namespace_drift_and_always_release(
    bundle: GenerationBundleFixture,
    registry: Path,
    tmp_path: Path,
    kind: str,
) -> None:
    snapshot = reader(bundle).read_bundle(bundle.spec)
    port = FileGenerationRequestOwner(bundle.root, registry)
    changed = False
    with port.claim(snapshot) as lease:
        observation = lease.observe(snapshot)
        path = registry / observation.lock_file_name
        try:
            if kind == "append":
                with path.open("ab") as stream:
                    stream.write(b"x")
            elif kind == "hardlink":
                os.link(path, tmp_path / "live-alias")
            else:
                path.rename(tmp_path / "old-lock")
                path.write_bytes(b"0")
            changed = True
        except OSError:
            # Native Windows denial also satisfies prevention before consumption.
            assert lease.observe(snapshot).native_lock_observed
        if changed:
            with pytest.raises(ValueError):
                lease.observe(snapshot)
    if kind == "hardlink" and changed:
        (tmp_path / "live-alias").unlink()
    path.write_bytes(b"0")
    with owned(bundle, registry):
        pass


@pytest.mark.parametrize("kind", ("last_recipe", "last_stream", "owner", "caps", "code"))
def test_should_reject_complete_domain_or_physical_corruption_before_claiming_any_owner(
    bundle: GenerationBundleFixture,
    kind: str,
) -> None:
    if kind == "last_recipe":
        bundle.body["generation_request"]["role_recipes"][-1]["allowed_use"] = "selection"
        reseal_generation_request(bundle.body)
    elif kind == "last_stream":
        bundle.body["generation_request"]["streams"][-1]["value"] = True
        reseal_generation_request(bundle.body)
    elif kind == "owner":
        bundle.body["owner_id"] = "F" * 32
    elif kind == "caps":
        bundle.body["resource_envelope"]["rss_bytes"] = 1
    else:
        (bundle.root / bundle.spec.code_paths[-1]).write_bytes(b"foreign code")
    if kind != "code":
        bundle.save()

    class ForbiddenOwner:
        def claim(self, snapshot: Any) -> Any:
            raise AssertionError("invalid full metadata reached a native owner")

    with pytest.raises(ValueError):
        with claim_prospective_generation_bundle(
            fixed_prospective_design(), bundle.spec, reader(bundle), ForbiddenOwner()
        ):
            pytest.fail("invalid full request yielded ownership")


@pytest.mark.parametrize(
    "kind",
    (
        "foreign",
        "owner",
        "identity_float",
        "identity_bool",
        "observed_float",
        "observed_naive",
        "prospective_future",
    ),
)
def test_should_reject_foreign_or_type_aliased_scopes_before_native_file_access(
    bundle: GenerationBundleFixture,
    registry: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    snapshot: Any = reader(bundle).read_bundle(bundle.spec)
    if kind == "foreign":
        snapshot = {"scope": "partial"}
    elif kind == "owner":
        snapshot = replace(snapshot, owner_id=1)
    elif kind == "observed_float":
        snapshot = replace(snapshot, observed_utc=1.0)
    elif kind == "observed_naive":
        snapshot = replace(snapshot, observed_utc="2026-10-05T13:00:00")
    elif kind == "prospective_future":
        snapshot = replace(snapshot, prospective_utc="2026-10-06T12:00:00+00:00")
    else:
        request = snapshot.generation_request
        value = float(request.request_identity.byte_count) if kind == "identity_float" else True
        snapshot = replace(
            snapshot,
            generation_request=replace(
                request, request_identity=replace(request.request_identity, byte_count=value)
            ),
        )
    port = FileGenerationRequestOwner(bundle.root, registry)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("invalid scope accessed a native registry file")

    monkeypatch.setattr(os, "open", forbidden)
    with pytest.raises(ValueError):
        with port.claim(snapshot):
            pytest.fail("foreign live owner scope was accepted")


@pytest.mark.parametrize(
    "kind",
    (
        "foreign",
        "sequence_bool",
        "sequence_float",
        "owner",
        "nonce",
        "device_bool",
        "inode_float",
        "unheld",
    ),
)
def test_should_deny_foreign_or_detached_observation_ports_before_yield(
    bundle: GenerationBundleFixture,
    registry: Path,
    kind: str,
) -> None:
    class InvalidLease:
        def observe(self, snapshot: Any) -> Any:
            observation: Any = GenerationOwnershipObservation(
                generation_ownership_scope(snapshot),
                registry.resolve().as_posix(),
                f".p67-generation-{snapshot.generation_request.request_identity.sha256}.owner.lock",
                1,
                2,
                "3" * 32,
                "2026-10-05T14:00:00+00:00",
                "2026-10-05T14:00:00+00:00",
                0,
            )
            if kind == "foreign":
                return asdict(observation)
            if kind == "owner":
                return replace(observation, scope=replace(observation.scope, owner_id="2" * 32))
            values: dict[str, dict[str, Any]] = {
                "sequence_bool": {"sequence": False},
                "sequence_float": {"sequence": 0.0},
                "nonce": {"lease_nonce": "F" * 32},
                "device_bool": {"physical_device": True},
                "inode_float": {"physical_inode": 2.0},
                "unheld": {"native_lock_observed": False},
            }
            return replace(observation, **values[kind])

    class InvalidOwner:
        @contextmanager
        def claim(self, snapshot: Any) -> Iterator[InvalidLease]:
            yield InvalidLease()

    with pytest.raises(ValueError):
        with claim_prospective_generation_bundle(
            fixed_prospective_design(), bundle.spec, reader(bundle), InvalidOwner()
        ):
            pytest.fail("foreign observation yielded a trusted ownership result")


def test_should_deny_a_backward_native_clock_without_advancing_the_observation(
    bundle: GenerationBundleFixture,
    registry: Path,
) -> None:
    snapshot = reader(bundle).read_bundle(bundle.spec)
    times = iter(datetime(2026, 10, 5, hour, tzinfo=timezone.utc) for hour in (14, 16, 15, 17))
    port = FileGenerationRequestOwner(bundle.root, registry, clock=lambda: next(times))
    with port.claim(snapshot) as lease:
        first = lease.observe(snapshot)
        with pytest.raises(ValueError, match="backward"):
            lease.observe(snapshot)
        final = lease.observe(snapshot)
        assert final.sequence == first.sequence + 1
        assert final.observed_utc > first.observed_utc


@pytest.mark.parametrize("value", (None, datetime(2026, 10, 5, 14), "2026-10-05T14:00:00+00:00"))
def test_should_release_native_lock_after_invalid_clock_without_yielding_ownership(
    bundle: GenerationBundleFixture,
    registry: Path,
    value: Any,
) -> None:
    snapshot = reader(bundle).read_bundle(bundle.spec)
    port = FileGenerationRequestOwner(bundle.root, registry, clock=lambda: value)
    with pytest.raises(ValueError, match="clock"):
        with port.claim(snapshot):
            pytest.fail("invalid clock yielded a native owner")
    with owned(bundle, registry):
        pass


@pytest.mark.parametrize("kind", ("sequence", "nonce", "inode", "scope", "clock"))
def test_should_reject_late_real_port_observation_drift_and_release_the_native_owner(
    bundle: GenerationBundleFixture,
    registry: Path,
    kind: str,
) -> None:
    native = FileGenerationRequestOwner(bundle.root, registry)

    class DriftingLease:
        def __init__(self, lease: Any) -> None:
            self.lease = lease

        def observe(self, snapshot: Any) -> GenerationOwnershipObservation:
            current = self.lease.observe(snapshot)
            if current.sequence != 2:
                return current
            if kind == "scope":
                return replace(current, scope=replace(current.scope, owner_id="2" * 32))
            fields: dict[str, dict[str, Any]] = {
                "sequence": {"sequence": 0},
                "nonce": {"lease_nonce": "f" * 32},
                "inode": {"physical_inode": current.physical_inode + 1},
                "clock": {"observed_utc": "2026-10-05T14:00:00+00:00"},
            }
            return replace(current, **fields[kind])

    class DriftingOwner:
        @contextmanager
        def claim(self, snapshot: Any) -> Iterator[DriftingLease]:
            with native.claim(snapshot) as lease:
                yield DriftingLease(lease)

    with pytest.raises(ValueError):
        with claim_prospective_generation_bundle(
            fixed_prospective_design(), bundle.spec, reader(bundle), DriftingOwner()
        ) as result:
            assert not result.execution_authorized
    with owned(bundle, registry):
        pass
