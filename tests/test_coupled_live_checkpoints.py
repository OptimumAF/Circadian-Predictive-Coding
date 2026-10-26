"""Live saver DATA boundaries, independent of operational native model methods.

Exact synthetic graphs exercise the real capture/codec/store pipeline. Routing
uses stubs; unchanged serving regressions and the separate native control own
the positive B/C continuation seam, without weakening exact capture schemas.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, cast

import numpy as np
import pytest

import scripts.run_coupled_candidate_handoff as cli
import src.adapters.coupled_live_checkpoints as module
import src.adapters.coupled_managed_serving as serving
import src.infra.checkpoint_files as files
from src.adapters.coupled_composite_codec import CoupledCompositeCheckpointCodec
from src.core.checkpoint_codec import CodecBinding
from src.infra.checkpoint_files import BoundedCheckpointFileStore
from test_coupled_composite_codec import assert_graph, metadata
from test_coupled_composite_files import (
    DISK_WORK as DISK_WORK,
    account_owned_file_work as account_owned_file_work,
)
from test_coupled_managed_composite_capture import (
    forbid_operations,
    graph_factory as graph_factory,
)


def options():
    return dict(
        binding=module.BINDING,
        capture_limits=module.CAPTURE_LIMITS,
        wire_limits=module.WIRE_LIMITS,
    )


def forbidden(*args, **kwargs):
    pytest.fail("operation must not be reached")


def account_synthetic_attachments(graph):
    life = graph["life"]
    ledger = life._copy_budget
    assert ledger is not None
    with life._registry._lease() as groups:
        retained = life._retained_bytes(groups)
    # Why: the rich fixture installs cache/metadata/promotion references directly,
    # bypassing the adapters that normally charge those newly owned bytes.
    ledger.reserve(max(0, retained - ledger._charged))
    return life.payload_byte_snapshot()


@pytest.mark.parametrize("family", ("pc", "neutral", "active"))
@pytest.mark.parametrize("guarded", (False, True))
def test_should_save_detached_complete_data_and_charge_only_original_raw_capture(
    graph_factory, monkeypatch, tmp_path, family, guarded
):
    graph = graph_factory(family, True, guarded)
    actor = graph["runtime"]._actor
    typed = np.array([[-0.0, 0.25], [0.5, -0.75]], dtype=">f8", order="F")
    typed.flags.writeable = False
    actor._slot.bundle.metadata["typed"] = typed
    before = account_synthetic_attachments(graph)
    inbox = graph["runtime"]._inbox.capture_cursor()
    identities = {name: id(graph[name]) for name in ("owner", "life", "runtime", "shared")}
    captured: list[Any] = []
    original = module.capture_coupled_managed_composite

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(module, "capture_coupled_managed_composite", capture)
    forbid_operations(monkeypatch)
    path = tmp_path / "complete.json"
    receipt = module.save_coupled_managed_checkpoint(graph["owner"], path, **options())
    raw = path.read_bytes()
    assert receipt == dict(
        path=str(path.absolute()),
        digest=sha256(raw).hexdigest(),
        data_only=True,
        retained_payload_bytes=before.observed_retained_bytes,
    )
    assert len(captured) == 1
    restored = BoundedCheckpointFileStore(
        path, CoupledCompositeCheckpointCodec(capture_limits=module.CAPTURE_LIMITS)
    ).load(binding=module.BINDING, expected_sha256=receipt["digest"], limits=module.WIRE_LIMITS)
    assert_graph(restored, captured[0].state)
    assert metadata(restored)["typed"].tobytes(order="A") == typed.tobytes(order="A")
    assert not hasattr(restored, "authority") and type(restored) is not type(captured[0])
    after = graph["life"].payload_byte_snapshot()
    assert after.charged_bytes == before.charged_bytes + before.observed_retained_bytes
    assert after.observed_retained_bytes == before.observed_retained_bytes
    assert after.limits is before.limits
    assert identities == {name: id(graph[name]) for name in identities}
    assert_graph(graph["runtime"]._inbox.capture_cursor(), inbox)
    assert graph["token"] in graph["checkpoint"]._pending
    assert sorted(item.name for item in tmp_path.iterdir()) == ["complete.json"]
    assert len(json.dumps(receipt, allow_nan=False).encode()) < 65536


@pytest.mark.parametrize("fault", ("path", "binding", "capture", "nested", "wire"))
def test_should_refuse_invalid_independent_controls_before_capture_or_io(
    monkeypatch, tmp_path, fault
):
    kwargs: dict[str, Any] = options()
    path: Any = tmp_path / "never.json"
    if fault == "path":
        path = str(path)
    elif fault == "binding":
        bad = CodecBinding("a" * 64, "b" * 64)
        object.__setattr__(bad, "source_sha256", True)
        kwargs["binding"] = bad
    elif fault == "capture":
        kwargs["capture_limits"] = object()
    elif fault == "nested":
        bad_capture = deepcopy(module.CAPTURE_LIMITS)
        object.__setattr__(bad_capture.records, "max_records", True)
        kwargs["capture_limits"] = bad_capture
    else:
        bad_wire = replace(module.WIRE_LIMITS)
        object.__setattr__(bad_wire, "max_encoded_bytes", False)
        kwargs["wire_limits"] = bad_wire
    monkeypatch.setattr(module, "capture_coupled_managed_composite", forbidden)
    monkeypatch.setattr(module, "BoundedCheckpointFileStore", forbidden)
    with pytest.raises(ValueError):
        module.save_coupled_managed_checkpoint(cast(Any, object()), path, **kwargs)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("present", (False, True))
@pytest.mark.parametrize("boundary", ("fsync", "replace"))
def test_should_preserve_target_and_spent_capture_charge_when_storage_fails(
    graph_factory, monkeypatch, tmp_path, present, boundary
):
    graph = graph_factory("active", True)
    before = account_synthetic_attachments(graph)
    path = tmp_path / "complete.json"
    old = b"previous independently owned bytes"
    if present:
        path.write_bytes(old)
    error = OSError(f"injected {boundary}")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(files.os, boundary, fail)
    forbid_operations(monkeypatch)
    with pytest.raises(OSError) as failure:
        module.save_coupled_managed_checkpoint(graph["owner"], path, **options())
    assert failure.value is error
    assert path.read_bytes() == old if present else not path.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ([path.name] if present else [])
    after = graph["life"].payload_byte_snapshot()
    assert after.charged_bytes == before.charged_bytes + before.observed_retained_bytes
    assert after.observed_retained_bytes == before.observed_retained_bytes


@pytest.mark.parametrize("bound", ("wire", "capture"))
def test_should_refuse_small_bounds_without_publishing_or_refunding_completed_capture(
    graph_factory, monkeypatch, tmp_path, bound
):
    graph = graph_factory("pc", True)
    before = account_synthetic_attachments(graph)
    kwargs = options()
    if bound == "wire":
        kwargs["wire_limits"] = replace(module.WIRE_LIMITS, max_encoded_bytes=1)
    else:
        kwargs["capture_limits"] = replace(module.CAPTURE_LIMITS, max_array_bytes=1)
    forbid_operations(monkeypatch)
    with pytest.raises(ValueError):
        module.save_coupled_managed_checkpoint(graph["owner"], tmp_path / "none", **kwargs)
    after = graph["life"].payload_byte_snapshot()
    assert after.charged_bytes == before.charged_bytes + (
        before.observed_retained_bytes if bound == "wire" else 0
    )
    assert after.observed_retained_bytes == before.observed_retained_bytes
    assert not list(tmp_path.iterdir())


def test_should_propagate_capture_refusal_without_constructing_file_payload(monkeypatch, tmp_path):
    error = ValueError("original owner is busy or expired")

    def capture(*args, **kwargs):
        raise error

    monkeypatch.setattr(module, "capture_coupled_managed_composite", capture)
    monkeypatch.setattr(BoundedCheckpointFileStore, "save", forbidden)
    with pytest.raises(ValueError) as failure:
        module.save_coupled_managed_checkpoint(cast(Any, object()), tmp_path / "none", **options())
    assert failure.value is error and not list(tmp_path.iterdir())


def test_should_route_each_fixed_case_through_two_named_savers_in_the_new_directory(
    monkeypatch, tmp_path
):
    directory = tmp_path / "new"
    calls: list[tuple[str, str]] = []
    saves: list[tuple[object, Path, dict[str, Any]]] = []

    def save(owner, path, **kwargs):
        assert directory.is_dir()
        saves.append((owner, path, kwargs))
        return dict(
            path=str(path.absolute()), digest="a" * 64, data_only=True, retained_payload_bytes=24
        )

    def case(family, mode, *, committed, checkpoint_observer):
        assert committed is True
        calls.append((family, mode))
        owner = object()
        return dict(
            family=family,
            mode=mode,
            complete_checkpoints=[
                dict(phase=phase, **checkpoint_observer(owner, phase)) for phase in ("B", "C")
            ],
        )

    def report(rows, *, committed):
        assert committed is True
        return dict(cases=rows, original_report_preserved=True)

    monkeypatch.setattr(module, "save_coupled_managed_checkpoint", save)
    monkeypatch.setattr(module, "_case", case)
    monkeypatch.setattr(module, "_report", report)
    result = module.run_complete_checkpoint_example(directory)
    assert calls == [(family, mode) for mode in module.MODES for family in module.FAMILIES]
    assert len(saves) == 12 and result["original_report_preserved"] is True
    assert result["complete_checkpoints_verified"] is result["data_only"] is True
    assert result["directory"] == str(directory.absolute())
    for (family, mode), row in zip(calls, result["cases"]):
        assert [r["phase"] for r in row["complete_checkpoints"]] == ["B", "C"]
        assert [Path(r["path"]).name for r in row["complete_checkpoints"]] == [
            f"{mode}-{family}-B.json",
            f"{mode}-{family}-C.json",
        ]
    assert all(kwargs == options() for _, _, kwargs in saves)
    assert all(saves[index][0] is saves[index + 1][0] for index in range(0, 12, 2))


@pytest.mark.parametrize("kind", ("file", "directory"))
def test_should_preserve_existing_output_before_any_case_or_save(monkeypatch, tmp_path, kind):
    path = tmp_path / "existing"
    if kind == "file":
        path.write_bytes(b"caller marker")
    else:
        path.mkdir()
    monkeypatch.setattr(module, "_case", forbidden)
    monkeypatch.setattr(module, "save_coupled_managed_checkpoint", forbidden)
    with pytest.raises(FileExistsError):
        module.run_complete_checkpoint_example(path)
    assert path.read_bytes() == b"caller marker" if kind == "file" else list(path.iterdir()) == []


def test_should_describe_original_controls_and_extra_data_charges_without_operational_work(
    monkeypatch,
):
    monkeypatch.setattr(module, "_case", forbidden)
    monkeypatch.setattr(module, "save_coupled_managed_checkpoint", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    description = module.describe_complete_checkpoint_example()
    assert description["protocol_id"] == "coupled_live_complete_checkpoints_engineering_v1"
    assert description["checkpoint_phases"] == ["B", "C"]
    assert description["capture_raw_charges"] == [328, 512]
    assert description["checkpoint_files"] == description["capture_calls"] == 12
    assert description["copy_charges"][-1] == 1496
    assert description["retained_raw_bytes"][-1] == 0
    assert description["capture_limits"] == asdict(module.CAPTURE_LIMITS)
    assert description["wire_limits"] == asdict(module.WIRE_LIMITS)
    assert description["binding"] == asdict(module.BINDING)
    old = serving.describe_committed_serving_example()
    changed = {
        "protocol_id",
        "retention_policy",
        "limitations",
        "copy_charges",
        "retained_raw_bytes",
    }
    assert description["copy_charges"] == [
        24,
        168,
        184,
        328,
        472,
        800,
        944,
        968,
        968,
        984,
        984,
        1496,
        1496,
    ]
    assert description["retained_raw_bytes"] == [
        24,
        168,
        184,
        328,
        328,
        328,
        472,
        496,
        496,
        512,
        512,
        512,
        0,
    ]
    assert all(description[key] == value for key, value in old.items() if key not in changed)


@pytest.mark.parametrize("mode", ("pending", "busy", "failure", "not_callable"))
def test_should_refuse_ineligible_observer_before_any_native_setup(monkeypatch, mode):
    kwargs: dict[str, Any] = dict(committed=True, checkpoint_observer=lambda *a: {})
    if mode == "pending":
        kwargs["committed"] = False
    elif mode == "busy":
        kwargs["busy_stop"] = True
    elif mode == "failure":
        kwargs["cleanup_failure"] = True
    else:
        kwargs["checkpoint_observer"] = object()
    monkeypatch.setattr(serving, "_model", forbidden)
    monkeypatch.setattr(serving.np, "array", forbidden)
    with pytest.raises(ValueError, match="checkpoint observer"):
        serving._case("pc", "fixed", **kwargs)


@pytest.mark.parametrize("flag", (True, 1, None, "true"))
def test_should_require_explicit_committed_exact_boolean_copy_policy(flag):
    with pytest.raises(ValueError):
        serving._retention_policy(complete_checkpoints=flag)
    for policy, expected in (
        (serving._retention_policy(), 640),
        (serving._retention_policy(committed=True), 656),
        (serving._retention_policy(committed=True, complete_checkpoints=True), 1496),
    ):
        copies = policy.owned_payload_copies
        assert copies is not None and copies.max_lifetime_owned_bytes == expected


@pytest.mark.parametrize("action", ("describe", "run"))
def test_should_route_cli_to_only_the_requested_complete_mode(monkeypatch, capsys, action):
    calls: list[str | Path] = []

    def describe():
        calls.append("describe")
        return {"tuple_boundary": ("B", "C")}

    def run(directory):
        calls.append(directory)
        return {"directory": str(directory), "data_only": True}

    monkeypatch.setattr(cli, "describe_complete_checkpoint_example", describe)
    monkeypatch.setattr(cli, "run_complete_checkpoint_example", run)
    monkeypatch.setattr(cli, "describe_example", forbidden)
    monkeypatch.setattr(cli, "run_example", forbidden)
    argv = ["handoff", f"--{action}", "--complete-checkpoints"]
    if action == "run":
        argv += ["--directory", "new"]
    monkeypatch.setattr(sys, "argv", argv)
    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert calls == (["describe"] if action == "describe" else [Path("new")])
    assert result == (
        {"tuple_boundary": ["B", "C"]}
        if action == "describe"
        else {"directory": "new", "data_only": True}
    )


@pytest.mark.parametrize(
    "arguments",
    (
        ["--run", "--complete-checkpoints"],
        ["--describe", "--complete-checkpoints", "--directory", "new"],
        ["--run", "--directory", "new"],
        ["--describe", "--complete-checkpoints", "--committed-serving-expiry"],
    ),
)
def test_should_refuse_ambiguous_cli_directory_or_mode_before_callbacks(monkeypatch, arguments):
    monkeypatch.setattr(cli, "describe_complete_checkpoint_example", forbidden)
    monkeypatch.setattr(cli, "run_complete_checkpoint_example", forbidden)
    monkeypatch.setattr(cli, "run_example", forbidden)
    monkeypatch.setattr(sys, "argv", ["handoff", *arguments])
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2


def test_should_construct_exact_unparameterized_owners_through_both_fake_serving_workflows(
    monkeypatch,
):
    from test_coupled_committed_serving import (
        test_should_expire_committed_current_and_previous_raw_bundles_without_native_rollback as committed,
    )
    from test_coupled_managed_serving import (
        test_should_expire_six_real_cache_promotion_transactions_with_owned_native_state as pending,
    )

    actor_type, runtime_type = serving.PromotableActor, serving.ActorShadowRuntime
    for workflow in (pending, committed):
        actors: list[Any] = []
        runtimes: list[Any] = []

        def actor(*args, **kwargs):
            value = actor_type(*args, **kwargs)
            assert type(value) is actor_type and "__orig_class__" not in vars(value)
            actors.append(value)
            return value

        def runtime(*args, **kwargs):
            value = runtime_type(*args, **kwargs)
            assert type(value) is runtime_type and "__orig_class__" not in vars(value)
            runtimes.append(value)
            return value

        # Why: each accepted workflow patches native fake ports and worker
        # methods; undo all of them before composing the next independent mode.
        with monkeypatch.context() as scoped:
            scoped.setattr(serving, "PromotableActor", actor)
            scoped.setattr(serving, "ActorShadowRuntime", runtime)
            workflow(scoped)
        assert len(actors) == len(runtimes) == 6
        assert all(type(value) is actor_type for value in actors)
        assert all(type(value) is runtime_type for value in runtimes)
        assert all("__orig_class__" not in vars(value) for value in (*actors, *runtimes))
