"""New literal workflow checks; no recorded data, SDK, neural or teacher work."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import push_cube_reserved_probe_inputs as inputs
from scripts import run_taskspace_memory_demo as cli
from scripts import taskspace_memory_native as native
from src.adapters import taskspace_memory_demo as demo
from src.adapters import numpy_nearest_continuous_bc as near
from src.core import continuous_bc_diagnostics as diagnostics
from test_taskspace_memory_demo import bundle
from test_push_cube_reserved_probe_inputs import reserved_episode
from test_taskspace_memory_native import receipt as native_receipt

WORK: defaultdict[str, int] = defaultdict(int)
CAPS = dict(
    inits=16,
    appends=128,
    rows=512,
    predictions=256,
    queries=4096,
    candidates=131072,
    features=1572864,
    kernels=32,
    pairs=65536,
    conversions=65536,
    snapshots=65536,
    snapshot_bytes=256 * 1024**2,
    forbidden=0,
)


def charge(key, value=1):
    WORK[key] += value
    assert WORK[key] <= CAPS[key], "permanent new fixture cap: " + key


def forbidden(*args, **kwargs):
    charge("forbidden")


@pytest.fixture(autouse=True)
def budget(monkeypatch):
    originals = {
        name: getattr(near.NumpyNearestContinuousBCLearner, name)
        for name in ("__init__", "train_batch", "predict_with_work", "snapshot_state")
    }

    def initialize(owner, policy):
        charge("inits")
        return originals["__init__"](owner, policy)

    def append(owner, features, targets):
        charge("appends")
        charge("rows", len(features))
        return originals["train_batch"](owner, features, targets)

    def predict(owner, features):
        charge("predictions")
        charge("queries", len(features))
        count = len(features) * owner._state.training_examples
        charge("candidates", count)
        charge("features", count * 12)
        return originals["predict_with_work"](owner, features)

    def snapshot(owner):
        charge("snapshots")
        result = originals["snapshot_state"](owner)
        charge("snapshot_bytes", result.prototype_features.nbytes + result.prototype_actions.nbytes)
        return result

    kernel = diagnostics.continuous_input_target_consistency

    def consistency(features, targets, spec, **options):
        charge("kernels")
        charge("pairs", len(features) * (len(features) - 1) // 2)
        charge("conversions", len(features) * 4)
        return kernel(features, targets, spec, **options)

    for name, function in (
        ("__init__", initialize),
        ("train_batch", append),
        ("predict_with_work", predict),
        ("snapshot_state", snapshot),
    ):
        monkeypatch.setattr(near.NumpyNearestContinuousBCLearner, name, function)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(diagnostics, "continuous_input_target_consistency", consistency)


def test_cpu_private_probe_learns_and_retains_delayed_charges_without_sdk():
    payload = bundle()
    original = deepcopy(payload)
    result = demo.run_demo(payload, training_probe=reserved_episode("probe_train"))
    assert result["complete"], result
    assert result["mode"] == "private_training_reproduction"
    probe = result["probe"]
    assert probe["plan"]["evaluation_id"] is None
    assert probe["source_updates"] == 7 and probe["source_training_examples"] == 7
    assert probe["progress"]["updates_completed"] == 1
    assert probe["events"][0]["deferred_exact"] and probe["events"][0]["success_validated"]
    assert probe["metrics"]["training_fit_gain"] > 0
    assert probe["work"]["score_rows_returned_by_role"] == dict(
        probe_train=36, probe_test=0, train=0
    )
    assert probe["training_consistency"]
    assert (
        probe["checkpoints"][-1]["scores"]["training"]["residual"]["overall"]["normalized_half_mse"]
        == 0
    )
    assert result["stream"]["model_state"]["candidate_prototype_rows"] == 7
    assert payload == original


@pytest.mark.parametrize("fault", ["after", "source"])
def test_paid_private_append_survives_late_endpoint_failure(fault):
    def endpoint(owner, phase, guard):
        if phase == "after_probe":
            if fault == "source":
                owner._state.prototype_features = owner._state.prototype_features.copy()
            else:
                raise RuntimeError("late paid observer failure")
        guard()

    result = demo.run_demo(
        bundle(), training_probe=reserved_episode("probe_train"), endpoint_observer=endpoint
    )
    assert result["complete"] is False and result["error"]
    assert result["probe"]["progress"]["updates_completed"] == 1
    assert len(result["probe"]["applied_updates"]) == 1
    assert result["probe"]["work"]["training_rows_attempted"] == 18
    assert result["stream"]["work"]["validated_updates"] == 6


@pytest.mark.parametrize("fault", ["role", "id", "history", "rows", "callback", "capacity"])
def test_api_refuses_bad_probe_before_any_model(monkeypatch, fault):
    training = reserved_episode("probe_train")
    payload = bundle()
    options: dict[str, Any] = {}
    if fault == "role":
        training = replace(training, role="probe_test")
    elif fault == "id":
        training = replace(training, episode_id="other")
    elif fault == "rows":
        training = replace(
            training,
            transitions=(
                *training.transitions[:-2],
                replace(training.transitions[-2], terminated=True),
            ),
        )
    elif fault == "history":
        row = training.transitions[0]
        training = replace(
            training, transitions=(replace(row, observation=(0.1,) * 12), *training.transitions[1:])
        )
    elif fault == "callback":
        options["endpoint_observer"] = True
    else:
        # Admission is already qualified separately; exercise the capacity guard.
        episodes, recordings, provenance = demo._admit(payload)
        padded = SimpleNamespace(transitions=(None,) * 239)
        monkeypatch.setattr(demo, "_admit", lambda value: ((padded,), recordings, provenance))
    monkeypatch.setattr(near.NumpyNearestContinuousBCLearner, "__init__", forbidden)
    with pytest.raises((ValueError, TypeError)):
        demo.run_demo(payload, training_probe=training, **options)


@pytest.mark.parametrize("option", ["endpoint_observer", "check_boundary"])
def test_default_api_refuses_probe_callback_without_probe(option):
    options: dict[str, Any] = {option: lambda: None}
    with pytest.raises(ValueError):
        demo.run_demo({}, **options)


@pytest.mark.parametrize("fault", ["good", "digest", "decoded", "replace", "receipt"])
def test_explicit_training_file_keeps_original_single_role_qualification(
    tmp_path, monkeypatch, fault
):
    episode = reserved_episode("probe_train")
    raw = json.dumps(asdict(episode)).encode()
    path = tmp_path / "explicit.json"
    path.write_bytes(raw)
    monkeypatch.setattr(
        inputs,
        "PROBES",
        {
            "probe_train": (tmp_path / "MUST-NOT-READ.json", hashlib.sha256(raw).hexdigest()),
            "probe_test": (tmp_path / "FORBIDDEN24.json", "never"),
        },
    )
    reader = inputs.read_continuous_probe_episode
    calls: list[str] = []

    def read(selected, *, role):
        assert selected == path and role == "probe_train"
        calls.append(role)
        value, size = reader(selected, role=role)
        if fault == "decoded":
            value = replace(value, episode_id="replaced")
        if fault == "replace":
            selected.write_bytes(raw + b" ")
        return value, size + (fault == "receipt")

    monkeypatch.setattr(inputs, "read_continuous_probe_episode", read)
    if fault == "digest":
        path.write_bytes(b"{}")
    if fault == "good":
        result, receipt = inputs.load_reserved_training_input(path)
        assert result == episode and receipt["path"] == str(path)
        assert receipt["sha256"] == hashlib.sha256(raw).hexdigest()
    else:
        with pytest.raises(ValueError):
            inputs.load_reserved_training_input(path)
    assert calls == ([] if fault == "digest" else ["probe_train"])


@pytest.mark.parametrize("mode", ["default", "cpu", "native", "development"])
def test_cli_dispatch_preserves_defaults_and_explicit_modes(tmp_path, monkeypatch, capsys, mode):
    calls: list[Any] = []
    payload: dict[str, Any] = {}
    training = reserved_episode("probe_train")
    output = tmp_path / "result.json"
    report = dict(complete=True, mode=mode)
    monkeypatch.setattr(cli, "load_bundle", lambda path: payload)
    monkeypatch.setattr(cli, "_load_default_bundle", lambda: payload)
    monkeypatch.setattr(
        inputs, "load_reserved_training_input", lambda path: (training, {"role": "probe_train"})
    )

    def run(value, **options):
        calls.append(options)
        assert value is payload
        return report

    def run_native(value, *arguments, **options):
        destination = arguments[-1]
        calls.append(options)
        assert destination == output and value is payload
        if "training_input_receipt" in options:
            report["training_input_receipt"] = options["training_input_receipt"]
        destination.write_text(json.dumps(report))
        return report

    monkeypatch.setattr(cli, "run_demo", run)
    monkeypatch.setattr(native, "run_native_demo", run_native)
    monkeypatch.setattr(native, "run_native_training_probe", run_native)
    args = ["--output", str(output)]
    if mode in ("cpu", "native"):
        args += ["--training-probe", str(tmp_path / "original23.json")]
    if mode in ("native", "development"):
        args.append("--native")
    if mode == "development":
        args.append("--development-probe")
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(output.read_text())
    if mode == "default":
        assert calls == [{}]
    elif mode == "cpu":
        assert calls == [{"training_probe": training}]
    elif mode == "development":
        assert calls == [
            {"development_scenes": (native.NativeScene("push", 19), native.NativeScene("push", 20))}
        ]


@pytest.mark.parametrize(
    "args",
    [
        ["--training-probe", "23.json"],
        ["--training-probe", "23.json", "--development-probe", "--native"],
        ["--development-probe"],
        ["--native"],
    ],
)
def test_incompatible_modes_refuse_before_reads(monkeypatch, args):
    monkeypatch.setattr(cli, "_load_default_bundle", forbidden)
    monkeypatch.setattr(inputs, "load_reserved_training_input", forbidden)
    with pytest.raises(SystemExit):
        cli.main(args)


@pytest.mark.parametrize("which", [0, 1, 2, 3])
def test_all_probe_capture_collisions_refuse_before_input(tmp_path, monkeypatch, which):
    output = tmp_path / "result.json"
    destinations = demo.training_probe_outputs(output)
    destinations[which].write_text("original paid data")
    monkeypatch.setattr(inputs, "load_reserved_training_input", forbidden)
    monkeypatch.setattr(cli, "_load_default_bundle", forbidden)
    with pytest.raises(ValueError):
        cli.main(["--training-probe", "23.json", "--output", str(output)])
    assert destinations[which].read_text() == "original paid data"


def fake_workflow(monkeypatch, fault=None):
    calls: list[str] = []
    holders: list[Any] = []
    capture_paths: list[Path] = []
    owner = SimpleNamespace(_state=SimpleNamespace(training_examples=7))

    def run(value, *, training_probe, endpoint_observer, check_boundary):
        for phase in ("before_probe", "after_probe"):
            if phase == "after_probe" and fault == "between_file":
                capture_paths[0].write_text("{}")
            elif phase == "after_probe" and fault == "between_holder":
                holders[0].report = deepcopy(holders[0].report)
            check_boundary()
            if phase == "after_probe":
                owner._state.training_examples = 25
            try:
                endpoint_observer(
                    owner, phase, lambda callback=None: callback() if callback else None
                )
            except Exception as error:
                return dict(
                    complete=False, error=str(error), paid_append=int(phase == "after_probe")
                )
        return dict(complete=True, paid_append=1)

    def rollout(path, *, learner, probe_train_seed, action_history_recipe):
        assert probe_train_seed == 23 and action_history_recipe.mode == "executed"
        phase = "before_probe" if not calls else "after_probe"
        calls.append(phase)
        if fault == "erase_attempt_throw":
            sys._getframe(1).f_locals["report"]["attempted"].clear()
            raise RuntimeError("lost native call after erased visible attempts")
        if fault == "throw":
            raise RuntimeError("SDK failed; cleanup belongs to frozen port")
        raw = native_receipt(17)
        rows = learner._state.training_examples
        raw.update(
            source_updates=7 + (phase == "after_probe"),
            source_training_rows=rows,
            role="probe_train",
            seed=23,
            scoring_only=True,
            nearest_work_scope="validated_native_work",
            nearest_distance_candidate_upper_bound=rows,
            nearest_distance_candidates_completed=rows,
            nearest_distance_feature_upper_bound=rows * 12,
            nearest_distance_feature_values_completed=rows * 12,
        )
        raw["episode"].update(
            context="pick-cube-reserved-probe-23-24", episode_id="pick-cube-probe-train-smoke-23"
        )
        if fault == "wrong_role":
            raw["role"] = "train"
        elif fault == "incomplete":
            raw["complete"] = False
        holder = SimpleNamespace(report=raw, corrections=None)
        holders.append(holder)
        path.write_text(json.dumps(raw))
        capture_paths.append(path)
        return holder

    monkeypatch.setattr(native, "run_demo", run)
    monkeypatch.setattr(native, "run_frozen_rollout", rollout)
    return calls, holders


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "wrong_role",
        "incomplete",
        "throw",
        "file",
        "holder",
        "late_io",
        "between_file",
        "between_holder",
        "last_read",
        "erase_attempt_throw",
    ],
)
def test_native_composition_retains_paid_prefix_and_refuses_corruption(
    tmp_path, monkeypatch, fault
):
    calls, holders = fake_workflow(monkeypatch, fault)
    reader = native._read_capture
    corrupted = False

    def read(path, maximum):
        nonlocal corrupted
        if (
            fault in ("file", "holder", "late_io")
            and path.name.endswith("before_probe.json")
            and not corrupted
        ):
            corrupted = True
            if fault == "file":
                path.write_text("{}")
            elif fault == "holder":
                holders[0].report = deepcopy(holders[0].report)
            else:
                raise OSError("late readback failed")
        raw = reader(path, maximum)
        if fault == "last_read" and holders and path == tmp_path / "result.json" and not corrupted:
            corrupted = True
            holders[0].report["sparse_return"] = 1.0
        return raw

    monkeypatch.setattr(native, "_read_capture", read)
    output = tmp_path / "result.json"
    result = native.run_native_training_probe({}, reserved_episode("probe_train"), output)
    if fault is None:
        assert result["complete"], result
        assert calls == ["before_probe", "after_probe"]
        assert (
            len(result["native"]) == 2
            and result["outcomes"]["after_probe"]["final_success"] is False
        )
        assert json.loads(output.read_text()) == result
    else:
        assert result["complete"] is False
        assert calls == ["before_probe"]
        if fault in ("between_file", "between_holder"):
            assert (result.get("workflow") or {}).get("paid_append", 0) == 0
        if fault not in ("throw", "erase_attempt_throw"):
            assert result["native"][0]["report"]["source_training_rows"] == 7
        else:
            assert result["unreturned_native_work_unknown"] is True
            assert result["attempted"] == ["before_probe"] and result["returned"] == []
        saved = output.with_suffix(".failure.json") if "capture_error" in result else output
        assert json.loads(saved.read_text()) == result


@pytest.mark.parametrize("seed", [23, 24, 29])
def test_native_scene_never_acquires_reserved_rights(seed):
    with pytest.raises(ValueError):
        native.NativeScene("pick", seed)


@pytest.mark.parametrize("both", [False, True])
def test_cpu_cli_preserves_paid_report_after_late_capture_failure(
    tmp_path, monkeypatch, capsys, both
):
    output = tmp_path / "result.json"
    training = reserved_episode("probe_train")
    monkeypatch.setattr(
        inputs, "load_reserved_training_input", lambda path: (training, {"role": "probe_train"})
    )
    monkeypatch.setattr(cli, "_load_default_bundle", lambda: {})
    monkeypatch.setattr(
        cli,
        "run_demo",
        lambda payload, **options: dict(
            complete=True, probe=dict(progress=dict(updates_completed=1), paid_rows=18)
        ),
    )
    original_open = Path.open

    def open_path(path, *args, **options):
        if path == output or both and path == output.with_suffix(".failure.json"):
            raise OSError("late capture failure")
        return original_open(path, *args, **options)

    monkeypatch.setattr(Path, "open", open_path)
    assert cli.main(["--training-probe", "original23.json", "--output", str(output)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["complete"] is False and report["capture_error"]
    assert (
        report["probe"]["progress"]["updates_completed"] == 1 and report["probe"]["paid_rows"] == 18
    )
    if not both:
        with original_open(output.with_suffix(".failure.json")) as handle:
            assert json.load(handle) == report
    else:
        assert report["failure_capture_error"]


def test_probe_output_refuses_symbolic_link_even_when_target_absent(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == output or original(path))
    with pytest.raises(ValueError):
        demo.training_probe_outputs(output)
