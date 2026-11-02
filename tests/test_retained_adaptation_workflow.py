"""NEW literal first-party CPU adaptation integration; no data/SDK/model RNG."""

from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import run_taskspace_memory_demo as cli
from scripts import push_cube_reserved_probe_inputs as inputs
from src.adapters import taskspace_memory_demo as demo
from src.adapters import taskspace_memory_adaptation as adapt
from src.adapters import numpy_retained_nearest_bc as retained
from src.adapters import continuous_nearest_probe as near
from src.adapters import continuous_stream_sequence as sequence
from src.adapters import continuous_probe as probe
from src.core.replay_retention import ReplayRetentionPolicy
from test_taskspace_memory_demo import bundle
from test_push_cube_reserved_probe_inputs import reserved_episode
from test_retained_nearest_memory import _process_peak_bytes

WORK: dict[str, int] = {}
PRIOR: dict[str, int] = {}
CAPS = dict(
    initializers=160,
    updates=2048,
    incoming_rows=4096,
    forks=4096,
    predictions=4096,
    query_rows=8192,
    candidates=524288,
    coordinates=6291456,
    snapshots=262144,
    validators=1048576,
    copy_bytes=1024**3,
    hash_bytes=2 * 1024**3,
    residual_rows=8192,
    endpoints=256,
)
FIFO = ReplayRetentionPolicy("recent_fifo")


def charge(key, count=1):
    WORK[key] = WORK.get(key, 0) + count
    assert WORK[key] + PRIOR.get(key, 0) <= CAPS[key], key


@pytest.fixture(scope="module", autouse=True)
def bounded_literal_work():
    PRIOR.update(json.loads(os.environ.get("RETAINED_ADAPT_PRIOR", "{}")))

    def wrap(method, kind):
        def call(owner, *args, **kwargs):
            charge(kind)
            if kind == "updates":
                charge("incoming_rows", len(args[0]))
            if kind == "predictions":
                rows = len(args[0])
                candidates = rows * len(owner._state.prototype_features)
                for key, count in (
                    ("query_rows", rows),
                    ("candidates", candidates),
                    ("coordinates", candidates * 12),
                ):
                    charge(key, count)
            return method(owner, *args, **kwargs)

        return call

    validate, ids = retained.retained_snapshot, retained._ids
    residual = probe.diagnostics.continuous_bc_residuals

    def validated(state, policy, rule):
        charge("validators")
        charge("copy_bytes", state.prototype_features.nbytes + state.prototype_actions.nbytes)
        return validate(state, policy, rule)

    def hashed(features, actions):
        charge("hash_bytes", features.nbytes + actions.nbytes + 24 * len(features))
        return ids(features, actions)

    def residuals(*args, **kwargs):
        charge("residual_rows", len(args[0]))
        return residual(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("literal workflow attempted model RNG")

    with pytest.MonkeyPatch.context() as patch:
        for name, kind in (
            ("__init__", "initializers"),
            ("train_batch", "updates"),
            ("fork", "forks"),
            ("predict_with_work", "predictions"),
            ("snapshot_state", "snapshots"),
        ):
            patch.setattr(
                retained.NumpyRetainedNearestBCLearner,
                name,
                wrap(getattr(retained.NumpyRetainedNearestBCLearner, name), kind),
            )
        for module in (retained, near, sequence):
            patch.setattr(module, "retained_snapshot", validated)
        patch.setattr(retained, "_ids", hashed)
        patch.setattr(probe.diagnostics, "continuous_bc_residuals", residuals)
        patch.setattr(np.random, "default_rng", forbidden)
        yield
    peak = _process_peak_bytes()
    receipt = os.environ.get("RETAINED_ADAPT_RECEIPT")
    if receipt:
        with Path(receipt).open("x") as handle:
            json.dump(
                dict(
                    work=WORK,
                    aggregate={k: WORK.get(k, 0) + PRIOR.get(k, 0) for k in CAPS},
                    peak_bytes=peak,
                ),
                handle,
                indent=2,
            )
    assert peak <= 512 * 1024**2


def plan():
    return probe.ContinuousProbePlan(
        reserved_episode("probe_train"), reserved_episode("probe_test")
    )


def run(**options):
    return demo.run_demo(
        bundle(), adaptation_probe=plan(), retention=FIFO, max_prototypes=2, **options
    )


@pytest.mark.parametrize("return_task", [None, "pick17"])
def test_first_party_composition_keeps_source_and_separate_paid_private_phases(return_task):
    phases = []

    def endpoint(child, phase, guard):
        charge("endpoints")
        phases.append(phase)
        guard()

    result = run(return_task=return_task, endpoint_observer=endpoint)
    assert result["complete"], result.get("error")
    assert result["native_outcomes"] is None and result["return_task"] == return_task
    assert result["stream"]["model_state"]["candidate_training_examples"] == 7
    assert result["stream"]["experiment_state"] == dict(completed_updates=6, next_tick=18)
    a = result["adaptation"]
    leaves: list[dict] = [a] if return_task is None else [a["probe"], a["return_phase"]]
    assert len(phases) == (2 if return_task is None else 3)
    assert all(
        x["complete"]
        and x["progress"] == dict(updates_completed=1, replay_examples_completed=0, next_tick=2)
        for x in leaves
    )
    assert leaves[0]["source_training_examples"] == 7
    assert all(
        len(x["checkpoints"]) == 2 and len(x["checkpoints"][0]["scores"]) == 6 for x in leaves
    )
    if return_task:
        assert leaves[1]["source_training_examples"] == 25
        assert leaves[1]["plan"]["training_role"] == "train"
        assert leaves[1]["plan"]["training_id"] == "training-a"
        assert leaves[0]["checkpoints"][-1]["scores"] == leaves[1]["checkpoints"][0]["scores"]


@pytest.mark.parametrize("task", list(adapt.RETURN_GROUPS))
def test_return_selection_is_original_group_metadata_not_a_probe_relabel(monkeypatch, task):
    episodes, _, _ = demo._admit(bundle())

    def leaf(source, recipe, **options):
        assert options["return_training"] == episodes[adapt.RETURN_GROUPS[task]]
        assert options["return_training"].role == "train"
        assert len(options["training_readouts"]) == 4
        return {"complete": True}

    monkeypatch.setattr(adapt, "run_retained_nearest_probe", leaf)
    # Test selection/dispatch with literal source metadata, not another model run.
    source = object()
    pin = SimpleNamespace(
        state=SimpleNamespace(completed_updates=7, training_examples=7, retained_rows=1)
    )
    monkeypatch.setattr(adapt, "pin_retained_native", lambda owner: pin)
    monkeypatch.setattr(adapt, "require_retained_native", lambda *args: None)

    def stream(first, rest, observer, **options):
        observer(
            source,
            dict(
                phase="after_release",
                packet_index=5,
                known_episode_ids=[e.episode_id for e in (first, *rest)],
                prototype_rows=1,
            ),
        )
        return {"complete": True}

    monkeypatch.setattr(adapt, "run_sequence", stream)
    result = run(return_task=task)
    assert result["complete"]  # Metadata dispatch; no numerical/source-isolation claim.


@pytest.mark.parametrize(
    "options",
    [
        dict(return_task="unknown"),
        dict(return_task=True),
        dict(endpoint_observer=1),
        dict(check_boundary=1),
    ],
)
def test_invalid_metadata_refuses_before_bundle_or_model(monkeypatch, options):
    monkeypatch.setattr(adapt, "_plan", lambda *a: pytest.fail("admission reached input work"))
    with pytest.raises(ValueError):
        run(**options)


@pytest.mark.parametrize("fault", ["updates", "diagnostics", "context", "identity", "history"])
def test_nonoriginal_or_nonplain_pair_refuses_before_bundle(monkeypatch, fault):
    p = plan()
    if fault == "updates":
        p = replace(p, updates=2)
    elif fault == "diagnostics":
        p = replace(p, diagnostics=True)
    elif fault == "context":
        p = probe.ContinuousProbePlan(
            replace(p.training, context="other"), replace(p.evaluation, context="other")
        )
    elif fault == "identity":
        p = replace(p, evaluation=replace(p.evaluation, episode_id="other"))
    else:
        row = replace(
            p.evaluation.transitions[0],
            observation=p.evaluation.transitions[0].observation[:7] + (1.0, 0.0, 0.0, 0.0, 0.0),
        )
        p = replace(
            p, evaluation=replace(p.evaluation, transitions=(row,) + p.evaluation.transitions[1:])
        )
    monkeypatch.setattr(demo, "_admit", lambda *a: pytest.fail("invalid pair reached bundle"))
    with pytest.raises(ValueError):
        demo.run_demo({}, adaptation_probe=p, retention=FIFO, max_prototypes=2)


def test_paid_return_is_restored_if_later_boundary_corrupts_report(monkeypatch):
    original = adapt.paid_report_guard

    def capture(report):
        guard, trusted = original(report)
        report["adaptation"]["complete"] = False
        return guard, trusted

    monkeypatch.setattr(adapt, "paid_report_guard", capture)
    result = run(return_task="pick17")
    assert not result["complete"]
    assert result["adaptation"]["complete"] is True
    assert result["adaptation"]["work"]["training_attempts"] == 2


def test_mutated_source_is_refused_and_paid_cost_is_not_hidden():
    def endpoint(child, phase, guard):
        charge("endpoints")
        if phase == "after_probe":
            child._state.prototype_actions[0, 0] += 0.25

    result = run(endpoint_observer=endpoint)
    assert not result["complete"]
    assert result["adaptation"]["work"]["training_attempts"] == 1
    assert not result["unreturned_private_work_unknown"]


def test_unreturned_private_call_stays_unknown_and_source_preparation_stays_paid(monkeypatch):
    def lost(*args, **kwargs):
        raise RuntimeError("private call did not return its report")

    monkeypatch.setattr(adapt, "run_retained_nearest_probe", lost)
    result = run()
    assert not result["complete"] and result["adaptation"] is None
    assert result["unreturned_private_work_unknown"]
    assert result["stream"]["experiment_state"]["completed_updates"] == 6


def command(output):
    return [
        "--adaptation-probe",
        "train.json",
        "test.json",
        "--retention",
        "recent_fifo",
        "--max-prototypes",
        "2",
        "--output",
        str(output),
    ]


def test_cli_return_without_adaptation_refuses_before_input(monkeypatch):
    monkeypatch.setattr(cli, "_load_default_bundle", lambda: pytest.fail("input opened"))
    with pytest.raises(SystemExit):
        cli.main(["--return-task", "pick17"])


@pytest.mark.parametrize("extra", [["--development-probe"], ["--training-probe", "x.json"]])
def test_cli_incompatible_mode_refuses_before_readers(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(cli, "_load_default_bundle", lambda: pytest.fail("input opened"))
    monkeypatch.setattr(inputs, "load_reserved_probe_inputs", lambda *a: pytest.fail("pair opened"))
    with pytest.raises(SystemExit):
        cli.main(command(tmp_path / "out.json") + extra)


@pytest.mark.parametrize("missing", ["--retention", "--max-prototypes", "--output"])
def test_cli_requires_explicit_retention_capacity_and_paid_output(tmp_path, missing):
    args = command(tmp_path / "out.json")
    i = args.index(missing)
    del args[i : i + 2]
    with pytest.raises(SystemExit):
        cli.main(args)


@pytest.mark.parametrize("suffix", [".json", ".failure.json"])
def test_cli_refuses_output_collision_before_loading_split(tmp_path, monkeypatch, suffix):
    output = tmp_path / "out.json"
    output.with_suffix(suffix).write_text("preserve")
    monkeypatch.setattr(inputs, "load_reserved_probe_inputs", lambda *a: pytest.fail("pair opened"))
    with pytest.raises(ValueError):
        cli.main(command(output))


def test_cli_cpu_uses_owned_pair_returns_scores_and_preserves_late_paid_capture(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "out.json"
    p = plan()
    calls: list[Path] = []

    def load(a, b):
        calls.extend((a, b))
        return replace(p, diagnostics=True), [{"role": "probe_train"}, {"role": "probe_test"}]

    monkeypatch.setattr(inputs, "load_reserved_probe_inputs", load)
    monkeypatch.setattr(cli, "_load_default_bundle", bundle)
    original = Path.open

    def opened(path, *args, **kwargs):
        if path == output and args and args[0] == "x":
            raise OSError("late denied main capture")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    assert cli.main(command(output) + ["--return-task", "pick17"]) == 1
    saved = json.loads(output.with_suffix(".failure.json").read_text())
    printed = json.loads(capsys.readouterr().out)
    assert saved == printed and not saved["complete"] and saved["adaptation"]["complete"]
    assert saved["adaptation"]["work"]["training_rows_attempted"] == 19
    assert calls == [Path("train.json"), Path("test.json")]
    assert len(saved["adaptation_input_receipts"]) == 2


@pytest.mark.parametrize("paths", [(None, None), (Path("a.json"), Path("b.json"))])
def test_pair_reader_explicit_paths_preserve_roles_and_default_profile(monkeypatch, paths):
    calls = []

    def load(role, path=None):
        calls.append((role, path))
        return reserved_episode(role), {"role": role}

    monkeypatch.setattr(inputs, "_load_role", load)
    p, receipts = inputs.load_reserved_probe_inputs(*paths)
    assert calls == [("probe_train", paths[0]), ("probe_test", paths[1])]
    assert p.diagnostics is True and len(receipts) == 2


def test_late_io_cannot_rewrite_paid_adaptation_before_failure_capture(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "out.json"
    paid: dict[str, Any] = dict(
        complete=True, adaptation=dict(complete=True, work=dict(training_attempts=2))
    )
    monkeypatch.setattr(cli, "run_demo", lambda *a, **k: paid)
    monkeypatch.setattr(cli, "_load_default_bundle", lambda: {})
    monkeypatch.setattr(inputs, "load_reserved_probe_inputs", lambda *a: (plan(), []))
    original = Path.open

    def changed(path, *args, **kwargs):
        if path == output and args and args[0] == "x":
            paid["adaptation"]["work"]["training_attempts"] = 0
            raise OSError("late capture changed the live holder")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", changed)
    assert cli.main(command(output)) == 1
    saved = json.loads(output.with_suffix(".failure.json").read_text())
    assert saved == json.loads(capsys.readouterr().out)
    assert not saved["complete"] and saved["adaptation"]["complete"]
    assert saved["adaptation"]["work"]["training_attempts"] == 2


def test_old_default_route_and_incompatibilities_use_original_dispatch(monkeypatch):
    sentinel = {"complete": True}
    monkeypatch.setattr(demo, "_run_training_probe_demo", lambda *a: sentinel)
    assert demo.run_demo({}, training_probe=reserved_episode("probe_train")) is sentinel
    with pytest.raises(ValueError):
        demo.run_demo({}, return_task="pick17")
    with pytest.raises(ValueError):
        demo.run_demo({}, training_probe=reserved_episode("probe_train"), retention=FIFO)
    with pytest.raises(ValueError):
        demo.run_demo({}, adaptation_probe=plan())
