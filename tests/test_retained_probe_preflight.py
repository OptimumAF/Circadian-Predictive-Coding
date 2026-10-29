"""No-model regressions for the actual source contract and child refusal."""

from contextlib import contextmanager
from dataclasses import asdict, replace
import subprocess
import sys

import pytest

from scripts import retained_probe_preflight as gate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget

WORK = dict(metadata_owners=0, plan_preparations=0, intercepted_models=0, child_attempts=0)


def owner():
    WORK["metadata_owners"] += 1
    return ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=6, max_wall_seconds=90, max_process_rss_bytes=512 * 1024**2
        ),
        lambda: 12.0,
    )


@pytest.mark.parametrize("updates", [0, 4, 6])
def test_should_accept_original_none_limit_and_zero_paid_replay_without_mutation(updates):
    session = owner()
    session.updates_completed = updates
    identity = tuple(
        id(x)
        for x in (session.budget, session.clock, session.progress, session.process_rss_sampler)
    )
    before = dict(vars(session))
    gate.require_source_budget(session)
    assert dict(vars(session)) == before
    assert (
        tuple(
            id(x)
            for x in (session.budget, session.clock, session.progress, session.process_rss_sampler)
        )
        == identity
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(max_replay_examples=0),
        dict(max_training_updates=7),
        dict(max_wall_seconds=91),
        dict(max_process_rss_bytes=1024),
        dict(max_hidden_width=8),
    ],
)
def test_should_refuse_changed_original_limits(change):
    session = owner()
    session.budget = replace(session.budget, **change)
    with pytest.raises(ValueError, match="limits"):
        gate.require_source_budget(session)


@pytest.mark.parametrize(
    "field,value",
    [
        ("replay_examples_completed", 1),
        ("replay_examples_completed", False),
        ("replay_examples_completed", 0.0),
        ("updates_completed", 7),
        ("updates_completed", True),
        ("updates_completed", None),
    ],
)
def test_should_refuse_unpaid_or_noninteger_work(field, value):
    session = owner()
    setattr(session, field, value)
    with pytest.raises(ValueError, match="completed"):
        gate.require_source_budget(session)


def test_should_accept_actual_source_setup_before_intercepted_model_body(monkeypatch):
    from scripts import run_minigrid_linear_fit_control as cli
    from src.adapters import minigrid_context_linear_learner as native
    from src.core.reference_controller import ReferencePolicy
    from tests.test_a4_adaptation import sources

    seen = []
    original = ToyBudgetSession.attach_memory

    def attach(session, memory):
        WORK["metadata_owners"] += 1
        value = original(session, memory)
        gate.require_source_budget(session)
        seen.append(asdict(session.budget))
        return value

    class BeforeModel(Exception):
        pass

    def no_model():
        WORK["intercepted_models"] += 1
        raise BeforeModel("constructor intercepted before original body")

    monkeypatch.setattr(ToyBudgetSession, "attach_memory", attach)
    monkeypatch.setattr(native, "ContextLinearLearner", no_model)
    WORK["plan_preparations"] += 1
    with pytest.raises(BeforeModel):
        cli.run_context_linear_stream(
            sources(),
            a4_training=True,
            retain_arrived=True,
            reference_policy=ReferencePolicy("drift"),
            ready_feedback=True,
        )
    assert len(seen) == 1 and seen[0]["max_replay_examples"] is None


def resolver(monkeypatch, path):
    @contextmanager
    def resolve():
        yield path

    monkeypatch.setattr(gate, "example_affine_training", resolve)


def child(marker):
    return [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('launched')",
        str(marker),
    ]


@pytest.mark.parametrize("kind", ["missing", "wrong_size", "directory"])
def test_should_prevent_actual_child_launch_when_asset_metadata_fails(monkeypatch, tmp_path, kind):
    asset, marker = tmp_path / "episodes.json", tmp_path / "child-ran"
    if kind == "wrong_size":
        asset.write_text("wrong")
    elif kind == "directory":
        asset.mkdir()
    resolver(monkeypatch, asset)
    with pytest.raises(ValueError, match="TRAIN asset"):
        gate.run_checked(child(marker), timeout_seconds=10)
    assert not marker.exists()


def test_should_launch_harmless_child_only_after_successful_check(monkeypatch, tmp_path):
    asset, marker = tmp_path / "episodes.json", tmp_path / "child-ran"
    asset.write_bytes(b"0" * 94705)
    resolver(monkeypatch, asset)
    WORK["child_attempts"] += 1
    result = gate.run_checked(child(marker), timeout_seconds=10)
    assert result.returncode == 0 and marker.read_text() == "launched"


def test_should_propagate_failed_child_without_reporting_success(monkeypatch, tmp_path):
    asset = tmp_path / "episodes.json"
    asset.write_bytes(b"0" * 94705)
    resolver(monkeypatch, asset)
    WORK["child_attempts"] += 1
    with pytest.raises(subprocess.CalledProcessError) as error:
        gate.run_checked([sys.executable, "-c", "raise SystemExit(7)"], timeout_seconds=10)
    assert error.value.returncode == 7


def test_should_check_real_package_alias_without_decoding_or_launching(monkeypatch, capsys):
    monkeypatch.setattr(gate.subprocess, "run", lambda *a, **k: pytest.fail("child launched"))
    assert gate.main([]) == 0
    assert '"child_launched": false' in capsys.readouterr().out


def test_should_fail_cli_before_child_on_bad_asset(monkeypatch, tmp_path, capsys):
    resolver(monkeypatch, tmp_path / "missing.json")
    marker = tmp_path / "child-ran"
    assert gate.main(["--", *child(marker)]) == 1
    assert not marker.exists() and "failed" in capsys.readouterr().err


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_should_refuse_invalid_timeout_before_child(tmp_path, timeout):
    marker = tmp_path / "child-ran"
    with pytest.raises(ValueError, match="timeout"):
        gate.run_checked(child(marker), timeout_seconds=timeout)
    assert not marker.exists()


@pytest.mark.parametrize("command", [[], "python", [""], [None]])
def test_should_refuse_invalid_argv(command):
    with pytest.raises(ValueError, match="argv"):
        gate.run_checked(command)
