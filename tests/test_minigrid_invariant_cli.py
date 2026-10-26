"""CLI source binding and publication checks without learner/physics execution."""

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch
from typing import Any

import pytest

import scripts.run_minigrid_invariant_pilot as cli
import scripts.run_minigrid_memory_probe as loader

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "artifacts/runs/minigrid-action-memory-live-20261017/native.json"
)


def forbidden(*args, **kwargs):
    pytest.fail("preview or refused admission reached execution")


def test_should_import_cli_without_importing_any_repository_runtime():
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("src", "scripts.run_minigrid_memory_probe")):
            pytest.fail(f"CLI import reached runtime {name}")
        return original(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("invariant_cli_import_only", cli.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch("builtins.__import__", guarded):
        spec.loader.exec_module(module)
    assert callable(module.main)


@pytest.mark.parametrize("arguments", [[], ["--describe"]])
def test_should_preview_without_loading_running_or_changing_thread_environment(
    monkeypatch, capsys, arguments
):
    monkeypatch.setenv("OMP_NUM_THREADS", "preview-preserved")
    monkeypatch.setattr(cli, "describe_protocol", lambda: {"protocol_id": "frozen-preview"})
    monkeypatch.setattr(cli, "load_inputs", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    assert cli.main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == {"protocol_id": "frozen-preview"}
    import os

    assert os.environ["OMP_NUM_THREADS"] == "preview-preserved"


@pytest.mark.parametrize(
    "arguments",
    [["--run"], ["--source", "x"], ["--output", "x"], ["--describe", "--run"]],
)
def test_should_refuse_ambiguous_or_missing_run_arguments_before_work(monkeypatch, arguments):
    monkeypatch.setattr(cli, "load_inputs", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)
    assert raised.value.code == 2


def test_should_read_once_validate_full_original_report_and_select_native0_and32(monkeypatch):
    document = json.loads(SOURCE.read_text(encoding="utf-8"))
    original = deepcopy(document)
    calls = []

    def read(path):
        calls.append(path)
        return document

    monkeypatch.setattr(loader, "_read_source", read)
    states, episodes = cli.load_inputs(SOURCE)
    assert calls == [SOURCE]
    assert [(s.completed_updates, s.training_examples) for s in states] == [(0, 0), (32, 288)]
    assert json.loads(json.dumps(asdict(states[1]))) == document["observations"][1]["snapshot"]
    assert [(e.episode_id, e.context, e.role, len(e.transitions)) for e in episodes] == [
        ("minigrid-source-A-blue", "A", "train", 9),
        ("minigrid-source-A-red", "A", "train", 9),
    ]
    object.__setattr__(states[1].policy, "max_entries", 1)
    object.__setattr__(episodes[0], "context", "changed")
    assert document == original


@pytest.mark.parametrize("fault", ["late_snapshot", "role", "A_record", "prefix_clock"])
def test_should_reuse_original_full_report_and_A_binding_refusals(monkeypatch, fault):
    document = json.loads(SOURCE.read_text(encoding="utf-8"))
    if fault == "late_snapshot":
        document["observations"][-1]["snapshot"]["training_examples"] = 1151
    elif fault == "role":
        document["training_sources"][2]["role"] = "final_test"
    elif fault == "A_record":
        transition = document["source_reports"][0]["transitions"][0]
        transition["action"] = (transition["action"] + 1) % 7
    else:
        document["observations"][1]["updates"] = True
    monkeypatch.setattr(loader, "_read_source", lambda path: document)
    with pytest.raises(ValueError):
        cli.load_inputs(SOURCE)


@pytest.mark.parametrize("status,expected", [("complete", 0), ("incomplete", 2), ("failed", 2)])
def test_should_publish_original_negative_or_complete_result_exclusively(
    tmp_path, monkeypatch, capsys, status, expected
):
    output = tmp_path / "new.json"
    states: Any = (object(), object())
    sources: Any = (object(), object())
    monkeypatch.setattr(cli, "load_inputs", lambda path: (states, sources))
    calls = []

    def run(actual_states, actual_sources):
        calls.append((actual_states, actual_sources))
        return {"status": status, "reason": "negative retained", "success": False}

    monkeypatch.setattr(cli, "run_pilot", run)
    ticks = iter((2.0, 2.25))
    monkeypatch.setattr(cli, "monotonic", lambda: next(ticks))
    assert cli.main(["--run", "--source", "prior.json", "--output", str(output)]) == expected
    report = json.loads(output.read_text(encoding="utf-8"))
    assert calls == [(states, sources)]
    assert report == dict(
        status=status,
        reason="negative retained",
        success=False,
        source_artifact="prior.json",
        source_load_seconds=0.25,
    )
    assert json.loads(capsys.readouterr().out) == report


@pytest.mark.parametrize("kind", ["existing", "source_collision", "missing_parent"])
def test_should_preflight_output_before_source_work(tmp_path, monkeypatch, kind):
    source = tmp_path / "source.json"
    source.write_bytes(b"source unchanged")
    if kind == "existing":
        output = tmp_path / "existing.json"
        output.write_bytes(b"user changes")
    elif kind == "source_collision":
        output = source
    else:
        output = tmp_path / "absent" / "out.json"
    monkeypatch.setattr(cli, "load_inputs", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    with pytest.raises(SystemExit) as raised:
        cli.main(["--run", "--source", str(source), "--output", str(output)])
    assert raised.value.code == 2
    assert source.read_bytes() == b"source unchanged"
    if kind == "existing":
        assert output.read_bytes() == b"user changes"


def test_should_keep_racing_output_and_report_publication_failure(tmp_path, monkeypatch, capsys):
    output = tmp_path / "raced.json"
    monkeypatch.setattr(cli, "load_inputs", lambda path: ((), ()))

    def run(states, sources):
        output.write_bytes(b"racing user data")
        return {"status": "complete"}

    monkeypatch.setattr(cli, "run_pilot", run)
    assert cli.main(["--run", "--source", "prior", "--output", str(output)]) == 2
    captured = capsys.readouterr()
    assert not captured.out and "FileExistsError" in captured.err
    assert output.read_bytes() == b"racing user data"


@pytest.mark.parametrize("fault", ["loader", "runner", "oversized", "nonfinite"])
def test_should_return_two_without_invented_report_on_operational_failure(
    tmp_path, monkeypatch, capsys, fault
):
    output = tmp_path / "unpublished.json"
    monkeypatch.setattr(cli, "load_inputs", lambda path: ((), ()))
    monkeypatch.setattr(cli, "run_pilot", lambda states, sources: {"status": "complete"})

    def fail(*args):
        raise ValueError("original evidence refused")

    if fault == "loader":
        monkeypatch.setattr(cli, "load_inputs", fail)
        monkeypatch.setattr(cli, "run_pilot", forbidden)
    elif fault == "runner":
        monkeypatch.setattr(cli, "run_pilot", fail)
    elif fault == "oversized":
        monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 4)
    else:
        monkeypatch.setattr(
            cli, "run_pilot", lambda *args: {"status": "complete", "x": float("nan")}
        )
    assert cli.main(["--run", "--source", "prior", "--output", str(output)]) == 2
    captured = capsys.readouterr()
    assert not captured.out and "invariant pilot failed" in captured.err
    assert not output.exists()


def test_should_lazy_dispatch_the_recipe_without_reentry(monkeypatch):
    from types import SimpleNamespace

    calls = []

    def run(states, sources):
        calls.append((states, sources))
        return {"status": "complete"}

    module = SimpleNamespace(
        describe_protocol=lambda: {"protocol_id": "lazy"},
        run_pilot=run,
    )
    monkeypatch.setitem(sys.modules, "src.adapters.minigrid_invariant_pilot", module)
    assert cli.describe_protocol() == {"protocol_id": "lazy"}
    states: Any = (object(), object())
    sources: Any = (object(), object())
    assert cli.run_pilot(states, sources) == {"status": "complete"}
    assert calls == [(states, sources)]
