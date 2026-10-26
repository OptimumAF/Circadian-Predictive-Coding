"""Portable data and actual stream handoff; scripted games are not task evidence."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import pickle
import sys
import tomllib
from zipfile import Path as ZipPath, ZipFile

import numpy as np
import pytest

from scripts import minigrid_affine_workflow as workflow
from scripts import run_minigrid_linear_fit_control as cli
from src.infra import minigrid_affine_example as resource
from test_after_c_linear_games import state

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/minigrid_affine_train/episodes.json"


@pytest.fixture(autouse=True)
def scripted_runtime(monkeypatch):
    # These existing stream/ownership fixtures already script all physical games.
    # The real optional-runtime prerequisite has its own fail-fast checks.
    monkeypatch.setattr(workflow, "_require_minigrid", lambda: None)


def complete_training():
    return dict(
        status="complete",
        source_unchanged=True,
        actor_unchanged=True,
        receipt_capture_error=None,
        committed_receipts=[{} for _ in range(6)],
        budget=dict(updates_completed=6),
    )


def fake_stream(monkeypatch, training=None, endpoint=None):
    training = complete_training() if training is None else training
    endpoint = state() if endpoint is None else endpoint

    def run(episodes, **options):
        assert len(episodes) == 6
        assert options["retain_arrived"] is options["abc_catalogue"] is True
        options["endpoint_observer"](endpoint)
        return training

    monkeypatch.setattr(cli, "run_context_linear_stream", run)
    return training


def complete_games():
    return dict(
        status="complete",
        state_unchanged=True,
        episodes=[dict(status="complete", environment=dict(closed=True)) for _ in range(4)],
    )


def test_should_bundle_only_complete_original_train_rows_and_package_them():
    raw = json.loads(DATA.read_bytes())
    assert set(raw) == {"training_sources"}
    assert [e["episode_id"] for e in raw["training_sources"]] == [
        f"minigrid-source-{context}-{cue}"
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    ]
    assert all(e["role"] == "train" and e["num_classes"] == 7 for e in raw["training_sources"])
    assert all(len(e["transitions"]) == 9 for e in raw["training_sources"])
    assert all(
        len(t["observation"]) == len(t["next_observation"]) == 148
        for e in raw["training_sources"]
        for t in e["transitions"]
    )
    episodes = cli.read_training_sources(DATA)
    assert all(len(e.transitions[0].observation) == 296 for e in episodes)
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf8"))
    assert (
        config["project"]["scripts"]["circadian-affine-game"]
        == "scripts.run_minigrid_linear_fit_control:main"
    )
    assert "src.infra.minigrid_affine_data" in config["tool"]["setuptools"]["packages"]


def test_should_decode_extracted_zip_resource_and_release_it(tmp_path, monkeypatch):
    archive_path = tmp_path / "data.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("episodes.json", DATA.read_bytes())
    monkeypatch.setattr(resource, "__file__", str(tmp_path / "repo/src/infra/example.py"))
    with ZipFile(archive_path) as archive:

        def files(package):
            assert package == "src.infra.minigrid_affine_data"
            return ZipPath(archive)

        monkeypatch.setattr(resource.resources, "files", files)
        with resource.example_affine_training() as path:
            assert len(cli.read_training_sources(path)) == 6
            extracted = path
        assert not extracted.exists()


def test_should_refuse_oversized_local_resource_without_fallback(tmp_path, monkeypatch):
    folder = tmp_path / "examples/minigrid_affine_train"
    folder.mkdir(parents=True)
    (folder / "episodes.json").write_bytes(b"x" * (512 * 1024 + 1))
    monkeypatch.setattr(resource, "__file__", str(tmp_path / "src/infra/example.py"))
    monkeypatch.setattr(resource.resources, "files", lambda *a: pytest.fail("fallback"))
    with pytest.raises(ValueError, match="512KiB"):
        with resource.example_affine_training():
            pytest.fail("oversized resource admitted")


def test_should_train_six_real_native_updates_and_handoff_actual_endpoint(monkeypatch):
    before = DATA.read_bytes()
    rng = pickle.dumps(np.random.get_state())
    seen = []

    def games(source, endpoint, **options):
        assert source.snapshot_state().completed_updates == 0
        assert (endpoint.completed_updates, endpoint.training_examples) == (6, 108)
        assert all(head.fit_rows == 27 for head in endpoint.heads)
        assert options["after_c_capacity"] is True
        assert options["budget"].updates_completed == 0
        options["check_boundary"]()
        seen.append(endpoint)
        return complete_games()

    monkeypatch.setattr(workflow, "evaluate_context_linear_stream_checkpoint", games)
    result = workflow.run_workflow()
    assert result["status"] == "complete", result.get("reason")
    assert len(seen) == 1
    training = result["training"]
    assert training["protocol"]["schedule"].startswith("A blue/red, B blue/red, C_train")
    assert training["catalogue"]["committed_rows"] == dict(arrivals=54, replay=54, native=108)
    for i, receipt in enumerate(training["committed_receipts"]):
        assert receipt["observed_at"] == i
        assert receipt["arrived_at"] == receipt["applied_at"] == i + 1
        assert receipt["update_number"] == i + 1
    assert [w["current_rows"] + w["replay_rows"] for w in training["catalogue"]["wakes"]] == [
        9,
        9,
        18,
        18,
        27,
        27,
    ]
    assert training["observed_checkpoints"] == [0, 2, 4, 6]
    assert training["endpoints"][-1]["state"] == __import__("dataclasses").asdict(seen[0])
    assert training["score_calls_completed"] == 24 and training["score_rows_completed"] == 216
    assert result["readout_budget"]["updates_completed"] == 0
    assert DATA.read_bytes() == before and pickle.dumps(np.random.get_state()) == rng


@pytest.mark.parametrize("fault", ["status", "source", "actor", "receipt", "charge", "endpoint"])
def test_should_refuse_incomplete_training_before_games(monkeypatch, fault):
    training = complete_training()
    endpoint = state()
    if fault == "status":
        training["status"] = "failed"
    elif fault == "source":
        training["source_unchanged"] = False
    elif fault == "actor":
        training["actor_unchanged"] = False
    elif fault == "receipt":
        training["committed_receipts"].pop()
    elif fault == "charge":
        training["budget"]["updates_completed"] = 5
    else:
        endpoint = replace(endpoint, training_examples=90)
    fake_stream(monkeypatch, training, endpoint)
    monkeypatch.setattr(
        workflow, "evaluate_context_linear_stream_checkpoint", lambda *a, **k: pytest.fail("game")
    )
    result = workflow.run_workflow()
    assert result["status"] == "failed" and result["games"] is None
    assert result["training"] is training


@pytest.mark.parametrize(
    "fault", ["training", "replacement", "source", "limits", "charge", "clock"]
)
def test_should_refuse_changed_preceding_evidence_or_readout_authority(monkeypatch, fault):
    training = fake_stream(monkeypatch)
    original = deepcopy(training)

    def games(source, endpoint, **options):
        budget = options["budget"]
        if fault == "training":
            training["status"] = "changed"
        elif fault == "replacement":
            training["committed_receipts"] = deepcopy(training["committed_receipts"])
        elif fault == "source":
            source.restore_state(endpoint)
        elif fault == "limits":
            object.__setattr__(budget.budget, "max_training_updates", 1)
        elif fault == "charge":
            budget.updates_completed = 1
        else:
            budget.started_at -= 1
        options["check_boundary"]()
        pytest.fail("changed evidence accepted")

    monkeypatch.setattr(workflow, "evaluate_context_linear_stream_checkpoint", games)
    result = workflow.run_workflow()
    assert result["status"] == "failed" and "changed" in result["reason"]
    assert result["training"] == original


def test_should_preserve_failed_game_prefix_without_retry(monkeypatch):
    fake_stream(monkeypatch)
    partial = dict(status="failed", reason="scripted failure", episodes=[dict(status="complete")])
    calls = []

    def games(*a, **k):
        calls.append(1)
        raise workflow.MiniGridEvaluationFailed(partial)

    monkeypatch.setattr(workflow, "evaluate_context_linear_stream_checkpoint", games)
    result = workflow.run_workflow()
    assert result["status"] == "failed" and result["games"] == partial and calls == [1]
    assert result["training"]["status"] == "complete"


@pytest.mark.parametrize(
    "args",
    [
        ["--describe"],
        ["--source", "missing"],
        ["--retained-source", "missing"],
        ["--fit-updates", "2"],
        ["--context-linear-probe"],
    ],
)
def test_should_describe_without_io_or_refuse_conflicting_cli_options(monkeypatch, capsys, args):
    monkeypatch.setattr(workflow, "run_workflow", lambda: pytest.fail("work"))
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("IO"))
    monkeypatch.setattr(sys, "argv", ["control", "--example-affine-play", *args])
    if args == ["--describe"]:
        cli.main()
        assert json.loads(capsys.readouterr().out)["training"]["fit_rows"] == 108
    else:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2


def test_should_keep_exclusive_output_and_use_normal_workflow_without_source(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["control", "--example-affine-play", "--output", str(output)])
    calls = []

    def run():
        calls.append(1)
        return dict(status="complete")

    monkeypatch.setattr(workflow, "run_workflow", run)
    cli.main()
    assert json.loads(output.read_bytes())["status"] == "complete" and calls == [1]
    with pytest.raises(FileExistsError):
        cli.main()
    assert calls == [1]


@pytest.mark.parametrize(
    "options", [dict(abc_catalogue=True), dict(abc_catalogue=1), dict(endpoint_observer=1)]
)
def test_should_refuse_ambiguous_stream_options_before_models(options):
    with pytest.raises(ValueError):
        cli.run_context_linear_stream((), **options)


def test_should_preserve_old_aba_and_catalogue_protocol_defaults():
    assert cli.describe_context_linear_stream()["fit_rows"] == 54
    assert cli.describe_context_linear_catalogue()["fit_rows"] == 90
    assert (
        "A blue, A red, B blue, B red, A blue, A red"
        in cli.describe_context_linear_stream()["schedule"]
    )
