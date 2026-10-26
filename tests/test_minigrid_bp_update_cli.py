"""Resource-free public BP diagnostic description and saved TRAIN admission."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import diagnose_minigrid_bp_cycle as cli


def forbidden(*args, **kwargs):
    raise AssertionError("description/admission constructed native resources")


def test_should_describe_fixed_diagnostic_without_loading_sources_or_torch(monkeypatch, capsys):
    monkeypatch.setattr(cli, "require_torch", forbidden)
    monkeypatch.setattr(cli, "_training_sources", forbidden)
    monkeypatch.setattr(sys, "argv", ["diagnose", "--describe"])
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert report["native_calls"] == 4 and report["native_training_rows"] == 36
    assert report["replay_rows"] == 18 and report["published_updates"] == 0
    assert report["final_data_used"] is report["probe_data_used"] is False


def test_should_read_only_saved_train_sources_as_complete_histories():
    path = (
        Path(__file__).resolve().parents[1] / "artifacts/runs/joint-catalogue-study-20261018/bp.txt"
    )
    sources = cli._training_sources(path)
    assert len(sources) == 6
    assert all(
        episode.role == "train" and len(episode.transitions) == 9 for episode in sources.values()
    )
    assert all(
        len(row.observation) == len(row.next_observation) == 296
        for episode in sources.values()
        for row in episode.transitions
    )


@pytest.mark.parametrize("argument", [None, "missing.json"])
def test_should_refuse_missing_sources_before_native_resources(monkeypatch, argument):
    monkeypatch.setattr(cli, "require_torch", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["diagnose"] + ([] if argument is None else ["--train-report", argument])
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("fault", ["diagnostic", "warmup"])
def test_should_preserve_paid_prefix_on_public_runtime_failure(monkeypatch, capsys, fault):
    # Fake learner makes the public accounting failure path resource-free.
    path = (
        Path(__file__).resolve().parents[1] / "artifacts/runs/joint-catalogue-study-20261018/bp.txt"
    )
    monkeypatch.setattr(
        cli,
        "require_torch",
        lambda: SimpleNamespace(set_num_threads=lambda _: None, get_num_interop_threads=lambda: 1),
    )

    class Fake:
        def __init__(self, *args):
            pass

        def train_batch(self, *args):
            if fault == "warmup":
                raise RuntimeError("paid warmup failure")

        def snapshot_state(self):
            return object()

    def fail(*args, **kwargs):
        raise cli.GradientDiagnosticFailed(
            dict(status="incomplete", reason="paid diagnostic failure", work={})
        )

    monkeypatch.setattr(cli, "InvariantTorchFeatureLearner", Fake)
    monkeypatch.setattr(cli, "diagnose_bp_updates", fail)
    monkeypatch.setattr(sys, "argv", ["diagnose", "--train-report", str(path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    report = json.loads(capsys.readouterr().out)
    prefix = report["warmup_work"]
    assert prefix["native_train_calls_attempted"] == prefix["charged_updates"] == 1
    assert prefix["native_training_rows_attempted"] == 9
    assert prefix["native_train_calls_returned"] == int(fault == "diagnostic")
    assert prefix["native_training_rows_returned"] == (9 if fault == "diagnostic" else 0)
    assert report["source_admission_work"]["validation_projection_rows"] == 108
