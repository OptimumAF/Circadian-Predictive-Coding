"""Thin saved-TRAIN CLI and preservation of paid results on late failure."""

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_linear_fit_control as control
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from test_minigrid_invariant_encoding import frame
from test_minigrid_linear_fit_control import records


def source_record():
    views = tuple(frame(cue_color=2, key_color=2, door_state=index) for index in range(3))
    return asdict(
        DemonstrationEpisode(
            "source-A-blue",
            "A",
            "train",
            tuple(
                DemonstrationTransition(i, views[i], action, 0.0, views[i + 1], i == 1, 7)
                for i, action in enumerate((0, 2))
            ),
            7,
        )
    )


def test_should_describe_without_reading_sources_or_fitting(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("description reached data/model access")

    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(cli, "diagnose_train_linear_fit", forbidden)
    monkeypatch.setattr(sys, "argv", ["control", "--describe"])
    cli.main()
    assert json.loads(capsys.readouterr().out) == control.describe_protocol()


def test_should_fit_only_saved_train_sources_and_create_a_new_json(tmp_path, monkeypatch, capsys):
    source, output = tmp_path / "source.json", tmp_path / "output.json"
    source.write_text(
        json.dumps(dict(training_sources=[source_record()], final_test="unavailable"))
    )
    before = source.read_bytes()
    monkeypatch.setattr(sys, "argv", ["control", "--source", str(source), "--output", str(output)])
    cli.main()
    result = json.loads(output.read_text())
    assert result == json.loads(capsys.readouterr().out)
    assert result["status"] == "complete" and len(result["prediction_probabilities"]) == 2
    assert result["work"]["fit_rows_completed"] == result["work"]["prediction_rows_completed"] == 2
    assert result["runtime"]["numpy"] and result["runtime"]["python"]
    assert "final_test" not in result and source.read_bytes() == before


@pytest.mark.parametrize(
    "fault", ["final_test", "probe_train", "probe_test", "missing", "empty", "extra", "raw296"]
)
def test_should_refuse_invalid_source_records_before_fit(tmp_path, monkeypatch, fault):
    record = source_record()
    payload = dict(training_sources=[record])
    if fault in ("final_test", "probe_train", "probe_test"):
        record["role"] = fault
    elif fault == "missing":
        payload = {}
    elif fault == "empty":
        payload["training_sources"] = []
    elif fault == "extra":
        record["unobserved"] = 1
    else:
        for transition in record["transitions"]:
            transition["observation"] *= 2
            transition["next_observation"] *= 2
    source = tmp_path / "source.json"
    source.write_text(json.dumps(payload))
    monkeypatch.setattr(
        cli, "diagnose_train_linear_fit", lambda *a: pytest.fail("invalid source fitted")
    )
    monkeypatch.setattr(sys, "argv", ["control", "--source", str(source)])
    with pytest.raises(ValueError):
        cli.main()


def test_should_refuse_large_input_before_reading_it(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text("{}")
    original = Path.stat

    def large(path, *args, **kwargs):
        return (
            SimpleNamespace(st_size=16 * 1024 * 1024 + 1)
            if path == source
            else original(path, *args, **kwargs)
        )

    monkeypatch.setattr(Path, "stat", large)
    with pytest.raises(ValueError, match="16MiB"):
        cli.read_training_sources(source)


def test_should_refuse_output_replacement_before_reading_or_fitting(tmp_path, monkeypatch):
    output = tmp_path / "output.json"
    output.write_text("original")
    monkeypatch.setattr(
        cli, "read_training_sources", lambda *a: pytest.fail("existing output admitted")
    )
    monkeypatch.setattr(
        sys, "argv", ["control", "--source", "unused.json", "--output", str(output)]
    )
    with pytest.raises(FileExistsError):
        cli.main()
    assert output.read_text() == "original"


@pytest.mark.parametrize("phase", ["fit", "prediction"])
def test_should_refuse_modified_detached_operation_inputs(monkeypatch, phase):
    name = "fit_categorical_linear" if phase == "fit" else "predict_categorical_linear"
    original = getattr(control, name)

    def mutate(*args):
        result = original(*args)
        args[0 if phase == "fit" else 1][0, 0] *= -1
        return result

    monkeypatch.setattr(control, name, mutate)
    result = control.diagnose_train_linear_fit(records())
    assert result["status"] == "failed" and result["source_unchanged"] is True
    assert "inputs changed" in result["reason"]
    assert result["work"][f"{'fit' if phase == 'fit' else 'prediction'}_calls_completed"] == 1


def test_should_keep_raw_predictions_and_completed_scores_when_later_reduction_fails(monkeypatch):
    original = control.probability_margins
    calls = []

    def fail_late(*args):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("injected second score refusal")
        return original(*args)

    monkeypatch.setattr(control, "probability_margins", fail_late)
    result = control.diagnose_train_linear_fit(records())
    assert result["status"] == "failed" and result["source_unchanged"] is True
    assert len(result["prediction_probabilities"]) == len(result["target_actions"]) == 5
    assert len(result["scores"]) == 1 and result["work"]["score_reductions_completed"] == 1
    assert result["fit"] is not None and result["work"]["prediction_rows_completed"] == 5
