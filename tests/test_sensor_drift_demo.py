"""Resource-free recipe/CLI checks; the one native qualification is recorded separately."""

import json

import pytest

from scripts.run_sensor_drift_demo import main
from src.adapters.sensor_drift_demo import describe_protocol


def test_should_describe_fixed_bp_delayed_recipe_without_claiming_comparison(capsys):
    assert main(["--describe"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == describe_protocol()
    assert report["training_updates"] == report["diagnostic_forward_calls"] == 12
    assert report["training_rows"] == report["diagnostic_forward_rows"] == 48
    assert report["seed"] == 17 and report["label_delay"] == 2
    assert report["comparison"] is report["final_evaluation"] is False


def test_should_refuse_existing_output_before_any_model_work(tmp_path):
    path = tmp_path / "prior.json"
    path.write_text("original", encoding="utf8")
    with pytest.raises(SystemExit):
        main(["--output", str(path)])
    assert path.read_text(encoding="utf8") == "original"


def test_should_refuse_describe_with_output(tmp_path):
    with pytest.raises(SystemExit):
        main(["--describe", "--output", str(tmp_path / "new.json")])
    assert not (tmp_path / "new.json").exists()
