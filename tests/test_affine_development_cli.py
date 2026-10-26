"""Fixed option admission and output behavior, no source/model/native access."""

import json
import pytest
from scripts import run_minigrid_linear_fit_control as cli
from scripts import minigrid_affine_development as development


def test_should_describe_before_source_or_model_io(monkeypatch, capsys):
    monkeypatch.setattr(
        development, "run_saved_development", lambda *a: pytest.fail("executed describe")
    )
    monkeypatch.setattr("sys.argv", ["affine", "--context-linear-a4-development", "--describe"])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(json.dumps(development.describe_development()))
    assert result["new_fit_calls"] == 0 and result["environment_data_role"] == "probe_test"


@pytest.mark.parametrize(
    "options",
    [
        ["--fit-updates", "0"],
        ["--retained-source", "unused"],
        ["--example-affine-recurring"],
        ["--describe", "--source", "unused"],
    ],
)
def test_should_refuse_conflicting_options_before_io(monkeypatch, options):
    monkeypatch.setattr(
        development, "run_saved_development", lambda *a: pytest.fail("executed conflict")
    )
    monkeypatch.setattr("sys.argv", ["affine", "--context-linear-a4-development", *options])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_preserve_failed_output_and_refuse_existing_output_before_more_work(
    monkeypatch, tmp_path, capsys
):
    output = tmp_path / "failed.json"
    calls = []

    def run(path):
        calls.append(path)
        return dict(status="failed", reason="controlled partial", games=dict(status="failed"))

    monkeypatch.setattr(development, "run_saved_development", run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "affine",
            "--context-linear-a4-development",
            "--source",
            "literal",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert (
        error.value.code == 2 and json.loads(output.read_text())["reason"] == "controlled partial"
    )
    with pytest.raises(FileExistsError):
        cli.main()
    assert len(calls) == 1
