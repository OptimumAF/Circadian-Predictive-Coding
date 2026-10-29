"""CLI contract with literal results; no learner, archived fit or task execution."""

import json
from pathlib import Path

import pytest

from scripts import run_bp_ticket_serving as demo
from scripts import ticket_controller_comparison as comparison


def arguments(output, *options):
    return ["--source", "unread-source", "--output", str(output), *options]


def test_should_describe_explicit_catalog_opt_in(capsys):
    with pytest.raises(SystemExit) as stopped:
        demo.main(["--help"])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "--catalog-fit" in help_text and "--controller-comparison" in help_text


@pytest.mark.parametrize(
    "mode",
    [
        [],
        ["--reference", "drift"],
        ["--learned-controller"],
        ["--outer-study"],
        ["--policy-change-controller-comparison", "--fitted-report", "unread-fit"],
        ["--serving-during-updates"],
        ["--publish-during-serving"],
        ["--continue-training-after-publication"],
        ["--context-controller-learning"],
        ["--context-stream-comparison", "--fitted-report", "unread-fit"],
    ],
)
def test_should_refuse_catalog_outside_explicit_comparison_before_io(
    tmp_path, capsys, monkeypatch, mode
):
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        comparison, "run_comparison", lambda *a, **k: pytest.fail("invalid dispatch")
    )
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("IO before validation"))
    with pytest.raises(SystemExit) as stopped:
        demo.main(arguments(output, "--catalog-fit", *mode))
    assert stopped.value.code == 2
    assert "--catalog-fit requires --controller-comparison" in capsys.readouterr().err
    assert not output.exists()


def test_should_require_fitted_report_before_creating_catalog_output(tmp_path, capsys):
    output = tmp_path / "result.json"
    with pytest.raises(SystemExit) as stopped:
        demo.main(arguments(output, "--controller-comparison", "--catalog-fit"))
    assert stopped.value.code == 2
    assert "--fitted-report" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("mode", ["catalog", "legacy", "policy_change"])
@pytest.mark.parametrize("complete", [True, False])
def test_should_forward_only_explicit_catalog_and_preserve_results(
    tmp_path, monkeypatch, mode, complete
):
    output = tmp_path / "result.json"
    calls = []
    report = dict(complete=complete, error=None if complete else "paid work failed", rows=7)

    def run(source, fitted, **kwargs):
        calls.append((source, fitted, kwargs))
        return report

    monkeypatch.setattr(comparison, "run_comparison", run)
    options = ["--fitted-report", "unread-fit"]
    options += (
        ["--policy-change-controller-comparison"]
        if mode == "policy_change"
        else ["--controller-comparison"]
    )
    if mode == "catalog":
        options += ["--catalog-fit"]
    assert demo.main(arguments(output, *options)) == (0 if complete else 1)
    expected = {"catalog_fit": True} if mode == "catalog" else {}
    if mode == "policy_change":
        expected = {"policy_change": True}
    assert calls == [(Path("unread-source"), Path("unread-fit"), expected)]
    assert json.loads(output.read_bytes()) == report


@pytest.mark.parametrize("catalog", [False, True])
def test_should_refuse_occupied_output_before_comparison(tmp_path, monkeypatch, catalog):
    output = tmp_path / "result.json"
    original = b"original paid result\n"
    output.write_bytes(original)
    monkeypatch.setattr(
        comparison, "run_comparison", lambda *a, **k: pytest.fail("occupied output dispatch")
    )
    options = ["--controller-comparison", "--fitted-report", "unread-fit"]
    if catalog:
        options += ["--catalog-fit"]
    with pytest.raises(FileExistsError):
        demo.main(arguments(output, *options))
    assert output.read_bytes() == original


@pytest.mark.parametrize("catalog", [False, True])
@pytest.mark.parametrize("extra_byte", [0, 1])
def test_should_preserve_original_comparison_byte_ceiling(
    tmp_path, monkeypatch, catalog, extra_byte
):
    output = tmp_path / "result.json"
    report = dict(complete=True, literal="")
    overhead = len((json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
    report["literal"] = "x" * (8 * 1024**2 - overhead + extra_byte)
    monkeypatch.setattr(comparison, "run_comparison", lambda *a, **k: report)
    options = ["--controller-comparison", "--fitted-report", "unread-fit"]
    if catalog:
        options += ["--catalog-fit"]
    if extra_byte:
        with pytest.raises(ValueError, match="declared byte limit"):
            demo.main(arguments(output, *options))
        assert output.read_bytes() == b"{}\n"
    else:
        assert demo.main(arguments(output, *options)) == 0
        assert output.stat().st_size == 8 * 1024**2


def test_should_propagate_comparison_exception_without_success_report(tmp_path, monkeypatch):
    output = tmp_path / "result.json"

    def fail(*args, **kwargs):
        raise ValueError("prior evidence changed")

    monkeypatch.setattr(comparison, "run_comparison", fail)
    with pytest.raises(ValueError, match="prior evidence changed"):
        demo.main(
            arguments(
                output, "--controller-comparison", "--catalog-fit", "--fitted-report", "unread-fit"
            )
        )
    assert output.read_bytes() == b"{}\n"
