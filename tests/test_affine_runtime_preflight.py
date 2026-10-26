"""Optional runtime refusal before any source, model or TRAIN work."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys

import pytest

from scripts import minigrid_affine_workflow as workflow
from scripts import run_minigrid_linear_fit_control as cli
from src.infra import minigrid_memory as runtime


def forbid_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("runtime refusal reached source/model/TRAIN/game work")

    monkeypatch.setattr(workflow, "example_affine_training", forbidden)
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(cli, "run_context_linear_stream", forbidden)
    monkeypatch.setattr(workflow, "ContextLinearLearner", forbidden)
    monkeypatch.setattr(workflow, "evaluate_context_linear_stream_checkpoint", forbidden)


def test_should_reuse_original_pins_before_importing_optional_runtime(monkeypatch):
    events = []
    gym = object()

    def version(name):
        events.append(("version", name))
        return runtime.VERSIONS[name]

    def imported(name):
        events.append(("import", name))
        return gym if name == "gymnasium" else object()

    monkeypatch.setattr(runtime.metadata, "version", version)
    monkeypatch.setattr(runtime, "import_module", imported)
    assert runtime._require_minigrid() is gym
    assert events == [("version", name) for name in runtime.VERSIONS] + [
        ("import", "gymnasium"),
        ("import", "minigrid"),
    ]


@pytest.mark.parametrize("name", tuple(runtime.VERSIONS))
def test_should_refuse_mismatched_version_before_optional_imports(monkeypatch, name):
    monkeypatch.setattr(
        runtime.metadata, "version", lambda key: "wrong" if key == name else runtime.VERSIONS[key]
    )
    monkeypatch.setattr(
        runtime, "import_module", lambda *a: pytest.fail("import after pin mismatch")
    )
    with pytest.raises(ValueError, match=f"{name} must be pinned"):
        runtime._require_minigrid()


def test_should_report_missing_distribution_without_optional_imports(monkeypatch):
    def missing(name):
        raise runtime.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runtime.metadata, "version", missing)
    monkeypatch.setattr(
        runtime, "import_module", lambda *a: pytest.fail("import after missing package")
    )
    with pytest.raises(RuntimeError, match="minigrid extra"):
        runtime._require_minigrid()


@pytest.mark.parametrize("missing", ("gymnasium", "minigrid"))
def test_should_report_incomplete_runtime_import_with_existing_setup_guidance(monkeypatch, missing):
    monkeypatch.setattr(runtime.metadata, "version", runtime.VERSIONS.__getitem__)

    def imported(name):
        if name == missing:
            raise ModuleNotFoundError(name)
        return object()

    monkeypatch.setattr(runtime, "import_module", imported)
    with pytest.raises(RuntimeError, match="requirements-minigrid.txt"):
        runtime._require_minigrid()


@pytest.mark.parametrize(
    "error",
    (RuntimeError("install minigrid extra"), ValueError("minigrid must be pinned")),
    ids=("missing", "mismatched"),
)
def test_should_capture_preflight_failure_before_source_owner_or_training(monkeypatch, error):
    forbid_work(monkeypatch)

    def preflight():
        raise error

    monkeypatch.setattr(workflow, "_require_minigrid", preflight)
    result = workflow.run_workflow()
    assert result["status"] == "failed" and str(error) in result["reason"]
    assert result["training"] is result["games"] is None
    assert "readout_budget" not in result


def test_should_check_runtime_before_decoding_and_admitting_original_stream(monkeypatch):
    events = []
    monkeypatch.setattr(workflow, "_require_minigrid", lambda: events.append("preflight"))

    @contextmanager
    def source():
        events.append("resource")
        yield Path("scripted.json")

    def decode(path):
        assert path == Path("scripted.json")
        events.append("decode")
        return ()

    def stream(*args, **options):
        assert options["retain_arrived"] is options["abc_catalogue"] is True
        events.append("stream")
        raise RuntimeError("deliberate stop before model work")

    monkeypatch.setattr(workflow, "example_affine_training", source)
    monkeypatch.setattr(cli, "read_training_sources", decode)
    monkeypatch.setattr(cli, "run_context_linear_stream", stream)
    result = workflow.run_workflow()
    assert result["status"] == "failed" and "deliberate stop" in result["reason"]
    assert events == ["preflight", "resource", "decode", "stream"]


def test_should_describe_without_optional_runtime_or_source_work(monkeypatch, capsys):
    forbid_work(monkeypatch)
    monkeypatch.setattr(
        workflow, "_require_minigrid", lambda: pytest.fail("describe required optional runtime")
    )
    monkeypatch.setattr(sys, "argv", ["control", "--example-affine-play", "--describe"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["training"]["fit_rows"] == 108


@pytest.mark.parametrize(
    "error",
    (RuntimeError("install minigrid extra"), ValueError("minigrid must be pinned")),
    ids=("missing", "mismatched"),
)
def test_should_publish_failed_cli_result_without_any_paid_work(
    tmp_path, monkeypatch, capsys, error
):
    forbid_work(monkeypatch)

    def preflight():
        raise error

    monkeypatch.setattr(workflow, "_require_minigrid", preflight)
    output = tmp_path / "failed.json"
    monkeypatch.setattr(sys, "argv", ["control", "--example-affine-play", "--output", str(output)])
    with pytest.raises(SystemExit) as failed:
        cli.main()
    assert failed.value.code == 2
    report = json.loads(output.read_bytes())
    assert report["status"] == "failed" and report["training"] is report["games"] is None
    assert json.loads(capsys.readouterr().out) == report
    monkeypatch.setattr(
        workflow, "_require_minigrid", lambda: pytest.fail("existing output repeated preflight")
    )
    with pytest.raises(FileExistsError):
        cli.main()
