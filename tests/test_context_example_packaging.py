"""Bounded resource/lifetime/CLI metadata checks; no learning or task execution."""

from hashlib import sha256
import json
from pathlib import Path
from zipfile import Path as ZipPath, ZipFile

import pytest

from scripts import run_bp_ticket_serving as cli
from scripts import context_head_serving as serving
from src.infra import context_head_example as example

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "examples/context_head_source/source.json"


def zip_resource(tmp_path, monkeypatch, *, raw=None):
    archive_path = tmp_path / "resources.zip"
    with ZipFile(archive_path, "w") as archive:
        if raw is not None:
            archive.writestr("source.json", raw)
    monkeypatch.setattr(example, "__file__", str(tmp_path / "repo/src/infra/example.py"))
    archive = ZipFile(archive_path)

    def files(package):
        assert package == "src.infra.context_head_data"
        return ZipPath(archive)

    monkeypatch.setattr(example.resources, "files", files)
    return archive


def test_should_preserve_exact_source_bytes_and_training_history(monkeypatch):
    raw = RAW.read_bytes()
    assert raw == (ROOT / "artifacts/runs/context-head-control-20261023/actual.json").read_bytes()
    assert sha256(raw).hexdigest() == example.SOURCE_SHA
    monkeypatch.setattr(
        example.resources, "files", lambda *a: pytest.fail("local lookup fell back")
    )
    with example.example_context_source() as path:
        record = json.loads(path.read_bytes())
        final = record["checkpoints"][-1]["state"]
        assert (
            final["completed_updates"],
            final["training_examples"],
            final["mistake_updates"],
        ) == (12, 72, 16)


def test_should_hold_zip_source_through_public_cli_and_release_after_output(tmp_path, monkeypatch):
    seen = []
    output = tmp_path / "result.json"
    with zip_resource(tmp_path, monkeypatch, raw=RAW.read_bytes()):

        def run(path):
            assert path.is_file() and path.read_bytes() == RAW.read_bytes()
            seen.append(path)
            return dict(complete=True, fixture=True)

        monkeypatch.setattr(serving, "run_case", run)
        assert (
            cli.main(
                ["--example-context-source", "--context-head-serving", "--output", str(output)]
            )
            == 0
        )
        assert not seen[0].exists()
    assert json.loads(output.read_bytes()) == dict(complete=True, fixture=True)


@pytest.mark.parametrize(
    "mode",
    [
        [],
        ["--context-head-serving-live"],
        ["--reference", "periodic"],
        ["--context-stream-protocol"],
    ],
)
def test_should_refuse_unsupported_bundled_mode_before_lookup_or_output(
    tmp_path, monkeypatch, mode
):
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        example, "example_context_source", lambda: pytest.fail("invalid mode looked up source")
    )
    with pytest.raises(SystemExit) as error:
        cli.main(["--example-context-source", *mode, "--output", str(output)])
    assert error.value.code == 2 and not output.exists()


def test_should_require_one_explicit_source_choice_before_output(tmp_path):
    output = tmp_path / "result.json"
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "--example-context-source",
                "--source",
                "other.json",
                "--context-head-serving",
                "--output",
                str(output),
            ]
        )
    assert error.value.code == 2 and not output.exists()


@pytest.mark.parametrize("raw", [b"{}", b"x" * (512 * 1024 + 1)], ids=["changed", "oversized"])
def test_should_refuse_corrupt_local_resource_without_fallback_or_output(
    tmp_path, monkeypatch, raw
):
    folder = tmp_path / "examples/context_head_source"
    folder.mkdir(parents=True)
    (folder / "source.json").write_bytes(raw)
    monkeypatch.setattr(example, "__file__", str(tmp_path / "src/infra/example.py"))
    monkeypatch.setattr(example.resources, "files", lambda *a: pytest.fail("corruption fell back"))
    output = tmp_path / "result.json"
    with pytest.raises(ValueError, match="qualified example"):
        cli.main(["--example-context-source", "--context-head-serving", "--output", str(output)])
    assert not output.exists()


def test_should_refuse_missing_zip_resource_before_output(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    with zip_resource(tmp_path, monkeypatch):
        with pytest.raises(FileNotFoundError):
            cli.main(
                ["--example-context-source", "--context-head-serving", "--output", str(output)]
            )
    assert not output.exists()


def test_should_refuse_changed_resource_on_context_exit(tmp_path, monkeypatch):
    with zip_resource(tmp_path, monkeypatch, raw=RAW.read_bytes()):
        with pytest.raises(ValueError, match="qualified example"):
            with example.example_context_source() as path:
                path.write_bytes(b"{}")
        assert not path.exists()


def test_should_release_zip_resource_after_runner_failure_and_retain_placeholder(
    tmp_path, monkeypatch
):
    output = tmp_path / "result.json"
    seen = []

    def fail(path):
        seen.append(path)
        raise RuntimeError("scripted failure")

    with zip_resource(tmp_path, monkeypatch, raw=RAW.read_bytes()):
        monkeypatch.setattr(serving, "run_case", fail)
        with pytest.raises(RuntimeError, match="scripted failure"):
            cli.main(
                ["--example-context-source", "--context-head-serving", "--output", str(output)]
            )
        assert not seen[0].exists()
    assert output.read_bytes() == b"{}\n"


def test_should_preserve_exclusive_output_before_runner(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    output.write_bytes(b"original")
    monkeypatch.setattr(
        serving, "run_case", lambda *a: pytest.fail("overwritten output entered runner")
    )
    with pytest.raises(FileExistsError):
        cli.main(["--example-context-source", "--context-head-serving", "--output", str(output)])
    assert output.read_bytes() == b"original"


def test_should_keep_explicit_source_default_without_example_lookup(tmp_path, monkeypatch):
    source, output = tmp_path / "explicit.json", tmp_path / "result.json"
    monkeypatch.setattr(
        example, "example_context_source", lambda: pytest.fail("explicit source looked up example")
    )
    seen = []

    def run(path):
        seen.append(path)
        return dict(complete=True, fixture=True)

    monkeypatch.setattr(cli, "run_case", run)
    assert cli.main(["--source", str(source), "--output", str(output)]) == 0
    assert seen == [source]


def test_should_declare_resource_and_license_in_existing_package_configuration():
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    setup = config["tool"]["setuptools"]
    assert "src.infra.context_head_data" in setup["packages"]
    assert setup["package-dir"]["src.infra.context_head_data"] == "examples/context_head_source"
    assert setup["package-data"]["src.infra.context_head_data"] == ["source.json", "README.md"]
    assert "MIT" in (RAW.parent / "README.md").read_text()
    assert "examples/context_head_source/source.json -text" in (ROOT / ".gitattributes").read_text()
