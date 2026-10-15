"""CLI boundaries and one actual four-case native completion demonstration."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import run_coupled_update_completion as cli
from src.adapters import coupled_update_completion as demo
from src.adapters import coupled_update_worker as worker


@pytest.mark.parametrize(
    "shapes,precisions",
    ((((2, 4), (4, 2)), (1.0, 1.0)), (((2, 3),), (1.0,)), (((2, 3), (3, 2)), (0.5, 1.0))),
)
def test_should_refuse_unfrozen_geometry_before_native_model_work(monkeypatch, shapes, precisions):
    snapshot = SimpleNamespace(
        parameters=SimpleNamespace(
            hidden_weights=tuple(SimpleNamespace(shape=shape) for shape in shapes)
        ),
        precisions=precisions,
    )

    class Codec:
        def decode(self, *args, **kwargs):
            return snapshot

    def forbidden(*args, **kwargs):
        raise AssertionError("unfrozen geometry constructed a native model")

    monkeypatch.setattr(worker, "_emit", lambda value: None)
    monkeypatch.setattr(
        worker,
        "_request",
        lambda: dict(
            op="update",
            binding=demo.describe_example()["binding"],
            limits=demo.describe_example()["limits"],
            expected_sha256="a" * 64,
            payload_utf8="{}",
        ),
    )
    monkeypatch.setattr(worker, "CoupledCheckpointCodec", Codec)
    monkeypatch.setattr(worker, "_model", forbidden)
    with pytest.raises(ValueError, match="frozen"):
        worker.main()


def test_should_describe_without_native_filesystem_or_process_work(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("description performed operational work")

    monkeypatch.setattr(demo, "_model", forbidden)
    monkeypatch.setattr(demo, "BoundedJsonChild", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(sys, "argv", ["native-completion", "--describe"])
    cli.main()
    assert json.loads(capsys.readouterr().out) == demo.describe_example()


def test_should_refuse_unsupported_platform_before_native_work(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported platform performed operational work")

    monkeypatch.setattr(demo, "_model", forbidden)
    monkeypatch.setattr(demo, "BoundedJsonChild", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    with monkeypatch.context() as local:
        local.setattr(demo.sys, "platform", "unsupported")
        with pytest.raises(OSError, match="Windows"):
            demo.run_example(tmp_path / "new")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows registered worker integration")
def test_should_preserve_existing_directory_before_native_work(monkeypatch, tmp_path):
    marker = tmp_path / "unrelated.txt"
    marker.write_text("preserve user data")

    def forbidden(*args, **kwargs):
        raise AssertionError("existing directory dispatched native work")

    monkeypatch.setattr(demo, "_model", forbidden)
    monkeypatch.setattr(demo, "BoundedJsonChild", forbidden)
    with pytest.raises(FileExistsError):
        demo.run_example(tmp_path)
    assert list(tmp_path.iterdir()) == [marker] and marker.read_text() == "preserve user data"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows registered worker integration")
def test_should_complete_validated_native_updates_and_refuse_written_b_without_validation(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_coupled_update_completion",
            "--directory",
            str(tmp_path / "example"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        timeout=60,
        check=True,
    )
    assert result.stderr == b"" and len(result.stdout) <= 2 * 1024**2
    report = json.loads(result.stdout)
    assert report["native_training_calls"] == 16 and report["admitted_child_updates"] == 4
    assert report["remaining_temporary_files"] == [] and len(report["rows"]) == 4
    for row in report["rows"]:
        assert Path(row["path_a"]).read_bytes() == row["checkpoint_a"]["payload_utf8"].encode()
        assert Path(row["path_b"]).read_bytes() == row["actual_b"]["payload_utf8"].encode()
        assert row["actual_b"] == row["expected_b"] and row["exit_code"] == 0
        assert all(value == row["predictions"]["before"] for value in row["predictions"].values())
        metadata = row["authority"]["terminal"]["metadata"]
        assert metadata["usage"]["updates_admitted"] == 1 and metadata["stopped"]
        assert metadata["usage"]["updates_completed"] == int(not row["expected_failure"])
        assert metadata["uncertain_work"] == row["expected_failure"]
        assert row["callbacks"]["refused_prepare"] == row["callbacks"]["refused_validate"] == 0
        assert row["response"]["training_calls"] == 1
