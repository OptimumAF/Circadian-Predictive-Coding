"""CLI preview, dedicated-directory refusal and actual supervised writer example."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_coupled_checkpoint_interruption as cli
from src.adapters import coupled_checkpoint_interruption as demo


def test_should_describe_without_model_process_or_filesystem_work(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("description performed operational work")

    monkeypatch.setattr(demo, "_model", forbidden)
    monkeypatch.setattr(demo, "BoundedJsonChild", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(sys, "argv", ["writer-interruption", "--describe"])
    cli.main()
    assert json.loads(capsys.readouterr().out) == demo.describe_example()
    assert demo.describe_example()["child_training_calls"] == 0


def test_should_refuse_unsupported_platform_before_directory_or_native_work(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported platform performed operational work")

    monkeypatch.setattr(demo, "_model", forbidden)
    monkeypatch.setattr(demo, "BoundedJsonChild", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    with monkeypatch.context() as local:
        local.setattr(demo.sys, "platform", "unsupported")
        with pytest.raises(OSError, match="Windows"):
            demo.run_example(tmp_path / "new")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows retained-process observation")
def test_should_preserve_existing_directory_before_models_or_processes(monkeypatch, tmp_path):
    marker = tmp_path / "user.txt"
    marker.write_text("preserve existing user file")

    def forbidden(*args, **kwargs):
        raise AssertionError("existing directory allowed native/process work")

    monkeypatch.setattr(demo, "_model", forbidden)
    monkeypatch.setattr(demo, "BoundedJsonChild", forbidden)
    with pytest.raises(FileExistsError):
        demo.run_example(tmp_path)
    assert marker.read_text() == "preserve existing user file"
    assert list(tmp_path.iterdir()) == [marker]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows retained-process observation")
def test_should_run_actual_cli_with_exact_actor_recovery_and_spent_attempts(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_coupled_checkpoint_interruption",
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
    assert report["owned_child_launches"] == 6 and report["child_training_calls"] == 0
    assert report["remaining_temporary_files"] == []
    for row in report["rows"]:
        assert Path(row["path"]).read_bytes() == row["checkpoint_a"]["payload_utf8"].encode()
        assert row["checkpoint_a"]["snapshot"] == row["recovered"]["snapshot"]
        assert row["staged_payload_utf8"] == row["candidate_b"]["payload_utf8"]
        assert row["staged_file_removed"] and row["refused_callback_calls"] == 0
        assert row["child_exit_codes"][0] != 0 and row["child_exit_codes"][1] == 0
        authority = row["authority"]
        assert authority["reserved"]["metadata"]["usage"]["checkpoint_attempts"] == 1
        assert authority["handed_off"]["metadata"]["usage"]["checkpoint_attempts"] == 2
        assert authority["terminal"]["metadata"]["stopped"] is True
        for observation in row["predictions"].values():
            assert observation == row["predictions"]["before"] == row["recovered"]["predictions"]
