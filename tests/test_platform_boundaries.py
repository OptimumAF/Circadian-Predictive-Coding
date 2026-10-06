"""Real native locks and explicit native-image support boundaries, without science."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import sys

import pytest

from src.infra.runtime_native_images import WindowsRuntimeImages
from src.infra.v14_resume_files import V14ResumeFiles


def _claim_in_child(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.infra.v14_resume_files import V14ResumeFiles; "
            "from pathlib import Path; import sys; "
            "files = V14ResumeFiles(Path(sys.argv[1]), 'platform-lock'); "
            "lease = files.claim(); lease.__enter__(); lease.__exit__(None, None, None)",
            str(root),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_should_exclude_another_process_and_release_after_consumer_failure(tmp_path: Path) -> None:
    files = V14ResumeFiles(tmp_path, "platform-lock")
    with pytest.raises(RuntimeError, match="consumer failed"):
        with files.claim():
            contender = _claim_in_child(tmp_path)
            assert contender.returncode != 0
            assert "v14 resume run is already active" in contender.stderr
            raise RuntimeError("consumer failed")
    acquired = _claim_in_child(tmp_path)
    assert acquired.returncode == 0, acquired.stdout + acquired.stderr
    # The permanent lock identity survives release and supports another claim.
    assert files.lock_path.is_file()
    assert files.lock_path.read_bytes() == b"0"
    with files.claim():
        assert files.lock_path.is_file()


def test_should_reject_native_observation_on_unsupported_platforms() -> None:
    if sys.platform == "win32" and ctypes.sizeof(ctypes.c_void_p) == 8:
        pytest.skip("positive 64-bit Windows observation runs in the native case")
    with pytest.raises(ValueError, match="requires 64-bit Windows"):
        WindowsRuntimeImages()
