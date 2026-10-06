"""The larger-input cost probe must be bound before touching any role."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from scripts import profile_cifar_representative_feasibility as profile


def test_should_save_fixed_development_request_exclusively(tmp_path: Path) -> None:
    path = tmp_path / "request.json"

    profile.prepare_request(path)
    request, digest = profile._read_request(path)

    assert digest == sha256(path.read_bytes()).hexdigest()
    assert request["config"]["image_size"] == 224
    assert request["roles"] == {"train": 4096, "guard": 512, "validation": 512}
    assert request["final_test_policy"] == "do_not_construct_or_iterate"
    assert request["probe_timeout_seconds"] == 120
    with pytest.raises(FileExistsError):
        profile.prepare_request(path)


def test_should_reject_changed_request_before_source_or_device_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "request.json"
    profile.prepare_request(path)
    path.write_bytes(path.read_bytes().replace(b'"image_size": 224', b'"image_size": 32'))

    def unexpected() -> None:
        raise AssertionError("source or CUDA runtime opened before request check")

    monkeypatch.setattr(profile, "_verify_cuda_runtime", unexpected)
    with pytest.raises(ValueError, match="predeclared"):
        profile.run_probe(path, tmp_path / "result.json", tmp_path / "failure.json")
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "failure.json").exists()
