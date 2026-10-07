"""Bounded artifact checks for the local pretrained CPU feature-profile writer."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from scripts import profile_cifar_feature_setup as profile  # noqa: E402


ROLES = ("train", "guard", "validation")


@dataclass(frozen=True)
class StubLoaders:
    test_loader: Any
    sample_ids: dict[str, tuple[int, ...]]


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite profile value: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    assert isinstance(record, dict)
    return record


def _install_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    paths = {name: tmp_path / f"profile-{name}.json" for name in ("request", "result", "failure")}
    monkeypatch.setattr(profile, "REQUEST_PATH", paths["request"])
    monkeypatch.setattr(profile, "RESULT_PATH", paths["result"])
    monkeypatch.setattr(profile, "FAILURE_PATH", paths["failure"])
    return paths


def _install_weight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    weight = tmp_path / "tiny-local-weight.pth"
    weight.write_bytes(b"bounded local stub weight")
    url = "stub://local-resnet50-v2"
    monkeypatch.setattr(profile, "_weight_source", lambda: (weight, url))
    return weight, url


def test_should_read_profile_request_and_result_without_final_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _install_paths(tmp_path, monkeypatch)
    weight, url = _install_weight(tmp_path, monkeypatch)
    feature = torch.ones((2, 5), dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    role_batches = ((feature, labels),)
    source_calls = 0

    def source_loader(config: Any, *, include_final_test: bool = True) -> StubLoaders:
        nonlocal source_calls
        source_calls += 1
        assert include_final_test is False
        assert config.device == "cpu"
        assert config.dataset_download is False
        assert config.backbone_weights == "imagenet"
        return StubLoaders(object(), {role: (0, 1) for role in ROLES})

    def feature_bank(
        torch_module: Any, device: Any, config: Any, *, include_final_test: bool = True
    ) -> Any:
        assert torch_module is torch
        assert str(device) == "cpu"
        assert include_final_test is False
        loaders = profile.matched._build_benchmark_loaders(config, include_final_test=False)
        with pytest.raises(AssertionError, match="opened final test"):
            iter(loaders.test_loader)
        return SimpleNamespace(
            train=role_batches,
            guard=role_batches,
            validation=role_batches,
            loaders=loaders,
            backbone_hash="a" * 64,
            split_hashes={role: role * 2 for role in ROLES},
            feature_hashes={role: role * 3 for role in ROLES},
        )

    monkeypatch.setattr(profile.matched, "_build_benchmark_loaders", source_loader)
    monkeypatch.setattr(profile, "_build_seed_bank", feature_bank)
    profile.main()

    assert source_calls == 1
    assert set(path.name for path in tmp_path.glob("profile-*.json")) == {
        paths["request"].name,
        paths["result"].name,
    }
    request = _read_json(paths["request"])
    result = _read_json(paths["result"])
    printed = json.loads(capsys.readouterr().out, parse_constant=_reject_nonfinite)
    expected_digest = sha256(weight.read_bytes()).hexdigest()
    assert request["weight_url"] == result["weight_url"] == url
    assert request["weight_sha256"] == result["weight_sha256"] == expected_digest
    assert request["weight_bytes"] == result["weight_bytes"] == weight.stat().st_size
    assert request["config"]["seed"] == result["seed"] == 101
    assert request["config"]["device"] == "cpu"
    assert request["config"]["dataset_download"] is False
    assert request["config"]["backbone_weights"] == "imagenet"
    assert request["wall_budget_seconds"] == profile.WALL_BUDGET_SECONDS == 60
    assert request["final_test_iterations_allowed"] == result["final_test_iterations"] == 0
    assert result["backbone_hash"] == "a" * 64
    assert result["split_hashes"] == {role: role * 2 for role in ROLES}
    assert result["feature_hashes"] == {role: role * 3 for role in ROLES}
    assert result["feature_bytes"] == {role: 56 for role in ROLES}
    assert result["role_counts"] == {role: 2 for role in ROLES}
    assert printed["role_counts"] == result["role_counts"]
    assert printed["result_path"] == str(paths["result"])
    assert not paths["failure"].exists()


@pytest.mark.parametrize("occupied", ("request", "result", "failure"))
def test_should_refuse_profile_occupied_path_before_weight_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, occupied: str
) -> None:
    paths = _install_paths(tmp_path, monkeypatch)
    paths[occupied].write_bytes(b"previous user content")
    monkeypatch.setattr(
        profile, "_weight_source", lambda: pytest.fail("opened weight before output preflight")
    )
    with pytest.raises(FileExistsError, match="already has an artifact"):
        profile.main()
    assert paths[occupied].read_bytes() == b"previous user content"


def test_should_keep_profile_request_and_failure_without_false_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _install_paths(tmp_path, monkeypatch)
    weight, _ = _install_weight(tmp_path, monkeypatch)

    def fail_bank(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("injected profile setup failure")

    monkeypatch.setattr(profile, "_build_seed_bank", fail_bank)
    with pytest.raises(RuntimeError, match="injected profile setup failure"):
        profile.main()
    assert paths["request"].is_file()
    assert not paths["result"].exists()
    failure = _read_json(paths["failure"])
    assert failure["error_type"] == "RuntimeError"
    assert failure["error"] == "injected profile setup failure"
    assert failure["elapsed_seconds"] >= 0
    assert _read_json(paths["request"])["weight_sha256"] == sha256(weight.read_bytes()).hexdigest()
