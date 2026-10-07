"""Keep the CIFAR final source physically absent during older development writers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5, sha256
import json
from pathlib import Path
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from scripts import profile_cifar_feature_setup as profile  # noqa: E402
from scripts import run_cifar_matched_validation as local_validation  # noqa: E402
from scripts import run_cifar_matched_confirmation as local_confirmation  # noqa: E402
from scripts import run_cifar_pretrained_validation as pretrained_validation  # noqa: E402
from scripts import run_cifar_pretrained_confirmation as pretrained_confirmation  # noqa: E402
from src.app import matched_head_benchmark as matched  # noqa: E402
from src.app.matched_head_tuning import (  # noqa: E402
    HEAD_NAMES,
    HeadTuningAttempt,
    MatchedHeadTuningError,
)
from src.app.repeated_head_confirmation import (  # noqa: E402
    REPEATED_CONFIRMATION_PROTOCOL,
    DescriptiveSummary,
    RepeatedConfirmationManifest,
    RepeatedConfirmationResult,
    restore_confirmation_manifest,
)


DEVELOPMENT_ROLES = ("train", "guard", "validation")


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite development writer value: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    assert isinstance(record, dict)
    return record


@dataclass(frozen=True)
class TinyLoaders:
    train_loader: Any
    guard_loader: Any
    validation_loader: Any
    test_loader: Any
    num_classes: int
    sample_ids: dict[str, tuple[str, ...]]
    split_hashes: dict[str, str]


@dataclass(frozen=True)
class StubFixedData:
    seeds: tuple[int, ...]
    attempts: tuple[dict[str, Any], ...]
    trials: tuple[dict[str, Any], ...]
    confirmations: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class StubMemory:
    seed: int
    reports: dict[str, dict[str, Any]]


def _synthetic_confirmation_result(
    manifest: RepeatedConfirmationManifest,
) -> RepeatedConfirmationResult:
    """Build writer-only finite rows; these are not model measurements."""
    choices = {item.head_name: item.candidate_id for item in manifest.selected_heads}
    cells = tuple((head, seed) for seed in manifest.confirmation_seeds for head in HEAD_NAMES)
    fixed = StubFixedData(
        seeds=manifest.confirmation_seeds,
        attempts=tuple(
            {"head_name": head, "candidate_id": choices[head], "seed": seed, "status": "complete"}
            for head, seed in cells
        ),
        trials=tuple(
            {
                "head_name": head,
                "candidate_id": choices[head],
                "seed": seed,
                "validation_accuracy": 0.25,
            }
            for head, seed in cells
        ),
        confirmations=tuple(
            {"head_name": head, "candidate_id": choices[head], "seed": seed, "test_accuracy": 0.25}
            for head, seed in cells
        ),
    )
    wall = tuple(
        {
            "seed": seed,
            "wall_time_budget_seconds": manifest.wall_time_budget_seconds,
            "reports": {head: {"test_accuracy": 0.25} for head in HEAD_NAMES},
        }
        for seed in manifest.confirmation_seeds
    )
    memory = tuple(
        StubMemory(
            seed=seed,
            reports={
                head: {"pid": 1000 + index * 3 + offset, "train_rss_bytes": 4096}
                for offset, head in enumerate(HEAD_NAMES)
            },
        )
        for index, seed in enumerate(manifest.confirmation_seeds)
    )

    def summaries(metric: str, value: float) -> dict[str, DescriptiveSummary]:
        return {
            head: DescriptiveSummary(
                metric_name=metric,
                seeds=manifest.confirmation_seeds,
                values=(value,) * len(manifest.confirmation_seeds),
                mean=value,
                population_std=0.0,
            )
            for head in HEAD_NAMES
        }

    return RepeatedConfirmationResult(
        protocol_id=REPEATED_CONFIRMATION_PROTOCOL,
        manifest=manifest,
        fixed_data=cast(Any, fixed),
        wall_time=cast(Any, wall),
        capacity_memory=cast(Any, memory),
        fixed_data_accuracy=summaries("accuracy", 0.25),
        wall_time_accuracy=summaries("accuracy", 0.25),
        observed_train_rss=summaries("train_rss_bytes", 4096.0),
    )


def _install_raising_final_source(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    requests: list[bool] = []
    features = torch.tensor(
        [[0.2, 0.1, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)
    role_batches = [(features, labels)]

    class SealedTestLoader:
        def __iter__(self) -> Any:
            raise AssertionError("final source iterated during development work")

    def build_loaders(config: Any, *, include_final_test: bool = True) -> TinyLoaders:
        del config
        requests.append(include_final_test)
        if include_final_test:
            raise AssertionError("final source constructed during development work")
        return TinyLoaders(
            train_loader=role_batches,
            guard_loader=list(role_batches),
            validation_loader=list(role_batches),
            test_loader=SealedTestLoader(),
            num_classes=10,
            sample_ids={role: (f"{role}/0", f"{role}/1") for role in DEVELOPMENT_ROLES},
            split_hashes={role: f"{role}-split" for role in DEVELOPMENT_ROLES},
        )

    monkeypatch.setattr(matched, "_build_benchmark_loaders", build_loaders)
    monkeypatch.setattr(
        matched,
        "_build_resnet50_backbone",
        lambda **kwargs: (torch.nn.Identity(), 5),
    )
    monkeypatch.setattr(
        matched,
        "_materialize_role_features",
        lambda *args: ((features, labels),),
    )
    return requests


def _install_tiny_verified_inputs(
    producer: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path | None, Path]:
    archive = tmp_path / "data" / "cifar-10-python.tar.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"bounded local CIFAR archive stub")
    output_dir = tmp_path / "out"
    monkeypatch.setattr(producer, "ARCHIVE", archive)
    monkeypatch.setattr(producer, "ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(producer, "ARCHIVE_MD5", md5(archive.read_bytes()).hexdigest())
    if producer is local_validation:
        monkeypatch.setattr(producer, "OUTPUT_DIR", output_dir)
        return archive, None, output_dir

    monkeypatch.setattr(producer, "ARTIFACT_DIR", output_dir)
    hub = tmp_path / "hub"
    checkpoint = hub / "checkpoints" / producer.WEIGHT_URL.rsplit("/", 1)[-1]
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"bounded local ImageNet V2 weight stub")
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(hub))
    monkeypatch.setattr(producer, "WEIGHT_BYTES", checkpoint.stat().st_size)
    monkeypatch.setattr(producer, "WEIGHT_SHA256", sha256(checkpoint.read_bytes()).hexdigest())
    return archive, checkpoint, output_dir


@pytest.fixture(autouse=True)
def _single_cpu_thread() -> Any:
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def test_should_profile_only_development_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = _install_raising_final_source(monkeypatch)
    weight = tmp_path / "stub-weight.pth"
    weight.write_bytes(b"local tiny weight stub")
    monkeypatch.setattr(profile, "_weight_source", lambda: (weight, "stub://local-weight"))
    monkeypatch.setattr(profile, "REQUEST_PATH", tmp_path / "profile-request.json")
    monkeypatch.setattr(profile, "RESULT_PATH", tmp_path / "profile-result.json")
    monkeypatch.setattr(profile, "FAILURE_PATH", tmp_path / "profile-failure.json")

    profile.main()

    assert requests == [False]
    assert not (tmp_path / "profile-failure.json").exists()
    result = json.loads((tmp_path / "profile-result.json").read_text(encoding="utf-8"))
    assert result["final_test_iterations"] == 0
    assert set(result["feature_hashes"]) == set(DEVELOPMENT_ROLES)


@pytest.mark.parametrize("producer", [local_validation, pretrained_validation])
def test_should_select_only_development_source(
    producer: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests = _install_raising_final_source(monkeypatch)
    archive, weight, output_dir = _install_tiny_verified_inputs(producer, tmp_path, monkeypatch)
    if producer is local_validation:
        prefix = "benchmark_cifar_v2"
        expected_seed = 73
        expected_confirmation = [83, 89, 97]
    else:
        prefix = "benchmark_cifar_pretrained_v1"
        expected_seed = 113
        expected_confirmation = [127, 131, 137]

    producer.main()

    assert requests == [False]
    request = _read_json(output_dir / f"{prefix}_request_smoke.json")
    selection = _read_json(output_dir / f"{prefix}_selection_smoke.json")
    manifest = _read_json(output_dir / f"{prefix}_manifest_smoke.json")
    printed = json.loads(capsys.readouterr().out, parse_constant=_reject_nonfinite)
    assert {path.name for path in output_dir.iterdir()} == {
        f"{prefix}_{name}_smoke.json" for name in ("request", "selection", "manifest")
    }
    assert request["archive"] == str(archive)
    assert request["archive_bytes"] == archive.stat().st_size
    assert request["archive_md5"] == md5(archive.read_bytes()).hexdigest()
    if weight is not None:
        assert request["weight_file"] == str(weight)
        assert request["weight_bytes"] == weight.stat().st_size
        assert request["weight_sha256"] == sha256(weight.read_bytes()).hexdigest()
    assert request["selection_seeds"] == selection["seeds"] == [expected_seed]
    assert request["confirmation_seeds"] == manifest["confirmation_seeds"] == expected_confirmation
    assert request["candidates_per_head"] == selection["candidates_per_head"] == 2
    assert {
        head: [candidate["candidate_id"] for candidate in options]
        for head, options in request["candidates"].items()
    } == {head: ["a", "b"] for head in HEAD_NAMES}
    assert request["base_config"] == selection["base_config"] == manifest["base_config"]
    assert request["base_config"]["device"] == "cpu"
    assert request["base_config"]["dataset_download"] is False
    assert request["base_config"]["backbone_weights"] == (
        "none" if producer is local_validation else "imagenet"
    )
    assert request["wall_time_epoch_cap"] == manifest["wall_time_epoch_cap"] == 1000
    assert request["wall_time_budget_seconds"] == manifest["wall_time_budget_seconds"]
    assert request[
        "selection_wall_budget_seconds"
        if producer is local_validation
        else "selection_limit_seconds"
    ] == (180 if producer is local_validation else 120)
    assert printed["manifest_digest"] == manifest["manifest_digest"]
    assert printed["selection_seeds"] == [expected_seed]
    assert printed["confirmation_seeds"] == expected_confirmation
    assert printed["attempts"] == 6
    assert printed["final_test_iterations"] == 0
    assert len(selection["attempts"]) == len(selection["trials"]) == 6
    assert all(row["status"] == "complete" for row in selection["attempts"])
    assert len(selection["selections"]) == 3
    assert selection["confirmations"] == []
    assert all(set(row["split_hashes"]) == set(DEVELOPMENT_ROLES) for row in selection["trials"])
    assert all(set(row["feature_hashes"]) == set(DEVELOPMENT_ROLES) for row in selection["trials"])
    assert (
        len({json.dumps(row["split_hashes"], sort_keys=True) for row in selection["trials"]}) == 1
    )
    assert len({row["backbone_hash"] for row in selection["trials"]}) == 1
    assert len({row["initial_head_hash"] for row in selection["trials"]}) == 1
    assert restore_confirmation_manifest(manifest).manifest_digest == manifest["manifest_digest"]
    assert (
        manifest["source_selection_digest"]
        == sha256(json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
    assert not (output_dir / f"{prefix}_failure_smoke.json").exists()


@pytest.mark.parametrize(
    ("producer", "occupied"),
    [(local_validation, name) for name in ("request", "selection", "manifest", "failure")]
    + [(pretrained_validation, name) for name in ("request", "selection", "manifest", "failure")],
)
def test_should_preflight_selection_outputs_before_source_access(
    producer: Any, occupied: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if producer is local_validation:
        monkeypatch.setattr(producer, "OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(
            producer,
            "_verify_archive",
            lambda: pytest.fail("opened archive before output preflight"),
        )
        prefix = "benchmark_cifar_v2"
    else:
        monkeypatch.setattr(producer, "ARTIFACT_DIR", tmp_path)
        monkeypatch.setattr(
            producer,
            "_verify_local_inputs",
            lambda: pytest.fail("opened local inputs before output preflight"),
        )
        prefix = "benchmark_cifar_pretrained_v1"
    path = tmp_path / f"{prefix}_{occupied}_smoke.json"
    path.write_bytes(b"previous user content")

    with pytest.raises(FileExistsError, match="already exist"):
        producer.main()

    assert path.read_bytes() == b"previous user content"


@pytest.mark.parametrize("producer", [local_validation, pretrained_validation])
def test_should_keep_attempt_failure_after_verified_request(
    producer: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, output_dir = _install_tiny_verified_inputs(producer, tmp_path, monkeypatch)
    prefix = (
        "benchmark_cifar_v2" if producer is local_validation else "benchmark_cifar_pretrained_v1"
    )
    attempt = HeadTuningAttempt(
        head_name="backprop_mlp",
        candidate_id="a",
        seed=producer.SELECTION_SEEDS[0],
        config=producer._base_config(),
        status="failed",
        error="RuntimeError: injected development failure",
    )

    def fail_selection(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["confirm_test"] is False
        assert kwargs["development_only_source"] is True
        raise MatchedHeadTuningError("injected development failure", (attempt,), ())

    monkeypatch.setattr(producer, "run_matched_head_tuning", fail_selection)
    with pytest.raises(MatchedHeadTuningError, match="injected development failure"):
        producer.main()

    request = _read_json(output_dir / f"{prefix}_request_smoke.json")
    assert request["selection_seeds"] == [producer.SELECTION_SEEDS[0]]
    assert not (output_dir / f"{prefix}_selection_smoke.json").exists()
    assert not (output_dir / f"{prefix}_manifest_smoke.json").exists()
    failure = _read_json(output_dir / f"{prefix}_failure_smoke.json")
    assert failure["error_type"] == "MatchedHeadTuningError"
    assert failure["error"] == "injected development failure"
    assert len(failure["attempts"]) == 1
    assert failure["attempts"][0]["status"] == "failed"
    assert failure["trials"] == []


def _prepare_confirmation_inputs(
    producer: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Path, str]:
    _install_raising_final_source(monkeypatch)
    archive, weight, output_dir = _install_tiny_verified_inputs(producer, tmp_path, monkeypatch)
    producer.main()
    confirmation = local_confirmation if producer is local_validation else pretrained_confirmation
    prefix = (
        "benchmark_cifar_v2" if producer is local_validation else "benchmark_cifar_pretrained_v1"
    )
    monkeypatch.setattr(confirmation, "ARTIFACT_DIR", output_dir)
    monkeypatch.setattr(confirmation, "RESULT_PATH", output_dir / f"{prefix}_result_smoke.json")
    monkeypatch.setattr(confirmation, "FAILURE_PATH", output_dir / f"{prefix}_failure_smoke.json")
    if producer is pretrained_validation:
        assert weight is not None
        monkeypatch.setattr(confirmation, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(confirmation, "EXPECTED_ARCHIVE_BYTES", archive.stat().st_size)
        monkeypatch.setattr(
            confirmation, "EXPECTED_ARCHIVE_MD5", md5(archive.read_bytes()).hexdigest()
        )
        monkeypatch.setattr(confirmation, "EXPECTED_WEIGHT_BYTES", weight.stat().st_size)
        monkeypatch.setattr(
            confirmation, "EXPECTED_WEIGHT_SHA256", sha256(weight.read_bytes()).hexdigest()
        )
    return confirmation, output_dir, prefix


@pytest.mark.parametrize("producer", [local_validation, pretrained_validation])
def test_should_write_three_scope_stub_confirmation_after_frozen_selection(
    producer: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    confirmation, output_dir, prefix = _prepare_confirmation_inputs(producer, tmp_path, monkeypatch)
    capsys.readouterr()
    manifest_payload = _read_json(output_dir / f"{prefix}_manifest_smoke.json")
    expected_manifest = restore_confirmation_manifest(manifest_payload)
    seen: list[str] = []

    def stub_confirmation(manifest: RepeatedConfirmationManifest) -> RepeatedConfirmationResult:
        seen.append(manifest.manifest_digest)
        return _synthetic_confirmation_result(manifest)

    monkeypatch.setattr(confirmation, "run_repeated_confirmation", stub_confirmation)
    confirmation.main()

    assert seen == [expected_manifest.manifest_digest]
    assert not confirmation.FAILURE_PATH.exists()
    result = _read_json(confirmation.RESULT_PATH)
    printed = json.loads(capsys.readouterr().out, parse_constant=_reject_nonfinite)
    assert result["protocol_id"] == REPEATED_CONFIRMATION_PROTOCOL
    assert result["manifest"] == manifest_payload
    assert printed["manifest_digest"] == manifest_payload["manifest_digest"]
    assert printed["result_path"] == str(confirmation.RESULT_PATH)
    seeds = manifest_payload["confirmation_seeds"]
    choices = {row["head_name"]: row["candidate_id"] for row in manifest_payload["selected_heads"]}
    fixed = result["fixed_data"]
    assert fixed["seeds"] == seeds
    assert len(fixed["attempts"]) == len(fixed["trials"]) == len(fixed["confirmations"]) == 9
    for kind in ("attempts", "trials", "confirmations"):
        assert {(row["head_name"], row["seed"], row["candidate_id"]) for row in fixed[kind]} == {
            (head, seed, choices[head]) for head in HEAD_NAMES for seed in seeds
        }
    assert [row["seed"] for row in result["wall_time"]] == seeds
    assert [row["seed"] for row in result["capacity_memory"]] == seeds
    for wall, memory in zip(result["wall_time"], result["capacity_memory"], strict=True):
        assert wall["wall_time_budget_seconds"] == manifest_payload["wall_time_budget_seconds"]
        assert set(wall["reports"]) == set(memory["reports"]) == set(HEAD_NAMES)
    assert (
        len(
            {
                report["pid"]
                for item in result["capacity_memory"]
                for report in item["reports"].values()
            }
        )
        == 9
    )
    for scope in ("fixed_data_accuracy", "wall_time_accuracy", "observed_train_rss"):
        assert set(result[scope]) == set(HEAD_NAMES)
        assert all(
            item["seeds"] == seeds and len(item["values"]) == 3 for item in result[scope].values()
        )


@pytest.mark.parametrize(
    ("confirmation", "occupied"),
    [
        (module, kind)
        for module in (local_confirmation, pretrained_confirmation)
        for kind in ("result", "failure")
    ],
)
def test_should_preflight_confirmation_output_before_inputs(
    confirmation: Any, occupied: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(confirmation, "RESULT_PATH", tmp_path / "result.json")
    monkeypatch.setattr(confirmation, "FAILURE_PATH", tmp_path / "failure.json")
    path = getattr(confirmation, f"{occupied.upper()}_PATH")
    path.write_bytes(b"previous user content")
    monkeypatch.setattr(
        confirmation,
        "_read_json",
        lambda name: pytest.fail("opened inputs before output preflight"),
    )
    with pytest.raises(FileExistsError, match="already has"):
        confirmation.main()
    assert path.read_bytes() == b"previous user content"


@pytest.mark.parametrize("producer", [local_validation, pretrained_validation])
def test_should_record_confirmation_failure_without_false_result(
    producer: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    confirmation, output_dir, prefix = _prepare_confirmation_inputs(producer, tmp_path, monkeypatch)
    manifest = _read_json(output_dir / f"{prefix}_manifest_smoke.json")

    def fail_confirmation(observed: RepeatedConfirmationManifest) -> Any:
        assert observed.manifest_digest == manifest["manifest_digest"]
        raise RuntimeError("injected confirmation interruption")

    monkeypatch.setattr(confirmation, "run_repeated_confirmation", fail_confirmation)
    with pytest.raises(RuntimeError, match="injected confirmation interruption"):
        confirmation.main()

    assert not confirmation.RESULT_PATH.exists()
    failure = _read_json(confirmation.FAILURE_PATH)
    assert failure["manifest_digest"] == manifest["manifest_digest"]
    assert failure["error_type"] == "RuntimeError"
    assert failure["error"] == "injected confirmation interruption"
    assert failure["elapsed_seconds"] >= 0
    assert (output_dir / f"{prefix}_selection_smoke.json").exists()


@pytest.mark.parametrize("producer", [local_validation, pretrained_validation])
def test_should_reject_changed_frozen_manifest_before_confirmation(
    producer: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    confirmation, output_dir, prefix = _prepare_confirmation_inputs(producer, tmp_path, monkeypatch)
    manifest_path = output_dir / f"{prefix}_manifest_smoke.json"
    payload = _read_json(manifest_path)
    payload["manifest_digest"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        confirmation,
        "run_repeated_confirmation",
        lambda manifest: pytest.fail("confirmation opened before manifest preflight"),
    )

    with pytest.raises(ValueError, match="manifest changed"):
        confirmation.main()

    assert not confirmation.RESULT_PATH.exists()
    assert not confirmation.FAILURE_PATH.exists()


@pytest.mark.parametrize(
    ("producer", "changed"),
    [
        (module, kind)
        for module in (local_validation, pretrained_validation)
        for kind in ("selection", "archive")
    ]
    + [(pretrained_validation, "weight")],
)
def test_should_reject_changed_selection_or_local_source_before_final_access(
    producer: Any, changed: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    confirmation, output_dir, prefix = _prepare_confirmation_inputs(producer, tmp_path, monkeypatch)
    request = _read_json(output_dir / f"{prefix}_request_smoke.json")
    if changed == "selection":
        selection_path = output_dir / f"{prefix}_selection_smoke.json"
        payload = _read_json(selection_path)
        payload["trials"][0]["validation_accuracy"] += 0.01
        selection_path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        source = Path(request["archive" if changed == "archive" else "weight_file"])
        contents = source.read_bytes()
        source.write_bytes(bytes([contents[0] ^ 1]) + contents[1:])
    monkeypatch.setattr(
        confirmation,
        "run_repeated_confirmation",
        lambda manifest: pytest.fail("final access started before provenance preflight"),
    )

    with pytest.raises(ValueError, match="changed|provenance|disagree"):
        confirmation.main()

    assert not confirmation.RESULT_PATH.exists()
    assert not confirmation.FAILURE_PATH.exists()
