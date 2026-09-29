"""Small CPU checks for equal-trial matched-head validation selection."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.app import matched_head_tuning as tuning  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402


def _base_config() -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        train_samples=4,
        guard_samples=4,
        validation_samples=4,
        test_samples=4,
        num_classes=3,
        image_size=32,
        batch_size=4,
        epochs=1,
        seed=47,
        device="cpu",
        target_accuracy=None,
        backprop_freeze_backbone=True,
        backbone_weights="none",
        predictive_head_hidden_dim=8,
        circadian_head_hidden_dim=8,
        circadian_min_hidden_dim=8,
        circadian_max_hidden_dim=8,
        predictive_inference_steps=1,
        circadian_inference_steps=1,
        circadian_sleep_interval=0,
        circadian_use_adaptive_sleep_trigger=False,
    )


def _candidates(
    base: ResNet50BenchmarkConfig,
) -> dict[str, tuple[tuning.HeadTuningCandidate, ...]]:
    changes = {
        "backprop_mlp": "backprop_learning_rate",
        "predictive_coding": "predictive_learning_rate",
        "circadian_predictive_coding": "circadian_learning_rate",
    }
    result: dict[str, tuple[tuning.HeadTuningCandidate, ...]] = {}
    for head_name, field_name in changes.items():
        result[head_name] = (
            tuning.HeadTuningCandidate("a", base),
            tuning.HeadTuningCandidate(
                "b",
                replace(base, **{field_name: getattr(base, field_name) * 0.8}),
            ),
        )
    return result


def _install_sealed_loaders(
    monkeypatch: pytest.MonkeyPatch,
    test_labels: tuple[int, ...],
) -> dict[str, Any]:
    state: dict[str, Any] = {"selected": False, "test_reads": 0, "test_iterations": 0}
    features = torch.tensor(
        [
            [0.2, 0.1, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.3, 0.2, 0.4, 0.5, 0.1],
            [0.1, 0.5, 0.2, 0.3, 0.4],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    role_loader = [(features, labels)]

    class TestLoader:
        def __iter__(self) -> Any:
            assert state["selected"], "Final-test labels opened before selection"
            state["test_iterations"] += 1
            return iter([(features, torch.tensor(test_labels, dtype=torch.long))])

    class Loaders:
        train_loader = role_loader
        guard_loader = list(role_loader)
        validation_loader = list(role_loader)
        num_classes = 3
        split_hashes = {role: role for role in ("train", "guard", "validation", "test")}

        @property
        def test_loader(self) -> TestLoader:
            assert state["selected"], "Final-test loader read before selection"
            state["test_reads"] += 1
            return TestLoader()

    monkeypatch.setattr(tuning.matched, "_build_benchmark_loaders", lambda config: Loaders())
    monkeypatch.setattr(
        tuning.matched,
        "_build_resnet50_backbone",
        lambda **kwargs: (torch.nn.Identity(), 5),
    )
    select = tuning._select_validation_candidates

    def select_and_freeze(*args: Any) -> tuple[tuning.HeadTuningSelection, ...]:
        selections = select(*args)
        state["selected"] = True
        return selections

    monkeypatch.setattr(tuning, "_select_validation_candidates", select_and_freeze)
    return state


def test_should_build_a_development_bank_without_requesting_final_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = torch.zeros((2, 5))
    labels = torch.tensor([0, 1], dtype=torch.long)
    roles = {name: object() for name in ("train", "guard", "validation")}
    requested: list[bool] = []

    class DevelopmentLoaders:
        train_loader = roles["train"]
        guard_loader = roles["guard"]
        validation_loader = roles["validation"]
        split_hashes = {name: name for name in roles}
        num_classes = 2

        @property
        def test_loader(self) -> Any:
            raise AssertionError("development bank requested final loader")

    def build_loaders(config: Any, *, include_final_test: bool = True) -> Any:
        del config
        requested.append(include_final_test)
        return DevelopmentLoaders()

    monkeypatch.setattr(tuning.matched, "_build_benchmark_loaders", build_loaders)
    monkeypatch.setattr(
        tuning.matched,
        "_build_resnet50_backbone",
        lambda **kwargs: (torch.nn.Identity(), 5),
    )
    monkeypatch.setattr(
        tuning.matched,
        "_materialize_role_features",
        lambda *args: ((features, labels),),
    )

    bank = tuning._build_seed_bank(
        torch, torch.device("cpu"), _base_config(), include_final_test=False
    )

    assert requested == [False]
    assert set(bank.split_hashes) == set(roles)


def test_should_select_cifar_candidates_without_constructing_final_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image
    from types import SimpleNamespace

    from src.infra import vision_datasets

    final_constructions = 0

    class FakeCifar:
        def __init__(self, root: str, train: bool, download: bool, transform: Any) -> None:
            nonlocal final_constructions
            del root, download
            if not train:
                final_constructions += 1
                raise AssertionError("selection constructed the CIFAR final source")
            self.transform = transform

        def __len__(self) -> int:
            return 30

        def __getitem__(self, index: int) -> tuple[Any, int]:
            return self.transform(Image.new("RGB", (32, 32), color=(index, 0, 0))), index % 10

    monkeypatch.setattr(
        vision_datasets,
        "require_torchvision_datasets",
        lambda: SimpleNamespace(CIFAR10=FakeCifar, CIFAR100=FakeCifar),
    )
    monkeypatch.setattr(
        tuning.matched,
        "_build_resnet50_backbone",
        lambda **kwargs: (
            torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 5)),
            5,
        ),
    )
    config = replace(
        _base_config(),
        dataset_name="cifar10",
        num_classes=10,
        dataset_download=False,
        dataset_train_subset_size=8,
        dataset_guard_subset_size=4,
        dataset_validation_subset_size=4,
        dataset_test_subset_size=4,
    )

    observed: list[tuple[str, bool]] = []
    result = tuning.run_matched_head_tuning(
        config,
        _candidates(config),
        seeds=(47,),
        candidates_per_head=2,
        confirm_test=False,
        development_only_source=True,
        attempt_observer=lambda attempt, trial: observed.append(
            (attempt.status, trial is not None)
        ),
    )

    assert final_constructions == 0
    assert len(result.attempts) == len(result.trials) == 6
    assert len(result.selections) == 3
    assert not result.confirmations
    assert observed == [("started", False), ("complete", True)] * 6
    assert all(
        set(trial.split_hashes) == {"train", "guard", "validation"} for trial in result.trials
    )


def test_equal_trial_ledger_seals_test_until_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_sealed_loaders(monkeypatch, (0, 1, 2, 1))
    base = _base_config()
    candidates = _candidates(base)
    result = tuning.run_matched_head_tuning(
        base,
        candidates,
        seeds=(47,),
        candidates_per_head=2,
    )

    assert result.protocol_id == "vision_matched_head_equal_trial_tuning_v1"
    assert result.candidates_per_head == 2
    assert result.trials_per_head == 2
    assert len(result.attempts) == 6
    assert all(
        attempt.status == "complete" and attempt.error is None for attempt in result.attempts
    )
    assert len(result.trials) == 6
    assert len(result.selections) == len(result.confirmations) == 3
    assert state == {"selected": True, "test_reads": 1, "test_iterations": 1}
    assert len({trial.initial_head_hash for trial in result.trials}) == 1
    for head_name in tuning.HEAD_NAMES:
        rows = [trial for trial in result.trials if trial.head_name == head_name]
        assert {trial.candidate_id for trial in rows} == {"a", "b"}
        assert {trial.seed for trial in rows} == {47}
        assert all(trial.guard_examples_scored == 4 for trial in rows)
        assert all(trial.validation_examples_scored == 4 for trial in rows)
        assert all(trial.wake_batches == 1 for trial in rows)
        assert all(trial.split_hashes.keys() == {"train", "guard", "validation"} for trial in rows)
        assert all(
            trial.feature_hashes.keys() == {"train", "guard", "validation"} for trial in rows
        )
        assert all("test_accuracy" not in asdict(trial) for trial in rows)
        assert all(
            trial.config == candidates[head_name][index].config for index, trial in enumerate(rows)
        )
        selected = next(item for item in result.selections if item.head_name == head_name)
        confirmed = next(item for item in result.confirmations if item.head_name == head_name)
        assert selected.trial_count == 2
        assert confirmed.candidate_id == selected.candidate_id
        assert confirmed.trained_head_hash == next(
            row.trained_head_hash for row in rows if row.candidate_id == selected.candidate_id
        )

    # Ties resolve by predeclared order, without consulting confirmation rows.
    tied = tuple(replace(row, validation_accuracy=0.5) for row in result.trials)
    tie_selection = tuning._select_validation_candidates(tied, candidates, (47,), 2)
    assert [item.candidate_id for item in tie_selection] == ["a", "a", "a"]


def test_validation_selection_mode_never_opens_final_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_sealed_loaders(monkeypatch, (0, 1, 2, 1))
    base = _base_config()
    result = tuning.run_matched_head_tuning(
        base,
        _candidates(base),
        seeds=(47,),
        candidates_per_head=2,
        confirm_test=False,
    )
    assert result.protocol_id == "vision_matched_head_validation_selection_v1"
    assert len(result.trials) == 6
    assert len(result.selections) == 3
    assert result.confirmations == ()
    assert state == {"selected": True, "test_reads": 0, "test_iterations": 0}


def test_test_label_changes_leave_trial_ledger_and_selection_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_config()
    candidates = _candidates(base)
    first_state = _install_sealed_loaders(monkeypatch, (0, 1, 2, 1))
    first = tuning.run_matched_head_tuning(
        base,
        candidates,
        seeds=(47,),
        candidates_per_head=2,
    )
    assert first_state["test_iterations"] == 1
    # Undo the selection wrapper before installing a second sealed fixture.
    monkeypatch.undo()
    second_state = _install_sealed_loaders(monkeypatch, (2, 2, 2, 2))
    second = tuning.run_matched_head_tuning(
        base,
        candidates,
        seeds=(47,),
        candidates_per_head=2,
    )
    assert second_state["test_iterations"] == 1
    for left, right in zip(first.trials, second.trials, strict=True):
        assert replace(left, train_seconds=0.0) == replace(right, train_seconds=0.0)
    assert first.selections == second.selections
    assert first.confirmations[0].test_feature_hash != second.confirmations[0].test_feature_hash


def test_candidate_order_keeps_each_seeded_trial_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_config()
    candidates = _candidates(base)
    _install_sealed_loaders(monkeypatch, (0, 1, 2, 1))
    first = tuning.run_matched_head_tuning(
        base,
        candidates,
        seeds=(47,),
        candidates_per_head=2,
    )
    monkeypatch.undo()
    _install_sealed_loaders(monkeypatch, (0, 1, 2, 1))
    reversed_candidates = {
        head_name: tuple(reversed(options)) for head_name, options in candidates.items()
    }
    second = tuning.run_matched_head_tuning(
        base,
        reversed_candidates,
        seeds=(47,),
        candidates_per_head=2,
    )
    first_rows = {
        (row.head_name, row.candidate_id): replace(row, train_seconds=0.0) for row in first.trials
    }
    second_rows = {
        (row.head_name, row.candidate_id): replace(row, train_seconds=0.0) for row in second.trials
    }
    assert first_rows == second_rows


def test_failed_candidate_retains_attempt_ledger_without_opening_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_sealed_loaders(monkeypatch, (0, 1, 2, 1))
    base = _base_config()
    original_train = tuning.matched._train_predictive_head

    def fail_second_candidate(*args: Any) -> Any:
        config = args[6]
        if config.predictive_learning_rate < base.predictive_learning_rate:
            raise RuntimeError("injected candidate failure")
        return original_train(*args)

    monkeypatch.setattr(tuning.matched, "_train_predictive_head", fail_second_candidate)
    with pytest.raises(tuning.MatchedHeadTuningError) as caught:
        tuning.run_matched_head_tuning(
            base,
            _candidates(base),
            seeds=(47,),
            candidates_per_head=2,
        )
    error = caught.value
    assert [(item.head_name, item.candidate_id, item.status) for item in error.attempts] == [
        ("backprop_mlp", "a", "complete"),
        ("backprop_mlp", "b", "complete"),
        ("predictive_coding", "a", "complete"),
        ("predictive_coding", "b", "failed"),
    ]
    assert error.attempts[-1].seed == 47
    assert error.attempts[-1].config.predictive_learning_rate < base.predictive_learning_rate
    assert "injected candidate failure" in (error.attempts[-1].error or "")
    assert len(error.trials) == 3
    assert error.selections == ()
    assert state == {"selected": False, "test_reads": 0, "test_iterations": 0}


@pytest.mark.parametrize("invalid", ["unequal", "fixed_field", "duplicate", "oversized"])
def test_tuning_rejects_unfair_requests_before_data_access(
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    base = _base_config()
    candidates = _candidates(base)
    seeds: tuple[int, ...] = (47,)
    if invalid == "unequal":
        candidates["backprop_mlp"] = candidates["backprop_mlp"][:1]
    elif invalid == "fixed_field":
        candidates["predictive_coding"] = (
            candidates["predictive_coding"][0],
            tuning.HeadTuningCandidate("b", replace(base, test_samples=8)),
        )
    elif invalid == "duplicate":
        candidates["circadian_predictive_coding"] = (
            candidates["circadian_predictive_coding"][0],
            tuning.HeadTuningCandidate("b", base),
        )
    else:
        seeds = tuple(range(9))
    monkeypatch.setattr(
        tuning.matched,
        "_build_benchmark_loaders",
        lambda config: pytest.fail("Invalid tuning request reached dataset loading"),
    )
    with pytest.raises(ValueError):
        tuning.run_matched_head_tuning(
            base,
            candidates,
            seeds=seeds,
            candidates_per_head=2,
        )
