"""Cross-runner typed sleep artifacts keep distinct attempts and retained work."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from src.app.continual_shift_benchmark import (
    CONTINUAL_VALIDATION_PROTOCOL,
    ContinualShiftConfig,
    ContinualGlobalSealConfig,
    run_continual_shift_benchmark,
)
from src.app import continual_arrived_benchmark as arrived
from src.app import continual_arrived_selection as selection
from src.app import continual_shift_benchmark as continual_base
from src.app.experiment_runner import ExperimentConfig, run_experiment
from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.sleep_telemetry import SleepEventTelemetry, SleepGuardMetrics
from src.infra.circadian_checkpoint_files import (
    TrustedLocalContinualCheckpointStore,
    TrustedLocalArrivedCheckpointStore,
    TrustedLocalArrivedSelectionCheckpointStore,
    TrustedLocalToyCheckpointStore,
)
from src.infra.local_result_json import write_local_result_json
from src.infra.toy_result_files import write_toy_result_json


class _StopAtAuditCursor(Exception):
    pass


class _InterruptingStore:
    def __init__(self, store: Any, stop_when: Callable[[Any], bool]) -> None:
        self.store = store
        self.stop_when = stop_when

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        if self.stop_when(checkpoint):
            raise _StopAtAuditCursor()


def _semantic_events(events: tuple[SleepEventTelemetry, ...]) -> list[dict[str, Any]]:
    records = [asdict(event) for event in events]
    for record in records:
        record.pop("durations")
    return records


def _audit_json_events(
    events: tuple[SleepEventTelemetry, ...], json_events: list[dict[str, Any]]
) -> dict[str, int]:
    """Audit the serialized schema and separate attempts from retained work."""
    assert len(events) == len(json_events)
    due_triggers = {
        "forced",
        "periodic",
        "adaptive",
        "periodic_and_adaptive",
        "budget_skipped",
    }
    event_fields = {field.name for field in fields(SleepEventTelemetry)}
    guard_fields = {field.name for field in fields(SleepGuardMetrics)}
    for event, record in zip(events, json_events, strict=True):
        assert type(event) is SleepEventTelemetry
        assert set(record) == event_fields
        assert record == json.loads(json.dumps(asdict(event), allow_nan=False))
        if event.guard is not None:
            assert set(record["guard"]) == guard_fields
        assert event.chemistry_before.count == event.before_width
        assert event.chemistry_proposed.count == event.proposed_width
        assert event.chemistry_final.count == event.final_width
        if event.outcome in {"rolled_back", "error"}:
            assert event.changes.applied_split_pairs == ()
            assert event.changes.applied_removed_prune_ids == ()
            assert event.replay.applied_updates == 0
    return {
        "attempts": sum(event.trigger_reason in due_triggers for event in events),
        "resolved": sum(event.outcome != "error" for event in events),
        "core_events": sum(event.outcome in {"applied", "accepted"} for event in events),
        "rejections": sum(event.outcome == "rolled_back" for event in events),
        "retained_splits": sum(len(event.changes.applied_split_pairs) for event in events),
        "retained_prunes": sum(len(event.changes.applied_removed_prune_ids) for event in events),
        "proposed_splits": sum(len(event.changes.proposed_split_pairs) for event in events),
    }


def _load_strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite {value}")),
    )


def test_toy_json_and_resumed_history_reconcile_legacy_counts(tmp_path: Path) -> None:
    config = ExperimentConfig(
        sample_count=80,
        hidden_dim=4,
        epoch_count=4,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval=2,
        random_seed=29,
        circadian_config=CircadianConfig(
            split_threshold=0.0,
            max_split_per_sleep=1,
            max_prune_per_sleep=0,
            replay_steps=2,
            replay_memory_size=4,
        ),
    )
    ordinary = run_experiment(config)
    store = TrustedLocalToyCheckpointStore(tmp_path / "toy.ckpt")
    with pytest.raises(_StopAtAuditCursor):
        run_experiment(
            config,
            checkpoint_store=_InterruptingStore(
                store,
                lambda checkpoint: (
                    checkpoint.combined.position.stage == "after_sleep"
                    and checkpoint.combined.position.completed_epoch == 2
                ),
            ),
        )
    assert len(store.load().sleep_events) == 2
    resumed = run_experiment(config, checkpoint_store=store, resume_from_checkpoint=True)
    events = resumed.circadian_sleep.events
    assert _semantic_events(events) == _semantic_events(ordinary.circadian_sleep.events)
    assert events == store.load().sleep_events
    output = tmp_path / "toy.json"
    write_toy_result_json(resumed, output)
    counts = _audit_json_events(events, _load_strict_json(output)["circadian_sleep"]["events"])
    assert counts["attempts"] == 2
    assert counts["resolved"] == config.epoch_count
    assert resumed.circadian_sleep.event_count == sum(
        bool(event.changes.applied_split_pairs or event.changes.applied_removed_prune_ids)
        for event in events
    )
    assert resumed.circadian_sleep.total_splits == counts["retained_splits"]
    assert resumed.circadian_sleep.total_prunes == counts["retained_prunes"]
    assert counts["attempts"] != resumed.circadian_sleep.event_count


def test_component_replay_only_artifact_counts_performed_without_retained_topology(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(
        sample_count=80,
        hidden_dim=4,
        epoch_count=2,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval=1,
        random_seed=31,
        circadian_config=CircadianConfig(
            sleep_mode="components",
            sleep_enable_chemical_reset=False,
            sleep_enable_replay=True,
            sleep_enable_homeostasis=False,
            sleep_enable_split=False,
            sleep_enable_prune=False,
            max_split_per_sleep=0,
            max_prune_per_sleep=0,
            replay_steps=1,
            replay_memory_size=4,
        ),
    )
    result = run_experiment(config)
    output = tmp_path / "replay-only.json"
    write_toy_result_json(result, output)
    counts = _audit_json_events(
        result.circadian_sleep.events,
        _load_strict_json(output)["circadian_sleep"]["events"],
    )
    assert counts["attempts"] == 2
    assert counts["core_events"] == result.circadian_sleep.event_count > 0
    assert counts["retained_splits"] == counts["retained_prunes"] == 0
    assert any(event.replay.applied_updates > 0 for event in result.circadian_sleep.events)


def test_continual_json_and_resumed_history_reconcile_legacy_counts(tmp_path: Path) -> None:
    config = ContinualShiftConfig(
        protocol_id=CONTINUAL_VALIDATION_PROTOCOL,
        sample_count_phase_a=80,
        sample_count_phase_b=80,
        phase_b_train_fraction=0.5,
        hidden_dim=4,
        phase_a_epochs=2,
        phase_b_epochs=2,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval_phase_a=2,
        circadian_sleep_interval_phase_b=1,
        circadian_config=CircadianConfig(
            split_threshold=0.0,
            max_split_per_sleep=1,
            max_prune_per_sleep=0,
            replay_steps=2,
            replay_memory_size=4,
        ),
    )
    ordinary = run_continual_shift_benchmark(config, [13])
    store = TrustedLocalContinualCheckpointStore(tmp_path / "continual.ckpt")
    with pytest.raises(_StopAtAuditCursor):
        run_continual_shift_benchmark(
            config,
            [13],
            checkpoint_store=_InterruptingStore(
                store,
                lambda checkpoint: (
                    checkpoint.phase == "b"
                    and checkpoint.phase_epoch_completed == 1
                    and checkpoint.combined.position.stage == "before_sleep"
                ),
            ),
        )
    assert len(store.load().sleep_events) == 2
    resumed = run_continual_shift_benchmark(
        config, [13], checkpoint_store=store, resume_from_checkpoint=True
    )
    report = resumed.seed_results[0].circadian_predictive_coding
    ordinary_report = ordinary.seed_results[0].circadian_predictive_coding
    assert _semantic_events(report.sleep_events) == _semantic_events(ordinary_report.sleep_events)
    saved = store.load()
    assert (
        saved.completed_results[0].circadian_predictive_coding.sleep_events == report.sleep_events
    )
    output = tmp_path / "continual.json"
    write_local_result_json(resumed, output)
    counts = _audit_json_events(
        report.sleep_events,
        _load_strict_json(output)["seed_results"][0]["circadian_predictive_coding"]["sleep_events"],
    )
    assert counts["attempts"] == 3
    assert counts["resolved"] == 4
    assert report.sleep_event_count == sum(
        bool(event.changes.applied_split_pairs or event.changes.applied_removed_prune_ids)
        for event in report.sleep_events
    )
    assert report.total_splits == counts["retained_splits"]
    assert report.total_prunes == counts["retained_prunes"]


def test_arrived_continual_json_keeps_rejected_proposals_distinct_from_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = ContinualGlobalSealConfig(
        sample_count_phase_a=40,
        sample_count_phase_b=40,
        hidden_dim=4,
        phase_a_epochs=2,
        phase_b_epochs=2,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval_phase_a=2,
        circadian_sleep_interval_phase_b=2,
        circadian_config=CircadianConfig(
            sleep_mode="components",
            max_split_per_sleep=0,
            max_prune_per_sleep=0,
            replay_steps=1,
            replay_memory_size=1,
        ),
        replay_max_examples=4,
        replay_max_bytes=96,
    )
    config = arrived.ContinualArrivedRolesConfig(training, 0.2, 0.2)
    original_sleep = continual_base._apply_scheduled_sleep

    def reject_guard(**kwargs: Any) -> tuple[int, int, int]:
        if kwargs["epoch_index"] != 2:
            return original_sleep(**kwargs)
        scores = iter((0.8, 0.2))
        with monkeypatch.context() as patch:
            patch.setattr(
                CircadianPredictiveCodingNetwork,
                "compute_accuracy",
                lambda *_: next(scores),
            )
            return original_sleep(**kwargs)

    monkeypatch.setattr(continual_base, "_apply_scheduled_sleep", reject_guard)
    ordinary = arrived.run_continual_arrived_benchmark(config, [17])
    store = TrustedLocalArrivedCheckpointStore(tmp_path / "arrived.ckpt")
    with pytest.raises(_StopAtAuditCursor):
        arrived.run_continual_arrived_benchmark(
            config,
            [17],
            checkpoint_store=_InterruptingStore(
                store,
                lambda checkpoint: (
                    checkpoint.seed_index == 0
                    and checkpoint.phase == "b"
                    and checkpoint.phase_epoch_completed == 2
                    and checkpoint.stage == "before_sleep"
                ),
            ),
        )
    resumed = arrived.run_continual_arrived_benchmark(
        config, [17], checkpoint_store=store, resume_from_checkpoint=True
    )
    seed = resumed.seed_results[0]
    report = seed.metrics.circadian_predictive_coding
    ordinary_events = ordinary.seed_results[0].metrics.circadian_predictive_coding.sleep_events
    assert _semantic_events(report.sleep_events) == _semantic_events(ordinary_events)
    assert store.load().unscored_seeds[0].sleep_events == report.sleep_events
    output = tmp_path / "arrived.json"
    write_local_result_json(resumed, output)
    counts = _audit_json_events(
        report.sleep_events,
        _load_strict_json(output)["seed_results"][0]["metrics"]["circadian_predictive_coding"][
            "sleep_events"
        ],
    )
    assert counts["attempts"] == counts["rejections"] == len(seed.guard_decisions) == 2
    assert counts["core_events"] == counts["retained_splits"] == counts["retained_prunes"] == 0
    assert report.sleep_event_count == report.total_splits == report.total_prunes == 0
    assert all(event.outcome == "rolled_back" for event in report.sleep_events[1::2])
    assert any(event.replay.proposed_updates > 0 for event in report.sleep_events)
    assert all(event.replay.applied_updates == 0 for event in report.sleep_events)


def test_selection_json_and_restart_bind_candidate_trial_and_chosen_histories(
    tmp_path: Path,
) -> None:
    training = ContinualGlobalSealConfig(
        sample_count_phase_a=40,
        sample_count_phase_b=40,
        hidden_dim=4,
        phase_a_epochs=1,
        phase_b_epochs=1,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval_phase_a=1,
        circadian_sleep_interval_phase_b=1,
        circadian_config=CircadianConfig(
            sleep_mode="components",
            max_split_per_sleep=0,
            max_prune_per_sleep=0,
            replay_steps=1,
            replay_memory_size=1,
        ),
        replay_max_examples=4,
        replay_max_bytes=96,
    )
    first = arrived.ContinualArrivedRolesConfig(training, 0.2, 0.2)
    second = replace(
        first,
        training=replace(
            training,
            backprop_learning_rate=training.backprop_learning_rate * 0.7,
            pc_learning_rate=training.pc_learning_rate * 0.7,
            circadian_learning_rate=training.circadian_learning_rate * 0.7,
        ),
    )
    candidates = (
        selection.ArrivedSelectionCandidate("default", first),
        selection.ArrivedSelectionCandidate("lower_rate", second),
    )
    ordinary = selection.run_arrived_outer_selection(candidates, [17])
    store = TrustedLocalArrivedSelectionCheckpointStore(tmp_path / "selection.ckpt")
    with pytest.raises(_StopAtAuditCursor):
        selection.run_arrived_outer_selection(
            candidates,
            [17],
            checkpoint_store=_InterruptingStore(
                store,
                lambda checkpoint: (
                    checkpoint.candidate_index == 1
                    and checkpoint.stage == "training"
                    and checkpoint.active_v6 is None
                ),
            ),
        )
    assert store.load().completed_candidates[0].unscored_seeds[0].sleep_events
    resumed = selection.run_arrived_outer_selection(
        candidates, [17], checkpoint_store=store, resume_from_checkpoint=True
    )
    ordinary_histories = {
        (item.candidate_id, item.seed): _semantic_events(item.sleep_events)
        for item in ordinary.candidate_sleep_histories
    }
    histories = {
        (item.candidate_id, item.seed): item.sleep_events
        for item in resumed.candidate_sleep_histories
    }
    assert {
        key: _semantic_events(events) for key, events in histories.items()
    } == ordinary_histories
    saved_candidates = store.load().completed_candidates
    assert {
        (candidate.candidate_id, seed.seed): seed.sleep_events
        for candidate in saved_candidates
        for seed in candidate.unscored_seeds
    } == histories
    output = tmp_path / "selection.json"
    write_local_result_json(resumed, output)
    payload = _load_strict_json(output)
    assert len(payload["candidate_sleep_histories"]) == len(candidates)
    for row in payload["candidate_sleep_histories"]:
        events = histories[row["candidate_id"], row["seed"]]
        counts = _audit_json_events(events, row["sleep_events"])
        assert counts["attempts"] == counts["resolved"] == 2
    for trial, row in zip(resumed.trials, payload["trials"], strict=True):
        if trial.method == "circadian_predictive_coding":
            events = histories[trial.candidate_id, trial.seed]
            _audit_json_events(events, row["sleep_events"])
            assert trial.sleep_events == events
            assert trial.inner_guard_examples_scored == sum(
                event.guard.examples_scored for event in events if event.guard is not None
            )
            assert trial.sleep_event_count == sum(event.outcome == "accepted" for event in events)
        else:
            assert trial.sleep_events == ()
            assert row["sleep_events"] == []
    chosen = next(
        item for item in resumed.selections if item.method == "circadian_predictive_coding"
    )
    selected_events = resumed.final_seed_results[0].metrics.circadian_predictive_coding.sleep_events
    assert selected_events == histories[chosen.candidate_id, 17]
    _audit_json_events(
        selected_events,
        payload["final_seed_results"][0]["metrics"]["circadian_predictive_coding"]["sleep_events"],
    )


def _small_torch_config() -> Any:
    from src.app.resnet50_benchmark import ResNet50BenchmarkConfig

    return ResNet50BenchmarkConfig(
        train_samples=8,
        guard_samples=8,
        validation_samples=8,
        test_samples=8,
        num_classes=3,
        image_size=32,
        batch_size=4,
        epochs=2,
        seed=73,
        device="cpu",
        target_accuracy=None,
        inference_batches=1,
        warmup_batches=0,
        backprop_freeze_backbone=True,
        backbone_weights="none",
        predictive_head_hidden_dim=4,
        predictive_inference_steps=2,
        circadian_head_hidden_dim=4,
        circadian_min_hidden_dim=4,
        circadian_max_hidden_dim=6,
        circadian_inference_steps=2,
        circadian_sleep_interval=1,
        circadian_force_sleep=True,
        circadian_sleep_warmup_steps=0,
        circadian_sleep_mode="components",
        circadian_use_adaptive_sleep_trigger=False,
        circadian_use_adaptive_sleep_budget=False,
        circadian_use_adaptive_thresholds=False,
        circadian_split_threshold=0.0,
        circadian_split_hysteresis_margin=0.0,
        circadian_split_cooldown_steps=0,
        circadian_sleep_max_change_fraction=1.0,
        circadian_max_prune_per_sleep=0,
        circadian_sleep_enable_prune=False,
        circadian_sleep_enable_homeostasis=False,
        circadian_sleep_enable_chemical_reset=False,
        circadian_sleep_rollback_tolerance=0.0,
    )


@pytest.fixture
def tiny_torch_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src.app import matched_head_benchmark as matched
    from src.core import resnet50_variants

    def build(device: Any, freeze_backbone: bool, backbone_weights: str) -> tuple[Any, int]:
        assert backbone_weights == "none"
        model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(), torch.nn.Linear(3, 16)
        ).to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(not freeze_backbone)
        return model, 16

    monkeypatch.setattr(resnet50_variants, "_build_resnet50_backbone", build)
    monkeypatch.setattr(matched, "_build_resnet50_backbone", build)


def test_fixed_feature_json_and_resumed_rejection_reconcile_attempts_and_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_torch_backbone: None
) -> None:
    from src.app import matched_head_benchmark as matched
    from src.infra.circadian_checkpoint_files import TrustedLocalCircadianCheckpointStore

    config = _small_torch_config()
    original_evaluate = matched._evaluate_head
    guard_calls = 0

    def score(*args: Any, **kwargs: Any) -> tuple[float, float]:
        nonlocal guard_calls
        result = original_evaluate(*args, **kwargs)
        if kwargs.get("on_examples_scored") is not None:
            guard_calls += 1
            return (0.7, 1.0) if guard_calls % 2 == 0 else (0.8, 0.5)
        return result

    monkeypatch.setattr(matched, "_evaluate_head", score)
    ordinary = matched.run_three_head_fixed_feature_benchmark(config)
    guard_calls = 0
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "fixed.ckpt")
    with pytest.raises(_StopAtAuditCursor):
        matched.run_three_head_fixed_feature_benchmark(
            config,
            checkpoint_store=_InterruptingStore(
                store,
                lambda checkpoint: (
                    checkpoint.combined.position.stage == "after_sleep"
                    and checkpoint.combined.position.completed_epoch == 1
                ),
            ),
        )
    assert len(store.load().sleep_events) == 1
    resumed = matched.run_three_head_fixed_feature_benchmark(
        config, checkpoint_store=store, resume_from_checkpoint=True
    )
    assert _semantic_events(resumed.circadian.sleep_events) == _semantic_events(
        ordinary.circadian.sleep_events
    )
    assert store.load().sleep_events == resumed.circadian.sleep_events
    assert resumed.trained_head_hashes == ordinary.trained_head_hashes
    output = tmp_path / "fixed.json"
    write_local_result_json(resumed, output)
    payload = _load_strict_json(output)
    assert payload["backprop"]["sleep_events"] == []
    assert payload["predictive_coding"]["sleep_events"] == []
    counts = _audit_json_events(
        resumed.circadian.sleep_events, payload["circadian"]["sleep_events"]
    )
    assert counts["attempts"] == counts["rejections"] == resumed.circadian.sleep_attempts == 1
    assert counts["resolved"] == 2
    assert counts["core_events"] == 0
    assert counts["proposed_splits"] > 0
    assert counts["retained_splits"] == resumed.circadian.total_splits == 0
    assert counts["retained_prunes"] == resumed.circadian.total_prunes == 0
    assert resumed.circadian.total_rollbacks == counts["rejections"]
    completed_sleep_guard_examples = sum(
        event.guard.examples_scored
        for event in resumed.circadian.sleep_events
        if event.guard is not None and event.outcome != "error"
    )
    # The legacy report also counts each epoch's stopping evaluation.
    assert resumed.circadian.guard_examples_scored == (
        config.guard_samples * resumed.circadian.epochs_ran + completed_sleep_guard_examples
    )


def test_vision_json_and_resumed_rejection_reconcile_attempts_and_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_torch_backbone: None
) -> None:
    from src.app import resnet50_benchmark as vision
    from src.app.resnet50_benchmark import VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL
    from src.infra.circadian_checkpoint_files import TrustedLocalVisionCheckpointStore

    config = replace(_small_torch_config(), protocol_id=VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL)
    original_score = vision._compute_pc_metrics
    guard_calls = 0

    def score(*args: Any, **kwargs: Any) -> tuple[float, float]:
        nonlocal guard_calls
        result = original_score(*args, **kwargs)
        if kwargs.get("on_examples_scored") is not None:
            guard_calls += 1
            return (0.7, 1.0) if guard_calls % 2 == 0 else (0.8, 0.5)
        return result

    monkeypatch.setattr(vision, "_compute_pc_metrics", score)
    ordinary = vision.run_resnet50_benchmark(config)
    guard_calls = 0
    control = vision.run_resnet50_benchmark(
        config,
        checkpoint_store=TrustedLocalVisionCheckpointStore(tmp_path / "vision-control.ckpt"),
    )
    guard_calls = 0
    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    with pytest.raises(_StopAtAuditCursor):
        vision.run_resnet50_benchmark(
            config,
            checkpoint_store=_InterruptingStore(
                store,
                lambda checkpoint: (
                    checkpoint.active_circadian is not None
                    and checkpoint.active_circadian.stage == "after_sleep"
                    and checkpoint.active_circadian.completed_epoch == 1
                ),
            ),
        )
    active = store.load().active_circadian
    assert active is not None
    assert len(active.sleep_events) == 1
    resumed = vision.run_resnet50_benchmark(
        config, checkpoint_store=store, resume_from_checkpoint=True
    )
    assert resumed.trained_model_hashes == control.trained_model_hashes
    ordinary_report = next(
        report for report in ordinary.reports if report.head_type == "circadian_predictive_coding"
    )
    report = next(
        report for report in resumed.reports if report.head_type == "circadian_predictive_coding"
    )
    assert _semantic_events(report.sleep_events) == _semantic_events(ordinary_report.sleep_events)
    assert store.load().completed_outcomes[-1].sleep_events == report.sleep_events
    output = tmp_path / "vision.json"
    write_local_result_json(resumed, output)
    payload = _load_strict_json(output)
    assert all(
        item["sleep_events"] == []
        for item in payload["reports"]
        if item["head_type"] != "circadian_predictive_coding"
    )
    json_report = next(
        item for item in payload["reports"] if item["head_type"] == "circadian_predictive_coding"
    )
    counts = _audit_json_events(report.sleep_events, json_report["sleep_events"])
    assert counts["attempts"] == counts["rejections"] == report.circadian_sleep_attempts == 1
    assert counts["resolved"] == 2
    assert counts["core_events"] == 0
    assert counts["proposed_splits"] > 0
    assert counts["retained_splits"] == report.circadian_total_splits == 0
    assert counts["retained_prunes"] == report.circadian_total_prunes == 0
    assert report.circadian_total_rollbacks == counts["rejections"]
