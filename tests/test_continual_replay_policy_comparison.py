"""Matched arrived-role replay policies keep a global final-test seal."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_replay_policy_comparison as comparison
from src.app.continual_shift_benchmark import ContinualGlobalSealConfig
from src.core.circadian_predictive_coding import CircadianConfig
from src.core.replay_retention import ReplayRetentionPolicy


def _manifest() -> comparison.ReplayPolicyComparisonManifest:
    training = ContinualGlobalSealConfig(
        sample_count_phase_a=40,
        sample_count_phase_b=40,
        hidden_dim=4,
        phase_a_epochs=2,
        phase_b_epochs=2,
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
    roles = arrived.ContinualArrivedRolesConfig(training, 0.2, 0.2)
    return comparison.ReplayPolicyComparisonManifest(
        arrived=roles,
        seeds=(17, 19),
        policies=(
            ReplayRetentionPolicy("recent_fifo"),
            ReplayRetentionPolicy("seeded_reservoir", seed=53),
        ),
    )


def test_should_reject_changed_manifest_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        arrived,
        "generate_two_cluster_dataset_with_transform",
        lambda **_: (_ for _ in ()).throw(AssertionError("source opened")),
    )
    manifest = _manifest()
    with pytest.raises(ValueError, match="unique.*seeds"):
        comparison.run_replay_policy_comparison(replace(manifest, seeds=(17, 17)))
    with pytest.raises(ValueError, match="FIFO.*reservoir"):
        comparison.run_replay_policy_comparison(
            replace(
                manifest,
                policies=(
                    ReplayRetentionPolicy("recent_fifo"),
                    ReplayRetentionPolicy("recent_fifo"),
                ),
            )
        )


def test_should_match_roles_baselines_and_replay_work_with_public_exposure() -> None:
    manifest = _manifest()
    result = comparison.run_replay_policy_comparison(manifest)
    repeated = comparison.run_replay_policy_comparison(manifest)
    assert result == repeated
    assert result.protocol_id == "continual_replay_policy_comparison_v8"
    assert result.manifest == manifest
    assert len(result.policies) == 2

    fifo, reservoir = result.policies
    assert fifo.policy == manifest.policies[0]
    assert reservoir.policy == manifest.policies[1]
    for first, second in zip(fifo.seeds, reservoir.seeds, strict=True):
        assert first.arrived.seed == second.arrived.seed
        assert first.arrived.role_ids == second.arrived.role_ids
        assert first.arrived.role_hashes == second.arrived.role_hashes
        assert first.arrived.metrics.backprop == second.arrived.metrics.backprop
        assert first.arrived.metrics.predictive_coding == second.arrived.metrics.predictive_coding
        assert first.arrived.metrics.training_order == second.arrived.metrics.training_order
        assert first.baseline_state_digests == second.baseline_state_digests
        assert first.exposure.observed_duplicate_occurrences > 0
        assert first.exposure.observed_duplicate_ids
        assert first.exposure.observed_duplicate_ids == second.exposure.observed_duplicate_ids
        assert first.exposure.after_b.replay_updates == second.exposure.after_b.replay_updates
        assert (
            first.arrived.metrics.circadian_predictive_coding.sleep_event_count
            == second.arrived.metrics.circadian_predictive_coding.sleep_event_count
        )
        for item in (first, second):
            retention = item.arrived.metrics.replay_retention
            assert retention.budget_examples == 4
            assert retention.budget_bytes == 96
            assert retention.phase_a.example_count <= 4
            assert retention.phase_b.example_count <= 4
            assert retention.phase_b.retained_bytes <= 96
            assert item.exposure.phase_a.replay_updates > 0
            assert item.exposure.after_b.replay_updates >= item.exposure.phase_a.replay_updates
            observed_ids = set(item.exposure.observed_ids)
            assert set(item.exposure.after_b.exposed_ids) <= observed_ids
            assert set(retention.phase_b.sample_ids) <= observed_ids
            assert all(
                event.event == "global_freeze"
                for event in item.arrived.role_accesses
                if event.role == "final_test"
            )
    assert {item.policy.name for item in result.policies} == {"recent_fifo", "seeded_reservoir"}


def test_should_open_final_sources_only_after_both_policies_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    finished: set[tuple[str, int]] = set()
    reads: list[str] = []
    original_source = arrived.generate_two_cluster_dataset_with_transform
    original_b_source = arrived._generate_phase_b_source
    original_train = arrived._train_arrived_seed

    class SealedSource:
        def __init__(self, source: Any) -> None:
            self.source = source
            self.train_input = source.train_input
            self.train_target = source.train_target

        @property
        def test_input(self) -> Any:
            assert len(finished) == 4
            reads.append("input")
            return self.source.test_input

        @property
        def test_target(self) -> Any:
            assert len(finished) == 4
            reads.append("target")
            return self.source.test_target

    def source(**kwargs: Any) -> Any:
        return SealedSource(original_source(**kwargs))

    def source_b(*args: Any, **kwargs: Any) -> Any:
        return SealedSource(original_b_source(*args, **kwargs))

    def train(*args: Any, **kwargs: Any) -> Any:
        pending = original_train(*args, **kwargs)
        policy = kwargs["retention_policy"]
        finished.add((policy.name, pending.seed))
        return pending

    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", source)
    monkeypatch.setattr(arrived, "_generate_phase_b_source", source_b)
    monkeypatch.setattr(arrived, "_train_arrived_seed", train)
    comparison.run_replay_policy_comparison(manifest)
    assert len(finished) == 4
    assert reads == ["input", "target"] * 8


def test_should_keep_training_and_replay_facts_when_final_labels_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    control = comparison.run_replay_policy_comparison(manifest)
    original_source = arrived.generate_two_cluster_dataset_with_transform

    def changed_final(**kwargs: Any) -> Any:
        source = original_source(**kwargs)
        if kwargs["seed"] != 17:
            return source
        return replace(source, test_target=1.0 - source.test_target)

    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", changed_final)
    changed = comparison.run_replay_policy_comparison(manifest)
    assert changed.manifest_digest == control.manifest_digest
    for before_policy, after_policy in zip(control.policies, changed.policies, strict=True):
        for before, after in zip(before_policy.seeds, after_policy.seeds, strict=True):
            assert before.exposure == after.exposure
            assert before.baseline_state_digests == after.baseline_state_digests
            assert before.arrived.guard_decisions == after.arrived.guard_decisions
            assert before.arrived.metrics.replay_retention == after.arrived.metrics.replay_retention
            for phase in ("a", "b"):
                for role in ("train", "inner_guard", "outer_selection"):
                    key = f"phase_{phase}_{role}"
                    assert before.arrived.role_hashes[key] == after.arrived.role_hashes[key]
            if before.arrived.seed == 17:
                assert (
                    before.arrived.role_hashes["phase_a_final_test"]
                    != after.arrived.role_hashes["phase_a_final_test"]
                )
            else:
                assert before.arrived == after.arrived
