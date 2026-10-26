"""Allocation is an opt-in train-only schedule, with unchanged v9 defaults."""

from dataclasses import replace
import json
import sys

import pytest

import scripts.run_continual_matched_replay_schedule_smoke as cli
from src.app import continual_arrived_benchmark as arrived
from src.app.continual_checkpoint import continual_config_digest
from src.app.continual_matched_replay_schedule import (
    MATCHED_REPLAY_SCHEDULE_PROTOCOL,
    PARTITIONED_REPLAY_SCHEDULE_PROTOCOL,
    MatchedReplayScheduleSession,
)
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.shared_replay_schedule import ReplayAllocation, SharedReplayBuffer


def allocation(recent=2):
    return ReplayAllocation(
        ReplayRetentionBudget(recent, recent * 24),
        ReplayRetentionBudget(4 - recent, (4 - recent) * 24),
    )


def manifest():
    return cli._manifest(ReplayRetentionPolicy("seeded_reservoir", 53))


def forbidden(*args, **kwargs):
    pytest.fail("unavailable source or route opened")


def test_should_preserve_v9_default_digest_and_schema():
    config = manifest()
    session = MatchedReplayScheduleSession(config, seed=17)
    assert session.protocol_id == MATCHED_REPLAY_SCHEDULE_PROTOCOL
    assert session.manifest_digest == continual_config_digest(config, config.seeds)
    assert session.allocation is session.partition_retention is None


def test_should_bind_partition_budgets_and_own_config_before_sources_change():
    config = manifest()
    first = allocation()
    session = MatchedReplayScheduleSession(config, seed=17, allocation=first)
    alternate = MatchedReplayScheduleSession(config, seed=17, allocation=allocation(1))
    assert session.protocol_id == PARTITIONED_REPLAY_SCHEDULE_PROTOCOL
    assert session.manifest_digest != continual_config_digest(config, config.seeds)
    assert session.manifest_digest != alternate.manifest_digest
    assert session.allocation == first
    object.__setattr__(first.recent, "max_examples", 99)
    assert session.allocation == allocation()
    assert session.complete_wake_epoch(config, source_role="train") is not None


def test_should_refuse_invalid_allocation_before_opening_training_source(monkeypatch):
    monkeypatch.setattr(arrived, "_build_phase_a_roles", forbidden)
    config = cli._manifest(ReplayRetentionPolicy("recent_fifo"))
    with pytest.raises(ValueError):
        MatchedReplayScheduleSession(config, seed=17, allocation=allocation())


def test_should_consume_only_arrived_train_and_propose_same_partitioned_rows_to_every_method(
    monkeypatch,
):
    original_a, original_b = arrived._build_phase_a_roles, arrived._build_phase_b_roles
    arrivals = []

    class SealedDecision:
        def __getattribute__(self, name):
            raise AssertionError("decision/final role was opened")

    def seal(roles):
        return replace(roles, inner_guard=SealedDecision(), outer_selection=SealedDecision())

    monkeypatch.setattr(arrived, "_build_phase_a_roles", lambda *a: seal(original_a(*a)))

    def build_b(*args):
        arrivals.append("b")
        return seal(original_b(*args))

    monkeypatch.setattr(arrived, "_build_phase_b_roles", build_b)
    monkeypatch.setattr(SharedReplayBuffer, "select_recent", forbidden)
    config = manifest()
    session = MatchedReplayScheduleSession(config, seed=17, allocation=allocation())
    assert arrivals == []
    with pytest.raises(ValueError, match="before Phase A"):
        session.arrive_phase_b(config)
    for phase in ("a", "b"):
        if phase == "b":
            session.arrive_phase_b(config)
        for _ in range(2):
            before = session.retention
            with pytest.raises(ValueError, match="source role"):
                session.complete_wake_epoch(config, source_role="inner_guard")
            assert session.retention == before
            boundary = session.complete_wake_epoch(config, source_role="train")
            assert boundary is not None
            partitions = session.partition_retention
            assert partitions is not None and partitions.unique == boundary.retention
            assert partitions.recent.example_count <= 2 and partitions.reservoir.example_count <= 2
            assert partitions.unique.retained_bytes <= 96
            assert len(boundary.selected_ids) == 2
            assert all(w.sample_ids == boundary.selected_ids for w in boundary.method_work)
            assert all(w.planned_optimizer_updates == 2 for w in boundary.method_work)
            with pytest.raises(ValueError, match="manifest"):
                session.complete_wake_epoch(replace(config, seeds=(17,)), source_role="train")
    assert arrivals == ["b"]


@pytest.mark.parametrize("partitioned", (False, True))
def test_should_keep_default_cli_payload_and_route_new_mode_explicitly(
    monkeypatch, tmp_path, partitioned
):
    calls = []

    class Session:
        phase = "a"
        manifest_digest = "stub-digest"
        protocol_id = PARTITIONED_REPLAY_SCHEDULE_PROTOCOL

        def __init__(self, config, *, seed, **options):
            calls.append((config.policy.name, seed, options))

        def arrive_phase_b(self, config):
            self.phase = "b"

        def complete_wake_epoch(self, config, *, source_role):
            assert source_role == "train"
            return None

    monkeypatch.setattr(cli, "MatchedReplayScheduleSession", Session)
    path = tmp_path / "schedule.json"
    args = ["schedule", "--result", str(path)] + (["--partitioned"] if partitioned else [])
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    payload = json.loads(path.read_text())
    assert len(calls) == len(payload["rows"]) == (2 if partitioned else 4)
    assert payload["protocol_id"] == (
        PARTITIONED_REPLAY_SCHEDULE_PROTOCOL if partitioned else MATCHED_REPLAY_SCHEDULE_PROTOCOL
    )
    assert all(("allocation" in options) == partitioned for _, _, options in calls)
    assert all(("allocation" in row) == partitioned for row in payload["rows"])
    assert all(row["boundaries"] == [] for row in payload["rows"])


def test_should_refuse_occupied_cli_output_before_opening_sources(monkeypatch, tmp_path):
    path = tmp_path / "occupied.json"
    path.write_text("preserved")
    monkeypatch.setattr(cli, "_manifest", forbidden)
    monkeypatch.setattr(cli, "MatchedReplayScheduleSession", forbidden)
    monkeypatch.setattr(sys, "argv", ["schedule", "--result", str(path), "--partitioned"])
    with pytest.raises(FileExistsError):
        cli.main()
    assert path.read_text() == "preserved"
