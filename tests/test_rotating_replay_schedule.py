"""Rotation is a separate schedule mode, preserving retained v1 identities."""

import json
import sys
from types import SimpleNamespace

import pytest

import scripts.run_continual_matched_replay_schedule_smoke as cli
from src.app import continual_arrived_benchmark as arrived
from src.app.continual_matched_replay_schedule import (
    PARTITIONED_REPLAY_SCHEDULE_PROTOCOL,
    ROTATING_REPLAY_SAMPLER,
    ROTATING_REPLAY_SCHEDULE_PROTOCOL,
    MatchedReplayScheduleSession,
)
from src.core.circadian_predictive_coding import ReplayRetentionBudget, ReplayRetentionSnapshot
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.shared_replay_schedule import (
    PartitionRetention,
    ReplayAllocation,
    ReplayRotationCursor,
    SharedReplayBuffer,
)


def allocation():
    return ReplayAllocation(ReplayRetentionBudget(2, 48), ReplayRetentionBudget(2, 48))


def manifest():
    return cli._manifest(ReplayRetentionPolicy("seeded_reservoir", 53))


def forbidden(*args, **kwargs):
    pytest.fail("unavailable route or source opened")


@pytest.mark.parametrize("mode", (None, 1, "true", True))
def test_should_refuse_bad_rotation_controls_before_source_access(monkeypatch, mode):
    monkeypatch.setattr(arrived, "_build_phase_a_roles", forbidden)
    with pytest.raises(ValueError, match="rotation"):
        MatchedReplayScheduleSession(manifest(), seed=17, rotate_reservoir=mode)


def test_should_preserve_default_partitioned_identity_and_cursor():
    config = manifest()
    default = MatchedReplayScheduleSession(config, seed=17, allocation=allocation())
    explicit = MatchedReplayScheduleSession(
        config, seed=17, allocation=allocation(), rotate_reservoir=False
    )
    assert default.protocol_id == explicit.protocol_id == PARTITIONED_REPLAY_SCHEDULE_PROTOCOL
    assert default.manifest_digest == explicit.manifest_digest
    # Frozen partitioned-v1 CLI identity, retained inline for portable regression.
    assert (
        default.manifest_digest
        == "6c97db4b8345322a71210cb70d3bfa8f5c596323c0ee9427de08aa346fb848f1"
    )
    assert default.rotation_cursor == explicit.rotation_cursor == ReplayRotationCursor(0, None)
    for _ in range(2):
        a = default.complete_wake_epoch(config, source_role="train")
        b = explicit.complete_wake_epoch(config, source_role="train")
        assert a is not None and b is not None and a.selected_ids == b.selected_ids
    assert default.rotation_cursor == explicit.rotation_cursor == ReplayRotationCursor(0, None)


def test_should_route_rotation_and_preserve_deterministic_scheduling_without_old_selector(
    monkeypatch,
):
    monkeypatch.setattr(SharedReplayBuffer, "select_partitioned", forbidden)
    config = manifest()
    session = MatchedReplayScheduleSession(
        config, seed=17, allocation=allocation(), rotate_reservoir=True
    )
    plain = MatchedReplayScheduleSession(config, seed=17, allocation=allocation())
    assert session.protocol_id == ROTATING_REPLAY_SCHEDULE_PROTOCOL
    assert session.manifest_digest != plain.manifest_digest
    first = session.complete_wake_epoch(config, source_role="train")
    assert first is not None and session.rotation_cursor.reservoir_after is not None
    # Role metadata contains mapping proxies: only the buffer promises copy/
    # pickle continuation. Independently constructed schedules must agree too.
    fork = MatchedReplayScheduleSession(
        config, seed=17, allocation=allocation(), rotate_reservoir=True
    )
    repeated = fork.complete_wake_epoch(config, source_role="train")
    assert repeated is not None and repeated.selected_ids == first.selected_ids
    for phase in ("a", "b", "b"):
        if phase == "b" and session.phase == "a":
            session.arrive_phase_b(config)
            fork.arrive_phase_b(config)
        a = session.complete_wake_epoch(config, source_role="train")
        b = fork.complete_wake_epoch(config, source_role="train")
        assert a is not None and b is not None
        assert a.selected_ids == b.selected_ids
        assert session.rotation_cursor == fork.rotation_cursor
        assert session.partition_retention == fork.partition_retention
        assert all(w.sample_ids == a.selected_ids for w in a.method_work)


def test_should_require_partitioned_cli_mode_before_any_source_or_output(monkeypatch, tmp_path):
    target = tmp_path / "unused.json"
    monkeypatch.setattr(cli, "_manifest", forbidden)
    monkeypatch.setattr(cli, "MatchedReplayScheduleSession", forbidden)
    monkeypatch.setattr(sys, "argv", ["schedule", "--rotate-reservoir", "--result", str(target)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and not target.exists()


def test_should_emit_rotating_sampler_identity_and_post_selection_cursor(monkeypatch, tmp_path):
    calls = []

    class Session:
        manifest_digest = "stub"
        protocol_id = ROTATING_REPLAY_SCHEDULE_PROTOCOL
        rotation_cursor = ReplayRotationCursor(1, "0" * 64)

        def __init__(self, config, *, seed, allocation, rotate_reservoir):
            calls.append(rotate_reservoir)
            self.steps = 0
            self.phase = "a"
            empty = ReplayRetentionSnapshot((), 0, 0)
            self.partition_retention = PartitionRetention(empty, empty, empty, 0)

        def arrive_phase_b(self, config):
            self.phase = "b"

        def complete_wake_epoch(self, config, *, source_role):
            self.steps += 1
            self.rotation_cursor = ReplayRotationCursor(self.steps % 2, f"{self.steps:064x}")
            return SimpleNamespace(
                phase=self.phase,
                epoch=(self.steps - 1) % 2 + 1,
                train_role_hash="stub",
                retention=self.partition_retention.unique,
                retained_order_ids=(),
                selected_ids=(),
                method_work=(),
            )

    monkeypatch.setattr(cli, "MatchedReplayScheduleSession", Session)
    target = tmp_path / "rotating.json"
    monkeypatch.setattr(
        sys, "argv", ["schedule", "--partitioned", "--rotate-reservoir", "--result", str(target)]
    )
    cli.main()
    data = json.loads(target.read_text())
    assert data["protocol_id"] == ROTATING_REPLAY_SCHEDULE_PROTOCOL
    assert calls == [True, True]
    assert all(row["sampling_policy"] == ROTATING_REPLAY_SAMPLER for row in data["rows"])
    assert all(row["protocol_id"] == ROTATING_REPLAY_SCHEDULE_PROTOCOL for row in data["rows"])
    for row in data["rows"]:
        assert [boundary["rotation_cursor"] for boundary in row["boundaries"]] == [
            dict(next_partition=n % 2, reservoir_after=f"{n:064x}") for n in range(1, 5)
        ]
