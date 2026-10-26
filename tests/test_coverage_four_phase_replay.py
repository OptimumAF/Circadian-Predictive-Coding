"""Equal-update coverage/recent/reservoir controls through existing fake native fixtures."""

from copy import deepcopy
import json
import sys

import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.train_acquisition_pilot as pilot
from src.app.stream_rehearsal import (
    CoverageStreamReplayPolicy,
    RecentStreamReplayPolicy,
    StreamReplayPolicy,
)
from test_four_phase_replay_pilot import environment, ReplayLearner
from test_torch_feature_rehearsal_pilot import forbidden
from test_train_acquisition_pilot import Learner


KEYS = ("A-red", "A-blue", "B-red", "B-blue", "C_train-red", "C_train-blue")
CONTROLS = ("coverage", "recent48", "partitioned48")


@pytest.mark.parametrize("control", CONTROLS)
def test_should_describe_matched48_transition_quotas_without_changing_existing16_controls(
    monkeypatch, control
):
    originals = [
        deepcopy(pilot.describe_four_phase_replay_protocol(control=name))
        for name in ("recent", "partitioned")
    ]
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    protocol = pilot.describe_four_phase_replay_protocol(control=control)
    assert originals == [
        pilot.describe_four_phase_replay_protocol(control=name)
        for name in ("recent", "partitioned")
    ]
    assert protocol["replay_control"] == control
    policy = protocol["replay"]["policy"]
    assert policy["every_updates"] == 2 and policy["rows_per_event"] == 8
    assert policy["retention"] == dict(max_examples=48, max_bytes=12096)
    if control == "coverage":
        assert tuple(policy["contexts"]) == KEYS
        assert "seed" not in policy and "allocation" not in policy
    elif control == "recent48":
        assert "seed" not in policy and "allocation" not in policy
    else:
        assert policy["seed"] == 53
        assert policy["allocation"] == dict(
            recent=dict(max_examples=24, max_bytes=6048),
            reservoir=dict(max_examples=24, max_bytes=6048),
        )
    assert protocol["planned_work"]["native_train_calls"] == 384
    assert protocol["planned_work"]["replay_rows"] == 1024


@pytest.mark.parametrize("control", CONTROLS)
def test_should_run_existing_stream_and_final_evaluator_with_equal_work_and_declared_storage(
    monkeypatch, control
):
    captured = environment(monkeypatch)
    report = pilot.run_four_phase_replay_pilot(control=control)
    assert report["status"] == "complete", report["reason"]
    helper = captured["helper"]
    if control == "coverage":
        assert type(helper._policy) is CoverageStreamReplayPolicy
    elif control == "recent48":
        assert type(helper._policy) is RecentStreamReplayPolicy
    else:
        assert type(helper._policy) is StreamReplayPolicy
    assert helper._policy.retention.max_examples == 48
    assert helper._policy.retention.max_bytes == 12096
    assert report["charged_updates"] == captured["wake"].updates_completed == 256
    assert (
        captured["replay_budget"].replay_examples_completed
        == report["replay"]["charged_examples"]
        == 1024
    )
    assert len(Learner.training_owners) == 384
    assert report["partial_stream"]["native_candidate_updates"] == 384
    assert report["partial_stream"]["native_candidate_training_examples"] == 3072
    replay = report["replay"]["rehearsal"]
    assert len(replay["events"]) == replay["committed_updates"] == 128
    assert len(report["replay"]["applied_consolidations"]) == 128
    assert report["replay"]["final_retention"]["example_count"] <= 48
    assert report["replay"]["final_retention"]["retained_bytes"] <= 12096
    if control == "coverage":
        assert replay["control"] == "context_coverage_v1"
        assert replay["coverage_keys"] == list(KEYS)
        sources = {
            batch.episode.episode_id: (index + 1, batch.context)
            for index, batch in enumerate(captured["training"])
        }
        latest = {}
        previous = None
        for event in replay["events"]:
            for batch in captured["training"][event["updates"] - 2 : event["updates"]]:
                latest[batch.context] = batch.episode.episode_id
            eligible = [key for key in KEYS if key in latest]
            assert event["coverage_keys"] == eligible
            start = 0 if previous is None else (KEYS.index(previous) + 1) % len(KEYS)
            selected_key = next(
                KEYS[(start + offset) % len(KEYS)]
                for offset in range(len(KEYS))
                if KEYS[(start + offset) % len(KEYS)] in latest
            )
            assert event["coverage_cursor"] == selected_key
            episode = event["selected_episodes"][0]
            assert episode["context"] == selected_key
            assert episode["episode_id"] == latest[selected_key]
            assert sources[episode["episode_id"]][0] <= event["updates"]
            assert episode["role"] == "train"
            previous = selected_key
    assert len(ReplayLearner.live_sessions) == 40
    assert all(
        slot["outcome"] == "illegal_action" for slot in report["policy_evaluation"]["episodes"]
    )
    assert (
        report["source_unchanged"]
        is report["isolated_source_unchanged"]
        is report["actor_unchanged"]
        is True
    )
    assert report["scientific_advantage_verified"] is False


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_only_explicit_coverage_family_cli_with_frozen_control(
    monkeypatch, capsys, control, describe
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", f"torch-key-door-four-phase-{control}"]
        + (["--describe"] if describe else []),
    )
    calls = []

    def selected(*, control):
        calls.append(control)
        return dict(status="complete", selected=control)

    monkeypatch.setattr(
        cli, "describe_four_phase_replay_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_four_phase_replay_pilot", forbidden if describe else selected)
    monkeypatch.setattr(cli, "run_four_phase_evaluation_pilot", forbidden)
    cli.main()
    assert calls == [control] and json.loads(capsys.readouterr().out)["selected"] == control
