"""Explicit environment instances through the existing fake-native quota recipe."""

from copy import deepcopy
import json
import sys
from typing import TypedDict

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_pilot as shared
import src.adapters.key_door_reward_pilot as pilot
from src.core.key_door_navigation import key_door_layout_record
from test_interaction_quota_pilot import environment
from test_legal_reward_pilot import Learner, Session
from test_torch_feature_rehearsal_pilot import forbidden


ROUTE = dict(condition_on_legal_actions=True, action_remapping=True, fill_interaction_quota=True)


class SeedOptions(TypedDict):
    environment_seed: int
    heldout_environment_seed: int


SEEDS: SeedOptions = dict(environment_seed=17, heldout_environment_seed=6)


def run():
    return pilot.run_pilot(**ROUTE, **SEEDS)


BAD_PAIRS = (
    (None, 6),
    (17, None),
    (6, 6),
    (-1, 6),
    (24, 6),
    (17, -1),
    (17, 24),
    (True, 6),
    (17, False),
    (17.0, 6),
    (17, 6.0),
    ("17", 6),
    (17, np.int64(6)),
)


@pytest.mark.parametrize("training,final", BAD_PAIRS)
@pytest.mark.parametrize("describe", (False, True))
def test_should_refuse_invalid_instance_pairs_before_resource_or_clock_work(
    monkeypatch, training, final, describe
):
    for name in ("monotonic", "ProcessRssSampler", "require_torch", "TorchRewardUpdateLearner"):
        monkeypatch.setattr(pilot, name, forbidden)
    selected = pilot.describe_protocol if describe else pilot.run_pilot
    with pytest.raises(ValueError):
        selected(**ROUTE, environment_seed=training, heldout_environment_seed=final)


@pytest.mark.parametrize(
    "route",
    (
        {},
        {"condition_on_legal_actions": True},
        {"condition_on_legal_actions": True, "action_remapping": True},
    ),
)
@pytest.mark.parametrize("describe", (False, True))
def test_should_refuse_instance_inputs_on_other_routes_before_resources(
    monkeypatch, route, describe
):
    for name in ("monotonic", "ProcessRssSampler", "require_torch", "TorchRewardUpdateLearner"):
        monkeypatch.setattr(pilot, name, forbidden)
    selected = pilot.describe_protocol if describe else pilot.run_pilot
    with pytest.raises(ValueError):
        selected(**route, **SEEDS)


@pytest.mark.parametrize(
    "route",
    (
        {},
        {"condition_on_legal_actions": True},
        {"condition_on_legal_actions": True, "action_remapping": True},
        ROUTE,
    ),
)
def test_should_preserve_legacy_descriptors_when_both_instance_inputs_are_absent(
    monkeypatch, route
):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    implicit = deepcopy(pilot.describe_protocol(**route))
    explicit = pilot.describe_protocol(
        **route, environment_seed=None, heldout_environment_seed=None
    )
    assert implicit == explicit
    assert "environment_seeds" not in implicit
    assert implicit["seed"] == 17 and implicit["sampling_seed"] == 29
    assert implicit["native_defaults"] == explicit["native_defaults"]
    assert pilot.action_remapping_layouts() == pilot.action_remapping_layouts(None, None)


def test_should_describe_catalogue_inputs_without_models_or_changing_optimizer_and_sampler_seeds(
    monkeypatch,
):
    for name in (
        "ProcessRssSampler",
        "require_torch",
        "TorchRewardUpdateLearner",
        "collect_key_door_reward_episode",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    old = deepcopy(pilot.describe_protocol(**ROUTE))
    descriptor = pilot.describe_protocol(**ROUTE, **SEEDS)
    assert descriptor["protocol_id"] == "actual_key_door_seeded_interaction_quota_v1"
    assert descriptor["environment_seeds"] == {"training": 17, "final_only": 6}
    assert descriptor["seed"] == old["seed"] == 17
    assert descriptor["sampling_seed"] == old["sampling_seed"] == 29
    for key in (
        "native_defaults",
        "planned_work",
        "checkpoint_physical_steps",
        "checkpoint_updates",
        "methods",
    ):
        assert descriptor[key] == old[key]
    expected = [
        key_door_layout_record(layout) for layout in pilot.action_remapping_layouts(**SEEDS)
    ]
    assert descriptor["layouts"] == expected
    for layout in descriptor["layouts"][:3]:
        assert (
            layout["rotation"],
            layout["branch_row"],
            layout["swap_keys"],
            layout["appearance"],
        ) == (1, 2, True, 0.0)
    assert descriptor["layouts"][3] == dict(
        layout_id="D", rotation=2, branch_row=2, swap_keys=False, appearance=0.0
    )
    descriptor["environment_seeds"]["training"] = 0
    descriptor["layouts"][0]["rotation"] = 0
    assert pilot.describe_protocol(**ROUTE, **SEEDS)["environment_seeds"] == {
        "training": 17,
        "final_only": 6,
    }
    assert pilot.describe_protocol(**ROUTE) == old


def test_should_keep_legacy_run_dispatch_without_forwarding_empty_seed_keywords(monkeypatch):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "sentinel"}

    monkeypatch.setattr(pilot, "_run_pilot", selected)
    assert pilot.run_pilot(**ROUTE)["status"] == "sentinel"
    assert (
        pilot.run_pilot(**ROUTE, environment_seed=None, heldout_environment_seed=None)["status"]
        == "sentinel"
    )
    assert calls == [dict(warm_start=False, **ROUTE)] * 2


def test_should_use_seeded_layouts_for_own_training_then_final_packets_and_actual_live_episodes(
    monkeypatch,
):
    budgets, finals, _ = environment(monkeypatch, scripted=False)
    declared = {layout.layout_id: layout for layout in pilot.action_remapping_layouts(**SEEDS)}
    collected, live, final_packets, policy_options = [], [], [], []
    original_collect = pilot.collect_key_door_reward_episode
    original_live = shared.run_key_door_episode
    original_final = shared._collect
    original_policy = pilot.TorchRecurrentPolicy

    def collect(source, layout, cue, **options):
        assert layout == declared[layout.layout_id] and layout.layout_id != "D"
        report = original_collect(source, layout, cue, **options)
        collected.append((layout, cue, report))
        return report

    def evaluate(*args, **options):
        assert sum(len(update.episode.episode.transitions) for update in Learner.updates) == 96
        assert options["layout"] == declared[options["context"]]
        live.append(deepcopy(options))
        return original_live(*args, **options)

    def final(*args, **options):
        assert args[1] == declared[args[1].layout_id] and args[3] == "final_test"
        batch = original_final(*args, **options)
        final_packets.append(batch)
        return batch

    def policy(*args, **options):
        policy_options.append((args, options))
        return original_policy(*args, **options)

    monkeypatch.setattr(pilot, "collect_key_door_reward_episode", collect)
    monkeypatch.setattr(shared, "run_key_door_episode", evaluate)
    monkeypatch.setattr(shared, "_collect", final)
    monkeypatch.setattr(pilot, "TorchRecurrentPolicy", policy)
    report = run()
    row = report["runs"][0]
    assert report["status"] == "complete", report
    assert (
        report["protocol"]["environment_seeds"]
        == row["environment_seeds"]
        == {"training": 17, "final_only": 6}
    )
    assert policy_options == [((8, 8, 5, 17), {})]
    assert len(budgets) == 1 and budgets[0].budget.max_training_updates == 16
    assert budgets[0].updates_completed == len(row["training"]) == len(Learner.updates)
    assert 8 <= len(collected) <= 16
    assert sum(item[2]["environment_steps"] for item in collected) == 96
    assert row["stream_result"]["training_examples"] == 96
    assert row["quota_checkpoints"][-1] == dict(
        physical_steps=96, native_updates=len(Learner.updates)
    )
    assert (
        row["physical_quotas"]
        == [dict(maximum_steps=12, attempted_steps=12, completed_steps=12, failed_steps=0)] * 8
    )
    expected_rng = np.random.Generator(np.random.PCG64(29))
    for index, slot in enumerate(row["training"]):
        assert slot["context"] in ("A", "B", "C")
        assert slot["layout"] == key_door_layout_record(declared[slot["context"]])
        assert slot["commit_validated"] is slot["update_committed"] is True
        assert slot["receipt"] is not None and slot["charged_updates_after"] == index + 1
        collection = slot["collection"]
        assert collection["sampler_before"] == expected_rng.bit_generator.state
        expected_rng.random(collection["sampling_draws"])
        assert collection["sampler_after"] == expected_rng.bit_generator.state
    assert row["sampling_state"] == expected_rng.bit_generator.state
    assert len(finals) == len(final_packets) == 8
    assert {batch.context for batch in final_packets} == {
        f"{context}-{color}" for context in ("A", "B", "C", "D") for color in ("red", "blue")
    }
    assert all(batch.role == "final_test" for batch in final_packets)
    assert len(Learner.classifications) == 40 and len(live) == 40
    for index, features in enumerate(Learner.classifications):
        np.testing.assert_array_equal(features, final_packets[index % 8].features)
    assert sum(options["context"] == "D" for options in live) == 10
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert all(update.episode.episode.role == "train" for update in Learner.updates)


@pytest.mark.parametrize("when", ("prediction", "training", "final_packet"))
def test_should_refuse_changed_seed_metadata_and_preserve_paid_work_before_more_final_access(
    monkeypatch, when
):
    budgets, finals, active = environment(monkeypatch, scripted=False)
    changed: list[bool] = []

    def mutate():
        if not changed:
            active["row"]["environment_seeds"]["training"] = 18
            changed.append(True)

    if when == "prediction":
        original_predict = Session.predict

        def predict(self, features):
            result = original_predict(self, features)
            mutate()
            return result

        monkeypatch.setattr(Session, "predict", predict)
    elif when == "training":
        original_train = Learner.train_batch

        def train(self, update, identity):
            result = original_train(self, update, identity)
            mutate()
            return result

        monkeypatch.setattr(Learner, "train_batch", train)
    else:
        original_final = shared._collect

        def final(*args, **options):
            result = original_final(*args, **options)
            mutate()
            return result

        monkeypatch.setattr(shared, "_collect", final)

    report = run()
    row = report["runs"][0]
    assert changed and report["status"] == "incomplete" and row["status"] == "failed"
    assert not Learner.classifications
    if when == "prediction":
        assert budgets[0].updates_completed == len(Learner.updates) == 0
        assert row["training"][0]["collection"]["environment_steps"] == 1
        assert row["physical_quotas"][0]["completed_steps"] == 1
        assert not finals
    elif when == "training":
        assert budgets[0].updates_completed == len(Learner.updates) == 1
        assert row["training"][0]["update_committed"] is True
        assert row["training"][0]["receipt"] is not None
        assert row["physical_quotas"][0]["completed_steps"] > 0
        assert not finals
    else:
        assert budgets[0].updates_completed == len(Learner.updates)
        assert sum(quota["completed_steps"] for quota in row["physical_quotas"]) == 96
        assert len(finals) == 1


def test_should_refuse_seed_metadata_changed_during_last_source_readback_without_erasing_paid_finals(
    monkeypatch,
):
    budgets, finals, active = environment(monkeypatch, scripted=False)
    original = Learner.snapshot_state
    changed: list[bool] = []

    def snapshot(self):
        result = original(self)
        if active.get("row", {}).get("status") == "complete" and not changed:
            active["row"]["environment_seeds"]["final_only"] = 7
            changed.append(True)
        return result

    monkeypatch.setattr(Learner, "snapshot_state", snapshot)
    report = run()
    row = report["runs"][0]
    assert changed and report["status"] == "incomplete" and row["status"] == "failed"
    assert row["quota_readback_error"]
    assert budgets[0].updates_completed == len(Learner.updates)
    assert sum(quota["completed_steps"] for quota in row["physical_quotas"]) == 96
    assert len(finals) == 8 and len(Learner.classifications) == 40
    assert row["stream_result"]["training_examples"] == 96
    assert row["stream_result"]["score_forward_calls"] == 40


@pytest.mark.parametrize("describe", (False, True))
def test_should_dispatch_seeded_pair_through_existing_quota_cli(monkeypatch, capsys, describe):
    calls = []

    def selected(**options):
        calls.append(options)
        return dict(status="complete")

    monkeypatch.setattr(cli, "run_legal_reward_pilot", selected)
    monkeypatch.setattr(cli, "describe_legal_reward_protocol", selected)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--scenario",
            "torch-key-door-legal-interaction-quota",
            "--environment-seed",
            "17",
            "--heldout-environment-seed",
            "6",
        ]
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [dict(**ROUTE, **SEEDS)]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "scenario,flags",
    (
        ("torch-key-door-legal-interaction-quota", ["--environment-seed", "17"]),
        ("torch-key-door-legal-interaction-quota", ["--heldout-environment-seed", "6"]),
        (
            "torch-key-door-legal-interaction-quota",
            ["--environment-seed", "6", "--heldout-environment-seed", "6"],
        ),
        (
            "torch-key-door-legal-interaction-quota",
            ["--environment-seed", "24", "--heldout-environment-seed", "6"],
        ),
        (
            "torch-key-door-legal-interaction-quota",
            ["--environment-seed", "17.0", "--heldout-environment-seed", "6"],
        ),
        (
            "torch-key-door-legal-rl",
            ["--environment-seed", "17", "--heldout-environment-seed", "6"],
        ),
        ("torch-key-door-rl", ["--environment-seed", "17", "--heldout-environment-seed", "6"]),
    ),
)
def test_should_refuse_invalid_seed_cli_before_invoking_any_selected_recipe(
    monkeypatch, scenario, flags
):
    for name in (
        "run_legal_reward_pilot",
        "describe_legal_reward_protocol",
        "run_reward_pilot",
        "describe_reward_protocol",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["run", "--scenario", scenario, *flags])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
