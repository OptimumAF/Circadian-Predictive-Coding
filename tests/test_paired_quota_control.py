"""Paired quota orchestration with literal fake states and zero native mathematics.

All collection/training/evaluation operations are stubs. One direct _run_arm
refusal constructs a literal fake port and must stop before runtime/RNG/collection.
No Torch or NumPy arrays, sampling, model initialization, forward, SGD or physical
game steps execute. Real metadata descriptors/comparisons and pair guards remain.

Per invocation: 30 cases, 13 fake pilot calls, 25 stub arm entries, 56 literal
final packets and 14 stub evaluation closures. The direct refusal adds one
literal learner construction and eight pure quota records, but no runtime.
Actual models/arrays/RNG/forwards/SGD/physics/native work: zero.
"""

from copy import deepcopy
from dataclasses import dataclass
import json
from types import SimpleNamespace
import sys
from typing import Any

import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import key_door_quota_comparison as paired

PHYSICAL = [0, 24, 48, 72, 96]
CELLS = [(context, cue) for context in "ABCD" for cue in (1, -1)]


@dataclass
class Memory:
    peak_bytes: int = 100


class Sampler:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def snapshot(self):
        return Memory()


@dataclass
class EpisodeMetadata:
    episode_id: str
    role: str = "final_test"


class LiteralArray:
    def __init__(self, values, shape, dtype):
        self.values, self.shape = values, shape
        self.dtype = SimpleNamespace(str=dtype)

    def tolist(self):
        return deepcopy(self.values)


@pytest.fixture
def orchestration(monkeypatch):
    evidence: dict[str, Any] = {
        "arms": [],
        "physics": {},
        "owners": {},
        "rows": {},
        "final_packets": [],
        "evaluations": [],
        "timeline": [],
        "thread_calls": [],
        "on_arm": None,
        "on_finish": None,
        "on_resource": None,
        "failure": None,
    }

    class ObservedSampler(Sampler):
        def snapshot(self):
            if evidence["on_resource"] is not None:
                evidence["on_resource"]()
            return super().snapshot()

    monkeypatch.setattr(paired.base, "ProcessRssSampler", ObservedSampler)
    monkeypatch.setattr(
        paired.base,
        "require_torch",
        lambda: SimpleNamespace(
            set_num_threads=lambda count: evidence["thread_calls"].append(count),
            get_num_interop_threads=lambda: 1,
        ),
    )
    monkeypatch.setattr(paired.base, "_same_state", lambda first, second: first == second)

    def valid(row):
        return row["status"] == "complete" and evidence["physics"].get(row["arm"]) == 96

    monkeypatch.setattr(paired.base, "_valid", valid)

    def collect(identity, layout, cue, role):
        assert evidence["physics"] == {"frozen": 96, "trained": 96}
        assert role == "final_test"
        evidence["timeline"].append(("final", layout.layout_id, cue))
        packet = SimpleNamespace(
            role=role,
            context=f"{layout.layout_id}-{'red' if cue == 1 else 'blue'}",
            row_ids=tuple(f"{identity}-{index}" for index in range(8)),
            num_classes=5,
            features=LiteralArray([[0.0] * 8 for _ in range(8)], (8, 8), "<f8"),
            targets=LiteralArray([0] * 8, (8,), "<i8"),
            episode=EpisodeMetadata(identity),
        )
        evidence["final_packets"].append(packet)
        return packet

    def validate(batch, role, seen):
        assert batch.role == role == "final_test"
        assert not (set(batch.row_ids) & seen)
        seen.update(batch.row_ids)

    monkeypatch.setattr(paired.base.shared, "_collect", collect)
    monkeypatch.setattr(paired.base, "_validate_batch", validate)

    def run_arm(row, final, deadline, sampler, **options):
        arm = row["arm"]
        evidence["arms"].append((arm, options))
        evidence["rows"][arm] = row
        if evidence["on_arm"] is not None:
            evidence["on_arm"](arm)
        if options["check_preceding"] is not None:
            options["check_preceding"]()
        assert evidence["final_packets"] == []
        initial = {
            "parameters": (0.25, -0.5),
            "momentum": (0.125, 0.0),
            "gradients": (None, None),
            "local_rng": (17, 0),
            "updates": 0,
            "rows": 0,
        }
        if arm == "frozen":
            assert options["quota_pair_initial"] == []
            options["quota_pair_initial"].append(deepcopy(initial))
        else:
            assert options["quota_pair_initial"] == [initial]
        row["initial_state_matches_control"] = True
        updates = 0 if arm == "frozen" else 16
        row["wake_updates"] = updates
        row["work"] = {"physical_steps": 96, "native_updates": updates}
        row["quota_checkpoints"] = [
            {"physical_steps": point, "native_updates": 0 if arm == "frozen" else index * 4}
            for index, point in enumerate(PHYSICAL)
        ]
        row["physical_quotas"] = [
            {"maximum_steps": 12, "attempted_steps": 12, "completed_steps": 12, "failed_steps": 0}
            for _ in range(8)
        ]
        row["training"] = [
            {
                "context": "AABBCCAA"[index // (1 if arm == "frozen" else 2)],
                "collection": {
                    "environment_steps": 12 if arm == "frozen" else 6,
                    "trajectory": [arm, index],
                },
                "update_committed": arm == "trained",
            }
            for index in range(8 if arm == "frozen" else 16)
        ]
        evidence["physics"][arm] = 96
        evidence["timeline"].append(("trained", arm, 96))
        if evidence["failure"] in (arm, "unreturned_" + arm):
            row.update(
                status="failed",
                reason="synthetic paid arm failure",
                wake_updates=0 if arm == "frozen" else 1,
            )
            row["training"] = row["training"][:1]
            evidence["physics"][arm] = 12 if arm == "frozen" else 6
            if evidence["failure"].startswith("unreturned_"):
                row["wake_updates"] = None
                raise RuntimeError("synthetic unreturned native update")
            return
        owner = dict(initial, updates=updates, rows=0 if arm == "frozen" else 96)
        evidence["owners"][arm] = owner
        sealed_owner, sealed_training = deepcopy(owner), deepcopy(row["training"])

        def check():
            if owner != sealed_owner or row["training"] != sealed_training:
                raise ValueError("synthetic preceding owner/training changed")

        options["completed_checks"].append(check)
        row.update(status="trained", reason=None, source_unchanged=True, actor_unchanged=True)

        def finish():
            options["check_final"]()
            packets = final(options["check_final"])
            evidence["evaluations"].append((arm, tuple(id(packet) for packet in packets)))
            native = [0] * 5 if arm == "frozen" else [0, 4, 8, 12, 16]
            ce, accuracy = (1.0, 0.5) if arm == "frozen" else (1.25, 0.25)
            row["stream_result"] = {
                "checkpoints": [
                    {
                        "updates": update,
                        "scores": {
                            f"{context}-{'red' if cue == 1 else 'blue'}": {
                                "ce": ce,
                                "accuracy": accuracy,
                            }
                            for context, cue in CELLS
                        },
                    }
                    for update in native
                ]
            }
            row["episodes"] = [
                {
                    "context": context,
                    "cue": cue,
                    "checkpoint_updates": native[stage],
                    "checkpoint_physical_steps": physical,
                    "success": arm == "frozen",
                    "return": 0.5 if arm == "frozen" else -0.25,
                    "steps_executed": 8 if arm == "frozen" else 9,
                }
                for stage, physical in enumerate(PHYSICAL)
                for context, cue in CELLS
            ]
            row["outcomes"] = {"kept_negative": True}
            row.update(status="complete", reason=None)
            if evidence["on_finish"] is not None:
                evidence["on_finish"](arm)

        options["evaluation_queue"].append(finish)

    monkeypatch.setattr(paired.base, "_run_arm", run_arm)
    return evidence


@pytest.fixture
def completed(orchestration):
    result = paired.run_pilot()
    assert result["status"] == "complete"
    return result, orchestration


def test_should_describe_same_state_equal_steps_and_separate_compute_without_resources(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("description constructed native/resource state")

    monkeypatch.setattr(paired.base, "require_torch", forbidden)
    monkeypatch.setattr(paired.base, "ProcessRssSampler", forbidden)
    protocol = paired.describe_protocol()
    assert protocol["protocol_id"] == "paired_key_door_own_policy_quota_v1"
    assert protocol["arms"] == ["frozen", "trained"]
    assert protocol["physical_steps_per_arm"] == 96
    assert protocol["physical_boundaries"] == PHYSICAL
    assert protocol["arm_protocol"]["environment_seeds"] == {"training": 0, "final_only": 23}
    assert protocol["planned_work"]["source_model_initializations"] == 2
    assert protocol["planned_work"]["native_model_initializations_min"] == 38
    assert protocol["planned_work"]["native_model_initializations_max"] == 54
    assert protocol["planned_work"]["initialization_normal_values_min"] == 38 * 168
    assert protocol["planned_work"]["initialization_normal_values_max"] == 54 * 168
    assert protocol["planned_work"]["collection_steps"] == 192
    assert protocol["planned_work"]["native_train_calls_min"] == 8
    assert protocol["planned_work"]["native_train_calls_max"] == 16
    assert "unequal" in protocol["limitations"]


@pytest.mark.parametrize(
    "seeds",
    [
        dict(environment_seed=True),
        dict(environment_seed=0.0),
        dict(environment_seed=-1),
        dict(environment_seed=24),
        dict(heldout_environment_seed=None),
        dict(heldout_environment_seed=0),
    ],
)
@pytest.mark.parametrize("run", [False, True])
def test_should_refuse_bad_instance_pair_before_resources_or_sampling(monkeypatch, seeds, run):
    monkeypatch.setattr(
        paired.base, "ProcessRssSampler", lambda: pytest.fail("bad seeds allocated resources")
    )
    selected = paired.run_pilot if run else paired.describe_protocol
    with pytest.raises(ValueError):
        selected(**seeds)


def test_should_finish_both_own_policy_allowances_before_any_final_packet(completed):
    result, evidence = completed
    assert [arm for arm, _ in evidence["arms"]] == ["frozen", "trained"]
    assert evidence["physics"] == {"frozen": 96, "trained": 96}
    assert len(evidence["final_packets"]) == 8
    assert [point[0] for point in evidence["timeline"][:2]] == ["trained", "trained"]
    assert len(evidence["evaluations"]) == 2
    assert evidence["evaluations"][0][1] == evidence["evaluations"][1][1]
    assert [row["wake_updates"] for row in result["runs"]] == [0, 16]
    assert [
        sum(slot["collection"]["environment_steps"] for slot in row["training"])
        for row in result["runs"]
    ] == [96, 96]
    assert (
        result["runs"][0]["training"][0]["collection"]["trajectory"]
        != result["runs"][1]["training"][0]["collection"]["trajectory"]
    )
    assert all(row["initial_state_matches_control"] is True for row in result["runs"])
    assert result["scientific_advantage_verified"] is result["full_R43_complete"] is False


def test_should_preserve_all_forty_negative_signed_cells_at_physical_not_native_points(completed):
    result, _ = completed
    assert len(result["comparison"]) == 40
    for index, row in enumerate(result["comparison"]):
        assert row["physical_steps"] == PHYSICAL[index // 8]
        assert row["native_updates"] == [0, 4 * (index // 8)]
        assert row["ce_reduction"] == -0.25
        assert row["accuracy_gain"] == -0.25
        assert row["success_gain"] == -1 and row["return_gain"] == -0.75
        assert row["extra_steps"] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("damage", ["owner", "training", "paid_work"])
def test_should_guard_first_paid_arm_before_any_second_arm_work(orchestration, damage):
    def mutate(arm):
        if arm == "trained":
            if damage == "owner":
                orchestration["owners"]["frozen"]["momentum"] = (99.0, 0.0)
            elif damage == "training":
                orchestration["rows"]["frozen"]["training"][0]["collection"][
                    "environment_steps"
                ] = 13
            else:
                orchestration["rows"]["frozen"]["work"]["physical_steps"] = 0

    orchestration["on_arm"] = mutate
    result = paired.run_pilot()
    assert result["status"] == "incomplete" and result["comparison"] is None
    assert orchestration["physics"] == {"frozen": 96}
    assert orchestration["final_packets"] == []
    assert result["runs"][0]["wake_updates"] == 0


@pytest.mark.parametrize("damage", ["prior_final", "cached_packet", "paid_work"])
def test_should_refuse_changed_preceding_final_or_shared_packets(orchestration, damage):
    def mutate(arm):
        if arm == "trained":
            if damage == "prior_final":
                orchestration["rows"]["frozen"]["episodes"][0]["return"] = 99.0
            elif damage == "cached_packet":
                orchestration["final_packets"][0].features.values[0][0] = 1.0
            else:
                orchestration["rows"]["frozen"]["work"]["physical_steps"] = 0

    orchestration["on_finish"] = mutate
    result = paired.run_pilot()
    assert result["status"] == "incomplete" and result["comparison"] is None
    assert len(orchestration["final_packets"]) == 8
    assert [row["wake_updates"] for row in result["runs"]] == [0, 16]
    assert result["global_errors"]


@pytest.mark.parametrize("damage", ["owner", "paid_row"])
def test_should_revalidate_pair_after_resource_snapshot_callback(orchestration, damage):
    def mutate():
        if damage == "owner":
            orchestration["owners"]["frozen"]["momentum"] = (99.0, 0.0)
        else:
            orchestration["rows"]["frozen"]["work"]["physical_steps"] = 0

    orchestration["on_resource"] = mutate
    result = paired.run_pilot()
    assert result["status"] == "incomplete"
    assert len(orchestration["evaluations"]) == 2
    assert len(orchestration["final_packets"]) == 8
    assert result["observed_process_rss"]["peak_bytes"] == 100
    assert any("resource readback" in error for error in result["global_errors"])


@pytest.mark.parametrize("failure", ["frozen", "trained", "unreturned_trained"])
def test_should_preserve_paid_partial_training_and_never_collect_finals(orchestration, failure):
    orchestration["failure"] = failure
    result = paired.run_pilot()
    assert result["status"] == "incomplete" and result["comparison"] is None
    assert orchestration["final_packets"] == []
    failed = result["runs"][0 if failure == "frozen" else 1]
    assert failed["status"] == "failed" and len(failed["training"]) == 1
    assert failed["wake_updates"] == (
        None if failure == "unreturned_trained" else 0 if failure == "frozen" else 1
    )
    assert failed["training"][0]["collection"]["environment_steps"] > 0


def test_should_reject_full_optimizer_initial_mismatch_before_runtime_rng_or_collection(
    monkeypatch,
):
    row = paired._row("trained", paired.describe_protocol())
    initial: dict[str, Any] = {
        "parameters": (0.25,),
        "optimizer": {"velocity": (0.125,)},
        "gradients": (None,),
        "updates": 0,
    }
    changed = deepcopy(initial)
    changed["optimizer"]["velocity"] = (0.5,)

    class LiteralLearner:
        def __init__(self, policy):
            self.policy = policy

        def snapshot_state(self):
            return deepcopy(changed)

    budget = SimpleNamespace(
        budget=SimpleNamespace(max_training_updates=16),
        clock=lambda: 0.0,
        progress=SimpleNamespace(),
        started_at=0.0,
        updates_completed=0,
        replay_examples_completed=0,
    )
    monkeypatch.setattr(paired.base.shared, "_session", lambda *args, **kwargs: budget)
    monkeypatch.setattr(paired.base, "TorchRewardUpdateLearner", LiteralLearner)
    monkeypatch.setattr(paired.base, "_same_state", lambda first, second: first == second)
    monkeypatch.setattr(
        paired.base,
        "ActorShadowRuntime",
        lambda *args, **kwargs: pytest.fail("mismatched state opened runtime"),
    )
    monkeypatch.setattr(
        paired.base,
        "np",
        SimpleNamespace(
            random=SimpleNamespace(
                Generator=lambda *args: pytest.fail("mismatched state created RNG"),
                PCG64=lambda *args: pytest.fail("mismatched state seeded RNG"),
            )
        ),
    )
    monkeypatch.setattr(
        paired.base,
        "collect_key_door_reward_episode",
        lambda *args, **kwargs: pytest.fail("mismatched state collected physics"),
    )
    queue: list[Any] = []
    checks: list[Any] = []
    paired.base._run_arm(
        row,
        lambda check: (),
        100.0,
        object(),
        evaluation_queue=queue,
        completed_checks=checks,
        check_final=lambda: None,
        quota_pair_initial=[initial],
    )
    assert row["status"] == "failed" and "initialization differs" in row["reason"]
    assert queue == checks == [] and row["wake_updates"] == 0


@pytest.mark.parametrize("describe", [False, True])
def test_should_route_explicit_cli_scenario_with_both_instance_seeds(monkeypatch, capsys, describe):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "complete", "paired": True}

    monkeypatch.setattr(
        cli,
        "describe_quota_comparison_protocol" if describe else "run_quota_comparison_pilot",
        selected,
    )
    arguments = [
        "pilot",
        "--scenario",
        "torch-key-door-paired-interaction-quota",
        "--environment-seed",
        "17",
        "--heldout-environment-seed",
        "6",
    ]
    if describe:
        arguments.append("--describe")
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    assert calls == [{"environment_seed": 17, "heldout_environment_seed": 6}]
    assert json.loads(capsys.readouterr().out)["paired"] is True


def test_should_leave_standalone_default_dispatch_and_false_keyword_shape_unchanged(monkeypatch):
    calls = []

    def selected(**options):
        calls.append(options)
        return {}

    monkeypatch.setattr(paired.base, "_run_pilot", selected)
    route = {
        "condition_on_legal_actions": True,
        "action_remapping": True,
        "fill_interaction_quota": True,
    }
    paired.base.run_pilot(**route)
    paired.base.run_pilot(**route, frozen_recurrent_control=False)
    assert calls == [dict(warm_start=False, **route)] * 2
