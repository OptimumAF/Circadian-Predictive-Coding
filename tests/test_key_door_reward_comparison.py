"""Saved-record comparison, with explicit native-shaped synthetic evidence."""

from copy import deepcopy
import json
from typing import Any

import pytest

from src.app.key_door_reward_comparison import (
    ENVIRONMENT_PAIRS,
    MODES,
    compare_key_door_reward_cases,
)
from src.core.key_door_instances import seeded_action_remapping_layouts
from src.core.key_door_navigation import key_door_layout_record


def _outcomes(points):
    def mean(stage, context):
        return (
            sum(points[stage]["scores"][f"{context}-{color}"]["ce"] for color in ("red", "blue"))
            / 2
        )

    return dict(
        new_B_gain=mean(0, "B") - mean(2, "B"),
        new_C_gain=mean(0, "C") - mean(3, "C"),
        retention_loss=mean(3, "A") - mean(1, "A"),
        return_gain=mean(3, "A") - mean(4, "A"),
    )


def make_complete_report(environment_seed, heldout_environment_seed, mode, *, split_first=False):
    """Construct scalar native-shaped records; no models, physics or RNG runs."""
    frozen = mode == "frozen"
    layouts = [
        key_door_layout_record(layout)
        for layout in seeded_action_remapping_layouts(environment_seed, heldout_environment_seed)
    ]
    protocol = dict(
        protocol_id="actual_key_door_frozen_interaction_quota_v1"
        if frozen
        else "actual_key_door_seeded_interaction_quota_v1",
        seed=17,
        sampling_seed=29,
        feature_dim=8,
        hidden_dim=8,
        num_classes=5,
        environment_seeds=dict(training=environment_seed, final_only=heldout_environment_seed),
        methods=["recurrent_rl"],
        replay_modes=[False],
        task_sequence=["A", "B", "C", "A"],
        cues=[1, -1],
        checkpoint_physical_steps=[0, 24, 48, 72, 96],
        native_defaults=dict(learning_rate=0.03, momentum=0.9, max_episode_steps=16, discount=1.0),
        action_conditioning="legal_actions_v1",
        action_mapping="movement_permutation_v1",
        interaction_quota="twelve_steps_per_phase_v1",
        heldout_layouts=["D"],
        layouts=layouts,
    )
    row: dict[str, Any] = dict(
        status="complete",
        method="recurrent_rl",
        replay_enabled=False,
        source_unchanged=True,
        actor_unchanged=True,
        environment_seeds=deepcopy(protocol["environment_seeds"]),
        action_conditioning="legal_actions_v1",
        action_mapping="movement_permutation_v1",
        interaction_quota_mode="twelve_steps_per_phase_v1",
        live_sessions_started=40,
        physical_quotas=[
            dict(maximum_steps=12, attempted_steps=12, completed_steps=12, failed_steps=0)
            for _ in range(8)
        ],
        training=[],
        episodes=[],
        observations=[],
        work=dict(classification_unroll_steps=320),
    )
    if frozen:
        protocol["learning_mode"] = row["learning_mode"] = "frozen_recurrent_control_v1"
    native = [0]
    cumulative = 0
    for phase in range(8):
        counts = (8, 4) if split_first and phase == 0 else (12,)
        phase_done = 0
        context = ("A", "B", "C", "A")[phase // 2]
        for episode_index, count in enumerate(counts):
            index = len(row["training"])
            collection = dict(
                status="complete",
                sampler_binding_preserved=True,
                action_conditioning="legal_actions_v1",
                environment_steps=count,
                environment_steps_attempted=count,
                prediction_calls_attempted=count,
                prediction_calls_returned=count,
                sampling_draws=count,
                transitions=[dict(applied=True) for _ in range(count)],
                interaction_quota_before=dict(
                    maximum_steps=12,
                    attempted_steps=phase_done,
                    completed_steps=phase_done,
                    failed_steps=0,
                ),
                interaction_quota_after=dict(
                    maximum_steps=12,
                    attempted_steps=phase_done + count,
                    completed_steps=phase_done + count,
                    failed_steps=0,
                ),
                episode=dict(episode=dict(role="train")),
                behavior_native_updates=0 if frozen else index,
                behavior_training_examples=0 if frozen else cumulative,
            )
            slot = dict(
                index=index,
                status="complete",
                quota_phase=phase,
                episode_in_quota=episode_index,
                context=context,
                cue=1 if phase % 2 == 0 else -1,
                layout=deepcopy(layouts[("A", "B", "C", "D").index(context)]),
                collection=collection,
                native_updates_before=0 if frozen else index,
                native_training_examples_before=0 if frozen else cumulative,
                charged_updates_before=0 if frozen else index,
                charged_updates_after=0 if frozen else index + 1,
                update_attempted=not frozen,
                update_committed=not frozen,
                commit_validated=not frozen,
                receipt=None if frozen else dict(update_number=index + 1),
                native_stages=[] if frozen else ["started", "completed"],
            )
            if frozen:
                slot.update(collection_only=True, collection_validated=True)
            row["training"].append(slot)
            phase_done += count
            cumulative += count
        if phase % 2:
            native.append(0 if frozen else len(row["training"]))
    updates = 0 if frozen else len(row["training"])
    row["wake_updates"] = updates
    row["quota_checkpoints"] = [
        dict(physical_steps=p, native_updates=n) for p, n in zip((0, 24, 48, 72, 96), native)
    ]
    curves = dict(
        A=(1.0, 0.8, 0.9, 1.0, 0.7),
        B=(1.1, 1.2, 0.6, 0.7, 0.8),
        C=(1.2, 1.1, 1.0, 0.8, 0.9),
        D=(1.3, 1.3, 1.3, 1.3, 1.3),
    )
    points = []
    for stage, (physical, count) in enumerate(zip((0, 24, 48, 72, 96), native)):
        scores = {
            f"{context}-{color}": dict(ce=curves[context][0 if frozen else stage], accuracy=0.5)
            for context in ("A", "B", "C", "D")
            for color in ("red", "blue")
        }
        point = dict(updates=count, scores=scores)
        observation = dict(
            native_updates=count,
            native_training_examples=0 if frozen else physical,
            parameters=dict(weight=[0.0 if frozen else stage * 0.1]),
        )
        if frozen:
            point["physical_steps"] = observation["physical_steps"] = physical
        points.append(point)
        row["observations"].append(observation)
        for context_index, context in enumerate(("A", "B", "C", "D")):
            for cue in (1, -1):
                success = not frozen and (
                    (stage >= 2 and context == "B") or (stage == 4 and context == "A")
                )
                wrong = not frozen and stage == 3 and context == "A"
                outcome = "success" if success else "wrong_key" if wrong else "step_limit"
                returned = 8 if success or wrong else 12
                total = 1.0 if success else -1.0 if wrong else 0.0
                slot = dict(
                    status="complete",
                    state_unchanged=True,
                    execution_origin="live_environment",
                    phase="evaluation",
                    learning_active=False,
                    context=context,
                    cue=cue,
                    layout=deepcopy(layouts[context_index]),
                    layout_partition="heldout_layout" if context == "D" else "known_layout",
                    checkpoint_updates=count,
                    success=success,
                    terminal=success or wrong,
                    outcome=outcome,
                    prediction_calls_attempted=returned,
                    prediction_calls_returned=returned,
                    steps_executed=returned,
                    transitions=[
                        dict(applied=True, reward=total if i == returned - 1 else 0.0)
                        for i in range(returned)
                    ],
                    **{"return": total},
                )
                if frozen:
                    slot["checkpoint_physical_steps"] = physical
                row["episodes"].append(slot)
    row["stream_result"] = dict(
        completed_updates=updates,
        training_examples=0 if frozen else 96,
        score_forward_calls=40,
        checkpoints=points,
        applied_updates=[] if frozen else [deepcopy(s["receipt"]) for s in row["training"]],
    )
    row["outcomes"] = _outcomes(points)
    return dict(
        protocol=protocol,
        status="complete",
        global_errors=[],
        runs=[row],
        total_seconds=1.0,
        observed_process_rss=dict(peak_bytes=1000),
    )


def make_cases():
    return [
        dict(
            case_id=f"env{s}-final{h}-{mode}",
            environment_seed=s,
            heldout_environment_seed=h,
            mode=mode,
            status="complete",
            reason=None,
            report=make_complete_report(s, h, mode),
        )
        for s, h in ENVIRONMENT_PAIRS
        for mode in MODES
    ]


def test_should_pair_all_physical_stages_and_preserve_signed_direction_and_input():
    cases = make_cases()
    before = deepcopy(cases)
    result = compare_key_door_reward_cases(cases)
    assert cases == before and result["status"] == "complete"
    assert len(result["pairs"]) == result["aggregate"]["pair_count"] == 2
    for pair in result["pairs"]:
        assert pair["outcomes"]["reward_minus_frozen"] == pytest.approx(
            dict(new_B_gain=0.5, new_C_gain=0.4, retention_loss=0.2, return_gain=0.3)
        )
        assert [s["physical_steps"] for s in pair["stages"]] == [0, 24, 48, 72, 96]
        assert pair["stages"][2]["contexts"][1]["reward_minus_frozen"]["success_rate"] == 1.0
        assert pair["stages"][3]["contexts"][0]["reward_minus_frozen"]["mean_return"] == -1.0
        assert pair["stages"][4]["contexts"][0]["reward_minus_frozen"]["mean_return"] == 1.0
    assert result["aggregate"]["outcomes"]["retention_loss"] > 0
    assert result["directions"]["retention_loss"] == "negative favors reward"
    assert [c["costs"]["native_updates"] for c in result["case_costs"]] == [8, 0, 8, 0]
    assert [c["costs"]["native_training_rows"] for c in result["case_costs"]] == [96, 0, 96, 0]
    result["pairs"][0]["stages"][0]["contexts"][0]["reward"]["live"][0]["success"] = True
    assert cases == before


def test_should_not_collapse_duplicate_native_zero_or_relabel_variable_reward_clocks():
    cases = make_cases()
    cases[0]["report"] = make_complete_report(9, 14, "reward", split_first=True)
    frozen = cases[1]["report"]["runs"][0]
    for index, point in enumerate(frozen["stream_result"]["checkpoints"]):
        point["scores"]["B-red"]["ce"] += 0.05 * index
    frozen["outcomes"] = _outcomes(frozen["stream_result"]["checkpoints"])
    result = compare_key_door_reward_cases(cases)
    assert result["case_costs"][0]["costs"]["native_updates"] == 9
    pair = result["pairs"][0]
    assert (
        pair["stages"][0]["contexts"][1]["frozen"]["mean_ce"]
        != pair["stages"][4]["contexts"][1]["frozen"]["mean_ce"]
    )
    assert pair["outcomes"]["frozen"]["new_B_gain"] == pytest.approx(-0.05)


def test_should_accept_native_tuple_layout_maps_and_their_saved_json_lists_identically():
    cases = make_cases()
    saved = json.loads(json.dumps(cases, allow_nan=False))
    assert type(cases[0]["report"]["protocol"]["layouts"][1]["movement_action_map"]) is tuple
    assert type(saved[0]["report"]["protocol"]["layouts"][1]["movement_action_map"]) is list
    assert compare_key_door_reward_cases(cases) == compare_key_door_reward_cases(saved)


@pytest.mark.parametrize("status", ["not_run", "running", "failed"])
def test_should_keep_every_failed_or_unreached_pair_and_leave_unknown_costs_and_aggregate_null(
    status,
):
    cases = make_cases()
    cases[2].update(status=status, reason="preserved", report=None)
    cases[3].update(status="not_run", reason="preceding failure", report=None)
    result = compare_key_door_reward_cases(cases)
    assert result["status"] == "incomplete" and result["aggregate"] is None
    assert len(result["pairs"]) == 2 and result["pairs"][0]["status"] == "complete"
    assert result["pairs"][1]["reasons"]["reward"] == "preserved"
    assert result["pairs"][1]["outcomes"] is None
    assert result["case_costs"][2]["costs"] is None


@pytest.mark.parametrize(
    "fault",
    [
        "seed_bool",
        "case_order",
        "extra_case",
        "protocol_seed",
        "sampler_seed",
        "layout",
        "defaults",
        "actor",
        "quota",
        "collection",
        "receipt",
        "frozen_commit",
        "frozen_receipt",
        "native_bool",
        "stage_bool",
        "frozen_stage_missing",
        "live_stage_missing",
        "score_nan",
        "score_bool",
        "accuracy",
        "signed",
        "live_bool",
        "live_return",
        "initial_parameters",
        "final_role",
        "native_updates_before",
        "native_training_examples_before",
        "charged_updates_before",
        "charged_updates_after",
        "failed_checkpoint",
    ],
)
def test_should_refuse_malformed_or_unmatched_complete_evidence(fault):
    cases = make_cases()
    reward = cases[0]["report"]["runs"][0]
    frozen = cases[1]["report"]["runs"][0]
    if fault == "seed_bool":
        cases[0]["environment_seed"] = True
    elif fault == "case_order":
        cases[0], cases[1] = cases[1], cases[0]
    elif fault == "extra_case":
        cases.append(deepcopy(cases[0]))
    elif fault == "protocol_seed":
        cases[0]["report"]["protocol"]["seed"] = 18
    elif fault == "sampler_seed":
        cases[1]["report"]["protocol"]["sampling_seed"] = 30
    elif fault == "layout":
        cases[1]["report"]["protocol"]["layouts"][0]["branch_row"] = 1
    elif fault == "defaults":
        cases[1]["report"]["protocol"]["native_defaults"]["momentum"] = 0.0
    elif fault == "actor":
        reward["actor_unchanged"] = False
    elif fault == "quota":
        reward["physical_quotas"][0]["completed_steps"] = 11
    elif fault == "collection":
        reward["training"][0]["collection"]["environment_steps"] = 11
    elif fault == "receipt":
        reward["training"][0]["receipt"]["update_number"] = 2
    elif fault == "frozen_commit":
        frozen["training"][0]["commit_validated"] = True
    elif fault == "frozen_receipt":
        frozen["training"][0]["receipt"] = dict(update_number=1)
    elif fault == "native_bool":
        frozen["wake_updates"] = False
    elif fault == "stage_bool":
        frozen["quota_checkpoints"][0]["physical_steps"] = False
    elif fault == "frozen_stage_missing":
        del frozen["stream_result"]["checkpoints"][1]["physical_steps"]
    elif fault == "live_stage_missing":
        del frozen["episodes"][8]["checkpoint_physical_steps"]
    elif fault == "score_nan":
        reward["stream_result"]["checkpoints"][0]["scores"]["A-red"]["ce"] = float("nan")
    elif fault == "score_bool":
        reward["stream_result"]["checkpoints"][0]["scores"]["A-red"]["ce"] = False
    elif fault == "accuracy":
        reward["stream_result"]["checkpoints"][0]["scores"]["A-red"]["accuracy"] = 1.1
    elif fault == "signed":
        reward["outcomes"]["retention_loss"] *= -1
    elif fault == "live_bool":
        reward["episodes"][0]["success"] = 0
    elif fault == "live_return":
        reward["episodes"][0]["return"] = float("inf")
    elif fault == "initial_parameters":
        frozen["observations"][0]["parameters"]["weight"][0] = 0.5
    elif fault == "final_role":
        reward["training"][0]["collection"]["episode"]["episode"]["role"] = "final_test"
    elif fault in (
        "native_updates_before",
        "native_training_examples_before",
        "charged_updates_before",
        "charged_updates_after",
    ):
        reward["training"][0][fault] += 1
    elif fault == "failed_checkpoint":
        frozen["stream_result"]["checkpoints"][1]["status"] = "failed"
    with pytest.raises(ValueError):
        compare_key_door_reward_cases(cases)
