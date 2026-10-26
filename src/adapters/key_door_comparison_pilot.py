"""Run the fixed small reward/control matrix through the existing game recipe.

Inputs are the declared protocol constants only. Outputs retain every raw arm,
including failures, plus a pure paired projection. Existing recipes own learners,
quotas, episode resets and final isolation. This adapter performs no training,
selection, artifact loading, baseline tuning or replacement of failed evidence.
"""

from copy import deepcopy
from time import monotonic
from typing import Any

from src.adapters.key_door_reward_pilot import describe_protocol as describe_single_arm
from src.adapters.key_door_reward_pilot import run_pilot as run_single_arm
from src.app.key_door_reward_comparison import (
    ENVIRONMENT_PAIRS,
    MODES,
    compare_key_door_reward_cases as project,
)

MAX_COMPARISON_SECONDS = 360.0


def _matrix() -> list[dict[str, Any]]:
    return [
        dict(
            case_id=f"env{s}-final{h}-{mode}",
            environment_seed=s,
            heldout_environment_seed=h,
            mode=mode,
        )
        for s, h in ENVIRONMENT_PAIRS
        for mode in MODES
    ]


def _options(case: dict[str, Any]) -> dict[str, Any]:
    options = dict(
        condition_on_legal_actions=True,
        action_remapping=True,
        fill_interaction_quota=True,
        environment_seed=case["environment_seed"],
        heldout_environment_seed=case["heldout_environment_seed"],
    )
    if case["mode"] == "frozen":
        options["frozen_recurrent_control"] = True
    return options


def describe_protocol() -> dict[str, Any]:
    matrix = _matrix()
    return dict(
        protocol_id="actual_key_door_repeated_reward_control_v1",
        classification="descriptive_two_instance_reward_vs_nonlearning_control",
        case_matrix=matrix,
        arm_protocols={case["case_id"]: describe_single_arm(**_options(case)) for case in matrix},
        learner_seed=17,
        sampling_seed=29,
        maximum_comparison_seconds=MAX_COMPARISON_SECONDS,
        execution="fixed order; fresh original learner/budget per arm; validate preceding result before more work; first failure stops remaining cells",
        final_isolation="finish-only per arm; no final data or summaries passed to later training, promotion, tuning or case selection",
        historical_evidence=dict(
            path="artifacts/runs/r43-frozen-recurrent-control-20261017/native.json",
            qualification="failed_spent",
            primary_admission="excluded_descriptive_only",
            loaded=False,
            retry=False,
        ),
        metric_differences="reward minus frozen; positive CE-gain/return/success differences favor reward; negative retention-loss difference favors reward",
        aggregation="both declared pairs required; incomplete aggregate null, preserve every failed/not_run cell and unknown cost",
        limitations="two related finite catalogue instances, fixed learner/sampler seeds; unequal SGD/reset/compute; frozen is not a trained BP/PC/CPC baseline; no statistical, generalization, CPC or full R4.3 claim",
        planned_work=dict(
            cases=4,
            environment_pairs=2,
            minimum_native_train_calls=16,
            maximum_native_train_calls=32,
            maximum_training_rows=192,
            exact_collection_rows=384,
            maximum_collection_episodes=64,
            score_forwards=160,
            score_rows=1280,
            live_episodes=160,
            live_sessions=160,
            maximum_live_forwards=1920,
            maximum_prediction_calls=2464,
            maximum_recurrent_unroll_steps=3776,
            maximum_environment_steps=2560,
            maximum_environment_constructions=256,
            replay_calls=0,
            probes=0,
        ),
    )


def run_pilot() -> dict[str, Any]:
    protocol = describe_protocol()
    cases = [
        dict(case, status="not_run", reason="not reached", report=None)
        for case in deepcopy(protocol["case_matrix"])
    ]
    # Why: fixed owned configuration and a captured runner prevent earlier final
    # outputs or producer aliases from selecting or rewriting subsequent cells.
    runner = run_single_arm
    planned_options = tuple(_options(case) for case in cases)
    clock = monotonic
    wall_limit = protocol["maximum_comparison_seconds"]
    started = clock()
    errors: list[str] = []
    comparison = project(cases)
    for case, options in zip(cases, planned_options, strict=True):
        try:
            if clock() - started >= wall_limit:
                raise TimeoutError("original comparison wall allowance exhausted")
            case.update(status="running", reason=None)
            raw = runner(**options)
            case["report"] = deepcopy(raw)
            if type(raw) is not dict or raw.get("status") != "complete":
                raise ValueError("original arm returned incomplete or malformed evidence")
            case.update(status="complete", reason=None)
            comparison = project(cases)
            if clock() - started > wall_limit:
                raise TimeoutError("comparison exceeded original wall allowance")
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            case.update(status="failed", reason=reason)
            errors.append(reason)
            break
    try:
        comparison = project(cases)
    except Exception as error:
        errors.append(f"projection {type(error).__name__}: {error}")
        comparison = dict(status="incomplete", pairs=None, aggregate=None)
    elapsed = clock() - started
    return dict(
        protocol=protocol,
        status="complete" if not errors and comparison["status"] == "complete" else "incomplete",
        cases=cases,
        comparison=comparison,
        global_errors=errors,
        total_seconds=elapsed,
        scientific_advantage_verified=False,
        full_R43_complete=False,
    )
