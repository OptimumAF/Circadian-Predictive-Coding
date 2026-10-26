"""Describe fixed disposable E-task probes on an existing four-phase control.

Inputs are a declared replay control and its existing fixed protocol. Outputs
are detached descriptions and prospective combined costs. No data collection,
learner, environment, selection, clock, budget renewal, scoring or IO occurs.
"""

from copy import deepcopy
from typing import Any

from src.adapters.four_phase_replay import validate_control


_BASE_COSTS = dict(
    native_train_calls=384,
    training_rows=3072,
    score_forwards=142,
    score_rows=1136,
    maximum_native_forwards=622,
    maximum_recurrent_unroll_steps=4688,
    teacher_episodes=14,
    environment_constructions=54,
    maximum_environment_steps=592,
)
_EXTRA_COSTS = dict(
    native_train_calls=32,
    training_rows=256,
    score_forwards=16,
    score_rows=128,
    maximum_native_forwards=16,
    maximum_recurrent_unroll_steps=384,
    teacher_episodes=18,
    environment_constructions=18,
    maximum_environment_steps=144,
)


def _validate_base(base: dict[str, Any], control: str) -> None:
    if (
        type(base) is not dict
        or base.get("protocol_id") != f"key_door_four_phase_{control}_replay_v1"
        or base.get("replay_control") != control
        or type(base.get("planned_work")) is not dict
        or type(base.get("final_evaluation")) is not dict
        or type(base.get("limits")) is not dict
        or type(base.get("limitations")) is not str
        or type(base.get("isolation")) is not str
    ):
        raise ValueError("four-phase probes require the matching existing replay protocol")
    for name, expected in _BASE_COSTS.items():
        value = base["planned_work"].get(name)
        if type(value) is not int or value != expected:
            raise ValueError(f"four-phase probe base cost {name} differs from the fixed protocol")


def describe_protocol(base: dict[str, Any], *, control: str) -> dict[str, Any]:
    """Add fixed initial/post-return E16 probes while preserving the base recipe.

    Why this: composition binds the reviewed benchmark cost envelope before
    adding diagnostic work; it never infers a fresh allowance from saved results.
    """
    validate_control(control)
    _validate_base(base, control)
    result = deepcopy(base)
    result.update(
        protocol_id=f"key_door_four_phase_{control}_fresh_task_probe_v1",
        classification="exploratory_single_control_fresh_task_diagnostic",
        probe=dict(
            layout="existing E(3,3,True,0.75)",
            source_wake_checkpoints=[0, 256],
            source_native_checkpoints=[0, 384],
            training_episodes=16,
            evaluation_episodes=2,
            checkpoints=[0, 4, 8, 16],
            cohorts=2,
            independent_updates_per_cohort=16,
            decision_diagnostics=True,
            metrics="all reserved E CE/accuracy/class/ordered sequence curves; CE0-minus-CE16 gain, higher accuracy/exact sequence better; all failed/negative results retained",
            comparison_scope="initial versus post-return within the selected fixed control; gain alone is not an isolated plasticity measurement or comparative CPC advantage",
            isolation="disposable owned checkpoint forks preserve native policy/optimizer state; independent16-update probe budgets never update benchmark/replay/deployed learners or choose training/promotion/probes",
            final_data_creation="after256benchmark wakes+stream.finish and both completed probes",
            budget=dict(
                max_training_updates_per_cohort=16,
                deadline="remaining original invocation deadline; no renewed wall allowance",
                process_rss="original invocation sampled RSS ceiling includes all probe work",
                computation="32SGD/256training rows and16reserved forwards/128score rows; no extra native prediction for decision reports",
            ),
            storage=dict(
                reserved_input_array_bytes=10368,
                scope="18reserved packets x8rows x(8features+1target) x8bytes; excludes episode/Python/native-state/copy/diagnostic payloads, all included in total time/process RSS",
            ),
            limitation="same deterministic E trace domain, not generalization; no isolated plasticity or CPC benefit established from probe gain alone",
        ),
    )
    work = result["planned_work"]
    for name, extra in _EXTRA_COSTS.items():
        work[name] += extra
    work.update(
        probes=2,
        probe_train_calls=32,
        probe_training_rows=256,
        probe_score_forwards=16,
        probe_score_rows=128,
        teacher_steps=256,
    )
    result["final_evaluation"]["final_data_creation"] = result["probe"]["final_data_creation"]
    result["isolation"] += (
        "; final packets remain unavailable until all256wakes+finish and both isolated probes complete"
    )
    result["limitations"] += (
        "; probe gain alone does not isolate plasticity; reserved E uses the same deterministic trace domain, not generalization; no comparative CPC advantage"
    )
    return result
