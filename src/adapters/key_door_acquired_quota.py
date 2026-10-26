"""Compose a fixed TRAIN prefix with the existing mapped reward recipe.

Inputs are fixed recipe constants. Outputs retain both original native arms,
their matched prefix evidence and descriptive physical-stage comparisons. The
existing actor/inbox, quota collector and final evaluator own execution. No
historical tensor loading, dose selection, promotion or new objective lives here.
"""

from copy import deepcopy
from dataclasses import asdict
from time import monotonic
from typing import Any

from src.adapters import key_door_reward_pilot as recipe
from src.adapters import key_door_pilot as shared

PREFIX = recipe.ACQUIRED_PREFIX_UPDATES
WALL_SECONDS = 240.0


def describe_protocol() -> dict[str, Any]:
    base = recipe.describe_protocol(
        condition_on_legal_actions=True,
        action_remapping=True,
        fill_interaction_quota=True,
        environment_seed=0,
        heldout_environment_seed=23,
    )
    return dict(
        protocol_id="actual_key_door_acquired_quota_prefix_v1",
        classification="fixed_train_prefix_reward_vs_frozen_not_guaranteed_acquired",
        base_protocol=base,
        arms=["frozen_prefix", "reward_prefix"],
        prefix=dict(
            updates=PREFIX,
            training_rows=8 * PREFIX,
            context="A",
            cues=[1, -1],
            schedule="alternating cues; mean CE; fixed dose, no extension or selection",
        ),
        native_caps=dict(frozen_prefix=PREFIX, reward_prefix=PREFIX + 16),
        maximum_wall_seconds=WALL_SECONDS,
        matching="full native parameters/gradients/optimizer/counters at192 before reward collection",
        final_isolation="both arms finish prefix and own96steps before constructing any final packet; no final feedback",
        physical_boundaries=[0, 24, 48, 72, 96],
        metric_differences="reward minus frozen; positive CE gain/success/return better, negative retention loss better",
        planned_work=dict(
            maximum_native_train_calls=400,
            maximum_training_rows=3168,
            prefix_updates=384,
            prefix_rows=3072,
            collection_steps=192,
            maximum_collection_episodes=32,
            score_calls=80,
            score_rows=640,
            live_episodes=80,
            maximum_live_steps=960,
            maximum_prediction_calls=1232,
            maximum_unroll_steps=4960,
            maximum_environment_steps=4288,
            maximum_environment_episodes=504,
            replay_updates=0,
            probe_updates=0,
        ),
        limitations="one finite instance/fixed initializer; prefix may fail acquisition; unequal compute, no trained BP/PC/CPC or advantage/fullR4.3 claim",
    )


def _row(frozen: bool) -> dict[str, Any]:
    row = recipe._new_row(warm_start=False)
    layouts = recipe.action_remapping_layouts(0, 23)
    by_name = {layout.layout_id: recipe.key_door_layout_record(layout) for layout in layouts}
    quota = row["training"]
    for index, slot in enumerate(quota):
        slot.update(
            index=PREFIX + index,
            quota_phase=index,
            episode_in_quota=0,
            layout=by_name[slot["context"]],
        )
        if frozen:
            slot.update(collection_only=True, collection_validated=False)
    prefix = []
    for index in range(PREFIX):
        slot = deepcopy(recipe._training_slots()[0])
        slot.update(
            index=index,
            context="A",
            cue=1 if index % 2 == 0 else -1,
            layout=deepcopy(by_name["A"]),
            phase="demonstration",
            demonstration=None,
            training_rows=None,
        )
        prefix.append(slot)
    row.update(
        arm="frozen_prefix" if frozen else "reward_prefix",
        status="not_run",
        acquired_prefix="a_only_192_v1",
        prefix=None,
        prefix_matches_control=None,
        action_conditioning="legal_actions_v1",
        action_mapping="movement_permutation_v1",
        interaction_quota_mode="twelve_steps_per_phase_v1",
        quota_checkpoints=[],
        environment_seeds=dict(training=0, final_only=23),
        demonstration_collection_attempts=0,
        training=prefix + quota,
    )
    if frozen:
        row["learning_mode"] = "frozen_recurrent_control_v1"
    return row


def _signature(batch: Any) -> Any:
    return (
        batch.role,
        batch.context,
        batch.row_ids,
        batch.num_classes,
        batch.features.dtype.str,
        batch.features.tolist(),
        batch.targets.dtype.str,
        batch.targets.tolist(),
        asdict(batch.episode),
    )


def _metrics(row: dict[str, Any], physical: int, context: str) -> dict[str, float]:
    episodes = [
        e
        for e in row["episodes"]
        if e["checkpoint_physical_steps"] == physical and e["context"] == context
    ]
    if len(episodes) != 2 or any(e["status"] != "complete" for e in episodes):
        raise ValueError("paired readout requires both original complete cue episodes")
    scores = next(
        p["scores"] for p in row["stream_result"]["checkpoints"] if p["physical_steps"] == physical
    )
    return dict(
        success_rate=sum(e["success"] for e in episodes) / 2,
        mean_return=sum(e["return"] for e in episodes) / 2,
        mean_ce=sum(scores[f"{context}-{cue}"]["ce"] for cue in ("red", "blue")) / 2,
        mean_accuracy=sum(scores[f"{context}-{cue}"]["accuracy"] for cue in ("red", "blue")) / 2,
    )


def _compare(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frozen, reward = rows
    stages = []
    for physical in (0, 24, 48, 72, 96):
        for context in ("A", "B", "C", "D"):
            f, r = _metrics(frozen, physical, context), _metrics(reward, physical, context)
            stages.append(
                dict(
                    physical_steps=physical,
                    context=context,
                    frozen=f,
                    reward=r,
                    reward_minus_frozen={key: r[key] - f[key] for key in f},
                )
            )
    acquired = {row["arm"]: _metrics(row, 0, "A")["success_rate"] == 1.0 for row in rows}
    return dict(
        status="complete",
        initial_A_acquired=acquired,
        stages=stages,
        outcomes={
            key: reward["outcomes"][key] - frozen["outcomes"][key] for key in frozen["outcomes"]
        },
        interpretation="descriptive retention/return after acquired A"
        if all(acquired.values())
        else "initial A acquisition unproven; retention/recovery interpretation inconclusive",
        scientific_advantage_verified=False,
    )


def run_pilot() -> dict[str, Any]:
    protocol = describe_protocol()
    runner = recipe._run_arm
    rows = [_row(True), _row(False)]
    checks: list[Any] = []
    finishes: list[Any] = []
    warm: list[Any] = []
    warm_seal: Any = None
    final: list[Any] = []
    final_seal: list[Any] = []
    finished: list[Any] = []
    errors: list[str] = []
    comparison = memory = None
    started = monotonic()
    deadline = started + WALL_SECONDS
    sampler = None

    def previous() -> None:
        checks[0]()
        if not recipe._same_state(warm, warm_seal):
            raise ValueError("original full prefix reference changed")

    def paired() -> None:
        if (
            len(checks) != 2
            or rows[1]["prefix_matches_control"] is not True
            or any(row["status"] not in ("trained", "complete") for row in rows)
        ):
            raise ValueError("finals require both original completed matched-prefix arms")
        for check in checks:
            check()
        if not recipe._same_state(warm, warm_seal) or not recipe._same_state(
            final_seal, [_signature(batch) for batch in final]
        ):
            raise ValueError("original prefix or final packet evidence changed")
        for row, expected in finished:
            if not recipe._same_state(
                expected, (row["stream_result"], row["episodes"], row["outcomes"])
            ):
                raise ValueError("preceding final result changed")

    def collect_final(check: Any) -> tuple[Any, ...]:
        check()
        if not final:
            for layout in recipe.action_remapping_layouts(0, 23):
                for cue in (1, -1):
                    check()
                    batch = shared._collect(
                        f"acquired-quota-final-{layout.layout_id}-{cue}", layout, cue, "final_test"
                    )
                    recipe._validate_batch(batch, "final_test", seen)
                    final.append(batch)
                    final_seal.append(deepcopy(_signature(batch)))
                    check()
        if len(final) != 8:
            raise ValueError("original final collection incomplete; no retry")
        return tuple(final)

    seen: set[str] = set()
    try:
        sampler = recipe.ProcessRssSampler()
        with sampler:
            torch = recipe.require_torch()
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            runner(
                rows[0],
                collect_final,
                deadline,
                sampler,
                warm_state=warm,
                evaluation_queue=finishes,
                completed_checks=checks,
                check_final=paired,
            )
            if rows[0]["status"] == "trained" and len(warm) == 1 and len(checks) == 1:
                warm_seal = deepcopy(warm)
                previous()
                runner(
                    rows[1],
                    collect_final,
                    deadline,
                    sampler,
                    warm_state=warm,
                    evaluation_queue=finishes,
                    completed_checks=checks,
                    check_preceding=previous,
                    check_final=paired,
                )
            if len(finishes) == 2 and all(row["status"] == "trained" for row in rows):
                rows[0]["prefix_matches_control"] = rows[1]["prefix_matches_control"]
                for row, finish in zip(rows, finishes, strict=True):
                    finish()
                    if row["status"] != "complete":
                        break
                    finished.append(
                        (row, deepcopy((row["stream_result"], row["episodes"], row["outcomes"])))
                    )
                if all(row["status"] == "complete" for row in rows):
                    paired()
                    comparison = _compare(rows)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        for row in rows:
            if row["status"] == "trained":
                row.update(
                    status="not_evaluated", reason="paired training or preceding final incomplete"
                )
        try:
            if sampler is not None:
                memory = asdict(sampler.snapshot())
        except Exception as error:
            errors.append(f"readback {type(error).__name__}: {error}")
        finishes.clear()
        checks.clear()
        warm.clear()
    elapsed = monotonic() - started
    complete = (
        comparison is not None
        and not errors
        and all(recipe._valid(row) for row in rows)
        and elapsed <= WALL_SECONDS
        and memory is not None
        and memory["peak_bytes"] <= shared.MAX_RSS
    )
    return dict(
        protocol=protocol,
        status="complete" if complete else "incomplete",
        runs=rows,
        prefix_matched=rows[1]["prefix_matches_control"] is True,
        comparison=comparison,
        demonstrations=[asdict(batch.episode) for batch in final],
        global_errors=errors,
        total_seconds=elapsed,
        observed_process_rss=memory,
        scientific_advantage_verified=False,
        full_R43_complete=False,
    )
