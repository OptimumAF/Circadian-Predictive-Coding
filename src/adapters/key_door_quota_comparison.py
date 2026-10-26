"""Pair existing recurrent reward/frozen routes at equal physical exposure.

Inputs are declared instance indices. Outputs retain both own-policy trajectories,
all five physical-boundary readouts and signed comparisons. The existing reward
runner owns training, native guards and evaluation. No new learning objective,
environment, replay, probe, promotion or outcome-based selection belongs here.
"""

from copy import deepcopy
from dataclasses import asdict
from math import isfinite
from time import monotonic, perf_counter
from typing import Any, Callable

from src.adapters import key_door_reward_pilot as base


def describe_protocol(
    *, environment_seed: int = 0, heldout_environment_seed: int = 23, wall_seconds: float = 90
) -> dict[str, Any]:
    """Validate declared instances without constructing a model or observing data."""
    if (
        type(wall_seconds) not in (int, float)
        or not 0 < wall_seconds <= 180
        or not isfinite(wall_seconds)
    ):
        raise ValueError("paired wall_seconds must be a finite native number in (0, 180]")
    arm = base.describe_protocol(
        condition_on_legal_actions=True,
        action_remapping=True,
        fill_interaction_quota=True,
        environment_seed=environment_seed,
        heldout_environment_seed=heldout_environment_seed,
    )
    return dict(
        protocol_id="paired_key_door_own_policy_quota_v1",
        classification="exploratory_paired_equal_interaction_control",
        arms=["frozen", "trained"],
        arm_protocol=arm,
        initialization="exact full native state equality before trained collection; distinct owners",
        sampling="independent PCG64(29) owners; realized own-policy draws/trajectories may differ",
        physical_steps_per_arm=96,
        physical_boundaries=[0, 24, 48, 72, 96],
        evaluation="both arms finish TRAIN before any shared final packet or score/live episode",
        directions=dict(
            ce_reduction="frozen minus trained; higher",
            accuracy_gain="trained minus frozen; higher",
            success_gain="trained minus frozen; higher",
            return_gain="trained minus frozen; higher",
            extra_steps="trained minus frozen; lower",
        ),
        planned_work=dict(
            source_model_initializations=2,
            native_model_initializations_min=38,
            native_model_initializations_max=54,
            initialization_normal_values_min=6384,
            initialization_normal_values_max=9072,
            native_train_calls_min=8,
            native_train_calls_max=16,
            training_rows=96,
            collection_steps=192,
            collection_episodes_max=32,
            score_forward_calls=80,
            score_rows=640,
            live_episodes=80,
            maximum_live_steps=960,
            final_teacher_episodes=8,
            final_teacher_steps=64,
            replay_rows=0,
            probes=0,
        ),
        resources=dict(
            wall_seconds=wall_seconds, cpu_threads=1, gpu=0, rss_bytes=base.shared.MAX_RSS
        ),
        timing="per-arm TRAIN/evaluation spans and inclusive overlapping guard times; telemetry only, no budget authority",
        limitations="equal interactions, unequal updates/resets/computation; legal support can force actions; no CPC/generalization/confirmation claim",
    )


def _timing_record() -> dict[str, Any]:
    return dict(attempts=0, returns=0, raised=0, inclusive_seconds=0.0, clock_error=None)


def _time_call(record: dict[str, Any], callback: Callable[[], Any]) -> Any:
    """Measure returned/raised calls without replacing their primary failure."""
    try:
        started = perf_counter()
        if type(started) not in (int, float) or not isfinite(started):
            raise ValueError("paired timing clock must return finite seconds")
    except Exception as error:
        record["clock_error"] = f"{type(error).__name__}: {error}"
        raise
    record["attempts"] += 1
    raised = False
    try:
        result = callback()
    except BaseException:
        raised = True
        record["raised"] += 1
        raise
    else:
        record["returns"] += 1
        return result
    finally:
        try:
            ended = perf_counter()
            if type(ended) not in (int, float) or not isfinite(ended) or ended < started:
                raise ValueError("paired timing clock must be finite and nondecreasing")
            total = record["inclusive_seconds"] + (ended - started)
            if not isfinite(total):
                raise ValueError("paired measured time overflowed")
            record["inclusive_seconds"] = total
        except Exception as error:
            record["clock_error"] = f"{type(error).__name__}: {error}"
            if not raised:
                raise


def _row(arm: str, protocol: dict[str, Any]) -> dict[str, Any]:
    row = base._new_row(warm_start=False)
    row.update(
        arm=arm,
        action_conditioning="legal_actions_v1",
        action_mapping="movement_permutation_v1",
        interaction_quota_mode="twelve_steps_per_phase_v1",
        quota_checkpoints=[dict(physical_steps=0, native_updates=0)],
        environment_seeds=deepcopy(protocol["arm_protocol"]["environment_seeds"]),
        initial_state_matches_control=None,
    )
    layouts = {layout.layout_id: layout for layout in base._row_layouts(row)}
    for index, slot in enumerate(row["training"]):
        slot.update(
            layout=base.key_door_layout_record(layouts[slot["context"]]),
            quota_phase=index,
            episode_in_quota=0,
        )
        if arm == "frozen":
            slot.update(collection_only=True, collection_validated=False)
    if arm == "frozen":
        row["learning_mode"] = "frozen_recurrent_control_v1"
    return row


def _final_signature(batch: Any) -> Any:
    return (
        batch.role,
        batch.context,
        batch.row_ids,
        batch.num_classes,
        batch.features.dtype.str,
        batch.features.shape,
        batch.features.tolist(),
        batch.targets.dtype.str,
        batch.targets.shape,
        batch.targets.tolist(),
        asdict(batch.episode),
    )


def _result(row: Any) -> Any:
    # Two bounded rows include paid collection/receipt/work and failure readbacks.
    # Later observers must not rewrite costs while leaving scores unchanged.
    return row


def _compare(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare every declared stage/cell, keeping actual native indices separate."""
    output = []
    for stage, physical in enumerate((0, 24, 48, 72, 96)):
        points = [row["stream_result"]["checkpoints"][stage] for row in rows]
        for offset in range(8):
            a, b = [row["episodes"][8 * stage + offset] for row in rows]
            if (a["context"], a["cue"]) != (b["context"], b["cue"]):
                raise ValueError("paired physical readout cells differ")
            key = f"{a['context']}-{'red' if a['cue'] == 1 else 'blue'}"
            left, right = [point["scores"][key] for point in points]
            output.append(
                dict(
                    physical_steps=physical,
                    context=key,
                    native_updates=[point["updates"] for point in points],
                    ce_reduction=left["ce"] - right["ce"],
                    accuracy_gain=right["accuracy"] - left["accuracy"],
                    success_gain=int(b["success"]) - int(a["success"]),
                    return_gain=b["return"] - a["return"],
                    extra_steps=b["steps_executed"] - a["steps_executed"],
                )
            )
    return output


def run_pilot(
    *, environment_seed: int = 0, heldout_environment_seed: int = 23, wall_seconds: float = 90
) -> dict[str, Any]:
    """Run frozen then trained collection; release finals only after both validate."""
    protocol = describe_protocol(
        environment_seed=environment_seed,
        heldout_environment_seed=heldout_environment_seed,
        wall_seconds=wall_seconds,
    )
    rows = [_row(arm, protocol) for arm in ("frozen", "trained")]
    checks: list[Any] = []
    finishes: list[Any] = []
    initial: list[Any] = []
    packets: list[Any] = []
    signatures: list[Any] = []
    completed_results: list[Any] = []
    pending_results: list[Any] = []
    demonstrations: list[Any] = []
    errors: list[str] = []
    comparison = None
    memory = sampler = None
    started = monotonic()
    # Capture once: mutable report metadata cannot grant time to later arms.
    deadline = started + wall_seconds
    # Keep telemetry outside paid rows: their guards must observe stable evidence.
    timing: dict[str, Any] = dict(
        definition="paired_quota_inclusive_wall_v1",
        clock="perf_counter",
        scope="wall spans include nested guards; guard categories overlap and must not be summed",
        arms={
            arm: {phase: _timing_record() for phase in ("train", "evaluation")}
            for arm in ("frozen", "trained")
        },
        guards={name: _timing_record() for name in ("pair", "preceding", "pending")},
    )

    def validate_pending() -> None:
        for row, expected in pending_results:
            if not base._same_state(row, expected):
                raise ValueError("preceding paired TRAIN evidence changed")

    def check_pending() -> None:
        _time_call(timing["guards"]["pending"], validate_pending)

    def validate_preceding() -> None:
        checks[0]()
        check_pending()

    def check_preceding() -> None:
        _time_call(timing["guards"]["preceding"], validate_preceding)

    def validate_pair() -> None:
        if len(checks) != 2 or any(row["status"] not in ("trained", "complete") for row in rows):
            raise ValueError("paired quota finals require both completed TRAIN allowances")
        for check in checks:
            check()
        check_pending()
        for row, expected in completed_results:
            if not base._same_state(_result(row), expected):
                raise ValueError("preceding paired final evidence changed")
        if not base._same_state(signatures, [_final_signature(batch) for batch in packets]):
            raise ValueError("paired final packets changed")
        if not base._same_state(demonstrations, [signature[-1] for signature in signatures]):
            raise ValueError("paired final demonstration evidence changed")

    def check_pair() -> None:
        _time_call(timing["guards"]["pair"], validate_pair)

    def collect_final(check_boundary: Any) -> Any:
        check_boundary()
        if not packets:
            seen: set[str] = set()
            for layout in base._row_layouts(rows[0]):
                for cue in (1, -1):
                    check_boundary()
                    batch = base.shared._collect(
                        f"paired-quota-final-{layout.layout_id}-{cue}", layout, cue, "final_test"
                    )
                    base._validate_batch(batch, "final_test", seen)
                    if batch.episode is None:
                        raise ValueError("paired final packet lacks episode metadata")
                    packets.append(batch)
                    signatures.append(_final_signature(batch))
                    demonstrations.append(asdict(batch.episode))
                    check_boundary()
        if len(packets) != 8:
            raise ValueError("paired final collection incomplete; no retry")
        check_boundary()
        return tuple(packets)

    try:
        sampler = base.ProcessRssSampler()
        with sampler:
            torch = base.require_torch()
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            for index, row in enumerate(rows):
                _time_call(
                    timing["arms"][row["arm"]]["train"],
                    lambda: base._run_arm(
                        row,
                        collect_final,
                        deadline,
                        sampler,
                        evaluation_queue=finishes,
                        completed_checks=checks,
                        check_preceding=check_preceding if index else None,
                        check_final=check_pair,
                        quota_pair_initial=initial,
                    ),
                )
                if row["status"] != "trained":
                    break
                pending_results.append((row, deepcopy(row)))
            if len(finishes) == len(checks) == 2 and all(
                row["status"] == "trained" for row in rows
            ):
                check_pair()
                for row, finish in zip(rows, finishes, strict=True):
                    # This row now changes legitimately during evaluation; the
                    # other pending row stays sealed until its own turn.
                    check_pair()
                    pending_results.pop(0)
                    _time_call(timing["arms"][row["arm"]]["evaluation"], finish)
                    check_pair()
                    if row["status"] != "complete":
                        raise ValueError("paired final readout failed")
                    completed_results.append((row, deepcopy(_result(row))))
                check_pair()
                comparison = _compare(rows)
                check_pair()
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error!r}")
    finally:
        for row in rows:
            if row["status"] == "trained":
                row.update(status="not_evaluated", reason="paired collection/evaluation incomplete")
        if sampler is not None:
            try:
                memory = asdict(sampler.snapshot())
                if comparison is not None:
                    check_pair()
            except Exception as error:
                errors.append(f"resource readback {type(error).__name__}: {error!r}")
        # These closures retain original live owners until the last boundary read.
        finishes.clear()
        checks.clear()
        initial.clear()
    elapsed = monotonic() - started
    complete = (
        not errors
        and comparison is not None
        and all(base._valid(row) for row in rows)
        and all(row["initial_state_matches_control"] is True for row in rows)
        and elapsed <= wall_seconds
        and memory is not None
        and memory["peak_bytes"] <= base.shared.MAX_RSS
    )
    return dict(
        protocol=protocol,
        status="complete" if complete else "incomplete",
        runs=rows,
        comparison=comparison,
        demonstrations=demonstrations,
        global_errors=errors,
        total_seconds=elapsed,
        timing=timing,
        observed_process_rss=memory,
        scientific_advantage_verified=False,
        full_R43_complete=False,
    )
