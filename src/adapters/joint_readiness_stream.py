"""Fixed two-stage joint TRAIN correction through the existing stream observer.

Own and release raw inputs before initializing the existing BP stream. Preserve
both paid branch reports separately and return only the final adapted candidate.
No collection, qualification, IO, native controller, probe, selection, optimizer
or alternate runner belongs here. Reference qualification remains caller-owned.
"""

from copy import deepcopy
from dataclasses import asdict
import json
from typing import Any, cast

from src.adapters.continuous_learnability_demo import (
    ContinuousLearnabilityResult,
    run_continuous_learnability,
)
from src.adapters.joint_readiness_corrections import (
    run_joint_readiness_corrections,
    _require_state,
    _require_sampling_option,
    _sampling_metadata,
)
from src.adapters.joint_readiness_data import prepare_joint_readiness_data
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner, _snapshot
from src.core.curated_readiness_corrections import (
    CuratedReadinessBatch,
    copy_curated_readiness_batch,
)
from src.core.experience import require_tick
from src.core.pick_cube_action_history import action_history_training_episode
from src.core.pick_cube_readiness_corrections import (
    GoalConsistentObservedGraspCorrectionBatch,
    copy_readiness_correction_batch,
    _observed_grasp_flags,
)
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.readiness_stage_diagnostics import ReadinessStageBinding


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _own_inputs(warm, new, corrections, curated, corrections_now, curated_now):
    require_tick(corrections_now, "joint stream correction release time")
    require_tick(curated_now, "joint stream curated release time")
    if curated_now < corrections_now:
        raise ValueError("curated release time cannot precede correction release time")
    if type(warm) is not ReadinessStageBinding or type(new) is not ReadinessStageBinding:
        raise ValueError("joint stream requires exact raw readiness stage bindings")
    if (
        type(corrections) is not GoalConsistentObservedGraspCorrectionBatch
        or type(curated) is not CuratedReadinessBatch
    ):
        raise ValueError("joint stream requires exact B corrections and nonnull A curation")
    _observed_grasp_flags(warm.episode)
    _observed_grasp_flags(new.episode)
    owned_warm, owned_new = (ReadinessStageBinding(v.episode, v.stages) for v in (warm, new))
    if (
        owned_new.episode.episode_id != "pick-cube-readiness-teacher-19"
        or owned_new.episode.context != "pick-cube-train-19"
        or owned_new.episode.role != "train"
        or owned_new.episode.spec.observation_dim != 42
        or owned_new.episode.spec != owned_warm.episode.spec
        or not 1 <= len(owned_new.episode.transitions) <= 128
    ):
        raise ValueError("joint stream requires the original qualified raw42 B19 demonstration")
    owned_b = cast(
        GoalConsistentObservedGraspCorrectionBatch, copy_readiness_correction_batch(corrections)
    )
    owned_c = copy_curated_readiness_batch(curated)
    recipe = PickCubeTaskSpaceHistoryRecipe(mode="executed")
    # Why: both stages are admitted before any model work, including C arrival.
    # The existing correction APIs re-own/project again; report that real cost.
    ab = prepare_joint_readiness_data(owned_warm, owned_b, None, now=corrections_now, recipe=recipe)
    abc = prepare_joint_readiness_data(owned_warm, owned_b, owned_c, now=curated_now, recipe=recipe)
    model_warm = action_history_training_episode(owned_warm.episode, recipe)
    model_new = action_history_training_episode(owned_new.episode, recipe)
    bindings = (
        ReadinessStageBinding(model_warm, owned_warm.stages),
        ReadinessStageBinding(model_new, owned_new.stages),
    )
    return owned_warm, owned_new, owned_b, owned_c, recipe, ab, abc, bindings


def _pin(owner):
    if type(owner) is not NumpyContinuousBCLearner:
        raise RuntimeError("joint stream requires an exact detached BP owner")
    return owner, owner._state, _snapshot(owner._state, owner._policy)


def run_joint_readiness_stream(
    warm_binding: ReadinessStageBinding,
    new_binding: ReadinessStageBinding,
    corrections: GoalConsistentObservedGraspCorrectionBatch,
    curated: CuratedReadinessBatch,
    *,
    corrections_now: int,
    curated_now: int,
    sampling: str | None = None,
) -> ContinuousLearnabilityResult:
    """Run fixed BP8/8/8, branching at B into A+B8 then A+B+C8.

    Invalid admission raises before initialization. Paid failures return all
    observed reports and no candidate. Mixed data never gain teacher authority.
    """
    _require_sampling_option(sampling)
    originals = (warm_binding, new_binding, corrections, curated)
    if any(
        type(v)
        not in (
            ReadinessStageBinding,
            GoalConsistentObservedGraspCorrectionBatch,
            CuratedReadinessBatch,
        )
        for v in originals
    ):
        raise ValueError("joint stream requires its four exact typed raw inputs")
    original_seal = _json([asdict(v) for v in originals])
    warm, new, b, c, recipe, ab, abc, model_bindings = _own_inputs(
        *originals, corrections_now, curated_now
    )
    # Both optional sampling schedules must be valid before the base initializes.
    sampling_preflight = None
    if sampling is not None:
        ab_plan, _ = _sampling_metadata(ab, sampling)
        abc_plan, _ = _sampling_metadata(abc, sampling)
        sampling_preflight = dict(
            calls=2,
            population_rows=len(ab.rows) + len(abc.rows),
            planned_index_entries=sum(len(batch) for plan in (ab_plan, abc_plan) for batch in plan),
            scope="additional preflight schedules; runtime plans/counting/validation/copies "
            "are charged separately; Python storage additional",
        )
    owned = (warm, new, b, c, recipe, *model_bindings)
    owned_seal = _json([asdict(v) for v in owned])
    preparation_seal = _json((asdict(ab), asdict(abc)))
    n_warm, n_new = len(warm.episode.transitions), len(new.episode.transitions)
    expected = [
        dict(checkpoint_index=i, phase=p, updates=u, training_examples=n)
        for i, (p, u, n) in enumerate(
            (
                ("initial", 0, 0),
                ("warm", 8, 8 * n_warm),
                ("new", 16, 8 * (n_warm + n_new)),
                ("warm_return", 24, 8 * (2 * n_warm + n_new)),
            )
        )
    ]
    report: dict[str, Any] = dict(
        protocol="joint_readiness_stream_v1",
        complete=False,
        stream=None,
        joint_ab=None,
        joint_abc=None,
        checkpoint=None,
        preparation_work=dict(
            preflight_calls=2,
            preflight_ab=deepcopy(ab.work),
            preflight_abc=deepcopy(abc.work),
            runtime_calls_attempted=0,
            runtime_calls_returned=0,
            source_projection=dict(
                episode_calls=2,
                observation_projections=2 * (n_warm + n_new),
                source_values_validated=84 * (n_warm + n_new),
                projected_values=24 * (n_warm + n_new),
                projected_numeric_bytes=192 * (n_warm + n_new),
                history_advances=n_warm + n_new,
            ),
            scope="preflight projections are additional to each branch preparation_work; original "
            "input ownership, validation, snapshots/copies and Python containers additional",
        ),
    )
    pins: list[tuple[NumpyContinuousBCLearner, Any, Any]] = []
    if sampling_preflight is not None:
        report["sampling"] = sampling
        report["preparation_work"]["sampling_preflight"] = sampling_preflight
    paid: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
    callback_count = 0
    final_candidate = None
    runtime_attempts = runtime_returns = 0
    preparation_base = deepcopy(report["preparation_work"])

    def input_guard():
        if (
            _json([asdict(v) for v in originals]) != original_seal
            or _json([asdict(v) for v in owned]) != owned_seal
            or _json((asdict(ab), asdict(abc))) != preparation_seal
        ):
            raise RuntimeError("joint stream original or owned input evidence changed")

    def guard():
        input_guard()
        if sampling is not None and report.get("sampling") != sampling:
            raise RuntimeError("joint stream original sampling selection changed")
        preparation_expected = dict(
            preparation_base,
            runtime_calls_attempted=runtime_attempts,
            runtime_calls_returned=runtime_returns,
        )
        if _json(report["preparation_work"]) != _json(preparation_expected) or _json(
            report["checkpoint"]
        ) != _json(None if callback_count <= 2 else expected[2]):
            raise RuntimeError("joint stream preparation/checkpoint evidence changed")
        for owner, native, state in pins:
            _require_state(owner, state, native)
        for key, raw, owned_report, payload in paid:
            if (
                report[key] is not owned_report
                or _json(raw) != payload
                or _json(owned_report) != payload
            ):
                raise RuntimeError("joint stream preceding raw/copied paid report changed")
        # A later public snapshot may change an owner checked earlier; close with
        # native-only checks so no final observer can silently become a baseline.
        for owner, native, state in pins:
            _require_state(owner, state, native, observe=False)
        input_guard()

    input_guard()

    def observe(readout, metadata):
        nonlocal callback_count, final_candidate, runtime_attempts, runtime_returns
        index = callback_count
        if (
            type(metadata) is not dict
            or index >= 4
            or metadata != expected[index]
            or any(
                type(metadata[k]) is not int
                for k in ("checkpoint_index", "updates", "training_examples")
            )
        ):
            raise RuntimeError("joint stream fixed checkpoint metadata changed")
        guard()
        callback_count += 1
        if index != 2:
            return dict(checkpoint_index=index, joint_updates=0)
        checkpoint_pin = _pin(readout)
        pins.append(checkpoint_pin)
        if checkpoint_pin[2].policy.spec != ab.spec or checkpoint_pin[2].policy.hidden_dim != 64:
            raise RuntimeError("joint stream checkpoint policy differs from fixed TS12/H64")
        report["checkpoint"] = deepcopy(metadata)
        for key, source, selected, tick in (
            ("joint_ab", readout, None, corrections_now),
            ("joint_abc", None, curated, curated_now),
        ):
            guard()
            if key == "joint_abc":
                source = final_candidate
            runtime_attempts += 1
            report["preparation_work"]["runtime_calls_attempted"] = runtime_attempts
            # Keep the caller's original training evidence in the existing
            # per-update input guard; preflight copies are not renewed authority.
            result = run_joint_readiness_corrections(
                source,
                warm_binding,
                corrections,
                selected,
                now=tick,
                action_history_recipe=recipe,
                **({} if sampling is None else {"sampling": sampling}),
            )
            owned_report = report[key] = deepcopy(result.report)
            paid.append((key, result.report, owned_report, _json(result.report)))
            runtime_returns += 1
            report["preparation_work"]["runtime_calls_returned"] = runtime_returns
            # Retain the complete paid report before observing the returned owner.
            if owned_report["complete"] is not True or result.candidate is None:
                raise RuntimeError(
                    "joint stream " + key + " incomplete: " + owned_report.get("error", "unknown")
                )
            candidate_pin = _pin(result.candidate)
            if any(result.candidate is owner for owner, _, _ in pins):
                raise RuntimeError("joint stream branch candidate aliases an earlier owner")
            previous = pins[-1][2]
            rows = len(ab.rows) if key == "joint_ab" else len(abc.rows)
            if (
                candidate_pin[2].policy != previous.policy
                or candidate_pin[2].completed_updates != previous.completed_updates + 8
                or candidate_pin[2].training_examples != previous.training_examples + 8 * rows
            ):
                raise RuntimeError("joint stream branch continuation counters/policy differ")
            pins.append(candidate_pin)
            final_candidate = result.candidate
            guard()
        return dict(checkpoint_index=2, joint_updates=16, complete=True)

    try:
        base = run_continuous_learnability(
            model_bindings[0].episode,
            model_bindings[1].episode,
            action_history_recipe=recipe,
            diagnostic_stage_bindings=model_bindings,
            learner_family="bp",
            phase_updates=8,
            head_gauss_newton=True,
            checkpoint_observer=observe,
        )
        report["stream"] = deepcopy(base.report)
        paid.append(("stream", base.report, report["stream"], _json(base.report)))
        if base.report["complete"] is not True or base.candidate is None:
            raise RuntimeError(
                "joint stream base incomplete: " + base.report.get("error", "unknown")
            )
        base_pin = _pin(base.candidate)
        if any(base.candidate is owner for owner, _, _ in pins):
            raise RuntimeError("joint stream benchmark candidate aliases adapted branch")
        pins.append(base_pin)
        if (
            callback_count != 4
            or final_candidate is None
            or report["joint_ab"] is None
            or report["joint_abc"] is None
            or (base_pin[2].completed_updates, base_pin[2].training_examples)
            != (24, 8 * (2 * n_warm + n_new))
            or base_pin[2].policy != pins[0][2].policy
        ):
            raise RuntimeError("joint stream base or correction prefix incomplete")
        guard()
        report["complete"] = True
    except Exception as error:
        report["complete"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        final_candidate = None
        report["unreturned_stream_work_unknown"] = report["stream"] is None
    finally:
        try:
            if final_candidate is not None:
                guard()
        except Exception as error:
            report["complete"] = False
            report["error"] = f"{type(error).__name__}: {error}"
            final_candidate = None
    return ContinuousLearnabilityResult(deepcopy(report), final_candidate)
