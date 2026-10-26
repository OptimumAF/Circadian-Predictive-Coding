"""Bounded recurrent RL recipes through the existing generic actor/inbox runtime.

The recipe records actual on-policy training, then uses existing checkpoint
scoring/live evaluation. A separate warm-start mode admits typed train-only
demonstrations before sampled rewards. Replay, selection, promotion, final
feedback and changes to the native objectives remain outside these recipes.
"""

from copy import deepcopy
from dataclasses import asdict
from math import isfinite
from time import monotonic
from typing import Any

import numpy as np

from src.adapters import key_door_pilot as shared
from src.adapters.key_door_physical_evaluation import evaluate_frozen_physical_checkpoints
from src.adapters.torch_demonstration_reward_learner import (
    ClassWeightedDemonstrationUpdate,
    DemonstrationUpdate,
    TorchDemonstrationRewardLearner,
)
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_recurrent_learner import TORCH_RECURRENT_CE_ID, TorchRecurrentPolicy
from src.adapters.torch_recurrent_weighted import TORCH_RECURRENT_WEIGHTED_CE_ID
from src.adapters.torch_reward_update_learner import (
    LegalRewardUpdate,
    RewardUpdate,
    TorchRewardUpdateLearner,
)
from src.adapters.torch_recurrent_reward import TORCH_RECURRENT_LEGAL_REWARD_ID
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.app.coupled_attribution import _score_checkpoints, _validate_batch
from src.app.key_door_training import KeyDoorTrainingFailed, collect_key_door_reward_episode
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.key_door_navigation import KeyDoorLayout, key_door_layout_record
from src.core.key_door_instances import seeded_action_remapping_layouts
from src.core.physical_interaction_quota import PhysicalInteractionQuota
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch


ACQUIRED_PREFIX_UPDATES = 192


def action_remapping_layouts(
    environment_seed: int | None = None, heldout_environment_seed: int | None = None
) -> tuple[KeyDoorLayout, ...]:
    """Change only B's movement controls; spatial inputs and A/C geometry agree."""
    if environment_seed is not None or heldout_environment_seed is not None:
        if type(environment_seed) is not int or type(heldout_environment_seed) is not int:
            raise ValueError("environment seeds must be a pair of native integers in0..23")
        return seeded_action_remapping_layouts(environment_seed, heldout_environment_seed)
    return (
        KeyDoorLayout("A"),
        KeyDoorLayout("B", movement_action_map=(1, 2, 3, 0)),
        KeyDoorLayout("C"),
        KeyDoorLayout("D", branch_row=1),
    )


def _row_layouts(row: Any) -> tuple[KeyDoorLayout, ...]:
    law = row.get("action_mapping")
    if law not in (None, "movement_permutation_v1"):
        raise ValueError("unknown movement action mapping")
    seeds = row.get("environment_seeds")
    if seeds is not None:
        if (
            not law
            or row.get("interaction_quota_mode") != "twelve_steps_per_phase_v1"
            or type(seeds) is not dict
            or set(seeds) != {"training", "final_only"}
        ):
            raise ValueError("environment seeds require the explicit mapped quota route")
        return action_remapping_layouts(seeds["training"], seeds["final_only"])
    return action_remapping_layouts() if law else shared.layouts()


def _environment_seed_options(
    environment_seed: int | None, heldout_environment_seed: int | None, quota: bool
) -> dict[str, Any]:
    if environment_seed is None and heldout_environment_seed is None:
        return {}
    if not quota:
        raise ValueError("environment seeds require the explicit mapped quota route")
    # Validate both values before sampler/model/clock construction, without using RNG.
    action_remapping_layouts(environment_seed, heldout_environment_seed)
    return dict(
        environment_seed=environment_seed, heldout_environment_seed=heldout_environment_seed
    )


def describe_protocol(
    *,
    condition_on_legal_actions: bool = False,
    action_remapping: bool = False,
    fill_interaction_quota: bool = False,
    environment_seed: int | None = None,
    heldout_environment_seed: int | None = None,
    frozen_recurrent_control: bool = False,
) -> dict[str, Any]:
    if type(condition_on_legal_actions) is not bool:
        raise ValueError("condition_on_legal_actions must be an exact bool")
    if type(action_remapping) is not bool or action_remapping and not condition_on_legal_actions:
        raise ValueError("action_remapping requires an exact bool and the legal reward route")
    if type(fill_interaction_quota) is not bool or fill_interaction_quota and not action_remapping:
        raise ValueError("interaction quota requires the explicit legal action-remapping route")
    seed_options = _environment_seed_options(
        environment_seed, heldout_environment_seed, fill_interaction_quota
    )
    if type(frozen_recurrent_control) is not bool or (
        frozen_recurrent_control and not fill_interaction_quota
    ):
        raise ValueError("frozen recurrent control requires the explicit mapped quota route")
    p = shared.describe_recurrent_protocol()
    p.update(
        protocol_id="actual_key_door_on_policy_recurrent_rl_v1",
        classification="exploratory_separate_recurrent_reward_learning",
        architecture="Elman tanh/full episode BPTT/reward logprob sum; zero hidden per packet",
        methods=["recurrent_rl"],
        replay_modes=[False],
        sampling_seed=29,
        training="eight actual sampled A+/A-/B+/B-/C+/C-/A+/A- episodes, max12 decisions each",
        teacher="eight final_test packets only; never training, collection or action repair",
        label_delay=1,
        release="matching episode ID only, no teacher class targets",
        replay=None,
        episode_schema=None,
        encoded_packet_array_bytes=None,
        native_defaults=dict(learning_rate=0.03, momentum=0.9, max_episode_steps=16, discount=1.0),
        objective="negative sum gamma^t discounted reward-to-go times log action probability; zero-reset BPTT/native SGD",
        sampling="one private PCG64 uniform per valid decision, raw cumulative mass; no mask, retry or teacher",
        baseline_reference=[
            "artifacts/runs/r42-key-door-20261016/pilot.json",
            "artifacts/runs/r42-recurrent-key-door-20261016/pilot.json",
        ],
        comparisons="reuse old results only; different objective/on-policy variable exposure, no equal work, architecture or causal comparison",
        limitations="one seed/eight episodes/local game; no useful learning, generalization, CPC advantage or full R4.2 guarantee",
    )
    p["planned_work"] = dict(
        wake_updates=8,
        native_train_calls=8,
        maximum_training_rows=96,
        bp_steps=8,
        pc_family_steps=0,
        replay_calls=0,
        probes=0,
        svd=0,
        resumes=0,
        collection_episodes=8,
        maximum_collection_forwards=96,
        score_forwards=40,
        score_rows=320,
        live_episodes=40,
        live_sessions=40,
        maximum_live_forwards=480,
        maximum_native_forwards=616,
        maximum_recurrent_unroll_steps=992,
        demonstration_steps=64,
        environment_constructions=56,
        maximum_environment_steps=640,
    )
    if condition_on_legal_actions:
        p.update(
            protocol_id="actual_key_door_legal_on_policy_recurrent_rl_v1",
            classification="exploratory_legal_conditioned_recurrent_reward_learning",
            action_conditioning="legal_actions_v1",
            sampling="one private PCG64 draw per decision from raw policy conditioned on actual legal support; no fallback or repair",
            objective="negative sum gamma^t discounted reward-to-go times conditional log action probability; zero-reset BPTT/native SGD",
            diagnostic=TORCH_RECURRENT_LEGAL_REWARD_ID,
            evaluation="after all8validated commits, construct final packets; raw-policy teacher CE/accuracy diagnostics and greedy first-tie conditional live decisions",
            score_scope="teacher CE and accuracy measure raw categorical probabilities; not conditional policy likelihood or live success",
            comparisons="legal support is additional information; historical unmasked outcomes are not matched controls; no comparative learner benefit established",
        )
    if action_remapping:
        p.update(
            protocol_id="actual_key_door_legal_action_remapping_v1",
            classification="engineering_action_remapping_qualification",
            action_mapping="movement_permutation_v1",
            layouts=[key_door_layout_record(layout) for layout in action_remapping_layouts()],
            observation_basis="spatial_directions_v1; channels do not follow action remapping",
            intervention="A/C identity, B movement controls (1,2,3,0); A/B/C geometry, keys and appearance identical; D heldout branch_row1",
            comparisons="no comparative experiment; legal support is extra information and may force corridor actions; own policy trajectories required for future equal-interaction repeated-seed comparisons",
            limitations="one engineering seed/unequal realized episode lengths; no learned remapping, useful adaptation, retention, generalization, CPC advantage or full R4.3 claim",
        )
    if fill_interaction_quota:
        p.update(
            protocol_id="actual_key_door_legal_interaction_quota_v1",
            classification="engineering_equal_physical_interaction_qualification",
            interaction_quota="twelve_steps_per_phase_v1",
            training="eight original context/cue phases, exactly12 returned physical steps each across fresh episodes; each episode trains once with original reward objective",
            checkpoint_updates=None,
            checkpoint_physical_steps=[0, 24, 48, 72, 96],
            phase_native_updates="record actual variable native indices at each physical boundary; never relabel them as fixed update indices",
            quota_failure="failed physical attempts stay charged and stop the arm; no replacement after unknown physics",
            comparisons="equal96physical TRAIN steps, variable8..16SGD/episode resets and unequal computation; no comparative experiment or repeated-instance claim",
        )
        p["planned_work"].update(
            wake_updates=None,
            native_train_calls=None,
            bp_steps=None,
            minimum_native_train_calls=8,
            maximum_native_train_calls=16,
            exact_training_rows=96,
            collection_episodes=None,
            maximum_collection_episodes=16,
            environment_constructions=None,
            maximum_environment_constructions=64,
        )
    if seed_options:
        p.update(
            protocol_id="actual_key_door_seeded_interaction_quota_v1",
            environment_seeds=dict(training=environment_seed, final_only=heldout_environment_seed),
            environment_catalogue="rotation=s%4; branch_row=1+(s//4)%3; swap_keys=bool(s//12); appearance=0; exact distinct native seeds0..23",
            layouts=[key_door_layout_record(x) for x in action_remapping_layouts(**seed_options)],
            intervention="A/C identity, B movement controls (1,2,3,0); A/B/C same training instance; D distinct final-only instance",
            limitations="one explicitly declared catalogue instance pair; no matched repeated-instance experiment, useful adaptation, generalization or CPC advantage",
        )
    if frozen_recurrent_control:
        p.update(
            protocol_id="actual_key_door_frozen_interaction_quota_v1",
            classification="engineering_nonlearning_control",
            learning_mode="frozen_recurrent_control_v1",
            architecture="initialized Elman tanh; frozen parameters and optimizer; zero hidden per episode; no BPTT",
            diagnostic=None,
            label_delay=None,
            release=None,
            checkpoint_native_updates=[0, 0, 0, 0, 0],
            phase_native_updates="all0; keep separate physical stage identifiers",
            training="no training; collect own96physical steps across eight12step phases using permanently frozen initialized recurrent weights/optimizer",
            evaluation="after all96validated physical collection steps, score five distinct physical stages and40fresh live episodes; actual native index0 throughout",
            objective="none; prepared reward packets never enter native training or actor inbox",
            comparisons="nonlearning control only; same physical allowance but unequal computation, no trained baseline or matched comparative experiment",
            limitations="engineering nonlearning control; no useful adaptation, trained-baseline equivalence, generalization, CPC advantage or full R4.3 claim",
        )
        p["planned_work"].update(
            wake_updates=0,
            native_train_calls=0,
            bp_steps=0,
            minimum_native_train_calls=0,
            maximum_native_train_calls=0,
            maximum_training_rows=0,
            exact_training_rows=0,
            exact_collection_rows=96,
            maximum_recurrent_unroll_steps=896,
        )
    return p


def describe_warm_start_protocol() -> dict[str, Any]:
    """A distinct fixed schedule; its scientific comparison remains unexecuted."""
    p = describe_protocol()
    p.update(
        protocol_id="actual_key_door_demonstration_then_reward_v1",
        classification="exploratory_warm_start_recurrent_reward_learning",
        architecture="Elman tanh/full episode BPTT; demonstration mean CE then reward logprob sum",
        training="eight train-role demonstrations, then eight actual sampled episodes; same A+/A-/B+/B-/C+/C-/A+/A- sequence in each phase",
        teacher="eight train-role warmup packets and eight disjoint-ID final_test packets; final packets never train or select",
        phase_schedule=["demonstration"] * 8 + ["reward"] * 8,
        checkpoint_updates=[0, 8, 10, 12, 14, 16],
        evaluation="all six complete owned checkpoints; argmax first tie, no masking/repair/retry; after all sixteen commits",
        ce_directions=dict(
            new_B_gain="mean B CE8 minus CE12",
            new_C_gain="mean C CE8 minus CE14",
            retention_loss="mean A CE14 minus CE10",
            return_gain="mean A CE14 minus CE16",
        ),
        release="matching episode ID only; exact typed demonstration or reward packet",
        objective="native demonstration mean CE, then unchanged gamma1 reward logprob sum; same optimizer/gradients with no reset",
        comparison_gate="requires separately frozen train-only chosen protocol, same-architecture/same-demonstration warm-start BC-only control and momentum accounting",
        outcome_scope="reward-stage CE changes on previously demonstrated contexts; original signed directions offset by eight updates, no new-task acquisition claim",
        limitations="one seed/eight demonstration and eight sampled episodes/local game; no matched comparison, useful learning, generalization, CPC advantage or full R4.2 guarantee",
    )
    p["planned_work"] = dict(
        wake_updates=16,
        native_train_calls=16,
        maximum_training_rows=160,
        demonstration_updates=8,
        demonstration_training_rows=64,
        reward_updates=8,
        maximum_reward_training_rows=96,
        bp_steps=16,
        pc_family_steps=0,
        replay_calls=0,
        probes=0,
        svd=0,
        resumes=0,
        collection_episodes=8,
        maximum_collection_forwards=96,
        score_forwards=48,
        score_rows=384,
        live_episodes=48,
        live_sessions=48,
        maximum_live_forwards=576,
        maximum_native_forwards=720,
        maximum_recurrent_unroll_steps=1216,
        demonstration_steps=128,
        environment_constructions=72,
        maximum_environment_steps=800,
    )
    return p


def _training_slots(warm_start: bool = False) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = [
        dict(
            index=2 * i + j,
            context=context,
            cue=cue,
            status="not_run",
            reason="not reached",
            collection=None,
            update_attempted=False,
            update_committed=False,
            commit_validated=False,
            receipt=None,
            native_stages=[],
            charged_updates_before=None,
            charged_updates_after=None,
            native_updates_before=None,
            native_training_examples_before=None,
            sampler_before=None,
            sampler_after=None,
            readback_error=None,
        )
        for i, context in enumerate(shared.TASKS)
        for j, cue in enumerate((1, -1))
    ]
    if not warm_start:
        return slots
    warm = deepcopy(slots)
    for slot in warm:
        slot.update(phase="demonstration", demonstration=None, training_rows=None)
    for slot in slots:
        slot.update(index=slot["index"] + 8, phase="reward", demonstration=None, training_rows=None)
    return warm + slots


def _serialize_collection(report: dict[str, Any]) -> dict[str, Any]:
    state = report["behavior_state"]
    return {
        **{k: deepcopy(v) for k, v in report.items() if k not in ("episode", "behavior_state")},
        "episode": None if report["episode"] is None else asdict(report["episode"]),
        "behavior_native_updates": None if state is None else state.completed_updates,
        "behavior_training_examples": None if state is None else state.training_examples,
    }


def _binding(budget: Any) -> tuple[Any, ...]:
    return (
        id(budget),
        id(budget.budget),
        deepcopy(budget.budget),
        id(budget.clock),
        id(budget.progress),
        budget.started_at,
    )


def _check_budget(
    budget: Any, birth: Any, updates: int, minimum_clock: float | None = None
) -> None:
    if (
        _binding(budget) != birth
        or type(budget.updates_completed) is not int
        or budget.updates_completed != updates
        or type(budget.replay_examples_completed) is not int
        or budget.replay_examples_completed != 0
        or type(budget.last_clock) is not float
        or not isfinite(budget.last_clock)
        or budget.last_clock < (budget.started_at if minimum_clock is None else minimum_clock)
    ):
        raise ValueError("original RL budget binding or observed progress changed")


def _prepare_update(
    row: Any,
    source: Any,
    state: Any,
    rng: Any,
    budget: Any,
    index: int,
    identity: str,
    *,
    interaction_quota: PhysicalInteractionQuota | None = None,
    on_collection: Any = None,
    before_collection_prediction: Any = None,
) -> tuple[Any, int, str]:
    slot = row["training"][index]
    conditioning = row.get("action_conditioning")
    if conditioning not in (None, "legal_actions_v1"):
        raise ValueError("unknown action conditioning")
    layout = next(x for x in _row_layouts(row) if x.layout_id == slot["context"])
    if row.get("action_mapping") and slot.get("layout") != key_door_layout_record(layout):
        raise ValueError("training layout differs from the declared movement action mapping")
    if slot.get("phase") == "demonstration":
        objective = row.get("demonstration_objective")
        row["demonstration_collection_attempts"] += 1
        batch = shared._collect(identity, layout, slot["cue"], "train")
        if batch.episode is None:
            raise ValueError("warmup training requires a complete demonstration packet")
        if row.get("demonstration_objective") != objective:
            raise ValueError("demonstration objective changed during collection")
        slot["demonstration"] = asdict(batch.episode)
        update: Any = DemonstrationUpdate(batch.episode)
        rows = len(batch.episode.transitions)
        diagnostic = TORCH_RECURRENT_CE_ID
        if objective == TORCH_RECURRENT_WEIGHTED_CE_ID:
            update = ClassWeightedDemonstrationUpdate(batch.episode)
            slot["class_weights"] = list(update.class_weights)
            diagnostic = TORCH_RECURRENT_WEIGHTED_CE_ID
    else:
        collector = source.fork()
        if collector is source:
            raise ValueError("RL collection requires a distinct owned policy fork")
        collector.restore_state(deepcopy(state))
        row["collection_attempts"] += 1
        options: dict[str, Any] = {"condition_on_legal_actions": True} if conditioning else {}
        if interaction_quota is not None:
            options["interaction_quota"] = interaction_quota

        def before_prediction() -> None:
            budget.before_final()
            before_collection_prediction()

        report = collect_key_door_reward_episode(
            collector,
            layout,
            slot["cue"],
            episode_id=identity,
            sampler=rng,
            before_prediction=before_prediction
            if before_collection_prediction is not None
            else budget.before_final,
            **options,
        )
        slot["collection"] = _serialize_collection(report)
        if on_collection is not None:
            on_collection()
        if row.get("action_mapping"):
            # Identity contexts intentionally retain their legacy collection shape.
            expected_layout = key_door_layout_record(layout)
            if report.get("layout") != (
                expected_layout if "movement_action_map" in expected_layout else None
            ):
                raise ValueError("collected action mapping differs from the declared layout")
        if not _same_state(state, report["behavior_state"]) or not _same_state(
            state, collector.snapshot_state()
        ):
            raise ValueError("collection changed captured or owned behavior checkpoint")
        if row.get("action_conditioning") != conditioning:
            raise ValueError("action conditioning changed during collection")
        if conditioning:
            if report.get("action_conditioning") != conditioning:
                raise ValueError("collection action conditioning differs from declared law")
            update = LegalRewardUpdate(report["episode"], report["behavior_state"])
            rows = len(update.episode.episode.transitions)
            diagnostic = TORCH_RECURRENT_LEGAL_REWARD_ID
        else:
            update = RewardUpdate(report["episode"], report["behavior_state"])
            rows = len(report["episode"].transitions)
            diagnostic = "torch_recurrent_reward_logprob_sum_v1"
    if "phase" in slot:
        slot["training_rows"] = rows
    return update, rows, diagnostic


def _train_one(
    row: Any,
    source: Any,
    runtime: Any,
    state: Any,
    rng: Any,
    budget: Any,
    birth: Any,
    index: int,
    clock: LogicalClock,
    *,
    check_boundary: Any = None,
    interaction_quota: PhysicalInteractionQuota | None = None,
    on_collection: Any = None,
    before_collection_prediction: Any = None,
) -> Any:
    slot = row["training"][index]
    identity = (
        f"warmup-train-{index}"
        if slot.get("phase") == "demonstration"
        else f"reward-train-{index - 8 if slot.get('phase') == 'reward' else index}"
    )
    slot.update(
        status="running",
        reason=None,
        charged_updates_before=budget.updates_completed,
        native_updates_before=state.completed_updates,
        native_training_examples_before=state.training_examples,
        sampler_before=deepcopy(rng.bit_generator.state),
    )
    committed, problem = None, None
    last_clock = budget.last_clock
    try:
        _check_budget(budget, birth, index)
        budget.before_update()
        if interaction_quota is not None and check_boundary is not None:
            check_boundary()
        update, rows, diagnostic = _prepare_update(
            row,
            source,
            state,
            rng,
            budget,
            index,
            identity,
            **({"interaction_quota": interaction_quota} if interaction_quota is not None else {}),
            **(
                {
                    "on_collection": on_collection,
                    "before_collection_prediction": before_collection_prediction,
                }
                if interaction_quota is not None
                else {}
            ),
        )
        if check_boundary is not None:
            check_boundary()
        runtime.record_experience(
            Experience(
                identity,
                identity,
                index,
                "a0",
                update,
                "train",
                ExperiencePermissions(training=True),
            )
        )
        clock.advance_to(index + 1)
        runtime.record_label(
            LabelArrival(f"release-{index}", identity, identity, index + 1, "a0", identity, "train")
        )

        def observe(stage: Any, read: Any) -> None:
            nonlocal committed
            origin = read()
            slot["native_stages"].append(stage)
            if stage == "started":
                slot["update_attempted"] = True
            if origin.receipt is not None:
                # A committed receipt remains evidence even if the boundary now refuses.
                slot.update(receipt=asdict(origin.receipt), update_committed=True)
            if check_boundary is not None:
                check_boundary()
            if origin.receipt is not None:
                observed = origin.learner.snapshot_state()
                expected_rows = state.training_examples + rows
                if (
                    origin.completed_updates != index + 1
                    or origin.receipt.update_number != index + 1
                    or observed.completed_updates != index + 1
                    or observed.training_examples != expected_rows
                    or origin.targets != identity
                    or origin.source.sample_id != identity
                    or origin.receipt.diagnostic.definition != diagnostic
                ):
                    raise ValueError("RL receipt/native checkpoint progression differs")
                committed = deepcopy(observed)
                slot["commit_validated"] = True

        receipts = runtime.train_ready(max_updates=1, native_observer=observe)
        _check_budget(budget, birth, index + 1, last_clock)
        if (
            len(receipts) != 1
            or slot["receipt"] != asdict(receipts[0])
            or committed is None
            or not slot["commit_validated"]
        ):
            raise ValueError("RL update did not return the validated original receipt")
        if not _same_state(committed, runtime.candidate_snapshot().state):
            raise ValueError("candidate changed after the preceding RL commit")
        slot.update(status="complete", reason=None)
        return committed
    except Exception as error:
        problem = error
        if isinstance(error, KeyDoorTrainingFailed):
            slot["collection"] = _serialize_collection(error.report)
        slot.update(status="failed", reason=f"{type(error).__name__}: {error!r}")
        raise
    finally:
        try:
            slot["charged_updates_after"] = budget.updates_completed
            slot["sampler_after"] = deepcopy(rng.bit_generator.state)
            _check_budget(budget, birth, index + int(slot["update_committed"]), last_clock)
            if (
                slot["collection"] is not None
                and slot["sampler_after"] != slot["collection"]["sampler_after"]
            ):
                raise ValueError("sampler changed after recorded collection")
        except Exception as readback:
            slot["readback_error"] = f"readback {type(readback).__name__}: {readback!r}"
            if problem is None:
                slot.update(status="failed", reason=slot["readback_error"])
                raise


def _phase_work(training: Any, *, frozen_control: bool = False) -> dict[str, Any]:
    if type(training) is not list or len(training) != (8 if frozen_control else 16):
        raise ValueError("phase summary requires all original training slots")
    for i, slot in enumerate(training):
        if (
            slot["phase"] != ("demonstration" if i < 8 else "reward")
            or type(slot["update_committed"]) is not bool
            or type(slot["commit_validated"]) is not bool
            or (
                slot["update_committed"]
                and (type(slot["training_rows"]) is not int or not 1 <= slot["training_rows"] <= 16)
            )
        ):
            raise ValueError("phase summary cannot account for corrupted committed evidence")
    return {
        phase: dict(
            completed_updates=sum(s["update_committed"] for s in training if s["phase"] == phase),
            validated_commits=sum(s["commit_validated"] for s in training if s["phase"] == phase),
            training_rows=sum(
                s["training_rows"]
                for s in training
                if s["phase"] == phase and s["update_committed"]
            ),
        )
        for phase in ("demonstration", "reward")
    }


def _collect_frozen_one(
    row: Any,
    source: Any,
    state: Any,
    rng: Any,
    budget: Any,
    birth: Any,
    index: int,
    *,
    check_boundary: Any,
    interaction_quota: PhysicalInteractionQuota,
    on_collection: Any,
    before_collection_prediction: Any,
    native_origin: int = 0,
) -> Any:
    """Validate paid own-policy collection without submitting work to the inbox.

    Why: a frozen control must preserve optimizer/counter state, not manufacture
    empty training receipts or advance a native update counter for physical work.
    """
    slot = row["training"][index]
    slot.update(
        status="running",
        reason=None,
        collection_only=True,
        collection_validated=False,
        charged_updates_before=budget.updates_completed,
        native_updates_before=state.completed_updates,
        native_training_examples_before=state.training_examples,
        sampler_before=deepcopy(rng.bit_generator.state),
    )
    try:
        _check_budget(budget, birth, native_origin)
        budget.before_final()
        check_boundary()
        _prepare_update(
            row,
            source,
            state,
            rng,
            budget,
            index,
            f"frozen-collect-{index}",
            interaction_quota=interaction_quota,
            on_collection=on_collection,
            before_collection_prediction=before_collection_prediction,
        )
        check_boundary()
        slot.update(status="complete", reason=None, collection_validated=True)
        return deepcopy(state)
    except Exception as error:
        if isinstance(error, KeyDoorTrainingFailed):
            slot["collection"] = _serialize_collection(error.report)
        slot.update(status="failed", reason=f"{type(error).__name__}: {error!r}")
        raise
    finally:
        slot["charged_updates_after"] = budget.updates_completed
        slot["sampler_after"] = deepcopy(rng.bit_generator.state)
        try:
            _check_budget(budget, birth, native_origin)
            if slot["collection"] is not None and (
                slot["sampler_after"] != slot["collection"]["sampler_after"]
            ):
                raise ValueError("frozen sampler changed after observed collection")
        except Exception as error:
            slot.update(status="failed", readback_error=f"{type(error).__name__}: {error}")
            raise


def _readback_arm(
    row: Any,
    source: Any,
    runtime: Any,
    budget: Any,
    before: Any,
    rng: Any,
    *,
    warm_start: bool,
    frozen_control: bool,
) -> None:
    row["wake_updates"] = 0 if budget is None else budget.updates_completed
    try:
        if rng is not None:
            row["sampling_state"] = deepcopy(rng.bit_generator.state)
        if source is not None and before is not None:
            row["source_unchanged"] = _same_state(before, source.snapshot_state())
        if runtime is not None and before is not None:
            row["actor_unchanged"] = _same_state(before, runtime.actor.snapshot_state().state)
        if row["source_unchanged"] is False or row["actor_unchanged"] is False:
            raise ValueError("observed source or serving actor isolation changed")
    except Exception as error:
        row["readback_error"] = f"readback {type(error).__name__}: {error!r}"
        if row["status"] != "failed":
            row.update(status="failed", reason=row["readback_error"])
    row["work"] = shared._recurrent_work(row)
    if row.get("learning_mode") == "frozen_recurrent_control_v1":
        collections = [s["collection"] for s in row["training"] if s["collection"] is not None]
        row["work"].update(
            collection_unroll_steps=sum(c["prediction_calls_returned"] for c in collections),
            collection_physical_steps=sum(c["environment_steps"] for c in collections),
            collection_physical_attempts=sum(c["environment_steps_attempted"] for c in collections),
            collection_episodes=row["collection_attempts"],
            native_training_updates=budget.updates_completed if budget is not None else None,
        )
    if warm_start:
        try:
            row["phase_work"] = _phase_work(row["training"], frozen_control=frozen_control)
        except Exception as error:
            row["phase_work"] = None
            row["phase_work_error"] = f"phase work {type(error).__name__}: {error!r}"
            if row["status"] != "failed":
                row.update(status="failed", reason=row["phase_work_error"])
        row["phase_work_scope"] = (
            "committed native updates/rows only; partial failed work requires execution receipt, never zero-imputed"
        )


def _quota_outcomes(scores: Any, boundaries: Any) -> dict[str, float]:
    """Read fixed physical boundaries without renaming the actual native clocks."""
    if [point["physical_steps"] for point in boundaries] != [0, 24, 48, 72, 96]:
        raise ValueError("quota outcomes require every original physical boundary")
    by_native = {point["updates"]: point["scores"] for point in scores}
    by_physical = {
        point["physical_steps"]: by_native[point["native_updates"]] for point in boundaries
    }

    def mean(steps: int, context: str) -> float:
        return sum(by_physical[steps][f"{context}-{color}"]["ce"] for color in ("red", "blue")) / 2

    return dict(
        new_B_gain=mean(0, "B") - mean(48, "B"),
        new_C_gain=mean(0, "C") - mean(72, "C"),
        retention_loss=mean(72, "A") - mean(24, "A"),
        return_gain=mean(72, "A") - mean(96, "A"),
    )


def _evaluate_arm(
    row: Any,
    source: Any,
    states: Any,
    final: Any,
    budget: Any,
    *,
    warm_start: bool,
    frozen_control: bool,
    check_boundary: Any = None,
) -> None:
    contexts = tuple(
        f"{s['context']}-{'red' if s['cue'] == 1 else 'blue'}" for s in row["training"]
    )
    if check_boundary is not None:
        check_boundary()
    scores = _score_checkpoints(
        source,
        states,
        (),
        final,
        budget,
        training_contexts=contexts,
        **(
            {"check_boundary": check_boundary, "verify_state": True}
            if check_boundary is not None
            else {}
        ),
    )
    if check_boundary is not None:
        check_boundary()
    if row.get("acquired_prefix"):
        for score, point in zip(scores, row["quota_checkpoints"], strict=True):
            score["physical_steps"] = point["physical_steps"]
    row["stream_result"] = dict(
        completed_updates=states[-1][0],
        training_examples=states[-1][1].training_examples,
        score_forward_calls=len(states) * len(final),
        checkpoints=scores,
        applied_updates=[s["receipt"] for s in row["training"]],
        num_classes=5,
    )
    if row.get("interaction_quota_mode"):
        row["outcomes"] = _quota_outcomes(scores, row["quota_checkpoints"])
    elif frozen_control:
        row["outcomes"] = None  # No reward phase or invented native16 checkpoint.
    elif warm_start:
        reward_scores = [dict(p, updates=p["updates"] - 8) for p in scores if p["updates"] >= 8]
        row["outcomes"] = shared._outcomes(dict(checkpoints=reward_scores))
    else:
        row["outcomes"] = shared._outcomes(row["stream_result"])
    live_options: dict[str, Any] = (
        {"declared_layouts": _row_layouts(row)} if row.get("action_mapping") else {}
    )
    shared._live(
        row,
        source,
        states,
        budget,
        **live_options,
        **(
            {"condition_on_legal_actions": True, "check_boundary": check_boundary}
            if row.get("action_conditioning")
            else {}
        ),
    )


def _run_arm(
    row: dict[str, Any],
    final: Any,
    deadline: float,
    sampler: Any,
    *,
    warm_start: bool = False,
    frozen_control: bool = False,
    warm_state: list[Any] | None = None,
    evaluation_queue: list[Any] | None = None,
    completed_checks: list[Any] | None = None,
    check_preceding: Any = None,
    check_final: Any = None,
    checkpoint_probe: Any = None,
    probe_only: bool = False,
    quota_pair_initial: list[Any] | None = None,
) -> None:
    source = runtime = budget = before = rng = None
    states: list[Any] = []
    quotas: tuple[PhysicalInteractionQuota, ...] = ()
    final_probe_guard: Any = None
    try:
        if frozen_control and not warm_start:
            raise ValueError("frozen BC control requires typed demonstration warm start")
        conditioning = row.get("action_conditioning")
        mapping = row.get("action_mapping")
        quota_mode = row.get("interaction_quota_mode")
        environment_seeds = deepcopy(row.get("environment_seeds"))
        learning_mode = row.get("learning_mode")
        if learning_mode not in (None, "frozen_recurrent_control_v1"):
            raise ValueError("unknown recurrent learning mode")
        nonlearning = learning_mode == "frozen_recurrent_control_v1"
        prefix_mode = row.get("acquired_prefix")
        if prefix_mode not in (None, "a_only_192_v1"):
            raise ValueError("unknown acquired prefix")
        prefix_updates = ACQUIRED_PREFIX_UPDATES if prefix_mode else 0
        if quota_pair_initial is not None and (
            type(quota_pair_initial) is not list
            or len(quota_pair_initial) != (0 if nonlearning else 1)
            or not quota_mode
            or prefix_updates
            or warm_start
            or frozen_control
            or probe_only
            or checkpoint_probe is not None
            or evaluation_queue is None
            or completed_checks is None
            or not callable(check_final)
        ):
            raise ValueError("quota pairing requires separate deferred frozen/reward arms")
        if type(probe_only) is not bool or (
            checkpoint_probe is not None and not callable(checkpoint_probe)
        ):
            raise ValueError("probe-only mode requires an exact bool and callable checkpoint probe")
        if (
            probe_only != (checkpoint_probe is not None)
            or probe_only
            and (
                not prefix_updates
                or nonlearning
                or final is not None
                or evaluation_queue is not None
                or completed_checks is not None
                or warm_state is not None
                or warm_start
                or frozen_control
            )
        ):
            raise ValueError(
                "probe-only mode requires standalone acquired reward quotas and no finals"
            )
        if prefix_updates and (not quota_mode or warm_start or frozen_control):
            raise ValueError("acquired prefix requires the standalone mapped quota recipe")
        frozen_prefix = expected_prefix = None
        if (
            quota_mode not in (None, "twelve_steps_per_phase_v1")
            or quota_mode
            and (
                not mapping
                or warm_start
                or frozen_control
                or evaluation_queue is not None
                and not prefix_updates
                and quota_pair_initial is None
            )
        ):
            raise ValueError("physical quota requires standalone mapped reward collection")
        if nonlearning and not quota_mode:
            raise ValueError("frozen recurrent collection requires physical quotas")
        _row_layouts(row)
        if mapping and (warm_start or frozen_control or not conditioning):
            raise ValueError("action remapping requires standalone legal reward collection")
        if conditioning not in (None, "legal_actions_v1") or (
            conditioning and warm_start and completed_checks is None
        ):
            raise ValueError("legal conditioning requires the standalone reward route")
        if check_preceding is not None:
            check_preceding()
        objective = row.get("demonstration_objective")
        if objective is not None and objective not in (
            TORCH_RECURRENT_CE_ID,
            TORCH_RECURRENT_WEIGHTED_CE_ID,
        ):
            raise ValueError("unknown demonstration objective")
        total = 16 if warm_start and not frozen_control else 8
        maximum_updates = (16 if quota_mode else total) + prefix_updates
        if quota_mode:
            quotas = tuple(PhysicalInteractionQuota(12) for _ in range(8))
        quota_binding = tuple((id(quota), quota.maximum_steps) for quota in quotas)
        expected_quotas = tuple(asdict(quota.snapshot()) for quota in quotas)
        active_index = -1
        active_phase = -1
        active_collection_seal: Any = None
        phase_templates = deepcopy(row["training"][prefix_updates:])
        expected_boundaries = deepcopy(row.get("quota_checkpoints"))
        probe_checks: list[Any] = []
        points = (
            (0, 8) if frozen_control else (0, 8, 10, 12, 14, 16) if warm_start else shared.POINTS
        )
        if prefix_updates:
            # One original session pays for prefix and continuation. The frozen
            # control keeps the prefix's native charges and optimizer history.
            budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=prefix_updates if nonlearning else maximum_updates,
                    max_wall_seconds=max(0.0, deadline - monotonic()),
                    max_process_rss_bytes=shared.MAX_RSS,
                ),
                monotonic,
            )
            budget.attach_memory(sampler)
            budget.before_final()
        else:
            budget = (
                shared._session(deadline, sampler, max_updates=0)
                if nonlearning
                else (
                    shared._session(deadline, sampler, max_updates=16)
                    if warm_start and not frozen_control or quota_mode
                    else shared._session(deadline, sampler)
                )
            )
        birth = _binding(budget)
        factory = (
            TorchDemonstrationRewardLearner
            if warm_start or prefix_updates
            else TorchRewardUpdateLearner
        )
        source = factory(TorchRecurrentPolicy(8, 8, 5, 17))
        before = source.snapshot_state()
        if quota_pair_initial is not None:
            if nonlearning:
                quota_pair_initial.append(deepcopy(before))
            elif not _same_state(quota_pair_initial[0], before):
                raise ValueError("paired quota native initialization differs from frozen control")
            row["initial_state_matches_control"] = True
        clock = LogicalClock()
        runtime = ActorShadowRuntime(
            source,
            actor_version="a0",
            candidate_version="c0",
            clock=clock,
            budget=budget,
            max_experiences=maximum_updates,
            max_consolidations=0,
        )
        rng = np.random.Generator(np.random.PCG64(29))

        def check_isolation() -> None:
            if (
                row.get("action_conditioning") != conditioning
                or row.get("action_mapping") != mapping
                or row.get("interaction_quota_mode") != quota_mode
                or not _same_state(row.get("environment_seeds"), environment_seeds)
                or row.get("learning_mode") != learning_mode
                or row.get("acquired_prefix") != prefix_mode
                or expected_prefix is not None
                and not _same_state(row.get("prefix"), expected_prefix)
                or nonlearning
                and (row.get("method") != "recurrent_rl" or row.get("replay_enabled") is not False)
                or not _same_state(before, source.snapshot_state())
                or not _same_state(before, runtime.actor.snapshot_state().state)
                or nonlearning
                and (not prefix_updates or frozen_prefix is not None)
                and not _same_state(
                    frozen_prefix if prefix_updates else before, runtime.candidate_snapshot().state
                )
                or quota_pair_initial is not None
                and (
                    len(quota_pair_initial) != 1
                    or not _same_state(quota_pair_initial[0], before)
                    or row.get("initial_state_matches_control") is not True
                )
            ):
                raise ValueError("action conditioning or source/serving actor isolation changed")

        def check_collection_boundary() -> None:
            check_isolation()
            if quota_mode:
                if tuple((id(q), q.maximum_steps) for q in quotas) != quota_binding:
                    raise ValueError("original physical quota binding or cap changed")
                for phase_index, quota in enumerate(quotas):
                    if (
                        phase_index != active_phase
                        and asdict(quota.snapshot()) != expected_quotas[phase_index]
                    ):
                        raise ValueError("inactive physical quota changed before collection")
                if row.get("quota_checkpoints") != expected_boundaries:
                    raise ValueError("preceding physical checkpoint boundary changed")
                if active_index >= 0:
                    actual = row["training"]
                    if not _same_state(
                        expected_training[:active_index], actual[:active_index]
                    ) or not _same_state(
                        expected_training[active_index + 1 :], actual[active_index + 1 :]
                    ):
                        raise ValueError("preceding or unreached quota episode evidence changed")
                    for key in (
                        "index",
                        "context",
                        "cue",
                        "layout",
                        "quota_phase",
                        "episode_in_quota",
                        "phase",
                    ):
                        if not _same_state(
                            actual[active_index].get(key), expected_training[active_index].get(key)
                        ):
                            raise ValueError("active quota episode plan changed")
                    if (
                        nonlearning
                        and active_phase >= 0
                        and actual[active_index].get("collection_only") is not True
                    ):
                        raise ValueError("frozen collection-only plan changed")

        def check_training_boundary() -> None:
            check_collection_boundary()
            for check in probe_checks:
                check()
            if nonlearning and any(
                s["update_attempted"]
                or s["update_committed"]
                or s["commit_validated"]
                or s["receipt"] is not None
                or s["native_stages"]
                for s in row["training"][prefix_updates:]
            ):
                raise ValueError("frozen control acquired native update evidence")
            if quota_mode:
                if tuple((id(q), q.maximum_steps) for q in quotas) != quota_binding:
                    raise ValueError("original physical quota binding or cap changed")
                allowed = deepcopy(expected_quotas)
                if active_index >= 0:
                    collected = active_collection_seal
                    if not _same_state(collected, row["training"][active_index]["collection"]):
                        raise ValueError("observed active collection evidence changed")
                    if collected is not None:
                        expected = allowed[active_phase]
                        attempted = collected["environment_steps_attempted"]
                        completed = collected["environment_steps"]
                        expected["attempted_steps"] += attempted
                        expected["completed_steps"] += completed
                        expected["failed_steps"] += attempted - completed
                        if (
                            collected.get("interaction_quota_before")
                            != expected_quotas[active_phase]
                            or collected.get("interaction_quota_after") != expected
                        ):
                            raise ValueError("collection quota evidence differs from observed work")
                if tuple(asdict(q.snapshot()) for q in quotas) != allowed:
                    raise ValueError("physical quota changed outside recorded collection")
            if check_preceding is not None:
                check_preceding()

        def seal_collection() -> None:
            nonlocal active_collection_seal
            if active_collection_seal is not None:
                raise ValueError("active physical collection was already sealed")
            active_collection_seal = deepcopy(row["training"][active_index]["collection"])
            check_training_boundary()

        previous = runtime.candidate_snapshot().state
        expected_training = deepcopy(row["training"])
        expected_rng = deepcopy(rng.bit_generator.state)
        expected_clock = budget.last_clock

        def probe_at(position: str) -> None:
            nonlocal final_probe_guard
            # Why: probes own another budget. This guard must never observe or
            # advance the original benchmark budget clock during a probe.
            state = deepcopy(previous)
            saved = deepcopy((runtime.applied_updates, states, row["observations"]))
            progress = deepcopy(
                (budget.updates_completed, budget.replay_examples_completed, budget.last_clock)
            )
            owners = (id(runtime._candidate), id(runtime.actor), id(runtime._inbox))

            def guard() -> None:
                check_training_boundary()
                if (
                    monotonic() > deadline
                    or _binding(budget) != birth
                    or not _same_state(
                        progress,
                        (
                            budget.updates_completed,
                            budget.replay_examples_completed,
                            budget.last_clock,
                        ),
                    )
                    or runtime._budget is not budget
                    or runtime._inbox._budget is not budget
                    or runtime._inbox._clock is not clock
                    or clock.now() != state.completed_updates
                    or owners != (id(runtime._candidate), id(runtime.actor), id(runtime._inbox))
                    or not _same_state(state, runtime.candidate_snapshot().state)
                    or not _same_state(state, previous)
                    or not _same_state(expected_training, row["training"])
                    or not _same_state(expected_rng, rng.bit_generator.state)
                    or not _same_state(
                        saved, (runtime.applied_updates, states, row["observations"])
                    )
                ):
                    raise ValueError(
                        "probe changed original quota state, evidence, budget or clock"
                    )

            guard()
            final_probe_guard = guard
            completed = checkpoint_probe(source, state, budget, position, guard)
            if not callable(completed):
                raise ValueError("completed probe must retain a preceding-success guard")
            probe_checks.append(completed)
            guard()

        if probe_only:
            probe_at("initial")
        if not prefix_updates:
            states.append((0, deepcopy(previous)))
            row["observations"].append(shared._recurrent_observation(0, previous))
            if nonlearning:
                row["observations"][-1]["physical_steps"] = 0
        for i in range(maximum_updates):
            if quota_mode and i >= len(row["training"]):
                break
            in_quota = bool(quota_mode and i >= prefix_updates)
            # Prefix collection and native observers must validate preceding and
            # unreached records before admitting the next paid native update too.
            active_index = i if quota_mode else -1
            active_phase = row["training"][i]["quota_phase"] if in_quota else -1
            charged = min(i, prefix_updates) if nonlearning else i
            _check_budget(budget, birth, charged, expected_clock)
            if (
                not _same_state(expected_training, row["training"])
                or expected_rng != rng.bit_generator.state
                or row.get("demonstration_objective") != objective
                or row.get("action_conditioning") != conditioning
            ):
                raise ValueError("preceding RL evidence or sampler progression changed")
            current = runtime.candidate_snapshot().state
            if not _same_state(previous, current) or current.completed_updates != (charged):
                raise ValueError("preceding RL success changed before more collection")
            if conditioning:
                check_training_boundary()
            training_options: dict[str, Any] = (
                {"check_boundary": check_training_boundary} if conditioning else {}
            )
            if in_quota:
                training_options.update(
                    interaction_quota=quotas[active_phase],
                    on_collection=seal_collection,
                    before_collection_prediction=check_collection_boundary,
                )
            if nonlearning and in_quota:
                if prefix_updates:
                    training_options["native_origin"] = prefix_updates
                previous = _collect_frozen_one(
                    row,
                    source,
                    current,
                    rng,
                    budget,
                    birth,
                    i,
                    **training_options,
                )
            else:
                previous = _train_one(
                    row, source, runtime, current, rng, budget, birth, i, clock, **training_options
                )
            if conditioning:
                check_training_boundary()
            if row.get("demonstration_objective") != objective:
                raise ValueError("demonstration objective changed during update")
            if row.get("action_conditioning") != conditioning:
                raise ValueError("action conditioning changed during update")
            other_before = expected_training[:i] + expected_training[i + 1 :]
            other_after = row["training"][:i] + row["training"][i + 1 :]
            if not _same_state(other_before, other_after):
                raise ValueError("prior or unreached RL records changed during update")
            if in_quota:
                quota = quotas[active_phase]
                collected = row["training"][i]["collection"]
                if collected["environment_steps"] <= 0:
                    raise ValueError("quota collection returned no physical steps")
                if quota.remaining_steps:
                    if (
                        row["training"][i]["episode_in_quota"] != 0
                        or len(row["training"]) >= maximum_updates
                    ):
                        raise ValueError("physical quota episode allowance exhausted")
                    extra = deepcopy(phase_templates[active_phase])
                    extra["episode_in_quota"] = 1
                    row["training"].insert(i + 1, extra)
                    for index, slot in enumerate(row["training"]):
                        slot["index"] = index
                elif active_phase % 2 == 1:
                    native_index = prefix_updates if nonlearning else i + 1
                    row["quota_checkpoints"].append(
                        dict(physical_steps=12 * (active_phase + 1), native_updates=native_index)
                    )
                    states.append((native_index, deepcopy(previous)))
                    row["observations"].append(
                        shared._recurrent_observation(native_index, previous)
                    )
                    if nonlearning:
                        row["observations"][-1]["physical_steps"] = 12 * (active_phase + 1)
                expected_quotas = tuple(asdict(q.snapshot()) for q in quotas)
                expected_boundaries = deepcopy(row["quota_checkpoints"])
                active_index = active_phase = -1
                active_collection_seal = None
            expected_training = deepcopy(row["training"])
            expected_rng = deepcopy(rng.bit_generator.state)
            expected_clock = budget.last_clock
            if prefix_updates and i + 1 == prefix_updates:
                frozen_prefix = deepcopy(previous) if nonlearning else None
                row["prefix"] = dict(
                    updates=prefix_updates,
                    training_examples=previous.training_examples,
                    source_role="train",
                    context="A",
                    acquired_skill_verified=False,
                    observation=shared._recurrent_observation(prefix_updates, previous),
                )
                expected_prefix = deepcopy(row["prefix"])
                row["quota_checkpoints"].append(
                    dict(physical_steps=0, native_updates=prefix_updates)
                )
                expected_boundaries = deepcopy(row["quota_checkpoints"])
                states.append((prefix_updates, deepcopy(previous)))
                row["observations"].append(shared._recurrent_observation(prefix_updates, previous))
                if nonlearning:
                    row["observations"][-1]["physical_steps"] = 0
            if warm_state is not None and i + 1 == (prefix_updates or 8):
                if frozen_control or nonlearning:
                    if warm_state:
                        raise ValueError("control warm state was already captured")
                    warm_state.append(deepcopy(previous))
                else:
                    row["warm_start_matches_control"] = len(warm_state) == 1 and _same_state(
                        warm_state[0], previous
                    )
                    if prefix_updates:
                        row["prefix_matches_control"] = row["warm_start_matches_control"]
                    if not row["warm_start_matches_control"]:
                        raise ValueError("full native warm start differs from BC-only control")
            if not quota_mode and i + 1 in points:
                states.append((i + 1, deepcopy(previous)))
                row["observations"].append(shared._recurrent_observation(i + 1, previous))

        if quota_mode:
            total = len(row["training"])
            if not 8 <= total - prefix_updates <= 16 or any(
                q.completed_steps != 12 or q.failed_steps for q in quotas
            ):
                raise ValueError(
                    "physical quota training ended before all original phases completed"
                )
            row["episodes"] = (
                []
                if probe_only
                else [
                    dict(
                        slot,
                        checkpoint_updates=point,
                        **(
                            {
                                "checkpoint_physical_steps": row["quota_checkpoints"][stage][
                                    "physical_steps"
                                ]
                            }
                            if nonlearning or prefix_updates
                            else {}
                        ),
                    )
                    for stage, (point, _) in enumerate(states)
                    for slot in shared._slots()
                    if slot["checkpoint_updates"] == 0
                ]
            )
        seal_completed = completed_checks is not None or bool(quota_mode)

        completed_receipts = deepcopy(runtime.applied_updates) if seal_completed else None
        completed_states = deepcopy(states) if seal_completed else None
        completed_observations = deepcopy(row["observations"]) if seal_completed else None
        native_total = prefix_updates if nonlearning else total

        if probe_only:
            probe_at("end")
            _check_budget(budget, birth, native_total, expected_clock)
            row.update(status="complete", reason=None)
            return

        def check_completed() -> None:
            if completed_checks is not None:
                budget.before_final()
            if conditioning:
                check_isolation()
            if quota_mode:
                check_training_boundary()
            _check_budget(budget, birth, native_total, expected_clock)
            if (
                not _same_state(expected_training, row["training"])
                or expected_rng != rng.bit_generator.state
                or row.get("demonstration_objective") != objective
                or row.get("action_conditioning") != conditioning
                or not _same_state(previous, runtime.candidate_snapshot().state)
            ):
                raise ValueError("completed RL evidence or state changed before final scoring")
            if seal_completed and (
                not _same_state(completed_receipts, runtime.applied_updates)
                or not _same_state(completed_states, states)
                or not _same_state(completed_observations, row["observations"])
                or runtime._budget is not budget
                or runtime._inbox._budget is not budget
                or runtime._inbox._clock is not clock
                or clock.now() != native_total
            ):
                raise ValueError(
                    "completed paired receipts, checkpoints or clock authority changed"
                )

        def guarded_completed() -> None:
            try:
                check_completed()
            except Exception as error:
                row.update(status="failed", reason=f"{type(error).__name__}: {error!r}")
                _readback_arm(
                    row,
                    source,
                    runtime,
                    budget,
                    before,
                    rng,
                    warm_start=warm_start,
                    frozen_control=frozen_control,
                )
                raise

        if completed_checks is not None:
            if len(completed_checks) >= 2:
                raise ValueError("paired evidence retains at most two completed local guards")
            completed_checks.append(guarded_completed)

        def finish() -> None:
            try:
                boundary = check_completed if check_final is None else check_final
                boundary()
                # Why this: the new route never constructs final packets before training ends.
                evaluation_final = final(boundary) if conditioning else final
                boundary()
                if nonlearning:
                    evaluate_frozen_physical_checkpoints(
                        row,
                        source,
                        states,
                        evaluation_final,
                        budget,
                        declared_layouts=_row_layouts(row),
                        check_boundary=boundary,
                        **(
                            {"native_origin": (prefix_updates, previous.training_examples)}
                            if prefix_updates
                            else {}
                        ),
                    )
                else:
                    _evaluate_arm(
                        row,
                        source,
                        states,
                        evaluation_final,
                        budget,
                        warm_start=warm_start,
                        frozen_control=frozen_control,
                        **({"check_boundary": boundary} if conditioning else {}),
                    )
                _check_budget(budget, birth, native_total)
                row.update(status="complete", reason=None)
            except Exception as error:
                row.update(status="failed", reason=f"{type(error).__name__}: {error!r}")
            finally:
                if evaluation_queue is not None:
                    _readback_arm(
                        row,
                        source,
                        runtime,
                        budget,
                        before,
                        rng,
                        warm_start=warm_start,
                        frozen_control=frozen_control,
                    )

        if evaluation_queue is None:
            finish()
        else:
            if len(evaluation_queue) >= 2:
                raise ValueError("paired evaluation holds at most two owned completion closures")
            row.update(status="trained", reason="paired final evaluation deferred")
            evaluation_queue.append(finish)
    except Exception as error:
        row.update(status="failed", reason=f"{type(error).__name__}: {error!r}")
    finally:
        _readback_arm(
            row,
            source,
            runtime,
            budget,
            before,
            rng,
            warm_start=warm_start,
            frozen_control=frozen_control,
        )
        if quotas:
            try:
                row["physical_quotas"] = [asdict(quota.snapshot()) for quota in quotas]
                if row["status"] == "complete" and (
                    tuple(row["physical_quotas"]) != expected_quotas
                    or not _same_state(expected_training, row["training"])
                    or row["quota_checkpoints"] != expected_boundaries
                    or not _same_state(row.get("environment_seeds"), environment_seeds)
                    or row.get("learning_mode") != learning_mode
                ):
                    raise ValueError("completed quota evidence changed during final readback")
            except Exception as error:
                row["quota_readback_error"] = f"{type(error).__name__}: {error}"
                row.update(status="failed", reason=row.get("reason") or row["quota_readback_error"])
                if row.get("work") is not None:
                    row["work"]["failure_work_may_be_unobserved"] = True
                if "physical_quotas" not in row:
                    row["physical_quotas"] = None
        if probe_only and final_probe_guard is not None:
            try:
                # Snapshot/readback ports above are external callbacks too.
                final_probe_guard()
            except Exception as error:
                row.update(
                    status="failed", reason=row.get("reason") or f"{type(error).__name__}: {error}"
                )
                if row.get("work") is not None:
                    row["work"]["failure_work_may_be_unobserved"] = True


def _valid(row: dict[str, Any], *, warm_start: bool = False, frozen_control: bool = False) -> bool:
    total, checkpoints = (8, 2) if frozen_control else (16, 6) if warm_start else (8, 5)
    prefix_updates = ACQUIRED_PREFIX_UPDATES if row.get("acquired_prefix") == "a_only_192_v1" else 0
    if prefix_updates:
        prefix = row["training"][:prefix_updates]
        if (
            len(prefix) != prefix_updates
            or any(
                s["phase"] != "demonstration"
                or s["context"] != "A"
                or type(s["cue"]) is not int
                or s["cue"] != (1 if i % 2 == 0 else -1)
                or s["status"] != "complete"
                or not s["commit_validated"]
                for i, s in enumerate(prefix)
            )
            or row.get("prefix", {}).get("training_examples") != 8 * prefix_updates
        ):
            return False
    if row.get("interaction_quota_mode"):
        total = len(row["training"])
        if (
            not 8 <= total - prefix_updates <= 16
            or row.get("physical_quotas")
            != [dict(maximum_steps=12, attempted_steps=12, completed_steps=12, failed_steps=0)] * 8
        ):
            return False
        if [point["physical_steps"] for point in row["quota_checkpoints"]] != [0, 24, 48, 72, 96]:
            return False
    if row.get("learning_mode") == "frozen_recurrent_control_v1":
        return bool(
            row["status"] == "complete"
            and row["wake_updates"] == prefix_updates
            and row["source_unchanged"] is True
            and row["actor_unchanged"] is True
            and len(row["observations"]) == 5
            and row["live_sessions_started"] == 40
            and row["stream_result"]["score_forward_calls"] == 40
            and row["stream_result"]["training_examples"] == 8 * prefix_updates
            and row["stream_result"]["applied_updates"] == []
            and [p["physical_steps"] for p in row["stream_result"]["checkpoints"]]
            == [0, 24, 48, 72, 96]
            and all(p["updates"] == prefix_updates for p in row["stream_result"]["checkpoints"])
            and all(
                s["status"] == "complete"
                and s.get("collection_only") is True
                and s.get("collection_validated") is True
                and not s["update_attempted"]
                and not s["update_committed"]
                and not s["commit_validated"]
                and s["receipt"] is None
                and s["native_stages"] == []
                for s in row["training"][prefix_updates:]
            )
            and len(row["episodes"]) == 40
            and all(
                e["status"] == "complete" and e["state_unchanged"] is True for e in row["episodes"]
            )
        )
    return bool(
        row["status"] == "complete"
        and row["wake_updates"] == total
        and row["source_unchanged"] is True
        and row["actor_unchanged"] is True
        and len(row["observations"]) == checkpoints
        and row["live_sessions_started"] == checkpoints * 8
        and row["stream_result"]["score_forward_calls"] == checkpoints * 8
        and all(s["status"] == "complete" and s["commit_validated"] for s in row["training"])
        and all(e["status"] == "complete" and e["state_unchanged"] is True for e in row["episodes"])
    )


def run_pilot(
    *,
    condition_on_legal_actions: bool = False,
    action_remapping: bool = False,
    fill_interaction_quota: bool = False,
    environment_seed: int | None = None,
    heldout_environment_seed: int | None = None,
    frozen_recurrent_control: bool = False,
) -> dict[str, Any]:
    seed_options = _environment_seed_options(
        environment_seed, heldout_environment_seed, fill_interaction_quota
    )
    describe_protocol(
        condition_on_legal_actions=condition_on_legal_actions,
        **seed_options,
        **(
            {"frozen_recurrent_control": frozen_recurrent_control}
            if frozen_recurrent_control is not False
            else {}
        ),
        **({"action_remapping": action_remapping} if action_remapping is not False else {}),
        **(
            {"fill_interaction_quota": fill_interaction_quota}
            if fill_interaction_quota is not False
            else {}
        ),
    )
    return _run_pilot(
        warm_start=False,
        condition_on_legal_actions=condition_on_legal_actions,
        **seed_options,
        **(
            {"frozen_recurrent_control": frozen_recurrent_control}
            if frozen_recurrent_control is not False
            else {}
        ),
        **({"action_remapping": action_remapping} if action_remapping is not False else {}),
        **(
            {"fill_interaction_quota": fill_interaction_quota}
            if fill_interaction_quota is not False
            else {}
        ),
    )


def run_warm_start_pilot() -> dict[str, Any]:
    return _run_pilot(warm_start=True)


def describe_warm_start_comparison_protocol(
    *, condition_on_legal_actions: bool = False
) -> dict[str, Any]:
    if type(condition_on_legal_actions) is not bool:
        raise ValueError("condition_on_legal_actions must be an exact bool")
    p = describe_warm_start_protocol()
    p.update(
        protocol_id="actual_key_door_matched_warm_start_control_v1",
        classification="exploratory_matched_bc_only_vs_warm_start_reward",
        methods=["recurrent_bc", "recurrent_rl"],
        training="BC-only: eight train demonstrations then frozen weights; reward arm: identical eight demonstrations then eight sampled episodes;64teacher rows per arm",
        teacher="sixteen separate train-only collections across two arms,128teacher rows total; eight shared final_test packets64rows never train",
        phase_schedule=dict(
            bc_only=["demonstration"] * 8, warm_start_rl=["demonstration"] * 8 + ["reward"] * 8
        ),
        checkpoint_updates=dict(bc_only=[0, 8], warm_start_rl=[0, 8, 10, 12, 14, 16]),
        arms=[
            dict(arm="bc_only", native_updates=8, checkpoint_updates=[0, 8], reward_updates=0),
            dict(
                arm="warm_start_rl",
                native_updates=16,
                checkpoint_updates=[0, 8, 10, 12, 14, 16],
                reward_updates=8,
            ),
        ],
        matching="fresh identical seed17 architecture/eight train-only demonstrations; full parameters/gradients/optimizer/policy/clocks must match at8 before sampled reward work",
        evaluation="both arms finish all24training commits before any final scoring/live; control endpoint is actual native8, reward endpoint native16",
        endpoint_directions=dict(
            ce="mean BC8 CE minus RL16 CE",
            accuracy="mean RL16 accuracy minus BC8 accuracy",
            success="RL16 success rate minus BC8 success rate",
            returns="RL16 mean return minus BC8 mean return",
        ),
        comparison_gate="one fixed local exploratory case; unequal computation/on-policy exposure and carried momentum prevent reward-gradient attribution from movement alone",
        limitations="one seed/local game; frozen BC-only control has8SGD64rows vs RL16SGD variable rows. No equal compute, reward-gradient attribution, useful-skill or CPC-advantage guarantee",
    )
    p["planned_work"] = dict(
        wake_updates=24,
        native_train_calls=24,
        maximum_training_rows=224,
        demonstration_updates=16,
        demonstration_training_rows=128,
        reward_updates=8,
        maximum_reward_training_rows=96,
        bp_steps=24,
        pc_family_steps=0,
        replay_calls=0,
        probes=0,
        svd=0,
        resumes=0,
        collection_episodes=8,
        maximum_collection_forwards=96,
        score_forwards=64,
        score_rows=512,
        live_episodes=64,
        live_sessions=64,
        maximum_live_forwards=768,
        maximum_native_forwards=928,
        maximum_recurrent_unroll_steps=1600,
        demonstration_steps=192,
        environment_constructions=96,
        maximum_environment_steps=1056,
    )
    if condition_on_legal_actions:
        p.update(
            protocol_id="actual_key_door_legal_warm_start_comparison_v1",
            classification="exploratory_same_legal_support_bc_only_vs_warm_start_reward",
            action_conditioning="legal_actions_v1",
            sampling="RL arm: one private PCG64seed29 draw per actual legal-conditioned decision; no fallback/repair/retry; BC-only has no reward collection",
            objective="same eight native mean-CE demonstration updates per arm; RL then gamma1 legal-conditioned reward logprob SUM with continuous carried optimizer",
            diagnostic=TORCH_RECURRENT_LEGAL_REWARD_ID,
            evaluation="both arms finish all24validated commits before eight shared final_test packets are collected once; raw teacher CE/accuracy and greedy first-tie legal-conditioned live decisions on both arms",
            score_scope="teacher CE/accuracy use unchanged raw categorical probabilities, not conditional policy likelihood or live success",
            matching=p["matching"]
            + "; both live arms receive the same actual current legal-action support; guard preceding BC evidence before more RL work and both completed arms around final callbacks",
            comparisons="same architecture/demonstrations/legal-support evaluation; RL adds8updates and sampled exposure, carried BC momentum; raw unmasked historical outcomes are not matched controls",
            limitations=p["limitations"]
            + "; legal support is additional information; no reward-gradient attribution, isolated plasticity, generalization or CPC advantage follows from an endpoint improvement",
        )
    return p


def run_warm_start_comparison_pilot(*, condition_on_legal_actions: bool = False) -> dict[str, Any]:
    describe_warm_start_comparison_protocol(condition_on_legal_actions=condition_on_legal_actions)
    return _run_pilot(
        warm_start=True,
        paired=True,
        **({"condition_on_legal_actions": True} if condition_on_legal_actions else {}),
    )


def describe_demonstration_protocol(*, class_weighted: bool = False) -> dict[str, Any]:
    """Describe an explicit BC-only condition without constructing native state."""
    if type(class_weighted) is not bool:
        raise ValueError("class_weighted must be a bool")
    p = describe_warm_start_protocol()
    objective = TORCH_RECURRENT_WEIGHTED_CE_ID if class_weighted else TORCH_RECURRENT_CE_ID
    p.update(
        protocol_id="actual_key_door_weighted_bc_v1"
        if class_weighted
        else "actual_key_door_mean_bc_v1",
        classification="exploratory_recurrent_demonstration_only",
        architecture="Elman tanh/full ordered episode BPTT; zero hidden per packet",
        methods=["recurrent_bc"],
        phase_schedule=["demonstration"] * 8,
        checkpoint_updates=[0, 8],
        training="eight train-only A+/A-/B+/B-/C+/C-/A+/A- demonstrations,64rows; no reward phase",
        evaluation="all eight training commits before any final scoring/live; checkpoints0/8, all A/B/C/D both cues",
        objective=objective,
        class_weights="current train packet only: N/(P*n_c) present,1 absent; observed target-weight mass normalization"
        if class_weighted
        else "none; unchanged native mean CE",
        sampling=None,
        sampling_seed=None,
        native_defaults=dict(learning_rate=0.03, momentum=0.9, max_episode_steps=16),
        release="matching episode ID only; exact typed train demonstration packet",
        ce_directions=None,
        outcome_scope="endpoint CE/accuracy/live success/return; no stage-specific acquisition or recovery claim",
        baseline_reference=["artifacts/runs/r42-warm-start-control-20261016/paired.json"],
        comparison_gate="freeze and review a separate small comparison; reuse historical BC-only reference only when source/data/initialization/evaluation match",
        limitations="one seed/local game; historical reference is not an independent concurrent replication; no useful skill or circadian advantage established",
    )
    p["planned_work"] = dict(
        wake_updates=8,
        native_train_calls=8,
        maximum_training_rows=64,
        demonstration_updates=8,
        demonstration_training_rows=64,
        reward_updates=0,
        maximum_reward_training_rows=0,
        bp_steps=8,
        pc_family_steps=0,
        replay_calls=0,
        probes=0,
        svd=0,
        resumes=0,
        collection_episodes=0,
        maximum_collection_forwards=0,
        score_forwards=16,
        score_rows=128,
        live_episodes=16,
        live_sessions=16,
        maximum_live_forwards=192,
        maximum_native_forwards=208,
        maximum_recurrent_unroll_steps=384,
        demonstration_steps=128,
        environment_constructions=32,
        maximum_environment_steps=320,
    )
    return p


def run_demonstration_pilot(*, class_weighted: bool = False) -> dict[str, Any]:
    """Run BC-only through the same owned runtime and original eight-update budget."""
    describe_demonstration_protocol(class_weighted=class_weighted)
    return _run_pilot(warm_start=True, frozen_bc=True, class_weighted=class_weighted)


def _new_row(*, warm_start: bool, frozen_control: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = dict(
        method="recurrent_rl",
        replay_enabled=False,
        status="not_run",
        reason="not reached",
        observations=[],
        episodes=shared._slots(),
        training=_training_slots(True) if warm_start else _training_slots(),
        wake_updates=0,
        stream_result=None,
        outcomes=None,
        source_unchanged=None,
        actor_unchanged=None,
        readback_error=None,
        live_sessions_started=0,
        collection_attempts=0,
        sampling_state=None,
        work=None,
    )
    if warm_start:
        row.update(
            phase_schedule=[s["phase"] for s in row["training"]],
            demonstration_collection_attempts=0,
            phase_work=None,
            phase_work_error=None,
            phase_work_scope=None,
        )
        points = (0, 8) if frozen_control else (0, 8, 10, 12, 14, 16)
        row["episodes"] = [
            dict(slot, checkpoint_updates=point)
            for point in points
            for slot in shared._slots()
            if slot["checkpoint_updates"] == 0
        ]
        if frozen_control:
            row.update(
                method="recurrent_bc",
                training=row["training"][:8],
                phase_schedule=["demonstration"] * 8,
            )
    return row


def _compare_endpoints(rows: list[dict[str, Any]]) -> dict[str, Any]:
    _, rl = rows
    endpoints = [
        next(p for p in row["stream_result"]["checkpoints"] if p["updates"] == n)
        for row, n in zip(rows, (8, 16), strict=True)
    ]
    episodes = [
        [e for e in row["episodes"] if e["checkpoint_updates"] == n]
        for row, n in zip(rows, (8, 16), strict=True)
    ]
    required = {(layout.layout_id, cue) for layout in shared.layouts() for cue in (1, -1)}
    if any(
        len(values) != 8 or {(e["context"], e["cue"]) for e in values} != required
        for values in episodes
    ):
        raise ValueError("paired comparison requires all eight unique final context/cue pairs")
    output = []
    for left, right in zip(*episodes, strict=True):
        if (left["context"], left["cue"]) != (right["context"], right["cue"]):
            raise ValueError("paired endpoints differ in final context/cue")
        context = f"{left['context']}-{'red' if left['cue'] == 1 else 'blue'}"
        a, b = [p["scores"][context] for p in endpoints]
        output.append(
            dict(
                context=context,
                ce_bc=a["ce"],
                ce_rl=b["ce"],
                ce_reduction_bc_minus_rl=a["ce"] - b["ce"],
                accuracy_bc=a["accuracy"],
                accuracy_rl=b["accuracy"],
                success_bc=left["success"],
                success_rl=right["success"],
                return_bc=left["return"],
                return_rl=right["return"],
                accuracy_gain_rl_minus_bc=b["accuracy"] - a["accuracy"],
                success_gain_rl_minus_bc=int(right["success"]) - int(left["success"]),
                return_gain_rl_minus_bc=right["return"] - left["return"],
            )
        )
    if len(output) != 8:
        raise ValueError("paired comparison requires all eight endpoint contexts")
    result: dict[str, Any] = dict(
        endpoint_native_updates=dict(bc_only=8, warm_start_rl=16), endpoint_rows=output
    )
    for metric, field in (
        ("mean_ce", "ce"),
        ("mean_accuracy", "accuracy"),
        ("success_rate", "success"),
        ("mean_return", "return"),
    ):
        result[metric] = {
            arm: sum(float(p[f"{field}_{suffix}"]) for p in output) / 8
            for arm, suffix in (("bc_only", "bc"), ("warm_start_rl", "rl"))
        }
    result.update(
        ce_reduction_bc_minus_rl=result["mean_ce"]["bc_only"] - result["mean_ce"]["warm_start_rl"],
        accuracy_gain_rl_minus_bc=result["mean_accuracy"]["warm_start_rl"]
        - result["mean_accuracy"]["bc_only"],
        success_gain_rl_minus_bc=result["success_rate"]["warm_start_rl"]
        - result["success_rate"]["bc_only"],
        return_gain_rl_minus_bc=result["mean_return"]["warm_start_rl"]
        - result["mean_return"]["bc_only"],
        nonzero_reward_episodes=sum(
            s["collection"]["total_return"] != 0 for s in rl["training"][8:]
        ),
        reward_diagnostic_values=[s["receipt"]["diagnostic"]["value"] for s in rl["training"][8:]],
        attribution="unequal compute/on-policy exposure and carried BC momentum; movement alone is not reward-gradient benefit",
        learning_benefit_established=False,
    )
    return result


def _run_pilot(
    *,
    warm_start: bool,
    paired: bool = False,
    frozen_bc: bool = False,
    class_weighted: bool = False,
    condition_on_legal_actions: bool = False,
    action_remapping: bool = False,
    fill_interaction_quota: bool = False,
    environment_seed: int | None = None,
    heldout_environment_seed: int | None = None,
    frozen_recurrent_control: bool = False,
) -> dict[str, Any]:
    if type(condition_on_legal_actions) is not bool:
        raise ValueError("condition_on_legal_actions must be an exact bool")
    if (
        type(action_remapping) is not bool
        or action_remapping
        and (not condition_on_legal_actions or warm_start or paired or frozen_bc or class_weighted)
    ):
        raise ValueError("action remapping requires standalone legal reward collection")
    if type(fill_interaction_quota) is not bool or fill_interaction_quota and not action_remapping:
        raise ValueError("interaction quota requires the explicit legal mapped reward route")
    if type(frozen_recurrent_control) is not bool or (
        frozen_recurrent_control and not fill_interaction_quota
    ):
        raise ValueError("frozen recurrent control requires the explicit mapped quota route")
    seed_options = _environment_seed_options(
        environment_seed, heldout_environment_seed, fill_interaction_quota
    )
    if condition_on_legal_actions and (
        frozen_bc or class_weighted or ((warm_start or paired) and not (warm_start and paired))
    ):
        raise ValueError("legal conditioning requires the standalone reward route")
    if frozen_bc and (not warm_start or paired) or class_weighted and not frozen_bc:
        raise ValueError("weighted demonstrations require a standalone BC-only condition")
    row = _new_row(warm_start=warm_start, frozen_control=frozen_bc)
    if condition_on_legal_actions:
        row["action_conditioning"] = "legal_actions_v1"
    if action_remapping:
        row["action_mapping"] = "movement_permutation_v1"
        declared = {layout.layout_id: layout for layout in action_remapping_layouts(**seed_options)}
        for slot in row["training"]:
            slot["layout"] = key_door_layout_record(declared[slot["context"]])
    if fill_interaction_quota:
        row["interaction_quota_mode"] = "twelve_steps_per_phase_v1"
        row["quota_checkpoints"] = [dict(physical_steps=0, native_updates=0)]
        for index, slot in enumerate(row["training"]):
            slot.update(quota_phase=index, episode_in_quota=0)
    if frozen_recurrent_control:
        row["learning_mode"] = "frozen_recurrent_control_v1"
        for slot in row["training"]:
            slot.update(collection_only=True, collection_validated=False)
    if seed_options:
        row["environment_seeds"] = dict(
            training=environment_seed, final_only=heldout_environment_seed
        )
    if frozen_bc:
        row["demonstration_objective"] = (
            TORCH_RECURRENT_WEIGHTED_CE_ID if class_weighted else TORCH_RECURRENT_CE_ID
        )
    rows = [_new_row(warm_start=True, frozen_control=True), row] if paired else [row]
    if paired:
        for arm, item in zip(("bc_only", "warm_start_rl"), rows, strict=True):
            item.update(arm=arm, warm_start_matches_control=None)
            if condition_on_legal_actions:
                item["action_conditioning"] = "legal_actions_v1"
    comparison = None
    errors, memory, sampler = [], None, None
    demonstrations: list[Any] = []
    started = monotonic()
    try:
        sampler = ProcessRssSampler()
        with sampler:
            torch = require_torch()
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)

            completed_checks: list[Any] = []
            final_cache: list[Any] = []
            final_seal: list[Any] = []
            final_seen: set[str] = set()
            completed_finals: list[tuple[Any, Any]] = []

            def final_signature(batch: Any) -> Any:
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

            def final_result(item: Any) -> Any:
                return (item["status"], item["stream_result"], item["episodes"], item["outcomes"])

            def check_pair() -> None:
                if len(completed_checks) != 2 or any(
                    r["status"] not in ("trained", "complete") for r in rows
                ):
                    raise ValueError("legal paired finals require both original completed arms")
                # Each retained guard checks only its own arm, avoiding recursive peer callbacks.
                for check in completed_checks:
                    check()
                for owner, expected in completed_finals:
                    if not _same_state(expected, final_result(owner)):
                        raise ValueError("completed paired final evidence changed")
                if not _same_state(
                    final_seal, [final_signature(batch) for batch in final_cache]
                ) or not _same_state(demonstrations, [signature[-1] for signature in final_seal]):
                    raise ValueError(
                        "shared final packets or recorded demonstration evidence changed"
                    )

            def collect_final(check_boundary: Any = None) -> Any:
                if check_boundary is None:
                    # Preserve legacy collection/failure ordering on every old route.
                    original = tuple(
                        shared._collect(
                            f"key-door-final-{layout.layout_id}-{cue}", layout, cue, "final_test"
                        )
                        for layout in (
                            action_remapping_layouts(**seed_options)
                            if action_remapping
                            else shared.layouts()
                        )
                        for cue in (1, -1)
                    )
                    for batch in original:
                        if batch.episode is None:
                            raise ValueError(
                                "final key/door packets require complete episode metadata"
                            )
                        demonstrations.append(asdict(batch.episode))
                    return original
                packets = []
                for layout in (
                    action_remapping_layouts(**seed_options)
                    if action_remapping
                    else shared.layouts()
                ):
                    for cue in (1, -1):
                        if check_boundary is not None:
                            check_boundary()
                        batch = shared._collect(
                            f"key-door-final-{layout.layout_id}-{cue}", layout, cue, "final_test"
                        )
                        if batch.episode is None:
                            raise ValueError(
                                "final key/door packets require complete episode metadata"
                            )
                        if paired and condition_on_legal_actions:
                            _validate_batch(batch, "final_test", final_seen)
                        demonstrations.append(asdict(batch.episode))
                        packets.append(batch)
                        if paired and condition_on_legal_actions:
                            final_cache.append(batch)
                            final_seal.append(final_signature(batch))
                        if check_boundary is not None:
                            check_boundary()
                return tuple(packets)

            final = collect_final if condition_on_legal_actions else collect_final()
            if paired:
                evaluation_queue: list[Any] = []
                warm_state: list[Any] = []

                def paired_final(check_boundary: Any) -> Any:
                    check_boundary()
                    if not final_cache:
                        collect_final(check_boundary)
                    if len(final_cache) != 8:
                        raise ValueError(
                            "paired final collection is incomplete; no retry is permitted"
                        )
                    check_boundary()
                    return tuple(final_cache)

                if condition_on_legal_actions:
                    final = paired_final
                first_options: dict[str, Any] = (
                    {"completed_checks": completed_checks, "check_final": check_pair}
                    if condition_on_legal_actions
                    else {}
                )
                _run_arm(
                    rows[0],
                    final,
                    started + 90,
                    sampler,
                    warm_start=True,
                    frozen_control=True,
                    warm_state=warm_state,
                    evaluation_queue=evaluation_queue,
                    **first_options,
                )
                if rows[0]["status"] == "trained":
                    next_options: dict[str, Any] = (
                        {
                            "completed_checks": completed_checks,
                            "check_final": check_pair,
                            "check_preceding": completed_checks[0],
                        }
                        if condition_on_legal_actions
                        else {}
                    )
                    _run_arm(
                        row,
                        final,
                        started + 90,
                        sampler,
                        warm_start=True,
                        warm_state=warm_state,
                        evaluation_queue=evaluation_queue,
                        **next_options,
                    )
                if row["warm_start_matches_control"] is True:
                    rows[0]["warm_start_matches_control"] = True
                if all(r["status"] == "trained" for r in rows) and len(evaluation_queue) == 2:
                    for index, finish in enumerate(evaluation_queue):
                        finish()
                        if condition_on_legal_actions and rows[index]["status"] == "complete":
                            # Keep at most two owned final records; later callbacks cannot rewrite a prior result.
                            completed_finals.append(
                                (rows[index], deepcopy(final_result(rows[index])))
                            )
                        if not all(r["status"] in ("trained", "complete") for r in rows):
                            for r in rows:
                                if r["status"] == "trained":
                                    r.update(
                                        status="not_evaluated",
                                        reason="preceding paired final evaluation failed",
                                    )
                                    r["work"] = shared._recurrent_work(r)
                            break
                    if all(r["status"] == "complete" for r in rows):
                        if condition_on_legal_actions:
                            check_pair()
                        comparison = _compare_endpoints(rows)
                else:
                    for r in rows:
                        if r["status"] == "trained":
                            r.update(status="not_evaluated", reason="paired training incomplete")
                            r["work"] = shared._recurrent_work(r)
                evaluation_queue.clear()
                warm_state.clear()
                completed_checks.clear()
            elif frozen_bc:
                _run_arm(row, final, started + 90, sampler, warm_start=True, frozen_control=True)
            elif warm_start:
                _run_arm(row, final, started + 90, sampler, warm_start=True)
            else:
                _run_arm(row, final, started + 90, sampler)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error!r}")
    try:
        if sampler is not None:
            memory = asdict(sampler.snapshot())
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error!r}")
    elapsed = monotonic() - started
    complete = (
        all(
            _valid(r, warm_start=warm_start, frozen_control=frozen_bc or paired and i == 0)
            for i, r in enumerate(rows)
        )
        and (not paired or comparison is not None)
        and not errors
        and elapsed <= 90
        and memory is not None
        and memory["peak_bytes"] <= shared.MAX_RSS
    )
    report = dict(
        protocol=describe_demonstration_protocol(class_weighted=class_weighted)
        if frozen_bc
        else describe_warm_start_comparison_protocol(
            **({"condition_on_legal_actions": True} if condition_on_legal_actions else {})
        )
        if paired
        else describe_warm_start_protocol()
        if warm_start
        else describe_protocol(
            condition_on_legal_actions=condition_on_legal_actions,
            **seed_options,
            **({"frozen_recurrent_control": True} if frozen_recurrent_control else {}),
            **({"action_remapping": True} if action_remapping else {}),
            **({"fill_interaction_quota": True} if fill_interaction_quota else {}),
        ),
        status="complete" if complete else "incomplete",
        runs=rows,
        demonstrations=demonstrations,
        global_errors=errors,
        total_seconds=elapsed,
        observed_process_rss=memory,
        scientific_advantage_verified=False,
        full_R42_complete=False,
    )
    if paired:
        report.update(
            warm_start_matched=row["warm_start_matches_control"] is True, comparison=comparison
        )
    if action_remapping:
        report["full_R43_complete"] = False
    return report
