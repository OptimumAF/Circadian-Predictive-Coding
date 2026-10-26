"""Fixed TRAIN-only MiniGrid recurrent reward smoke through the existing runtime.

The descriptor is resource free. Execution collects eight on-policy episodes,
releases matching identifiers and records native receipts/charges. No teacher,
evaluation, replay, promotion, selection or new training framework lives here.
This demonstrates a port, not useful learning or a comparison with BC/CPC.
"""

from copy import deepcopy
from dataclasses import asdict
import pickle
from time import monotonic
from typing import Any

import numpy as np

from src.adapters.key_door_reward_pilot import _binding, _check_budget, _serialize_collection
from src.adapters.minigrid_reward_collection import (
    MiniGridRewardCollectionFailed,
    collect_minigrid_reward_episode,
)
from src.adapters.minigrid_momentum_control import new_momentum_control, run_momentum_control
from src.adapters.minigrid_warm_start import prepare_warm_start
from src.adapters.minigrid_warm_readouts import new_train_diagnostics, run_train_diagnostics
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.adapters.torch_reward_update_learner import RewardUpdate, TorchRewardUpdateLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.minigrid_key_door import predefined_layouts
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch

BODY_SECONDS = 45.0
RSS_LIMIT = 1024**3
OBJECTIVE = "torch_recurrent_reward_logprob_sum_v1"
SCHEDULE = tuple(
    (context, cue) for context in ("A", "B", "C_train", "A") for cue in ("blue", "red")
)


def describe_protocol(
    *,
    warm_start: bool = False,
    train_readouts: bool = False,
    momentum_control: bool = False,
    extended_control_readouts: bool = False,
) -> dict[str, Any]:
    """Freeze dose, source roles, categorical law and resource limits before work."""
    if type(warm_start) is not bool:
        raise ValueError("warm_start must be an exact bool")
    if type(train_readouts) is not bool or (train_readouts and not warm_start):
        raise ValueError("train_readouts must be an exact bool and requires warm_start=True")
    if type(momentum_control) is not bool or (
        momentum_control and not (warm_start and train_readouts)
    ):
        raise ValueError(
            "momentum_control must be an exact bool and requires warm_start/train_readouts"
        )
    if type(extended_control_readouts) is not bool or (
        extended_control_readouts and not momentum_control
    ):
        raise ValueError(
            "extended_control_readouts must be an exact bool and requires momentum_control"
        )
    protocol: dict[str, Any] = dict(
        protocol_id="minigrid_cue_key_door_recurrent_reward_smoke_v1",
        classification="exploratory_train_only_recurrent_reward_port",
        methods=["recurrent_rl"],
        source_role="train",
        feature_dim=148,
        hidden_dim=8,
        num_classes=7,
        seed=17,
        learning_rate=0.03,
        momentum=0.9,
        discount=1.0,
        max_episode_steps=16,
        sampling_seed=29,
        layouts=[asdict(layout) for layout in predefined_layouts()[:3]],
        schedule=[dict(context=context, cue=cue) for context, cue in SCHEDULE],
        sampling="one private PCG64 uniform per decision; raw seven-action CDF scaled by actual total; no mask/repair/retry",
        objective=OBJECTIVE,
        release="matching episode identifier one logical tick later; no class targets",
        evaluation="none; TRAIN rollout outcomes are collection evidence, not policy readouts",
        planned_work=dict(
            native_train_calls=8,
            maximum_training_rows=128,
            collection_episodes=8,
            maximum_environment_resets=8,
            maximum_environment_closes=8,
            maximum_collection_sessions=8,
            maximum_collection_predictions=128,
            maximum_environment_steps=128,
            maximum_recurrent_unroll_steps=256,
            score_forwards=0,
            teacher_episodes=0,
            final_episodes=0,
            probe_updates=0,
            replay_updates=0,
            promotions=0,
        ),
        limits=dict(
            body_seconds=BODY_SECONDS,
            parent_seconds=60,
            process_rss_bytes=RSS_LIMIT,
            output_bytes=16 * 1024**2,
            cpu_threads=1,
            interop_threads=1,
            gpu=0,
        ),
        storage="initial/current adapter checkpoints only; existing inbox retains at most8 owned RewardUpdate payloads with behavior checkpoints until invocation ends; report omits opaque checkpoints; RSS includes retained payloads/copies",
        limitations="eight variable-length TRAIN rollouts/one seed; no acquisition, retention, generalization, comparative benefit or CPC advantage claim; momentum can move parameters even with zero episode reward",
    )
    if warm_start:
        protocol.update(
            protocol_id="minigrid_unweighted_a128_warm_reward_smoke_v1",
            classification="engineering_exact_bc_prefix_then_sampled_reward_continuation",
            warm_start="newly charged original unweighted A128/1152 reconstruction; all5raw and0/128native cue-pair readouts EXACT accepted-reference parity; full optimizer/gradients/momentum restored without reset",
            reference_path="artifacts/runs/r21-fixed-a-acquisition-20261016/native-result.json",
            training="128 original A mean-CE updates then fixed8 delayed sampled reward episodes; reward receipt/budget clocks local1..8, native counters128..136",
            evaluation="warm-prefix parity only before reward collection; no postreward policy readout or comparison claim",
            planned_work=dict(
                native_train_calls=136,
                maximum_training_rows=1280,
                warm_updates=128,
                warm_training_rows=1152,
                reward_updates=8,
                maximum_reward_training_rows=128,
                score_forwards=10,
                score_rows=90,
                warm_live_episodes=4,
                maximum_warm_live_predictions=64,
                collection_episodes=8,
                maximum_collection_predictions=128,
                maximum_environment_steps=192,
                maximum_environment_resets=12,
                maximum_environment_closes=12,
                maximum_collection_sessions=8,
                maximum_recurrent_unroll_steps=1562,
                teacher_episodes=0,
                final_episodes=0,
                probe_updates=0,
                replay_updates=0,
                promotions=0,
            ),
            limits=dict(
                protocol["limits"],
                body_seconds=180.0,
                parent_seconds=210,
                reward_phase_seconds=45.0,
                inherited_warm_arm_rss_bytes=768 * 1024**2,
            ),
            storage=protocol["storage"]
            + "; five bounded original warm checkpoints and parity reports retained; old historical source/control costs labelled separately",
            limitations=protocol["limitations"]
            + "; necessary native warm reconstruction is not a repeated baseline experiment; no causal reward-effect claim without shared warm/additional-update momentum controls",
        )
    if train_readouts:
        protocol.update(
            protocol_id="minigrid_warm_reward_train_readouts_v1",
            classification="exploratory_train_only_before_after_reward_endpoints",
            evaluation="after ALL8 reward commits/complete_memory/fullguard, freeze original45s reward budget/clock; independentzero-update min60seconds/whole-remainder readout budget; owned128/136knownTRAINA/B/C only, no feedback totraining",
            readout_method="recurrent_warm_reward",
            readout_points=[128, 136],
            readout_role="train",
            endpoint_joint_acquisition="each endpoint: both cue exact_sequence_accuracy==1 AND both native successes",
            endpoint_directions="mean2cue CE before-minus-after; accuracy/exact_sequence_accuracy/live success/return after-minus-before; positive gain desirable; initialjointbothcue fit+nativegoals is required retention premise; endpoint-only, no recovery or causal RL benefit claim",
            planned_work=dict(
                protocol["planned_work"],
                score_forwards=22,
                score_rows=198,
                train_diagnostic_raw_forwards=12,
                train_diagnostic_raw_rows=108,
                train_diagnostic_live_episodes=12,
                maximum_train_diagnostic_predictions=192,
                maximum_environment_steps=384,
                maximum_environment_resets=24,
                maximum_environment_closes=24,
                maximum_recurrent_unroll_steps=1862,
            ),
            limits=dict(protocol["limits"], train_readout_phase_seconds=60.0),
            storage=protocol["storage"]
            + "; two bounded added diagnostic snapshots/reports; original training resource observation frozen separately from later shared-sampler growth",
        )
    if momentum_control:
        protocol.update(
            protocol_id="minigrid_warm_reward_momentum_control_v1",
            classification="exploratory_shared_warm_observed_episode_additional_update_control",
            methods=["recurrent_warm_reward", "recurrent_warm_momentum_control"],
            control="AFTER all8 validated reward commits and frozen original45s clock: independent warm128 source fork, same8 original observed TRAIN episodes unchanged, densezero nativeSGD/momentum; delayedIDrelease via existingruntime",
            readout_method="distinct reward/control identities; shared warm128 then reward136/control136 after ALLtraining",
            comparison_directions="control-minus-reward gains: rewardCE-minus-controlCE; controlaccuracy/exact/success/return-minus-reward; allknownTRAIN, same observed rows conditioned on rewardarm collection, not independentonpolicycontrol",
            planned_work=dict(
                protocol["planned_work"],
                native_train_calls=144,
                maximum_training_rows=1408,
                control_updates=8,
                maximum_control_repeated_observation_rows=128,
                score_forwards=28,
                score_rows=252,
                train_diagnostic_raw_forwards=18,
                train_diagnostic_raw_rows=162,
                train_diagnostic_live_episodes=18,
                maximum_train_diagnostic_predictions=288,
                maximum_environment_steps=480,
                maximum_environment_resets=30,
                maximum_environment_closes=30,
                maximum_recurrent_unroll_steps=2140,
            ),
            limits=dict(
                protocol["limits"], control_phase_seconds=45.0, train_readout_phase_seconds=90.0
            ),
            storage=protocol["storage"]
            + "; one control actor/candidate runtime, bounded8 unchanged episode DTOs/receipts and one endpoint, shared RSS covers all copies",
            limitations=protocol["limitations"]
            + "; matchedadditionalupdates/originalobservations isolates zero-currentgradient momentum control conditional on rewardarm observations, not equaltrainingFLOPs or independent exploration; compare actual costs explicitly",
        )
    if extended_control_readouts:
        # Why: the spent v1 case paid90s for95 diagnostic predictions. This is
        # a separately selected engineering allowance, never a renewed budget.
        protocol.update(
            protocol_id="minigrid_warm_reward_momentum_control_v2",
            engineering_allowance="explicitv2: measuredv1readout90s/95pred; qualified3to2parentproof reduction projects~182s for288maxpred;210readoutAND300whole is boundedplanning, not fitguarantee; preserve spentv1case",
            limits=dict(
                protocol["limits"],
                body_seconds=300.0,
                parent_seconds=330,
                train_readout_phase_seconds=210.0,
            ),
        )
    return protocol


def _new_report() -> dict[str, Any]:
    return dict(
        protocol=describe_protocol(),
        status="running",
        reason=None,
        training=[
            dict(
                index=i,
                context=context,
                cue=cue,
                episode_id=f"minigrid-reward-train-{i}",
                status="not_run",
                reason="not reached",
                collection=None,
                early_release_receipts=None,
                native_stages=[],
                update_attempted=False,
                update_committed=False,
                commit_validated=False,
                receipt=None,
                charged_updates_before=None,
                charged_updates_after=None,
                native_updates_before=None,
                native_updates_after=None,
                native_training_examples_before=None,
                native_training_examples_after=None,
                sampler_before=None,
                sampler_after=None,
                readback_error=None,
            )
            for i, (context, cue) in enumerate(SCHEDULE)
        ],
        source_unchanged=None,
        actor_unchanged=None,
        wake_updates=0,
        training_examples=0,
        work=None,
        observed_process_rss=None,
        total_seconds=None,
        readback_error=None,
        scientific_metrics=None,
        scientific_advantage_verified=False,
    )


def _counts(state: Any, updates: int, rows: int) -> None:
    if (
        type(state.completed_updates) is not int
        or state.completed_updates != updates
        or type(state.training_examples) is not int
        or state.training_examples != rows
    ):
        raise ValueError("native update/row counters differ from original committed work")


def _work(training: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "snapshots_attempted",
        "snapshots_returned",
        "sessions_attempted",
        "sessions_started",
        "environment_constructions_attempted",
        "environment_constructions_returned",
        "prediction_calls_attempted",
        "prediction_calls_returned",
        "sampling_draws",
        "environment_steps_attempted",
        "environment_steps_returned",
        "close_calls_attempted",
        "close_calls_returned",
    )
    collections = [slot["collection"] for slot in training if slot["collection"] is not None]
    result: dict[str, Any] = {key: sum(c["work"][key] for c in collections) for key in keys}
    result.update(
        native_updates_attempted=sum(s["update_attempted"] for s in training),
        native_updates_committed=sum(s["update_committed"] for s in training),
        validated_commits=sum(s["commit_validated"] for s in training),
        committed_training_rows=sum(
            len(s["collection"]["episode"]["transitions"])
            for s in training
            if s["update_committed"] and s["collection"]["episode"] is not None
        ),
        failure_work_may_be_unobserved=any(
            c["work"]["failure_work_may_be_unobserved"] for c in collections
        )
        or any(s["update_attempted"] and not s["commit_validated"] for s in training),
        score_forwards=0,
        teacher_episodes=0,
        final_episodes=0,
        probe_updates=0,
        replay_updates=0,
        promotions=0,
    )
    result["observed_recurrent_unroll_steps"] = (
        result["prediction_calls_returned"] + result["committed_training_rows"]
    )
    result["unroll_scope"] = (
        "returned collection predictions plus committed native training rows; failed native work may be unobserved"
    )
    return result


def run_pilot(
    *,
    warm_start: bool = False,
    train_readouts: bool = False,
    momentum_control: bool = False,
    extended_control_readouts: bool = False,
) -> dict[str, Any]:
    """Execute one fixed bounded case; never retry a failed or uncertain update."""
    if type(warm_start) is not bool:
        raise ValueError("warm_start must be an exact bool")
    if type(train_readouts) is not bool or (train_readouts and not warm_start):
        raise ValueError("train_readouts must be an exact bool and requires warm_start=True")
    if type(momentum_control) is not bool or (
        momentum_control and not (warm_start and train_readouts)
    ):
        raise ValueError(
            "momentum_control must be an exact bool and requires warm_start/train_readouts"
        )
    if type(extended_control_readouts) is not bool or (
        extended_control_readouts and not momentum_control
    ):
        raise ValueError(
            "extended_control_readouts must be an exact bool and requires momentum_control"
        )
    report = _new_report()
    warm_report: dict[str, Any] = {}
    if warm_start:
        report["protocol"] = (
            describe_protocol(
                warm_start=True,
                train_readouts=True,
                momentum_control=True,
                extended_control_readouts=True,
            )
            if extended_control_readouts
            else describe_protocol(warm_start=True, train_readouts=True, momentum_control=True)
            if momentum_control
            else describe_protocol(warm_start=True, train_readouts=True)
            if train_readouts
            else describe_protocol(warm_start=True)
        )
        report["warm_start"] = warm_report
    diagnostics: dict[str, Any] = (
        (
            new_train_diagnostics(momentum_control=True)
            if momentum_control
            else new_train_diagnostics()
        )
        if train_readouts
        else {}
    )
    if train_readouts:
        report["train_diagnostics"] = diagnostics
    control_report = new_momentum_control() if momentum_control else {}
    if momentum_control:
        report["momentum_control"] = control_report
    control_payload: bytes | None = None
    control_guard: Any = None
    control_state: Any = None
    control_episodes: list[Any] = []
    control_episode_payloads: list[bytes] = []
    protocol = report["protocol"]
    protocol_bytes = pickle.dumps(protocol, protocol=5)
    training = report["training"]
    identities = tuple(id(slot) for slot in training)
    metadata = tuple((s["index"], s["context"], s["cue"], s["episode_id"]) for s in training)
    expected_slots = [pickle.dumps(slot, protocol=5) for slot in training]
    completed_records: list[dict[str, Any]] = []
    source: Any = None
    runtime: Any = None
    budget: Any = None
    initial: Any = None
    previous: Any = None
    rng: Any = None
    memory: Any = None
    birth = None
    receipts: list[Any] = []
    expected_rng = None
    updates = rows = 0
    active = -1
    expected_tick = 0
    last_clock: float | None = None
    actor_owner: Any = None
    candidate_owner: Any = None
    inbox_owner: Any = None
    collection_bytes = None
    captured_receipt = None
    primary_error = None
    started = monotonic()
    whole_seconds = protocol["limits"]["body_seconds"]
    deadline = started + whole_seconds
    base_updates = base_rows = 0
    warm_guard: Any = None
    warm_payload: bytes | None = None
    diagnostic_payload: bytes | None = None
    diagnostic_guard: Any = None
    diagnostic_sources: Any = None
    frozen_training_clock: float | None = None
    frozen_training_rss: Any = None
    frozen_training_rss_owner: Any = None
    in_completed_warm_guard = False
    clock = LogicalClock()

    def structural_evidence() -> None:
        """Check identities, fixed metadata and the small frozen protocol."""
        if (
            report["protocol"] is not protocol
            or pickle.dumps(protocol, protocol=5) != protocol_bytes
            or report["training"] is not training
            or tuple(id(s) for s in training) != identities
            or tuple((s["index"], s["context"], s["cue"], s["episode_id"]) for s in training)
            != metadata
        ):
            raise ValueError("protocol, fixed slots or preceding reward evidence changed")
        if warm_start and report.get("warm_start") is not warm_report:
            raise ValueError("preceding warm-prefix evidence changed")
        if train_readouts and report.get("train_diagnostics") is not diagnostics:
            raise ValueError("preceding TRAIN diagnostic evidence changed")
        if momentum_control and report.get("momentum_control") is not control_report:
            raise ValueError("original momentum control evidence owner changed")
        if in_completed_warm_guard:
            if (
                runtime._budget is not budget
                or runtime._inbox._budget is not budget
                or runtime._inbox._clock is not clock
                or clock.now() != expected_tick
                or type(clock.now()) is not int
                or budget.process_rss_sampler is not memory
            ):
                raise ValueError("original read-only warm callback bindings changed")
            _check_budget(budget, birth, updates, last_clock)
            if frozen_training_clock is not None and budget.last_clock != frozen_training_clock:
                raise ValueError("completed original reward phase clock changed")

    def full_evidence(*, check_warm_payload: bool = True) -> None:
        """Compare every original paid payload before work and after callbacks."""
        structural_evidence()
        if any(
            pickle.dumps(slot, protocol=5) != expected_slots[i]
            for i, slot in enumerate(training)
            if i != active
        ):
            raise ValueError("protocol, fixed slots or preceding reward evidence changed")
        if (
            check_warm_payload
            and warm_payload is not None
            and pickle.dumps(warm_report, protocol=5) != warm_payload
        ):
            raise ValueError("preceding warm-prefix evidence changed")
        if (
            diagnostic_payload is not None
            and pickle.dumps(diagnostics, protocol=5) != diagnostic_payload
        ):
            raise ValueError("preceding TRAIN diagnostic evidence changed")

        if momentum_control and (
            len(control_episodes) != len(control_episode_payloads)
            or any(
                pickle.dumps(episode, protocol=5) != payload
                for episode, payload in zip(control_episodes, control_episode_payloads, strict=True)
            )
            or (
                control_payload is not None
                and pickle.dumps(control_report, protocol=5) != control_payload
            )
        ):
            raise ValueError("original observed control inputs or completed evidence changed")

    def warm_boundary() -> None:
        # Why: the measured duplicate checks occur inside a completed read-only
        # warm guard. Its caller proves full reward/diagnostic evidence on entry
        # and exit before admitting work; helper.common still proves warm data.
        # Preparation and direct warm-guard calls retain the original full check.
        if in_completed_warm_guard:
            structural_evidence()
        else:
            full_evidence(check_warm_payload=False)
        if monotonic() >= deadline:
            raise ValueError("original whole-case warm/reward deadline exhausted")
        if memory is not None:
            memory.sample()
            if memory.snapshot().peak_bytes > RSS_LIMIT:
                raise ValueError("whole-case warm/reward process RSS limit exhausted")
        if in_completed_warm_guard:
            structural_evidence()
        else:
            full_evidence(check_warm_payload=False)

    def check_owned(*, candidate: bool, sampling: bool) -> None:
        nonlocal in_completed_warm_guard
        full_evidence()
        # Warm guards read their original owners/budgets. Their callbacks must
        # finish before proving the current reward candidate/RNG/phase budget.
        if warm_guard is not None:
            previous_scope = in_completed_warm_guard
            in_completed_warm_guard = True
            try:
                warm_guard()
            finally:
                in_completed_warm_guard = previous_scope
        if monotonic() - started >= whole_seconds:
            raise ValueError("fixed whole-case wall-time limit exhausted")
        _check_budget(budget, birth, updates, last_clock)
        if frozen_training_clock is not None and budget.last_clock != frozen_training_clock:
            raise ValueError("completed original reward phase clock changed")
        if frozen_training_clock is not None and (
            budget.process_rss_segment is not frozen_training_rss_owner
            or not _same_state(frozen_training_rss, budget.process_rss_segment)
        ):
            raise ValueError("completed original reward resource observation changed")
        if (
            runtime._budget is not budget
            or runtime._inbox._budget is not budget
            or runtime._inbox._clock is not clock
            or clock.now() != expected_tick
            or type(clock.now()) is not int
            or runtime.actor is not actor_owner
            or runtime._candidate is not candidate_owner
            or runtime._inbox is not inbox_owner
            or runtime.actor.version != "a0"
            or runtime._candidate_version != "c0"
            or budget.process_rss_sampler is not memory
            or not _same_state(initial, source.snapshot_state())
            or not _same_state(initial, runtime.actor.snapshot_state().state)
            or tuple(runtime._inbox.applied_updates) != tuple(receipts)
        ):
            raise ValueError("original source/actor/receipts/budget/clock binding changed")
        if (
            collection_bytes is not None
            and active >= 0
            and pickle.dumps(training[active]["collection"], protocol=5) != collection_bytes
        ):
            raise ValueError("completed current collection evidence changed before update")
        if candidate and not _same_state(previous, runtime.candidate_snapshot().state):
            raise ValueError("preceding committed candidate changed before more work")
        if sampling and rng.bit_generator.state != expected_rng:
            raise ValueError("sampling state changed outside recorded collection")
        # Snapshot ports above can invoke external work. Bind pure authority and
        # evidence again after their last callback, before admitting anything.
        _check_budget(budget, birth, updates, last_clock)
        if frozen_training_clock is not None and budget.last_clock != frozen_training_clock:
            raise ValueError("completed original reward phase clock changed")
        if frozen_training_clock is not None and (
            budget.process_rss_segment is not frozen_training_rss_owner
            or not _same_state(frozen_training_rss, budget.process_rss_segment)
        ):
            raise ValueError("completed original reward resource observation changed")
        full_evidence()

    def guard(*, candidate: bool = True, sampling: bool = True) -> None:
        nonlocal last_clock
        check_owned(candidate=candidate, sampling=sampling)
        if frozen_training_clock is None:
            budget.before_final()
            _check_budget(budget, birth, updates, last_clock)
            last_clock = budget.last_clock
            check_owned(candidate=candidate, sampling=sampling)

    try:
        with ProcessRssSampler() as memory:
            if warm_start:
                try:
                    prepared = prepare_warm_start(warm_report, memory, deadline, warm_boundary)
                except Exception:
                    warm_payload = pickle.dumps(warm_report, protocol=5)
                    raise
                warm_payload = pickle.dumps(warm_report, protocol=5)
                warm_guard = prepared["guard"]
                if (
                    type(prepared["updates"]) is not int
                    or prepared["updates"] != 128
                    or type(prepared["rows"]) is not int
                    or prepared["rows"] != 1152
                ):
                    raise ValueError("warm prefix must retain original128/1152 native counters")
                base_updates, base_rows = 128, 1152
                source = prepared["source"]
                if train_readouts:
                    diagnostic_sources = prepared["sources"]
                warm_guard()
            budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=8,
                    max_wall_seconds=min(BODY_SECONDS, max(0.0, deadline - monotonic()))
                    if warm_start
                    else BODY_SECONDS,
                    max_process_rss_bytes=RSS_LIMIT,
                ),
                monotonic,
            )
            budget.attach_memory(memory)
            birth = _binding(budget)
            last_clock = budget.last_clock
            require_torch()
            budget.before_final()
            if not warm_start:
                source = TorchRewardUpdateLearner(TorchRecurrentPolicy(148, 8, 7, 17))
            initial = deepcopy(source.snapshot_state())
            _counts(initial, base_updates, base_rows)
            if warm_guard is not None:
                warm_guard()
            runtime = ActorShadowRuntime(
                source,
                actor_version="a0",
                candidate_version="c0",
                clock=clock,
                budget=budget,
                max_experiences=8,
                max_consolidations=0,
            )
            actor_owner, candidate_owner, inbox_owner = (
                runtime.actor,
                runtime._candidate,
                runtime._inbox,
            )
            previous = deepcopy(runtime.candidate_snapshot().state)
            if not _same_state(initial, previous):
                raise ValueError("initial candidate differs from original source")
            rng = np.random.Generator(np.random.PCG64(29))
            expected_rng = deepcopy(rng.bit_generator.state)
            for index, slot in enumerate(training):
                guard()
                _counts(previous, base_updates + index, base_rows + rows)
                active = index
                collection_bytes = None
                slot.update(
                    status="running",
                    reason=None,
                    charged_updates_before=updates,
                    native_updates_before=previous.completed_updates,
                    native_training_examples_before=previous.training_examples,
                    sampler_before=deepcopy(expected_rng),
                )
                captured_receipt = committed = collection = None
                try:
                    budget.before_update()
                    guard()
                    collector = source.fork()
                    if collector is source:
                        raise ValueError("collection requires an independent owned source fork")
                    guard()
                    collector.restore_state(deepcopy(previous))
                    guard()
                    if not _same_state(previous, collector.snapshot_state()):
                        raise ValueError("collector failed to restore the current candidate")
                    layout = next(
                        v for v in predefined_layouts()[:3] if v.context == slot["context"]
                    )
                    collection = collect_minigrid_reward_episode(
                        collector,
                        layout,
                        slot["cue"],
                        episode_id=slot["episode_id"],
                        sampler=rng,
                        check_boundary=lambda: guard(sampling=False),
                    )
                    slot["collection"] = _serialize_collection(collection)
                    collection_bytes = pickle.dumps(slot["collection"], protocol=5)
                    if (
                        collection["status"] != "complete"
                        or collection["state_unchanged"] is not True
                        or collection["sampler_binding_preserved"] is not True
                        or collection["sampler_before"] != expected_rng
                        or collection["sampler_after"] != rng.bit_generator.state
                        or not _same_state(previous, collection["behavior_state"])
                        or not _same_state(previous, collector.snapshot_state())
                    ):
                        raise ValueError(
                            "collection differs from original behavior/source/sampling evidence"
                        )
                    expected_rng = deepcopy(rng.bit_generator.state)
                    guard()
                    update = RewardUpdate(collection["episode"], collection["behavior_state"])
                    identity = slot["episode_id"]
                    if (
                        update.episode.episode_id != identity
                        or update.episode.context != slot["context"]
                        or update.episode.role != "train"
                    ):
                        raise ValueError("reward packet differs from fixed TRAIN identity")
                    packet_rows = len(update.episode.transitions)
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
                    guard()
                    early = runtime.train_ready(max_updates=1)
                    slot["early_release_receipts"] = [asdict(value) for value in early]
                    if early:
                        raise ValueError(
                            "unreleased reward packet trained before matching identifier"
                        )
                    guard()
                    clock.advance_to(index + 1)
                    expected_tick = index + 1
                    runtime.record_label(
                        LabelArrival(
                            f"release-{index}",
                            identity,
                            identity,
                            index + 1,
                            "a0",
                            identity,
                            "train",
                        )
                    )
                    guard()

                    def observe(stage, read):
                        nonlocal captured_receipt, committed, updates, rows
                        origin = read()
                        slot["native_stages"].append(stage)
                        if stage == "started":
                            slot["update_attempted"] = True
                        if origin.receipt is not None:
                            if captured_receipt is None:
                                captured_receipt = deepcopy(origin.receipt)
                                receipts.append(captured_receipt)
                                updates += 1
                                rows += packet_rows
                            slot.update(receipt=asdict(captured_receipt), update_committed=True)
                        guard(candidate=False)
                        if origin.receipt is not None:
                            observed = origin.learner.snapshot_state()
                            _counts(observed, base_updates + index + 1, base_rows + rows)
                            if (
                                origin.receipt != captured_receipt
                                or origin.completed_updates != index + 1
                                or origin.targets != identity
                                or origin.source.sample_id != identity
                                or origin.receipt.episode_id != identity
                                or origin.receipt.event_id != f"release-{index}"
                                or origin.receipt.actor_version != "a0"
                                or origin.receipt.learner_version != "c0"
                                or origin.receipt.observed_at != index
                                or origin.receipt.arrived_at != index + 1
                                or origin.receipt.applied_at != index + 1
                                or origin.receipt.update_number != index + 1
                                or origin.receipt.diagnostic.definition != OBJECTIVE
                                or slot["receipt"] != asdict(captured_receipt)
                                or not _same_state(update, origin.features)
                                or not _same_state(update, origin.source.features)
                            ):
                                raise ValueError(
                                    "native reward receipt/state differs from original released update"
                                )
                            committed = deepcopy(observed)
                            slot["commit_validated"] = True

                    returned = runtime.train_ready(max_updates=1, native_observer=observe)
                    if len(returned) != 1 or captured_receipt != returned[0] or committed is None:
                        raise ValueError(
                            "native reward update did not return its validated receipt"
                        )
                    previous = committed
                    guard()
                    slot.update(status="complete", reason=None)
                except Exception as error:
                    if isinstance(error, MiniGridRewardCollectionFailed):
                        collection = error.report
                        slot["collection"] = _serialize_collection(collection)
                    slot.update(status="failed", reason=f"{type(error).__name__}: {error}")
                    raise
                finally:
                    slot.update(
                        charged_updates_after=budget.updates_completed,
                        sampler_after=deepcopy(rng.bit_generator.state),
                    )
                    if captured_receipt is not None:
                        slot.update(receipt=asdict(captured_receipt), update_committed=True)
                    try:
                        actual = runtime.candidate_snapshot().state
                        slot.update(
                            native_updates_after=actual.completed_updates,
                            native_training_examples_after=actual.training_examples,
                        )
                    except Exception as error:
                        slot["readback_error"] = f"{type(error).__name__}: {error}"
                    if collection is not None:
                        slot["collection"] = (
                            pickle.loads(collection_bytes)
                            if collection_bytes is not None
                            else _serialize_collection(collection)
                        )
                if momentum_control:
                    owned_episode = deepcopy(update.episode)
                    if not _same_state(asdict(owned_episode), slot["collection"]["episode"]):
                        raise ValueError("control input differs from original observed collection")
                    control_episodes.append(owned_episode)
                    control_episode_payloads.append(pickle.dumps(owned_episode, protocol=5))
                completed_records.append(deepcopy(slot))
                expected_slots[index] = pickle.dumps(slot, protocol=5)
                active = -1
                guard()
            budget.complete_memory()
            guard()
            if train_readouts:
                _counts(initial, 128, 1152)
                _counts(previous, 136, 1152 + rows)
                if (
                    updates != 8
                    or len(receipts) != 8
                    or not all(s["commit_validated"] for s in training)
                ):
                    raise ValueError(
                        "TRAIN diagnostics require all eight validated original reward commits"
                    )
                frozen_training_clock = budget.last_clock
                frozen_training_rss_owner = budget.process_rss_segment
                frozen_training_rss = deepcopy(budget.process_rss_segment)
                training_phase = dict(
                    updates_completed=updates,
                    reward_training_rows=rows,
                    native_updates=136,
                    native_training_examples=1152 + rows,
                    replay_examples_completed=budget.replay_examples_completed,
                    started_at=budget.started_at,
                    last_clock=frozen_training_clock,
                    elapsed_seconds=frozen_training_clock - budget.started_at,
                    limits=asdict(budget.budget),
                    process_rss=None
                    if budget.process_rss_segment is None
                    else asdict(budget.process_rss_segment),
                )
                if momentum_control:
                    guard()
                    if len(control_episodes) != 8 or any(
                        not _same_state(asdict(episode), slot["collection"]["episode"])
                        for episode, slot in zip(control_episodes, training, strict=True)
                    ):
                        raise ValueError(
                            "control requires all eight original observed TRAIN episodes"
                        )
                    try:
                        completed_control = run_momentum_control(
                            control_report,
                            source,
                            tuple(control_episodes),
                            memory,
                            deadline,
                            check_boundary=guard,
                        )
                    except Exception:
                        control_payload = pickle.dumps(control_report, protocol=5)
                        raise
                    control_payload = pickle.dumps(control_report, protocol=5)
                    control_guard = completed_control["guard"]
                    control_state = deepcopy(completed_control["state"])
                    _counts(control_state, 136, 1152 + rows)
                    control_guard()
                    guard()

                def completed_training_guard():
                    # The completed control proof calls this original parent
                    # guard before its own snapshot callbacks. Prove the parent
                    # again after its last callback, without a duplicate entry.
                    if control_guard is not None:
                        control_guard()
                    guard()

                try:
                    completed_diagnostics = run_train_diagnostics(
                        diagnostics,
                        source,
                        ((128, deepcopy(initial)), (136, deepcopy(previous))),
                        diagnostic_sources,
                        memory,
                        deadline,
                        training_phase=training_phase,
                        check_boundary=completed_training_guard if momentum_control else guard,
                        **({"control_state": control_state} if momentum_control else {}),
                        **(
                            {"extended_control_readouts": True} if extended_control_readouts else {}
                        ),
                    )
                except Exception:
                    diagnostic_payload = pickle.dumps(diagnostics, protocol=5)
                    raise
                diagnostic_payload = pickle.dumps(diagnostics, protocol=5)
                diagnostic_guard = completed_diagnostics["guard"]
                guard()
            report.update(status="complete", reason=None)
    except Exception as error:
        primary_error = f"{type(error).__name__}: {error}"
        report.update(status="incomplete", reason=primary_error)
    finally:
        try:
            if source is not None and initial is not None:
                report["source_unchanged"] = _same_state(initial, source.snapshot_state())
            if runtime is not None and initial is not None:
                report["actor_unchanged"] = _same_state(
                    initial, runtime.actor.snapshot_state().state
                )
            if report["source_unchanged"] is False or report["actor_unchanged"] is False:
                raise ValueError("source or serving actor changed during TRAIN reward route")
            if memory is not None and memory.start_bytes is not None:
                report["observed_process_rss"] = asdict(memory.snapshot())
                if report["observed_process_rss"]["peak_bytes"] > RSS_LIMIT:
                    raise ValueError("final observed RSS exceeded original cap")
            if budget is not None:
                _check_budget(budget, birth, updates, last_clock)
            if runtime is not None and rng is not None:
                # Final readback/RSS callbacks must finish before validation and
                # publication. No repair of public evidence is admitted here.
                guard()
            if control_guard is not None:
                control_guard()
                guard()
            if diagnostic_guard is not None:
                diagnostic_guard()
                guard()
        except Exception as error:
            report["readback_error"] = f"{type(error).__name__}: {error}"
            if primary_error is None:
                primary_error = report["readback_error"]
        # Paid output comes from bounded private captures only AFTER refusal
        # validation. This preserves evidence, never accepts a corrupt continuation.
        canonical = [pickle.loads(payload) for payload in expected_slots]
        for index, record in enumerate(completed_records):
            canonical[index] = deepcopy(record)
        if active >= 0:
            failed = deepcopy(slot)
            failed.update(status="failed", reason=primary_error)
            if captured_receipt is not None:
                failed.update(receipt=asdict(captured_receipt), update_committed=True)
            if collection_bytes is not None:
                failed["collection"] = pickle.loads(collection_bytes)
            canonical[active] = failed
        report["protocol"] = pickle.loads(protocol_bytes)
        report["training"] = canonical
        report["wake_updates"] = updates
        report["observed_budget_updates"] = None if budget is None else budget.updates_completed
        report["training_examples"] = rows
        report["work"] = _work(canonical)
        if warm_start:
            canonical_warm = (
                pickle.loads(warm_payload) if warm_payload is not None else deepcopy(warm_report)
            )
            report["warm_start"] = canonical_warm
            warm_work = canonical_warm.get("work") or {}
            report["total_new_native_updates"] = warm_work.get("charged_updates", 0) + updates
            report["total_new_training_rows"] = (
                warm_work.get("observed_native_training_rows", 0) + rows
            )
            report["total_new_work"] = dict(
                native_updates=report["total_new_native_updates"],
                training_rows=report["total_new_training_rows"],
                raw_score_forwards=warm_work.get("raw_score_forwards_completed", 0),
                raw_score_rows=warm_work.get("raw_score_rows_completed", 0),
                live_or_collection_predictions=warm_work.get("live_prediction_calls_returned", 0)
                + report["work"]["prediction_calls_returned"],
                environment_steps=warm_work.get("live_environment_steps_returned", 0)
                + report["work"]["environment_steps_returned"],
                observed_recurrent_unroll_steps=warm_work.get("observed_recurrent_unroll_steps", 0)
                + report["work"]["observed_recurrent_unroll_steps"],
                failure_work_may_be_unobserved=warm_work.get(
                    "failure_work_may_be_unobserved", False
                )
                or report["work"]["failure_work_may_be_unobserved"],
                teacher_episodes=0,
                final_episodes=0,
                probe_updates=0,
                replay_updates=0,
            )
            report["historical_work"] = canonical_warm.get("historical_work")
        if train_readouts:
            canonical_diagnostics = (
                pickle.loads(diagnostic_payload)
                if diagnostic_payload is not None
                else deepcopy(diagnostics)
            )
            report["train_diagnostics"] = canonical_diagnostics
            readout_work = canonical_diagnostics.get("work") or {}
            total = report["total_new_work"]
            total["raw_score_forwards"] += readout_work.get("raw_score_forwards_completed", 0)
            total["raw_score_rows"] += readout_work.get("raw_score_rows_completed", 0)
            total["live_or_collection_predictions"] += readout_work.get(
                "live_prediction_calls_returned", 0
            )
            total["environment_steps"] += readout_work.get("environment_steps_returned", 0)
            total["observed_recurrent_unroll_steps"] += readout_work.get(
                "observed_recurrent_unroll_steps", 0
            )
            total["train_diagnostic_sessions"] = readout_work.get("live_sessions_started", 0)
            total["failure_work_may_be_unobserved"] |= readout_work.get(
                "failure_work_may_be_unobserved", False
            )
        if momentum_control:
            canonical_control = (
                pickle.loads(control_payload)
                if control_payload is not None
                else deepcopy(control_report)
            )
            report["momentum_control"] = canonical_control
            control_work = canonical_control.get("work") or {}
            total = report["total_new_work"]
            control_updates = control_work.get("native_updates_committed", 0)
            control_rows = control_work.get("committed_training_rows", 0)
            report["total_new_native_updates"] += control_updates
            report["total_new_training_rows"] += control_rows
            total["native_updates"] += control_updates
            total["training_rows"] += control_rows
            total["control_repeated_observation_exposure_rows"] = control_rows
            total["control_native_updates_attempted"] = control_work.get(
                "native_updates_attempted", 0
            )
            total["observed_recurrent_unroll_steps"] += control_rows
            total["failure_work_may_be_unobserved"] |= control_work.get(
                "failure_work_may_be_unobserved", False
            )
        report.update(
            status="complete" if primary_error is None else "incomplete", reason=primary_error
        )
        report["total_seconds"] = monotonic() - started
        if report["status"] == "complete" and report["total_seconds"] >= whole_seconds:
            report.update(status="incomplete", reason="fixed whole-case wall-time limit exhausted")
    return report
