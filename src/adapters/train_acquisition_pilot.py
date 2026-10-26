"""Fixed train-role task-fit curve through the existing delayed stream.

Inputs are the fixed local teacher recipe and native learner defaults. Outputs
are resubstitution diagnostics, permanent work charges and isolation readbacks.
An explicit policy-evaluation route creates final packets only after training.
These recipes perform no selection, promotion, hyperparameter tuning or broad
generalization/adaptation certification.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from time import monotonic
from typing import Any

import numpy as np

from src.adapters import key_door_pilot as shared
from src.core.key_door_navigation import key_door_layout_record
from src.adapters import four_phase_replay as matched_replay
from src.adapters import four_phase_probes
from src.adapters import train_policy_diagnostics as train_live
from src.adapters import acquired_task_return as acquired_tasks
from src.adapters.return_policy_evaluation import (
    EVALUATION_POINTS,
    FOUR_PHASE_EVALUATION_POINTS,
    evaluate_policy,
    new_evaluation,
)
from src.adapters.return_replay_accounting import check_replay_progress, replay_evidence
from src.adapters.return_task_probes import run_return_checkpoint_probe
from src.adapters.key_door_learnability import probe_slots, valid_probe, valid_probes
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_recurrent_learner import TorchRecurrentLearner, TorchRecurrentPolicy
from src.adapters.torch_train_weighted_learner import TorchTrainWeightedLearner
from src.adapters.torch_recurrent_weighted import TORCH_RECURRENT_WEIGHTED_CE_ID
from src.app.coupled_attribution import AttributionBatch, AttributionStream, _validate_batch
from src.app.fresh_task_probe import (
    run_fresh_task_probe,
    _validate_batches as validate_probe_batches,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.app.stream_rehearsal import StreamRehearsal
from src.core.categorical_probability_scores import probability_scores
from src.core.context_return_metrics import (
    summarize_context_return,
    summarize_four_phase_return,
    summarize_acquired_return,
)
from src.core.key_door_navigation import KeyDoorLayout
from src.core.demonstration_episode import EpisodeReplaySchema, episode_arrays
from src.core.demonstration_class_weights import episode_class_weights
from src.core.probe_decision_diagnostics import decision_class_diagnostics
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch

UPDATES = 64
POINTS = (0, 8, 16, 32, 64)
FIXED_A_UPDATES = 128
FIXED_A_POINTS = (0, 8, 16, 32, 64, 72, 80, 96, 128)
RETURN_UPDATES = 192
RETURN_POINTS = (0, 8, 16, 32, 64, 72, 80, 96, 128, 136, 144, 160, 192)
ACQUIRED_RETURN_UPDATES = 320
ACQUIRED_RETURN_POINTS = (*RETURN_POINTS, 200, 208, 224, 256, 264, 272, 288, 320)
FOUR_PHASE_UPDATES = 256
FOUR_PHASE_POINTS = (*RETURN_POINTS, 200, 208, 224, 256)
FOUR_PHASE_CONTEXTS = ("A", "B", "C_train", "A")


def _third_training_layout() -> KeyDoorLayout:
    # A distinct identity/geometry avoids repurposing historical final C records.
    return shared.third_training_layout()


def describe_four_phase_protocol() -> dict[str, Any]:
    """Declare four train-only phases without collecting data or native resources."""
    p = describe_protocol()
    p.update(
        protocol_id="key_door_train_only_four_phase_v1",
        classification="descriptive_four_phase_train_resubstitution_not_generalization",
        training="A64/B64/C_train64/A64 alternating cues; six train-role traces/48source records,2048repeated exposures; continuous original optimizer",
        training_layouts=[
            key_door_layout_record(layout)
            for layout in (*shared.layouts()[:2], _third_training_layout())
        ],
        task_sequence=list(FOUR_PHASE_CONTEXTS),
        context_aliases={"C": "C_train"},
        phase_schedule=[
            dict(context=context, start=64 * i, end=64 * (i + 1))
            for i, context in enumerate(FOUR_PHASE_CONTEXTS)
        ],
        checkpoint_updates=list(FOUR_PHASE_POINTS),
        endpoint_updates=[0, 64, 128, 192, 256],
        scoring="six original train packets on isolated checkpoint forks at all17points; no final/probe/live data, promotion or selection",
        reference="pooled and per-layout empirical train-label priors/majority fixed before learning; offline train references, not causal online controls",
        metric_differences=dict(
            direction="d=-1 for CE, +1 for accuracy/exact_sequence; positive gain desirable, positive loss/gap worse",
            initial_A_gain="d*(A64-A0)",
            new_B_gain="d*(B128-B64)",
            new_C_gain="d*(C192-C128)",
            old_A_loss_after_B="d*(A64-A128)",
            old_A_loss_before_return="d*(A64-A192)",
            old_B_loss_after_C="d*(B128-B192)",
            old_B_loss_after_return="d*(B128-B256)",
            old_C_loss_after_return="d*(C192-C256)",
            return_A_gain="d*(A256-A192)",
            return_gap="d*(A64-A256)",
        ),
        limitations="one fixed seed/train resubstitution; C aliases new C_train, not historical final C; incomplete acquisition retained; no useful live skill, generalization, recovery savings, equal-compute comparison or CPC advantage",
    )
    p["planned_work"].update(
        native_train_calls=256,
        training_rows=2048,
        collected_source_records=48,
        score_forwards=102,
        score_rows=816,
        recurrent_unroll_steps=2864,
        teacher_episodes=6,
        environment_constructions=6,
        environment_steps=48,
    )
    return p


def describe_four_phase_evaluation_protocol() -> dict[str, Any]:
    """Freeze final readout after unchanged training, without executing it."""
    p = describe_four_phase_protocol()
    p.update(
        protocol_id="key_door_four_phase_policy_evaluation_v1",
        classification="exploratory_post_training_four_phase_cached_and_live_policy",
        final_evaluation=dict(
            checkpoint_updates=list(FOUR_PHASE_EVALUATION_POINTS),
            layouts=["A", "B", "C_train", "D"],
            cues=[1, -1],
            known_layouts=["A", "B", "C_train"],
            heldout_layouts=["D"],
            task_sequence=list(FOUR_PHASE_CONTEXTS),
            final_data_creation="only after all256training updates and stream.finish",
            prediction="argmax first tie/no masking/repair/retry/teacher assistance; max12decisions",
            selection="all fixed endpoints/layouts/cues retained; final never trains/promotes/tunes/selects",
        ),
        historical_prefix_reference="artifacts/runs/r42-four-phase-train-20261016/native.json",
        prefix_recreation="opaque native checkpoint states not saved; recreate unchanged256prefix for fixed final readout; require exact original train-field projection",
        limitations="one seed/local deterministic layouts; known finals repeat train geometry, D heldout for this invocation but evaluated in prior protocols; initialAfit incomplete/no equalAexposure control; no broad generalization/recovery savings/CPC advantage",
    )
    p["planned_work"].update(
        score_forwards=142,
        score_rows=1136,
        train_fit_score_forwards=102,
        train_fit_score_rows=816,
        final_cached_forwards=40,
        final_cached_rows=320,
        final_episodes=8,
        live_episodes=40,
        maximum_live_forwards=480,
        maximum_native_forwards=622,
        maximum_recurrent_unroll_steps=3664,
        recurrent_unroll_steps=None,
        teacher_episodes=14,
        environment_constructions=54,
        maximum_environment_steps=592,
        environment_steps=None,
    )
    return p


def describe_four_phase_replay_protocol(*, control: str) -> dict[str, Any]:
    """Describe one matched existing replay control without native work."""
    matched_replay.validate_control(control)
    return matched_replay.describe_protocol(
        describe_four_phase_evaluation_protocol(), control=control
    )


def describe_protocol() -> dict[str, Any]:
    """Describe without collecting data, importing Torch or creating resources."""
    return dict(
        protocol_id="key_door_train_only_acquisition_v1",
        classification="descriptive_train_resubstitution_not_generalization",
        architecture=dict(input_dim=8, hidden_dim=8, num_classes=5, seed=17),
        native_defaults=dict(learning_rate=0.03, momentum=0.9),
        objective="unchanged native recurrent episode mean CE/full ordered BPTT",
        training="two A cue +/- complete train-role traces collected once;64 unique packet IDs alternating copies,512exposures/16collected source records (numeric values may repeat)",
        checkpoint_updates=list(POINTS),
        label_delay=1,
        scoring="same two train packets at each point on owned checkpoint forks; no final/probe-test/live evaluation",
        reference="pooled train-label empirical prior, argmax first-tie majority; fixed before learning",
        directions=dict(
            ce="reference CE minus learner CE",
            accuracy="learner minus majority",
            exact_sequence="learner minus majority",
        ),
        isolation="fresh source -> isolated stream source -> owned candidate; train-fit scoring uses separate owned checkpoint forks; no benchmark/deployed learner",
        limits=dict(
            body_seconds=90,
            parent_seconds=120,
            process_rss_bytes=shared.MAX_RSS,
            json_bytes=8 * 1024**2,
            cpu_threads=1,
            gpu=0,
        ),
        planned_work=dict(
            native_train_calls=64,
            training_rows=512,
            collected_source_records=16,
            score_forwards=10,
            score_rows=80,
            recurrent_unroll_steps=592,
            teacher_episodes=2,
            environment_constructions=2,
            environment_steps=16,
            final_episodes=0,
            probes=0,
            replay_calls=0,
            reward_updates=0,
            live_episodes=0,
            svd=0,
        ),
        limitations="one seed/one layout/resubstitution only; no useful new-task retention/recovery, generalization or circadian benefit claim; fixed budget has no early stopping",
    )


def describe_live_protocol() -> dict[str, Any]:
    """Describe the unchanged training dose plus isolated train-only live diagnostics."""
    p = describe_protocol()
    p.update(
        protocol_id="key_door_train_only_closed_loop_acquisition_v1",
        classification="descriptive_train_only_closed_loop_acquisition",
        diagnostic_points=[0, 64],
        diagnostic_data_role="train",
        action_conditioning="legal_actions_v1",
        live_policy="owned checkpoint forks; first greedy conditional argmax on current visible legal support, max12decisions; no fallback/retry/repair",
        diagnostic_timing="all64commits and stream.finish before four learner A+/- episodes at0/64 and two fixed prior episodes once",
        reference="unchanged pooled train-label empirical prior; same legal information; no SGD/native forward; diagnostic reference, not matched-compute learner",
        reference_refusal="finite normalized prior with exactly zero actual legal support mass is recorded as refused; no completed metric imputation; comparison null unless both prior cues complete",
        scoring="original ten train teacher-forced scores unchanged; additional closed-loop train resubstitution only; no final/probe data",
        live_directions="learner success/mean-return64 minus0; endpoint learner minus prior only if both prior cues complete",
        limitations=p["limitations"]
        + "; legal information is additional; closed-loop train fit is not generalization, retained/recovered skill or CPC advantage; undefined prior comparison is not a learner win",
    )
    p["planned_work"].update(
        recurrent_unroll_steps=640,
        maximum_native_predict_calls=58,
        learner_live_episodes=4,
        prior_live_episodes=2,
        live_episodes=6,
        maximum_learner_live_forwards=48,
        maximum_prior_callbacks=24,
        environment_constructions=8,
        maximum_environment_steps=88,
    )
    # An upper bound includes both teacher and diagnostic physics; actual work is reported.
    p["planned_work"].pop("environment_steps")
    return p


def run_live_pilot() -> dict[str, Any]:
    return _run_pilot(train_policy_diagnostics=True)


def describe_weighted_live_protocol() -> dict[str, Any]:
    """Fix the existing weighted objective at the same train-only dose/readout."""
    p = describe_live_protocol()
    p.update(
        protocol_id="key_door_weighted_train_only_closed_loop_acquisition_v1",
        objective=TORCH_RECURRENT_WEIGHTED_CE_ID,
        weight_rule="current_train_episode_inverse_present_class_frequency_v1",
        scoring="same raw unweighted train CE/accuracy/sequences at all five fixed points; same legal train-only live0/64 after finish",
        historical_mean_reference="artifacts/runs/r21-train-live-20261016/pilot.json",
        comparison_directions="mean minus weighted raw CE; weighted minus mean accuracy/exact sequence/live success/return; all fixed points and cues retained",
        limitations=p["limitations"]
        + "; existing weighted BC8 negative retained; equal updates/exposures, objective-validation/owned-metadata overhead counted; no end-to-end walltime advantage claim",
    )
    return p


def run_weighted_live_pilot() -> dict[str, Any]:
    return _run_pilot(train_policy_diagnostics=True, weighted_train=True)


def describe_fixed_exposure_protocol() -> dict[str, Any]:
    """Declare the missing A128 control; this dose matches existing return exposure."""
    p = describe_live_protocol()
    p.update(
        protocol_id="key_door_train_only_fixed_a_exposure_v1",
        training="two original A cue+/- train traces,128 alternating unique episode IDs/1024exposures/16source records; mean CE, no B training",
        checkpoint_updates=list(FIXED_A_POINTS),
        diagnostic_points=[0, 128],
        scoring="eighteen raw unweighted train scores/144rows at nine fixed points; isolated legal train-only live0/128 after finish, no final/probe data",
        diagnostic_timing="all128commits and stream.finish before four owned learner A+/- episodes at0/128 and two fixed prior episodes once",
        live_directions="learner success/mean-return128 minus0; endpoint learner minus prior only if both prior cues complete",
        historical_return_reference="artifacts/runs/r21-train-context-return-20261016/native.json",
        exposure_alignment=[
            dict(a_exposures=n, control_updates=n, return_updates=n if n <= 64 else n + 64)
            for n in FIXED_A_POINTS
        ],
        comparison="raw A train metrics only; return minus control CE, control minus return accuracy/exact sequence; preserve all points/cues and postB128 separately at64Aexposures",
        comparison_work="control128nativeupdates/1024trainrows vs return192/1536 (128A+64B); scoring/source/diagnostic costs differ and remain explicit",
        limitations=p["limitations"]
        + "; initialA64 incomplete; dose128 follows existing A exposure, not outcome tuning; no acquired-skill restoration/savings/causal transfer; standalone legal live not comparable to old unmasked return live",
    )
    p["planned_work"].update(
        native_train_calls=128,
        training_rows=1024,
        score_forwards=18,
        score_rows=144,
        recurrent_unroll_steps=1216,
        maximum_native_predict_calls=66,
    )
    return p


def run_fixed_exposure_pilot() -> dict[str, Any]:
    return _run_pilot(train_policy_diagnostics=True, fixed_a_exposure=True)


def describe_acquired_return_protocol() -> dict[str, Any]:
    """Declare fixed acquisition/retention readouts after an unchanged trained prefix."""
    p = describe_protocol()
    p.update(
        protocol_id="key_door_train_only_acquired_task_return_v1",
        classification="descriptive_train_acquisition_retention_relearning",
        training="A64/B64/A64/C_train64/A64; original192prefix unchanged,320unique IDs/2560exposures/48collected source records; continuous original optimizer",
        training_layouts=[
            key_door_layout_record(layout)
            for layout in (*shared.layouts()[:2], _third_training_layout())
        ],
        task_sequence=["A", "B", "A", "C_train", "A"],
        checkpoint_updates=list(ACQUIRED_RETURN_POINTS),
        diagnostic_points=[192, 256, 320],
        diagnostic_data_role="train",
        action_conditioning="legal_actions_v1",
        scoring="six original TRAINpackets on owned forks at21points/126raw scores1008rows; no final/probe data",
        live_policy="same legal greedy first argmax, max12steps; A+/- and C_train+/- at each owned192/256/320 state after ALL320commits+finish",
        initial_condition="both A192 complete teacher sequences AND both actual same-law A192 successes; affects interpretation only, all fixed training retained if unmet",
        reference="pooled empirical prior from all declared TRAINlabels including future C; descriptive offline reference, not a causal online control; no extra prior rollouts",
        metric_differences=dict(
            directions="d=-1CE,+1accuracy/sequence/success/return; positive gains better, positive losses/gap worse",
            new_C_gain="d*(C256-C192)",
            old_A_loss="d*(A192-A256)",
            return_A_gain="d*(A320-A256)",
            return_gap="d*(A192-A320)",
            new_C_loss_after_return="d*(C256-C320)",
        ),
        historical_prefix_reference="artifacts/runs/r21-train-context-return-20261016/native.json",
        prefix_recreation="native optimizer tensors unavailable; necessary unchanged192prefix to own continuation states; exact originalA/Braw projections/source/first192receipts required, not inferred old tensor identity",
        limitations="TRAINresubstitution only/C_train new to this prefix, not unseen generalization; no final-selected context/dose/seed, gating/extension or promotion; initial-condition failure inconclusive; no savings/causaltransfer/CPC advantage; equal-total A192 causalB-history control deferred",
    )
    p["planned_work"].update(
        native_train_calls=320,
        training_rows=2560,
        collected_source_records=48,
        score_forwards=126,
        score_rows=1008,
        recurrent_unroll_steps=3712,
        teacher_episodes=6,
        environment_constructions=18,
        live_episodes=12,
        maximum_learner_live_forwards=144,
        maximum_native_predict_calls=270,
        maximum_environment_steps=192,
    )
    p["planned_work"].pop("environment_steps")
    return p


def run_acquired_return_pilot() -> dict[str, Any]:
    return _run_pilot(acquired_return=True)


def _validate_acquired_replay(control: str) -> None:
    if type(control) is not str or control not in ("recent48", "coverage"):
        raise ValueError("acquired replay control must be recent48 or coverage")


def describe_acquired_replay_protocol(*, control: str) -> dict[str, Any]:
    """Compose existing matched rehearsal without selecting data or native work."""
    _validate_acquired_replay(control)
    p = describe_acquired_return_protocol()
    p.update(
        protocol_id=f"key_door_acquired_task_{control}_replay_v1",
        replay_control=control,
        training="A64/B64/A64/C_train64/A64;320wake episodes/2560rows plus160released episode replay updates/1280rows; original continuous mean-CE optimizer, prefix differs under replay",
        live_policy="same legal greedy first argmax, max12steps; A+/- and C_train+/- at owned wake192/256/320 (native288/384/480) only after all480native commits and finish",
        replay=dict(
            policy=asdict(matched_replay.replay_policy(control)),
            wake_updates=320,
            replay_updates=160,
            replay_exposures=1280,
            native_updates=480,
            native_checkpoint_updates=[288, 384, 480],
            selection="existing released complete episode policy; no future labels",
            retention_scope="48 transitions/12096 encoded array bytes; metadata, event records, copies and native states in total RSS/time",
        ),
        historical_prefix_reference=None,
        prefix_recreation="replay changes the prefix; each arm verifies its own acquisition, no old192 raw/tensor parity claim",
        acquisition_conditions="A192 and C256 each require both complete teacher sequences and both matching-law live successes; missing premise makes its retention interpretation inconclusive, never gates fixed training",
        comparison="coverage minus recent48 pooled A/C live success and mean return at320; all endpoints/cues/raw curves and signed metrics retained; CE recent minus coverage, other raw metrics coverage minus recent",
        comparison_limits="same wake/replay counts and payload ceilings, not equal walltime; coverage uses declared context/cue metadata, recent selection can depend on cue parity; historical no-replay320 reference is unequal compute; no pure mechanism causality",
    )
    p["planned_work"].update(
        native_train_calls=480,
        training_rows=3840,
        wake_training_rows=2560,
        replay_calls=160,
        replay_rows=1280,
        recurrent_unroll_steps=4992,
    )
    return p


def run_acquired_replay_pilot(*, control: str) -> dict[str, Any]:
    _validate_acquired_replay(control)
    return _run_pilot(acquired_replay=control)


def _weighted_objective(base: tuple[AttributionBatch, ...]) -> dict[str, Any]:
    """Describe only current train-episode weights, never diagnostic/final labels."""
    if any(batch.episode is None for batch in base):
        raise ValueError("weighted training requires complete original train episodes")
    return dict(
        objective_id=TORCH_RECURRENT_WEIGHTED_CE_ID,
        weight_rule="current_train_episode_inverse_present_class_frequency_v1",
        class_weights=[
            dict(
                episode_id=batch.episode.episode_id,
                values=list(episode_class_weights(batch.episode)),
            )
            for batch in base
            if batch.episode is not None
        ],
    )


def describe_return_protocol() -> dict[str, Any]:
    """A separately fixed A→B→A experiment; no native work during description."""
    p = describe_protocol()
    p.update(
        protocol_id="key_door_train_only_context_return_v1",
        classification="descriptive_train_only_context_return_not_generalization",
        training="A64/B64/A64 alternating cues; four collected train-only traces/32source records,1536repeated exposures; same continuous optimizer",
        checkpoint_updates=list(RETURN_POINTS),
        phase_schedule=[
            dict(context=c, start=n, end=n + 64) for c, n in (("A", 0), ("B", 64), ("A", 128))
        ],
        scoring="all four original A/B cue train packets at all13points on owned checkpoint forks; resubstitution only",
        reference="pooled and per-layout empirical train-label priors/majority fixed before learning",
        directions=dict(
            ce="gains earlier minus later; old loss A128 minus A64; return gap A192 minus A64",
            accuracy="gains later minus earlier; old loss A64 minus A128; return gap A64 minus A192",
            exact_sequence="same directions as accuracy",
        ),
        historical_prefix_reference="artifacts/runs/r21-train-acquisition-20261016/native.json",
        prefix_recreation="opaque post64 optimizer/gradient checkpoint not saved; recreate necessary prefix for continuous B/return updates; compare all five original A score points",
        limitations="one seed/train resubstitution, initial A fit incomplete; CE return gains do not establish restored acquired skill, held-out success or CPC benefit",
    )
    p["planned_work"].update(
        native_train_calls=192,
        training_rows=1536,
        collected_source_records=32,
        score_forwards=52,
        score_rows=416,
        recurrent_unroll_steps=1952,
        teacher_episodes=4,
        environment_constructions=4,
        environment_steps=32,
    )
    return p


def _packets(
    *,
    return_context: bool = False,
    four_phase: bool = False,
    fixed_a_exposure: bool = False,
    acquired_return: bool = False,
) -> tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]]:
    if type(acquired_return) is not bool or (
        acquired_return
        and any(value is not False for value in (return_context, four_phase, fixed_a_exposure))
    ):
        raise ValueError("acquired return requires an exact bool and standalone five-phase recipe")
    if type(fixed_a_exposure) is not bool or (
        fixed_a_exposure and (return_context is not False or four_phase is not False)
    ):
        raise ValueError("fixed A exposure requires an exact bool and no context-switch recipe")
    base = tuple(
        shared._collect(f"train-fit-base-{cue}", shared.layouts()[0], cue, "train")
        for cue in (1, -1)
    )
    if return_context or four_phase or acquired_return:
        base += tuple(
            shared._collect(f"train-return-base-B-{cue}", shared.layouts()[1], cue, "train")
            for cue in (1, -1)
        )
    if four_phase or acquired_return:
        base += tuple(
            shared._collect(f"train-four-phase-C-{cue}", _third_training_layout(), cue, "train")
            for cue in (1, -1)
        )
    seen: set[str] = set()
    for batch in base:
        _validate_batch(batch, "train", seen)
        if batch.episode is None or batch.num_classes != 5 or batch.features.shape != (8, 8):
            raise ValueError("train-fit requires complete8step D8/K5 training packets")
    training = []
    updates = (
        ACQUIRED_RETURN_UPDATES
        if acquired_return
        else FIXED_A_UPDATES
        if fixed_a_exposure
        else FOUR_PHASE_UPDATES
        if four_phase
        else RETURN_UPDATES
        if return_context
        else UPDATES
    )
    for i in range(updates):
        offset = (
            (0, 2, 0, 4, 0)[i // 64]
            if acquired_return
            else (0, 2, 4, 0)[i // 64]
            if four_phase
            else 2
            if return_context and 64 <= i < 128
            else 0
        )
        original = base[offset + i % 2]
        assert original.episode is not None
        episode = replace(original.episode, episode_id=f"train-fit-{i}")
        features, targets = episode_arrays(episode)
        training.append(
            AttributionBatch(
                "train",
                episode.context,
                tuple(f"{episode.episode_id}:step:{t.step}" for t in episode.transitions),
                features,
                targets,
                episode,
                5,
            )
        )
    return base, tuple(training)


def describe_return_evaluation_protocol() -> dict[str, Any]:
    """Separate fixed final readout; training recipe and all its costs remain."""
    p = describe_return_protocol()
    p.update(
        protocol_id="key_door_return_policy_evaluation_v1",
        classification="exploratory_post_training_cached_and_live_return_policy",
        final_evaluation=dict(
            checkpoint_updates=list(EVALUATION_POINTS),
            layouts=["A", "B", "C", "D"],
            cues=[1, -1],
            known_layouts=["A", "B"],
            heldout_layouts=["C", "D"],
            task_sequence=["A", "B", "A"],
            final_data_creation="only after all192training updates and stream.finish",
            prediction="argmax first tie/no masking/repair/retry/teacher assistance; max12decisions",
            selection="all fixed endpoints and contexts retained; final never trains/promotes/tunes/selects",
        ),
        historical_prefix_reference="artifacts/runs/r21-train-context-return-20261016/native.json",
        prefix_recreation="restorable opaque native optimizer checkpoints not saved; recreate unchanged192prefix for fixed final readout; require exact old train-field projection",
        limitations="one seed/local layouts; known A/B finals repeat deterministic trace domain, C/D unseen layouts; initialAfit incomplete/no equalAexposurecontrol; no broad generalization, recovery savings or CPC advantage established",
    )
    p["planned_work"].update(
        score_forwards=84,
        score_rows=672,
        train_fit_score_forwards=52,
        train_fit_score_rows=416,
        final_cached_forwards=32,
        final_cached_rows=256,
        final_episodes=8,
        live_episodes=32,
        maximum_live_forwards=384,
        maximum_native_forwards=468,
        maximum_recurrent_unroll_steps=2592,
        recurrent_unroll_steps=None,
        teacher_episodes=12,
        environment_constructions=44,
        environment_steps=None,
        maximum_environment_steps=480,
        teacher_steps=96,
    )
    return p


def describe_return_replay_protocol() -> dict[str, Any]:
    """Fixed episode rehearsal on the same wake recipe, with all extra work explicit."""
    p = describe_return_evaluation_protocol()
    p.update(
        protocol_id="key_door_return_episode_replay_v1",
        replay=dict(
            policy=asdict(shared.replay_policy()),
            episode_schema=asdict(EpisodeReplaySchema(8, 8, 5)),
            max_replay_examples=768,
            checkpoint_clocks="updates identify wake counts; native includes committed replay",
            motivation="B training-fit forgetting observed before final evaluation",
            comparison="accepted fixed replay-off result; additional compute/exposure, not equal compute",
        ),
        historical_prefix_reference="artifacts/runs/r42-return-policy-evaluation-20261016/native.json",
        prefix_recreation="replay changes the trajectory from wake2; no claim of unchanged training scores",
        limitations="one seed/fixed existing replay policy; extra computation and repeated exposure; no final-driven tuning, broad generalization, recovery savings or CPC advantage established",
    )
    p["planned_work"].update(
        wake_train_calls=192,
        wake_training_rows=1536,
        native_train_calls=288,
        training_rows=2304,
        replay_calls=96,
        replay_rows=768,
        maximum_recurrent_unroll_steps=3360,
    )
    return p


def describe_return_probe_protocol(*, episode_replay: bool = False) -> dict[str, Any]:
    """The existing fixed E16 probe at initial and final owned return states."""
    if type(episode_replay) is not bool:
        raise ValueError("episode_replay must be an exact bool")
    p = (
        describe_return_replay_protocol()
        if episode_replay
        else describe_return_evaluation_protocol()
    )
    p.update(
        protocol_id="key_door_return_fresh_task_probe_v1"
        + ("_replay" if episode_replay else "_off"),
        probe=dict(
            layout="existing E(3,3,True,0.75)",
            source_wake_checkpoints=[0, 192],
            source_native_checkpoints=[0, 288 if episode_replay else 192],
            training_episodes=16,
            evaluation_episodes=2,
            checkpoints=[0, 4, 8, 16],
            independent_updates_per_cohort=16,
            metrics="all reserved E CE/accuracy/class/sequence curves; CE0-minus-CE16 gain, higher accuracy/sequence better",
            isolation="disposable owned forks and independent16-update budgets; no benchmark/replay/deployed update or selection",
            final_data_creation="after192benchmark wakes+finish and both completed probes",
            limitation="same deterministic E trace domain, not generalization; no equal-compute or CPC benefit established",
        ),
    )
    p["planned_work"].update(
        native_train_calls=p["planned_work"]["native_train_calls"] + 32,
        training_rows=p["planned_work"]["training_rows"] + 256,
        probes=2,
        probe_train_calls=32,
        probe_training_rows=256,
        probe_score_forwards=16,
        probe_score_rows=128,
        score_forwards=100,
        score_rows=800,
        maximum_native_forwards=484,
        maximum_recurrent_unroll_steps=p["planned_work"]["maximum_recurrent_unroll_steps"] + 384,
        teacher_episodes=30,
        teacher_steps=240,
        environment_constructions=62,
        maximum_environment_steps=624,
    )
    return p


def _point(updates: int, predictions: tuple[Any, ...], base: Any) -> dict[str, Any]:
    scores, details = {}, {}
    for prediction, batch in zip(predictions, base, strict=True):
        scores[batch.context] = probability_scores(prediction, batch.targets, 5)
        detail = decision_class_diagnostics(prediction, batch.targets, 5)
        detail.update(
            row_ids=list(batch.row_ids),
            episode_id=batch.episode.episode_id,
            sequence_kind="complete_train_episode",
            role="train",
        )
        details[batch.context] = detail
    return dict(
        updates=updates,
        scores=scores,
        decision_diagnostics=details,
        mean_ce=sum(s["ce"] for s in scores.values()) / len(base),
        mean_accuracy=sum(s["accuracy"] for s in scores.values()) / len(base),
        exact_sequence_accuracy=sum(
            all(d["correct"] for d in detail["decisions"]) for detail in details.values()
        )
        / len(base),
    )


def _reference(base: Any) -> dict[str, Any]:
    targets = np.concatenate([b.targets[:, 0] for b in base]).astype(np.int64)
    counts = np.bincount(targets, minlength=5)
    probabilities = counts.astype(np.float64) / len(targets)
    result = _point(0, tuple(np.tile(probabilities, (len(b.features), 1)) for b in base), base)
    result.update(
        class_counts=counts.tolist(),
        probabilities=probabilities.tolist(),
        majority_class=int(np.argmax(probabilities)),
        source_role="train",
    )
    return result


def _score(
    source: Any, state: Any, base: Any, budget: Any, work: Any, count: int
) -> dict[str, Any]:
    budget.before_final()
    evaluator = source.fork()
    if evaluator is source:
        raise ValueError("train-fit scorer requires an owned fork")
    evaluator.restore_state(deepcopy(state))
    predictions = []
    for batch in base:
        budget.before_final()
        work["score_forwards_attempted"] += 1
        work["score_rows_attempted"] += len(batch.features)
        prediction = evaluator.predict(batch.features.copy())
        work["score_forwards_completed"] += 1
        work["score_rows_completed"] += len(batch.features)
        if not _same_state(state, evaluator.snapshot_state()):
            raise ValueError("train-fit scoring changed checkpoint state")
        predictions.append(prediction)
        budget.before_final()
    return _point(count, tuple(predictions), base)


def _isolation(source: Any, isolated: Any, initial: Any, stream: Any) -> dict[str, bool]:
    observed = dict(
        source_unchanged=_same_state(initial, source.snapshot_state()),
        isolated_source_unchanged=_same_state(initial, isolated.snapshot_state()),
    )
    if stream is not None:
        observed["actor_unchanged"] = _same_state(
            initial, stream._runtime.actor.snapshot_state().state
        )
    if not all(observed.values()):
        raise ValueError("train-fit source or serving actor isolation changed")
    return observed


def run_pilot() -> dict[str, Any]:
    """Run the complete fixed curve; outcome quality never controls admission."""
    return _run_pilot()


def run_return_pilot() -> dict[str, Any]:
    """Run fixed acquisition/retention/return phases without evaluation feedback."""
    return _run_pilot(return_context=True)


def run_four_phase_pilot() -> dict[str, Any]:
    """Run the declared third context and return on the existing train-only stream."""
    return _run_pilot(four_phase=True)


def run_four_phase_evaluation_pilot() -> dict[str, Any]:
    """Finish unchanged four-phase training before collecting all final outcomes."""
    return _run_pilot(four_phase_policy=True)


def run_four_phase_replay_pilot(*, control: str) -> dict[str, Any]:
    """Run one frozen replay arm; an isolated CLI process owns its final data."""
    return _run_pilot(four_phase_replay=matched_replay.validate_control(control))


def describe_four_phase_probe_protocol(*, control: str) -> dict[str, Any]:
    """Describe disposable E16 probes without collecting any role or native work."""
    control = matched_replay.validate_control(control)
    return four_phase_probes.describe_protocol(
        describe_four_phase_replay_protocol(control=control), control=control
    )


def run_four_phase_probe_pilot(*, control: str) -> dict[str, Any]:
    """Probe initial/final owned states with separate budgets and disposable weights."""
    return _run_pilot(
        four_phase_replay=matched_replay.validate_control(control), four_phase_probes=True
    )


def run_return_evaluation_pilot() -> dict[str, Any]:
    """Run unchanged training, then every frozen cached/live final outcome."""
    return _run_pilot(return_context=True, evaluate_policy_after_training=True)


def run_return_replay_pilot() -> dict[str, Any]:
    """Use the existing bounded episode helper, then fixed post-training readouts."""
    return _run_pilot(return_context=True, evaluate_policy_after_training=True, episode_replay=True)


def run_return_probe_pilot(*, episode_replay: bool = False) -> dict[str, Any]:
    """Measure initial/post-return E learning without changing benchmark learners."""
    if type(episode_replay) is not bool:
        raise ValueError("episode_replay must be an exact bool")
    return _run_pilot(
        return_context=True,
        evaluate_policy_after_training=True,
        episode_replay=episode_replay,
        fresh_task_probes=True,
    )


def _return_outcomes(points: list[dict[str, Any]]) -> dict[str, Any]:
    endpoints = []
    for point in points:
        if point["updates"] not in (0, 64, 128, 192):
            continue
        row: dict[str, Any] = dict(updates=point["updates"])
        for layout in ("A", "B"):
            names = (f"{layout}-red", f"{layout}-blue")
            row[layout] = dict(
                ce=sum(point["scores"][c]["ce"] for c in names) / 2,
                accuracy=sum(point["scores"][c]["accuracy"] for c in names) / 2,
                exact_sequence_accuracy=sum(
                    all(d["correct"] for d in point["decision_diagnostics"][c]["decisions"])
                    for c in names
                )
                / 2,
            )
        endpoints.append(row)
    return summarize_context_return(tuple(endpoints))


def _four_phase_outcomes(points: list[dict[str, Any]]) -> dict[str, Any]:
    endpoints = []
    for point in points:
        if point["updates"] not in (0, 64, 128, 192, 256):
            continue
        row: dict[str, Any] = dict(updates=point["updates"])
        for alias, layout in (("A", "A"), ("B", "B"), ("C", "C_train")):
            names = (f"{layout}-red", f"{layout}-blue")
            row[alias] = dict(
                ce=sum(point["scores"][c]["ce"] for c in names) / 2,
                accuracy=sum(point["scores"][c]["accuracy"] for c in names) / 2,
                exact_sequence_accuracy=sum(
                    all(d["correct"] for d in point["decision_diagnostics"][c]["decisions"])
                    for c in names
                )
                / 2,
            )
        endpoints.append(row)
    return summarize_four_phase_return(tuple(endpoints))


def _acquired_return_outcomes(points: list[dict[str, Any]]) -> dict[str, Any]:
    endpoints = []
    for point in points:
        if point["updates"] not in (192, 256, 320):
            continue
        row: dict[str, Any] = {"updates": point["updates"]}
        for context in ("A", "C_train"):
            names = (f"{context}-red", f"{context}-blue")
            row[context] = dict(
                ce=sum(point["scores"][c]["ce"] for c in names) / 2,
                accuracy=sum(point["scores"][c]["accuracy"] for c in names) / 2,
                exact_sequence_accuracy=sum(
                    all(d["correct"] for d in point["decision_diagnostics"][c]["decisions"])
                    for c in names
                )
                / 2,
            )
        endpoints.append(row)
    return summarize_acquired_return(tuple(endpoints))


def _run_pilot(
    *,
    return_context: bool = False,
    evaluate_policy_after_training: bool = False,
    episode_replay: bool = False,
    fresh_task_probes: bool = False,
    four_phase: bool = False,
    four_phase_policy: bool = False,
    four_phase_replay: str | None = None,
    four_phase_probes: bool = False,
    train_policy_diagnostics: bool = False,
    weighted_train: bool = False,
    fixed_a_exposure: bool = False,
    acquired_return: bool = False,
    acquired_replay: str | None = None,
) -> dict[str, Any]:
    if acquired_replay is not None:
        _validate_acquired_replay(acquired_replay)
        if four_phase_replay is not None or any(
            value is not False
            for value in (
                acquired_return,
                return_context,
                evaluate_policy_after_training,
                episode_replay,
                fresh_task_probes,
                four_phase,
                four_phase_policy,
                four_phase_probes,
                train_policy_diagnostics,
                weighted_train,
                fixed_a_exposure,
            )
        ):
            raise ValueError("acquired replay requires its standalone five-phase recipe")
        acquired_return = True
    if type(acquired_return) is not bool or (
        acquired_return
        and (
            four_phase_replay is not None
            or any(
                value is not False
                for value in (
                    return_context,
                    evaluate_policy_after_training,
                    episode_replay,
                    fresh_task_probes,
                    four_phase,
                    four_phase_policy,
                    four_phase_probes,
                    train_policy_diagnostics,
                    weighted_train,
                    fixed_a_exposure,
                )
            )
        )
    ):
        raise ValueError("acquired return requires an exact bool and standalone five-phase recipe")
    diagnostics_enabled = train_policy_diagnostics is True or acquired_return
    if type(fixed_a_exposure) is not bool or (
        fixed_a_exposure
        and (
            train_policy_diagnostics is not True
            or any(
                value is not False
                for value in (
                    weighted_train,
                    return_context,
                    evaluate_policy_after_training,
                    episode_replay,
                    fresh_task_probes,
                    four_phase,
                    four_phase_policy,
                    four_phase_probes,
                )
            )
            or four_phase_replay is not None
        )
    ):
        raise ValueError(
            "fixed A exposure requires an exact bool and standalone mean train diagnostics"
        )
    if type(weighted_train) is not bool or (
        weighted_train and train_policy_diagnostics is not True
    ):
        raise ValueError(
            "weighted training requires an exact bool and standalone train diagnostics"
        )
    if type(train_policy_diagnostics) is not bool or (
        train_policy_diagnostics
        and (
            any(
                value is not False
                for value in (
                    return_context,
                    evaluate_policy_after_training,
                    episode_replay,
                    fresh_task_probes,
                    four_phase,
                    four_phase_policy,
                    four_phase_probes,
                )
            )
            or four_phase_replay is not None
        )
    ):
        raise ValueError(
            "train policy diagnostics require an exact bool and the standalone64-update recipe"
        )
    if type(four_phase_probes) is not bool or (four_phase_probes and four_phase_replay is None):
        raise ValueError("four-phase probes require an exact bool and a declared replay control")
    if four_phase_replay is not None:
        matched_replay.validate_control(four_phase_replay)
        if any(
            (
                return_context,
                evaluate_policy_after_training,
                episode_replay,
                fresh_task_probes,
                four_phase,
                four_phase_policy,
            )
        ):
            raise ValueError("four-phase replay cannot combine other recipe options")
    if type(four_phase_policy) is not bool or (
        four_phase_policy
        and (
            four_phase
            or return_context
            or evaluate_policy_after_training
            or episode_replay
            or fresh_task_probes
        )
    ):
        raise ValueError("four-phase policy recipe cannot combine other recipe options")
    if type(four_phase) is not bool or (
        four_phase
        and (
            return_context or evaluate_policy_after_training or episode_replay or fresh_task_probes
        )
    ):
        raise ValueError("four-phase train-only recipe cannot combine other recipe options")
    if four_phase_policy:
        four_phase = evaluate_policy_after_training = True
    if four_phase_replay is not None:
        four_phase = four_phase_policy = evaluate_policy_after_training = episode_replay = True
    if evaluate_policy_after_training and not (return_context or four_phase_policy):
        raise ValueError("post-training policy evaluation requires the fixed return recipe")
    if episode_replay and not (
        (return_context or four_phase_replay is not None) and evaluate_policy_after_training
    ):
        raise ValueError("episode replay requires the fixed return policy recipe")
    # Why: a distinct opt-in preserves existing mixed-recipe refusals unchanged.
    if four_phase_probes:
        fresh_task_probes = True
    if fresh_task_probes and not (
        (return_context or four_phase_probes) and evaluate_policy_after_training
    ):
        raise ValueError("fresh-task probes require the fixed return policy recipe")
    updates = (
        ACQUIRED_RETURN_UPDATES
        if acquired_return
        else FIXED_A_UPDATES
        if fixed_a_exposure
        else FOUR_PHASE_UPDATES
        if four_phase
        else RETURN_UPDATES
        if return_context
        else UPDATES
    )
    points = (
        ACQUIRED_RETURN_POINTS
        if acquired_return
        else FIXED_A_POINTS
        if fixed_a_exposure
        else FOUR_PHASE_POINTS
        if four_phase
        else RETURN_POINTS
        if return_context
        else POINTS
    )
    evaluation_points = (
        (192, 256, 320)
        if acquired_return
        else (0, 128)
        if fixed_a_exposure
        else (0, 64)
        if train_policy_diagnostics
        else FOUR_PHASE_EVALUATION_POINTS
        if four_phase_policy
        else EVALUATION_POINTS
    )
    replay_enabled = episode_replay or acquired_replay is not None
    replay_check_options: dict[str, Any] = (
        {"total_updates": 320}
        if acquired_replay is not None
        else {}
        if four_phase_replay is None
        else {"total_updates": 256}
    )
    started = monotonic()
    source = isolated = stream = budget = sampler = initial = None
    replay_budget = helper = None
    evaluation_states: list[tuple[int, Any]] = []
    acquired_completed_evidence = None
    acquired_replay_claims = None
    result: dict[str, Any] = dict(
        protocol=describe_acquired_replay_protocol(control=acquired_replay)
        if acquired_replay is not None
        else describe_acquired_return_protocol()
        if acquired_return
        else describe_fixed_exposure_protocol()
        if fixed_a_exposure
        else describe_weighted_live_protocol()
        if weighted_train
        else describe_live_protocol()
        if train_policy_diagnostics
        else describe_four_phase_probe_protocol(control=four_phase_replay)
        if four_phase_probes and four_phase_replay is not None
        else describe_four_phase_replay_protocol(control=four_phase_replay)
        if four_phase_replay is not None
        else describe_four_phase_evaluation_protocol()
        if four_phase_policy
        else describe_four_phase_protocol()
        if four_phase
        else describe_return_probe_protocol(episode_replay=episode_replay)
        if fresh_task_probes
        else describe_return_replay_protocol()
        if episode_replay
        else describe_return_evaluation_protocol()
        if evaluate_policy_after_training
        else describe_return_protocol()
        if return_context
        else describe_protocol(),
        status="incomplete",
        reason=None,
        stream_result=None,
        checkpoints=[],
        training_summary=None,
        majority_control=None,
        charged_updates=0,
        source_unchanged=None,
        isolated_source_unchanged=None,
        actor_unchanged=None,
        readback_error=None,
        observed_process_rss=None,
        partial_stream=None,
        partial_readback_error=None,
        scientific_advantage_verified=False,
    )
    if train_policy_diagnostics:
        result["train_policy_diagnostics"] = (
            train_live.new_diagnostic(endpoint=128)
            if fixed_a_exposure
            else train_live.new_diagnostic()
        )
    if return_context:
        result.update(context_return=None, majority_controls=None)
    acquired_report = None
    if acquired_return:
        acquired_report = acquired_tasks.new_diagnostic(
            **({"replay_control": acquired_replay} if acquired_replay is not None else {})
        )
        result.update(
            acquired_return=None,
            acquired_return_diagnostics=acquired_report,
            acquired_initial_condition_met=None,
            acquired_interpretation="not_reached",
        )
    if acquired_replay is not None:
        result.update(
            replay_control=acquired_replay,
            replay=None,
            replay_readback_error=None,
            acquired_new_task_condition_met=None,
        )
    if four_phase:
        result.update(four_phase_return=None, majority_controls=None)
    if evaluate_policy_after_training:
        result["policy_evaluation"] = (
            new_evaluation(four_phase=True) if four_phase_policy else new_evaluation()
        )
    if episode_replay:
        result["replay"] = None
        result["replay_readback_error"] = None
        result["policy_evaluation"]["replay_enabled"] = True
        if four_phase_replay is not None:
            result["replay_control"] = result["policy_evaluation"]["replay_control"] = (
                four_phase_replay
            )
    if fresh_task_probes:
        result["probes"] = probe_slots(checkpoint_updates=(0, updates))
    work = dict(
        score_forwards_attempted=0,
        score_forwards_completed=0,
        score_rows_attempted=0,
        score_rows_completed=0,
    )
    result["work"] = work
    try:
        sampler = ProcessRssSampler()
        with sampler:
            torch = require_torch()
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=updates,
                    max_wall_seconds=max(0.0, started + 90 - monotonic()),
                    max_process_rss_bytes=shared.MAX_RSS,
                ),
                monotonic,
            )
            budget.attach_memory(sampler)
            budget.before_final()
            base, training = (
                _packets(acquired_return=True)
                if acquired_return
                else _packets(fixed_a_exposure=True)
                if fixed_a_exposure
                else _packets(four_phase=True)
                if four_phase
                else _packets(return_context=True)
                if return_context
                else _packets()
            )
            expected_objective = _weighted_objective(base) if weighted_train else None
            if weighted_train:
                result["training_objective"] = deepcopy(expected_objective)

            def check_objective() -> None:
                if acquired_replay is not None and result.get("replay_control") != acquired_replay:
                    raise ValueError("original acquired replay control evidence changed")
                if weighted_train and (
                    result["training_objective"] != expected_objective
                    or _weighted_objective(base) != expected_objective
                ):
                    raise ValueError("original weighted train objective evidence changed")

            probe_data = None
            if fresh_task_probes:
                probe_data = shared.build_probe_stream()
                validate_probe_batches(*probe_data)
                seen = {identity for b in (*base, *training) for identity in b.row_ids}
                for batches, role in zip(probe_data, ("probe_train", "probe_test"), strict=True):
                    for packet in batches:
                        _validate_batch(packet, role, seen)
            replay_sources = (
                {}
                if not replay_enabled
                else {
                    b.episode.episode_id: (i + 1, asdict(b.episode))
                    for i, b in enumerate(training)
                    if b.episode is not None
                }
            )
            result["training_summary"] = dict(
                episodes=len(training),
                unique_source_episodes=len(base),
                collected_source_records=sum(len(b.features) for b in base),
                training_exposures=sum(len(b.features) for b in training),
                roles=[b.role for b in base],
                record_count_scope="collected source records; numeric values can repeat; unique IDs do not establish independent samples",
                memory_scope="encoded bytes count original input arrays only; owned copies/native states/graphs/report allocations included in total time/process RSS",
                source_episodes=[asdict(b.episode) for b in base if b.episode is not None],
                episode_ids=[b.episode.episode_id for b in training if b.episode is not None],
                encoded_training_array_bytes=sum(
                    b.features.nbytes + b.targets.nbytes for b in training
                ),
                encoded_scoring_array_bytes=sum(b.features.nbytes + b.targets.nbytes for b in base),
            )
            result["majority_control"] = _reference(base)
            if return_context:
                result["majority_controls"] = dict(A=_reference(base[:2]), B=_reference(base[2:]))
                result["training_summary"]["phase_schedule"] = describe_return_protocol()[
                    "phase_schedule"
                ]
            if four_phase:
                result["majority_controls"] = {
                    name: _reference(base[2 * i : 2 * i + 2])
                    for i, name in enumerate(("A", "B", "C_train"))
                }
                result["training_summary"]["phase_schedule"] = result["protocol"]["phase_schedule"]
            if weighted_train and any(batch.episode is None for batch in training):
                raise ValueError("weighted schedule requires complete train episodes")
            source = (
                TorchTrainWeightedLearner(
                    TorchRecurrentPolicy(8, 8, 5, 17),
                    tuple(batch.episode for batch in training if batch.episode is not None),
                )
                if weighted_train
                else TorchRecurrentLearner(TorchRecurrentPolicy(8, 8, 5, 17))
            )
            initial = source.snapshot_state()
            isolated = source.fork()
            if isolated is source or not _same_state(initial, isolated.snapshot_state()):
                raise ValueError("train-fit stream source requires distinct identical owned fork")
            if replay_enabled:
                # Both original sessions share this invocation's fixed deadline.
                replay_budget = ToyBudgetSession(
                    ToyExecutionBudget(
                        max_replay_examples=1280
                        if acquired_replay is not None
                        else 1024
                        if four_phase_replay is not None
                        else 768,
                        max_wall_seconds=max(0.0, started + 90 - monotonic()),
                        max_process_rss_bytes=shared.MAX_RSS,
                    ),
                    monotonic,
                )
                replay_budget.attach_memory(sampler)
                helper = StreamRehearsal(
                    isolated,
                    budget,
                    matched_replay.replay_policy(acquired_replay)
                    if acquired_replay is not None
                    else matched_replay.replay_policy(four_phase_replay)
                    if four_phase_replay is not None
                    else shared.replay_policy(),
                    replay_budget=replay_budget,
                    total_updates=updates,
                    input_dim=8,
                    episode_schema=EpisodeReplaySchema(8, 8, 5),
                )

            initial_probe_evidence = None

            def check_initial_probe() -> None:
                if (
                    initial_probe_evidence is not None
                    and result["probes"][0] != initial_probe_evidence
                ):
                    raise ValueError("completed initial probe evidence changed before more work")

            def observe(count: int, state: Any, batch: Any) -> None:
                check_objective()
                if four_phase_probes:
                    check_initial_probe()
                _isolation(source, isolated, initial, stream)
                clocks = (
                    None
                    if helper is None
                    else check_replay_progress(
                        helper,
                        replay_budget,
                        count,
                        state,
                        training_sources=replay_sources,
                        **replay_check_options,
                    )
                )
                point = _score(source, state, base, budget, work, count)
                check_objective()
                if four_phase_probes:
                    check_initial_probe()
                _isolation(source, isolated, initial, stream)
                if helper is not None:
                    check_replay_progress(
                        helper,
                        replay_budget,
                        count,
                        state,
                        training_sources=replay_sources,
                        **replay_check_options,
                    )
                    point["clocks"] = clocks
                reference = result["majority_control"]
                point.update(
                    ce_reduction_vs_prior=reference["mean_ce"] - point["mean_ce"],
                    accuracy_gain_vs_majority=point["mean_accuracy"] - reference["mean_accuracy"],
                    exact_sequence_gain_vs_majority=point["exact_sequence_accuracy"]
                    - reference["exact_sequence_accuracy"],
                )
                result["checkpoints"].append(point)
                if (
                    evaluate_policy_after_training or diagnostics_enabled
                ) and count in evaluation_points:
                    if helper is None and (
                        state.completed_updates != count or state.training_examples != 8 * count
                    ):
                        raise ValueError(
                            "policy checkpoint native progression differs from observed update"
                        )
                    evaluation_states.append((count, deepcopy(state)))
                    if helper is not None and not acquired_return:
                        for episode in result["policy_evaluation"]["episodes"]:
                            if episode["checkpoint_updates"] == count:
                                episode["native_checkpoint_updates"] = state.completed_updates

            rehearsal_options: dict[str, Any] = {} if helper is None else {"rehearsal": helper}
            stream = AttributionStream(
                isolated,
                training,
                (),
                checkpoint_updates=points,
                label_delay=1,
                budget=budget,
                observe_checkpoint=observe,
                training_only=True,
                **rehearsal_options,
            )

            def probe_at(count: int) -> None:
                nonlocal initial_probe_evidence
                if four_phase_probes:
                    check_initial_probe()
                assert probe_data is not None
                state = next(state for n, state in evaluation_states if n == count)
                run_return_checkpoint_probe(
                    result,
                    source,
                    state,
                    probe_data,
                    checkpoint=count,
                    stream=stream,
                    wake=budget,
                    replay=replay_budget,
                    helper=helper,
                    deadline=started + 90,
                    sampler=sampler,
                    same_state=_same_state,
                    isolation=lambda: _isolation(source, isolated, initial, stream),
                    probe_runner=run_fresh_task_probe,
                )
                if four_phase_probes:
                    check_initial_probe()
                    slot = next(p for p in result["probes"] if p["checkpoint_updates"] == count)
                    if (
                        not valid_probe(slot)
                        or slot["report"]["status"] != "complete"
                        or slot["native_updates_before"] != state.completed_updates
                        or slot["native_training_examples_before"] != state.training_examples
                    ):
                        raise ValueError("four-phase probe must complete before more work")
                    if count == 0:
                        initial_probe_evidence = deepcopy(slot)

            if fresh_task_probes:
                probe_at(0)
            if four_phase_probes:
                # Check the one preceding probe record before each wake admission.
                # Existing stream ticks/label delay/optimizer remain continuous.
                for next_wake in range(1, updates + 1):
                    check_initial_probe()
                    stream.run_until(next_wake)
                    check_initial_probe()
            else:
                stream.run_until(updates)
            result["stream_result"] = stream.finish()
            check_objective()
            if fresh_task_probes:
                probe_at(updates)
                if not valid_probes(result):
                    raise ValueError("return recipe requires both complete isolated E16 probes")
            if return_context:
                result["context_return"] = _return_outcomes(result["checkpoints"])
            if four_phase:
                result["four_phase_return"] = _four_phase_outcomes(result["checkpoints"])
            if acquired_return:
                result["acquired_return"] = _acquired_return_outcomes(result["checkpoints"])
            if evaluate_policy_after_training or diagnostics_enabled:
                if diagnostics_enabled and not _same_state(
                    _reference(base), result["majority_control"]
                ):
                    raise ValueError(
                        "train diagnostic prior differs from original empirical train labels"
                    )
                completed = deepcopy(stream._runtime._candidate.snapshot_state())
                receipts = deepcopy(stream._runtime.applied_updates)
                expected_states = deepcopy(
                    [(n, state) for n, state in stream._states if n in evaluation_points]
                )
                expected_replay = None if helper is None else replay_evidence(helper, replay_budget)
                expected_probes = deepcopy(result["probes"]) if fresh_task_probes else None

                def completed_evidence() -> Any:
                    return (
                        result["stream_result"],
                        result["training_summary"],
                        result["majority_control"],
                        result["checkpoints"],
                        *((result["acquired_return"],) if acquired_return else ()),
                    )

                diagnostic_evidence = (
                    deepcopy(completed_evidence()) if diagnostics_enabled else None
                )

                def input_signature() -> Any:
                    return [
                        (
                            b.role,
                            b.context,
                            b.row_ids,
                            b.num_classes,
                            b.features.dtype.str,
                            b.features.shape,
                            b.features.tobytes(),
                            b.targets.dtype.str,
                            b.targets.shape,
                            b.targets.tobytes(),
                            None if b.episode is None else asdict(b.episode),
                        )
                        for b in (*base, *training)
                    ]

                expected_inputs = input_signature() if diagnostics_enabled else None

                def final_guard() -> None:
                    check_objective()
                    if four_phase_probes:
                        check_initial_probe()
                    budget.before_final()
                    stream._check_original()
                    _isolation(source, isolated, initial, stream)
                    if acquired_return and (
                        result["acquired_return_diagnostics"] is not acquired_report
                        or result["acquired_initial_condition_met"] is not None
                        or result["acquired_interpretation"] != "not_reached"
                        or (
                            acquired_replay is not None
                            and result.get("acquired_new_task_condition_met") is not None
                        )
                    ):
                        raise ValueError(
                            "acquired return result binding or pending interpretation changed"
                        )
                    if diagnostics_enabled and (
                        not _same_state(
                            diagnostic_evidence,
                            completed_evidence(),
                        )
                        or not _same_state(expected_inputs, input_signature())
                    ):
                        raise ValueError(
                            "completed train diagnostic inputs or score evidence changed"
                        )
                    if fresh_task_probes and result["probes"] != expected_probes:
                        raise ValueError("completed probe evidence changed before more final work")
                    if helper is not None:
                        check_replay_progress(
                            helper,
                            replay_budget,
                            updates,
                            completed,
                            training_sources=replay_sources,
                            **replay_check_options,
                        )
                        if replay_evidence(helper, replay_budget) != expected_replay:
                            raise ValueError(
                                "completed replay evidence changed before more final work"
                            )
                    if (
                        not _same_state(completed, stream._runtime._candidate.snapshot_state())
                        or not _same_state(receipts, stream._runtime.applied_updates)
                        or not _same_state(expected_states, evaluation_states)
                        or not _same_state(
                            expected_states,
                            [(n, state) for n, state in stream._states if n in evaluation_points],
                        )
                    ):
                        raise ValueError(
                            "completed training evidence changed before more final work"
                        )
                    if acquired_completed_evidence is not None and not _same_state(
                        acquired_completed_evidence, result["acquired_return_diagnostics"]
                    ):
                        raise ValueError("completed acquired diagnostic changed at final boundary")

                if acquired_return:
                    final_guard()
                    if acquired_report is None:
                        raise ValueError(
                            "acquired task return requires its original diagnostic record"
                        )
                    acquired_tasks.evaluate_acquired_tasks(
                        acquired_report,
                        source,
                        evaluation_states,
                        budget,
                        check_boundary=final_guard,
                        **(
                            {"replay_control": acquired_replay}
                            if acquired_replay is not None
                            else {}
                        ),
                    )
                    # Why: the final caller boundary can invoke external clock/state
                    # hooks after the helper has published its derived acquisition.
                    acquired_completed_evidence = deepcopy(acquired_report)
                    final_guard()
                    if acquired_report is None or acquired_report["status"] != "complete":
                        raise ValueError("acquired task return diagnostic did not complete")
                    result["acquired_initial_condition_met"] = (
                        not result["acquired_return"]["incomplete_initial_A_fit"]
                        and acquired_report["initial_A_acquired"]
                    )
                    result["acquired_interpretation"] = (
                        "descriptive_train_retention_relearning"
                        if result["acquired_initial_condition_met"]
                        else "inconclusive_initial_A_not_acquired"
                    )
                    if acquired_replay is not None:
                        raw_c = result["acquired_return"]["endpoint_values"][1]["C_train"]
                        live_c = acquired_report["outcomes"][1]["by_layout"]["C_train"]
                        result["acquired_new_task_condition_met"] = (
                            raw_c["exact_sequence_accuracy"] == 1.0
                            and live_c["success_rate"] == 1.0
                        )
                        acquired_replay_claims = (
                            result["acquired_initial_condition_met"],
                            result["acquired_interpretation"],
                            result["acquired_new_task_condition_met"],
                        )
                elif train_policy_diagnostics:
                    final_guard()
                    train_live.evaluate_train_policy(
                        result["train_policy_diagnostics"],
                        source,
                        evaluation_states,
                        result["majority_control"],
                        budget,
                        check_boundary=final_guard,
                        **({"endpoint": 128} if fixed_a_exposure else {}),
                    )
                    final_guard()
                    if result["train_policy_diagnostics"]["status"] != "complete":
                        raise ValueError("train-only closed-loop diagnostic did not complete")
                else:
                    evaluate_policy(
                        result["policy_evaluation"],
                        source,
                        evaluation_states,
                        budget,
                        training=training,
                        check_boundary=final_guard,
                        **({"four_phase": True} if four_phase_policy else {}),
                    )
                if helper is not None and not acquired_return:
                    clocks_by_wake = {p["updates"]: p["clocks"] for p in result["checkpoints"]}
                    for point in result["policy_evaluation"]["checkpoints"]:
                        point["clocks"] = deepcopy(clocks_by_wake[point["updates"]])
            budget.before_final()
            if replay_budget is not None:
                replay_budget.before_final()
            result["status"] = "complete"
    except Exception as error:
        result.update(status="incomplete", reason=f"{type(error).__name__}: {error!r}")
    finally:
        result["charged_updates"] = 0 if budget is None else budget.updates_completed
        if helper is not None:
            try:
                result["replay"] = replay_evidence(helper, replay_budget)
            except Exception as error:
                result["replay_readback_error"] = (
                    f"replay readback {type(error).__name__}: {error!r}"
                )
                result["status"] = "incomplete"
                if result["reason"] is None:
                    result["reason"] = result["replay_readback_error"]
        if stream is not None:
            try:
                candidate = stream._runtime._candidate.snapshot_state()
                result["partial_stream"] = dict(
                    completed_wake_updates=stream.completed_updates,
                    native_candidate_updates=candidate.completed_updates,
                    native_candidate_training_examples=candidate.training_examples,
                    applied_updates=[asdict(r) for r in stream._runtime.applied_updates],
                )
            except Exception as error:
                result["partial_readback_error"] = (
                    f"partial readback {type(error).__name__}: {error!r}"
                )
                result["status"] = "incomplete"
        try:
            if initial is not None and source is not None:
                result["source_unchanged"] = _same_state(initial, source.snapshot_state())
            if initial is not None and isolated is not None:
                result["isolated_source_unchanged"] = _same_state(
                    initial, isolated.snapshot_state()
                )
            if stream is not None:
                result["actor_unchanged"] = _same_state(
                    initial, stream._runtime.actor.snapshot_state().state
                )
            if sampler is not None:
                result["observed_process_rss"] = asdict(sampler.snapshot())
            if any(
                result[k] is not True
                for k in ("source_unchanged", "isolated_source_unchanged", "actor_unchanged")
            ):
                raise ValueError("train-fit source or serving actor isolation unconfirmed")
        except Exception as error:
            result["readback_error"] = f"readback {type(error).__name__}: {error!r}"
            result["status"] = "incomplete"
    result["total_seconds"] = monotonic() - started
    if acquired_completed_evidence is not None and not _same_state(
        acquired_completed_evidence, result.get("acquired_return_diagnostics")
    ):
        result["status"] = "incomplete"
        if result["readback_error"] is None:
            result["readback_error"] = "completed acquired diagnostic changed during readback"
    if acquired_replay is not None and (
        result.get("replay_control") != acquired_replay
        or (
            acquired_replay_claims is not None
            and not _same_state(
                acquired_replay_claims,
                (
                    result.get("acquired_initial_condition_met"),
                    result.get("acquired_interpretation"),
                    result.get("acquired_new_task_condition_met"),
                ),
            )
        )
    ):
        result["status"] = "incomplete"
        if result["readback_error"] is None:
            result["readback_error"] = (
                "acquired replay control or acquisition claims changed during readback"
            )
    if result["status"] == "complete" and (
        result["charged_updates"] != updates
        or [p["updates"] for p in result["checkpoints"]] != list(points)
        or result["total_seconds"] > 90
        or result["observed_process_rss"]["peak_bytes"] > shared.MAX_RSS
    ):
        result.update(status="incomplete", reason="train-fit work/curve/resource gate incomplete")
    return result
