"""Disposable, capped fitting on the original six TRAIN sources.

Uses the existing learner and delayed training-only stream. Returns raw TRAIN
probabilities, original metrics and paid-work facts, including on failure. No
benchmark/deployed learner, probe/final data, optimizer selection or IO enters.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, replace
from math import isfinite
from time import monotonic
from typing import Any

import numpy as np

from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
    copy_invariant_torch_feature_snapshot,
)
from src.adapters.cue_relative_torch_feature_learner import (
    CueRelativeTorchFeatureLearner,
    CueRelativeTorchFeaturePolicy,
    CueRelativeTorchFeatureSnapshot,
)
from src.core.minigrid_cue_relative_encoding import CUE_RELATIVE_HISTORY_ENCODING_ID
from src.adapters.minigrid_probe_support import summarize_train_fit_bounds, fit_gap_outcomes
from src.adapters.minigrid_train_catalogue import compose_catalogue_training
from src.adapters import minigrid_train_rollout as train_rollout
from src.adapters.minigrid_train_rollout import run_train_rollout
from src.adapters.torch_feature_learner import TORCH_CE_ID, TorchFeaturePolicy
from src.core.resnet50_variants import TORCH_PC_ENERGY_ID
from src.adapters.torch_feature_gradient_diagnostics import diagnose_prior_gradients, ROW_DEFINITION
from src.adapters.torch_recurrent_gradient_diagnostics import GradientDiagnosticFailed
from src.adapters.torch_feature_rehearsal_pilot import _same_state as _same_torch_state
from src.adapters.invariant_perceptron_learner import (
    InvariantPerceptronLearner,
    InvariantPerceptronSnapshot,
    copy_invariant_perceptron_snapshot,
)
from src.adapters.invariant_kernel_perceptron_learner import (
    InvariantKernelPerceptronLearner,
    InvariantKernelPerceptronSnapshot,
    copy_invariant_kernel_perceptron_snapshot,
)
from src.core.learner_ports import TrainingUpdateFailed
from src.app.coupled_attribution import AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_probability_scores import probability_scores
from src.core.demonstration_episode import DemonstrationEpisode, episode_arrays
from src.core.demonstration_schedule_witness import DemonstrationScheduleWitness
from src.shared.process_memory import ProcessRssSampler

DEFINITION = "minigrid_constructive_h8_bp_train_fit_v1"
CUE_RELATIVE_DEFINITION = "minigrid_cue_relative_constructive_h8_bp_train_fit_v1"
PERCEPTRON_DEFINITION = "minigrid_invariant_perceptron_train_fit_v1"
KERNEL_PERCEPTRON_DEFINITION = "minigrid_invariant_polynomial_kernel_perceptron_train_fit_v1"
WIDE_NEURAL_DEFINITION = "minigrid_invariant_h32_train_fit_v1"
WIDE_METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
RSS_LIMIT = 768 * 1024**2


def _same_state(left: Any, right: Any) -> bool:
    if (
        type(left) is InvariantKernelPerceptronSnapshot
        or type(right) is InvariantKernelPerceptronSnapshot
    ):
        if type(left) is not type(right):
            return False
        return copy_invariant_kernel_perceptron_snapshot(
            left
        ) == copy_invariant_kernel_perceptron_snapshot(right)
    if type(left) is InvariantPerceptronSnapshot or type(right) is InvariantPerceptronSnapshot:
        if type(left) is not type(right):
            return False
        # This state contains immutable numeric records; no optional Torch import
        # is needed to validate or compare the separate non-neural control.
        return copy_invariant_perceptron_snapshot(left) == copy_invariant_perceptron_snapshot(right)
    return _same_torch_state(left, right)


def _points(updates: int, seconds: float) -> tuple[int, ...]:
    if type(updates) is not int or not 1 <= updates <= 512:
        raise ValueError("constructive fit updates must be a native integer in1..512")
    if type(seconds) not in (int, float) or not isfinite(seconds) or not 0 < seconds <= 90:
        raise ValueError("constructive fit wall seconds must be finite in(0,90]")
    return tuple(sorted({0, min(128, updates), updates}))


def _wide_policy(method: str) -> InvariantTorchFeaturePolicy:
    if type(method) is not str or method not in WIDE_METHODS:
        raise ValueError("wide_method requires bp, pc, neutral_cpc or gated_cpc")
    # Why: expose one already supported capacity, with no width/optimizer sweep.
    return InvariantTorchFeaturePolicy(TorchFeaturePolicy(method, 296, 32, 17, num_classes=7))


def _learner(
    *,
    cue_relative: bool = False,
    perceptron: bool = False,
    kernel_perceptron: bool = False,
    wide_method: str | None = None,
) -> (
    InvariantTorchFeatureLearner
    | CueRelativeTorchFeatureLearner
    | InvariantPerceptronLearner
    | InvariantKernelPerceptronLearner
):
    if wide_method is not None:
        return InvariantTorchFeatureLearner(_wide_policy(wide_method))
    if kernel_perceptron:
        return InvariantKernelPerceptronLearner()
    if perceptron:
        return InvariantPerceptronLearner()
    if cue_relative:
        return CueRelativeTorchFeatureLearner(
            CueRelativeTorchFeaturePolicy(TorchFeaturePolicy("bp", 296, 8, 17, num_classes=7))
        )
    return InvariantTorchFeatureLearner(
        InvariantTorchFeaturePolicy(TorchFeaturePolicy("bp", 296, 8, 17, num_classes=7))
    )


def describe_protocol(
    *,
    updates: int = 512,
    max_wall_seconds: float = 90,
    cue_turn_gradients: bool = False,
    cue_relative: bool = False,
    perceptron: bool = False,
    kernel_perceptron: bool = False,
    wide_method: str | None = None,
    train_a_rollout: bool = False,
) -> dict[str, Any]:
    if type(train_a_rollout) is not bool or (
        train_a_rollout and (wide_method is None or type(updates) is not int or updates != 128)
    ):
        raise ValueError(
            "train_a_rollout requires an exact bool and explicit wide_method/128updates"
        )
    if wide_method is not None:
        _wide_policy(wide_method)
        if any((cue_turn_gradients, cue_relative, perceptron, kernel_perceptron)):
            raise ValueError(
                "wide_method requires its original encoding without other fit families/gradients"
            )
    if type(kernel_perceptron) is not bool or (
        kernel_perceptron and (perceptron or cue_relative or cue_turn_gradients)
    ):
        raise ValueError(
            "kernel_perceptron requires an exact bool without other fit families/gradients"
        )
    if type(perceptron) is not bool or (perceptron and (cue_relative or cue_turn_gradients)):
        raise ValueError("perceptron must be an exact bool without alternate encoding/gradients")
    if type(cue_turn_gradients) is not bool:
        raise ValueError("cue_turn_gradients must be an exact bool")
    if type(cue_relative) is not bool:
        raise ValueError("cue_relative must be an exact bool")
    if cue_relative and cue_turn_gradients:
        # Why: the accepted gradient reader binds the original projection.
        raise ValueError("cue-relative fit requires a projection-aware gradient reader")
    points = _points(updates, max_wall_seconds)
    protocol = dict(
        definition=DEFINITION,
        classification="disposable_TRAIN_optimization_witness_not_replacement_baseline",
        method="public_invariant_bp",
        feature_dim=296,
        hidden_dim=8,
        num_classes=7,
        seed=17,
        learning_rate=0.03,
        momentum=0.9,
        objective="original unweighted native BP CE",
        updates=updates,
        checkpoint_updates=list(points),
        label_delay=1,
        training_only=True,
        heldout=[],
        replay_rows=0,
        probe_updates=0,
        final_episodes=0,
        max_wall_seconds=max_wall_seconds,
        max_process_rss_bytes=RSS_LIMIT,
        training_rows=54 * updates,
        score_forward_calls=6 * len(points),
        score_rows=54 * len(points),
        stream_input_array_bytes=54 * 297 * 8 * updates,
        template_array_bytes=54 * 297 * 8,
        score_input_array_bytes=54 * 297 * 8,
        prediction_array_bytes=54 * 7 * 8 * len(points),
        storage="one shared input template; existing stream owns one array copy per wake; at most3 full checkpoint states, bounded receipts/probability records; Python/native/copies/RSS additional",
        scoring="after complete training only, all original six TRAIN episodes at every fixed checkpoint; no best-checkpoint or early-success selection",
        acceptance="all six final TRAIN traces exact after all requested updates; constructive fit only; failure cannot prove finite H8 incapacity",
        comparison="larger optimization budget than original128 BP45/54; not matched exposure/compute or a baseline replacement",
    )
    if cue_relative:
        protocol.update(
            definition=CUE_RELATIVE_DEFINITION,
            method="public_cue_relative_bp",
            encoding_id=CUE_RELATIVE_HISTORY_ENCODING_ID,
            input_contract="raw public296 through catalogue/stream; bound learner projects once for train and prediction; never pass explicit codec output",
            comparison="representation control versus original unweighted BP at equal requested updates; not evidence of circadian benefit, retention or generalization",
        )
    if perceptron or kernel_perceptron:
        for key in ("hidden_dim", "seed", "learning_rate", "momentum"):
            protocol.pop(key)
        protocol.update(
            definition=PERCEPTRON_DEFINITION,
            method="public_invariant_perceptron",
            objective="supplied-order online multiclass mistake rule; not CE optimization",
            initialization="all coefficients zero; no model RNG",
            step_size=1.0,
            coefficients=297 * 7,
            coefficient_array_equivalent_bytes=297 * 7 * 8,
            training_decision_rows=54 * updates,
            maximum_mistake_updates=54 * updates,
            native_update_definition="one complete54-row pass; row mistakes additionally counted",
            prediction="stable softmax of linear scores; first argmax ties",
            comparison="separate linear classifier control; original least-squares/BP/PC recipes unchanged; no matched computation or neural capacity claim",
            acceptance="all six final TRAIN traces exact after all requested passes; a negative finite run does not prove linear inseparability",
            storage="immutable coefficient table/counters; at most3 full checkpoints; existing stream owns one input copy per pass; Python/copies/RSS additional",
        )
    if kernel_perceptron:
        protocol.update(
            definition=KERNEL_PERCEPTRON_DEFINITION,
            method="public_invariant_polynomial_kernel_perceptron",
            objective="supplied-order unit multiclass mistake rule in a fixed degree2 kernel; not CE optimization",
            initialization="empty prototype dictionary/zero scores; no model RNG",
            prediction="stable softmax of fixed quadratic-kernel scores; first argmax ties",
            acceptance="all six final TRAIN traces exact after all requested passes; a negative finite run does not prove kernel inseparability",
            kernel="(1 + dot(x,y)/296) ** 2",
            maximum_prototypes=128,
            maximum_dictionary_array_equivalent_bytes=128 * (296 + 7) * 8,
            maximum_training_kernel_pairs=54 * updates * 128,
            maximum_prediction_kernel_pairs=54 * len(points) * 128,
            capacity="retain exact mistaken keys, update hits; new mistaken key at128 refuses entire pass atomically; no eviction or removal of canceled entries",
            storage="bounded immutable prototype/weight dictionary and counters; at most3 complete checkpoints; existing stream input copies plus validation/temporary arrays/Python/copies/RSS additional",
            comparison="separate nonlinear capacity control; original linear/least-squares/neural recipes unchanged; not matched computation or a circadian advantage claim",
        )
        for key in ("coefficients", "coefficient_array_equivalent_bytes"):
            protocol.pop(key)
    if wide_method is not None:
        protocol.update(
            definition=WIDE_NEURAL_DEFINITION,
            wide_neural_method=wide_method,
            method=f"public_invariant_{wide_method}",
            hidden_dim=32,
            inference_steps=2,
            inference_learning_rate=0.04,
            objective="original unweighted native BP CE"
            if wide_method == "bp"
            else "original unweighted categorical PC relaxation/parameter rule; returned legacy energy diagnostic is separate from prior CE; neutral/active gates retain original configuration",
            native_diagnostic_definition=TORCH_CE_ID if wide_method == "bp" else TORCH_PC_ENERGY_ID,
            parameter_elements=9735,
            parameter_array_equivalent_bytes=38940,
            nominal_latent_row_steps=0 if wide_method == "bp" else 54 * updates * 2,
            work_scope="shared four trainable float32 parameter arrays; nominal solver steps derive from fixed policy; native adaptive/optimizer/gradient/RNG state, projection/validation/copies/Python/RSS/time additional",
            acceptance="all six final TRAIN traces exact after all prescribed updates; a finite negative run does not prove width32 incapable",
            comparison="separate fixed width32 neural capacity recipe; same seed/data/exposure across four methods, native computation differs; original width8 recipes unchanged; no learning benefit or matched-compute claim",
        )
    if cue_turn_gradients:
        protocol["cue_turn_gradients"] = dict(
            role="train",
            context="A",
            row_indices=[2],
            checkpoint_updates=list(points),
            source_episode_ids=["minigrid-source-A-blue", "minigrid-source-A-red"],
            pair_calls=len(points),
            forward_calls=2 * len(points),
            forward_rows=2 * len(points),
            backward_calls=2 * len(points),
            backward_rows=2 * len(points),
            admission_projection_rows=36 * len(points),
            projected_selected_rows=2 * len(points),
            optimizer_steps=0,
            published_updates=0,
            interpretation="original A-step2 prior CE gradient geometry; descriptive, not native PC energy or causal interference",
        )
    if train_a_rollout:
        protocol["train_a_rollout"] = dict(
            checkpoint_updates=128,
            checkpoint_training_rows=6912,
            context="A",
            cues=["blue", "red"],
            role="train",
            learner_live_episodes=2,
            maximum_live_prediction_forwards=32,
            maximum_live_environment_steps=32,
            max_wall_seconds=train_rollout.LIVE_SECONDS,
            order="after all fixed TRAIN readouts on genuine owned endpoint; original parent clock/charges remain active",
            policy="frozen raw seven-action first argmax, no mask/repair/teacher/target in actions",
            budget="separate original zero-update/replay readout budget within original parent time/RSS limit",
            scope="original known TRAIN A games only; no probe/final/training/selection/generalization or CPC benefit",
        )
    return protocol


def _cue_turn_readouts(
    states: list[tuple[int, Any]],
    episodes: tuple[DemonstrationEpisode, ...],
    report: dict[str, Any],
    guard: Any,
) -> None:
    for point, state in states:
        guard()
        started = monotonic()
        guard()
        owned = deepcopy(state)
        saved = deepcopy(owned)
        record: dict[str, Any] = dict(updates=point, diagnostic=None, elapsed_seconds=None)
        report["cue_turn_gradients"].append(record)
        try:
            diagnostic = diagnose_prior_gradients(owned, episodes[:2], row_indices=(2,))
        except GradientDiagnosticFailed as error:
            record["diagnostic"] = deepcopy(error.report)
            record["elapsed_seconds"] = monotonic() - started
            raise
        record["diagnostic"] = deepcopy(diagnostic)
        record["elapsed_seconds"] = monotonic() - started
        guard()
        if not _same_state(saved, owned):
            raise ValueError("cue-turn gradient reader changed its detached checkpoint")
        if record["diagnostic"]["status"] != "complete":
            raise ValueError("cue-turn gradient readout did not complete")
        rows = record["diagnostic"].get("row_indices")
        if (
            record["diagnostic"].get("definition") != ROW_DEFINITION
            or type(rows) is not list
            or len(rows) != 1
            or type(rows[0]) is not int
            or rows[0] != 2
        ):
            raise ValueError("cue-turn gradient readout must identify the original selected row2")


def _counter(
    state: Any,
    updates: int,
    *,
    cue_relative: bool = False,
    perceptron: bool = False,
    kernel_perceptron: bool = False,
    wide_method: str | None = None,
) -> None:
    if wide_method is not None:
        wide = copy_invariant_torch_feature_snapshot(state)
        if (
            type(updates) is not int
            or wide.policy != _wide_policy(wide_method)
            or wide.completed_updates != updates
            or wide.training_examples != 54 * updates
        ):
            raise ValueError("wide checkpoint differs from its fixed policy or paid54-row clock")
        return
    if kernel_perceptron:
        owned_kernel = copy_invariant_kernel_perceptron_snapshot(state)
        if (
            type(updates) is not int
            or owned_kernel.completed_updates != updates
            or owned_kernel.training_examples != 54 * updates
        ):
            raise ValueError("kernel checkpoint differs from the paid54-row pass clock")
        return
    if perceptron:
        owned = copy_invariant_perceptron_snapshot(state)
        if owned.completed_updates != updates or owned.training_examples != 54 * updates:
            raise ValueError("perceptron checkpoint differs from the paid54-row pass clock")
        return
    native = TorchFeaturePolicy("bp", 296, 8, 17, num_classes=7)
    expected = (
        CueRelativeTorchFeaturePolicy(native)
        if cue_relative
        else InvariantTorchFeaturePolicy(native)
    )
    if (
        (cue_relative and type(state) is not CueRelativeTorchFeatureSnapshot)
        or state.policy != expected
        or state.completed_updates != updates
        or state.training_examples != 54 * updates
    ):
        raise ValueError("constructive checkpoint policy or paid native counters changed")


def _score_point(
    source: Any,
    state: Any,
    point: int,
    episodes: tuple[DemonstrationEpisode, ...],
    report: dict[str, Any],
    guard: Any,
) -> None:
    """Keep returned raw predictions before any later metric/guard can refuse."""
    guard()
    saved = deepcopy(state)
    reader = source.fork()
    if (
        type(reader) is not type(source)
        or reader is source
        or (hasattr(source, "_native") and reader._native is source._native)
    ):
        raise ValueError("constructive scoring requires a separately owned reader")
    reader.restore_state(state)
    guard()
    _counter(
        reader.snapshot_state(),
        point,
        cue_relative=report["protocol"].get("encoding_id") == CUE_RELATIVE_HISTORY_ENCODING_ID,
        perceptron=report["definition"] == PERCEPTRON_DEFINITION,
        kernel_perceptron=report["definition"] == KERNEL_PERCEPTRON_DEFINITION,
        wide_method=report["protocol"].get("wide_neural_method"),
    )
    before = deepcopy(reader.snapshot_state())
    record: dict[str, Any] = dict(updates=point, status="running", scores=[])
    report["readouts"].append(record)
    kernel_prototypes = None
    if report["definition"] == KERNEL_PERCEPTRON_DEFINITION:
        kernel_prototypes = len(state.native_state.prototypes)
        record["kernel_prototype_count"] = kernel_prototypes
    for episode in episodes:
        guard()
        if not _same_state(before, reader.snapshot_state()) or not _same_state(saved, state):
            raise ValueError("detached TRAIN reader state changed")
        features, targets = episode_arrays(episode)
        inputs, labels = features.copy(), targets.copy()
        work = report["work"]
        work["score_forward_calls_attempted"] += 1
        work["score_rows_attempted"] += len(features)
        work["score_work_may_be_unobserved"] = True
        if kernel_prototypes is not None:
            work["kernel_prediction_pairs_derived_attempted"] += kernel_prototypes * len(features)
        predictions = reader.predict(features)
        work["score_forward_calls_completed"] += 1
        work["score_rows_completed"] += len(features)
        work["score_work_may_be_unobserved"] = False
        if kernel_prototypes is not None:
            work["kernel_prediction_pairs_derived_returned"] += kernel_prototypes * len(features)
        raw = dict(
            episode_id=episode.episode_id,
            context=episode.context,
            probabilities=predictions.tolist(),
            target_actions=targets[:, 0].tolist(),
        )
        record["scores"].append(raw)
        guard()
        if (
            not _same_state(before, reader.snapshot_state())
            or not _same_state(saved, state)
            or not np.array_equal(inputs, features)
            or not np.array_equal(labels, targets)
        ):
            raise ValueError("detached prediction changed its state or TRAIN inputs")
        scores = probability_scores(predictions, targets, 7)
        raw.update(
            **scores,
            predicted_actions=np.argmax(predictions, axis=1).tolist(),
            exact_sequence_accuracy=float(scores["accuracy"] == 1.0),
        )
    record["status"] = "complete"


def diagnose_train_fit(
    episodes: tuple[DemonstrationEpisode, ...],
    *,
    updates: int = 512,
    max_wall_seconds: float = 90,
    cue_turn_gradients: bool = False,
    cue_relative: bool = False,
    perceptron: bool = False,
    kernel_perceptron: bool = False,
    wide_method: str | None = None,
    train_a_rollout: bool = False,
) -> dict[str, Any]:
    protocol = describe_protocol(
        updates=updates,
        max_wall_seconds=max_wall_seconds,
        cue_turn_gradients=cue_turn_gradients,
        cue_relative=cue_relative,
        perceptron=perceptron,
        kernel_perceptron=kernel_perceptron,
        wide_method=wide_method,
        train_a_rollout=train_a_rollout,
    )
    # Admit the full fixed TRAIN schema before constructing models or resources.
    bounds = (
        summarize_train_fit_bounds(episodes, cue_relative=True)
        if cue_relative
        else summarize_train_fit_bounds(episodes)
    )
    witness = DemonstrationScheduleWitness(episodes)
    started = monotonic()
    report: dict[str, Any] = dict(
        definition=protocol["definition"],
        role="train",
        status="incomplete",
        reason=None,
        protocol=protocol,
        train_fit_bounds=bounds,
        readouts=[],
        stream_result=None,
        fit_witness=False,
        source_unchanged=None,
        actor_unchanged=None,
        charged_updates=0,
        committed_updates=0,
        native_completed_updates=None,
        native_training_rows=None,
        applied_updates=[],
        work=dict(
            score_forward_calls_attempted=0,
            score_forward_calls_completed=0,
            score_rows_attempted=0,
            score_rows_completed=0,
            score_work_may_be_unobserved=False,
            training_work_may_be_unobserved=False,
        ),
    )
    source = stream = budget = initial = actor = native_before_scoring = None
    live_binding: Callable[[], None] | None = None
    if train_a_rollout:
        report["train_a_rollout"] = train_rollout.empty_record()
    if kernel_perceptron:
        report["work"].update(
            kernel_prediction_pairs_derived_attempted=0, kernel_prediction_pairs_derived_returned=0
        )
    if cue_turn_gradients:
        report["cue_turn_gradients"] = []
    states: list[tuple[int, Any]] = []
    phase = "construction"
    publication = None
    limits = ToyExecutionBudget(
        max_training_updates=updates,
        max_wall_seconds=max_wall_seconds,
        max_process_rss_bytes=RSS_LIMIT,
    )
    try:
        with ProcessRssSampler() as sampler:
            # Account for construction since admission; never renew time for scoring.
            budget = ToyBudgetSession(
                replace(
                    limits, max_wall_seconds=max(0.0, max_wall_seconds - (monotonic() - started))
                ),
                monotonic,
            )
            limits = budget.budget
            budget.attach_memory(sampler)
            born_clock, born_start, born_limits = budget.clock, budget.started_at, deepcopy(limits)
            report["budget"] = dict(limits=asdict(born_limits), started_at=born_start)

            def guard() -> None:
                assert budget is not None
                budget.before_final()
                if live_binding is not None:
                    live_binding()
                if (
                    budget.budget is not limits
                    or limits != born_limits
                    or budget.clock is not born_clock
                    or budget.started_at != born_start
                    or budget.process_rss_sampler is not sampler
                    or budget.replay_examples_completed != 0
                ):
                    raise ValueError("original constructive budget binding changed")
                witness.check(episodes)
                if source is not None and initial is not None:
                    if not _same_state(initial, source.snapshot_state()):
                        raise ValueError("original constructive source changed")
                if actor is not None and not _same_state(initial, actor.snapshot_state().state):
                    raise ValueError("original constructive actor changed")
                if native_before_scoring is not None and not _same_state(
                    native_before_scoring, stream._runtime._candidate.snapshot_state()
                ):
                    raise ValueError("completed constructive candidate changed during scoring")
                # Snapshot ports run before the closing pure source/budget proof.
                if (
                    source is not None
                    and initial is not None
                    and not _same_state(initial, source.snapshot_state())
                ):
                    raise ValueError("original constructive source changed after snapshot reads")
                if stream is not None:
                    stream._check_original()
                witness.check(episodes)
                if (
                    budget.budget is not limits
                    or limits != born_limits
                    or budget.clock is not born_clock
                    or budget.started_at != born_start
                    or budget.process_rss_sampler is not sampler
                    or budget.replay_examples_completed != 0
                ):
                    raise ValueError("original constructive budget changed after snapshot reads")
                if live_binding is not None:
                    live_binding()

            guard()
            plan = (
                compose_catalogue_training(episodes, 1, cue_relative=True)
                if cue_relative
                else compose_catalogue_training(episodes, 1)
            )
            template = plan.training[0]
            training = tuple(
                replace(
                    template,
                    row_ids=tuple(f"fit:{wake}:{identity}" for identity in template.row_ids),
                )
                for wake in range(updates)
            )
            guard()
            source = (
                _learner(wide_method=wide_method)
                if wide_method is not None
                else _learner(kernel_perceptron=True)
                if kernel_perceptron
                else _learner(perceptron=True)
                if perceptron
                else _learner(cue_relative=True)
                if cue_relative
                else _learner()
            )
            initial = deepcopy(source.snapshot_state())
            _counter(
                initial,
                0,
                cue_relative=cue_relative,
                perceptron=perceptron,
                kernel_perceptron=kernel_perceptron,
                wide_method=wide_method,
            )
            guard()

            def observe(point: int, state: Any, batch: Any) -> None:
                guard()
                _counter(
                    state,
                    point,
                    cue_relative=cue_relative,
                    perceptron=perceptron,
                    kernel_perceptron=kernel_perceptron,
                    wide_method=wide_method,
                )
                if point != protocol["checkpoint_updates"][len(states)]:
                    raise ValueError("constructive checkpoint order changed")
                states.append((point, deepcopy(state)))

            stream = AttributionStream(
                source,
                training,
                (),
                checkpoint_updates=tuple(protocol["checkpoint_updates"]),
                label_delay=1,
                budget=budget,
                observe_checkpoint=observe,
                training_only=True,
            )
            actor = stream._runtime.actor
            guard()
            phase = "training"
            for point in range(1, updates + 1):
                guard()
                stream.run_until(point)
                guard()
            report["stream_result"] = stream.finish()
            native_before_scoring = deepcopy(stream._runtime._candidate.snapshot_state())
            _counter(
                native_before_scoring,
                updates,
                cue_relative=cue_relative,
                perceptron=perceptron,
                kernel_perceptron=kernel_perceptron,
                wide_method=wide_method,
            )
            result = report["stream_result"]
            if (
                result["evaluation_scope"] != "training_only"
                or result["score_forward_calls"] != 0
                or result["completed_updates"] != updates
                or budget.updates_completed != updates
                or result["training_examples"] != 54 * updates
                or len(result["applied_updates"]) != updates
                or [p for p, _ in states] != protocol["checkpoint_updates"]
            ):
                raise ValueError("constructive training completion or permanent charges differ")
            guard()
            phase = "scoring"
            for point, state in states:
                _score_point(source, state, point, episodes, report, guard)
            guard()
            if cue_turn_gradients:
                _cue_turn_readouts(states, episodes, report, guard)
                guard()
            if train_a_rollout:
                phase = "live"
                report["work"]["live_work_may_be_unobserved"] = True
                live_record, live_binding = run_train_rollout(
                    dict(source=source, states=states),
                    check_boundary=guard,
                    wide_method=wide_method,
                )
                # Why: preserve returned paid game work before later guards refuse.
                report["train_a_rollout"] = live_record
                live_work = dict(
                    total_score_forwards_completed=report["work"]["score_forward_calls_completed"],
                    total_score_rows_completed=report["work"]["score_rows_completed"],
                    source_environment_steps_returned=0,
                )
                train_rollout.add_work(live_work, live_record)
                report["work"].update(live_work)
                report["work"]["live_work_may_be_unobserved"] = live_work[
                    "failure_work_may_be_unobserved"
                ]
                guard()
                if live_record["status"] != "complete":
                    raise ValueError(
                        "wide TRAIN A rollout incomplete: " + str(live_record["reason"])
                    )
        elapsed = monotonic() - started
        guard()
        outcomes = [
            dict(
                updates=record["updates"],
                **{
                    context: {
                        metric: sum(
                            score[metric]
                            for score in record["scores"]
                            if score["context"] == context
                        )
                        / 2
                        for metric in ("ce", "accuracy", "exact_sequence_accuracy")
                    }
                    for context in ("A", "B", "C_train")
                },
            )
            for record in report["readouts"]
        ]
        publication = dict(
            status="complete",
            outcomes=outcomes,
            fit_gap_outcomes=fit_gap_outcomes(bounds, outcomes),
            fit_witness=all(
                score["exact_sequence_accuracy"] == 1.0
                for score in report["readouts"][-1]["scores"]
            ),
            source_unchanged=True,
            actor_unchanged=True,
            elapsed_seconds=elapsed,
            peak_process_rss_bytes=sampler.peak_bytes,
        )
    except Exception as error:
        report.update(status="incomplete", reason=f"{type(error).__name__}: {error}")
        if kernel_perceptron and isinstance(error, TrainingUpdateFailed):
            report["native_failure_definition"] = error.definition
            report["native_failure_work"] = error.work
        report["work"]["training_work_may_be_unobserved"] = phase == "training"
    finally:
        if budget is not None:
            report["charged_updates"] = budget.updates_completed
        if stream is not None:
            try:
                report["committed_updates"] = stream.completed_updates
                report["applied_updates"] = [
                    asdict(value) for value in stream._runtime.applied_updates
                ]
                native = stream._runtime._candidate.snapshot_state()
                report["native_completed_updates"] = native.completed_updates
                report["native_training_rows"] = native.training_examples
                if wide_method is not None:
                    report["work"]["wide_neural"] = dict(
                        method=wide_method,
                        hidden_dim=32,
                        completed_native_calls=native.completed_updates,
                        completed_training_rows=native.training_examples,
                        parameter_elements=9735,
                        parameter_array_equivalent_bytes=38940,
                        bp_backward_calls_derived=native.completed_updates
                        if wide_method == "bp"
                        else 0,
                        bp_optimizer_steps_derived=native.completed_updates
                        if wide_method == "bp"
                        else 0,
                        nominal_latent_row_steps=0
                        if wide_method == "bp"
                        else native.training_examples * 2,
                        pc_parameter_updates_derived=0
                        if wide_method == "bp"
                        else native.completed_updates,
                        scope="observed native completed prefix; BP backward/optimizer and PC parameter/latent work derived from unchanged fixed algorithms, not an external operation meter; failed work may be unobserved; complete adaptive/optimizer/gradient/RNG state, projection/validation/copies/Python/RSS/time additional",
                    )
                if kernel_perceptron:
                    state = native.native_state
                    prototype_count = len(state.prototypes)
                    report["work"]["kernel_perceptron"] = dict(
                        completed_pass_calls=native.completed_updates,
                        completed_training_rows=native.training_examples,
                        completed_mistake_row_updates=state.mistake_updates,
                        completed_training_kernel_evaluations=state.training_kernel_evaluations,
                        prototype_count=prototype_count,
                        prototype_array_equivalent_bytes=prototype_count * 296 * 8,
                        coefficient_array_equivalent_bytes=prototype_count * 7 * 8,
                        maximum_prototypes=128,
                        optimizer_steps=0,
                        backward_calls=0,
                        scope="native committed kernel-pair counter; prediction pairs derived from owned checkpoint prototype count and attempted/returned rows; admission/validation/copies/RSS/time additional; failed uncommitted work retained separately when available",
                    )
                if perceptron:
                    report["work"]["perceptron"] = dict(
                        completed_pass_calls=native.completed_updates,
                        completed_training_rows=native.training_examples,
                        completed_online_score_rows=native.training_examples,
                        completed_mistake_row_updates=native.native_state.mistake_updates,
                        coefficient_array_equivalent_bytes=297 * 7 * 8,
                        optimizer_steps=0,
                        backward_calls=0,
                        scope="native completed prefix; each row performs a7-class linear decision and each mistake updates two297-coordinate columns; successful prediction rows counted separately; projection/validation/copies/RSS/time additional; failed work may be unobserved",
                    )
            except Exception as error:
                report["native_readback_error"] = f"{type(error).__name__}: {error}"
                report["work"]["training_work_may_be_unobserved"] = True
                publication = None
                report.update(
                    status="incomplete", reason=report["reason"] or report["native_readback_error"]
                )
    if publication is not None:
        try:
            # Readback and the closing clock precede the final source/state checks.
            guard()
            report.update(publication)
        except Exception as error:
            report.update(status="incomplete", reason=f"{type(error).__name__}: {error}")
    return report
