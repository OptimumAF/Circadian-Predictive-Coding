"""Fixed MiniGrid acquisition/return experiment on the existing delayed stream.

The descriptor is resource free. The opt-in pilot owns six TRAIN teacher traces,
five bounded learners and fixed post-training readouts. It performs no selection,
promotion, replay, reward learning or file IO; final D never supplies labels.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, replace
from itertools import chain
from operator import attrgetter, is_
from math import isfinite
import pickle
from time import monotonic
from typing import Any

import numpy as np

from src.adapters.minigrid_policy_evaluation import (
    MiniGridEvaluationFailed,
    evaluate_checkpoint,
)
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy, _config
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_recurrent_learner import TorchRecurrentLearner, TorchRecurrentPolicy
from src.adapters import torch_recurrent_episode_batch as paired
from src.adapters import minigrid_relaxation_control as relaxation
from src.adapters import minigrid_cue_relative_route as cue_relative
from src.adapters.invariant_action_memory_learner import copy_invariant_memory_snapshot
from src.adapters.invariant_torch_feature_learner import copy_invariant_torch_feature_snapshot
from src.app.coupled_attribution import (
    AttributionBatch,
    AttributionStream,
    _validate_batch,
    _same_value,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import DemonstrationEpisode, episode_arrays
from src.core.action_memory import ActionMemoryPolicy, copy_action_memory_snapshot
from src.core.minigrid_key_door import CueKeyDoorTeacher, predefined_layouts
from src.infra.minigrid_key_door import MiniGridKeyDoor
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch

METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc", "recurrent_bc")
POINTS = (0, 32, 64, 96, 128)
CONTEXTS = ("A", "B", "C_train", "A")
RSS_LIMIT = 768 * 1024**2
BODY_SECONDS = 180.0
_INVARIANT_COUNT_METHODS = (
    "public_invariant_observation_memory",
    "public_invariant_observation_memory_fifo",
)
_COUNT_METHODS = ("observation_memory", *_INVARIANT_COUNT_METHODS)
_NATIVE_INVARIANT_FEATURE_METHODS = tuple(
    f"public_invariant_{method}" for method in ("bp", "pc", "neutral_cpc", "gated_cpc")
)
_INVARIANT_FEATURE_METHODS = (
    _NATIVE_INVARIANT_FEATURE_METHODS + relaxation.BOUND_METHODS + cue_relative.METHODS
)
_RECEIPT_METHODS = (*_COUNT_METHODS, *_INVARIANT_FEATURE_METHODS)


def describe_protocol() -> dict[str, Any]:
    """Declare all methods, exposures, comparisons and caps before execution."""
    return dict(
        protocol_id="minigrid_cue_key_door_learning_v1",
        classification="exploratory_single_seed_train_resubstitution_and_fixed_final",
        methods=list(METHODS),
        feature_dim=148,
        hidden_dim=8,
        num_classes=7,
        seed=17,
        learning_rate=0.03,
        momentum=0.9,
        trainable_parameters=dict(feature_methods=1255, recurrent_bc=1319),
        inference_steps=2,
        inference_learning_rate=0.04,
        cpc_configs={
            name: asdict(_config(TorchFeaturePolicy(name, 148, 8, 17, num_classes=7)))
            for name in ("neutral_cpc", "gated_cpc")
        },
        layouts=[asdict(layout) for layout in predefined_layouts()],
        cues=["blue", "red"],
        task_sequence=list(CONTEXTS),
        phase_updates=32,
        checkpoint_updates=list(POINTS),
        training="six TRAIN public-teacher traces collected once; A32/B32/C_train32/A32 alternating cues; original continuous optimizer",
        stream=dict(training_only=True, heldout=[], label_delay=1),
        evaluation="all five streams finish before any readout; six original TRAIN traces plus six known live episodes per checkpoint; final D endpoint128 only, without labels",
        action_rule="all seven native inputs; raw first argmax; no mask/repair/retry",
        metric_direction="CE lower; accuracy/exact_sequence_accuracy/success/return higher; positive gains desirable, positive losses/gaps worse",
        differences=dict(
            initial_A_gain="A32-A0",
            new_B_gain="B64-B32",
            new_C_gain="C96-C64",
            old_A_loss="A32-A96",
            return_A_gain="A128-A96",
            return_gap="A32-A128",
        ),
        primary_comparisons=["gated_cpc-minus-pc", "gated_cpc-minus-neutral_cpc"],
        acquisition_premise="raw exact_sequence_accuracy=1 and both live successes at A32/B64/C96; unmet premises make forgetting/recovery claims inconclusive",
        limits=dict(
            body_seconds=BODY_SECONDS,
            parent_seconds=210,
            process_rss_bytes=RSS_LIMIT,
            output_bytes=64 * 1024**2,
            cpu_threads=1,
            interop_threads=1,
            gpu=0,
        ),
        planned_work=dict(
            native_train_calls=640,
            training_rows=5760,
            teacher_episodes=6,
            teacher_decisions=54,
            source_environment_steps=54,
            raw_score_forwards=150,
            raw_score_rows=1350,
            known_live_episodes=150,
            final_live_episodes=10,
            maximum_live_predictions=2560,
            maximum_live_environment_steps=2560,
            maximum_recurrent_unroll_steps=1934,
        ),
        limitations="same SGD/row dose is not equal computation; recurrent_bc has extra recurrence parameters; single fixed seed and familiar TRAIN geometries; no generalization, equivalence or circadian advantage established",
    )


def _batch(episode: Any) -> AttributionBatch:
    features, targets = episode_arrays(episode)
    return AttributionBatch(
        "train",
        episode.context,
        tuple(f"{episode.episode_id}:step:{i}" for i in range(len(features))),
        features,
        targets,
        episode,
        7,
    )


def _episode(batch: AttributionBatch) -> DemonstrationEpisode:
    if batch.episode is None:
        raise ValueError("complete TRAIN episode required")
    return batch.episode


def _collect_sources(
    check_boundary: Callable[[], None], reports: list[dict[str, Any]]
) -> tuple[AttributionBatch, ...]:
    sources = []
    for layout in predefined_layouts()[:3]:
        for cue in ("blue", "red"):
            check_boundary()
            environment = MiniGridKeyDoor(
                layout,
                cue,
                seed=17,
                max_steps=16,
                role="train",
                episode_id=f"minigrid-source-{layout.context}-{cue}",
            )
            teacher = CueKeyDoorTeacher()
            problem: Exception | None = None
            try:
                for _ in range(16):
                    check_boundary()
                    action = teacher.choose_action(environment.observation)
                    check_boundary()
                    environment.step(action)
                    check_boundary()
                    if environment.terminal:
                        break
                episode = environment.to_demonstration()
                if len(episode.transitions) != 9 or environment.report["total_return"] <= 0:
                    raise ValueError(
                        "fixed teacher must produce a genuine nine-step successful TRAIN episode"
                    )
                sources.append(_batch(episode))
            except Exception as error:
                problem = error
                raise
            finally:
                cleanup_problem: Exception | None = None
                secondary: dict[str, Any] = {}
                try:
                    environment.close()
                except Exception as error:
                    cleanup_problem = error
                    secondary["close_error"] = f"{type(error).__name__}: {error}"
                try:
                    observed = environment.report
                except Exception as error:
                    observed = dict(failure_work_may_be_unobserved=True)
                    secondary["readback_error"] = f"{type(error).__name__}: {error}"
                    if cleanup_problem is None:
                        cleanup_problem = error
                observed.update(secondary)
                if problem is not None:
                    observed["primary_error"] = f"{type(problem).__name__}: {problem}"
                reports.append(observed)
                if cleanup_problem is not None and problem is None:
                    raise cleanup_problem
    return tuple(sources)


def _packets(
    sources: tuple[AttributionBatch, ...], *, context_sequence: tuple[str, ...] = CONTEXTS
) -> tuple[AttributionBatch, ...]:
    if (
        type(context_sequence) is not tuple
        or len(context_sequence) != 4
        or any(type(context) is not str for context in context_sequence)
        or context_sequence not in (CONTEXTS, ("A",) * 4)
    ):
        raise ValueError("context_sequence must be the original tuple or exactly four A phases")
    if type(sources) is not tuple or len(sources) != 6:
        raise ValueError("fixed collection requires six TRAIN source episodes")
    seen: set[str] = set()
    for index, source in enumerate(sources):
        _validate_batch(source, "train", seen)
        if (
            source.context != ("A", "B", "C_train")[index // 2]
            or source.features.shape != (9, 148)
            or source.num_classes != 7
            or source.episode is None
            or not source.episode.episode_id.endswith(("blue", "red")[index % 2])
        ):
            raise ValueError(
                "source order/role/shape/classes must bind the fixed TRAIN context/cue"
            )
    phase_indices = tuple({"A": 0, "B": 2, "C_train": 4}[context] for context in context_sequence)
    return tuple(
        _batch(
            replace(
                _episode(sources[phase_indices[i // 32] + i % 2]),
                episode_id=f"minigrid-training-{i}",
            )
        )
        for i in range(128)
    )


def _learner(method: str) -> Any:
    if method == "recurrent_bc":
        return TorchRecurrentLearner(TorchRecurrentPolicy(148, 8, 7, 17))
    return TorchFeatureLearner(TorchFeaturePolicy(method, 148, 8, 17, num_classes=7))


_EPISODE_LEAVES = attrgetter("episode_id", "context", "role", "num_classes", "transitions")
_TRANSITION_LEAVES = attrgetter(
    "step", "observation", "action", "reward", "next_observation", "terminal", "num_classes"
)


def _episode_signature(episode: Any) -> tuple[Any, ...]:
    # Why: observations are already validated native immutable tuples. Hold each
    # original leaf strongly and bind identity as well as value/type, so replacing
    # even an equal tuple is refused without serializing 148 coordinates per step
    # at every callback. Mutable array values remain fully compared as bytes.
    if type(episode.transitions) is not tuple:
        raise ValueError("episode signature requires the original immutable transitions tuple")
    return (
        episode,
        _EPISODE_LEAVES(episode),
        tuple(chain.from_iterable(map(_TRANSITION_LEAVES, episode.transitions))),
    )


def _signature(batches: tuple[AttributionBatch, ...]) -> tuple[Any, ...]:
    return tuple(
        (
            id(batch),
            batch.role,
            batch.context,
            batch.row_ids,
            batch.num_classes,
            batch.features.dtype.str,
            batch.features.shape,
            batch.features.tobytes(),
            batch.targets.dtype.str,
            batch.targets.shape,
            batch.targets.tobytes(),
            _episode_signature(_episode(batch)) if batch.episode is not None else None,
        )
        for batch in batches
    )


def _same_packet_bytes(value: Any, frozen: Any) -> bool:
    """Compare every bit of large native arrays without copying their full payload.

    Why: completed H32 guards repeatedly copy54x296 arrays just to compare bytes.
    Integer views preserve NaN payloads and signed zero; float equality would not.
    Small arrays and opaque inputs retain the original byte-comparison behavior.
    """
    if (
        type(value) is np.ndarray
        and value.dtype == np.float64
        and value.nbytes >= 65536
        and type(frozen) is bytes
        and len(frozen) == value.nbytes
    ):
        return np.array_equal(
            value.view(np.uint64), np.frombuffer(frozen, np.uint64).reshape(value.shape)
        )
    return value.tobytes() == frozen


def _check_signature(batches: tuple[AttributionBatch, ...], expected: tuple[Any, ...]) -> bool:
    # Compare the captured immutable leaves directly. Rebuilding 128 complete
    # signatures at every live callback dominated the first qualification budget.
    if len(batches) != len(expected):
        return False
    for batch, frozen in zip(batches, expected, strict=True):
        if (
            id(batch) != frozen[0]
            or batch.role is not frozen[1]
            or batch.context is not frozen[2]
            or batch.row_ids is not frozen[3]
            or batch.num_classes is not frozen[4]
            or batch.features.dtype.str != frozen[5]
            or batch.features.shape != frozen[6]
            or not _same_packet_bytes(batch.features, frozen[7])
            or batch.targets.dtype.str != frozen[8]
            or batch.targets.shape != frozen[9]
            or not _same_packet_bytes(batch.targets, frozen[10])
        ):
            return False
        if frozen[11] is None:
            if batch.episode is not None:
                return False
            continue
        episode, binding = _episode(batch), frozen[11]
        # Why: the measured hotspot was Python getattr/generator work per leaf.
        # C-level attribute collection and identity comparison still visit every
        # captured field, retaining strong references and exact identity checks.
        if (
            episode is not binding[0]
            or len(binding[1]) != 5
            or not all(map(is_, _EPISODE_LEAVES(episode), binding[1]))
        ):
            return False
        # Why: the original immutable transitions tuple above binds every
        # transition owner. C iterators still visit all seven fields per item,
        # avoiding a Python loop and wrapper per transition on this measured path.
        if len(binding[2]) != 7 * len(episode.transitions) or not all(
            map(
                is_,
                chain.from_iterable(map(_TRANSITION_LEAVES, episode.transitions)),
                binding[2],
            )
        ):
            return False
    return True


def _counter(state: Any, point: int, *, catalogue_rows: int | None = None) -> None:
    if catalogue_rows is not None and (
        type(catalogue_rows) is not int
        or catalogue_rows != 54
        or type(point) is not int
        or not 0 <= point <= 128
    ):
        raise ValueError("joint catalogue requires exact54-row wakes and a native point in0..128")
    rows = 9 if catalogue_rows is None else catalogue_rows
    if (
        type(state.completed_updates) is not int
        or state.completed_updates != point
        or type(state.training_examples) is not int
        or state.training_examples != rows * point
    ):
        raise ValueError(
            "native update/row clocks differ from the fixed nine-row schedule"
            if catalogue_rows is None
            else "native update/row clocks differ from the explicit54-row catalogue"
        )


def _memory_counter(state: Any, point: int) -> None:
    """Bind this control to its explicit geometry/cap and original exposure law."""
    owned = copy_action_memory_snapshot(state)
    if owned.policy != ActionMemoryPolicy(feature_dim=296, num_classes=7, max_payload_bytes=155136):
        raise ValueError("MiniGrid memory requires its explicit296/7 bounded policy")
    _counter(owned, point)


def _invariant_memory_counter(state: Any, point: int, method: str, *, fifo36: bool = False) -> None:
    """Keep representation identity and native clocks bound to the declared arm."""
    if type(fifo36) is not bool or (
        fifo36 and method != "public_invariant_observation_memory_fifo"
    ):
        raise ValueError("FIFO36 requires its explicit invariant FIFO recipe")
    owned = copy_invariant_memory_snapshot(state)
    policy = owned.policy
    if (
        policy.method != method
        or policy.feature_dim != 296
        or policy.num_classes != 7
        or policy.max_entries != (36 if fifo36 else 64)
        or policy.max_payload_bytes != (87264 if fifo36 else 155136)
    ):
        raise ValueError("invariant count policy differs from the fixed MiniGrid recipe")
    _counter(owned, point)


def _invariant_feature_counter(
    state: Any,
    point: int,
    method: str,
    *,
    catalogue_rows: int | None = None,
    wide_neural: bool = False,
) -> None:
    """Bind the shared stream to its explicit representation and native objective."""
    _invariant_feature_snapshot(
        state, point, method, catalogue_rows=catalogue_rows, wide_neural=wide_neural
    )


def _invariant_feature_snapshot(
    state: Any,
    point: int,
    method: str,
    *,
    catalogue_rows: int | None = None,
    wide_neural: bool = False,
) -> Any:
    """Return the owned state that passed the existing recipe and clock checks.

    Why: a caller needing both validation and an owned snapshot must use the same
    observed copy, rather than validate one copy and then reread the input.
    """
    if type(wide_neural) is not bool or (
        wide_neural
        and (
            type(catalogue_rows) is not int
            or catalogue_rows != 54
            or type(method) is not str
            or method not in (*_NATIVE_INVARIANT_FEATURE_METHODS, *relaxation.BOUND_METHODS)
        )
    ):
        raise ValueError("wide neural counter requires an explicit original54-row H32 recipe")
    if type(method) is str and method in cue_relative.METHODS:
        return cue_relative.snapshot(state, point, method, catalogue_rows=catalogue_rows)
    owned = copy_invariant_torch_feature_snapshot(state)
    policy = owned.policy
    native = policy.native_policy
    if wide_neural:
        # Why: width is an explicit study recipe, never inferred from a foreign state.
        valid = (
            policy == relaxation.policy(method, wide_neural=True)
            if method in relaxation.BOUND_METHODS
            else policy.method == method
            and native
            == TorchFeaturePolicy(
                method.removeprefix("public_invariant_"), 296, 32, 17, num_classes=7
            )
        )
        if not valid:
            raise ValueError("wide invariant neural policy differs from the fixed H32 recipe")
    elif type(method) is str and method in relaxation.BOUND_METHODS:
        if catalogue_rows is None or policy != relaxation.policy(method):
            raise ValueError("fixed64 control requires its exact policy and joint54-row law")
    elif policy.method != method or native != TorchFeaturePolicy(
        native.method, 296, 8, 17, num_classes=7
    ):
        raise ValueError("invariant neural policy differs from the fixed MiniGrid recipe")
    if catalogue_rows is None:
        _counter(owned, point)
    else:
        _counter(owned, point, catalogue_rows=catalogue_rows)
    return owned


def _paired_counter(state: Any, point: int) -> None:
    """Validate the explicit paired type and actual step/episode/row clocks."""
    if type(point) is not int or not 0 <= point <= 64:
        raise ValueError("paired checkpoint must be a native step in0..64")
    owned = paired._snapshot(state)
    if (
        owned.policy != TorchRecurrentPolicy(148, 8, 7, 17)
        or owned.completed_updates != point
        or owned.training_episodes != 2 * point
        or owned.training_examples != 18 * point
    ):
        raise ValueError("paired native clocks/policy differ from the fixed two-episode recipe")


def _paired_history_counter(state: Any, point: int) -> None:
    """Admit only the explicit296 paired recipe, retaining actual episode clocks."""
    if type(point) is not int or not 0 <= point <= 64:
        raise ValueError("paired history checkpoint must be a native step in0..64")
    owned = paired._snapshot(state)
    if (
        owned.policy != TorchRecurrentPolicy(296, 8, 7, 17)
        or owned.completed_updates != point
        or owned.training_episodes != 2 * point
        or owned.training_examples != 18 * point
    ):
        raise ValueError("paired history clocks/policy differ from the explicit296 recipe")


def _row(method: str) -> dict[str, Any]:
    return dict(
        method=method,
        status="not_run",
        reason=None,
        stream_result=None,
        checkpoint_updates=[],
        charged_updates=None,
        native_training_examples=None,
        source_unchanged=None,
        actor_unchanged=None,
        partial_stream=None,
        wake_calls_attempted=0,
        wake_calls_returned=0,
        readouts=[dict(updates=point, role="train", status="not_run") for point in POINTS],
        final_evaluation=dict(updates=128, role="final_test", status="not_run"),
        outcomes=None,
        signed_metrics=None,
        acquisition_conditions=None,
        work=None,
    )


def _readback(
    row: dict[str, Any],
    source: Any,
    initial: Any,
    stream: Any,
    budget: Any,
    *,
    state_equal: Callable[[Any, Any], bool] | None = None,
    catalogue_rows: int | None = None,
) -> None:
    if catalogue_rows is not None and (type(catalogue_rows) is not int or catalogue_rows != 54):
        raise ValueError("joint readback requires exact54-row exposure")
    rows_per_update = 9 if catalogue_rows is None else catalogue_rows
    state_equal = _same_state if state_equal is None else state_equal
    if budget is not None:
        row["charged_updates"] = budget.updates_completed
    if stream is not None and row["method"] in _RECEIPT_METHODS:
        # Why: a failed native port may no longer admit model reads. Preserve
        # original committed receipts first without bypassing that closed port
        # or inferring the unavailable current model clock from its ledger.
        receipts = [asdict(receipt) for receipt in stream._runtime.applied_updates]
        row["partial_stream"] = dict(
            completed_wake_updates=stream.completed_updates,
            native_candidate_updates=None,
            native_candidate_training_examples=None,
            applied_updates=receipts,
            committed_training_examples=rows_per_update * len(receipts),
            current_native_state_unavailable=True,
        )
    if source is not None:
        row["source_unchanged"] = state_equal(initial, source.snapshot_state())
    if stream is not None:
        candidate = stream._runtime._candidate.snapshot_state()
        row["native_training_examples"] = candidate.training_examples
        if row["method"] == "recurrent_paired_bc":
            row["native_training_episodes"] = candidate.training_episodes
        row["actor_unchanged"] = state_equal(initial, stream._runtime.actor.snapshot_state().state)
        row["partial_stream"] = dict(
            completed_wake_updates=stream.completed_updates,
            native_candidate_updates=candidate.completed_updates,
            native_candidate_training_examples=candidate.training_examples,
            applied_updates=[asdict(receipt) for receipt in stream._runtime.applied_updates],
        )
        if row["method"] in _RECEIPT_METHODS:
            row["partial_stream"].update(
                committed_training_examples=rows_per_update
                * len(row["partial_stream"]["applied_updates"]),
                current_native_state_unavailable=False,
            )
            row["source_unchanged"] = state_equal(initial, source.snapshot_state())


def _training_budget_evidence(budget: Any, origin: tuple[Any, ...] | None) -> dict[str, Any]:
    """Observe original authority without granting it or calling model ports."""
    if origin is None:
        return dict(born=False, original=None, observed=None, bindings=None)
    owner, config, limits, clock, started = origin
    original = dict(limits=deepcopy(limits), started_at=started)
    invalid: list[str] = []

    def number(name: str, value: Any, nullable: bool = False) -> Any:
        integer = name.endswith("completed") or (
            name.startswith("max_") and name != "max_wall_seconds"
        )
        if (value is None and nullable) or (
            (type(value) is int if integer else type(value) in (int, float))
            and isfinite(value)
            and value >= 0
        ):
            return value
        invalid.append(name)
        return None

    try:
        values = vars(owner)
        observed = {
            name: number(name, values.get(name))
            for name in (
                "started_at",
                "last_clock",
                "updates_completed",
                "replay_examples_completed",
            )
        }
        current = vars(owner.budget) if type(owner.budget) is ToyExecutionBudget else {}
        current_limits = {name: number(name, current.get(name), name in current) for name in limits}
        return dict(
            born=True,
            original=original,
            observed=observed,
            observed_limits=current_limits,
            invalid_fields=invalid,
            phase="before_final_model_readback",
            bindings=dict(
                session=budget is owner,
                config=owner.budget is config,
                clock=owner.clock is clock,
                started_at=type(owner.started_at) is type(started) and owner.started_at == started,
                limits=not any(name in invalid for name in limits)
                and _same_value(current_limits, limits),
            ),
        )
    except Exception as error:
        # Keep the trusted birth even when external corruption prevents observation.
        return dict(
            born=True,
            original=original,
            observed=None,
            bindings=None,
            observation_error=type(error).__name__,
        )


def _train_arm(
    row: dict[str, Any],
    packets: tuple[AttributionBatch, ...],
    sampler: Any,
    deadline: float,
    check_boundary: Callable[[], None],
    *,
    learner_factory: Callable[[str], Any] | None = None,
    checkpoint_updates: tuple[int, ...] = POINTS,
    public_history: bool = False,
    invariant_fifo36: bool = False,
    catalogue_rows: int | None = None,
    wide_neural: bool = False,
    recurrent_replay: str | None = None,
    record_budget_evidence: bool = False,
) -> dict[str, Any]:
    if type(record_budget_evidence) is not bool or (
        record_budget_evidence and (type(row) is not dict or "budget_evidence" in row)
    ):
        raise ValueError("budget evidence requires an exact bool and no replaced prior evidence")
    if type(row) is not dict or type(row.get("method")) is not str:
        raise ValueError("training row requires its declared method")
    if recurrent_replay is not None:
        from src.adapters import minigrid_recurrent_replay as replay

        replay.validate_mode(recurrent_replay)
        replay.validate_packets(packets)
        if (
            row["method"] != "recurrent_public_cue"
            or learner_factory is None
            or public_history
            or invariant_fifo36
            or catalogue_rows is not None
            or wide_neural
        ):
            raise ValueError("recurrent replay requires its explicit raw148 two-head factory")
    if type(wide_neural) is not bool or (
        wide_neural
        and (
            type(catalogue_rows) is not int
            or catalogue_rows != 54
            or row["method"] not in (*_NATIVE_INVARIANT_FEATURE_METHODS, *relaxation.BOUND_METHODS)
            or learner_factory is None
            or type(checkpoint_updates) is not tuple
            or any(type(point) is not int for point in checkpoint_updates)
            or checkpoint_updates != POINTS
            or len(packets) != 128
            or public_history is not False
            or invariant_fifo36 is not False
        )
    ):
        raise ValueError("wide neural training requires its explicit original128/54-row factory")
    if catalogue_rows is not None and (
        type(catalogue_rows) is not int
        or catalogue_rows != 54
        or row["method"] not in _INVARIANT_FEATURE_METHODS
        or learner_factory is None
    ):
        raise ValueError("joint training requires exact54-row bound neural exposure and factory")
    if type(invariant_fifo36) is not bool or (
        invariant_fifo36
        and (row["method"] != "public_invariant_observation_memory_fifo" or learner_factory is None)
    ):
        raise ValueError("FIFO36 requires an explicit invariant FIFO learner factory")
    if type(public_history) is not bool or (
        public_history and (row["method"] != "recurrent_paired_bc" or learner_factory is None)
    ):
        raise ValueError(
            "public_history must be an exact bool and requires an explicit paired factory"
        )
    if (
        type(checkpoint_updates) is not tuple
        or any(type(point) is not int for point in checkpoint_updates)
        or checkpoint_updates
        not in (
            ((0, 16, 32, 48, 64), (0, 4))
            if row["method"] == "recurrent_paired_bc"
            else (POINTS, (0, 32))
        )
        or len(packets) != checkpoint_updates[-1]
        or (row["method"] == "recurrent_paired_bc" and learner_factory is None)
    ):
        raise ValueError(
            "paired training requires64steps or the explicit4step qualification recipe"
            if row["method"] == "recurrent_paired_bc"
            else "training requires the original128 or explicit32 checkpoint recipe"
        )
    stop = checkpoint_updates[-1]
    original_method = row["method"]
    if original_method in _RECEIPT_METHODS and learner_factory is None:
        raise ValueError("MiniGrid encoded/count control requires its explicit native factory")
    # Why: tensor-aware comparison is retained for every original neural arm;
    # pure count snapshots use the existing backend-neutral full-value comparator.
    same_state = _same_value if original_method in _COUNT_METHODS else _same_state
    completed_paid_payload: bytes | None = None
    if (
        catalogue_rows is None
        and row["method"] != "recurrent_paired_bc"
        and any(p.episode is None for p in packets)
    ):
        raise ValueError("legacy MiniGrid training requires complete single-episode metadata")
    counter = (
        (
            lambda state, point: _invariant_feature_counter(
                state, point, original_method, catalogue_rows=54, wide_neural=True
            )
        )
        if wide_neural
        else (
            lambda state, point: _invariant_feature_counter(
                state, point, original_method, catalogue_rows=54
            )
        )
        if catalogue_rows is not None
        else (lambda state, point: _invariant_feature_counter(state, point, original_method))
        if original_method in _INVARIANT_FEATURE_METHODS
        else (
            lambda state, point: _invariant_memory_counter(
                state, point, original_method, fifo36=invariant_fifo36
            )
        )
        if original_method in _INVARIANT_COUNT_METHODS
        else _memory_counter
        if original_method == "observation_memory"
        else _paired_history_counter
        if public_history
        else _paired_counter
        if row["method"] == "recurrent_paired_bc"
        else (lambda state, point: replay.check_snapshot(state, point))
        if recurrent_replay is not None
        else _counter
    )
    rows_per_update = (
        54 if catalogue_rows is not None else 18 if row["method"] == "recurrent_paired_bc" else 9
    )
    if row["method"] == "recurrent_paired_bc" or catalogue_rows is not None:
        seen: set[str] = set()
        for packet in packets:
            _validate_batch(packet, "train", seen)
    if catalogue_rows is not None and any(
        packet.episode is not None
        or packet.context != "train-catalogue"
        or packet.num_classes != 7
        or packet.features.shape != (54, 296)
        or packet.targets.shape != (54, 1)
        for packet in packets
    ):
        raise ValueError("joint stream requires original independent54-row TRAIN296/K7 arrays")
    if row["method"] == "recurrent_paired_bc" and any(
        packet.episode is not None
        or packet.role != "train"
        or packet.context != "A"
        or packet.num_classes != 7
        or packet.features.shape != (18, 296 if public_history else 148)
        or packet.targets.shape != (18, 1)
        for packet in packets
    ):
        raise ValueError("paired stream requires original two-episode18row TRAIN A arrays")
    source = initial = stream = budget = None
    budget_origin: tuple[Any, ...] | None = None
    budget_exit_owner: dict[str, Any] | None = None
    budget_exit_expected: dict[str, Any] | None = None
    replay_witness = None
    problem: Exception | None = None
    states: list[tuple[int, Any]] = []
    row["status"] = "training"

    def readback() -> None:
        # Keep original neural callback signatures and dynamic comparator;
        # explicit count arms require their backend-neutral comparison.
        if catalogue_rows is not None:
            _readback(row, source, initial, stream, budget, catalogue_rows=54)
        elif original_method in _COUNT_METHODS:
            _readback(row, source, initial, stream, budget, state_equal=same_state)
        else:
            _readback(row, source, initial, stream, budget)

    try:
        check_boundary()
        source = (_learner if learner_factory is None else learner_factory)(row["method"])
        initial = deepcopy(source.snapshot_state())
        counter(initial, 0)
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=stop,
                max_wall_seconds=max(0.0, deadline - monotonic()),
                max_process_rss_bytes=RSS_LIMIT,
            ),
            monotonic,
        )
        if record_budget_evidence:
            # Why: failed arms do not return their budget. Retain its real birth
            # before sampler/native callbacks and never reconstruct it later.
            budget_origin = (
                budget,
                budget.budget,
                asdict(budget.budget),
                budget.clock,
                budget.started_at,
            )
        budget.attach_memory(sampler)
        original_config, original_clock, original_start = (
            budget.budget,
            budget.clock,
            budget.started_at,
        )
        if recurrent_replay is not None:
            replay_witness = replay.RecurrentReplayWitness(
                source, budget, packets, recurrent_replay, sampler
            )

        def observe(point: int, state: Any, batch: Any) -> None:
            check_boundary()
            if replay_witness is not None:
                replay_witness.admit(point, include_inputs=False)
            counter(state, point)
            if point != checkpoint_updates[len(states)]:
                raise ValueError("checkpoint order changed")
            states.append((point, deepcopy(state)))

        stream_options: dict[str, Any] = {}
        if replay_witness is not None:
            stream_options["rehearsal"] = replay_witness.helper
        stream = AttributionStream(
            source,
            packets,
            (),
            checkpoint_updates=checkpoint_updates,
            label_delay=1,
            budget=budget,
            observe_checkpoint=observe,
            training_only=True,
            **stream_options,
        )
        runtime, candidate, actor = (
            stream._runtime,
            stream._runtime._candidate,
            stream._runtime.actor,
        )
        original_training = stream._training
        training_signature = _signature(original_training)

        def active_guard(*, include_boundary: bool = True) -> None:
            if include_boundary:
                check_boundary()
            if (
                row["method"] != original_method
                or stream._runtime is not runtime
                or runtime._candidate is not candidate
                or runtime.actor is not actor
                or stream._learner is not source
                or stream._budget is not budget
                or budget.budget is not original_config
                or budget.clock is not original_clock
                or budget.started_at != original_start
                or budget.process_rss_sampler is not sampler
                or budget.progress is not None
                or stream._training is not original_training
                or not _check_signature(original_training, training_signature)
                or (
                    record_budget_evidence
                    and (
                        (budget_exit_owner is None and "budget_evidence" in row)
                        or (
                            budget_exit_owner is not None
                            and (
                                row.get("budget_evidence") is not budget_exit_owner
                                or not _same_value(row["budget_evidence"], budget_exit_expected)
                            )
                        )
                    )
                )
            ):
                raise ValueError("original source/runtime/budget/training bindings changed")
            stream._check_original()
            budget.before_final()
            if not same_state(initial, source.snapshot_state()) or not same_state(
                initial, actor.snapshot_state().state
            ):
                raise ValueError("source or serving actor changed during isolated training")
            if replay_witness is not None:
                if stream._rehearsal is not replay_witness.helper:
                    raise ValueError("original recurrent replay helper was replaced")
                replay_witness.admit(stream.completed_updates, include_inputs=False)

        for point in range(1, stop + 1):
            active_guard()
            row["wake_calls_attempted"] += 1
            stream.run_until(point)
            row["wake_calls_returned"] += 1
            active_guard()
            counter(candidate.snapshot_state(), point)
        active_guard()
        result = stream.finish()
        active_guard()
        if (
            result["evaluation_scope"] != "training_only"
            or result["score_forward_calls"] != 0
            or result["checkpoints"] != []
            or result["completed_updates"] != stop
            or result["training_examples"] != rows_per_update * stop
            or len(result["applied_updates"]) != stop
            or (
                original_method == "recurrent_paired_bc"
                and any(
                    receipt["diagnostic"]["definition"] != paired.TORCH_RECURRENT_EPISODE_PAIR_CE_ID
                    for receipt in result["applied_updates"]
                )
            )
            or budget.updates_completed != stop
            or budget.replay_examples_completed != 0
        ):
            raise ValueError("training-only completion receipts or permanent charges differ")
        row.update(
            stream_result=result,
            checkpoint_updates=[point for point, _ in states],
            status="trained",
        )
        readback()
        # These three records contain no native tensors. Whole-record bytes keep
        # exact fields/types while avoiding the measured recursive scalar walk.
        frozen_result, frozen_states = pickle.dumps(result, protocol=5), deepcopy(states)
        frozen_native, frozen_receipts = (
            deepcopy(candidate.snapshot_state()),
            pickle.dumps(runtime.applied_updates, protocol=5),
        )
        frozen_partial = pickle.dumps(row["partial_stream"], protocol=5)
        original_states = stream._states
        frozen_stream_states = deepcopy(original_states)
        original_state_objects = tuple(id(state) for _, state in states)
        original_stream_state_objects = tuple(id(state) for _, state in original_states)
        last_clock = budget.last_clock

        def completed_guard() -> None:
            nonlocal last_clock
            if budget.last_clock < last_clock:
                raise ValueError("original budget clock moved backwards")
            active_guard(include_boundary=False)
            if (
                not stream._finished
                or stream.completed_updates != stop
                or budget.updates_completed != stop
                or budget.replay_examples_completed != 0
                or budget.last_clock < last_clock
                or row["stream_result"] is not result
                or pickle.dumps(result, protocol=5) != frozen_result
                or tuple(id(state) for _, state in states) != original_state_objects
                or stream._states is not original_states
                or tuple(id(state) for _, state in original_states) != original_stream_state_objects
                or not same_state(frozen_states, states)
                or not same_state(frozen_stream_states, original_states)
                or not same_state(frozen_native, candidate.snapshot_state())
                or pickle.dumps(runtime.applied_updates, protocol=5) != frozen_receipts
                or pickle.dumps(row["partial_stream"], protocol=5) != frozen_partial
                or row["checkpoint_updates"] != list(checkpoint_updates)
                or row["charged_updates"] != stop
                or row["native_training_examples"]
                != rows_per_update
                * (replay.native_updates(stop) if replay_witness is not None else stop)
                or (
                    row["method"] == "recurrent_paired_bc"
                    and row["native_training_episodes"] != 2 * stop
                )
                or row["source_unchanged"] is not True
                or row["actor_unchanged"] is not True
            ):
                raise ValueError("preceding completed training evidence changed")
            if original_method in _RECEIPT_METHODS and (
                not same_state(initial, actor.snapshot_state().state)
                or not same_state(initial, source.snapshot_state())
            ):
                raise ValueError("source or serving actor changed during candidate readback")
            last_clock = budget.last_clock

        # Peer checks are performed by the caller's loop. Completed guards only
        # inspect their own arm, keeping the dependency graph nonrecursive.
        completed_guard()
        if original_method == "recurrent_paired_bc" or original_method in (
            *_INVARIANT_COUNT_METHODS,
            *_INVARIANT_FEATURE_METHODS,
        ):
            completed_paid_payload = pickle.dumps(row, protocol=5)
        return dict(
            row=row,
            source=source,
            states=states,
            stream=stream,
            budget=budget,
            guard=completed_guard,
            **({"replay": replay_witness} if replay_witness is not None else {}),
        )
    except Exception as error:
        problem = error
        row.update(status="incomplete", reason=f"{type(error).__name__}: {error}")
        if replay_witness is not None:
            row["replay_failure_work"] = deepcopy(getattr(error, "work", None))
        if original_method == "recurrent_paired_bc":
            row["paired_failure_work"] = (
                deepcopy(error.work) if isinstance(error, paired.EpisodePairUpdateFailed) else {}
            )
        raise
    finally:
        # Sample counters/clocks before a failed candidate's readback can raise.
        # This is detached report evidence; original runtime authority is untouched.
        evidence = (
            _training_budget_evidence(budget, budget_origin) if record_budget_evidence else None
        )
        try:
            if replay_witness is not None:
                row["replay"] = replay_witness.evidence()
                row["replay_budget"] = dict(
                    limits=asdict(replay_witness.replay_budget.budget),
                    started_at=replay_witness.replay_budget.started_at,
                    last_clock=replay_witness.replay_budget.last_clock,
                    charged_examples=replay_witness.replay_budget.replay_examples_completed,
                )
            readback()
            if completed_paid_payload is not None:
                completed_guard()
                if pickle.dumps(row, protocol=5) != completed_paid_payload:
                    raise ValueError("final readback changed preceding paid evidence")
        except Exception as error:
            if completed_paid_payload is not None:
                # Repair only the report from observed private charges, never
                # model state or an unknown change made by a callback.
                row.clear()
                row.update(pickle.loads(completed_paid_payload))
            row["readback_error"] = f"{type(error).__name__}: {error}"
            row["status"] = "incomplete"
            if problem is None:
                row["reason"] = f"readback {type(error).__name__}: {error}"
                raise
        finally:
            if evidence is not None:
                budget_exit_owner = evidence
                budget_exit_expected = deepcopy(evidence)
                row["budget_evidence"] = evidence


def _outcomes(
    readouts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, bool]]:
    points = []
    for readout in readouts:
        point: dict[str, Any] = dict(updates=readout["updates"])
        for context in ("A", "B", "C_train"):
            scores = [score for score in readout["scores"] if score["context"] == context]
            episodes = [episode for episode in readout["episodes"] if episode["context"] == context]
            if (
                len(scores) != 2
                or len(episodes) != 2
                or any(e["status"] != "complete" for e in episodes)
            ):
                raise ValueError("outcome requires both fixed cues and genuine complete readouts")
            point[context] = {
                key: sum(score[key] for score in scores) / 2
                for key in ("ce", "accuracy", "exact_sequence_accuracy")
            }
            point[context].update(
                success=sum(e["success"] for e in episodes) / 2,
                mean_return=sum(e["return"] for e in episodes) / 2,
            )
        points.append(point)
    comparisons = dict(
        initial_A_gain=("A", 0, 1),
        new_B_gain=("B", 1, 2),
        new_C_gain=("C_train", 2, 3),
        old_A_loss=("A", 3, 1),
        return_A_gain=("A", 3, 4),
        return_gap=("A", 4, 1),
    )
    metrics = {
        metric: {
            name: (-1 if metric == "ce" else 1)
            * (points[right][context][metric] - points[left][context][metric])
            for name, (context, left, right) in comparisons.items()
        }
        for metric in ("ce", "accuracy", "exact_sequence_accuracy", "success", "mean_return")
    }
    acquired = {
        context: bool(
            points[index][context]["exact_sequence_accuracy"] == 1
            and points[index][context]["success"] == 1
        )
        for context, index in (("A", 1), ("B", 2), ("C_train", 3))
    }
    return points, metrics, acquired


def _work(row: dict[str, Any]) -> dict[str, Any]:
    readouts = [*row["readouts"], row["final_evaluation"]]
    observed = [readout["work"] for readout in readouts if "work" in readout]
    keys = set().union(*(work.keys() for work in observed))
    work = {
        key: any(item[key] for item in observed)
        if key == "failure_work_may_be_unobserved"
        else sum(item[key] for item in observed)
        for key in keys
    }
    work.update(
        native_train_calls=row["charged_updates"], training_rows=row["native_training_examples"]
    )
    if row["method"] in _RECEIPT_METHODS:
        partial = row.get("partial_stream")
        work["observed_committed_training_rows"] = (
            None if partial is None else partial["committed_training_examples"]
        )
    work.update(
        wake_calls_attempted=row["wake_calls_attempted"],
        wake_calls_returned=row["wake_calls_returned"],
        native_training_work_may_be_unobserved=row["wake_calls_attempted"]
        > row["wake_calls_returned"],
    )
    if row["method"] == "recurrent_bc" and row["native_training_examples"] is not None:
        work["recurrent_unroll_steps"] = (
            row["native_training_examples"]
            + work.get("raw_score_rows_completed", 0)
            + work.get("live_prediction_calls_returned", 0)
        )
    return work


def _restore_readouts(
    row: dict[str, Any], owned_slots: list[bytes], owned_final: bytes
) -> dict[str, Any]:
    """Decode only this invocation's private captured payloads, never public slots.

    Why: completed reports grow with every episode. Serializing that entire
    history at each callback dominated native readouts. Immutable owned bytes
    retain every original field; public slot identity still rejects replacement.
    Callers supply private bytes produced here, not file or public-row input.
    """
    return dict(
        row,
        readouts=[pickle.loads(payload) for payload in owned_slots],
        final_evaluation=pickle.loads(owned_final),
    )


def run_pilot() -> dict[str, Any]:
    """Execute one frozen budgeted case, preserving every failed observed prefix."""
    started, sampler = monotonic(), None
    rows = [_row(method) for method in METHODS]
    report: dict[str, Any] = dict(
        protocol=describe_protocol(),
        status="running",
        reason=None,
        training_sources=[],
        source_reports=[],
        methods=rows,
        training_summary=None,
        comparisons=None,
        observed_process_rss=None,
        total_seconds=None,
        scientific_advantage_verified=False,
    )
    arms: list[dict[str, Any]] = []
    source_signature = packet_signature = None
    sources: tuple[AttributionBatch, ...] = ()
    packets: tuple[AttributionBatch, ...] = ()
    row_identities = tuple(id(row) for row in rows)
    original_protocol = report["protocol"]
    protocol_signature = pickle.dumps(original_protocol, protocol=5)
    readout_lists = tuple(row["readouts"] for row in rows)
    expected_readouts = [list(row["readouts"]) for row in rows]
    expected_finals = [row["final_evaluation"] for row in rows]
    expected_slots = [[pickle.dumps(slot, protocol=5) for slot in row["readouts"]] for row in rows]
    expected_final_slots = [pickle.dumps(row["final_evaluation"], protocol=5) for row in rows]
    source_records = source_reports = None
    expected_source_records = report["training_sources"]
    expected_source_reports = report["source_reports"]
    expected_summary = pickle.dumps(None, protocol=5)
    publication: tuple[list[dict[str, Any]], dict[str, Any]] | None = None
    rendered: list[dict[str, Any]] = []

    def resources() -> None:
        if monotonic() - started > BODY_SECONDS:
            raise ValueError("fixed whole-case wall-time budget exhausted")
        if sampler is not None:
            sampler.sample()
            if sampler.snapshot().peak_bytes > RSS_LIMIT:
                raise ValueError("fixed whole-case sampled process RSS budget exhausted")

    def common_guard() -> None:
        resources()
        if (
            report["methods"] is not rows
            or tuple(id(row) for row in rows) != row_identities
            or tuple(row["method"] for row in rows) != METHODS
            or report["protocol"] is not original_protocol
            or pickle.dumps(original_protocol, protocol=5) != protocol_signature
            or report["training_sources"] is not expected_source_records
            or report["source_reports"] is not expected_source_reports
            or pickle.dumps(report["training_summary"], protocol=5) != expected_summary
        ):
            raise ValueError("fixed learner slots changed")
        for index, row in enumerate(rows):
            if (
                row["readouts"] is not readout_lists[index]
                or len(row["readouts"]) != 5
                or any(
                    actual is not expected
                    for actual, expected in zip(
                        row["readouts"], expected_readouts[index], strict=True
                    )
                )
                or row["final_evaluation"] is not expected_finals[index]
                or any(
                    pickle.dumps(slot, protocol=5) != frozen
                    for slot, frozen in zip(row["readouts"], expected_slots[index], strict=True)
                    if type(slot) is not bytes
                )
                or (
                    type(row["final_evaluation"]) is not bytes
                    and pickle.dumps(row["final_evaluation"], protocol=5)
                    != expected_final_slots[index]
                )
            ):
                raise ValueError("fixed readout slots or completed evidence changed")
        if source_signature is not None and (
            not _check_signature(sources, source_signature)
            or not _check_signature(packets, packet_signature)
            or source_records != pickle.dumps(report["training_sources"], protocol=5)
            or source_reports != pickle.dumps(report["source_reports"], protocol=5)
        ):
            raise ValueError("original TRAIN sources or observed source evidence changed")

    def all_completed() -> None:
        common_guard()
        for arm in arms:
            arm["guard"]()
        common_guard()

    try:
        torch = require_torch()
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        with ProcessRssSampler() as active_sampler:
            sampler = active_sampler
            resources()
            sources = _collect_sources(resources, report["source_reports"])
            packets = _packets(sources)
            report["training_sources"] = [asdict(_episode(packet)) for packet in sources]
            expected_source_records = report["training_sources"]
            source_records, source_reports = (
                pickle.dumps(report["training_sources"], protocol=5),
                pickle.dumps(report["source_reports"], protocol=5),
            )
            source_signature, packet_signature = _signature(sources), _signature(packets)
            for row in rows:
                all_completed()
                # Freeze the preceding-arm list for this closure, avoiding a
                # recursive dependency when this successful arm is appended.
                preceding = tuple(arm["guard"] for arm in arms)

                def before_arm(preceding=preceding) -> None:
                    common_guard()
                    for guard in preceding:
                        guard()

                arms.append(_train_arm(row, packets, sampler, started + BODY_SECONDS, before_arm))
            all_completed()
            report["training_summary"] = dict(
                completed_methods=5,
                native_train_calls=640,
                training_rows=5760,
                final_data_constructed_before_completion=False,
            )
            expected_summary = pickle.dumps(report["training_summary"], protocol=5)
            for row_index, arm in enumerate(arms):
                row = arm["row"]
                for index, (point, state) in enumerate(arm["states"]):
                    all_completed()
                    try:
                        readout = evaluate_checkpoint(
                            arm["source"],
                            state,
                            method=row["method"],
                            updates=point,
                            layouts=predefined_layouts()[:3],
                            role="train",
                            sources=sources,
                            budget=arm["budget"],
                            check_boundary=all_completed,
                        )
                    except MiniGridEvaluationFailed as error:
                        frozen = pickle.dumps(error.report, protocol=5)
                        row["readouts"][index] = frozen
                        expected_readouts[row_index][index] = frozen
                        expected_slots[row_index][index] = frozen
                        row.update(status="incomplete", reason=str(error))
                        raise
                    frozen = pickle.dumps(readout, protocol=5)
                    row["readouts"][index] = frozen
                    expected_readouts[row_index][index] = frozen
                    expected_slots[row_index][index] = frozen
                    all_completed()
            # Final-only construction is reachable only after every fixed TRAIN
            # readout succeeds; these outcomes cannot affect any update.
            for row_index, arm in enumerate(arms):
                row = arm["row"]
                all_completed()
                try:
                    final = evaluate_checkpoint(
                        arm["source"],
                        arm["states"][-1][1],
                        method=row["method"],
                        updates=128,
                        layouts=predefined_layouts()[3:],
                        role="final_test",
                        sources=(),
                        budget=arm["budget"],
                        check_boundary=all_completed,
                    )
                except MiniGridEvaluationFailed as error:
                    frozen = pickle.dumps(error.report, protocol=5)
                    row["final_evaluation"] = frozen
                    expected_finals[row_index] = frozen
                    expected_final_slots[row_index] = frozen
                    row.update(status="incomplete", reason=str(error))
                    raise
                frozen = pickle.dumps(final, protocol=5)
                row["final_evaluation"] = frozen
                expected_finals[row_index] = frozen
                expected_final_slots[row_index] = frozen
                all_completed()
            derived: list[dict[str, Any]] = []
            for index, row in enumerate(rows):
                decoded = _restore_readouts(row, expected_slots[index], expected_final_slots[index])
                points, metrics, acquired = _outcomes(decoded["readouts"])
                derived.append(
                    dict(
                        outcomes=points,
                        signed_metrics=metrics,
                        acquisition_conditions=acquired,
                        initial_A_acquired=acquired["A"],
                        interpretation="descriptive_acquired_task_return"
                        if acquired["A"]
                        else "inconclusive_initial_A_not_acquired",
                        work=_work(decoded),
                        status="complete",
                    )
                )
            endpoint = {row["method"]: derived[i]["outcomes"][-1] for i, row in enumerate(rows)}
            comparisons = {
                f"gated_cpc-minus-{control}": {
                    metric: sum(
                        endpoint["gated_cpc"][context][metric] - endpoint[control][context][metric]
                        for context in ("A", "B", "C_train")
                    )
                    / 3
                    for metric in ("success", "mean_return")
                }
                for control in ("pc", "neutral_cpc")
            }
            all_completed()
            publication = derived, comparisons
    except Exception as error:
        report.update(status="incomplete", reason=f"{type(error).__name__}: {error}")
    finally:
        for arm in arms:
            try:
                _readback(
                    arm["row"], arm["source"], arm["states"][0][1], arm["stream"], arm["budget"]
                )
                arm["guard"]()
            except Exception as error:
                reason = f"readback {type(error).__name__}: {error}"
                arm["row"].update(status="incomplete", readback_error=reason)
                if arm["row"]["reason"] is None:
                    arm["row"]["reason"] = reason
                report["readback_error"] = reason
                report["status"] = "incomplete"
                if report["reason"] is None:
                    report["reason"] = reason
        for index, row in enumerate(rows):
            decoded = _restore_readouts(row, expected_slots[index], expected_final_slots[index])
            rendered.append(decoded)
            row["work"] = _work(decoded)
        if sampler is not None:
            report["observed_process_rss"] = asdict(sampler.snapshot())
        report["total_seconds"] = monotonic() - started
    if publication is not None and report["status"] == "running":
        try:
            all_completed()
        except Exception as error:
            report.update(
                status="incomplete", reason=f"publication {type(error).__name__}: {error}"
            )
        else:
            # Publish derived claims only after all external work/readbacks and
            # the final guard; no callback can alter a published acquisition flag.
            for row, value in zip(rows, publication[0], strict=True):
                row.update(value)
            report.update(status="complete", comparisons=publication[1])
    # Decoded payloads stay private through every callback and the final guard.
    # On rejection these are the original paid records, with status incomplete;
    # observed replacement bytes are never unpickled or accepted as repaired work.
    for row, decoded in zip(rows, rendered, strict=True):
        row["readouts"] = decoded["readouts"]
        row["final_evaluation"] = decoded["final_evaluation"]
    return report
