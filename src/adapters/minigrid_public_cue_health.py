"""Compose existing native activation/health diagnostics on six TRAIN packets.

Inputs are an original public-cue source/checkpoint, its caller guard and born-once
readout budget. Outputs are bounded detached metrics and paid work, never hidden
arrays or updated learners. Caller guards own actor, RNG and collection provenance.
"""

from copy import deepcopy
from dataclasses import asdict
import pickle
from time import monotonic
from typing import Any, Callable

import numpy as np

from src.adapters import minigrid_learning_pilot as shared
from src.adapters import minigrid_public_cue_recurrent_learner as public
from src.adapters.minigrid_policy_evaluation import _return_budget_guard
from src.adapters.torch_activation_diagnostics import read_native_activations
from src.app.coupled_attribution import AttributionBatch, _same_value, _validate_batch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.learnability_diagnostics import diagnose_activations


class ActivationHealthFailed(RuntimeError):
    """Detached last verified paid prefix, including uncertain failed work."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = deepcopy(report)
        super().__init__(str(report["reason"]))


def _sources(sources: tuple[AttributionBatch, ...]) -> None:
    if type(sources) is not tuple or len(sources) != 6:
        raise ValueError("public health requires six ordered original TRAIN packets")
    seen: set[str] = set()
    for i, packet in enumerate(sources):
        _validate_batch(packet, "train", seen)
        context, cue = ("A", "B", "C_train")[i // 2], ("blue", "red")[i % 2]
        if (
            packet.context != context
            or packet.num_classes != 7
            or packet.features.shape != (9, 148)
            or packet.episode is None
            or packet.episode.episode_id != f"minigrid-source-{context}-{cue}"
            or public._cue(public._raw(packet.features)[0]) != i % 2
        ):
            raise ValueError("public health requires original raw148/K7 TRAIN order and cues")
        public._consistent_cues(packet.features, i % 2)


def read_checkpoint_health(
    source: public.PublicCueRecurrentLearner,
    state: public.PublicCueRecurrentSnapshot,
    *,
    updates: int,
    sources: tuple[AttributionBatch, ...],
    budget: ToyBudgetSession,
    check_boundary: Callable[[], None],
    recurrent_replay: str | None = None,
) -> dict[str, Any]:
    """Read existing zero-reset recurrent health; publish no learner updates.

    Native forward attempts inside a failed reader are unknown: record admitted
    call/row upper bounds and only count complete returned readers as completed.
    Snapshot counts here exclude work performed inside the caller's own guard.
    """
    if type(source) is not public.PublicCueRecurrentLearner or not callable(check_boundary):
        raise ValueError("public health requires its exact source and original caller guard")
    if type(updates) is not int or updates not in (0, 32, 64, 96, 128):
        raise ValueError("public health requires a declared original checkpoint")
    expected_updates = updates
    if recurrent_replay is not None:
        from src.adapters import minigrid_recurrent_replay as replay

        replay.validate_mode(recurrent_replay)
        expected_updates = replay.native_updates(updates)
    _sources(sources)  # Whole-role/geometry admission before native validation.
    source_owner, policy_owner = source._state, source._policy
    signature = shared._signature(sources)
    budget_guard = _return_budget_guard(budget)
    limits = ToyExecutionBudget(
        max_training_updates=0,
        max_replay_examples=0,
        max_wall_seconds=200,
        max_process_rss_bytes=768 * 1024**2,
    )
    if not _same_value(vars(budget.budget), vars(limits)):
        raise ValueError("public health requires original200s/768MiB zero-work readout budget")
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="public_cue_TRAIN_ordered_recurrent_activation_health_v1",
        updates=updates,
        native_updates=expected_updates,
        status="running",
        reason=None,
        source_unchanged=None,
        state_unchanged=None,
        scores=[],
        work=dict(
            public_snapshot_copies_attempted=0,
            public_snapshot_copies_completed=0,
            source_snapshot_reads_attempted=0,
            source_snapshot_reads_completed=0,
            activation_reads_attempted=0,
            activation_reads_completed=0,
            forward_calls_attempted_upper_bound=0,
            forward_rows_attempted_upper_bound=0,
            forward_calls_completed=0,
            forward_rows_completed=0,
            discarded_output_projections_completed=0,
            activation_array_bytes_returned=0,
            svd_calls_attempted=0,
            svd_calls_completed=0,
            optimizer_steps=0,
            published_updates=0,
            failure_work_may_be_unobserved=False,
            seconds=0.0,
        ),
    )
    paid = pickle.dumps(report, protocol=5)

    def verify() -> None:
        if pickle.dumps(report, protocol=5) != paid:
            raise ValueError("preceding public health evidence changed")

    def capture() -> None:
        nonlocal paid
        paid = pickle.dumps(report, protocol=5)

    def charge(**values: int) -> None:
        verify()
        for key, value in values.items():
            report["work"][key] += value
        capture()

    def boundary() -> None:
        verify()
        budget_guard()
        check_boundary()
        budget_guard()
        budget.before_final()
        budget_guard()
        check_boundary()
        budget_guard()
        verify()
        if not shared._check_signature(sources, signature):
            raise ValueError("public health original TRAIN source changed")

    def snapshot() -> public.PublicCueRecurrentSnapshot:
        charge(source_snapshot_reads_attempted=1)
        value = source.snapshot_state()
        charge(source_snapshot_reads_completed=1)
        return value

    try:
        boundary()
        charge(public_snapshot_copies_attempted=1)
        owned = public.copy_snapshot(state)
        charge(public_snapshot_copies_completed=1)
        if (
            owned.policy != public.PublicCueRecurrentPolicy()
            or owned.failed_work is not None
            or any(
                (h.completed_updates, h.training_examples)
                != (expected_updates // 2, 9 * expected_updates // 2)
                for h in owned.heads
            )
        ):
            raise ValueError("public health policy/failed state or exact head clocks differ")
        boundary()
        before = snapshot()
        if (
            before.policy != owned.policy
            or before.failed_work is not None
            or before.completed_updates != 0
            or before.training_examples != 0
        ):
            raise ValueError("public health requires original unchanged cold TRAIN source")

        def check() -> None:
            boundary()
            if not _same_value(owned, state) or not _same_value(before, snapshot()):
                raise ValueError("public health original source or checkpoint changed")
            boundary()  # Snapshot callbacks cannot silently replace preceding evidence.
            if not _same_value(owned, state):
                raise ValueError("public health checkpoint changed during final callback")
            if (
                source._state is not source_owner
                or source._policy is not policy_owner
                or not _same_value(source._state, before)
                or not _same_value(source._policy, before.policy)
            ):
                raise ValueError("public health source changed during final callback")

        check()
        for packet in sources:
            check()
            selected = public._cue(packet.features[0])
            features = packet.features.copy()
            charge(
                activation_reads_attempted=1,
                forward_calls_attempted_upper_bound=9,
                forward_rows_attempted_upper_bound=9,
            )
            head = deepcopy(owned.heads[selected])
            hidden = read_native_activations(head, features)
            charge(
                activation_reads_completed=1,
                forward_calls_completed=9,
                forward_rows_completed=9,
                discarded_output_projections_completed=9,
            )
            check()
            if not _same_value(head, owned.heads[selected]) or not np.array_equal(
                features, packet.features
            ):
                raise ValueError("public health owned head or features changed during diagnostic")
            if (
                type(hidden) is not np.ndarray
                or hidden.dtype != np.float64
                or hidden.shape != (9, 8)
            ):
                raise ValueError("public health requires nine-row/eight-unit float64 hidden states")
            if not np.isfinite(hidden).all() or np.any(np.abs(hidden) > 1):
                raise ValueError("public health requires finite tanh states in [-1, 1]")
            charge(activation_array_bytes_returned=hidden.nbytes, svd_calls_attempted=1)
            metrics = diagnose_activations((hidden,))
            charge(svd_calls_completed=1)
            check()
            assert packet.episode is not None
            report["scores"].append(
                dict(
                    context=packet.context,
                    head=selected,
                    episode_id=packet.episode.episode_id,
                    metrics=[asdict(value) for value in metrics],
                )
            )
            capture()
        check()
        report["status"] = "complete"
        report["source_unchanged"] = report["state_unchanged"] = True
        report["work"]["seconds"] = monotonic() - started
        capture()
    except Exception as error:
        # Why: never reseal callback-corrupted fields or discard preceding charges.
        failed = pickle.loads(paid)
        failed.update(status="incomplete", reason=f"{type(error).__name__}: {error}")
        failed["work"].update(failure_work_may_be_unobserved=True, seconds=monotonic() - started)
        raise ActivationHealthFailed(failed) from error
    return deepcopy(report)
