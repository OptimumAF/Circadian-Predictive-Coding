"""Matched conventional references using existing generic probe and coverage replay.

One common live checkpoint/data collection, two isolated arms, fixed roles and
separate permanent budgets. No optimizer, replay algorithm or model persistence.
"""

from dataclasses import replace
from time import monotonic
from typing import Any, Callable

from scripts.recurrent_readout_example import _budget_report, _check_paid_evidence
from src.adapters.minigrid_learning_pilot import _signature, _check_signature, _episode
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app.coupled_attribution import AttributionBatch
from src.adapters.minigrid_learnability import _collect_probe_sources
from src.adapters.minigrid_probe_rollout import ProbeRollouts
from src.adapters.minigrid_public_cue_recurrent_pilot import _typed_sources
from src.adapters.minigrid_public_cue_recurrent_learner import (
    PublicCueRecurrentLearner,
    PublicCueRecurrentPolicy,
    copy_snapshot,
)
from src.adapters.minigrid_return_fit import return_training_sources
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.probe_return_phase import paid_report_guard
from src.app.stream_rehearsal import StreamRehearsal, CoverageStreamReplayPolicy
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.demonstration_episode import EpisodeReplaySchema

ARMS = ("extra_F", "old_ABC")
KEYS = tuple(f"{context}-{cue}" for context in ("A", "B", "C_train") for cue in ("blue", "red"))
SCHEMA = EpisodeReplaySchema(148, 9, 7)
BANK_BYTES = 6 * (SCHEMA.encoded_dim + 1) * 8
RSS = 768 * 1024**2


def describe():
    return dict(
        definition="joint128_F48_vs_F16_ABC32_A16_v1",
        arms=list(ARMS),
        consent="--compare-rehearsal explicitly retains original54arrivedTRAIN rows for this disposable comparison",
        bank=dict(rows=54, encoded_array_bytes=BANK_BYTES, keys=list(KEYS)),
        replay="two old nine-row TRAIN episodes after every F wake, fixed six-key coverage rotation",
        probe_native_updates=48,
        probe_rows=432,
        return_updates=16,
        return_rows=144,
        primary_updates=dict(extra_F=48, old_ABC=16),
        auxiliary_updates=dict(extra_F=0, old_ABC=32),
        old_ABC_rows=dict(F=144, old_train=288),
        extra_F_rows=dict(F=432, old_train=0),
        teacher_points=dict(extra_F=[0, 12, 24, 48], old_ABC=[0, 4, 8, 16]),
        native_points=[128, 140, 152, 176, 192],
        endpoint_seconds=120,
        phase_seconds=300,
        native_games="F2probe_test+alloldABC6TRAIN after EACH F/A phase, botharms",
        final_available=False,
        promotion=False,
        limitation="equal native updates/rows, not allcompute/time, causal benefit or CPC advantage",
    )


def _bank(old_sources, consent):
    if type(consent) is not bool or consent is not True:
        raise ValueError("matched rehearsal requires explicit old TRAIN retention/replay consent")
    _validate_old_sources(old_sources)
    packets = _typed_sources(tuple(_episode(p) for p in old_sources))
    return tuple(replace(_episode(p), context=key) for p, key in zip(packets, KEYS, strict=True))


def _validate_old_sources(old_sources):
    if type(old_sources) is not tuple or len(old_sources) != 6:
        raise ValueError("matched rehearsal requires all six original arrived TRAIN sources")
    if any(
        type(p) is not AttributionBatch
        or p.role != "train"
        or p.episode is None
        or p.episode.role != "train"
        for p in old_sources
    ):
        raise ValueError("matched bank contains only original TRAIN episodes")


def build_rehearsal(learner, wake, replay, old_sources, *, consent):
    bank = _bank(old_sources, consent)
    if (
        type(learner) is not PublicCueRecurrentLearner
        or learner.policy != PublicCueRecurrentPolicy()
    ):
        raise ValueError("matched rehearsal requires the original public-cue recurrent learner")
    return StreamRehearsal(
        learner,
        wake,
        CoverageStreamReplayPolicy(1, 18, ReplayRetentionBudget(54, BANK_BYTES), KEYS),
        replay_budget=replay,
        total_updates=16,
        input_dim=148,
        episode_schema=SCHEMA,
        fixed_training_episodes=bank,
        wake_role="probe_train",
    )


def _budget(updates, memory, *, replay=False):
    limits = ToyExecutionBudget(
        max_training_updates=None if replay else updates,
        max_replay_examples=288 if replay else None,
        max_wall_seconds=300,
        max_process_rss_bytes=RSS,
    )
    result = ToyBudgetSession(limits, monotonic)
    result.attach_memory(memory)
    return result


def run_comparison(
    source,
    state,
    old_sources,
    *,
    check_boundary,
    memory,
    old_training_consent=False,
    cold_source=False,
):
    report: dict[str, Any] = dict(
        status="incomplete",
        reason=None,
        collection=dict(reports=[]),
        arms=[],
        consent=old_training_consent,
    )
    current: dict[str, Any] | None = None
    try:
        if type(cold_source) is not bool or cold_source and old_training_consent is not False:
            raise ValueError("cold control requires its exact separate no-replay mode")
        if cold_source:
            _validate_old_sources(old_sources)  # Monitors/A acquisition, no replay bank.
        else:
            _bank(old_sources, old_training_consent)
        source_updates = 0 if cold_source else 128
        saved = copy_snapshot(state)
        if (
            type(source) is not PublicCueRecurrentLearner
            or saved.policy != PublicCueRecurrentPolicy()
            or saved.failed_work is not None
            or (saved.completed_updates, saved.training_examples)
            != (source_updates, 9 * source_updates)
            or any(
                (h.completed_updates, h.training_examples)
                != (source_updates // 2, 9 * source_updates // 2)
                for h in saved.heads
            )
        ):
            raise ValueError("probe comparison requires its original cold0 or joint128 state")
        if cold_source and not _same_state(source.snapshot_state(), saved):
            raise ValueError("cold control requires the exact original cold source state")
        collection = report["collection"]
        records = report["arms"]
        sealed: list[tuple[dict[str, Any], Callable[[], None]]] = []
        collection_check = None
        signatures: list[tuple[Any, Any]] = []
        games = None

        def guard():
            check_boundary()
            if report["consent"] is not old_training_consent:
                raise ValueError("matched original TRAIN consent evidence changed")
            if report["arms"] is not records or len(records) != len(sealed) + (current is not None):
                raise ValueError("matched arm evidence list changed")
            for actual, (owner, proof) in zip(records, sealed):
                _check_paid_evidence(actual, owner, proof, "preceding matched arm changed")
            if current is not None and records[-1] is not current:
                raise ValueError("active matched arm replaced")
            if collection_check is not None:
                _check_paid_evidence(
                    report["collection"],
                    collection,
                    collection_check,
                    "matched source collection changed",
                )
            if not all(_check_signature(data, sig) for data, sig in signatures):
                raise ValueError("matched original inputs changed")
            if games is not None:
                games.check()
                if current is None:
                    raise ValueError("matched observer lost its active arm")
                if current["endpoints"] is not games.records:
                    raise ValueError("matched endpoint evidence list replaced")

        training, evaluation = _collect_probe_sources(
            guard, collection["reports"], distinct_probe=True
        )
        roster = (
            ("probe_train", "blue"),
            ("probe_train", "red"),
            ("probe_test", "blue"),
            ("probe_test", "red"),
        )
        rows = collection["reports"]
        if len(rows) != 4 or any(
            r.get("status") != "complete"
            or r.get("closed") is not True
            or r.get("environment_steps") != 9
            or (r.get("role"), r.get("cue")) != expected
            for r, expected in zip(rows, roster, strict=True)
        ):
            raise ValueError("matched comparison requires complete closed original F sources")
        collection_check, _ = paid_report_guard(collection)
        monitors = old_sources[:2]
        templates = return_training_sources(monitors)
        signatures = [
            (data, _signature(data)) for data in (training, evaluation, old_sources, templates)
        ]
        for arm in ("cold_F",) if cold_source else ARMS:
            guard()  # The preceding arm must qualify before the next is even allocated.
            current = dict(
                arm=arm,
                status="incomplete",
                probe=None,
                endpoints=[],
                replay=None,
                probe_budget=None,
                return_budget=None,
                replay_budget=None,
            )
            records.append(current)
            games = ProbeRollouts(
                source_updates, public_cue=True, joint_public_cue=True, comparison_arm=arm
            )
            current["endpoints"] = games.records
            wake = _budget(16 if arm == "old_ABC" else 48, memory)
            returned = _budget(16, memory)
            replay = _budget(None, memory, replay=True) if arm == "old_ABC" else None
            helper = None

            def factory(learner, budget, count):
                nonlocal helper
                guard()
                if budget is not wake or count != 16:
                    raise ValueError("matched rehearsal lost its original F16 authority")
                helper = build_rehearsal(
                    learner, budget, replay, old_sources, consent=old_training_consent
                )
                guard()
                return helper

            try:
                current["probe"] = run_fresh_task_probe(
                    source,
                    state,
                    training,
                    evaluation,
                    budget=wake,
                    checkpoint_updates=(0, 4, 8, 16) if arm == "old_ABC" else (0, 12, 24, 48),
                    decision_diagnostics=True,
                    training_readouts=monitors,
                    return_training=templates,
                    return_budget=returned,
                    endpoint_observer=games.observe_qualified,
                    check_boundary=guard,
                    **(
                        dict(extra_f_exposure=True)
                        if arm != "old_ABC"
                        else dict(rehearsal_factory=factory)
                    ),
                )
                guard()
                result = current["probe"]
                if (
                    result.get("status") != "complete"
                    or result.get("completed_updates") != (16 if arm == "old_ABC" else 48)
                    or result.get("return_phase", {}).get("status") != "complete"
                    or result["return_phase"].get("completed_updates") != 16
                    or (wake.updates_completed, returned.updates_completed)
                    != (16 if arm == "old_ABC" else 48, 16)
                    or len(games.records) != 2
                    or any(r["status"] != "complete" for r in games.records)
                ):
                    raise ValueError("matched probe/return or endpoint did not complete")
                if helper is not None:
                    current["replay"] = helper.report
                    if any(
                        current["replay"].get(k) != v
                        for k, v in dict(
                            charged_exposures=288,
                            attempted_updates=32,
                            returned_updates=32,
                            committed_updates=32,
                        ).items()
                    ):
                        raise ValueError("matched old TRAIN replay dose differs")
                current["status"] = "complete"
            finally:
                current["probe_budget"] = _budget_report(wake)
                current["return_budget"] = _budget_report(returned)
                current["replay_budget"] = _budget_report(replay)
                if helper is not None and current["replay"] is None:
                    try:
                        current["replay"] = helper.report
                    except Exception as readback:
                        current["replay_readback_error"] = f"{type(readback).__name__}: {readback}"
            proof, _ = paid_report_guard(current)
            sealed.append((current, proof))
            current, games = None, None
            guard()
        report["status"] = "complete"
    except Exception as error:
        report["reason"] = f"{type(error).__name__}: {error}"
    return report
