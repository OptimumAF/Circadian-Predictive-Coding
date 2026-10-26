"""Fixed synthetic categorical packets through the ordinary attribution stream.

Inputs are the frozen local protocol; outputs retain every method/mode/checkpoint
and incurred work. These packets are synthetic, not live environment rewards or
autonomous episodes. No new trainer, loss, promotion or tuning policy lives here.
"""

from dataclasses import asdict
from time import monotonic
from typing import Any

from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.adapters.torch_feature_rehearsal_pilot import _observation, _same_state
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.stream_rehearsal import StreamReplayPolicy, StreamRehearsal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    EpisodeReplaySchema,
    episode_arrays,
)
from src.core.shared_replay_schedule import ReplayAllocation
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch

METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
POINTS = (0, 1, 2, 3, 4)
MAX_RSS = 768 * 1024**2
MAX_BYTES = 2 * 1024**2


def replay_policy() -> StreamReplayPolicy:
    return StreamReplayPolicy(
        2,
        3,
        ReplayRetentionBudget(6, 1936),
        ReplayAllocation(ReplayRetentionBudget(3, 968), ReplayRetentionBudget(3, 968)),
        53,
    )


def describe_protocol() -> dict[str, Any]:
    return dict(
        protocol_id="native_categorical_complete_packets_v1",
        classification="synthetic_categorical_stream_engineering_not_environment_rollout",
        methods=list(METHODS),
        replay_modes=[False, True],
        seed=17,
        feature_dim=2,
        hidden_dim=3,
        num_classes=3,
        task_sequence=["A", "B", "C", "A"],
        checkpoint_updates=list(POINTS),
        label_delay=1,
        packet_steps=3,
        episode_schema=asdict(EpisodeReplaySchema(2, 3, 3)),
        training_x=[-0.75, -0.25, 0.25, 0.75],
        training_y=dict(A=-0.75, B=0.0, C=0.75),
        final_x=[-0.6, -0.2, 0.2, 0.6],
        final_y=dict(A=-0.5, B=0.25, C=0.5),
        actions=dict(A=[0, 1, 2], B=[1, 2, 0], C=[2, 0, 1]),
        transition_metadata="synthetic contiguous observations/rewards0,0,1/last terminal; NOT live measured reward",
        native_defaults=dict(
            learning_rate=0.03, inference_steps=2, inference_learning_rate=0.04, momentum=0.9
        ),
        replay=asdict(replay_policy()),
        encoded_packet_array_bytes=968,
        pause_updates=[2],
        metrics=dict(
            ce="mean -log clip(p_true,1e-12,1); lower",
            accuracy="argmax first index on ties; higher",
            row_sum_atol=1e-6,
        ),
        directions=dict(
            new_B_gain="B CE0 minus CE2",
            new_C_gain="C CE0 minus CE3",
            retention_loss="A CE3 minus CE1",
            return_gain="A CE3 minus CE4",
        ),
        planned_work=dict(
            wake_updates=32,
            replay_updates=8,
            native_train_calls=40,
            training_row_exposures=120,
            pc_family_steps=30,
            bp_steps=10,
            score_forwards=120,
            resumes=8,
            probes=0,
        ),
        resources=dict(
            cpu_threads=1,
            gpu=0,
            wall_seconds=60,
            process_rss_bytes=MAX_RSS,
            storage_bytes=MAX_BYTES,
        ),
        final_isolation="all training completes before detached checkpoint scoring; no feedback or selection",
        limitations="one seed/synthetic packets; no environment interaction, acquisition from failure, recurrent RL, held-out layouts, equal off/on work or circadian advantage",
    )


def _packet(identity: str, context: str, role: str) -> AttributionBatch:
    final = role == "final_test"
    x = (-0.6, -0.2, 0.2, 0.6) if final else (-0.75, -0.25, 0.25, 0.75)
    y = ({"A": -0.5, "B": 0.25, "C": 0.5} if final else {"A": -0.75, "B": 0.0, "C": 0.75})[context]
    offset = ("A", "B", "C").index(context)
    transitions = tuple(
        DemonstrationTransition(
            step,
            (x[step], y),
            (step + offset) % 3,
            float(step == 2),
            (x[step + 1], y),
            step == 2,
            3,
        )
        for step in range(3)
    )
    episode = DemonstrationEpisode(identity, context, role, transitions, 3)
    features, targets = episode_arrays(episode)
    return AttributionBatch(
        role,
        context,
        tuple(f"{identity}:step:{s}" for s in range(3)),
        features,
        targets,
        episode,
        3,
    )


def build_stream() -> tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]]:
    training = tuple(
        _packet(f"categorical-train-{i}", context, "train")
        for i, context in enumerate(("A", "B", "C", "A"))
    )
    final = tuple(
        _packet(f"categorical-final-{context}", context, "final_test")
        for context in ("A", "B", "C")
    )
    return training, final


def _session(
    deadline: float, sampler: ProcessRssSampler, *, replay: bool = False
) -> ToyBudgetSession:
    budget = ToyExecutionBudget(
        max_training_updates=None if replay else 4,
        max_replay_examples=6 if replay else None,
        max_wall_seconds=max(0.0, deadline - monotonic()),
        max_process_rss_bytes=MAX_RSS,
    )
    session = ToyBudgetSession(budget, monotonic)
    session.attach_memory(sampler)
    session.before_final()
    return session


def _continue(stream: AttributionStream) -> dict[str, Any]:
    stream.run_until(2)
    runtime = stream._runtime
    helper = stream._rehearsal
    old_candidate, old_inbox = runtime._candidate, runtime._inbox
    old_buffer = None if helper is None else helper._buffer
    budgets = tuple(
        (id(s), id(s.budget), s.started_at, s.updates_completed, s.replay_examples_completed)
        for s in stream._sessions()
    )
    token = stream.pause()
    stream.resume(token)
    current = tuple(
        (id(s), id(s.budget), s.started_at, s.updates_completed, s.replay_examples_completed)
        for s in stream._sessions()
    )
    return dict(
        wake_updates=2,
        runtime_preserved=stream._runtime is runtime,
        original_budgets_and_charges=budgets == current,
        candidate_reconstructed=runtime._candidate is not old_candidate,
        inbox_reconstructed=runtime._inbox is not old_inbox,
        buffer_reconstructed=helper is None or helper._buffer is not old_buffer,
        pause_consumed=token._owner is None and not token._snapshot,
    )


def _outcomes(report: dict[str, Any]) -> dict[str, float]:
    points = {p["updates"]: p["scores"] for p in report["checkpoints"]}
    return dict(
        new_B_gain=points[0]["B"]["ce"] - points[2]["B"]["ce"],
        new_C_gain=points[0]["C"]["ce"] - points[3]["C"]["ce"],
        retention_loss=points[3]["A"]["ce"] - points[1]["A"]["ce"],
        return_gain=points[3]["A"]["ce"] - points[4]["A"]["ce"],
    )


def _run_arm(
    row: dict[str, Any],
    training: tuple[AttributionBatch, ...],
    final: tuple[AttributionBatch, ...],
    deadline: float,
    sampler: ProcessRssSampler,
) -> None:
    wake, replay, helper, source, before, stream = None, None, None, None, None, None
    try:
        wake = _session(deadline, sampler)
        source = TorchFeatureLearner(TorchFeaturePolicy(row["method"], 2, 3, 17, num_classes=3))
        before = source.snapshot_state()
        if row["replay_enabled"]:
            replay = _session(deadline, sampler, replay=True)
            helper = StreamRehearsal(
                source,
                wake,
                replay_policy(),
                replay_budget=replay,
                total_updates=4,
                input_dim=2,
                episode_schema=EpisodeReplaySchema(2, 3, 3),
            )

        def observe(updates: int, state: Any, batch: AttributionBatch | None) -> None:
            row["observations"].append(_observation(updates, state))

        stream = AttributionStream(
            source,
            training,
            final,
            checkpoint_updates=POINTS,
            label_delay=1,
            budget=wake,
            observe_checkpoint=observe,
            rehearsal=helper,
        )
        row["continuation"] = _continue(stream)
        stream.run_until(4)
        result = stream.finish()
        row.update(status="complete", reason=None, stream_result=result, outcomes=_outcomes(result))
    except Exception as error:
        row.update(status="failed", reason=f"{type(error).__name__}: {error}")
    finally:
        row["wake_updates"] = 0 if wake is None else wake.updates_completed
        row["replay_charged_examples"] = 0 if replay is None else replay.replay_examples_completed
        try:
            if helper is not None:
                row["rehearsal"] = helper.report
                row["final_retention"] = asdict(helper.retention)
            if source is not None and before is not None:
                row["source_unchanged"] = _same_state(before, source.snapshot_state())
            if stream is not None and before is not None:
                row["actor_unchanged"] = _same_state(
                    before, stream._runtime.actor.snapshot_state().state
                )
        except Exception as error:
            secondary = f"readback {type(error).__name__}: {error}"
            row["readback_error"] = secondary
            if row["status"] != "failed":
                row.update(status="failed", reason=secondary)


def _valid_arm(row: dict[str, Any]) -> bool:
    if row["status"] != "complete":
        return False
    updates = [0, 1, 3, 4, 6] if row["replay_enabled"] else list(POINTS)
    continuation = row["continuation"]
    valid = (
        row["wake_updates"] == 4
        and row["source_unchanged"] is True
        and row["actor_unchanged"] is True
        and row["stream_result"]["score_forward_calls"] == 15
        and row["stream_result"]["num_classes"] == 3
        and [p["native_updates"] for p in row["observations"]] == updates
        and [p["native_training_examples"] for p in row["observations"]] == [3 * u for u in updates]
        and continuation is not None
        and all(v is True for k, v in continuation.items() if k != "wake_updates")
    )
    if not row["replay_enabled"]:
        return valid and row["replay_charged_examples"] == 0
    events = row["rehearsal"]["events"]
    return bool(
        valid
        and row["replay_charged_examples"] == 6
        and len(events) == 2
        and row["final_retention"]["example_count"] <= 6
        and row["final_retention"]["retained_bytes"] <= 1936
        and all(
            e["status"] == "complete"
            and e["charged_exposures"] == 3
            and e["attempted_updates"] == e["returned_updates"] == e["committed_updates"] == 1
            and e["receipt"] is not None
            and len(e["selected_episodes"]) == 1
            and e["selected_episodes"][0]["num_classes"] == 3
            for e in events
        )
    )


def run_pilot() -> dict[str, Any]:
    rows: list[dict[str, Any]] = [
        dict(
            method=method,
            replay_enabled=enabled,
            status="not_run",
            reason="not reached",
            wake_updates=0,
            replay_charged_examples=0,
            observations=[],
            continuation=None,
            stream_result=None,
            outcomes=None,
            rehearsal=None,
            final_retention=None,
            source_unchanged=None,
            actor_unchanged=None,
            readback_error=None,
        )
        for enabled in (False, True)
        for method in METHODS
    ]
    errors, memory, sampler = [], None, None
    started = monotonic()
    try:
        sampler = ProcessRssSampler()
        with sampler:
            torch = require_torch()
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            training, final = build_stream()
            for row in rows:
                if monotonic() - started >= 60:
                    break
                _run_arm(row, training, final, started + 60, sampler)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        if sampler is not None:
            memory = asdict(sampler.snapshot())
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    complete = all(_valid_arm(row) for row in rows)
    plans = [
        [e["selected_ids"] for e in r["rehearsal"]["events"]]
        for r in rows
        if r["rehearsal"] is not None
    ]
    shared = len(plans) == 4 and all(p == plans[0] for p in plans)
    neutral = complete and all(
        rows[i + 1]["stream_result"]["checkpoints"] == rows[i + 2]["stream_result"]["checkpoints"]
        and all(
            a["parameters"] == b["parameters"]
            for a, b in zip(rows[i + 1]["observations"], rows[i + 2]["observations"], strict=True)
        )
        for i in (0, 4)
    )
    initial = complete and all(
        r["observations"][0]["parameters"] == rows[0]["observations"][0]["parameters"] for r in rows
    )
    elapsed = monotonic() - started
    complete = bool(
        complete
        and shared
        and neutral
        and initial
        and not errors
        and elapsed <= 60
        and memory is not None
        and memory["peak_bytes"] <= MAX_RSS
    )
    return dict(
        protocol=describe_protocol(),
        status="complete" if complete else "incomplete",
        runs=rows,
        global_errors=errors,
        total_seconds=elapsed,
        observed_process_rss=memory,
        shared_replay_plans_exact=shared,
        pc_neutral_exact=neutral,
        shared_initial_parameters_exact=initial,
        scientific_advantage_verified=False,
        full_R42_complete=False,
    )
