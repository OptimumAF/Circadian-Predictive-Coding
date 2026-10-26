"""Bounded CPU fixed-feature rehearsal using the existing native Torch heads.

Owns a fixed engineering protocol and reporting, not a backbone, task selector,
probe, or whole-stream checkpoint format. The existing stream runner owns label
arrival, candidate isolation, replay admission and deferred final scoring.
"""

from dataclasses import asdict, fields, is_dataclass
import pickle
from time import monotonic
from typing import Any

import numpy as np

from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.app.coupled_attribution import AttributionBatch, AttributionStream, run_attribution_stream
from src.app.coupled_attribution_summary import METHODS
from src.app.stream_rehearsal import StreamRehearsal, StreamReplayPolicy
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.shared_replay_schedule import ReplayAllocation
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    EpisodeReplaySchema,
    episode_arrays,
)
from src.core.line_navigation import LineNavigation
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch

PROTOCOL_ID = "native_torch_fixed_feature_rehearsal_v1"
CHECKPOINTS = (0, 2, 4, 6)
WALL_SECONDS = 60.0
PROCESS_RSS_BYTES = 768 * 1024**2
STORAGE_BYTES = 2 * 1024**2
PARAMETERS = ("weight_feature_hidden", "bias_hidden", "weight_hidden_output", "bias_output")


def replay_policy() -> StreamReplayPolicy:
    return StreamReplayPolicy(
        2,
        1,
        ReplayRetentionBudget(4, 96),
        ReplayAllocation(ReplayRetentionBudget(2, 48), ReplayRetentionBudget(2, 48)),
        53,
    )


def describe_protocol() -> dict[str, Any]:
    """Describe without importing Torch, constructing data, or training."""
    return dict(
        protocol_id=PROTOCOL_ID,
        classification="fixed_cpu_feature_engineering_demo",
        methods=list(METHODS),
        replay_modes=[False, True],
        seed=17,
        feature_dim=2,
        hidden_dim=3,
        classes=2,
        learning_rate=0.03,
        inference_steps=2,
        inference_learning_rate=0.04,
        bp_momentum=0.9,
        training_rng=7301,
        heldout_rng=9101,
        phase_lengths=[2, 2, 2],
        phase_contexts=["A", "B", "A"],
        rows_per_wake=2,
        heldout_rows_per_context=8,
        checkpoint_updates=list(CHECKPOINTS),
        label_delay=1,
        replay=asdict(replay_policy()),
        metrics=dict(bce="clip1e-12; lower", accuracy="probability>=0.5; higher"),
        directions=dict(
            new_B_gain="B BCE at0 minus4",
            retention_loss="A BCE at4 minus2",
            return_gain="A BCE at4 minus6",
        ),
        planned_work=dict(
            wake_updates=48,
            replay_updates=12,
            native_train_calls=60,
            training_row_exposures=108,
            score_forwards=64,
            pc_family_train_steps=45,
            bp_optimizer_steps=15,
            probes=0,
            svd=0,
        ),
        resources=dict(
            cpu_threads=1,
            gpu=0,
            wall_seconds=WALL_SECONDS,
            process_rss_bytes=PROCESS_RSS_BYTES,
            storage_bytes=STORAGE_BYTES,
        ),
        budget_scope="new Torch engineering route including optional runtime overhead; existing scientific pilot limits unchanged",
        clock_semantics="separate permanent app wake/replay charges; every external replay call also advances native head update, age and chemistry clocks",
        limitations="fixed features/one seed; BP CE and native PC energy differ; replay on/off work differs; no causal replay estimate, equal compute, circadian advantage, backbone, probe, durable or whole-stream resume claim",
    )


def build_stream() -> tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]]:
    rng = np.random.default_rng(7301)
    training = []
    for index in range(6):
        context = "B" if 2 <= index < 4 else "A"
        features = rng.uniform(-1, 1, (2, 2))
        targets = (features[:, [1 if context == "B" else 0]] >= 0).astype(np.float64)
        training.append(
            AttributionBatch(
                "train",
                context,
                tuple(f"torch-train-{index}-{i}" for i in range(2)),
                features,
                targets,
            )
        )
    rng = np.random.default_rng(9101)
    final = []
    for axis, context in enumerate(("A", "B")):
        features = rng.uniform(-1, 1, (8, 2))
        final.append(
            AttributionBatch(
                "final_test",
                context,
                tuple(f"torch-final-{context}-{i}" for i in range(8)),
                features,
                (features[:, [axis]] >= 0).astype(np.float64),
            )
        )
    return tuple(training), tuple(final)


def describe_continuation_protocol() -> dict[str, Any]:
    protocol = describe_protocol()
    protocol.update(
        protocol_id="native_torch_fixed_feature_continuation_v1",
        classification="owner_local_whole_stream_continuation_engineering",
        pause_updates=[2, 4],
        planned_pauses=16,
        planned_resumes=16,
        restore_scope="new native candidate/inbox/buffer; same runtime/actor/original budgets, clocks, receipts; no process restart or durability",
    )
    return protocol


def trajectory_replay_policy() -> StreamReplayPolicy:
    """Quotas count transitions; bytes include the complete encoded episode."""
    return StreamReplayPolicy(
        2,
        2,
        ReplayRetentionBudget(4, 1808),
        ReplayAllocation(ReplayRetentionBudget(2, 904), ReplayRetentionBudget(2, 904)),
        53,
    )


def describe_trajectory_protocol() -> dict[str, Any]:
    protocol = describe_continuation_protocol()
    protocol.update(
        protocol_id="native_torch_complete_demonstration_rehearsal_v1",
        trajectory_enabled=True,
        classification="complete_two_action_episode_behavior_cloning_engineering",
        replay=asdict(trajectory_replay_policy()),
        episode_schema=asdict(EpisodeReplaySchema(2, 2)),
        environment="integer line -2..2, action0 left/action1 right, terminal at goal",
        training_episodes="start0, goals+2(A)/-2(B), A,A,B,B,A,A; two legal unit steps",
        final_episodes="A start-1 goal+1; B start+1 goal-1; unseen goal cues; final_test only",
        observations="position/2, goal/2; action labels drive unchanged native hard-binary objectives",
        transition_metadata="ordered step/state/action/reward/next-state/terminal retained whole; reward/next-state do not alter objective",
        collection="scripted demonstrator reward0 intermediate/reward1 terminal; no autonomous learner rollout claim",
        training_rng=None,
        heldout_rng=None,
        heldout_rows_per_context=2,
        encoded_episode_array_bytes=904,
        total_collected_transitions=16,
        limitations="two-action feedforward behavior cloning/one seed; no recurrent RL, TD target, autonomous return, generalization, causal replay or circadian advantage claim; same-mode matched memory/exposure, replayon/off unequal work; no durable restart",
    )
    protocol["planned_work"] = dict(protocol["planned_work"], training_row_exposures=120)
    return protocol


def _demonstration(
    episode_id: str, context: str, role: str, start: int, goal: int
) -> AttributionBatch:
    environment = LineNavigation(start, goal)
    transitions = []
    for step in range(2):
        observation = environment.observation
        action = int(goal > observation[0] * 2)
        result = environment.step(action)
        transitions.append(
            DemonstrationTransition(
                step,
                observation,
                action,
                result.reward,
                result.observation,
                result.terminal,
            )
        )
    episode = DemonstrationEpisode(episode_id, context, role, tuple(transitions))
    features, targets = episode_arrays(episode)
    return AttributionBatch(
        role,
        context,
        tuple(f"{episode_id}:step:{t.step}" for t in episode.transitions),
        features,
        targets,
        episode,
    )


def build_trajectory_stream() -> tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]]:
    training = tuple(
        _demonstration(f"line-train-{index}", context, "train", 0, 2 if context == "A" else -2)
        for index, context in enumerate(("A", "A", "B", "B", "A", "A"))
    )
    final = (
        _demonstration("line-final-A", "A", "final_test", -1, 1),
        _demonstration("line-final-B", "B", "final_test", 1, -1),
    )
    return training, final


def _continued_stream(
    row: dict[str, Any],
    learner: TorchFeatureLearner,
    training: tuple[AttributionBatch, ...],
    final: tuple[AttributionBatch, ...],
    wake: ToyBudgetSession,
    rehearsal: StreamRehearsal | None,
    observe: Any,
) -> dict[str, Any]:
    stream = AttributionStream(
        learner,
        training,
        final,
        checkpoint_updates=CHECKPOINTS,
        label_delay=1,
        budget=wake,
        rehearsal=rehearsal,
        observe_checkpoint=observe,
    )
    for point in (2, 4):
        stream.run_until(point)
        runtime = stream._runtime
        candidate, inbox = runtime._candidate, runtime._inbox
        buffer = None if rehearsal is None else rehearsal._buffer
        buffer_bytes = (
            pickle.dumps(buffer)
            if row.get("trajectory_enabled", False) and buffer is not None
            else None
        )
        row["pause_attempts"] += 1
        token = stream.pause()
        snapshot = token.snapshot_state()
        cursor = snapshot["inbox_cursor"]
        applied = {(receipt.episode_id, receipt.sample_id) for receipt in cursor.applied}
        pending = [
            experience.sample_id
            for experience in cursor.experiences
            if experience.key not in applied
        ]
        stream.resume(token)
        row["continuation"].append(
            dict(
                wake_updates=point,
                next_tick=stream.next_tick,
                pending_sample_ids=pending,
                candidate_reconstructed=runtime._candidate is not candidate,
                inbox_reconstructed=runtime._inbox is not inbox,
                buffer_reconstructed=True if rehearsal is None else rehearsal._buffer is not buffer,
                runtime_preserved=stream._runtime is runtime,
                original_wake_budget=runtime._budget is wake,
                pause_consumed=token._owner is None and not token._snapshot,
            )
        )
        if row.get("trajectory_enabled", False):
            row["continuation"][-1].update(
                pending_episode_ids=[
                    experience.episode_id
                    for experience in cursor.experiences
                    if experience.key not in applied
                ],
                whole_episode_buffer_exact=buffer is None
                or (rehearsal is not None and pickle.dumps(rehearsal._buffer) == buffer_bytes),
                episode_records_exact=tuple(batch.episode for batch in stream._training)
                == tuple(batch.episode for batch in training),
            )
    stream.run_until(6)
    return stream.finish()


def _same_state(left: Any, right: Any) -> bool:
    """Compare values, avoiding pickle's unstable Torch storage identifiers."""
    if type(left) is not type(right):
        return False
    # Why: native readout guards compare many nested checkpoints. Resolve the
    # optional dependency once per comparison, including every scalar leaf.
    torch = require_torch()

    def equal_values(first: Any, second: Any) -> bool:
        kind = type(first)
        if kind is not type(second):
            return False
        # Prefix history is mostly plain records. Avoid tensor/dataclass
        # inspection for these exact builtins while still visiting every value.
        if (
            kind is str
            or kind is int
            or kind is float
            or kind is bool
            or kind is type(None)
            or kind is bytes
            or kind is complex
        ):
            return bool(first == second)
        if kind is dict:
            return first.keys() == second.keys() and all(
                equal_values(first[k], second[k]) for k in first
            )
        if kind is list or kind is tuple:
            return len(first) == len(second) and all(
                equal_values(a, b) for a, b in zip(first, second)
            )
        if kind is torch.Tensor:
            return (
                first.dtype == second.dtype
                and first.device == second.device
                and torch.equal(first, second)
            )
        if is_dataclass(first):
            return all(
                equal_values(getattr(first, f.name), getattr(second, f.name)) for f in fields(first)
            )
        # Keep the legacy fallback order for custom types and metaclasses.
        if type(first) in (list, tuple):
            return len(first) == len(second) and all(
                equal_values(a, b) for a, b in zip(first, second)
            )
        return bool(first == second)

    return equal_values(left, right)


def _tensor_bytes(value: Any) -> int:
    torch = require_torch()
    if type(value) is torch.Tensor:
        return int(value.numel() * value.element_size())
    if type(value) is dict:
        return sum(_tensor_bytes(item) for item in value.values())
    if type(value) in (list, tuple):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _observation(updates: int, state: Any) -> dict[str, Any]:
    native = state.head_state
    return dict(
        wake_updates=updates,
        native_updates=state.completed_updates,
        native_training_examples=state.training_examples,
        parameters={key: native[key].tolist() for key in PARAMETERS},
        native_counters={
            key: native[key]
            for key in ("traffic_steps", "wake_examples", "steps_since_sleep", "sleep_events")
            if key in native
        },
        adaptive_state={
            key: native[key].tolist()
            for key in ("neuron_age", "chemical", "chemical_fast", "chemical_slow")
            if key in native
        },
        head_tensor_bytes=_tensor_bytes(native),
        optimizer_tensor_bytes=_tensor_bytes(state.optimizer_state),
        gradient_tensor_bytes=_tensor_bytes(state.gradients),
    )


def _outcomes(report: dict[str, Any]) -> dict[str, float]:
    points = {point["updates"]: point["scores"] for point in report["checkpoints"]}
    return dict(
        new_B_gain=points[0]["B"]["bce"] - points[4]["B"]["bce"],
        retention_loss=points[4]["A"]["bce"] - points[2]["A"]["bce"],
        return_gain=points[4]["A"]["bce"] - points[6]["A"]["bce"],
    )


def _rows() -> list[dict[str, Any]]:
    return [
        dict(
            method=method,
            replay_enabled=enabled,
            status="not_run",
            reason="not reached",
            wake_updates=0,
            replay_charged_examples=0,
            rehearsal=None,
            final_retention=None,
            observations=[],
            initial_parameters=None,
            source_learner_unchanged=None,
            deployed_actor_unchanged=None,
            stream_result=None,
            outcomes=None,
        )
        for enabled in (False, True)
        for method in METHODS
    ]


def _run_one(
    row: dict[str, Any],
    training: tuple[AttributionBatch, ...],
    final: tuple[AttributionBatch, ...],
    deadline: float,
    sampler: ProcessRssSampler,
) -> None:
    wake = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=6,
            max_wall_seconds=max(0.0, deadline - monotonic()),
            max_process_rss_bytes=PROCESS_RSS_BYTES,
        ),
        monotonic,
    )
    replay = None
    rehearsal = None
    learner = None
    before = None
    try:
        wake.attach_memory(sampler)
        wake.before_final()
        learner = TorchFeatureLearner(TorchFeaturePolicy(row["method"], 2, 3, 17))
        before = learner.snapshot_state()
        row["initial_parameters"] = {key: before.head_state[key].tolist() for key in PARAMETERS}
        if row["replay_enabled"]:
            replay = ToyBudgetSession(
                ToyExecutionBudget(
                    max_replay_examples=6 if row.get("trajectory_enabled", False) else 3,
                    max_wall_seconds=max(0.0, deadline - monotonic()),
                    max_process_rss_bytes=PROCESS_RSS_BYTES,
                ),
                monotonic,
            )
            replay.attach_memory(sampler)
            if row.get("trajectory_enabled", False):
                rehearsal = StreamRehearsal(
                    learner,
                    wake,
                    trajectory_replay_policy(),
                    replay_budget=replay,
                    total_updates=6,
                    input_dim=2,
                    episode_schema=EpisodeReplaySchema(2, 2),
                )
            else:
                rehearsal = StreamRehearsal(
                    learner,
                    wake,
                    replay_policy(),
                    replay_budget=replay,
                    total_updates=6,
                    input_dim=2,
                )

        def observe(updates: int, state: Any, batch: AttributionBatch | None) -> None:
            row["observations"].append(_observation(updates, state))

        if row.get("continuation_enabled", False):
            report = _continued_stream(row, learner, training, final, wake, rehearsal, observe)
        else:
            report = run_attribution_stream(
                learner,
                training,
                final,
                checkpoint_updates=CHECKPOINTS,
                label_delay=1,
                budget=wake,
                observe_checkpoint=observe,
                rehearsal=rehearsal,
            )
        row.update(status="complete", reason=None, stream_result=report, outcomes=_outcomes(report))
    except Exception as error:
        row.update(status="failed", reason=f"{type(error).__name__}: {error}")
    # Record permanent costs before optional readback can fail.
    row.update(
        wake_updates=wake.updates_completed,
        replay_charged_examples=0 if replay is None else replay.replay_examples_completed,
    )
    if rehearsal is not None:
        row["rehearsal"] = rehearsal.report
    try:
        if learner is not None and before is not None:
            row["source_learner_unchanged"] = _same_state(before, learner.snapshot_state())
        if rehearsal is not None and row["status"] == "complete":
            row["final_retention"] = asdict(rehearsal.retention)
            if rehearsal._runtime is None:
                raise RuntimeError("completed rehearsal has no observed runtime")
            row["deployed_actor_unchanged"] = _same_state(
                before, rehearsal._runtime.actor.snapshot_state().state
            )
    except Exception as error:
        row.update(status="failed", reason=f"readback {type(error).__name__}: {error}")


def _plan(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            key: event[key]
            for key in (
                "updates",
                "selected_ids",
                "retention",
                "partition_retention",
                "rotation_cursor",
            )
        }
        for event in report["events"]
    ]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trajectory = all(row.get("trajectory_enabled", False) for row in rows)
    reports = [row["rehearsal"] for row in rows if row["rehearsal"] is not None]
    completed = all(row["status"] == "complete" for row in rows)
    plans = [_plan(report) for report in reports]
    parity = completed and all(
        rows[offset + 1]["stream_result"] == rows[offset + 2]["stream_result"]
        and [point["parameters"] for point in rows[offset + 1]["observations"]]
        == [point["parameters"] for point in rows[offset + 2]["observations"]]
        for offset in (0, 4)
    )
    wake = sum(row["wake_updates"] for row in rows)
    attempted = sum(report["attempted_updates"] for report in reports)
    return dict(
        wake_updates=wake,
        replay_charged_examples=sum(row["replay_charged_examples"] for row in rows),
        replay_attempted_updates=attempted,
        replay_returned_updates=sum(report["returned_updates"] for report in reports),
        replay_committed_updates=sum(report["committed_updates"] for report in reports),
        known_native_train_calls=wake + attempted,
        native_train_calls=wake + attempted if completed else None,
        training_row_exposures=2 * wake + (2 if trajectory else 1) * attempted
        if completed
        else None,
        final_score_forwards=sum(row["stream_result"]["score_forward_calls"] for row in rows)
        if completed
        else None,
        shared_replay_plans_exact=len(plans) == 4 and all(plan == plans[0] for plan in plans),
        pc_neutral_exact=parity,
        shared_initial_parameters_exact=completed
        and all(row["initial_parameters"] == rows[0]["initial_parameters"] for row in rows),
        native_clocks_exact=completed
        and all(
            [point["native_updates"] for point in row["observations"]]
            == ([0, 3, 6, 9] if row["replay_enabled"] else [0, 2, 4, 6])
            and [point["native_training_examples"] for point in row["observations"]]
            == (
                ([0, 6, 12, 18] if trajectory else [0, 5, 10, 15])
                if row["replay_enabled"]
                else [0, 4, 8, 12]
            )
            for row in rows
        ),
        replay_bounds_and_receipts_exact=completed
        and all(
            row["final_retention"] is not None
            and row["final_retention"]["example_count"] <= 4
            and row["final_retention"]["retained_bytes"] <= (1808 if trajectory else 96)
            and len(row["rehearsal"]["events"]) == 3
            and all(
                event["receipt"] is not None
                and event["committed_updates"] == 1
                and event["retention"]["example_count"] <= 4
                and event["retention"]["retained_bytes"] <= (1808 if trajectory else 96)
                and (
                    not trajectory
                    or (
                        len(event["selected_episodes"]) == 1
                        and event["charged_exposures"] == 2
                        and len(event["selected_episodes"][0]["transitions"]) == 2
                    )
                )
                for event in row["rehearsal"]["events"]
            )
            for row in rows
            if row["replay_enabled"]
        ),
        partial_work_note="permanent replay charges/attempts and wake receipts retained; failed wake/scoring work may include unknown additional work",
    )


def _execute(continuation: bool, *, trajectory: bool = False) -> dict[str, Any]:
    rows = _rows()
    if trajectory:
        for row in rows:
            row["trajectory_enabled"] = True
    if continuation:
        for row in rows:
            row.update(continuation_enabled=True, pause_attempts=0, continuation=[])
    errors = []
    sampler = None
    started = monotonic()
    try:
        sampler = ProcessRssSampler()
        with sampler:
            torch = require_torch()
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            training, final = build_trajectory_stream() if trajectory else build_stream()
            for row in rows:
                if monotonic() - started >= WALL_SECONDS:
                    break
                try:
                    _run_one(row, training, final, started + WALL_SECONDS, sampler)
                except Exception as error:
                    row.update(status="failed", reason=f"{type(error).__name__}: {error}")
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        if sampler is None:
            raise RuntimeError("RSS sampler unavailable")
        memory = asdict(sampler.snapshot())
    except Exception as error:
        memory = None
        errors.append(f"{type(error).__name__}: {error}")
    totals = _summarize(rows)
    elapsed = monotonic() - started
    complete = (
        not errors
        and all(row["status"] == "complete" for row in rows)
        and all(
            totals[key] is True
            for key in (
                "shared_replay_plans_exact",
                "pc_neutral_exact",
                "shared_initial_parameters_exact",
                "native_clocks_exact",
                "replay_bounds_and_receipts_exact",
            )
        )
        and totals["wake_updates"] == 48
        and totals["replay_charged_examples"] == (24 if trajectory else 12)
        and totals["replay_attempted_updates"]
        == totals["replay_returned_updates"]
        == totals["replay_committed_updates"]
        == 12
        and totals["native_train_calls"] == 60
        and totals["training_row_exposures"] == (120 if trajectory else 108)
        and totals["final_score_forwards"] == 64
        and all(row["source_learner_unchanged"] is True for row in rows)
        and all(row["deployed_actor_unchanged"] is True for row in rows if row["replay_enabled"])
        and elapsed <= WALL_SECONDS
        and memory is not None
        and memory["peak_bytes"] <= PROCESS_RSS_BYTES
    )
    continuation_totals: dict[str, Any] = {}
    if continuation:
        successes = sum(len(row["continuation"]) for row in rows)
        reconstructed = successes == 16 and all(
            item["next_tick"] == item["wake_updates"] + 1
            and len(item["pending_sample_ids"]) == 1
            and all(
                item[key] is True
                for key in (
                    "candidate_reconstructed",
                    "inbox_reconstructed",
                    "buffer_reconstructed",
                    "runtime_preserved",
                    "original_wake_budget",
                    "pause_consumed",
                )
            )
            for row in rows
            for item in row["continuation"]
        )
        complete = complete and reconstructed
        if trajectory:
            complete = complete and all(
                item["whole_episode_buffer_exact"] is True
                and item["episode_records_exact"] is True
                and item["pending_episode_ids"] == [f"line-train-{item['wake_updates']}"]
                for row in rows
                for item in row["continuation"]
            )
        continuation_totals.update(
            completed_resumes=successes, complete_owned_reconstruction=reconstructed
        )
    return dict(
        protocol=describe_trajectory_protocol()
        if trajectory
        else (describe_continuation_protocol() if continuation else describe_protocol()),
        status="complete" if complete else "incomplete",
        runs=rows,
        global_errors=errors,
        total_seconds=elapsed,
        observed_process_rss=memory,
        **totals,
        scientific_advantage_verified=False,
        full_R24_complete=False,
        **continuation_totals,
    )


def run_pilot() -> dict[str, Any]:
    """Execute the original eight fixed arms once, preserving every outcome."""
    return _execute(False)


def run_continuation_pilot() -> dict[str, Any]:
    """Execute the same arms with actual snapshot reconstruction at wakes two and four."""
    return _execute(True)


def run_trajectory_pilot() -> dict[str, Any]:
    """Execute complete demonstration episodes with bounded replay and owned resume."""
    return _execute(True, trajectory=True)
