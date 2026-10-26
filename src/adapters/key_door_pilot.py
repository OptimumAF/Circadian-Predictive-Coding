"""Actual key/door demonstrations, ordinary adaptation, then frozen live policies.

The fixed local recipe composes existing native learners and AttributionStream.
Known-room recall and an unseen layout are scored separately. The explicit
recurrent control uses fresh episode sessions; no RL objective, final feedback,
promotion, checkpoint selection or tuning is added.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from time import monotonic
from typing import Any, Callable

from src.adapters.key_door_learnability import probe_slots, run_checkpoint_probe, valid_probes
from src.adapters.key_door_public_history_session import PublicHistoryPredictionSession
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.adapters.torch_feature_rehearsal_pilot import _observation, _same_state, _tensor_bytes
from src.adapters.torch_recurrent_learner import TorchRecurrentLearner, TorchRecurrentPolicy
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.key_door_policy_evaluation import KeyDoorEvaluationFailed, run_key_door_episode
from src.app.line_policy_evaluation import EvaluationIdentity
from src.app.stream_rehearsal import StreamReplayPolicy, StreamRehearsal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    EpisodeReplaySchema,
    episode_arrays,
)
from src.core.key_door_navigation import (
    KeyDoorLayout,
    KeyDoorNavigation,
    key_door_layout_record,
    scripted_actions,
)
from src.core.shared_replay_schedule import ReplayAllocation
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch

METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
POINTS = (0, 2, 4, 6, 8)
TASKS = ("A", "B", "C", "A")
MAX_RSS = 768 * 1024**2
MAX_BYTES = 8 * 1024**2


def layouts() -> tuple[KeyDoorLayout, ...]:
    return (
        KeyDoorLayout("A"),
        KeyDoorLayout("B", 1, 2, True, 0.5),
        KeyDoorLayout("C", 2, 2, False, -0.5),
        KeyDoorLayout("D", 0, 1, False, 0.25),
    )


def third_training_layout() -> KeyDoorLayout:
    """Declare the third training geometry without repurposing historical final C."""
    return KeyDoorLayout("C_train", 2, 3, True, -0.75)


def replay_policy() -> StreamReplayPolicy:
    return StreamReplayPolicy(
        2,
        8,
        ReplayRetentionBudget(16, 4032),
        ReplayAllocation(ReplayRetentionBudget(8, 2016), ReplayRetentionBudget(8, 2016)),
        53,
    )


def describe_protocol() -> dict[str, Any]:
    return dict(
        protocol_id="actual_hidden_cue_key_door_v1",
        classification="local_key_door_behavior_cloning_and_live_evaluation",
        environment="deterministic local MiniGrid-like feature game; not official MiniGrid integration",
        layouts=[key_door_layout_record(layout) for layout in layouts()],
        heldout_layouts=["D"],
        methods=list(METHODS),
        replay_modes=[False, True],
        seed=17,
        feature_dim=8,
        hidden_dim=8,
        num_classes=5,
        task_sequence=list(TASKS),
        cues=[1, -1],
        checkpoint_updates=list(POINTS),
        label_delay=1,
        training="A+/A-/B+/B-/C+/C-/A+/A-; eight actual teacher episodes, eight steps each",
        final="fresh known A/B/C and unseen geometry D, both cues; final_test only",
        observations="four physical neighbor tiles/current tile/held color/initial-only cue/appearance; no map or hidden target",
        actions=["up", "right", "down", "left", "interact"],
        teacher="scripted cue/layout oracle for collection only; never policy evaluation",
        rewards="goal1, wrong-key terminal-1, other legal0; illegal decisions are unapplied evaluation failures",
        episode_schema=asdict(EpisodeReplaySchema(8, 8, 5)),
        encoded_packet_array_bytes=2016,
        replay=asdict(replay_policy()),
        native_defaults=dict(
            learning_rate=0.03, inference_steps=2, inference_learning_rate=0.04, momentum=0.9
        ),
        metrics=dict(
            success="actual goal reached",
            returns="sum actual rewards",
            steps="applied legal steps",
            ce="mean -log clip(p_true,1e-12,1); lower",
            accuracy="argmax first tie; higher",
        ),
        ce_directions=dict(
            new_B_gain="mean B CE0 minus CE4",
            new_C_gain="mean C CE0 minus CE6",
            retention_loss="mean A CE6 minus CE2",
            return_gain="mean A CE6 minus CE8",
        ),
        evaluation="all five complete owned checkpoints, argmax first tie, no masking/repair/retry; after all arm training",
        max_episode_steps=12,
        planned_work=dict(
            wake_updates=64,
            replay_calls=16,
            native_train_calls=80,
            training_rows=640,
            pc_family_steps=60,
            bp_steps=20,
            score_forwards=320,
            score_rows=2560,
            live_episodes=320,
            maximum_live_forwards=3840,
            maximum_native_forwards=4160,
            demonstration_steps=128,
            maximum_environment_steps=3968,
            environment_constructions=336,
            probes=0,
            resumes=0,
        ),
        resources=dict(
            cpu_threads=1,
            gpu=0,
            wall_seconds=90,
            process_rss_bytes=MAX_RSS,
            storage_bytes=MAX_BYTES,
        ),
        final_isolation="no final scoring/rollout data enter training/replay/promotion/tuning/probe selection",
        limitations="one seed/tiny local game/feedforward imitation; no recurrent RL, acquisition/generalization guarantee, equal off/on work or circadian advantage",
    )


def describe_recurrent_protocol() -> dict[str, Any]:
    """The same small data/exposure protocol, with only the new recurrent control."""
    protocol = describe_protocol()
    protocol.update(
        protocol_id="actual_hidden_cue_key_door_recurrent_bc_v1",
        classification="local_recurrent_behavior_cloning_and_live_evaluation",
        methods=["recurrent_bc"],
        native_defaults=dict(learning_rate=0.03, momentum=0.9, max_episode_steps=16),
        architecture="Elman tanh/full episode BPTT/mean action cross entropy; zero hidden per packet",
        live_activation="new owned frozen session per episode; no teacher cue injection",
        baseline_reference="artifacts/runs/r42-key-door-20261016/pilot.json",
        baseline_execution="reuse accepted original eight-arm evidence; never retrain or load it into learning",
        comparisons="PC-neutral parity not applicable; initial parity only between recurrent off/on; architecture/initialization/objective/memory/compute differ from feedforward",
        limitations="one seed/eight wakes/local game; no recurrent RL, useful adaptation guarantee, equal off/on work, equal-architecture comparison or circadian advantage",
    )
    protocol["planned_work"] = dict(
        wake_updates=16,
        replay_calls=4,
        native_train_calls=20,
        training_rows=160,
        pc_family_steps=0,
        bp_steps=20,
        score_forwards=80,
        score_rows=640,
        live_episodes=80,
        maximum_live_forwards=960,
        maximum_native_forwards=1040,
        maximum_recurrent_unroll_steps=1760,
        live_sessions=80,
        demonstration_steps=128,
        maximum_environment_steps=1088,
        environment_constructions=96,
        probes=0,
        resumes=0,
    )
    return protocol


def _recurrent_observation(updates: int, state: Any) -> dict[str, Any]:
    return dict(
        wake_updates=updates,
        native_updates=state.completed_updates,
        native_training_examples=state.training_examples,
        parameters={key: value.tolist() for key, value in state.parameters.items()},
        architecture="elman_tanh",
        head_tensor_bytes=_tensor_bytes(state.parameters),
        optimizer_tensor_bytes=_tensor_bytes(state.optimizer_state),
        gradient_tensor_bytes=_tensor_bytes(state.gradients),
        activation_tensor_bytes=4 * state.policy.hidden_dim,
        training_unroll_steps=state.training_examples,
        memory_scope="tensor payload only; excludes owned checkpoint/session birth copies, metadata, temporary graph and process RSS",
    )


def describe_learnability_protocol(*, decision_diagnostics: bool = False) -> dict[str, Any]:
    if type(decision_diagnostics) is not bool:
        raise ValueError("decision_diagnostics must be an exact bool")
    protocol = describe_recurrent_protocol()
    protocol.update(
        protocol_id="actual_key_door_owned_checkpoint_learnability_v1",
        classification="reserved_complete_episode_probe_at_owned_stream_checkpoints",
        probe=dict(
            layout=key_door_layout_record(KeyDoorLayout("E", 3, 3, True, 0.75)),
            source_checkpoints=[0, 8],
            training_episodes=16,
            evaluation_episodes=2,
            training="sixteen unique probe_train episodes alternating E cues +1/-1",
            evaluation="two reserved probe_test E±; same deterministic trace domain, not generalization",
            checkpoints=[0, 4, 8, 16],
            independent_updates_per_cohort=16,
            metrics="paired E mean CE0 minus CE16; mean accuracy0 and16; all curves retained",
            isolation="full owned opaque checkpoint fork; no benchmark final data, promotion, tuning, selection or source update",
        ),
    )
    protocol["planned_work"].update(
        native_train_calls=84,
        training_rows=672,
        bp_steps=84,
        score_forwards=112,
        score_rows=896,
        maximum_native_forwards=1072,
        maximum_recurrent_unroll_steps=2528,
        demonstration_steps=272,
        maximum_environment_steps=1232,
        environment_constructions=114,
        probes=4,
        probe_train_calls=64,
        probe_training_rows=512,
        probe_score_forwards=32,
        probe_score_rows=256,
    )
    if decision_diagnostics:
        protocol["protocol_id"] = "actual_key_door_owned_checkpoint_decisions_v1"
        protocol["probe"].update(
            decision_diagnostics=True,
            decision_diagnostics_scope=(
                "already returned reserved probe_test predictions; no extra native calls; "
                "bounded per-class support/confusion/loss and ordered episode decisions; "
                "at most4points/2contexts/8rows/5classes in this protocol; "
                "original reserved_data_bytes excludes added Python/temp NumPy storage; "
                "total invocation time and sampled process RSS include diagnostic cost"
            ),
        )
    return protocol


def build_probe_stream() -> tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]]:
    layout = KeyDoorLayout("E", 3, 3, True, 0.75)
    return (
        tuple(
            _collect(f"reserved-E-train-{i}", layout, 1 if i % 2 == 0 else -1, "probe_train")
            for i in range(16)
        ),
        tuple(_collect(f"reserved-E-test-{cue}", layout, cue, "probe_test") for cue in (1, -1)),
    )


def _context(layout: KeyDoorLayout, cue: int) -> str:
    return f"{layout.layout_id}-{'red' if cue == 1 else 'blue'}"


def _collect(identity: str, layout: KeyDoorLayout, cue: int, role: str) -> AttributionBatch:
    environment = KeyDoorNavigation(layout, cue)
    transitions = []
    for step, action in enumerate(scripted_actions(layout, cue)):
        observation = environment.observation
        result = environment.step(action)
        transitions.append(
            DemonstrationTransition(
                step, observation, action, result.reward, result.observation, result.terminal, 5
            )
        )
    if not environment.terminal or not environment.success:
        raise RuntimeError("scripted key/door collection did not reach the goal")
    episode = DemonstrationEpisode(identity, _context(layout, cue), role, tuple(transitions), 5)
    features, targets = episode_arrays(episode)
    return AttributionBatch(
        role,
        episode.context,
        tuple(f"{identity}:step:{i}" for i in range(8)),
        features,
        targets,
        episode,
        5,
    )


def build_training_stream() -> tuple[AttributionBatch, ...]:
    """Collect only the original TRAIN episodes, without constructing finals."""
    declared = {layout.layout_id: layout for layout in layouts()}
    return tuple(
        _collect(f"key-door-train-{2 * i + j}", declared[context], cue, "train")
        for i, context in enumerate(TASKS)
        for j, cue in enumerate((1, -1))
    )


def build_stream() -> tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]]:
    training = build_training_stream()
    final = tuple(
        _collect(f"key-door-final-{layout.layout_id}-{cue}", layout, cue, "final_test")
        for layout in layouts()
        for cue in (1, -1)
    )
    return training, final


def _session(
    deadline: float, sampler: ProcessRssSampler, *, replay: bool = False, max_updates: int = 8
) -> ToyBudgetSession:
    if (
        type(max_updates) is not int
        or max_updates not in (0, 8, 16)
        or (replay and max_updates != 8)
    ):
        raise ValueError(
            "key/door wake limit must be 0, 8 or 16; replay retains its original limit"
        )
    session = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=None if replay else max_updates,
            max_replay_examples=32 if replay else None,
            max_wall_seconds=max(0.0, deadline - monotonic()),
            max_process_rss_bytes=MAX_RSS,
        ),
        monotonic,
    )
    session.attach_memory(sampler)
    session.before_final()
    return session


def _slots() -> list[dict[str, Any]]:
    return [
        dict(
            checkpoint_updates=point,
            context=layout.layout_id,
            cue=cue,
            layout_partition="heldout_layout" if layout.layout_id == "D" else "known_layout",
            status="not_run",
            reason="not reached",
            state_unchanged=None,
        )
        for point in POINTS
        for layout in layouts()
        for cue in (1, -1)
    ]


def _live(
    row: dict[str, Any],
    source: Any,
    states: list[tuple[int, Any]],
    budget: ToyBudgetSession,
    *,
    task_sequence: tuple[str, ...] = TASKS,
    check_boundary: Callable[[], None] | None = None,
    declared_layouts: tuple[KeyDoorLayout, ...] | None = None,
    condition_on_legal_actions: bool = False,
    public_history: bool = False,
    observe_budget: Callable[[], None] | None = None,
    same_state: Callable[[Any, Any], bool] | None = None,
) -> None:
    if same_state is not None and not callable(same_state):
        raise ValueError("same_state must be callable or None")
    compare = _same_state if same_state is None else same_state
    if type(condition_on_legal_actions) is not bool:
        raise ValueError("condition_on_legal_actions must be an exact bool")
    if check_boundary is not None and not callable(check_boundary):
        raise ValueError("check_boundary must be callable or None")
    if type(public_history) is not bool:
        raise ValueError("public_history must be an exact bool")
    if observe_budget is not None and not callable(observe_budget):
        raise ValueError("observe_budget must be callable or None")
    if public_history and (
        row["method"] not in (*METHODS, "observation_memory")
        or source.policy.feature_dim != 16
        or source.policy.num_classes != 5
        or condition_on_legal_actions
    ):
        raise ValueError("public history requires the original unmasked16-input feature methods")

    def before_prediction() -> None:
        if observe_budget is None:
            budget.before_final()
        else:
            observe_budget()

    if declared_layouts is not None:
        if (
            type(declared_layouts) is not tuple
            or not 1 <= len(declared_layouts) <= 32
            or any(type(layout) is not KeyDoorLayout for layout in declared_layouts)
        ):
            raise ValueError(
                "declared_layouts must be an exact tuple of 1..32 KeyDoorLayout records"
            )
        declared_layouts = tuple(replace(layout) for layout in declared_layouts)
        if len({layout.layout_id for layout in declared_layouts}) != len(declared_layouts):
            raise ValueError("declared_layouts require unique layout IDs")
        if any(
            slot["context"] not in {layout.layout_id for layout in declared_layouts}
            for slot in row["episodes"]
        ):
            raise ValueError("live episode context requires its declared layout")
    # Admit the complete evaluation identity before any external learner work.
    identities = {
        point: EvaluationIdentity(
            row["method"],
            17,
            f"eval:{row['method']}:replay-{row['replay_enabled']}:wake-{point}",
            task_sequence,
            point,
            row["replay_enabled"],
        )
        for point, _ in states
    }
    if not states:
        EvaluationIdentity(row["method"], 17, "eval:empty", task_sequence, 0, row["replay_enabled"])
    if condition_on_legal_actions:
        row["action_conditioning"] = "legal_actions_v1"

    def check() -> None:
        if check_boundary is not None:
            check_boundary()

    declared = {
        layout.layout_id: layout
        for layout in (layouts() if declared_layouts is None else declared_layouts)
    }
    for point, state in states:
        expected = deepcopy(state)
        before_prediction()
        check()
        predictor = source.fork()
        check()
        if predictor is source:
            raise ValueError("evaluation predictor must own a distinct native learner")
        check()
        predictor.restore_state(deepcopy(expected))
        check()
        frozen = predictor.snapshot_state()
        if not compare(expected, frozen):
            raise ValueError("evaluation native state differs from its captured state")
        current = predictor

        def predict(features: Any) -> Any:
            before_prediction()
            check()
            if not compare(frozen, current.snapshot_state()):
                raise ValueError("evaluation native state changed before prediction")
            if public_history and not compare(frozen, predictor.snapshot_state()):
                raise ValueError("public history checkpoint owner changed before prediction")
            result = current.predict(features)
            check()
            before_prediction()
            if not compare(frozen, current.snapshot_state()):
                raise ValueError("evaluation native state changed during prediction")
            if public_history and not compare(frozen, predictor.snapshot_state()):
                raise ValueError("public history checkpoint owner changed during prediction")
            return result

        identity = identities[point]
        for slot in row["episodes"]:
            if slot["checkpoint_updates"] != point:
                continue
            history_session = None
            history_attempted = False
            try:
                if public_history:
                    before_prediction()
                    check()
                    history_attempted = True
                    history_session = (
                        PublicHistoryPredictionSession(predictor)
                        if same_state is None
                        else PublicHistoryPredictionSession(predictor, same_state=compare)
                    )
                    current = history_session
                    row["live_sessions_started"] += 1
                    check()
                    if current.completed_steps != 0 or not compare(
                        frozen, current.snapshot_state()
                    ):
                        raise ValueError("public history requires a fresh owned zero-step session")
                elif row["method"] in ("recurrent_bc", "recurrent_rl"):
                    before_prediction()
                    check()
                    current = predictor.start_episode()
                    row["live_sessions_started"] += 1
                    check()
                    if (
                        current is predictor
                        or type(current.completed_steps) is not int
                        or current.completed_steps != 0
                        or not compare(frozen, current.snapshot_state())
                    ):
                        raise ValueError(
                            "recurrent evaluation requires a fresh owned zero-step session"
                        )
                check()
                evaluation_options: dict[str, Any] = (
                    {"condition_on_legal_actions": True} if condition_on_legal_actions else {}
                )
                result = run_key_door_episode(
                    predict,
                    identity=identity,
                    context=slot["context"],
                    layout=declared[slot["context"]],
                    cue=slot["cue"],
                    layout_partition=slot["layout_partition"],
                    max_steps=12,
                    before_prediction=before_prediction,
                    **evaluation_options,
                )
                slot.update(result, reason=None)
                check()
                if not compare(frozen, current.snapshot_state()) or not compare(
                    frozen, predictor.snapshot_state()
                ):
                    raise ValueError("evaluation native state changed after episode")
                slot["state_unchanged"] = True
            except KeyDoorEvaluationFailed as error:
                slot.update(error.report, state_unchanged=False)
                raise
            except Exception as error:
                slot.update(
                    status="failed",
                    reason=f"{type(error).__name__}: {error}",
                    state_unchanged=False,
                )
                raise
            finally:
                if public_history:
                    slot["history_session_work"] = dict(
                        session_construction_attempts=int(history_attempted),
                        sessions_started=int(history_session is not None),
                        attempted_steps=0
                        if history_session is None
                        else history_session.attempted_steps,
                        encoded_frames=0
                        if history_session is None
                        else history_session.encoded_frames,
                        native_prediction_returns=0
                        if history_session is None
                        else history_session.native_prediction_returns,
                        completed_history_steps=0
                        if history_session is None
                        else history_session.completed_steps,
                        failed_work_may_be_unobserved=slot["status"] != "complete",
                    )
    check()


def _outcomes(report: dict[str, Any]) -> dict[str, float]:
    points = {p["updates"]: p["scores"] for p in report["checkpoints"]}

    def mean(point: int, context: str) -> float:
        return sum(points[point][f"{context}-{color}"]["ce"] for color in ("red", "blue")) / 2

    return dict(
        new_B_gain=mean(0, "B") - mean(4, "B"),
        new_C_gain=mean(0, "C") - mean(6, "C"),
        retention_loss=mean(6, "A") - mean(2, "A"),
        return_gain=mean(6, "A") - mean(8, "A"),
    )


def _run_arm(
    row: dict[str, Any],
    training: tuple[AttributionBatch, ...],
    final: tuple[AttributionBatch, ...],
    deadline: float,
    sampler: ProcessRssSampler,
    probe_data: tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]] | None = None,
    *,
    decision_diagnostics: bool = False,
) -> None:
    wake, replay, helper, source, before, stream = None, None, None, None, None, None
    states: list[tuple[int, Any]] = []
    try:
        wake = _session(deadline, sampler)
        source = (
            TorchRecurrentLearner(TorchRecurrentPolicy(8, 8, 5, 17))
            if row["method"] == "recurrent_bc"
            else TorchFeatureLearner(TorchFeaturePolicy(row["method"], 8, 8, 17, num_classes=5))
        )
        before = source.snapshot_state()
        if row["replay_enabled"]:
            replay = _session(deadline, sampler, replay=True)
            helper = StreamRehearsal(
                source,
                wake,
                replay_policy(),
                replay_budget=replay,
                total_updates=8,
                input_dim=8,
                episode_schema=EpisodeReplaySchema(8, 8, 5),
            )

        def observe(updates: int, state: Any, batch: AttributionBatch | None) -> None:
            row["observations"].append(
                _recurrent_observation(updates, state)
                if row["method"] == "recurrent_bc"
                else _observation(updates, state)
            )
            states.append((updates, deepcopy(state)))
            if probe_data is not None and updates in (0, 8):
                probe_options: dict[str, Any] = (
                    {"decision_diagnostics": True} if decision_diagnostics else {}
                )
                run_checkpoint_probe(
                    row,
                    source,
                    state,
                    probe_data,
                    checkpoint=updates,
                    wake=wake,
                    replay=replay,
                    deadline=deadline,
                    sampler=sampler,
                    same_state=_same_state,
                    probe_runner=run_fresh_task_probe,
                    **probe_options,
                )

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
        stream.run_until(8)
        row["stream_result"] = stream.finish()
        row["outcomes"] = _outcomes(row["stream_result"])
        _live(row, source, states, wake)
        row.update(status="complete", reason=None)
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
            row["readback_error"] = f"readback {type(error).__name__}: {error}"
            if row["status"] != "failed":
                row.update(status="failed", reason=row["readback_error"])
        if row["method"] == "recurrent_bc":
            row["work"] = _recurrent_work(row)


def _recurrent_work(row: dict[str, Any]) -> dict[str, Any]:
    scores = row["stream_result"]
    work = dict(
        observed_checkpoint_training_unroll_steps=(
            row["observations"][-1]["native_training_examples"] if row["observations"] else None
        ),
        classification_unroll_steps=None if scores is None else scores["score_forward_calls"] * 8,
        live_callback_attempts=sum(e.get("prediction_calls_attempted", 0) for e in row["episodes"]),
        live_callback_returns=sum(e.get("prediction_calls_returned", 0) for e in row["episodes"]),
        live_applied_steps=sum(e.get("steps_executed", 0) for e in row["episodes"]),
        live_sessions_started=row["live_sessions_started"],
        failure_work_may_be_unobserved=row["status"] != "complete",
        scope="complete successful rows count exact unrolled steps; partial failed native work requires execution receipt, never zero-imputed",
    )
    if "probes" in row:
        reports = [p["report"] for p in row["probes"] if p["report"] is not None]
        work.update(
            observed_probe_completed_training_unroll_steps=sum(
                r["completed_training_examples"] for r in reports
            ),
            observed_probe_score_unroll_steps=8
            * sum(r["score_forward_calls_completed"] for r in reports),
            probe_reports_available=len(reports),
        )
    return work


def _valid(row: dict[str, Any]) -> bool:
    if row["status"] != "complete":
        return False
    if "probes" in row and not valid_probes(row):
        return False
    expected = [0, 3, 6, 9, 12] if row["replay_enabled"] else list(POINTS)
    valid = (
        row["source_unchanged"] is True
        and row["actor_unchanged"] is True
        and row["wake_updates"] == 8
        and row["stream_result"]["score_forward_calls"] == 40
        and len(row["episodes"]) == 40
        and all(e["status"] == "complete" and e["state_unchanged"] is True for e in row["episodes"])
        and [p["native_updates"] for p in row["observations"]] == expected
        and [p["native_training_examples"] for p in row["observations"]]
        == [8 * n for n in expected]
    )
    if not row["replay_enabled"]:
        return (
            valid
            and row["replay_charged_examples"] == 0
            and (row["method"] != "recurrent_bc" or row["live_sessions_started"] == 40)
        )
    events = row["rehearsal"]["events"]
    return bool(
        valid
        and (row["method"] != "recurrent_bc" or row["live_sessions_started"] == 40)
        and row["replay_charged_examples"] == 32
        and len(events) == 4
        and row["final_retention"]["example_count"] <= 16
        and row["final_retention"]["retained_bytes"] <= 4032
        and all(
            e["status"] == "complete"
            and e["charged_exposures"] == 8
            and e["attempted_updates"] == e["returned_updates"] == e["committed_updates"] == 1
            and e["receipt"] is not None
            for e in events
        )
    )


def run_pilot() -> dict[str, Any]:
    return _run_pilot(METHODS, describe_protocol())


def run_recurrent_pilot() -> dict[str, Any]:
    return _run_pilot(("recurrent_bc",), describe_recurrent_protocol())


def run_learnability_pilot(*, decision_diagnostics: bool = False) -> dict[str, Any]:
    if type(decision_diagnostics) is not bool:
        raise ValueError("decision_diagnostics must be an exact bool")
    options = {"decision_diagnostics": True} if decision_diagnostics else {}
    protocol = describe_learnability_protocol(**options)
    return _run_pilot(("recurrent_bc",), protocol, with_probes=True, **options)


def _run_pilot(
    methods: tuple[str, ...],
    protocol: dict[str, Any],
    *,
    with_probes: bool = False,
    decision_diagnostics: bool = False,
) -> dict[str, Any]:
    if type(decision_diagnostics) is not bool or (decision_diagnostics and not with_probes):
        raise ValueError("decision diagnostics require probe mode and an exact bool")
    rows: list[dict[str, Any]] = [
        dict(
            method=m,
            replay_enabled=r,
            status="not_run",
            reason="not reached",
            observations=[],
            episodes=_slots(),
            wake_updates=0,
            replay_charged_examples=0,
            stream_result=None,
            outcomes=None,
            rehearsal=None,
            final_retention=None,
            source_unchanged=None,
            actor_unchanged=None,
            readback_error=None,
        )
        for r in (False, True)
        for m in methods
    ]
    for row in rows:
        if row["method"] == "recurrent_bc":
            row.update(live_sessions_started=0, work=None)
        if with_probes:
            row["probes"] = probe_slots()
    errors, memory, sampler, demonstrations = [], None, None, []
    probe_demonstrations = []
    started = monotonic()
    try:
        sampler = ProcessRssSampler()
        with sampler:
            torch = require_torch()
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            training, final = build_stream()
            for batch in (*training, *final):
                if batch.episode is None:
                    raise ValueError("key/door collection requires complete episode metadata")
                demonstrations.append(asdict(batch.episode))
            probe_data = build_probe_stream() if with_probes else None
            if probe_data is not None:
                for batch in (*probe_data[0], *probe_data[1]):
                    if batch.episode is None:
                        raise ValueError("reserved probes require complete episode records")
                    probe_demonstrations.append(asdict(batch.episode))
            for row in rows:
                if monotonic() - started >= 90:
                    break
                if probe_data is None:
                    _run_arm(row, training, final, started + 90, sampler)
                else:
                    _run_arm(
                        row,
                        training,
                        final,
                        started + 90,
                        sampler,
                        probe_data,
                        **({"decision_diagnostics": True} if decision_diagnostics else {}),
                    )
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        if sampler is not None:
            memory = asdict(sampler.snapshot())
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    complete = all(_valid(row) for row in rows)
    plans = [
        [e["selected_ids"] for e in r["rehearsal"]["events"]]
        for r in rows
        if r["rehearsal"] is not None
    ]
    shared = len(plans) == len(methods) and all(p == plans[0] for p in plans)
    recurrent = methods == ("recurrent_bc",)
    neutral = (
        (not recurrent)
        and complete
        and all(
            rows[i + 1]["stream_result"]["checkpoints"]
            == rows[i + 2]["stream_result"]["checkpoints"]
            and all(
                a["parameters"] == b["parameters"]
                for a, b in zip(
                    rows[i + 1]["observations"], rows[i + 2]["observations"], strict=True
                )
            )
            and all(
                (a["outcome"], a["return"], a["transitions"])
                == (b["outcome"], b["return"], b["transitions"])
                for a, b in zip(rows[i + 1]["episodes"], rows[i + 2]["episodes"], strict=True)
            )
            for i in (0, 4)
        )
    )
    initial = complete and all(
        r["observations"][0]["parameters"] == rows[0]["observations"][0]["parameters"] for r in rows
    )
    elapsed = monotonic() - started
    complete = bool(
        complete
        and shared
        and (recurrent or neutral)
        and initial
        and not errors
        and elapsed <= 90
        and memory is not None
        and memory["peak_bytes"] <= MAX_RSS
    )
    report = dict(
        protocol=protocol,
        status="complete" if complete else "incomplete",
        runs=rows,
        demonstrations=demonstrations,
        global_errors=errors,
        total_seconds=elapsed,
        observed_process_rss=memory,
        shared_replay_plans_exact=None if recurrent else shared,
        pc_neutral_exact=None if recurrent else neutral,
        shared_initial_parameters_exact=initial,
        scientific_advantage_verified=False,
        full_R42_complete=False,
    )
    if with_probes:
        report["probe_demonstrations"] = probe_demonstrations
    return report
