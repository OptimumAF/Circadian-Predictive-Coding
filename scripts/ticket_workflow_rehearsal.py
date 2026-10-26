"""Fixed synthetic TRAIN episode compilation and existing replay composition.

No learned selection, evaluation data, new objective or external-data consent.
Terminal zero features are codec evidence; only original observations/actions fit.
"""

from dataclasses import asdict, replace

import numpy as np

from src.app.coupled_attribution import AttributionBatch, _validate_batch
from src.app.stream_rehearsal import (
    CoverageStreamReplayPolicy,
    RecentStreamReplayPolicy,
    StreamRehearsal,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.decision_features import encode_decision
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    EpisodeReplaySchema,
)
from src.core.temporal_decision_sequence import policy_for_phase
from src.core.ticket_workflow import (
    ACTION_IDS,
    WorkflowCapabilities,
    WorkflowState,
    dispatch_workflow,
    workflow_request,
    workflow_success,
)


def trajectory_batches(spec, batches, *, context_return=False):
    """Certify the fixed teacher transitions against every supplied TRAIN batch."""
    if type(context_return) is not bool:
        raise ValueError("context_return must be an exact boolean")
    if type(batches) is not tuple or len(batches) != 12:
        raise ValueError("ticket trajectories need the fixed twelve TRAIN batches")
    seen: set[str] = set()
    for batch in batches:
        _validate_batch(batch, "train", seen)
    if context_return:
        return _context_queue_batches(spec, batches)
    contexts = {"A": "invoice", "B": "parcel", "A_return": "invoice"}
    class_ids = tuple(c.candidate_id for c in spec.catalog)
    prepared: list[AttributionBatch] = []
    for index, batch in enumerate(batches):
        phase = ("A", "B", "A_return")[index // 4]
        if batch.role != "train" or batch.context != phase or batch.episode is not None:
            raise ValueError("ticket trajectory source must be original TRAIN phase data")
        concept = contexts[phase]
        state, last = WorkflowState("teacher-ticket", concept), "initial"
        caps = WorkflowCapabilities(ACTION_IDS)
        actions = ("open", policy_for_phase("after_return")["rules"][concept], "commit")
        observations, transitions = [], []
        for step, action in enumerate(actions):
            request = workflow_request(state, caps, f"teacher-{step}", last)
            request = replace(request, query=request.query + " train")
            features, _ = encode_decision(request, spec)
            observations.append(tuple(float(x) for x in features[0]))
            transition = dispatch_workflow(state, action, state.revision, caps)
            if transition.status != "applied":
                raise ValueError("teacher transition did not apply")
            state, last = transition.state, transition.status
        expected = np.array(observations, dtype=np.float64)
        labels = np.array([[float(class_ids.index(a))] for a in actions])
        if (
            not np.array_equal(batch.features, expected)
            or not np.array_equal(batch.targets, labels)
            or batch.num_classes != len(class_ids)
        ):
            raise ValueError("TRAIN observations/actions differ from authentic ticket teacher")
        if not workflow_success(state):
            raise ValueError("teacher ticket did not complete correctly")
        for step, action in enumerate(actions):
            transitions.append(
                DemonstrationTransition(
                    step,
                    observations[step],
                    class_ids.index(action),
                    float(step == 2),
                    observations[step + 1] if step < 2 else (0.0,) * len(spec.vocabulary),
                    step == 2,
                    len(class_ids),
                )
            )
        episode = DemonstrationEpisode(
            f"{phase}-{index % 4}", concept, "train", tuple(transitions), len(class_ids)
        )
        prepared.append(
            replace(
                batch,
                context=concept,
                episode=episode,
                row_ids=tuple(f"{episode.episode_id}:step:{i}" for i in range(3)),
            )
        )
    return tuple(prepared)


def _ticket_trace(spec, version, concept):
    """Validate one real ticket; observation means the next policy decision."""
    state = WorkflowState("queue-teacher", concept, policy_version=version)
    caps, last = WorkflowCapabilities(ACTION_IDS), "initial"
    rules = policy_for_phase("after_A" if version == "v1" else "after_return")["rules"]
    actions = ("open", rules[concept], "commit")
    observations = []
    for step, action in enumerate(actions):
        request = workflow_request(state, caps, f"queue-{step}", last, include_policy_context=True)
        observations.append(tuple(float(x) for x in encode_decision(request, spec)[0][0]))
        transition = dispatch_workflow(state, action, state.revision, caps)
        if transition.status != "applied":
            raise ValueError("queue teacher transition did not apply")
        state, last = transition.state, transition.status
    if not workflow_success(state):
        raise ValueError("queue teacher ticket did not solve its declared policy")
    return observations, actions


def _context_queue_batches(spec, batches):
    from scripts.run_ticket_workflow import CONTEXT_FEATURES
    from scripts.ticket_controller_streams import CONTEXT_RETURN_PHASES
    from src.core.decision_features import DecisionFeatureSpec
    from src.core.ticket_workflow import CATALOG

    if type(spec) is not DecisionFeatureSpec or spec != DecisionFeatureSpec(
        CONTEXT_FEATURES, CATALOG
    ):
        raise ValueError("queue rehearsal requires the fixed D9/catalog schema")
    prepared = []
    class_ids = tuple(c.candidate_id for c in spec.catalog)
    for index, batch in enumerate(batches):
        phase, version, concepts = CONTEXT_RETURN_PHASES[index // 4]
        if batch.context != phase or batch.episode is not None or batch.num_classes != 6:
            raise ValueError("queue rehearsal requires original contextual TRAIN phases")
        traces = [_ticket_trace(spec, version, concept) for concept in concepts]
        observations = [o for obs, _ in traces for o in obs]
        actions = [a for _, act in traces for a in act]
        labels = np.array([[float(class_ids.index(a))] for a in actions])
        if not np.array_equal(batch.features, np.array(observations)) or not np.array_equal(
            batch.targets, labels
        ):
            raise ValueError("queue TRAIN inputs differ from authentic contextual teacher")
        # Why: the environment advances the queue after the first correct commit;
        # both tickets are validated above. Only the complete queue is terminal.
        transitions = tuple(
            DemonstrationTransition(
                step,
                observations[step],
                class_ids.index(action),
                float(step in (2, 5)),
                observations[step + 1] if step < 5 else (0.0,) * 9,
                step == 5,
                6,
            )
            for step, action in enumerate(actions)
        )
        context = "-".join((version, *concepts))
        episode = DemonstrationEpisode(f"{phase}-{index % 4}", context, "train", transitions, 6)
        prepared.append(
            replace(
                batch,
                context=context,
                episode=episode,
                row_ids=tuple(f"{episode.episode_id}:step:{i}" for i in range(6)),
            )
        )
    return tuple(prepared)


def make_rehearsal(mode, learner, wake_budget, sampler, *, context_return=False):
    if type(context_return) is not bool:
        raise ValueError("context_return must be an exact boolean")
    if mode not in ("recent", "coverage"):
        raise ValueError("explicit recent or coverage replay required")
    width, steps = (9, 6) if context_return else (7, 3)
    schema = EpisodeReplaySchema(width, steps, 6)
    limits = ReplayRetentionBudget(2 * steps, 2 * 8 * (schema.encoded_dim + 1))
    policy = (
        RecentStreamReplayPolicy(2, steps, limits)
        if mode == "recent"
        else CoverageStreamReplayPolicy(
            2,
            steps,
            limits,
            ("v1-invoice-password", "v2-invoice-parcel")
            if context_return
            else ("invoice", "parcel"),
        )
    )
    replay_budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_replay_examples=6 * steps, max_wall_seconds=30, max_process_rss_bytes=512 * 1024**2
        ),
        wake_budget.clock,
    )
    replay_budget.attach_memory(sampler)
    helper = StreamRehearsal(
        learner,
        wake_budget,
        policy=policy,
        replay_budget=replay_budget,
        total_updates=12,
        input_dim=width,
        episode_schema=schema,
    )
    return helper, replay_budget


def require_progress(
    stream, helper, replay_budget, point, state, previous, *, context_return=False
):
    """Native additions require exact published original replay receipts."""
    if type(context_return) is not bool:
        raise ValueError("context_return must be an exact boolean")
    steps = 6 if context_return else 3
    receipts = tuple(asdict(r) for r in stream.applied_consolidations)
    if receipts[: len(previous)] != previous:
        raise ValueError("previous replay publication receipt prefix changed")
    report = helper.report
    events = [e for e in report["events"] if e["updates"] <= point]
    expected_ids = tuple(f"rehearsal-{p}" for p in range(2, point + 1, 2))
    if (
        tuple(r["event_id"] for r in receipts) != expected_ids
        or len(events) != len(receipts)
        or any(
            e["status"] != "complete"
            or e["receipt"] != r
            or e["committed_updates"] != 1
            or e["attempted_updates"] != 1
            or e["returned_updates"] != 1
            or e["charged_exposures"] != steps
            or len(e["selected_episodes"]) != 1
            or e["selected_episodes"][0]["role"] != "train"
            for e, r in zip(events, receipts, strict=True)
        )
    ):
        raise ValueError("replay publication receipts or complete TRAIN event work differ")
    committed = len(receipts)
    if (
        report["committed_updates"] != committed
        or report["charged_exposures"] != steps * committed
        or replay_budget.replay_examples_completed != steps * committed
        or replay_budget.updates_completed != 0
        or state.completed_updates != point + committed
        or state.training_examples != steps * (point + committed)
    ):
        raise ValueError("native replay progress differs from published receipts and budgets")
    return receipts
