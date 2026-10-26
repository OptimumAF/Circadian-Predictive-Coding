"""Bounded additional-update control on eight unchanged observed TRAIN episodes.

The existing runtime owns delayed releases and native commits. This adapter
retains paid evidence and full-state guards; it collects no data, evaluates no
policy and grants no final/probe/replay access. Reusing observations is reported
as additional exposure, even though this distinct objective has zero gradient.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from math import isfinite
import pickle
from time import monotonic
from typing import Any

from src.adapters.key_door_reward_pilot import _binding, _check_budget
from src.adapters.torch_demonstration_reward_learner import MomentumOnlyUpdate
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_recurrent_zero_loss import TORCH_RECURRENT_ZERO_LOSS_ID
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.policy_gradient_episode import PolicyGradientEpisode, copy_policy_gradient_episode

RSS_LIMIT = 1024**3
CONTEXTS = ("A", "A", "B", "B", "C_train", "C_train", "A", "A")


def new_momentum_control() -> dict[str, Any]:
    """Describe the fixed eight slots without constructing native resources."""
    return dict(
        status="not_run",
        reason="not reached",
        objective=TORCH_RECURRENT_ZERO_LOSS_ID,
        training=[
            dict(
                index=i,
                context=context,
                cue=("blue", "red")[i % 2],
                episode_id=f"minigrid-reward-train-{i}",
                status="not_run",
                reason="not reached",
                episode=None,
                early_release_receipts=None,
                native_stages=[],
                update_attempted=False,
                update_committed=False,
                commit_validated=False,
                receipt=None,
                native_updates_before=None,
                native_updates_after=None,
                native_training_examples_before=None,
                native_training_examples_after=None,
                readback_error=None,
            )
            for i, context in enumerate(CONTEXTS)
        ],
        work=None,
        budget=None,
        source_unchanged=None,
        actor_unchanged=None,
        readback_error=None,
    )


def _counts(state: Any, updates: int, rows: int) -> None:
    if (
        type(state.completed_updates) is not int
        or state.completed_updates != updates
        or type(state.training_examples) is not int
        or state.training_examples != rows
    ):
        raise ValueError("momentum control native counters differ from paid work")


def _episodes(episodes: tuple) -> tuple[PolicyGradientEpisode, ...]:
    if type(episodes) is not tuple or len(episodes) != 8:
        raise ValueError("momentum control requires exactly eight original TRAIN episodes")
    owned = tuple(copy_policy_gradient_episode(episode) for episode in episodes)
    for i, episode in enumerate(owned):
        if (
            episode.episode_id != f"minigrid-reward-train-{i}"
            or episode.context != CONTEXTS[i]
            or episode.role != "train"
            or episode.num_classes != 7
            or any(len(step.observation) != 148 for step in episode.transitions)
        ):
            raise ValueError("momentum control requires the fixed observed TRAIN148/K7 schedule")
    return owned


def run_momentum_control(
    report: dict[str, Any],
    source: Any,
    episodes: tuple,
    memory: Any,
    deadline: float,
    *,
    check_boundary: Callable[[], None],
) -> dict[str, Any]:
    """Return an owned136 checkpoint and a guard over this completed runtime.

    The caller attests original collection provenance. This helper binds those
    exact records and never substitutes on-policy probabilities for this control.
    Failures publish original paid prefixes before raising; no automatic retry.
    """
    if (
        type(report) is not dict
        or not callable(check_boundary)
        or type(deadline) not in (int, float)
        or not isfinite(deadline)
    ):
        raise ValueError("momentum control requires a report, caller guard and finite deadline")
    owned = _episodes(episodes)
    input_payload = pickle.dumps(episodes, protocol=5)
    input_owners = tuple(id(episode) for episode in episodes)
    report.clear()
    report.update(new_momentum_control())
    report.update(status="running", reason=None)
    records: list[dict[str, Any]] = report["training"]
    record_owners = tuple(records)
    canonical = deepcopy(records)
    payloads = [pickle.dumps(record, protocol=5) for record in records]
    updates = rows = attempted_rows = attempts = tick = 0
    receipts: list[Any] = []
    runtime: Any = None
    budget: Any = None
    initial = previous = birth = last_clock = frozen_clock = frozen_rss = None
    frozen_rss_owner = None
    actor_owner: Any = None
    candidate_owner: Any = None
    inbox_owner: Any = None
    current_update: Any = None
    update_payload = None
    completed_payload = None
    active = -1
    failure: Exception | None = None
    clock = LogicalClock()

    def evidence() -> None:
        if (
            report.get("training") is not records
            or len(records) != 8
            or report.get("objective") != TORCH_RECURRENT_ZERO_LOSS_ID
            or tuple(id(episode) for episode in episodes) != input_owners
            or pickle.dumps(episodes, protocol=5) != input_payload
            or (
                current_update is not None
                and pickle.dumps(current_update, protocol=5) != update_payload
            )
            or any(
                record is not owner or pickle.dumps(record, protocol=5) != payload
                for record, owner, payload in zip(records, record_owners, payloads, strict=True)
            )
            or (
                completed_payload is not None
                and pickle.dumps(report, protocol=5) != completed_payload
            )
        ):
            raise ValueError("momentum control input or preceding paid evidence changed")
        if budget is not None:
            _check_budget(budget, birth, updates, last_clock)
            if budget.process_rss_sampler is not memory or (
                frozen_clock is not None
                and (
                    budget.last_clock != frozen_clock
                    or budget.process_rss_segment is not frozen_rss_owner
                    or not _same_state(frozen_rss, budget.process_rss_segment)
                )
            ):
                raise ValueError("original momentum budget/clock/resource observation changed")
        if runtime is not None and (
            runtime._budget is not budget
            or runtime._inbox._budget is not budget
            or runtime._inbox._clock is not clock
            or type(clock.now()) is not int
            or clock.now() != tick
            or runtime.actor is not actor_owner
            or runtime._candidate is not candidate_owner
            or runtime._inbox is not inbox_owner
            or runtime.actor.version != "a0"
            or runtime._candidate_version != "c0"
            or tuple(runtime._inbox.applied_updates) != tuple(receipts)
        ):
            raise ValueError("original momentum runtime/receipt/clock binding changed")

    def publish(index: int) -> None:
        records[index].clear()
        records[index].update(deepcopy(canonical[index]))
        payloads[index] = pickle.dumps(records[index], protocol=5)

    def proof(*, candidate: bool = True) -> None:
        evidence()
        check_boundary()
        if monotonic() >= deadline:
            raise ValueError("original whole-case deadline exhausted during momentum control")
        if initial is not None and not _same_state(initial, source.snapshot_state()):
            raise ValueError("original warm source changed during momentum control")
        if runtime is not None:
            if not _same_state(initial, runtime.actor.snapshot_state().state):
                raise ValueError("momentum serving actor changed")
            if candidate and not _same_state(previous, runtime.candidate_snapshot().state):
                raise ValueError("preceding momentum candidate changed")
        # Snapshot/caller ports above can mutate records. Pure proof follows the
        # last callback, before another release, native call or publication.
        evidence()

    def guard(*, candidate: bool = True) -> None:
        nonlocal last_clock
        proof(candidate=candidate)
        if budget is not None and frozen_clock is None:
            budget.before_final()
            _check_budget(budget, birth, updates, last_clock)
            last_clock = budget.last_clock
            proof(candidate=candidate)

    try:
        check_boundary()
        evidence()
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise ValueError("whole-case deadline exhausted before momentum control")
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=8,
                max_wall_seconds=min(45.0, remaining),
                max_process_rss_bytes=RSS_LIMIT,
            ),
            monotonic,
        )
        birth, last_clock = _binding(budget), budget.last_clock
        budget.attach_memory(memory)
        guard()
        initial = deepcopy(source.snapshot_state())
        _counts(initial, 128, 1152)
        policy = initial.policy
        if (
            policy.feature_dim,
            policy.hidden_dim,
            policy.num_classes,
            policy.seed,
            policy.learning_rate,
            policy.momentum,
            policy.max_episode_steps,
        ) != (148, 8, 7, 17, 0.03, 0.9, 16):
            raise ValueError("momentum control requires the original warm native policy")
        guard()
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
            raise ValueError("momentum candidate did not begin at the original warm checkpoint")
        guard()
        for index, episode in enumerate(owned):
            guard()
            active = index
            slot = canonical[index]
            _counts(previous, 128 + index, 1152 + rows)
            slot.update(
                status="running",
                reason=None,
                episode=asdict(episode),
                native_updates_before=previous.completed_updates,
                native_training_examples_before=previous.training_examples,
            )
            publish(index)
            update = MomentumOnlyUpdate(episode)
            current_update = update
            update_payload = pickle.dumps(update, protocol=5)
            identity = episode.episode_id
            packet_rows = len(episode.transitions)
            captured = committed = None
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
            evidence()
            slot["early_release_receipts"] = [asdict(value) for value in early]
            publish(index)
            if early:
                raise ValueError("momentum episode trained before its matching release")
            guard()
            clock.advance_to(index + 1)
            tick = index + 1
            runtime.record_label(
                LabelArrival(
                    f"momentum-release-{index}", identity, identity, tick, "a0", identity, "train"
                )
            )
            guard()

            def observe(stage, read):
                nonlocal captured, committed, updates, rows, attempts, attempted_rows
                origin = read()
                slot["native_stages"].append(stage)
                if stage == "started" and not slot["update_attempted"]:
                    slot["update_attempted"] = True
                    attempts += 1
                    attempted_rows += packet_rows
                # A committed receipt is permanent even if the following proof
                # refuses a callback mutation or an invalid native readback.
                if origin.receipt is not None:
                    if captured is None:
                        captured = deepcopy(origin.receipt)
                        receipts.append(captured)
                        updates += 1
                        rows += packet_rows
                    slot.update(receipt=asdict(captured), update_committed=True)
                corrupt = None
                try:
                    evidence()
                except Exception as error:
                    corrupt = error
                publish(index)
                if corrupt is not None:
                    raise corrupt
                guard(candidate=False)
                if (
                    origin.targets != identity
                    or origin.source.sample_id != identity
                    or origin.source.episode_id != identity
                    or not _same_state(update, origin.features)
                    or not _same_state(update, origin.source.features)
                ):
                    raise ValueError("momentum native input differs from original observed episode")
                if origin.receipt is not None:
                    observed = origin.learner.snapshot_state()
                    evidence()
                    _counts(observed, 128 + index + 1, 1152 + rows)
                    expected = dict(
                        sample_id=identity,
                        episode_id=identity,
                        event_id=f"momentum-release-{index}",
                        actor_version="a0",
                        learner_version="c0",
                        observed_at=index,
                        arrived_at=index + 1,
                        applied_at=index + 1,
                        update_number=index + 1,
                    )
                    if (
                        origin.receipt != captured
                        or origin.completed_updates != index + 1
                        or any(getattr(captured, name) != value for name, value in expected.items())
                        or captured.diagnostic.definition != TORCH_RECURRENT_ZERO_LOSS_ID
                        or captured.diagnostic.value != 0.0
                    ):
                        raise ValueError(
                            "momentum native receipt/objective differs from paid zero-loss work"
                        )
                    committed = deepcopy(observed)
                    slot["commit_validated"] = True
                    publish(index)

            returned = runtime.train_ready(max_updates=1, native_observer=observe)
            evidence()
            if len(returned) != 1 or returned[0] != captured or committed is None:
                raise ValueError("momentum update did not return one validated receipt")
            previous = committed
            guard()
            slot.update(
                status="complete",
                native_updates_after=previous.completed_updates,
                native_training_examples_after=previous.training_examples,
            )
            publish(index)
            active = -1
            guard()
        budget.complete_memory()
        guard()
        frozen_clock = budget.last_clock
        frozen_rss_owner = budget.process_rss_segment
        frozen_rss = deepcopy(frozen_rss_owner)
        guard()
    except Exception as error:
        failure = error
        if active >= 0:
            canonical[active].update(status="failed", reason=f"{type(error).__name__}: {error}")

    # Readback faults never erase original commits, and callback-corrupted public
    # rows never become the source of permanent cost or restored paid evidence.
    try:
        if budget is not None and frozen_clock is None:
            budget.record_memory_snapshot()
        if initial is not None:
            report["source_unchanged"] = _same_state(initial, source.snapshot_state())
        if runtime is not None:
            report["actor_unchanged"] = _same_state(initial, runtime.actor.snapshot_state().state)
        if failure is None:
            guard()
    except Exception as error:
        report["readback_error"] = f"{type(error).__name__}: {error}"
        failure = failure or error
    for index in range(8):
        publish(index)
    report.update(
        status="complete" if failure is None else "incomplete",
        reason=None if failure is None else f"{type(failure).__name__}: {failure}",
        work=dict(
            native_updates_attempted=attempts,
            native_updates_committed=updates,
            committed_training_rows=rows,
            attempted_training_rows=attempted_rows,
            observed_recurrent_unroll_steps=rows,
            repeated_observation_exposure_rows=rows,
            environment_steps=0,
            collection_episodes=0,
            failure_work_may_be_unobserved=failure is not None,
        ),
        budget=None
        if budget is None
        else dict(
            updates_completed=updates,
            replay_examples_completed=budget.replay_examples_completed,
            started_at=budget.started_at,
            last_clock=budget.last_clock,
            elapsed_seconds=budget.last_clock - budget.started_at,
            limits=asdict(budget.budget),
            process_rss=None
            if budget.process_rss_segment is None
            else asdict(budget.process_rss_segment),
        ),
    )
    if failure is not None:
        raise RuntimeError(f"momentum control incomplete: {report['reason']}") from failure
    completed_payload = pickle.dumps(report, protocol=5)
    return dict(state=deepcopy(previous), guard=guard)
