"""Two isolated original mean-CE cue updates from one accepted TRAIN checkpoint.

Inputs are the warm source/full state, six original TRAIN packets and their paid
baseline scores. Output retains both finite-update effects and separate costs.
No source update, environment, final/probe access, optimizer reset or selection
belongs here. Carried momentum is part of each original SGD update.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from math import isfinite
import pickle
from time import monotonic
from typing import Any

from src.adapters import minigrid_learning_pilot as shared
from src.adapters.key_door_reward_pilot import _binding, _check_budget
from src.adapters.minigrid_gradient_readouts import validate_sources
from src.adapters.minigrid_learnability import _score_checkpoint
from src.adapters.torch_demonstration_reward_learner import DemonstrationUpdate
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_recurrent_learner import TORCH_RECURRENT_CE_ID
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.learner_ports import TrainingDiagnostic

RSS_LIMIT = 768 * 1024**2
SCORE_KEYS = (
    "score_forward_calls_attempted",
    "score_forward_calls_completed",
    "score_rows_attempted",
    "score_rows_completed",
)


def _baseline(baseline: dict[str, Any], sources: tuple) -> None:
    if (
        type(baseline) is not dict
        or baseline.get("status") != "complete"
        or baseline.get("method") != "recurrent_bc"
        or baseline.get("role") != "train"
        or type(baseline.get("updates")) is not int
        or baseline["updates"] != 128
        or type(baseline.get("scores")) is not list
        or len(baseline["scores"]) != 6
    ):
        raise ValueError("cue updates require the accepted complete128 TRAIN raw baseline")
    for score, packet in zip(baseline["scores"], sources, strict=True):
        episode = shared._episode(packet)
        if (
            type(score) is not dict
            or score.get("context") != packet.context
            or score.get("cue") != episode.episode_id.rsplit("-", 1)[-1]
            or score.get("episode_id") != episode.episode_id
            or score.get("row_ids") != list(packet.row_ids)
            or type(score.get("ce")) not in (int, float)
            or not isfinite(score["ce"])
            or score["ce"] < 0
        ):
            raise ValueError("baseline CE/identity must match each original ordered TRAIN packet")


def _deltas(baseline: dict[str, Any], arms: list[dict[str, Any]], sources: tuple) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in arms:
        entries = []
        readout = arm["readout"]
        if readout["status"] != "complete" or len(readout["scores"]) != 6:
            raise ValueError("finite-update deltas require both complete six-packet readouts")
        for old, score, packet in zip(baseline["scores"], readout["scores"], sources, strict=True):
            if (
                score["episode_id"] != old["episode_id"]
                or score["context"] != old["context"]
                or score["cue"] != old["cue"]
                or score["row_ids"] != list(packet.row_ids)
                or type(score["ce"]) not in (int, float)
                or not isfinite(score["ce"])
                or score["ce"] < 0
            ):
                raise ValueError("post-update score identity/CE differs from original TRAIN packet")
            entries.append(
                dict(
                    context=score["context"],
                    cue=score["cue"],
                    episode_id=score["episode_id"],
                    baseline_ce=old["ce"],
                    after_ce=score["ce"],
                    ce_delta=score["ce"] - old["ce"],
                )
            )
        result[arm["cue"]] = entries
    result["blue_minus_red"] = [
        dict(
            context=blue["context"],
            cue=blue["cue"],
            episode_id=blue["episode_id"],
            ce_difference=blue["after_ce"] - red["after_ce"],
        )
        for blue, red in zip(result["blue"], result["red"], strict=True)
    ]
    return result


def run_cue_updates(
    report: dict[str, Any],
    source: Any,
    state: Any,
    sources: tuple,
    baseline: dict[str, Any],
    budget: ToyBudgetSession,
    *,
    check_boundary: Callable[[], None],
) -> None:
    """Apply one mean-CE update per independent cue fork; retain paid failures.

    The caller's zero-update readout budget and warm guard remain authoritative.
    A separate two-update budget accounts for these disposable training steps.
    Unknown input/state mutations are refused, never restored on caller owners.
    """
    if type(report) is not dict or report or not callable(check_boundary):
        raise ValueError("cue updates require an empty report and callable boundary")
    signature = validate_sources(sources)
    _baseline(baseline, sources)
    shared._counter(state, 128)
    caller_birth = _binding(budget)
    _check_budget(budget, caller_birth, 0)
    if budget.budget.max_training_updates != 0 or budget.process_rss_sampler is None:
        raise ValueError("cue updates require the original zero-update readout/RSS budget")
    memory_owner = budget.process_rss_sampler
    expected_state = deepcopy(state)
    baseline_payload = pickle.dumps(baseline, protocol=5)
    baseline_scores = tuple(baseline["scores"])
    arms: list[dict[str, Any]] = [
        dict(
            cue=cue,
            episode_id=shared._episode(sources[index]).episode_id,
            status="not_run",
            reason="not reached",
            diagnostic=None,
            update_attempted=False,
            update_committed=False,
            commit_validated=False,
            native_updates_before=None,
            native_updates_after=None,
            native_training_examples_before=None,
            native_training_examples_after=None,
            readout=None,
        )
        for index, cue in enumerate(("blue", "red"))
    ]
    report.update(status="running", reason=None, arms=arms, work=None, budget=None, deltas=None)
    owners = tuple(arms)
    canonical = deepcopy(arms)
    payloads = [pickle.dumps(arm, protocol=5) for arm in arms]
    forks: list[Any] = []
    fork_states: list[Any] = []
    own_budget: Any = None
    own_birth = last_clock = frozen_clock = frozen_rss = None
    frozen_rss_owner = None
    recorded_budget = None
    caller_clock = budget.last_clock
    attempted = committed = attempted_rows = committed_rows = 0
    fork_attempted = fork_returned = 0
    active = -1
    current_update = update_payload = None
    problem = None

    def evidence() -> None:
        if (
            report.get("status") != "running"
            or report.get("reason") is not None
            or report.get("work") is not None
            or report.get("budget") is not None
            or report.get("deltas") is not None
            or budget.process_rss_sampler is not memory_owner
            or report.get("arms") is not arms
            or len(arms) != 2
            or any(
                arm is not owner or pickle.dumps(arm, protocol=5) != payload
                for arm, owner, payload in zip(arms, owners, payloads, strict=True)
            )
            or pickle.dumps(baseline, protocol=5) != baseline_payload
            or any(
                score is not owner
                for score, owner in zip(baseline["scores"], baseline_scores, strict=True)
            )
            or not shared._check_signature(sources, signature)
            or not _same_state(expected_state, state)
            or (
                current_update is not None
                and pickle.dumps(current_update, protocol=5) != update_payload
            )
        ):
            raise ValueError("original cue-update inputs or preceding paid evidence changed")
        _check_budget(budget, caller_birth, 0, caller_clock)
        if own_budget is not None:
            _check_budget(own_budget, own_birth, committed, last_clock)
            if own_budget.process_rss_sampler is not memory_owner or (
                frozen_clock is not None
                and (
                    own_budget.last_clock != frozen_clock
                    or own_budget.process_rss_segment is not frozen_rss_owner
                    or not _same_state(frozen_rss, own_budget.process_rss_segment)
                )
            ):
                raise ValueError("original cue-update budget/clock/resource binding changed")

    def publish(index: int) -> None:
        arms[index].clear()
        arms[index].update(deepcopy(canonical[index]))
        payloads[index] = pickle.dumps(arms[index], protocol=5)

    def proof() -> None:
        evidence()
        check_boundary()
        if not _same_state(expected_state, source.snapshot_state()):
            raise ValueError("original warm source full state changed during cue updates")
        for fork, expected in zip(forks, fork_states, strict=True):
            if not _same_state(expected, fork.snapshot_state()):
                raise ValueError("original or preceding updated cue fork changed")
        evidence()

    def guard() -> None:
        nonlocal caller_clock, last_clock, recorded_budget
        proof()
        budget.before_final()
        _check_budget(budget, caller_birth, 0, caller_clock)
        caller_clock = budget.last_clock
        if own_budget is not None and frozen_clock is None:
            own_budget.before_final()
            _check_budget(own_budget, own_birth, committed, last_clock)
            last_clock = own_budget.last_clock
        proof()
        if own_budget is not None:
            recorded_budget = dict(
                limits=asdict(own_budget.budget),
                started_at=own_budget.started_at,
                last_clock=own_budget.last_clock,
                elapsed_seconds=own_budget.last_clock - own_budget.started_at,
                updates_completed=committed,
                replay_examples_completed=own_budget.replay_examples_completed,
                process_rss=None
                if own_budget.process_rss_segment is None
                else asdict(own_budget.process_rss_segment),
            )

    try:
        guard()
        remaining = 20.0
        if budget.budget.max_wall_seconds is not None:
            remaining = min(
                remaining,
                max(0.0, budget.started_at + budget.budget.max_wall_seconds - monotonic()),
            )
        own_budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=2, max_wall_seconds=remaining, max_process_rss_bytes=RSS_LIMIT
            ),
            monotonic,
        )
        own_birth, last_clock = _binding(own_budget), own_budget.last_clock
        own_budget.attach_memory(budget.process_rss_sampler)
        guard()
        for _ in range(2):
            guard()
            fork_attempted += 1
            fork = source.fork()
            fork_returned += 1
            if fork is source or any(fork is prior for prior in forks):
                raise ValueError("cue-update forks must be independent from source and each other")
            actual = fork.snapshot_state()
            if not _same_state(expected_state, actual):
                raise ValueError(
                    "cue fork full parameters/optimizer/gradients differ before update"
                )
            forks.append(fork)
            fork_states.append(deepcopy(actual))
            guard()
        for index, fork in enumerate(forks):
            guard()
            active = index
            arm = canonical[index]
            arm.update(
                status="running",
                reason=None,
                native_updates_before=128,
                native_training_examples_before=1152,
            )
            publish(index)
            episode = shared._episode(sources[index])
            current_update = DemonstrationUpdate(episode)
            update_payload = pickle.dumps(current_update, protocol=5)
            own_budget.before_update()
            guard()
            attempted += 1
            attempted_rows += 9
            arm["update_attempted"] = True
            publish(index)
            diagnostic = fork.train_batch(current_update, episode.episode_id)
            # A returned native update and its charge survive every later refusal.
            committed += 1
            committed_rows += 9
            arm["update_committed"] = True
            valid_diagnostic = (
                type(diagnostic) is TrainingDiagnostic
                and set(vars(diagnostic)) == {"definition", "value"}
                and diagnostic.definition == TORCH_RECURRENT_CE_ID
                and type(diagnostic.value) in (int, float)
                and isfinite(diagnostic.value)
                and diagnostic.value >= 0
            )
            if valid_diagnostic:
                arm["diagnostic"] = asdict(diagnostic)
            own_budget.record_update()
            _check_budget(own_budget, own_birth, committed, last_clock)
            if recorded_budget is not None:
                recorded_budget["updates_completed"] = own_budget.updates_completed
            corrupt = None
            try:
                evidence()
            except Exception as error:
                corrupt = error
            publish(index)
            if corrupt is not None:
                raise corrupt
            if not valid_diagnostic:
                raise ValueError("cue update must return the original native mean-CE diagnostic")
            actual = fork.snapshot_state()
            if (
                type(actual.completed_updates) is not int
                or actual.completed_updates != 129
                or type(actual.training_examples) is not int
                or actual.training_examples != 1161
            ):
                raise ValueError("cue update native counters must be129/1161")
            fork_states[index] = deepcopy(actual)
            arm.update(
                commit_validated=True, native_updates_after=129, native_training_examples_after=1161
            )
            publish(index)
            guard()
            current = _score_checkpoint(
                fork,
                deepcopy(actual),
                method="recurrent_bc",
                updates=129,
                sources=sources,
                budget=budget,
                check_boundary=guard,
            )
            corrupt = None
            try:
                evidence()
            except Exception as error:
                corrupt = error
            arm["readout"] = deepcopy(current)
            publish(index)
            if corrupt is not None:
                raise corrupt
            guard()
            if current["status"] != "complete":
                raise ValueError("cue-update raw readout incomplete; paid prefix retained")
            arm.update(status="complete", reason=None)
            publish(index)
            active = -1
            guard()
        own_budget.complete_memory()
        guard()
        frozen_clock = own_budget.last_clock
        frozen_rss_owner = own_budget.process_rss_segment
        frozen_rss = deepcopy(frozen_rss_owner)
        guard()
        deltas = _deltas(pickle.loads(baseline_payload), canonical, sources)
        guard()
    except Exception as error:
        problem = f"{type(error).__name__}: {error}"
        deltas = None
        if active >= 0:
            canonical[active].update(status="failed", reason=problem)
    # Private paid records, rather than callback-mutated public dictionaries,
    # supply all final costs. Caller-owned source/baseline changes stay refused.
    raw_work = [arm["readout"]["work"] for arm in canonical if arm["readout"] is not None]
    work = dict(
        native_updates_attempted=attempted,
        native_updates_committed=committed,
        attempted_training_rows=attempted_rows,
        committed_training_rows=committed_rows,
        fork_calls_attempted=fork_attempted,
        fork_calls_returned=fork_returned,
        **{key: sum(value.get(key, 0) for value in raw_work) for key in SCORE_KEYS},
        observed_recurrent_unroll_steps=committed_rows
        + sum(value.get("score_rows_completed", 0) for value in raw_work),
        failure_work_may_be_unobserved=problem is not None
        or any(value.get("failure_work_may_be_unobserved", False) for value in raw_work),
    )
    report.update(
        arms=deepcopy(canonical),
        status="complete" if problem is None else "incomplete",
        reason=problem,
        work=work,
        budget=recorded_budget,
        deltas=deltas,
    )
