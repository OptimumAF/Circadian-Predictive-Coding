"""Bounded continuation of an existing common-MSE BP on released TRAIN rows.

Use existing head-GN and actor/shadow ports. Qualified raw/model binding remains
with the caller; this adapter neither qualifies data nor collects, predicts,
initializes, fits, promotes, restores budgets or retries failed work.
"""

from copy import deepcopy
from dataclasses import asdict
import json
import math
from time import monotonic
from typing import Any, cast

import numpy as np

from src.adapters.continuous_head_learnability import (
    RecordedContinuousHead,
    _native_work,
    head_protocol,
)
from src.adapters.continuous_learnability_demo import ContinuousLearnabilityResult
from src.adapters.continuous_learnability_modes import same_continuous_state
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner, _snapshot
from src.adapters.continuous_mixed_training import (
    RecordedContinuousMixed,
    _admit_maximum_rows,
    mixed_native_work,
    mixed_training_source,
)
from src.app.actor_shadow import ActorShadowRuntime, StableActor
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.continuous_bc import (
    Array,
    OBJECTIVE,
    MIXED_OBJECTIVE,
    WEIGHTED_MIXED_OBJECTIVE,
    _binary_gripper_targets,
    _row_weights,
    _row_masses,
)
from src.core.continuous_demonstration import (
    ContinuousEpisode,
    copy_continuous_episode,
    continuous_training_pairs,
)
from src.core.observation_standardization import standardization_numeric_bytes
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _require_owner(owner, expected, state_owner=None, *, observe=False):
    if type(owner) is not NumpyContinuousBCLearner:
        raise RuntimeError("continuation native BP owner changed")
    before, policy = owner._state, owner._policy
    if state_owner is not None and before is not state_owner:
        raise RuntimeError("continuation native state owner changed")
    if not same_continuous_state(expected, _snapshot(before, policy)):
        raise RuntimeError("continuation native state changed")
    if observe and not same_continuous_state(expected, owner.snapshot_state()):
        raise RuntimeError("continuation public/native snapshot differs")
    if (
        owner._state is not before
        or owner._policy is not policy
        or not same_continuous_state(expected, _snapshot(owner._state, owner._policy))
    ):
        raise RuntimeError("continuation owner changed during observation")


def _require_disjoint(states):
    for index, state in enumerate(states):
        arrays = (*state.parameters, *state.velocity)
        for previous in states[:index]:
            if any(
                np.shares_memory(left, right)
                for left in arrays
                for right in (*previous.parameters, *previous.velocity)
            ):
                raise RuntimeError("continuation model arrays alias a live owner")


def _admit_inputs(training, references, spec, *, maximum=130):
    if type(references) is not tuple or len(references) > 8:
        raise ValueError(
            "continuation references require an exact tuple with zero to eight episodes"
        )
    originals = (training, *references)
    if any(type(value) is not ContinuousEpisode for value in originals):
        raise ValueError("continuation requires exact ContinuousEpisode TRAIN records")
    if any(value.role != "train" or value.spec != spec for value in originals):
        raise ValueError("continuation admits only original same-spec TRAIN episodes")
    episodes = tuple(copy_continuous_episode(value) for value in originals)
    if (
        any(value.role != "train" or value.spec != spec for value in episodes)
        or len({value.episode_id for value in episodes}) != len(episodes)
        or not 1 <= sum(len(value.transitions) for value in episodes) <= maximum
    ):
        raise ValueError(
            f"continuation requires distinct same-spec complete TRAIN episodes and one to {maximum} total rows"
        )
    return originals, episodes


def _describe_inputs(episodes):
    metadata: list[dict[str, Any]] = []
    groups: list[tuple[str, int, int]] = []
    bindings: list[dict[str, Any]] = []
    start = 0
    for index, episode in enumerate(episodes):
        name = "training" if index == 0 else f"reference_{index - 1}"
        stop = start + len(episode.transitions)
        groups.append((name, start, stop))
        item = dict(
            group=name, episode_id=episode.episode_id, context=episode.context, role=episode.role
        )
        metadata.append(dict(item, rows=stop - start))
        bindings.extend(dict(item, step=row.step) for row in episode.transitions)
        start = stop
    return tuple(metadata), tuple(groups), tuple(bindings)


def run_continuous_train_continuation(
    source,
    training: ContinuousEpisode,
    *,
    references: tuple[ContinuousEpisode, ...] = (),
    updates: int = 8,
    update_rule: str = "head_gauss_newton",
    row_weights: Array | None = None,
    maximum_rows: int = 148,
) -> ContinuousLearnabilityResult:
    """Continue released TRAIN data without changing source or actor.

    Admission raises before forks. New rows precede references; derived exposure
    clocks are not original acquisition times. Later failures retain paid work
    and release no learner. No normalization fit or additional prediction occurs.
    Explicit mixed SGD permits at most64 updates for a fixed longer TRAIN fit;
    the default and head-GN bound remain eight to preserve existing scopes.
    Mixed row weights are fixed batch supervision in training/reference order;
    all observations remain exposed and original clocks/authority stay intact.
    Explicit maximum_rows=198 admits both correction releases and old references;
    the default mixed148/head130 scopes and update ceilings remain unchanged.
    """
    if type(update_rule) is not str or update_rule not in ("head_gauss_newton", "bp_mixed"):
        raise ValueError("continuation update_rule must be head_gauss_newton or bp_mixed")
    mixed = update_rule == "bp_mixed"
    maximum_rows = _admit_maximum_rows(maximum_rows)
    if maximum_rows == 198 and not mixed:
        raise ValueError("expanded198-row scope requires explicit bp_mixed continuation")
    row_options = {"maximum_rows": maximum_rows} if maximum_rows == 198 else {}
    weighted = row_weights is not None
    if weighted and not mixed:
        raise ValueError("row weights require explicit bp_mixed continuation")
    objective = WEIGHTED_MIXED_OBJECTIVE if weighted else MIXED_OBJECTIVE if mixed else OBJECTIVE
    maximum_updates = 64 if mixed else 8
    if type(updates) is not int or not 1 <= updates <= maximum_updates:
        raise ValueError(f"continuation updates must be a native integer in 1..{maximum_updates}")
    if type(source) is not NumpyContinuousBCLearner:
        raise ValueError("continuation requires an exact native BP learner")
    source_state, source_policy = source._state, source._policy
    source_standardization = source_policy.observation_standardization
    source_policy_seal = _json(asdict(source_policy))
    initial = _snapshot(source_state, source_policy)
    policy = initial.policy
    if (
        policy.training_role != "train"
        or (
            policy.binary_gripper_axis is not None
            and (
                not mixed
                or type(policy.binary_gripper_axis) is not int
                or policy.binary_gripper_axis != 3
            )
        )
        or policy.spec.observation_dim > 64
        or not 1 <= policy.hidden_dim <= 64
    ):
        raise ValueError(
            "mixed continuation requires TRAIN BP with D/H at most64 and a supported objective"
            if mixed
            else "continuation requires TRAIN common-MSE BP with D/H at most 64"
        )
    # Why: default mixed scope keeps the original148 rows; explicit198 admits
    # both50-row correction releases without deleting old data. Head-GN stays130.
    originals, episodes = (
        _admit_inputs(training, references, policy.spec, maximum=maximum_rows)
        if mixed
        else _admit_inputs(training, references, policy.spec)
    )
    original_episode_seal = _json([asdict(value) for value in originals])
    episode_seal = _json([asdict(value) for value in episodes])
    original_inputs, groups, row_bindings = _describe_inputs(episodes)
    rows = sum(len(value.transitions) for value in episodes)
    new_rows = len(episodes[0].transitions)
    owned_weights = _row_weights(row_weights, rows) if weighted else None
    masses = _row_masses(owned_weights) if weighted else None
    weight_seal = None if owned_weights is None else owned_weights.copy()
    weight_options: dict[str, Any] = {} if owned_weights is None else {"row_weights": owned_weights}

    def require_weights():
        if weighted and (
            weight_seal is None
            or type(row_weights) is not np.ndarray
            or row_weights.dtype != np.float64
            or not np.array_equal(row_weights, weight_seal)
            or type(owned_weights) is not np.ndarray
            or owned_weights.dtype != np.float64
            or not np.array_equal(owned_weights, weight_seal)
        ):
            raise RuntimeError("original or owned continuation row weights changed")

    # Refuse an impossible continuation before any fork or paid proposal.
    counter_limit = 2**53 - 1
    if (
        initial.completed_updates + updates > counter_limit
        or initial.training_examples + updates * rows > counter_limit
    ):
        raise ValueError("continuation would exceed native counter capacity")
    pairs = tuple(continuous_training_pairs(value) for value in episodes)
    x = tuple(row for features, _ in pairs for row in features)
    y = tuple(row for _, targets in pairs for row in targets)
    features, targets = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    owned_features, owned_targets = features.copy(), targets.copy()
    if mixed:
        _binary_gripper_targets(targets, policy.spec, 3)
    if (
        source._state is not source_state
        or source._policy is not source_policy
        or source._policy.observation_standardization is not source_standardization
        or _json(asdict(source_policy)) != source_policy_seal
    ):
        raise ValueError("original source policy or frozen normalization changed during admission")
    _require_owner(source, initial, source_state, observe=True)
    require_weights()
    training_source = mixed_training_source(source) if mixed else source
    training_state, training_policy = training_source._state, training_source._policy
    training_initial = _snapshot(training_state, training_policy)
    training_standardization = training_policy.observation_standardization
    _require_owner(source, initial, source_state)
    require_weights()
    started = monotonic()
    native: list[dict[str, Any]] = []
    facade: RecordedContinuousHead | RecordedContinuousMixed = (
        RecordedContinuousMixed(training_source, native, updates, **weight_options, **row_options)
        if mixed
        else RecordedContinuousHead(source, native, updates)
    )
    clock = LogicalClock()
    clock.advance_to(1000)
    limits = ToyExecutionBudget(max_training_updates=updates)
    budget = ToyBudgetSession(limits, monotonic)
    born = (budget.budget, budget.clock, budget.started_at, budget.progress)
    limit_seal, last_wall = _json(asdict(limits)), budget.last_clock
    _require_owner(source, initial, source_state)
    runtime_options: dict[str, Any] = {}
    if mixed:
        # Why: changing the training objective must not change the stable actor.
        # The runtime already supports an explicitly owned independent actor.
        actor_facade = RecordedContinuousMixed(
            source,
            native,
            updates,
            _records=facade._records,
            _latest=facade._latest,
            **weight_options,
            **row_options,
        )
        runtime_options["actor"] = StableActor(actor_facade, version="actor-0")
    runtime = ActorShadowRuntime(
        facade,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=updates,
        max_consolidations=0,
        **runtime_options,
    )
    actor, inbox = runtime._actor, runtime._inbox
    candidateport = cast(RecordedContinuousHead | RecordedContinuousMixed, runtime._candidate)
    actorport = cast(RecordedContinuousHead | RecordedContinuousMixed, actor._learner)
    raw_actor, raw_candidate = actorport._learner, candidateport._learner
    actor_state, admitted_state = raw_actor._state, raw_candidate._state
    actor_policy, candidate_policy = raw_actor._policy, raw_candidate._policy
    actor_standardization = actor_policy.observation_standardization
    candidate_standardization = candidate_policy.observation_standardization
    seed_states = (training_state,) if mixed else ()
    _require_disjoint((source_state, *seed_states, actor_state, admitted_state))
    admitted, committed, attempts, tick = training_initial, 0, 0, 1000
    receipt_objects: tuple[Any, ...] = ()
    receipt_prefix: list[dict[str, Any]] = []
    native_prefix: list[dict[str, Any]] = []
    owners = (
        runtime._actor,
        runtime._candidate,
        runtime._inbox,
        actor._learner,
        actorport._learner,
        candidateport._learner,
        runtime._attempted_ids,
        runtime._consolidations,
        inbox._applied,
        inbox._experiences,
        inbox._labels,
        inbox._event_ids,
    )
    report: dict[str, Any] = dict(
        protocol=(
            "continuous_train_row_weighted_mixed_gripper_continuation_v1"
            if weighted
            else "continuous_train_mixed_gripper_continuation_v1"
            if mixed
            else "continuous_train_continuation_v1"
        ),
        complete=False,
        head_gauss_newton=None if mixed else head_protocol(),
        rows=rows,
        original_inputs=deepcopy(original_inputs),
        groups=deepcopy(groups),
        row_bindings=deepcopy(row_bindings),
        derived_batch=dict(
            episode_id="continuous-TRAIN-continuation", role="train", qualified_demonstration=False
        ),
        budget=asdict(limits),
        initial_counts=dict(
            updates=initial.completed_updates, training_rows=initial.training_examples
        ),
        final_counts=None,
        derived_exposure_clocks=dict(
            observed_start=1000,
            arrived_start=1001,
            stride=2,
            scope="new released exposure clocks; original teacher metadata unchanged",
        ),
        events=[],
        source_exact=False,
        actor_exact=False,
    )
    metadata_keys: tuple[str, ...] = (
        "protocol",
        "head_gauss_newton",
        "rows",
        "original_inputs",
        "groups",
        "row_bindings",
        "derived_batch",
        "budget",
        "initial_counts",
        "derived_exposure_clocks",
    )
    if mixed:
        report.update(
            training_rule="bp_mixed",
            policy_transition=dict(
                source_objective=(
                    MIXED_OBJECTIVE if policy.binary_gripper_axis == 3 else OBJECTIVE
                ),
                candidate_objective=objective,
                binary_gripper_axis=3,
                parameters="copied",
                momentum="preserved",
                counters="preserved",
                normalization="preserved",
                serving_policy=(
                    "original_mixed" if policy.binary_gripper_axis == 3 else "original_common_mse"
                ),
            ),
        )
        metadata_keys += ("training_rule", "policy_transition")
    if maximum_rows == 198:
        report["row_scope"] = dict(
            maximum_rows=198,
            default_maximum_rows=148,
            scope="explicit expanded released TRAIN inputs; no authority or update-budget renewal",
        )
        metadata_keys += ("row_scope",)
    if owned_weights is not None:
        report["row_weighting"] = dict(
            definition=WEIGHTED_MIXED_OBJECTIVE,
            values=owned_weights.tolist(),
            normalized_masses=([1.0 / rows] * rows if masses is None else masses.tolist()),
            row_order="training then references in declared order; see row_bindings",
            scope="positive TRAIN supervision; exposure counters count actual rows",
        )
        metadata_keys += ("row_weighting",)
    metadata_owners = tuple(report[key] for key in metadata_keys)
    metadata = _json({key: report[key] for key in metadata_keys})
    event_items = report["events"]
    event_seals: list[tuple[Any, str]] = []
    capture_seal = None
    work_seal = None
    detached: NumpyContinuousBCLearner | None = None
    detached_pin: tuple[NumpyContinuousBCLearner, Any, Any] | None = None
    detached_standardization = None
    paid_keys = (
        "native_updates",
        "sealed_native_updates",
        "validated_native_updates",
        "applied_updates",
        "sealed_applied_updates",
        "progress",
        "update_attempts",
        "attempted_training_rows",
        "attempted_new_training_rows",
        "attempted_reference_training_rows",
        "unreturned_training_work_unknown",
    )

    def capture():
        nonlocal capture_seal
        previous = {k: report.get(k) for k in paid_keys}
        changed = capture_seal is not None and _json(previous) != capture_seal
        if changed:
            report.setdefault("corrupted_copied_evidence", deepcopy(previous))
        report.update(
            native_updates=deepcopy(native),
            sealed_native_updates=[json.loads(payload) for _, payload in facade._records],
            validated_native_updates=deepcopy(native_prefix),
            applied_updates=[asdict(v) for v in inbox._applied.values()],
            sealed_applied_updates=deepcopy(receipt_prefix),
            progress=dict(
                updates_completed=budget.updates_completed,
                replay_examples_completed=budget.replay_examples_completed,
                logical_tick=clock._time,
            ),
            update_attempts=attempts,
            attempted_training_rows=attempts * rows,
            attempted_new_training_rows=attempts * new_rows,
            attempted_reference_training_rows=attempts * (rows - new_rows),
            unreturned_training_work_unknown=attempts > len(native),
        )
        capture_seal = _json({k: report[k] for k in paid_keys})
        if changed:
            raise RuntimeError("preceding copied paid continuation evidence changed")

    def require(*, observe=False):
        nonlocal last_wall
        current_owners = (
            runtime._actor,
            runtime._candidate,
            runtime._inbox,
            actor._learner,
            actorport._learner,
            candidateport._learner,
            runtime._attempted_ids,
            runtime._consolidations,
            inbox._applied,
            inbox._experiences,
            inbox._labels,
            inbox._event_ids,
        )
        if any(a is not b for a, b in zip(owners, current_owners, strict=True)):
            raise RuntimeError("continuation runtime owner was replaced")
        if (
            source._policy is not source_policy
            or source._policy.observation_standardization is not source_standardization
            or _json(asdict(source._policy)) != source_policy_seal
            or raw_actor._policy.observation_standardization is not actor_standardization
            or raw_candidate._policy.observation_standardization is not candidate_standardization
            or raw_actor._policy is not actor_policy
            or raw_candidate._policy is not candidate_policy
            or budget.budget is not born[0]
            or budget.clock is not born[1]
            or budget.progress is not born[3]
            or type(budget.started_at) is not float
            or budget.started_at != born[2]
            or _json(asdict(limits)) != limit_seal
            or runtime._budget is not budget
            or inbox._budget is not budget
            or inbox._clock is not clock
            or inbox._learner is not candidateport
            or type(clock._time) is not int
            or clock._time != tick
            or type(budget.updates_completed) is not int
            or budget.updates_completed != committed
            or type(budget.replay_examples_completed) is not int
            or budget.replay_examples_completed != 0
            or type(budget.last_clock) is not float
            or not math.isfinite(budget.last_clock)
            or budget.last_clock < last_wall
            or runtime._consolidation_limit != 0
            or runtime._attempted_ids
            or runtime._consolidations
            or runtime._stopped is not False
            or runtime._retired is not False
            or inbox._stopped is not False
            or runtime._candidate_version != "candidate-0"
            or inbox._learner_version != "candidate-0"
            or actor._version != "actor-0"
            or runtime._training_role != "train"
            or inbox._training_role != "train"
            or type(inbox._capacity) is not int
            or inbox._capacity != updates
        ):
            raise RuntimeError("original continuation budget/clock/authority changed")
        if mixed and (
            training_source._policy is not training_policy
            or training_policy.observation_standardization is not training_standardization
        ):
            raise RuntimeError("private mixed training seed policy changed")
        if mixed:
            _require_owner(training_source, training_initial, training_state, observe=observe)
        last_wall = budget.last_clock
        for port in (facade, actorport, candidateport):
            port.require_reports(native)
        candidateport.require_latest(admitted)
        _require_owner(source, initial, source_state, observe=observe)
        _require_owner(raw_actor, initial, actor_state, observe=observe)
        _require_owner(raw_candidate, admitted, admitted_state, observe=observe)
        current = tuple(inbox._applied.values())
        if (
            len(current) != len(receipt_objects)
            or any(a is not b for a, b in zip(current, receipt_objects, strict=True))
            or _json([asdict(v) for v in current]) != _json(receipt_prefix)
            or _json(native) != _json(native_prefix)
        ):
            raise RuntimeError("preceding native/receipt continuation evidence changed")
        if (
            report["events"] is not event_items
            or len(event_items) != len(event_seals)
            or any(
                event_items[i] is not obj or _json(obj) != payload
                for i, (obj, payload) in enumerate(event_seals)
            )
        ):
            raise RuntimeError("original continuation event prefix changed")
        if (
            capture_seal is not None
            and _json({k: report.get(k) for k in paid_keys}) != capture_seal
        ):
            raise RuntimeError("copied paid evidence changed")
        if (
            _json([asdict(value) for value in originals]) != original_episode_seal
            or _json([asdict(value) for value in episodes]) != episode_seal
            or not np.array_equal(features, owned_features)
            or not np.array_equal(targets, owned_targets)
            or any(
                report[key] is not owner
                for key, owner in zip(metadata_keys, metadata_owners, strict=True)
            )
            or metadata != _json({key: report[key] for key in metadata_keys})
        ):
            raise RuntimeError("original TRAIN input or continuation metadata changed")
        require_weights()
        for i, event in enumerate(inbox._experiences.values()):
            if (
                event.sample_id != f"continuation-{i}"
                or event.episode_id != "continuous-TRAIN-continuation"
                or event.observed_at != 1000 + 2 * i
                or event.role != "train"
                or event.model_version != "actor-0"
                or event.permissions != ExperiencePermissions(training=True)
                or not np.array_equal(event.features, owned_features)
            ):
                raise RuntimeError("retained continuation TRAIN exposure changed")
        for i, label in enumerate(inbox._labels.values()):
            if (
                label.sample_id != f"continuation-{i}"
                or label.event_id != f"continuation-label-{i}"
                or label.episode_id != "continuous-TRAIN-continuation"
                or label.arrived_at != 1001 + 2 * i
                or label.role != "train"
                or label.model_version != "actor-0"
                or not np.array_equal(label.targets, owned_targets)
            ):
                raise RuntimeError("retained continuation TRAIN target changed")
        # Budget/public snapshot getters above can mutate an earlier owner.
        # Close with native fields only, including the frozen transform identity.
        if (
            source._policy is not source_policy
            or source_policy.observation_standardization is not source_standardization
            or _json(asdict(source_policy)) != source_policy_seal
            or raw_actor._policy is not actor_policy
            or actor_policy.observation_standardization is not actor_standardization
            or raw_candidate._policy is not candidate_policy
            or candidate_policy.observation_standardization is not candidate_standardization
        ):
            raise RuntimeError("continuation policy or frozen normalization owner changed")
        _require_owner(source, initial, source_state)
        if mixed:
            _require_owner(training_source, training_initial, training_state)
        _require_owner(raw_actor, initial, actor_state)
        _require_owner(raw_candidate, admitted, admitted_state)
        _require_disjoint((source._state, *seed_states, raw_actor._state, raw_candidate._state))

    def stable():
        require(observe=True)
        # A later public snapshot may alter an earlier native owner: close purely.
        require()

    def require_detached(*, observe=False):
        if (
            detached_pin is None
            or detached is not detached_pin[0]
            or detached._policy is not detached_pin[2]
            or detached._policy.observation_standardization is not detached_standardization
        ):
            raise RuntimeError("detached continuation owner or policy changed")
        _require_owner(detached, admitted, detached_pin[1], observe=observe)
        _require_disjoint(
            (source._state, *seed_states, raw_actor._state, raw_candidate._state, detached._state)
        )

    def observer(stage, read_origin):
        nonlocal admitted, admitted_state, committed, receipt_objects, receipt_prefix, native_prefix
        if stage == "started":
            # This hook runs after the original budget observer, before native math.
            require()
            origin = read_origin()
            if (
                origin.learner is not candidateport
                or not np.array_equal(origin.features, owned_features)
                or not np.array_equal(origin.targets, owned_targets)
            ):
                raise RuntimeError("actual native input/owner differs")
        elif stage == "completed":
            observed = _snapshot(raw_candidate._state, raw_candidate._policy)
            current = tuple(inbox._applied.values())
            capture()  # Returned-native work and receipt are paid even if checks fail.
            candidateport.require_reports(native)
            candidateport.require_latest(observed)
            if (
                len(current) != committed + 1
                or len(native) != committed + 1
                or any(a is not b for a, b in zip(current[:-1], receipt_objects, strict=True))
                or _json(native[:-1]) != _json(native_prefix)
                or _json([asdict(v) for v in current[:-1]]) != _json(receipt_prefix)
            ):
                raise RuntimeError("native publication replaced preceding evidence")
            receipt = current[-1]
            i = committed
            if (
                receipt.sample_id != f"continuation-{i}"
                or receipt.event_id != f"continuation-label-{i}"
                or receipt.episode_id != "continuous-TRAIN-continuation"
                or (receipt.observed_at, receipt.arrived_at, receipt.applied_at)
                != (1000 + 2 * i, 1001 + 2 * i, 1001 + 2 * i)
                or receipt.update_number != i + 1
                or receipt.actor_version != "actor-0"
                or receipt.learner_version != "candidate-0"
                or receipt.diagnostic.definition != objective
                or asdict(receipt.diagnostic) != native[-1]["diagnostic"]
                or native[-1]["training_rows"] != rows
                or type(native[-1]["work"].get("published_updates")) is not int
                or native[-1]["work"]["published_updates"] != 1
                or (
                    not mixed
                    and any(
                        not np.array_equal(observed.parameters[j], initial.parameters[j])
                        or not np.array_equal(observed.velocity[j], initial.velocity[j])
                        for j in (0, 1)
                    )
                )
                or observed.policy != training_initial.policy
                or (observed.completed_updates, observed.training_examples)
                != (initial.completed_updates + i + 1, initial.training_examples + (i + 1) * rows)
            ):
                raise RuntimeError("continuation returned receipt/state differs")
            admitted, admitted_state, committed = observed, raw_candidate._state, i + 1
            receipt_objects, receipt_prefix = current, [asdict(v) for v in current]
            native_prefix = deepcopy(native)
            capture()
            require()

    detached = None
    try:
        capture()
        stable()
        for i in range(updates):
            tick = 1000 + 2 * i
            clock.advance_to(tick)
            capture()
            stable()
            event = dict(
                index=i,
                observed_at=tick,
                arrived_at=tick + 1,
                deferred_exact=False,
                update_attempted=False,
                success_validated=False,
            )
            event_items.append(event)
            event_seals.append((event, _json(event)))
            runtime.record_experience(
                Experience(
                    f"continuation-{i}",
                    "continuous-TRAIN-continuation",
                    tick,
                    "actor-0",
                    features,
                    "train",
                    ExperiencePermissions(training=True),
                )
            )
            stable()
            if runtime.train_ready(max_updates=1) != ():
                raise RuntimeError("unarrived labels admitted native work")
            stable()
            event["deferred_exact"] = True
            event_seals[-1] = (event, _json(event))
            tick += 1
            clock.advance_to(tick)
            capture()
            runtime.record_label(
                LabelArrival(
                    f"continuation-label-{i}",
                    f"continuation-{i}",
                    "continuous-TRAIN-continuation",
                    tick,
                    "actor-0",
                    targets,
                    "train",
                )
            )
            stable()
            attempts += 1
            event["update_attempted"] = True
            event_seals[-1] = (event, _json(event))
            try:
                returned = runtime.train_ready(max_updates=1, native_observer=observer)
            finally:
                capture()
            if len(returned) != 1 or returned[0] is not receipt_objects[-1]:
                raise RuntimeError("original returned receipt differs")
            event["success_validated"] = True
            event_seals[-1] = (event, _json(event))
            stable()
        if runtime.train_ready(max_updates=1) != ():
            raise RuntimeError("continuation has unexpected remaining work")
        stable()
        detached = training_source.fork()
        if type(detached) is not NumpyContinuousBCLearner or any(
            detached is v for v in (source, training_source, raw_actor, raw_candidate)
        ):
            raise RuntimeError("detached continuation aliases a live owner")
        detached_policy = detached._policy
        detached_standardization = detached_policy.observation_standardization
        detached.restore_state(admitted)
        if (
            detached._policy is not detached_policy
            or detached_policy.observation_standardization is not detached_standardization
        ):
            raise RuntimeError("detached policy changed during restore")
        detached_pin = (detached, detached._state, detached_policy)
        require_detached(observe=True)
        stable()
        counts, flags, weights = (
            mixed_native_work(native, **row_options) if mixed else _native_work(native)
        )
        report["work"] = dict(
            optimizer_updates=committed,
            head_gauss_newton_updates=0 if mixed else committed,
            training_rows=committed * rows,
            new_training_rows=committed * new_rows,
            reference_training_rows=committed * (rows - new_rows),
            group_training_exposures={
                name: committed * (stop - start) for name, start, stop in groups
            },
            native_update_counts=counts,
            native_work_flags=flags,
            native_event_row_weights=weights,
            initializations=0,
            fits=0,
            sgd_updates=committed if mixed else 0,
            prediction_calls=0,
            solver_work_known=mixed,
            named_model_owners=5 if mixed else 4,
            retained_native_records=len(native),
            latest_native_model_pins=1,
            inbox_feature_target_numeric_bytes=committed
            * rows
            * (policy.spec.observation_dim + len(policy.spec.action_units))
            * 8,
            parameter_momentum_numeric_bytes_per_owner=16
            * sum(array.size for array in initial.parameters),
            frozen_standardization_numeric_bytes_per_owner=standardization_numeric_bytes(
                policy.observation_standardization
            ),
            scope="selected native work; solver workspace/arithmetic, validation, copies and Python/RSS additional",
        )
        if mixed:
            report["work"]["scope"] = (
                "successful SGD work source-derived from unchanged native train_batch; "
                "no solver or extra loss readout; validation/copies/Python/RSS additional"
            )
        if weighted:
            report["work"].update(
                objective_weight_values_per_update=rows,
                retained_receipt_objective_weight_values=committed * rows,
                retained_receipt_objective_weight_numeric_bytes=committed * rows * 8,
                scope=(
                    "selected successful native SGD and weight-input counts; "
                    "native mass-admission calls source-derived; facade/metadata "
                    "validation, normalization arithmetic, copies, JSON/Python/RSS additional"
                ),
            )
        report["final_counts"] = dict(
            updates=admitted.completed_updates, training_rows=admitted.training_examples
        )
        work_seal = _json((report["work"], report["final_counts"]))
        report["elapsed_seconds"] = monotonic() - started
        stable()
        require_detached(observe=True)
        require()
        require_detached()
        if _json((report["work"], report["final_counts"])) != work_seal:
            raise RuntimeError("final continuation work changed")
        report.update(source_exact=True, actor_exact=True, complete=True)
    except Exception as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}")
        if hasattr(error, "work"):
            report["failed_native_work"] = deepcopy(error.work)
            if weighted and hasattr(error, "definition"):
                report["failed_native_definition"] = error.definition
        detached = None
    finally:
        try:
            capture()
        except Exception as error:
            report.update(complete=False, capture_error=f"{type(error).__name__}: {error}")
            detached = None
        if "elapsed_seconds" not in report:
            report["elapsed_seconds"] = monotonic() - started
        if report["complete"]:
            try:
                require()
                require_detached()
                if _json((report["work"], report["final_counts"])) != work_seal:
                    raise RuntimeError("final continuation work changed")
            except Exception as error:
                report.update(complete=False, final_error=f"{type(error).__name__}: {error}")
                detached = None
    return ContinuousLearnabilityResult(deepcopy(report), detached if report["complete"] else None)
