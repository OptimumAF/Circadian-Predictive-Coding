"""Bounded cumulative TRAIN releases through the existing actor/shadow runtime.

Owns episode admission, a frozen nearest baseline and delayed packet sequencing.
Readout callbacks receive detached learners and bounded metadata, never runtime
authority. No simulator, IO, promotion, replay, probe or final evaluation lives
here. The two-episode adapter delegates only its explicitly selected sequence.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
import json
from time import monotonic
from typing import Any, cast

import numpy as np

from src.adapters.numpy_continuous_bc import fit_training_standardization
from src.adapters.numpy_nearest_continuous_bc import (
    OBJECTIVE,
    FixedInputNearestContinuousBCPolicy,
    NearestContinuousBCPolicy,
    NearestContinuousBCSnapshot,
    NumpyNearestContinuousBCLearner,
    _snapshot,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.continuous_demonstration import (
    ContinuousEpisode,
    continuous_training_pairs,
    copy_continuous_episode,
)
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.observation_standardization import standardize_observations

Observer = Callable[[NumpyNearestContinuousBCLearner, dict[str, Any]], dict[str, Any]]


def _episodes(
    warm: ContinuousEpisode, releases: tuple[ContinuousEpisode, ...], *, maximum_releases=3
):
    if type(releases) is not tuple or not 1 <= len(releases) <= maximum_releases:
        raise ValueError(f"sequence requires1..{maximum_releases} new TRAIN releases")
    records = (warm, *releases)
    if any(type(item) is not ContinuousEpisode or item.role != "train" for item in records):
        raise ValueError("sequence requires exact released TRAIN episodes")
    owned = tuple(copy_continuous_episode(item) for item in records)
    if any(item.spec != owned[0].spec for item in owned):
        raise ValueError("sequence episodes must have the same spec")
    if len({item.episode_id for item in owned}) != len(owned):
        raise ValueError("sequence episodes require distinct identities")
    if sum(len(item.transitions) for item in owned) > 256:
        raise ValueError("sequence exceeds256 prototype rows")
    return owned


def _same(first: NearestContinuousBCSnapshot, second: NearestContinuousBCSnapshot) -> bool:
    return (
        first.policy == second.policy
        and first.completed_updates == second.completed_updates
        and first.training_examples == second.training_examples
        and np.array_equal(first.prototype_features, second.prototype_features)
        and np.array_equal(first.prototype_actions, second.prototype_actions)
    )


def _owned_state(learner: NumpyNearestContinuousBCLearner, expected):
    if type(learner) is not NumpyNearestContinuousBCLearner or learner._policy != expected.policy:
        raise ValueError("sequence learner owner or frozen policy changed")
    if not _same(_snapshot(learner._state, learner._policy), expected):
        raise ValueError("sequence native state changed outside its declared update")


def _json_result(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("observer must return a finite JSON dictionary")
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=True)
    if len(encoded.encode("utf8")) > 256 * 1024:
        raise ValueError("observer result exceeds256KiB")
    return json.loads(encoded)


class _Sequence:
    def __init__(self, records, observer: Observer | None, *, fixed_input: bool = False):
        self.records, self.observer = records, observer
        self.fixed_input = fixed_input
        self.work = dict(
            fit_attempts=0 if fixed_input else 1,
            fit_returns=0,
            fit_rows=0 if fixed_input else len(records[0].transitions),
            model_initializations=0,
            model_initialization_attempts=0,
            model_random_draws=0,
            warm_append_attempts=0,
            warm_append_returns=0,
            warm_append_rows=0,
            runtime_append_attempts=0,
            runtime_append_returns=0,
            runtime_append_rows=0,
            validated_updates=0,
            observer_attempts=0,
            observer_returns=0,
            observer_validated=0,
            optimizer_updates=0,
            replayed_training_rows=0,
            validation_normalization_rows=0,
        )
        self.report: dict[str, Any] = dict(
            schema_version=1,
            protocol_id="continuous_nearest_sequence_v1",
            scope="saved_train_integration",
            complete=False,
            events=[],
            checkpoints=[],
            applied=[],
            work=self.work,
        )
        if fixed_input:
            self.report["fixed_input"] = True
            self.report["input_normalization"] = dict(
                definition="continuous_nearest_explicit_fixed_input_identity_v1",
                empirical_fit=False,
                policy_standardization=None,
                normalization_calls="identity_copy_of_already_encoded_inputs",
            )
        self.tick = 0

    def setup(self):
        records = self.records
        policy: NearestContinuousBCPolicy
        if self.fixed_input:
            # Why: already encoded physical inputs must not acquire an empirical
            # transform merely to enter the existing delayed-release sequence.
            policy = FixedInputNearestContinuousBCPolicy(records[0].spec)
        else:
            statistics = fit_training_standardization(records[0])
            self.work["fit_returns"] = 1
            policy = NearestContinuousBCPolicy(records[0].spec, statistics)
        self.work["model_initialization_attempts"] = 1
        self.source = NumpyNearestContinuousBCLearner(policy)
        self.work["model_initializations"] = 1
        self.work["warm_append_attempts"] = 1
        self.source.train_episode(records[0])
        self.work["warm_append_returns"] = 1
        self.work["warm_append_rows"] = len(records[0].transitions)
        self.initial = self.source.snapshot_state()
        self.expected = deepcopy(self.initial)
        self.clock = LogicalClock(0)
        self.limit = ToyExecutionBudget(max_training_updates=len(records) - 1)
        self.birth_limit = deepcopy(self.limit)
        self.budget = ToyBudgetSession(self.limit, monotonic)
        self.runtime = ActorShadowRuntime(
            self.source,
            actor_version="actor-0",
            candidate_version="candidate-0",
            clock=self.clock,
            budget=self.budget,
            max_experiences=len(records) - 1,
            max_consolidations=0,
        )
        self.actor = self.runtime._actor
        self.candidate = cast(NumpyNearestContinuousBCLearner, self.runtime._candidate)
        self.backend = cast(NumpyNearestContinuousBCLearner, self.actor._learner)
        self.inbox = self.runtime._inbox
        self.budget_clock, self.budget_started = self.budget.clock, self.budget.started_at
        self.receipts: tuple = ()
        self.receipt_owners: tuple = ()
        self.known = [records[0].episode_id]
        self.report["policy"] = asdict(policy)

    def guard(self):
        """Check the owned native state after all externally supplied callbacks."""
        runtime = self.runtime
        if (
            runtime._actor is not self.actor
            or self.actor._learner is not self.backend
            or runtime._candidate is not self.candidate
            or runtime._inbox is not self.inbox
            or runtime._budget is not self.budget
            or self.inbox._budget is not self.budget
            or self.inbox._clock is not self.clock
            or self.inbox._learner is not self.candidate
            or self.budget.budget is not self.limit
            or self.limit != self.birth_limit
            or self.budget.clock is not self.budget_clock
            or self.budget.started_at != self.budget_started
            or type(self.clock._time) is not int
            or self.clock._time != self.tick
            or type(self.budget.updates_completed) is not int
            or self.budget.updates_completed != len(self.receipts)
            or type(self.budget.replay_examples_completed) is not int
            or self.budget.replay_examples_completed != 0
            or type(self.limit.max_training_updates) is not int
            or self.inbox._capacity != len(self.records) - 1
            or self.inbox._learner_version != "candidate-0"
            or runtime._base_actor_version != "actor-0"
            or runtime._candidate_version != "candidate-0"
            or runtime._training_role != "train"
            or runtime._original_training_role != "train"
            or self.inbox._training_role != "train"
            or self.inbox._original_training_role != "train"
            or tuple(self.inbox._applied.values()) != self.receipts
            or any(
                actual is not original
                for actual, original in zip(
                    self.inbox._applied.values(), self.receipt_owners, strict=True
                )
            )
            or any(
                key != (value.episode_id, value.sample_id)
                for key, value in self.inbox._applied.items()
            )
            or self.actor._version != "actor-0"
        ):
            raise ValueError("sequence original owners, clock, receipts or charges changed")
        _owned_state(self.source, self.initial)
        _owned_state(self.backend, self.initial)
        _owned_state(self.candidate, self.expected)

    def observe(self, phase: str, packet_index: int | None, episode_id: str):
        self.guard()
        metadata: dict[str, Any] = dict(
            phase=phase,
            packet_index=packet_index,
            tick=self.tick,
            episode_id=episode_id,
            known_episode_ids=list(self.known),
            prototype_rows=self.expected.training_examples,
            completed_updates=self.expected.completed_updates,
        )
        entry = dict(metadata, result=None)
        self.report["checkpoints"].append(entry)
        if self.observer is None:
            return
        # Why: every checkpoint gets its own fork; a retained earlier readout
        # never gains subsequent labels. Forks use no initializer or RNG.
        readout = self.candidate.fork()
        baseline = deepcopy(self.expected)
        if any(readout is owner for owner in (self.source, self.backend, self.candidate)):
            raise ValueError("checkpoint readout must be detached from protected owners")
        _owned_state(readout, baseline)
        self.guard()
        self.work["observer_attempts"] += 1
        returned = self.observer(readout, deepcopy(metadata))
        self.work["observer_returns"] += 1
        entry["result"] = _json_result(returned)  # preserve paid output before validation
        _owned_state(readout, baseline)
        self.guard()
        self.work["observer_validated"] += 1

    def release(self, index: int, episode: ContinuousEpisode):
        self.guard()  # preceding paid success must still validate before admission
        self.tick = 3 * index
        self.clock.advance_to(self.tick)
        self.observe("before_release", index, episode.episode_id)
        features, targets = continuous_training_pairs(episode, role="train")
        x, y = np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)
        self.runtime.record_experience(
            Experience(
                "episode-pairs",
                episode.episode_id,
                self.tick,
                "actor-0",
                x,
                "train",
                ExperiencePermissions(training=True),
            )
        )
        self.guard()
        for offset in range(3):
            self.tick = 3 * index + offset
            self.clock.advance_to(self.tick)
            if offset == 2:
                self.runtime.record_label(
                    LabelArrival(
                        "label-" + str(index),
                        "episode-pairs",
                        episode.episode_id,
                        self.tick,
                        "actor-0",
                        y,
                    )
                )
                self.guard()
                self.work["runtime_append_attempts"] += 1
            paid = self.runtime.train_ready(max_updates=1)
            # Preserve paid receipts before snapshot/diagnostic validation.
            self.report["applied"].extend(asdict(item) for item in paid)
            self.work["runtime_append_returns"] += len(paid)
            if offset == 2:
                self.work["runtime_append_rows"] += len(x) * len(paid)
                self.accept_update(index, x, y, paid)
            elif paid:
                raise ValueError("sequence applied labels before their arrival")
            self.guard()
            self.report["events"].append(
                dict(
                    tick=self.tick,
                    packet_index=index,
                    label_available=offset == 2,
                    applied=[asdict(item) for item in paid],
                    candidate_prototype_rows=self.expected.training_examples,
                    candidate_publications=self.expected.completed_updates,
                    budget_updates=self.budget.updates_completed,
                )
            )
        self.known.append(episode.episode_id)
        self.observe("after_release", index, episode.episode_id)

    def accept_update(self, index, x, y, paid):
        if len(paid) != 1:
            raise ValueError("sequence requires exactly one arrived append")
        receipt = paid[0]
        if (
            receipt.observed_at != 3 * index
            or receipt.arrived_at != self.tick
            or receipt.applied_at != self.tick
            or receipt.update_number != index + 1
            or receipt.diagnostic.definition != OBJECTIVE
            or receipt.diagnostic.value != len(x)
            or receipt.episode_id != self.records[index + 1].episode_id
            or receipt.sample_id != "episode-pairs"
            or receipt.event_id != "label-" + str(index)
            or receipt.actor_version != "actor-0"
            or receipt.learner_version != "candidate-0"
        ):
            raise ValueError("sequence arrived receipt differs from declared packet")
        normalized = standardize_observations(
            x, self.initial.policy.observation_standardization, self.initial.policy.spec
        )
        self.work["validation_normalization_rows"] += len(x)
        before = self.expected
        expected = NearestContinuousBCSnapshot(
            before.policy,
            np.concatenate((before.prototype_features, normalized)),
            np.concatenate((before.prototype_actions, y)),
            before.completed_updates + 1,
            before.training_examples + len(x),
        )
        _owned_state(self.candidate, expected)
        self.expected = expected
        self.receipts = (*self.receipts, deepcopy(receipt))
        self.receipt_owners = (*self.receipt_owners, receipt)
        self.work["validated_updates"] += 1

    def finish(self):
        self.guard()
        if self.runtime.train_ready(max_updates=1):
            raise ValueError("sequence repeated a previously paid update")
        self.guard()
        row_bytes = (
            self.initial.policy.spec.observation_dim + len(self.initial.policy.spec.action_units)
        ) * 8
        self.report["memory"] = dict(
            source_prototype_bytes=self.initial.training_examples * row_bytes,
            actor_prototype_bytes=self.initial.training_examples * row_bytes,
            candidate_prototype_bytes=self.expected.training_examples * row_bytes,
            readout_prototype_bytes_max=self.expected.training_examples * row_bytes,
            retained_episode_rows=sum(len(item.transitions) for item in self.records),
            checkpoint_records=len(self.report["checkpoints"]),
            scope="per-owner numeric payloads; excludes Python, snapshots, callback retention and RSS",
        )
        self.report["experiment_state"] = dict(
            completed_updates=self.budget.updates_completed,
            next_tick=self.tick + 1,
        )
        self.report["model_state"] = dict(
            source_publications=self.initial.completed_updates,
            candidate_publications=self.expected.completed_updates,
            candidate_prototype_rows=self.expected.training_examples,
            durable_checkpoint=False,
        )
        self.guard()
        return self.report


def run_sequence(
    warm: ContinuousEpisode,
    releases: tuple[ContinuousEpisode, ...],
    observer: Observer | None = None,
    *,
    fixed_input: bool = False,
    maximum_releases: int = 3,
) -> dict[str, Any]:
    """Run bounded cumulative delayed releases; retain paid partial reports.

    Invalid input raises before fitting. An execution/callback refusal returns
    complete=False and never admits another packet. Callback computation/native
    costs belong in its returned result; this adapter performs no predictions.
    Fixed input is explicit and leaves already encoded features unfitted.
    """
    if type(fixed_input) is not bool:
        raise ValueError("fixed_input must be an exact bool")
    if type(maximum_releases) is not int or maximum_releases not in (3, 5, 6):
        raise ValueError("maximum_releases must be the native integer3,5 or6")
    if maximum_releases > 3 and not fixed_input:
        raise ValueError("extended releases require explicit fixed_input=True")
    if observer is not None and not callable(observer):
        raise ValueError("checkpoint observer must be callable or None")
    # Why: complete B22 coverage is a separate seventh episode; keep every
    # existing correction identity/order rather than inventing a merged path.
    records = (
        _episodes(warm, releases, maximum_releases=maximum_releases)
        if maximum_releases > 3
        else _episodes(warm, releases)
    )
    started = monotonic()
    sequence = (
        _Sequence(records, observer, fixed_input=True)
        if fixed_input
        else _Sequence(records, observer)
    )
    if maximum_releases > 3:
        sequence.report["release_scope"] = dict(
            maximum_releases=maximum_releases,
            default_maximum_releases=3,
            scope=(
                "explicit fixed-input "
                + ("five" if maximum_releases == 5 else "six")
                + "-release sequence; no prior budget renewal"
            ),
        )
    try:
        sequence.setup()
        sequence.observe("warm", None, records[0].episode_id)
        for index, episode in enumerate(records[1:]):
            sequence.release(index, episode)
        sequence.finish()
        sequence.report["elapsed_seconds"] = monotonic() - started
        sequence.guard()
        sequence.report["complete"] = True
    except Exception as error:
        sequence.report["error"] = dict(type=type(error).__name__, message=str(error)[:1024])
        # A native step may commit and charge before its public return raises.
        # Preserve that evidence separately without blessing it as validated.
        budget = getattr(sequence, "budget", None)
        if budget is not None:
            sequence.report["experiment_state"] = dict(
                completed_updates=budget.updates_completed,
                next_tick=sequence.tick + 1,
            )
        inbox = getattr(sequence, "inbox", None)
        if inbox is not None:
            try:
                sequence.report["retained_applied_on_failure"] = [
                    asdict(item) for item in inbox._applied.values()
                ]
            except Exception as capture_error:
                sequence.report["failure_capture_error"] = str(capture_error)[:1024]
        observed = {}
        for name in ("source", "candidate"):
            owner = getattr(sequence, name, None)
            if type(owner) is NumpyNearestContinuousBCLearner:
                try:
                    state = _snapshot(owner._state, owner._policy)
                    observed[name] = dict(
                        completed_updates=state.completed_updates,
                        training_examples=state.training_examples,
                        status="observed_native_counts_not_validated_success",
                    )
                except Exception:
                    observed[name] = dict(status="invalid_native_state")
        sequence.report["observed_native_on_failure"] = observed
    if "elapsed_seconds" not in sequence.report:
        sequence.report["elapsed_seconds"] = monotonic() - started
    return sequence.report
