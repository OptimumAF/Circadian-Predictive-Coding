"""Measure matched promotion guards on detached native model copies.

Inputs: frozen actor/candidate snapshots, declared policy/role IDs and configured
native probes. Output: complete local rejection/measurement report. No training,
serving mutation, promotion ticket, rollback or scientific release belongs here.
"""

from copy import deepcopy
from math import isfinite
import re
from typing import Callable, Generic, TypeVar

from src.core.actor_ports import CandidateState, VersionedState
from src.core.experience import SampleKey, require_identifier, require_tick
from src.core.learner_ports import NativeLearner
from src.core.promotion_guard import (
    GuardBatch,
    PromotionEvidence,
    PromotionGuardReport,
    PromotionPolicy,
    decide_promotion,
    require_sample_keys,
)

Features = TypeVar("Features")
Targets = TypeVar("Targets")
Prediction = TypeVar("Prediction")
State = TypeVar("State")


class PromotionGuardEvaluator(Generic[Features, Targets, Prediction, State]):
    def __init__(
        self,
        *,
        build_learner: Callable[[State], NativeLearner[Features, Targets, Prediction, State]],
        utility: Callable[[Prediction, Targets], float],
        actions: Callable[[Prediction], tuple[str, ...]],
        state_valid: Callable[[State], bool],
        prediction_valid: Callable[[Prediction], bool],
        resource_bytes: Callable[[State], int],
        state_digest: Callable[[State], str],
        clock: Callable[[], float],
    ) -> None:
        if not all(
            callable(p)
            for p in (
                build_learner,
                utility,
                actions,
                state_valid,
                prediction_valid,
                resource_bytes,
                state_digest,
                clock,
            )
        ):
            raise ValueError("guard evaluator requires callable native probes and clock")
        self._build, self._utility, self._actions = build_learner, utility, actions
        self._state_valid, self._prediction_valid = state_valid, prediction_valid
        self._resource, self._digest, self._clock = resource_bytes, state_digest, clock

    def evaluate(
        self,
        policy: PromotionPolicy,
        actor: VersionedState[State],
        candidate: CandidateState[State],
        new: GuardBatch[Features, Targets],
        old: GuardBatch[Features, Targets],
        *,
        now: int,
        training_ids: frozenset[SampleKey],
    ) -> PromotionGuardReport:
        self._require_metadata(policy, actor, candidate, new, old, now, training_ids)
        actor_state, candidate_state = deepcopy(actor.state), deepcopy(candidate.state)
        actor_digest, candidate_digest = (
            self._state_digest(actor_state),
            self._state_digest(candidate_state),
        )
        valid = self._valid(self._state_valid(deepcopy(actor_state)))
        valid = self._valid(self._state_valid(deepcopy(candidate_state))) and valid
        if valid:
            evidence = self._evaluate_models(
                policy, actor_state, candidate_state, deepcopy(new), deepcopy(old)
            )
        else:
            evidence = PromotionEvidence(None, None, None, None, None, None, False, False)
        return PromotionGuardReport(
            policy,
            actor.actor_version,
            candidate.learner_version,
            (
                candidate.wake_updates,
                candidate.consolidation_attempts,
                candidate.consolidations_completed,
            ),
            actor_digest,
            candidate_digest,
            new.task_id,
            old.task_id,
            new.sample_keys,
            old.sample_keys,
            (new.labels_arrived_at, old.labels_arrived_at),
            tuple(sorted(training_ids)),
            now,
            evidence,
            decide_promotion(policy, evidence),
        )

    @staticmethod
    def _require_metadata(
        policy: PromotionPolicy,
        actor: VersionedState[State],
        candidate: CandidateState[State],
        new: GuardBatch[Features, Targets],
        old: GuardBatch[Features, Targets],
        now: int,
        training_ids: frozenset[SampleKey],
    ) -> None:
        if (
            type(policy) is not PromotionPolicy
            or type(actor) is not VersionedState
            or type(candidate) is not CandidateState
        ):
            raise ValueError("promotion guard requires declared policy and versioned snapshots")
        require_tick(now, "guard observation time")
        for batch in (new, old):
            if type(batch) is not GuardBatch or batch.role != "inner_guard":
                raise ValueError(
                    "promotion measurements require inner_guard role before payload access"
                )
            require_identifier(batch.task_id, "guard task_id")
            require_sample_keys(batch.sample_keys)
            require_tick(batch.labels_arrived_at, "guard label arrival")
            if batch.labels_arrived_at > now:
                raise ValueError("guard labels have not arrived")
        if new.task_id == old.task_id:
            raise ValueError("new and old guard task IDs must differ")
        if type(training_ids) is not frozenset:
            raise ValueError("training_ids must be a frozen identity set")
        require_sample_keys(tuple(training_ids), allow_empty=True)
        new_keys, old_keys = set(new.sample_keys), set(old.sample_keys)
        if new_keys & old_keys or (new_keys | old_keys) & training_ids:
            raise ValueError("new/old guards and declared training IDs must be disjoint")
        for version in (actor.actor_version, candidate.actor_version, candidate.learner_version):
            require_identifier(version, "model version")
        if (
            actor.actor_version != candidate.actor_version
            or actor.actor_version == candidate.learner_version
        ):
            raise ValueError("candidate base actor version is stale or not distinct")
        for counter in (
            candidate.wake_updates,
            candidate.consolidation_attempts,
            candidate.consolidations_completed,
        ):
            require_tick(counter, "candidate revision")
        if candidate.consolidations_completed > candidate.consolidation_attempts:
            raise ValueError("candidate revision has more completions than attempts")

    @staticmethod
    def _valid(value: bool) -> bool:
        if type(value) is not bool:
            raise ValueError("numerical probe must return an exact boolean")
        return value

    def _state_digest(self, state: State) -> str:
        digest = self._digest(deepcopy(state))
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("state digest must be a lowercase SHA256 identifier")
        return digest

    def _evaluate_models(
        self,
        policy: PromotionPolicy,
        actor_state: State,
        candidate_state: State,
        new: GuardBatch[Features, Targets],
        old: GuardBatch[Features, Targets],
    ) -> PromotionEvidence:
        actor_model = self._build(deepcopy(actor_state))
        candidate_model = self._build(deepcopy(candidate_state))
        if actor_model is candidate_model:
            raise ValueError("guard factory must return independent predictors")
        actor_model.restore_state(deepcopy(actor_state))
        candidate_model.restore_state(deepcopy(candidate_state))
        resource = self._resource(deepcopy(candidate_state))
        require_tick(resource, "candidate_resource_bytes")
        last: list[float | None] = [None]
        actor_new = self._measure(actor_model, new, policy, last)
        candidate_new = self._measure(candidate_model, new, policy, last)
        actor_old = self._measure(actor_model, old, policy, last)
        candidate_old = self._measure(candidate_model, old, policy, last)
        valid = all(result[1] for result in (actor_new, candidate_new, actor_old, candidate_old))
        valid = (
            self._state_digest(actor_model.snapshot_state()) == self._state_digest(actor_state)
            and valid
        )
        valid = (
            self._state_digest(candidate_model.snapshot_state())
            == self._state_digest(candidate_state)
            and valid
        )
        return PromotionEvidence(
            actor_new[0],
            candidate_new[0],
            actor_old[0],
            candidate_old[0],
            max(candidate_new[3], candidate_old[3]),
            resource,
            valid,
            candidate_new[2] and candidate_old[2],
        )

    def _read_clock(self, last: list[float | None]) -> float:
        now = self._clock()
        if type(now) not in (int, float) or not isfinite(now):
            raise ValueError("guard clock must return finite seconds")
        if last[0] is not None and now < last[0]:
            raise ValueError("guard clock moved backwards")
        last[0] = float(now)
        return float(now)

    def _measure(
        self,
        model: NativeLearner[Features, Targets, Prediction, State],
        guard: GuardBatch[Features, Targets],
        policy: PromotionPolicy,
        last: list[float | None],
    ) -> tuple[float | None, bool, bool, float]:
        features = deepcopy(guard.features)
        started = self._read_clock(last)
        prediction = deepcopy(model.predict(features))
        elapsed = self._read_clock(last) - started
        valid = self._valid(self._prediction_valid(deepcopy(prediction)))
        if not valid:
            return None, False, False, elapsed
        utility = self._utility(deepcopy(prediction), deepcopy(guard.targets))
        decoded = self._actions(deepcopy(prediction))
        if type(decoded) is not tuple:
            raise ValueError("action decoder must return an immutable tuple")
        safe = len(decoded) == len(guard.sample_keys) and all(
            action in policy.allowed_actions for action in decoded
        )
        return utility, valid, safe, elapsed
