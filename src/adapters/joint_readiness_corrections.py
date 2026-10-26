"""Simultaneous released TRAIN reference and correction updates.

Own typed raw evidence, project once, then use the existing private actor/shadow
and head-GN ports. Return a detached candidate only on complete success; paid
failures retain their evidence. No qualification, IO, collection or promotion.
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
from src.adapters.joint_readiness_data import (
    OLD_INFORMATION_CONTROL_DEFINITION,
    prepare_joint_readiness_data,
)
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner, _snapshot
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.continuous_bc_diagnostics import continuous_bc_residuals
from src.core.joint_stage_sampling import STAGES, plan_joint_stage_sampling
from src.core.curated_readiness_corrections import CuratedReadinessBatch
from src.core.experience import (
    Experience,
    ExperiencePermissions,
    LabelArrival,
    LogicalClock,
    require_tick,
    validate_experience,
    validate_label_arrival,
)
from src.core.pick_cube_readiness_corrections import GoalConsistentObservedGraspCorrectionBatch
from src.core.readiness_stage_diagnostics import ReadinessStageBinding

UPDATES = 8
EPISODE = "joint-TRAIN-A17-B19-curated-A17"


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _require_state(owner, expected, native_state=None, *, observe=True):
    if type(owner) is not NumpyContinuousBCLearner:
        raise RuntimeError("joint correction native owner was replaced")
    before = owner._state
    if native_state is not None and before is not native_state:
        raise RuntimeError("joint correction native state owner was replaced")
    if not same_continuous_state(expected, _snapshot(owner._state, owner._policy)):
        raise RuntimeError("joint correction native state changed")
    if observe and not same_continuous_state(expected, owner.snapshot_state()):
        raise RuntimeError("joint correction public state differs from its native owner")
    if owner._state is not before or not same_continuous_state(
        expected, _snapshot(owner._state, owner._policy)
    ):
        raise RuntimeError("joint correction native state changed during observation")


def _admit_source(source):
    if type(source) is not NumpyContinuousBCLearner:
        raise ValueError("joint corrections require an exact native BP learner")
    initial = _snapshot(source._state, source._policy)
    policy = initial.policy
    # Exact BP with no binary-gripper override uses the existing common MSE.
    if (
        policy.training_role != "train"
        or policy.binary_gripper_axis is not None
        or not 1 <= policy.hidden_dim <= 64
        or policy.spec.observation_dim != 12
        or policy.observation_standardization is not None
    ):
        raise ValueError(
            "joint corrections require TRAIN common-MSE BP with identity TS12 and H<=64"
        )
    _require_state(source, initial, source._state)
    return initial


def _require_sampling_option(sampling):
    if sampling is not None and (type(sampling) is not str or sampling != "stage_balanced"):
        raise ValueError("joint sampling must be None or exact stage_balanced")


def _sampling_metadata(data, sampling):
    _require_sampling_option(sampling)
    if sampling is None:
        return None, None
    plan = plan_joint_stage_sampling(tuple(row["stage"] for row in data.rows))
    if len(plan) != UPDATES:
        raise ValueError("joint sampling plan differs from the fixed eight-update scope")
    exposures = [0] * len(data.rows)
    for batch in plan:
        for index in batch:
            exposures[index] += 1
    return plan, dict(
        method="stage_balanced_round_robin_v1",
        stage_order=list(STAGES),
        quota_per_stage=len(data.rows) // len(STAGES),
        planned_row_indices=[list(batch) for batch in plan],
        planned_row_training_exposures=exposures,
        planned_group_training_exposures={
            name: sum(exposures[start:stop]) for name, start, stop in data.groups
        },
        planning_work=dict(
            population_rows=len(data.rows),
            planned_index_entries=UPDATES * len(data.rows),
            logical_int64_index_bytes=8 * UPDATES * len(data.rows),
            scope="logical indices only; validation/coverage/counting/copies/Python storage additional",
        ),
    )


class _JointSession:
    """One bounded private update scope; no model state history is retained."""

    def __init__(self, source, initial, data, now, input_guard, report, sampling_plan=None):
        self.source, self.initial, self.data = source, initial, data
        self.source_state, self.input_guard, self.report = source._state, input_guard, report
        self.report_owner = report
        self.old_information_control_owner = report.get("old_information_control")
        self.sampling_plan = self.sampling_plan_owner = sampling_plan
        self.episode_id = report["derived_batch"]["episode_id"]
        self.capture_seal = None
        self.now, self.attempts, self.committed = now, 0, 0
        self.features = np.asarray([r["features"] for r in data.rows], dtype=np.float64)
        self.targets = np.asarray([r["target"] for r in data.rows], dtype=np.float64)
        self.features_owned, self.targets_owned = self.features.copy(), self.targets.copy()
        self.row_seal = _json(data.rows)
        self.metadata_keys = (
            "rows",
            "groups",
            "preparation_work",
            "recipe",
            "source_updates",
            "source_training_rows",
            "head_gauss_newton",
            "derived_batch",
            "budget",
        ) + (() if sampling_plan is None else ("sampling",))
        if "retained_reference" in report:
            self.metadata_keys += ("retained_reference",)
        if self.old_information_control_owner is not None:
            self.metadata_keys += ("old_information_control",)
        self.metadata_seal = _json({key: report[key] for key in self.metadata_keys})
        self.residual_seals: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
        self.work_seal = None
        self.native: list[dict[str, Any]] = []
        self.native_prefix: list[dict[str, Any]] = []
        self.receipts: list[dict[str, Any]] = []
        self.receipt_objects: tuple[Any, ...] = ()
        self.facade = RecordedContinuousHead(source, self.native, UPDATES)
        self.clock = LogicalClock()
        self.clock.advance_to(now)
        self.limits = ToyExecutionBudget(max_training_updates=UPDATES)
        self.budget = ToyBudgetSession(self.limits, monotonic)
        self.runtime = ActorShadowRuntime(
            self.facade,
            actor_version="actor-0",
            candidate_version="candidate-0",
            clock=self.clock,
            budget=self.budget,
            max_experiences=UPDATES,
            max_consolidations=0,
        )
        self.actor, self.inbox = self.runtime._actor, self.runtime._inbox
        self.actorport = cast(RecordedContinuousHead, self.actor._learner)
        self.candidateport = cast(RecordedContinuousHead, self.runtime._candidate)
        self.raw_actor, self.raw_candidate = self.actorport._learner, self.candidateport._learner
        self.actor_state = self.raw_actor._state
        self.admitted = initial
        self.owners = self._owners()
        self.born = (
            self.budget.budget,
            self.budget.clock,
            self.budget.started_at,
            self.budget.progress,
        )
        self.limit_seal = _json(asdict(self.limits))
        self.last_wall = self.budget.last_clock

    def _owners(self):
        return (
            self.runtime._actor,
            self.runtime._candidate,
            self.runtime._inbox,
            self.actor._learner,
            self.actorport._learner,
            self.candidateport._learner,
            self.runtime._attempted_ids,
            self.runtime._consolidations,
            self.inbox._applied,
            self.inbox._experiences,
            self.inbox._labels,
            self.inbox._event_ids,
        )

    def _require_runtime(self):
        if any(a is not b for a, b in zip(self._owners(), self.owners, strict=True)):
            raise RuntimeError("joint correction runtime owner identity changed")
        b, r, i = self.budget, self.runtime, self.inbox
        if (
            b.budget is not self.born[0]
            or b.clock is not self.born[1]
            or b.progress is not self.born[3]
            or b.started_at != self.born[2]
            or type(b.started_at) is not float
            or not math.isfinite(b.started_at)
            or _json(asdict(self.limits)) != self.limit_seal
            or r._budget is not b
            or i._budget is not b
            or i._clock is not self.clock
            or i._learner is not self.candidateport
            or type(self.clock._time) is not int
            or self.clock._time != self.now
            or type(b.updates_completed) is not int
            or b.updates_completed != self.committed
            or type(b.replay_examples_completed) is not int
            or b.replay_examples_completed != 0
            or type(b.last_clock) is not float
            or not math.isfinite(b.last_clock)
            or b.last_clock < self.last_wall
            or type(r._consolidation_limit) is not int
            or r._consolidation_limit != 0
            or r._attempted_ids
            or r._consolidations
            or r._stopped is not False
            or r._retired is not False
            or i._stopped is not False
            or r._candidate_version != "candidate-0"
            or i._learner_version != "candidate-0"
            or self.actor._version != "actor-0"
            or r._training_role != "train"
            or i._training_role != "train"
            or type(i._capacity) is not int
            or i._capacity != UPDATES
        ):
            raise RuntimeError("joint correction original budget/clock/authority changed")
        self.last_wall = b.last_clock

    def _row_exposures(self, updates):
        counts = [0] * len(self.data.rows)
        if self.sampling_plan_owner is None:
            return [updates] * len(self.data.rows)
        # Refusal/capture must retain the original paid mapping even if the
        # active schedule was replaced by an opaque observer before failure.
        for batch in self.sampling_plan_owner[:updates]:
            for index in batch:
                counts[index] += 1
        return counts

    def _group_exposures(self):
        if self.sampling_plan_owner is None:
            return {name: len(self.native) * (end - start) for name, start, end in self.data.groups}
        counts = self._row_exposures(len(self.native_prefix))
        return {name: sum(counts[start:end]) for name, start, end in self.data.groups}

    def _batch(self, index, *, owned=False):
        features = self.features_owned if owned else self.features
        targets = self.targets_owned if owned else self.targets
        if self.sampling_plan_owner is None:
            return features, targets
        indices = list(self.sampling_plan_owner[index])
        return features[indices], targets[indices]

    def _require_receipts(self):
        current = tuple(self.inbox._applied.values())
        if (
            len(current) != len(self.receipt_objects)
            or any(a is not b for a, b in zip(current, self.receipt_objects, strict=True))
            or _json([asdict(r) for r in current]) != _json(self.receipts)
            or _json(self.native) != _json(self.native_prefix)
        ):
            raise RuntimeError("joint correction preceding paid evidence changed")
        if "native_updates" in self.report and (
            _json(self.report["native_updates"]) != _json(self.native)
            or _json(self.report["validated_native_prefix"]) != _json(self.native_prefix)
            or _json(self.report["sealed_applied_updates"]) != _json(self.receipts)
            or _json(self.report["applied_updates"]) != _json(self.receipts)
            or _json(self.report["sealed_native_updates"])
            != _json([json.loads(payload) for _, payload in self.facade._records])
            or self.report["update_attempts"] != self.attempts
            or self.report["attempted_training_rows"] != self.attempts * len(self.data.rows)
            or _json(self.report["progress"])
            != _json(
                dict(
                    updates_completed=self.budget.updates_completed,
                    replay_examples_completed=self.budget.replay_examples_completed,
                    logical_tick=self.clock._time,
                )
            )
            or _json(self.report["group_training_exposures"]) != _json(self._group_exposures())
        ):
            raise RuntimeError("joint correction copied paid evidence changed")
        if (
            self.sampling_plan_owner is not None
            and "native_updates" in self.report
            and (
                self.report["row_training_exposures"]
                != self._row_exposures(len(self.native_prefix))
                or self.report["attempted_row_training_exposures"]
                != self._row_exposures(self.attempts)
                or self.report["unvalidated_sampling_work_unknown"]
                is not (self.attempts > len(self.native_prefix))
            )
        ):
            raise RuntimeError("joint correction sampled paid exposure evidence changed")

    def _require_payloads(self):
        i = self.inbox
        if len(i._experiences) != len(i._labels) or len(i._experiences) > UPDATES:
            raise RuntimeError("joint correction retained payload count changed")
        for n, (event, label) in enumerate(
            zip(i._experiences.values(), i._labels.values(), strict=True)
        ):
            expected_features, expected_targets = self._batch(n, owned=True)
            validate_experience(event)
            validate_label_arrival(label)
            if (
                event.sample_id != f"joint-{n}"
                or label.sample_id != event.sample_id
                or label.event_id != f"joint-label-{n}"
                or event.episode_id != self.episode_id
                or label.episode_id != self.episode_id
                or event.observed_at != self.now
                or label.arrived_at != self.now
                or event.role != "train"
                or label.role != "train"
                or event.model_version != "actor-0"
                or label.model_version != "actor-0"
                or event.permissions != ExperiencePermissions(training=True)
                or not np.array_equal(event.features, expected_features)
                or not np.array_equal(label.targets, expected_targets)
            ):
                raise RuntimeError("joint correction derived packet or authority changed")

    def _require(self, observe):
        if self.report.get("old_information_control") is not self.old_information_control_owner:
            raise RuntimeError("joint correction old-information metadata owner changed")
        if self.sampling_plan is not self.sampling_plan_owner:
            raise RuntimeError("joint correction original sampling plan was replaced")
        if self.episode_id != self.report["derived_batch"]["episode_id"]:
            raise RuntimeError("joint correction packet identity changed")
        if self.report is not self.report_owner:
            raise RuntimeError("joint correction report owner was replaced")
        self.input_guard()
        self._require_runtime()
        for port in (self.facade, self.actorport, self.candidateport):
            port.require_reports(self.native)
        self.candidateport.require_latest(self.admitted)
        _require_state(self.source, self.initial, self.source_state, observe=observe)
        _require_state(self.raw_actor, self.initial, self.actor_state, observe=observe)
        _require_state(self.raw_candidate, self.admitted, observe=observe)
        self._require_receipts()
        self._require_payloads()
        if (
            _json(self.data.rows) != self.row_seal
            or _json({key: self.report[key] for key in self.metadata_keys}) != self.metadata_seal
            or not np.array_equal(self.features, self.features_owned)
            or not np.array_equal(self.targets, self.targets_owned)
        ):
            raise RuntimeError("joint correction owned training rows changed")
        for (when, name), (item, payload) in self.residual_seals.items():
            if self.report["residuals"][when].get(name) is not item or _json(item) != payload:
                raise RuntimeError("joint correction preceding residual evidence changed")
        if self.work_seal is not None and _json(self._completed_work()) != self.work_seal:
            raise RuntimeError("joint correction completed work evidence changed")
        # Public snapshot observers must not change authority after its first check.
        self._require_runtime()

    def require(self):
        self._require(True)
        # Later public getters may mutate an owner checked by an earlier getter.
        self.require_native()

    def require_native(self):
        self._require(False)

    def _completed_work(self):
        return {
            key: self.report[key]
            for key in ("work", "candidate_updates", "candidate_training_rows")
        }

    def _seal_residual(self, when, name, item):
        self.residual_seals[(when, name)] = (item, _json(item))

    def _copied_paid_evidence(self):
        keys = (
            "native_updates",
            "validated_native_prefix",
            "sealed_applied_updates",
            "progress",
            "applied_updates",
            "sealed_native_updates",
            "group_training_exposures",
            "update_attempts",
            "attempted_training_rows",
            "unreturned_training_work_unknown",
        ) + (
            ()
            if self.sampling_plan_owner is None
            else (
                "row_training_exposures",
                "attempted_row_training_exposures",
                "unvalidated_sampling_work_unknown",
            )
        )
        return {key: self.report.get(key) for key in keys}

    def capture(self):
        r = self.report
        previous = self._copied_paid_evidence()
        changed = self.capture_seal is not None and _json(previous) != self.capture_seal
        if changed:
            # Keep the first observed corruption alongside the original seals.
            r.setdefault("corrupted_copied_evidence", deepcopy(previous))
        r["update_attempts"], r["attempted_training_rows"] = (
            self.attempts,
            self.attempts * len(self.data.rows),
        )
        r["native_updates"] = deepcopy(self.native)
        r["validated_native_prefix"] = deepcopy(self.native_prefix)
        r["sealed_applied_updates"] = deepcopy(self.receipts)
        r["progress"] = dict(
            updates_completed=self.budget.updates_completed,
            replay_examples_completed=self.budget.replay_examples_completed,
            logical_tick=self.clock._time,
        )
        r["applied_updates"] = [asdict(row) for row in self.inbox._applied.values()]
        r["sealed_native_updates"] = [json.loads(payload) for _, payload in self.facade._records]
        r["group_training_exposures"] = self._group_exposures()
        if self.sampling_plan_owner is not None:
            r["row_training_exposures"] = self._row_exposures(len(self.native_prefix))
            r["attempted_row_training_exposures"] = self._row_exposures(self.attempts)
            r["unvalidated_sampling_work_unknown"] = self.attempts > len(self.native_prefix)
        r["unreturned_training_work_unknown"] = self.attempts > len(self.native)
        self.capture_seal = _json(self._copied_paid_evidence())
        if changed:
            raise RuntimeError("joint correction copied paid evidence changed before capture")

    def residuals(self, when):
        for name, start, end in self.data.groups:
            self.require()
            item: dict[str, Any] = dict(
                prediction_attempted=True,
                prediction_returned=False,
                prediction_rows_attempted=end - start,
                residual_attempted=False,
                residual_returned=False,
            )
            self.report["residuals"][when][name] = item
            self._seal_residual(when, name, item)
            predicted = self.candidateport.predict(self.features[start:end])
            owned_prediction = predicted.tolist()
            if _json(item) != self.residual_seals[(when, name)][1]:
                self.report["unvalidated_prediction_return"] = dict(
                    when=when, group=name, predictions=owned_prediction, prediction_rows=end - start
                )
                raise RuntimeError("joint correction prediction evidence changed during return")
            item.update(
                predictions=owned_prediction,
                prediction_returned=True,
                prediction_rows=end - start,
            )
            self._seal_residual(when, name, item)
            self.require_native()
            self.require()
            item["residual_attempted"] = True
            self._seal_residual(when, name, item)
            item["result"] = asdict(
                continuous_bc_residuals(
                    predicted,
                    self.targets[start:end],
                    self.data.spec,
                    tuple(row["stage"] for row in self.data.rows[start:end]),
                )
            )
            item["residual_returned"] = True
            self._seal_residual(when, name, item)
            self.capture()
            self.require()

    def _commit_return(self, index, returned):
        observed = _snapshot(self.raw_candidate._state, self.raw_candidate._policy)
        self.candidateport.require_latest(observed)
        self.candidateport.require_reports(self.native)
        receipts = list(self.inbox._applied.values())
        if (
            len(returned) != 1
            or len(receipts) != index + 1
            or len(self.native) != index + 1
            or _json(self.native[:-1]) != _json(self.native_prefix)
            or _json([asdict(r) for r in receipts[:-1]]) != _json(self.receipts)
            or any(a is not b for a, b in zip(receipts[:-1], self.receipt_objects, strict=True))
        ):
            raise RuntimeError("joint correction previous or returned paid prefix changed")
        row = receipts[-1]
        if (
            row.sample_id != f"joint-{index}"
            or row.event_id != f"joint-label-{index}"
            or row.episode_id != self.episode_id
            or (row.observed_at, row.arrived_at, row.applied_at) != (self.now,) * 3
            or row.update_number != index + 1
            or row.actor_version != "actor-0"
            or row.learner_version != "candidate-0"
            or _json(asdict(row.diagnostic)) != _json(self.native[-1]["diagnostic"])
            or self.native[-1]["training_rows"] != len(self.data.rows)
            or observed.policy != self.initial.policy
            or (observed.completed_updates, observed.training_examples)
            != (
                self.initial.completed_updates + index + 1,
                self.initial.training_examples + (index + 1) * len(self.data.rows),
            )
        ):
            raise RuntimeError("joint correction native publication/receipt/counters differ")
        self.admitted, self.committed = observed, index + 1
        self.receipt_objects, self.receipts = tuple(receipts), [asdict(r) for r in receipts]
        self.native_prefix = deepcopy(self.native)

    def apply_one(self, index):
        self.require()
        features, targets = self._batch(index)
        self.runtime.record_experience(
            Experience(
                f"joint-{index}",
                self.episode_id,
                self.now,
                "actor-0",
                features,
                "train",
                ExperiencePermissions(training=True),
            )
        )
        self.runtime.record_label(
            LabelArrival(
                f"joint-label-{index}",
                f"joint-{index}",
                self.episode_id,
                self.now,
                "actor-0",
                targets,
                "train",
            )
        )
        self.require()
        self.attempts += 1
        try:
            returned = self.runtime.train_ready(max_updates=1)
        finally:
            self.capture()
        self._commit_return(index, returned)
        self.capture()
        self.require()

    def finish(self):
        self.require()
        candidate = self.source.fork()
        if candidate is self.source or candidate in (self.raw_actor, self.raw_candidate):
            raise RuntimeError("joint correction detached candidate aliases a live owner")
        candidate.restore_state(self.admitted)
        _require_state(candidate, self.admitted)
        self.require()
        counts, flags, weights = _native_work(self.native)
        self.report["work"] = dict(
            optimizer_updates=len(self.native),
            head_gauss_newton_updates=len(self.native),
            training_rows=sum(row["training_rows"] for row in self.native),
            native_update_counts=counts,
            native_work_flags=flags,
            native_event_row_weights=weights,
            initializations=0,
            fits=0,
            sgd_updates=0,
            solver_work_known=False,
            inbox_feature_target_numeric_bytes=UPDATES * len(self.data.rows) * 16 * 8,
            parameter_momentum_numeric_bytes_per_owner=16
            * (
                12 * self.initial.policy.hidden_dim
                + self.initial.policy.hidden_dim
                + self.initial.policy.hidden_dim * 4
                + 4
            ),
        )
        if self.sampling_plan_owner is not None:
            self.report["work"].update(
                sampled_training_gather_values=UPDATES * len(self.data.rows) * 16,
                sampled_training_gather_numeric_bytes=UPDATES * len(self.data.rows) * 128,
                sampling_scope="training payload gathers only; repeated guard gathers, exposure "
                "counting/validation/Python containers additional; no normalization or new history",
            )
        self.report["candidate_updates"] = self.admitted.completed_updates
        self.report["candidate_training_rows"] = self.admitted.training_examples
        self.work_seal = _json(self._completed_work())
        self.require()
        _require_state(candidate, self.admitted)
        return candidate


def run_joint_readiness_corrections(
    source,
    reference,
    corrections,
    curated,
    *,
    now: int,
    action_history_recipe=None,
    sampling=None,
    retained_reference: ReadinessStageBinding | None = None,
    old_information_control: bool = False,
) -> ContinuousLearnabilityResult:
    """Apply eight simultaneous released batches; keep failed paid work, never a learner.

    The reference must already be qualified at the trusted input boundary. This
    API preserves provisional correction authority and does not qualify episodes.
    Explicit curated=None selects the preceding A+B-only stage. Supplied curated
    evidence must pass its ordinary release gate; it is never silently omitted.
    An optional separately qualified observed-grasp TRAIN20 reference retains its
    original history and authority in a distinct group. Qualification is caller-owned.
    The explicit old-information control instead repeats original A17 information
    at equal row weight, without admitting a new source or changing targets.
    Preadmission raises; failures after a private scope exists return partial work.
    """
    if type(old_information_control) is not bool:
        raise ValueError("old_information_control must be an exact bool")
    if old_information_control and (
        curated is None or retained_reference is not None or sampling is not None
    ):
        raise ValueError(
            "old information control requires curation, no retained reference and no sampling"
        )
    _require_sampling_option(sampling)
    require_tick(now, "joint correction release time")
    if (
        type(reference) is not ReadinessStageBinding
        or type(corrections) is not GoalConsistentObservedGraspCorrectionBatch
        or (curated is not None and type(curated) is not CuratedReadinessBatch)
        or (
            retained_reference is not None and type(retained_reference) is not ReadinessStageBinding
        )
    ):
        raise ValueError(
            "joint corrections require exact typed reference, B corrections and A curation"
        )
    if old_information_control and len(reference.episode.transitions) != 15:
        raise ValueError("old information control requires the original fifteen-row A17 reference")
    inputs = (
        (reference, corrections)
        + (() if curated is None else (curated,))
        + (() if action_history_recipe is None else (action_history_recipe,))
        + (() if retained_reference is None else (retained_reference,))
    )
    input_seal = _json([asdict(value) for value in inputs])

    def input_guard():
        if _json([asdict(value) for value in inputs]) != input_seal:
            raise RuntimeError("original joint correction evidence changed")

    initial = _admit_source(source)
    input_guard()
    source_state = source._state
    preparation_options: dict[str, Any] = {}
    if retained_reference is not None:
        preparation_options["retained_reference"] = retained_reference
    if old_information_control:
        preparation_options["old_information_control"] = True
    data = prepare_joint_readiness_data(
        reference,
        corrections,
        curated,
        now=now,
        recipe=action_history_recipe,
        **preparation_options,
    )
    plan, sampling_report = _sampling_metadata(data, sampling)
    input_guard()
    if data.spec != initial.policy.spec:
        raise ValueError("joint correction feature spec differs from the frozen learner")
    _require_state(source, initial, source_state)
    input_guard()
    report: dict[str, Any] = dict(
        protocol="joint_readiness_head_gauss_newton_v1",
        complete=False,
        head_gauss_newton=head_protocol(),
        rows=deepcopy(data.rows),
        groups=deepcopy(data.groups),
        preparation_work=deepcopy(data.work),
        recipe=asdict(data.recipe),
        source_updates=initial.completed_updates,
        source_training_rows=initial.training_examples,
        residuals=dict(before={}, after={}),
        derived_batch=dict(
            episode_id=EPISODE if curated is not None else "joint-TRAIN-A17-B19",
            observed_at=now,
            arrived_at=now,
            role="train",
            original_row_clocks_preserved=True,
            qualified_demonstration=False,
        ),
        budget=asdict(ToyExecutionBudget(max_training_updates=UPDATES)),
    )
    if sampling_report is not None:
        report["sampling"] = sampling_report
    if retained_reference is not None:
        # Why: this is another released TRAIN source, not a relabeled A/B row or
        # newly qualified demonstration. Validated group exposures remain paid.
        report["retained_reference"] = dict(
            episode_id=retained_reference.episode.episode_id,
            context=retained_reference.episode.context,
            role=retained_reference.episode.role,
            rows=len(retained_reference.episode.transitions),
            group="retained_train20",
            qualification="caller-owned at the existing raw input boundary",
        )
        report["derived_batch"]["episode_id"] += "-retained-TRAIN20"
    if old_information_control:
        report["old_information_control"] = dict(
            definition=OLD_INFORMATION_CONTROL_DEFINITION,
            group="repeated_a",
            repeated_rows=15,
            source_row_indices=list(range(15)),
            source_episode_id=reference.episode.episode_id,
            source_context=reference.episode.context,
            source_role=reference.episode.role,
        )
        report["derived_batch"]["episode_id"] += "-old-reference-control"
    session = None
    candidate = None
    try:
        session = _JointSession(
            source,
            initial,
            data,
            now,
            input_guard,
            report,
            **({} if plan is None else {"sampling_plan": plan}),
        )
        session.require()
        session.residuals("before")
        for index in range(UPDATES):
            session.apply_one(index)
        session.residuals("after")
        candidate = session.finish()
        session.capture()
        session.require()
        _require_state(candidate, session.admitted)
        session.require_native()
        _require_state(candidate, session.admitted, observe=False)
        report["complete"] = True
    except Exception as error:
        report["complete"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        if hasattr(error, "work"):
            report["failed_native_work"] = deepcopy(error.work)
        candidate = None
    finally:
        if session is not None:
            try:
                session.capture()
                if candidate is not None:
                    session.require_native()
                    _require_state(candidate, session.admitted, observe=False)
            except Exception as error:
                report["capture_error"] = f"{type(error).__name__}: {error}"
                report.setdefault("error", report["capture_error"])
                report["complete"], candidate = False, None
    return ContinuousLearnabilityResult(deepcopy(report), candidate)
