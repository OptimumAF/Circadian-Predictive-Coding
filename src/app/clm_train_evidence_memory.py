"""Phase-delayed TRAIN memory beside the existing fixed CLM stream.

Consumes validated catalog update receipts and separately consented raw TRAIN
mirrors. Returns bounded, unscored retrieval observations; scoring happens only
after TRAIN. No backend, head update, probe, final data, encoder or promotion.
"""

from copy import deepcopy
from dataclasses import asdict
import json
from typing import Any

from src.app.decision_evidence import (
    EvidenceDecisionPredictor,
    apply_decision_feedback,
    decision_from_evidence,
)
from src.core.clm_catalog import CATALOG_OBJECTIVE
from src.core.decision_evidence import DecisionEvidenceStore, EvidencePolicy
from src.core.experience import Experience, ExperiencePermissions, LabelArrival
from src.core.temporal_decision_sequence import CHECKPOINTS, policy_for_phase, temporal_examples


def _pin(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _require_receipts(report, count):
    """Check the complete paid prefix before releasing any raw TRAIN label."""
    stream = report.get("stream", {})
    receipts, rows = stream.get("receipts"), report.get("rows")
    if (
        report.get("training_objective") != CATALOG_OBJECTIVE
        or report.get("catalog_support", "none") != "none"
        or report.get("training_replay_consent") is not True
        or stream.get("completed_updates") != count
        or stream.get("training_pairs") != 3 * count
        or type(receipts) is not list
        or len(receipts) != count
        or type(rows) is not list
        or len(rows) != count
    ):
        raise ValueError("memory requires the complete consented original catalog prefix")
    for index, (receipt, row) in enumerate(zip(receipts, rows, strict=True)):
        phase = CHECKPOINTS[1 + index // 4][1]
        sample = f"clm-{phase}-{index % 4}"
        expected = dict(
            sample_id=sample,
            episode_id="clm-temporal",
            event_id="label-" + sample,
            actor_version=phase,
            learner_version="clm-temporal",
            observed_at=2 * index,
            arrived_at=2 * index + 1,
            applied_at=2 * index + 1,
            update_number=index + 1,
        )
        examples = temporal_examples(phase, "train", exposure=index % 4)
        ids = [candidate.candidate_id for candidate in examples[0].request.allowed_candidates]
        if (
            type(receipt) is not dict
            or type(row) is not dict
            or any(
                type(receipt.get(k)) is not type(v) or receipt[k] != v for k, v in expected.items()
            )
            or receipt.get("diagnostic", {}).get("definition") != CATALOG_OBJECTIVE
            or row.get("update") != index + 1
            or row.get("phase") != phase
            or row.get("exposure") != index % 4
            or row.get("candidate_ids") != ids
            or row.get("candidate_indices") != [ids.index(e.target_id) for e in examples]
        ):
            raise ValueError("memory receipt clock, TRAIN labels or catalog changed")


class CLMTrainEvidenceMemory:
    """One finite checkpoint observer; default TRAIN never creates this owner."""

    def __init__(self, report, *, permissions):
        if type(report) is not dict or "train_evidence_memory" in report:
            raise ValueError("memory requires a fresh report slot")
        if type(permissions) is not ExperiencePermissions or permissions != ExperiencePermissions(
            training=True, replay=True
        ):
            raise ValueError("raw TRAIN memory requires explicit original reuse consent")
        self._report, self._permissions = report, deepcopy(permissions)
        self._store = DecisionEvidenceStore(EvidencePolicy(36, 65536, 768, 1))
        self._record: dict[str, Any] = dict(
            definition="clm_phase_delayed_train_evidence_v1",
            complete=False,
            recording="phase checkpoint; original observation and label clocks retained",
            permissions=asdict(permissions),
            phases=[],
            events=[],
            work=self._store.work,
            storage_bytes=self._store.storage_bytes,
        )
        report["train_evidence_memory"] = self._record
        self._completed = 0
        self._failed = False
        self._paid_prefix: list = []
        self._serving_attempted = False
        self._serving_work: dict[str, Any] | None = None
        self._publish()

    @property
    def serving_work(self):
        return deepcopy(self._serving_work)

    def require_serving_source(self):
        """The phase owner remains sealed even while its separate view is read."""
        self._require()
        _require_receipts(self._report, 12)
        if self._completed != 4 or self._record["complete"] is not True:
            raise ValueError("serving requires a completed original TRAIN-memory source")

    def bind_fallback(self, predict):
        """Copy one bounded serving view without renewing the phase-store budget."""
        self.require_serving_source()
        if self._serving_attempted or not callable(predict):
            raise ValueError("serving view requires an unspent binding and frozen callback")
        self._serving_attempted = True
        self._serving_work = dict(
            binding_attempts=1,
            binding_returns=0,
            copy_write_attempts=0,
            copy_write_returns=0,
            source_events=36,
            source_storage_bytes=self._store.storage_bytes,
            view_storage_bytes=0,
            source_work=self._store.work,
            cutoff=23,
            policy_version="v2",
            max_calls=5,
            copy_return_unknown=False,
        )
        view = DecisionEvidenceStore(EvidencePolicy(36, 65536, 768, 1))
        try:
            # Why: separate paid copies keep the completed observation immutable;
            # a single-attempt bind prevents repeatedly manufacturing new stores.
            for event in self._store.events:
                self._serving_work["copy_write_attempts"] += 1
                self._serving_work["copy_return_unknown"] = True
                view.add(event.evidence)
                self._serving_work["copy_write_returns"] += 1
                self._serving_work["copy_return_unknown"] = False
            result = EvidenceDecisionPredictor(
                view, cutoff=23, policy_version="v2", mode="fallback", predict=predict, max_calls=5
            )
            self.require_serving_source()
            self._serving_work["binding_returns"] = 1
            return result
        finally:
            self._serving_work["view_storage_bytes"] = view.storage_bytes
            self._serving_work["view_write_work"] = view.work

    def _publish(self):
        self._record.update(
            events=[asdict(event) for event in self._store.events],
            work=self._store.work,
            storage_bytes=self._store.storage_bytes,
        )
        self._proof = _pin(self._record)

    def _require(self):
        self._permissions.__post_init__()
        if (
            self._failed
            or self._report.get("train_evidence_memory") is not self._record
            or _pin(self._record) != self._proof
            or self._permissions != ExperiencePermissions(training=True, replay=True)
            or self._record["events"] != [asdict(event) for event in self._store.events]
            or self._record["work"] != self._store.work
            or self._record["storage_bytes"] != self._store.storage_bytes
            or self._report.get("stream", {}).get("receipts", [])[: len(self._paid_prefix)]
            != self._paid_prefix
        ):
            raise ValueError("memory stopped or preceding evidence/receipt prefix changed")

    def _record_phase(self, phase, count):
        # Why: receipt validation authorizes release, not a retrospective write
        # at the earlier label clock. All twelve raw mirrors are recorded now.
        cutoff = max(0, 2 * count - 1)
        for index in range(count - 4, count):
            for example in temporal_examples(phase, "train", exposure=index % 4):
                request = example.request
                source = Experience(
                    request.request_id,
                    "clm-temporal-raw",
                    2 * index,
                    phase,
                    request,
                    "train",
                    self._permissions,
                    tuple(candidate.candidate_id for candidate in request.allowed_candidates),
                )
                label = LabelArrival(
                    "label-" + request.request_id,
                    request.request_id,
                    "clm-temporal-raw",
                    2 * index + 1,
                    phase,
                    example.target_id,
                    "train",
                )
                apply_decision_feedback(
                    self._store,
                    source,
                    label,
                    policy_version=example.policy_version,
                    source_group=example.source_group,
                    recorded_at=cutoff,
                )

    def __call__(self, phase, learner):
        self._require()
        if self._completed >= len(CHECKPOINTS) or CHECKPOINTS[self._completed][1] != phase:
            raise ValueError("memory checkpoint order or finite allowance differs")
        count = CHECKPOINTS[self._completed][0]
        _require_receipts(self._report, count)
        if learner.completed_updates != count:
            raise ValueError("memory checkpoint differs from actual learner progress")
        try:
            if count:
                self._record_phase(phase, count)
            cutoff = max(0, 2 * count - 1)
            row: dict[str, Any] = dict(phase=phase, source_update=count, cutoff=cutoff, calls=[])
            self._record["phases"].append(row)
            for example in temporal_examples(phase, "outer_selection"):
                call: dict[str, Any] = dict(request=asdict(example.request))
                row["calls"].append(call)
                selection = self._store.retrieve(
                    example.request,
                    cutoff=cutoff,
                    policy_version=policy_for_phase(phase)["version"],
                )
                call["selection"] = asdict(selection)
                result = decision_from_evidence(example.request, selection)
                call["result"] = None if result is None else asdict(result)
            self._paid_prefix = deepcopy(self._report["stream"]["receipts"])
            self._completed += 1
        except Exception as error:
            self._failed = True
            self._record["error"] = type(error).__name__ + ": " + str(error)[:512]
            raise
        finally:
            self._publish()

    def finish(self):
        """Score observed choices only after the protected TRAIN owner finishes."""
        self._require()
        _require_receipts(self._report, 12)
        if self._completed != 4 or self._report.get("sequence_complete") is not True:
            raise ValueError("memory scoring requires all TRAIN and four original checkpoints")
        if self._record["complete"]:
            raise ValueError("memory already completed; its work allowance cannot renew")
        scores = []
        for row in self._record["phases"]:
            cases = []
            for call, example in zip(
                row["calls"], temporal_examples(row["phase"], "outer_selection"), strict=True
            ):
                result = call["result"]
                selected = None if result is None else result["selected_id"]
                cases.append(
                    dict(
                        concept=example.concept,
                        selected_id=selected,
                        target_id=example.target_id,
                        correct=selected == example.target_id,
                    )
                )
            answered = sum(case["selected_id"] is not None for case in cases)
            correct = sum(case["correct"] is True for case in cases)
            scores.append(
                dict(
                    phase=row["phase"],
                    query_count=4,
                    answered_count=answered,
                    correct_count=correct,
                    coverage=answered / 4,
                    accuracy_answered=correct / answered if answered else None,
                    cases=cases,
                )
            )
        self._record.update(complete=True, scores=scores)
        self._publish()
