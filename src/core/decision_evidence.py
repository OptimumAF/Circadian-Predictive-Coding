"""Bounded TRAIN evidence, causal label revisions and deterministic retrieval.

Owns immutable local records and canonical UTF-8/context budgets. No IO, model,
instruction editing, fitting, final/probe data or source-provenance attestation.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
import json
import re

from src.core.experience import ExperiencePermissions, ExperienceRole
from src.core.typed_decision import DecisionRequest, _count, _text


def _exact(value, kind):
    if type(value) is not kind or set(vars(value)) != {f.name for f in fields(kind)}:
        raise ValueError(f"expected exact complete {kind.__name__}")
    value.__post_init__()


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class DecisionEvidence:
    evidence_id: str
    observed_at: int
    available_at: int
    recorded_at: int
    policy_version: str
    source_group: str
    request: DecisionRequest
    target_id: str
    role: ExperienceRole = "train"
    permissions: ExperiencePermissions = ExperiencePermissions()

    def __post_init__(self):
        for name in ("evidence_id", "policy_version", "source_group", "target_id"):
            _text(getattr(self, name), name, 64)
        for name in ("observed_at", "available_at", "recorded_at"):
            _count(getattr(self, name), name)
        if not self.observed_at <= self.available_at <= self.recorded_at:
            raise ValueError("evidence requires observation <= label availability <= recording")
        _exact(self.request, DecisionRequest)
        _exact(self.permissions, ExperiencePermissions)
        if type(self.role) is not str or self.role != "train":
            raise ValueError("decision evidence admits TRAIN only, never heldout/probe data")
        if not self.permissions.training or not self.permissions.replay:
            raise ValueError("decision evidence requires explicit TRAIN and replay reuse consent")
        if self.target_id not in {c.candidate_id for c in self.request.allowed_candidates}:
            raise ValueError("evidence target must be an observed allowed candidate")


@dataclass(frozen=True)
class EvidencePolicy:
    max_events: int = 32
    max_storage_bytes: int = 32768
    max_context_bytes: int = 1536
    top_k: int = 4

    def __post_init__(self):
        for name, low, high in (
            ("max_events", 1, 64),
            ("max_storage_bytes", 64, 65536),
            ("max_context_bytes", 32, 1536),
            ("top_k", 1, 8),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in {low}..{high}")


@dataclass(frozen=True)
class EvidenceRevision:
    evidence: DecisionEvidence
    revision: int


@dataclass(frozen=True)
class EvidenceSelection:
    items: tuple[EvidenceRevision, ...]
    context: str
    work: dict[str, int]


class DecisionEvidenceStore:
    def __init__(self, policy=EvidencePolicy()):
        _exact(policy, EvidencePolicy)
        self._policy = deepcopy(policy)
        self._events: tuple[EvidenceRevision, ...] = ()
        self._read_through = -1
        self._storage_bytes = len(_json([]).encode("utf8"))
        self._work = dict.fromkeys(
            (
                "writes_attempted",
                "writes_completed",
                "retrieval_attempts",
                "retrievals_completed",
                "metadata_scans",
                "candidate_comparisons",
                "token_visits",
                "rendered_bytes",
            ),
            0,
        )

    @property
    def events(self):
        return deepcopy(self._events)

    @property
    def storage_bytes(self):
        """Canonical logical event bytes; Python objects/copies are additional."""
        return self._storage_bytes

    @property
    def work(self):
        return dict(self._work)

    def _commit(self, evidence, revision):
        _exact(evidence, DecisionEvidence)
        # A delivered (or attempted) view seals its cutoff. Later writes cannot
        # backfill that view, including additions of previously unseen IDs.
        if evidence.recorded_at <= self._read_through:
            raise ValueError("evidence write would change a previously observed cutoff")
        if self._events and evidence.recorded_at < self._events[-1].evidence.recorded_at:
            raise ValueError("evidence writes cannot move recording time backwards")
        if len(self._events) >= self._policy.max_events:
            raise ValueError("evidence event allowance exhausted; no automatic eviction")
        staged = (*self._events, EvidenceRevision(evidence, revision))
        size = len(_json([asdict(event) for event in staged]).encode("utf8"))
        if size > self._policy.max_storage_bytes:
            raise ValueError("evidence canonical storage byte allowance exceeded")
        self._events = deepcopy(staged)
        self._storage_bytes = size
        self._work["writes_completed"] += 1
        return deepcopy(self._events[-1])

    def add(self, evidence):
        self._work["writes_attempted"] += 1
        _exact(evidence, DecisionEvidence)
        if any(e.evidence.evidence_id == evidence.evidence_id for e in self._events):
            raise ValueError("evidence ID already exists; use an explicit revision edit")
        return self._commit(evidence, 0)

    def edit(self, evidence_id, target_id, *, available_at, recorded_at, expected_revision):
        self._work["writes_attempted"] += 1
        _text(evidence_id, "evidence_id", 64)
        _count(expected_revision, "expected_revision")
        prior = next(
            (e for e in reversed(self._events) if e.evidence.evidence_id == evidence_id), None
        )
        if prior is None or prior.revision != expected_revision:
            raise ValueError("evidence edit requires the current expected revision")
        if type(available_at) is not int or available_at < prior.evidence.available_at:
            raise ValueError("evidence edit cannot backdate label availability")
        if type(recorded_at) is not int or recorded_at <= prior.evidence.recorded_at:
            raise ValueError("evidence edit requires a later recording time")
        # Preserve observation, request, policy, source and consent. A new policy
        # observation is a new record; editing cannot rewrite that historical fact.
        updated = replace(
            prior.evidence, target_id=target_id, available_at=available_at, recorded_at=recorded_at
        )
        return self._commit(updated, prior.revision + 1)

    def retrieve(self, request, *, cutoff, policy_version):
        _exact(request, DecisionRequest)
        _count(cutoff, "cutoff")
        _text(policy_version, "policy_version", 64)
        self._work["retrieval_attempts"] += 1
        self._read_through = max(self._read_through, cutoff)
        work = dict(metadata_scans=0, candidate_comparisons=0, token_visits=0, rendered_bytes=0)
        latest = {}
        for event in self._events:
            work["metadata_scans"] += 1
            record = event.evidence
            if record.policy_version == policy_version and record.recorded_at <= cutoff:
                latest[record.evidence_id] = event
        words = re.findall(r"\w+", request.query.casefold())
        work["token_visits"] += len(words)
        query = set(words)
        allowed = {c.candidate_id for c in request.allowed_candidates}
        ranked = []
        for event in latest.values():
            record = event.evidence
            # Resolve latest before filtering: never resurrect an older allowed
            # label after a newer revision names a currently inactive target.
            if record.target_id not in allowed:
                continue
            terms = re.findall(r"\w+", record.request.query.casefold())
            work["token_visits"] += len(terms)
            work["candidate_comparisons"] += 1
            union = query | set(terms)
            score = len(query & set(terms)) / len(union) if union else 0
            if score > 0:
                ranked.append((-score, -record.recorded_at, record.evidence_id, event))
        ranked.sort(key=lambda row: row[:3])
        items = tuple(row[3] for row in ranked[: self._policy.top_k])
        # Structured data only. Instructions and candidate definitions are not
        # editable fields of this context. This is not semantic injection defense.
        context = _json(
            {
                "decision_evidence": [
                    dict(
                        evidence_id=e.evidence.evidence_id,
                        revision=e.revision,
                        query=e.evidence.request.query,
                        target_id=e.evidence.target_id,
                        observed_at=e.evidence.observed_at,
                        available_at=e.evidence.available_at,
                        recorded_at=e.evidence.recorded_at,
                        policy_version=e.evidence.policy_version,
                        source_group=e.evidence.source_group,
                    )
                    for e in items
                ]
            }
        )
        work["rendered_bytes"] = len(context.encode("utf8"))
        for key, value in work.items():
            self._work[key] += value
        if work["rendered_bytes"] > self._policy.max_context_bytes:
            raise ValueError("evidence context byte allowance exceeded; no truncation")
        self._work["retrievals_completed"] += 1
        return EvidenceSelection(deepcopy(items), context, dict(work))
