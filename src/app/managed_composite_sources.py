"""Explicit complete supported source schemas, read only under original leases.

Projects every field as data, nested state or an original reference. Native
projection is an inward port. No model operations, payload copying or publication.
"""

from src.app.actor_shadow import ActorShadowRuntime, StableActor
from src.app.managed_record_capture import _runtime_schema
from src.app.candidate_checkpoint import (
    CandidateCheckpointController,
    _Pending as CheckpointPending,
)
from src.app.experience_inbox import ExperienceInbox
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.resource_sharing import ServingPriorityGate
from src.app.serving_promotion import (
    PromotableActor,
    ServingPromotionController,
    _Bundle,
    _Slot,
    _Pending as PromotionPending,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionProgress
from src.core.experience import LogicalClock
from src.core.managed_composite_state import AuthorityPath, SourceRecord, NativeProjection
from src.core.managed_lifecycle_state import AuthorityReference
from src.shared.process_memory import ProcessRssSampler
from src.core.serving_ports import PreparedPromotion


def schema(data="", refs="", **nested):
    result = {name: "data" for name in data.split()}
    result.update({name: "ref" for name in refs.split()})
    result.update(nested)
    return result


SCHEMAS = {
    StableActor: schema(
        "_version _payload_ready", "_read_gate _payload_registry", _learner="native"
    ),
    PromotableActor: schema(
        "_version _payload_ready", "_read_gate _payload_registry _feature_digest", _slot="record"
    ),
    _Slot: schema("generation", bundle="record", previous="record"),
    _Bundle: schema("version configuration cache metadata last_cache_tick", learner="native"),
    ActorShadowRuntime: schema(
        "_base_actor_version _candidate_version _consolidation_limit _attempted_ids _consolidations _stopped _retired _revision _payload_ready",
        "_actor _budget _write_gate _payload_lineage",
        _candidate="native",
        _inbox="record",
    ),
    ExperienceInbox: schema(
        "_learner_version _capacity _experiences _labels _event_ids _applied _erased _historical_completed_updates _draining _stopped _last_time",
        "_learner _clock _budget _registration_guard _training_guard _payload_copy_guard",
    ),
    CandidateCheckpointController: schema(
        "_limit _attempt_limit _attempts _payload_ready",
        "_shared _build _digest _policy _gate",
        _models="models",
        _pending="pending",
    ),
    CheckpointPending: schema(
        "revision view integrity native_digest policy_digest",
        "owner build policy state_probe budget wall_clock event_clock sampler progress resource sharing_gate actor",
        sampler_state="sampler_state",
    ),
    ServingPromotionController: schema(
        "_policy _limit _payload_ready",
        "_actor _build _digest _gate",
        _evaluator="record",
        _pending="pending",
        _latest="receipt_state",
    ),
    PromotionPending: schema("report actor_generation", "runtime builder", bundle="record"),
    PromotionGuardEvaluator: schema(
        refs="_build _utility _actions _state_valid _prediction_valid _resource _digest _clock"
    ),
    ServingPriorityGate: schema(
        "_limits _serving _admitted _training _paused _checkpointing _deferrals",
        "_resource _retention_hold _gate",
    ),
    ToyBudgetSession: schema(
        "budget started_at last_clock updates_completed replay_examples_completed hidden_width_observed peak_hidden_width_observed rejected_proposed_hidden_width checkpoint_position process_rss_segment",
        "clock",
        progress="record",
        process_rss_sampler="record",
    ),
    ToyExecutionProgress: schema(
        "updates_completed replay_examples_completed checkpoint_position hidden_width_observed peak_hidden_width_observed rejected_proposed_hidden_width process_rss_segment"
    ),
    LogicalClock: schema("_time"),
    ProcessRssSampler: schema(
        "interval_seconds start_bytes peak_bytes sample_count",
        "read_rss_bytes _lock _error",
        _stop="event",
        _thread="thread",
    ),
}


class SourceReader:
    def __init__(self, project, limits):
        self.project, self.limits = project, limits
        self.authority: list[AuthorityReference] = []
        self.memo: dict[int, SourceRecord] = {}
        self.active: set[int] = set()

    def refresh_observations(self, budget, clock):
        """Rebuild only mutable observation records after final original admission."""
        for original in (budget, budget.progress, budget.process_rss_sampler, clock):
            if original is not None:
                self.memo.pop(id(original), None)
        self.authority = [
            ref
            for ref in self.authority
            if not ref.path.startswith(("composite.budget.", "composite.clock."))
        ]
        return self.record(budget, "composite.budget"), self.record(clock, "composite.clock")

    def reference(self, value, path):
        self.authority.append(AuthorityReference(path, value))
        return AuthorityPath(path, value is not None)

    def native(self, value, path):
        if id(value) in self.memo:
            return self.memo[id(value)]
        result = self.project(value, path, self.limits)
        if type(result) is not NativeProjection or type(result.state) is not SourceRecord:
            raise ValueError("native source port requires complete typed projection")
        self.memo[id(value)] = result.state
        self.authority.extend(result.authority)
        return result.state

    def record(self, value, path):
        if value is None:
            return None
        if id(value) in self.memo:
            return self.memo[id(value)]
        spec = SCHEMAS.get(type(value))
        if spec is None or vars(value).keys() != spec.keys():
            raise ValueError("composite source differs from complete explicit field schema")
        if type(value) is ActorShadowRuntime:
            _runtime_schema(value)
            if (
                value._inbox._learner is not value._candidate
                or value._inbox._budget is not value._budget
                or value._inbox._learner_version != value._candidate_version
            ):
                raise ValueError("retained runtime inbox learner/budget/version changed")
        if id(value) in self.active:
            raise ValueError("unsupported recursive source record")
        self.active.add(id(value))
        self.reference(value, path + ".original")
        items = tuple(
            (name, self.read(getattr(value, name), path + "." + name, role))
            for name, role in sorted(spec.items())
        )
        result = SourceRecord(type(value).__module__ + "." + type(value).__name__, items)
        self.active.remove(id(value))
        self.memo[id(value)] = result
        return result

    def read(self, value, path, role):
        if role == "data":
            return value
        if role == "ref":
            return self.reference(value, path)
        if role == "native":
            return self.native(value, path)
        if role == "record":
            return self.record(value, path)
        if role == "models":
            if type(value) is not list or len(value) > self.limits.records.max_records:
                raise ValueError("unsupported or oversized retained model list")
            return tuple(self.native(model, path + f".{i}") for i, model in enumerate(value))
        if role == "pending":
            if type(value) is not dict or len(value) > self.limits.records.max_records:
                raise ValueError("unsupported or oversized original pending history")
            return tuple(
                (
                    self.reference(token, path + f".{i}.token"),
                    self.record(pending, path + f".{i}.state"),
                    token if type(token) is PreparedPromotion else None,
                )
                for i, (token, pending) in enumerate(value.items())
            )
        if role == "receipt_state":
            if value is not None:
                self.reference(value[0], path + ".token")
            return value
        if role == "sampler_state":
            if value is None:
                return None
            if type(value) is not tuple or len(value) != 2:
                raise ValueError("unsupported original pending sampler observation")
            return self.reference(value[0], path + ".reader"), value[1]
        if role == "event":
            return self.reference(value, path), value.is_set()
        if role == "thread":
            return self.reference(value, path), value is not None and value.is_alive()
        raise ValueError("unknown trusted source projection role")
