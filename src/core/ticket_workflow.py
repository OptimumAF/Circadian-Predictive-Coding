"""Pure bounded ticket transitions, model-external capabilities and observations.

Inputs are owned scalar state, selected tool, observed revision and capabilities.
Outputs are typed local requests/transitions. No IO, learner, scoring schedule,
arbitrary tool handlers or Python-process containment is provided here.
"""

from dataclasses import dataclass, fields, replace
from typing import Literal

from src.core.temporal_decision_sequence import CATALOG as ROUTING_CATALOG
from src.core.temporal_decision_sequence import VOCABULARY, policy_for_phase
from src.core.typed_decision import DecisionCandidate, DecisionRequest, _text

CATALOG = tuple(
    sorted(
        (
            *ROUTING_CATALOG,
            DecisionCandidate("open", "open or refresh the ticket before choosing a workflow"),
            DecisionCandidate("commit", "commit the staged ticket workflow"),
        ),
        key=lambda c: c.candidate_id,
    )
)
ACTION_IDS = tuple(c.candidate_id for c in CATALOG)
POLICY_CONTEXT_TOKENS = ("policy_v1", "policy_v2")
Status = Literal[
    "applied",
    "permission_denied",
    "stale_revision",
    "wrong_phase",
    "inactive_tool",
    "already_complete",
]
STATUSES = (
    "applied",
    "permission_denied",
    "stale_revision",
    "wrong_phase",
    "inactive_tool",
    "already_complete",
)


def _count(value, name):
    if type(value) is not int or not 0 <= value <= 2**53 - 1:
        raise ValueError(f"{name} must be a nonnegative safe native integer")


def _policy(version):
    if type(version) is not str or version not in ("v1", "v2"):
        raise ValueError("unknown workflow policy version")
    return policy_for_phase("after_A" if version == "v1" else "after_return")


def _exact(value, kind):
    if type(value) is not kind or set(vars(value)) != {f.name for f in fields(kind)}:
        raise ValueError(f"expected exact complete {kind.__name__}")
    value.__post_init__()


@dataclass(frozen=True)
class WorkflowState:
    ticket_id: str
    concept: str
    policy_version: str = "v2"
    revision: int = 0
    phase: str = "sealed"
    staged_tool: str | None = None
    committed_tool: str | None = None

    def __post_init__(self):
        _text(self.ticket_id, "ticket_id", 64)
        policy = _policy(self.policy_version)
        if type(self.concept) is not str or self.concept not in VOCABULARY:
            raise ValueError("unknown ticket concept")
        _count(self.revision, "revision")
        if type(self.phase) is not str or self.phase not in (
            "sealed",
            "ready",
            "staged",
            "committed",
        ):
            raise ValueError("unknown ticket phase")
        if self.revision < {"sealed": 0, "ready": 1, "staged": 2, "committed": 3}[self.phase]:
            raise ValueError("ticket revision precedes its phase")
        if self.phase in ("sealed", "ready"):
            if self.staged_tool is not None or self.committed_tool is not None:
                raise ValueError("unstaged ticket contains a workflow effect")
        elif type(self.staged_tool) is not str or self.staged_tool not in policy["allowed_ids"]:
            raise ValueError("staged tool is not active under the ticket policy")
        if self.phase == "staged" and self.committed_tool is not None:
            raise ValueError("uncommitted ticket contains a committed effect")
        if self.phase == "committed" and self.committed_tool != self.staged_tool:
            raise ValueError("committed tool must equal the staged tool")


@dataclass(frozen=True)
class WorkflowCapabilities:
    allowed: tuple[str, ...]
    version: int = 0

    def __post_init__(self):
        _count(self.version, "capability version")
        if (
            type(self.allowed) is not tuple
            or len(self.allowed) > 6
            or any(type(a) is not str or a not in ACTION_IDS for a in self.allowed)
            or tuple(sorted(set(self.allowed))) != self.allowed
        ):
            raise ValueError("capabilities require unique sorted known tool IDs")


@dataclass(frozen=True)
class WorkflowTransition:
    state: WorkflowState
    status: Status

    def __post_init__(self):
        _exact(self.state, WorkflowState)
        if type(self.status) is not str or self.status not in STATUSES:
            raise ValueError("unknown workflow transition status")


class NoWorkflowCandidates(ValueError):
    """No active tool is permitted; this is a terminal task refusal, not a fit."""


def workflow_request(
    state, capabilities, request_id, last_status="initial", *, include_policy_context=False
) -> DecisionRequest:
    if type(include_policy_context) is not bool:
        raise ValueError("include_policy_context must be an exact bool")
    _exact(state, WorkflowState)
    _exact(capabilities, WorkflowCapabilities)
    if type(last_status) is not str or (last_status != "initial" and last_status not in STATUSES):
        raise ValueError("unknown workflow feedback status")
    if state.phase == "committed":
        raise ValueError("completed ticket has no next decision")
    active = (*_policy(state.policy_version)["allowed_ids"], "open", "commit")
    # Why this: phase legality is not a candidate mask; the policy chooses every step.
    candidates = tuple(
        c for c in CATALOG if c.candidate_id in active and c.candidate_id in capabilities.allowed
    )
    if not candidates:
        raise NoWorkflowCandidates("no active workflow tool is permitted")
    query = state.phase + (" " + state.concept if state.phase == "ready" else "")
    # Why: public policy context distinguishes incompatible rules on identical
    # phase/concept inputs. It supplies no target or extra permission.
    if include_policy_context:
        query += " policy_" + state.policy_version
    return DecisionRequest(request_id, query + " feedback " + last_status, candidates)


def dispatch_workflow(state, action_id, observed_revision, capabilities) -> WorkflowTransition:
    _exact(state, WorkflowState)
    _exact(capabilities, WorkflowCapabilities)
    _count(observed_revision, "observed revision")
    if type(action_id) is not str or action_id not in ACTION_IDS:
        raise ValueError("unknown workflow tool")
    if observed_revision != state.revision:
        return WorkflowTransition(state, "stale_revision")
    if action_id not in capabilities.allowed:
        return WorkflowTransition(state, "permission_denied")
    active = _policy(state.policy_version)["allowed_ids"]
    if action_id not in (*active, "open", "commit"):
        return WorkflowTransition(state, "inactive_tool")
    if state.phase == "committed":
        return WorkflowTransition(state, "already_complete")
    changes: dict = {}
    if state.phase == "sealed" and action_id == "open":
        changes = dict(phase="ready")
    elif state.phase == "ready" and action_id in active:
        changes = dict(phase="staged", staged_tool=action_id)
    elif state.phase == "staged" and action_id == "commit":
        changes = dict(phase="committed", committed_tool=state.staged_tool)
    else:
        return WorkflowTransition(state, "wrong_phase")
    # Ground truth does not repair a permitted wrong choice; success is scored later.
    return WorkflowTransition(replace(state, revision=state.revision + 1, **changes), "applied")


def invalidate_workflow(state) -> WorkflowState:
    _exact(state, WorkflowState)
    if state.phase == "committed":
        raise ValueError("cannot invalidate a completed ticket")
    return replace(
        state, revision=state.revision + 1, phase="sealed", staged_tool=None, committed_tool=None
    )


def workflow_success(state) -> bool:
    _exact(state, WorkflowState)
    return (
        state.phase == "committed"
        and state.committed_tool == _policy(state.policy_version)["rules"][state.concept]
    )
