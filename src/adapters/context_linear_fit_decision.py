"""Frozen two-head typed LS readout using established public-context routing.

Counts the extra full-query projection and both selected/unfitted head reads.
No fitting, context discovery, labels for routing, mutation, scheduling or IO.
"""

from copy import deepcopy
from dataclasses import dataclass, fields, replace
import re

from src.adapters.context_perceptron import _head_spec
from src.adapters.linear_fit_decision import LinearFitDecisionPredictor, LinearDecisionWork
from src.core.context_linear_state import copy_context_linear_state, routing_policy
from src.core.context_perceptron import route_context_features
from src.core.decision_features import encode_decision
from src.core.typed_decision import validate_decision_result


@dataclass(frozen=True)
class ContextRoutingWork:
    decision_attempts: int = 0
    decision_returns: int = 0
    routing_attempts: int = 0
    routing_returns: int = 0
    routing_rows: int = 0
    routing_token_visits: int = 0


@dataclass(frozen=True)
class ContextLinearDecisionWork(LinearDecisionWork):
    routing_attempts: int = 0
    routing_returns: int = 0
    routing_rows: int = 0
    routing_token_visits: int = 0


class ContextLinearFitDecisionPredictor:
    def __init__(self, spec, state):
        state = copy_context_linear_state(state)
        head_spec = _head_spec(spec, state)
        self.spec, self.state = deepcopy((spec, state))
        self._pin = deepcopy((self.spec, self.state))
        self._owners = (id(self.spec), id(self.state))
        self._readers = tuple(LinearFitDecisionPredictor(head_spec, h.model) for h in state.heads)
        self._original_readers = self._readers
        self._routing = ContextRoutingWork()
        self._routing_owner = self._routing
        self._routing_pin = deepcopy(self._routing)

    @property
    def work(self):
        reads = [reader.work for reader in self._original_readers]
        values = {
            f.name: sum(getattr(w, f.name) for w in reads) for f in fields(LinearDecisionWork)
        }
        for key in ("decision_attempts", "decision_returns"):
            values[key] = getattr(self._routing_pin, key)
        values["token_visits"] += self._routing_pin.routing_token_visits
        values.update(
            {
                f.name: getattr(self._routing_pin, f.name)
                for f in fields(ContextRoutingWork)
                if f.name.startswith("routing_")
            }
        )
        return ContextLinearDecisionWork(**values)

    def assert_unchanged(self):
        if (
            self._owners != (id(self.spec), id(self.state))
            or (self.spec, self.state) != self._pin
            or self._readers is not self._original_readers
            or self._routing is not self._routing_owner
            or self._routing != self._routing_pin
        ):
            raise ValueError("frozen context linear readout binding/work changed")
        head_spec = _head_spec(self.spec, self.state)
        for reader, head in zip(self._original_readers, self.state.heads, strict=True):
            if reader.spec != head_spec or reader.model != head.model:
                raise ValueError("frozen context linear head binding changed")
            reader.assert_unchanged()

    def _charge(self, **increments):
        self.assert_unchanged()
        self._routing = replace(
            self._routing, **{k: getattr(self._routing, k) + v for k, v in increments.items()}
        )
        self._routing_owner = self._routing
        self._routing_pin = deepcopy(self._routing)

    def __call__(self, request):
        self._charge(decision_attempts=1, routing_attempts=1)
        features, visits = encode_decision(request, self.spec)
        self._charge(routing_token_visits=visits)
        tokens = [
            w for w in re.findall(r"\w+", request.query.casefold()) if w.startswith("policy_")
        ]
        if len(tokens) != 1 or tokens[0] not in ("policy_v1", "policy_v2"):
            raise ValueError("typed context must be one unambiguous known public policy token")
        index, _ = route_context_features(features, routing_policy(self.state.policy))
        self._charge(routing_returns=1, routing_rows=1)
        result = self._original_readers[index](request)
        # Both full-context projection and selected-head tokenization are paid.
        result = replace(result, token_visits=result.token_visits + visits)
        validate_decision_result(request, result)
        self._charge(decision_returns=1)
        return result
