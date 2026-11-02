"""Ordinary shared TRAIN publication through existing actor/guard/sharing ports.

Own one birth actor/gate and one bounded current source/paid record. The existing
runner owns training, data, budgets and task protocol. No saved-source loader,
learner, solver, policy search, probe/final data or renewed authority lives here.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from typing import Any

from scripts.context_head_serving import policy
from scripts.run_ticket_workflow import CONTEXT_TASKS
from src.adapters import linear_fit_serving as binding
from src.app.coupled_attribution import _same_value
from src.app.resource_sharing import ServingPriorityGate
from src.app.serving_promotion import PromotableActor, ServingPromotionController, PromotionRejected
from src.app.ticket_workflow import run_workflow_episode
from src.core.actor_ports import VersionedState
from src.core.promotion_guard import GuardBatch
from src.core.resource_sharing import SharingLimits
from src.core.serving_ports import ServingConfiguration
from src.core.ticket_workflow import WorkflowState, WorkflowCapabilities, ACTION_IDS


def require_mode(options):
    value = options.get("shared_serving", False)
    if type(value) is not bool:
        raise ValueError("shared serving requires an exact boolean opt-in")
    if not value:
        return
    if (
        options.get("method") != "linear_fit"
        or options.get("context_return") is not True
        or options.get("train_update") != "minimum_change"
        or options.get("replay") != "coverage"
        or options.get("replay_joint") not in ("joint", "current_only")
        or any(
            options.get(k) not in (None, False)
            for k in (
                "separate_context_heads",
                "policy_change",
                "private_adaptation",
                "reserved_probe",
                "paired_serving",
                "adaptive_serving",
                "calibrated_serving",
                "catalog_action",
                "catalog_context",
                "catalog_study",
                "reference_validation",
                "record_train_fit",
                "class_balanced_bp",
                "train_readout",
                "probe_rehearsal",
                "probe_return_once",
                "probe_initialization",
                "probe_update",
                "incident_stream",
                "train_readout_cohort",
                "probe_output_only",
                "probe_update_diagnostics",
            )
        )
    ):
        raise ValueError(
            "shared serving requires ordinary shared minimum-change coverage replay alone"
        )


class SharedLinearPublication:
    def __init__(self, learner, spec, budget, batches):
        self._source_owner = learner
        self.spec = deepcopy(spec)
        self._spec_pin = deepcopy(spec)
        self.budget = budget
        self._budget_owner = budget
        self.stream: Any = None
        self.actor = PromotableActor(
            learner,
            version="a0",
            configuration=ServingConfiguration(3, 2),
            feature_digest=lambda x: sha256(x.tobytes()).hexdigest(),
            metadata={"mode": "ordinary_shared"},
        )
        self.gate = ServingPriorityGate(SharingLimits(1, 16, 1), resource_available=lambda: True)
        self._gate_owners = (id(self.gate), id(self.gate._limits), id(self.gate._resource))
        self._gate_pin = self.gate.snapshot()
        self._actor_pin = self.actor.snapshot_state()
        self._generation = 0
        self._initial_actor = deepcopy(self._actor_pin)
        self._initial_serving = self.actor.serving_snapshot()
        self._serving_pin = deepcopy(self._initial_serving)
        self._lease: Any = None
        self._source_pin: Any = None
        self.reader = binding.LinearFitServingPredictor(self.actor, self.spec)
        self.row: dict[str, Any] = dict(
            complete=False,
            status="training",
            fit_admissions=0,
            fit_returns=0,
            fit_work_unknown=False,
            guard_attempts=0,
            guard_returns=0,
            guard_rows=0,
            commit_attempts=0,
            commit_returns=0,
            commit_unknown=False,
            rollback_attempts=0,
            rollback_returns=0,
            rollback_unknown=False,
            requests=0,
            request_returns=0,
            episodes=[],
            guard_report=None,
            promotion_receipt=None,
            guard_scope="fixed TRAIN-derived development inner_guard copies; disjoint IDs, not heldout data",
        )
        self._row_pin = deepcopy(self.row)
        self.guards = tuple(
            GuardBatch(
                name,
                tuple((name, str(i)) for i in range(6)),
                0,
                batches[index].features.copy(),
                batches[index].targets.copy(),
                "inner_guard",
            )
            for name, index in (("shared-inner-new", 4), ("shared-inner-old", 0))
        )
        self._guards_pin = deepcopy(self.guards)

    @property
    def expected_actor(self):
        return deepcopy(self._actor_pin)

    def _record(self, **values):
        if not _same_value(self.row, self._row_pin):
            raise ValueError("shared paid serving evidence changed")
        self.row.update(values)
        self._row_pin = deepcopy(self.row)

    def require(self):
        if self.budget is not self._budget_owner or (
            self.stream is not None and self.stream._budget is not self._budget_owner
        ):
            raise ValueError("original shared publication budget replaced")
        if (
            not _same_value(self.row, self._row_pin)
            or not _same_value(self.guards, self._guards_pin)
            or not _same_value(self.spec, self._spec_pin)
        ):
            raise ValueError("shared paid serving evidence or guard inputs changed")
        if (
            id(self.gate),
            id(self.gate._limits),
            id(self.gate._resource),
        ) != self._gate_owners or not _same_value(self.gate.snapshot(), self._gate_pin):
            raise ValueError("original shared gate changed")
        if self.actor.snapshot_state() != self._actor_pin or not _same_value(
            self.actor.serving_snapshot(), self._serving_pin
        ):
            raise ValueError("unobserved shared actor change")

    def _gate_changed(self, expected):
        if not _same_value(self.gate.snapshot(), expected):
            raise ValueError("unobserved sharing-counter change")
        self._gate_pin = expected

    def before_fit(self):
        self.require()
        self._lease = self.gate.training()
        admission = self._lease.__enter__()
        if not admission.allowed:
            if admission.reason is None:
                raise ValueError("denied source admission has no reason")
            # Only this denied admission owns the exact one deferral increment.
            # Never refresh from arbitrary observed sharing-counter changes.
            deferrals = dict(self._gate_pin.deferrals)
            deferrals[admission.reason] = deferrals.get(admission.reason, 0) + 1
            self._gate_changed(replace(self._gate_pin, deferrals=tuple(sorted(deferrals.items()))))
            self._record(sharing=asdict(self._gate_pin))
            self._lease.__exit__(None, None, None)
            self._lease = None
            raise ValueError("source fitting deferred: " + str(admission.reason))
        self._gate_changed(
            replace(
                self._gate_pin,
                training_active=True,
                admitted_updates=self._gate_pin.admitted_updates + 1,
            )
        )
        self._record(fit_admissions=self.row["fit_admissions"] + 1, fit_work_unknown=True)

    def after_fit(self):
        self.require()
        self._lease.__exit__(None, None, None)
        self._lease = None
        self._gate_changed(replace(self._gate_pin, training_active=False))
        self._record(fit_returns=self.row["fit_returns"] + 1, fit_work_unknown=False)

    def _source_value(self):
        stream = self.stream
        runtime = stream._runtime
        return dict(
            owners=(
                id(stream),
                id(stream._learner),
                id(runtime),
                id(runtime._candidate),
                id(runtime._inbox),
            ),
            candidate=runtime._candidate.snapshot_state(),
            revision=runtime._revision,
            inbox=runtime._inbox.capture_cursor(),
            # Guard evaluation already holds the runtime's candidate lease.
            # Read its exact receipts without recursively acquiring that lease;
            # source fitting is paused, and before/after checks retain the pin.
            wake=runtime._inbox.applied_updates,
            replay=tuple(runtime._consolidations),
            completed=stream.completed_updates,
            sessions=[
                (
                    id(b),
                    id(b.budget),
                    asdict(b.budget),
                    b.started_at,
                    id(b.clock),
                    id(b.process_rss_sampler),
                    id(b.progress),
                    b.updates_completed,
                    b.replay_examples_completed,
                )
                for b in stream._sessions()
            ],
            clocks=[b.last_clock for b in stream._sessions()],
            rehearsal=stream._rehearsal.report,
        )

    def _readonly(self):
        self.require()
        self.context_guard()
        self.stream._check_original()
        self.stream._rehearsal._check_schema()
        if not _same_value(self._source_value(), self._source_pin):
            raise ValueError("original shared source/paid prefix changed during publication")

    def _budget_read(self):
        self._readonly()
        self.budget.before_final()
        self.observe_clock()

    def observe_clock(self):
        expected = deepcopy(self._source_pin)
        expected["clocks"][0] = self.budget.last_clock
        if not _same_value(self._source_value(), expected):
            raise ValueError("source changed beyond observed original clock check")
        self._source_pin = expected

    def _guard_measure(self, model, batch, policy, last):
        # Keep the shared factory plain: the controller also uses it to build
        # the published model, whose predict must never capture source locks.
        self._budget_read()
        self._record(
            guard_attempts=self.row["guard_attempts"] + 1,
            guard_rows=self.row["guard_rows"] + len(batch.features),
        )
        try:
            result = self._measure(model, batch, policy, last)
            self._record(guard_returns=self.row["guard_returns"] + 1)
            return result
        finally:
            self._readonly()

    def _predict(self, request):
        self._budget_read()
        self._record(requests=self.row["requests"] + 1)
        with self.gate.serving():
            self._gate_changed(replace(self._gate_pin, active_serving_requests=1))
            try:
                served = self.reader.serve(request)
                if (served.actor_version, served.generation) != (
                    self._actor_pin.actor_version,
                    self._generation,
                ):
                    raise ValueError("serving returned a foreign slot")
                self._record(request_returns=self.row["request_returns"] + 1)
                return served.result
            finally:
                self._record(
                    typed_work=asdict(self.reader.native_work), slot_work=dict(self.reader.work)
                )
                # The owned serving lease releases before the next observation.
                self._readonly()

    def _served(self, request):
        try:
            return self._predict(request)
        finally:
            self._gate_changed(replace(self._gate_pin, active_serving_requests=0))

    def _episodes(self, phase, scenarios):
        for scenario in scenarios:
            for version, concept in CONTEXT_TASKS:
                self._budget_read()
                episode = run_workflow_episode(
                    WorkflowState(f"shared-{phase}-{concept}", concept, policy_version=version),
                    self._served,
                    capabilities=WorkflowCapabilities(ACTION_IDS),
                    max_steps=6,
                    scenario=scenario,
                    include_policy_context=True,
                )
                self._readonly()
                self._record(
                    episodes=[
                        *self.row["episodes"],
                        dict(
                            phase=phase,
                            policy_version=version,
                            concept=concept,
                            role="outer_selection",
                            episode=episode,
                        ),
                    ]
                )
                if episode["complete"] is not True:
                    raise ValueError("shared task work incomplete")

    def run(self, stream, require_source, context_guard):
        if (
            self.row["status"] != "training"
            or self.row["fit_admissions"] != self.row["fit_returns"]
            or self.row["fit_returns"] != 16
        ):
            raise ValueError("shared publication requires its one complete original TRAIN sequence")
        self.stream = stream
        self.context_guard = context_guard
        if (
            stream._learner is not self._source_owner
            or stream._runtime.actor is not self.actor
            or stream._budget is not self._budget_owner
        ):
            raise ValueError("original shared source or actor replaced")
        require_source()
        self.require()
        self.gate.pause()
        self._gate_changed(replace(self._gate_pin, paused=True))
        self._source_pin = deepcopy(self._source_value())
        guard = binding.categorical_guard(self.spec, self.budget.clock)
        self._measure = guard._measure
        setattr(guard, "_measure", self._guard_measure)
        controller = ServingPromotionController(
            self.actor,
            policy=replace(policy(), resource_metric_id="complete_shared_linear_record_bytes"),
            evaluator=guard,
            build_learner=binding.build_linear,
            state_digest=binding.snapshot_digest,
            max_pending=1,
        )
        self.controller = controller
        self._record(status="guarding")
        self._budget_read()
        try:
            ticket = controller.prepare(
                stream._runtime, *self.guards, now=stream._next_tick, training_ids=frozenset()
            )
        except PromotionRejected as refused:
            self._readonly()
            self._record(guard_report=asdict(refused.report), complete=True, status="rejected")
            return
        self._readonly()
        self._record(guard_report=asdict(ticket.report))
        require_source()
        self._budget_read()
        self._record(commit_attempts=1, commit_unknown=True, status="committing")
        receipt = controller.commit(stream._runtime, ticket)
        expected = VersionedState("c0", deepcopy(self._source_pin["candidate"]))
        if not _same_value(
            asdict(receipt), dict(generation=1, previous_version="a0", actor_version="c0")
        ):
            raise ValueError("returned shared publication receipt differs")
        self._actor_pin = expected
        self._generation = 1
        self._serving_pin = replace(
            self._initial_serving,
            generation=1,
            actor_version="c0",
            model_state=deepcopy(expected.state),
        )
        self._readonly()
        self._record(
            commit_returns=1,
            commit_unknown=False,
            promotion_receipt=asdict(receipt),
            status="published",
        )
        self._episodes("published", ("ordinary", "stale_once", "revoke_commit"))
        self._budget_read()
        self._record(rollback_attempts=1, rollback_unknown=True)
        generation = controller.rollback(receipt)
        if type(generation) is not int or generation != 2:
            raise ValueError("shared rollback generation differs")
        self._actor_pin = deepcopy(self._initial_actor)
        self._generation = 2
        self._serving_pin = replace(self._initial_serving, generation=2)
        self._readonly()
        self._record(rollback_returns=1, rollback_unknown=False, status="rolled_back")
        self._episodes("rollback", ("ordinary",))
        require_source()
        self._record(complete=True, status="published_and_rolled_back")

    def close(self):
        if self._lease is not None:
            self._lease.__exit__(None, None, None)
            self._lease = None
            self._gate_changed(replace(self._gate_pin, training_active=False))
        self._record(sharing=asdict(self.gate.snapshot()))

    def preserved_work(self):
        return deepcopy(self._row_pin)
