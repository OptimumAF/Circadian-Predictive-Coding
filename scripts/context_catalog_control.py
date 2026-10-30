"""D9 paired-head binding for the existing catalog/live schedule contract.

Reuse the original chooser, pin, candidate guard and capture methods. Override
only D7/BP source/packet/bank/accounting assumptions; no new scheduling or fit.
"""

from copy import deepcopy
from dataclasses import asdict
from time import monotonic

from scripts.context_catalog_replay import compose_context_replay
from scripts.run_ticket_workflow import CONTEXT_FEATURES
from scripts.ticket_catalog_control import CatalogTicketControl
from src.app.counterfactual_controller import CatalogCounterfactualRequest
from src.app.coupled_attribution import _same_value, _validated_batches
from src.app.resource_sharing import ResourceSharedRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.catalog_contextual_controller import ContextualArmCatalog, FittedCatalogController
from src.core.contextual_controller import training_context
from src.core.context_linear_state import ContextLinearFitSnapshot
from src.core.decision_features import DecisionFeatureSpec
from src.core.experience import require_identifier, require_tick
from src.core.ticket_workflow import CATALOG


def context_catalog():
    return ContextualArmCatalog(
        AdaptationEnvelope(
            (
                AdaptationAction("wait"),
                AdaptationAction("wake1", 1),
                AdaptationAction("replay6-wake1", 1, 6),
            ),
            14,
            2,
            12,
        )
    )


class ContextCatalogControl(CatalogTicketControl):
    def __init__(
        self, trial=None, policy=None, *, base_snapshot=None, resume_wake=False, linear_fit=False
    ):
        if type(linear_fit) is not bool:
            raise ValueError("context catalog LS binding requires an exact Boolean opt-in")
        self._linear_fit = self._linear_fit_pin = linear_fit
        if type(resume_wake) is not bool:
            raise ValueError("context wake continuation requires an exact Boolean opt-in")
        self.resume_wake = self._resume_pin = resume_wake
        self._continuation = self._continuation_owner = None
        if (trial is None) == (policy is None):
            raise ValueError("context catalog requires one private trial or frozen application")
        record = trial if trial is not None else policy
        if trial is not None:
            if type(trial) is not CatalogCounterfactualRequest or base_snapshot is None:
                raise ValueError("context catalog trial requires exact TRAIN request and full base")
            if trial.role != "controller_train" or trial.context not in ("clean", "mistakes"):
                raise ValueError("context catalog trial requires original TRAIN context")
            require_tick(trial.started_at, "trial start")
        elif type(policy) is not FittedCatalogController or base_snapshot is not None:
            raise ValueError("context catalog application requires exact frozen fit, no rebasing")
        else:
            policy.__post_init__()
        if (
            record.learner_version != ("context-ls-12" if linear_fit else "qualified-12")
            or type(record.catalog) is not ContextualArmCatalog
            or record.catalog != context_catalog()
        ):
            raise ValueError(
                "context catalog requires its explicit family source version and envelope"
            )
        record.catalog.__post_init__()
        require_identifier(record.state_digest, "context source digest")
        if trial is not None:
            record.catalog.envelope.select(trial.action_id)
        # Why: shared base behavior is geometry-independent. Its constructor is
        # deliberately D7-only, so admit this exact D9 variant here, not by rebasing.
        self.record, self.catalog = record, record.catalog
        self._record_owner = record
        self._pin, self._catalog_owner = deepcopy(record), self.catalog
        self.trial, self._batches = trial is not None, 1 if trial is not None else 2
        self.stream_recipe = self._stream_pin = None
        self.live_recipe = self._live_recipe_owner = self._live_recipe_pin = None
        self.replay = self.replay_budget = self._wake = self._wake_limits = None
        self._base = deepcopy(base_snapshot)
        self._packets = self._packet_pin = self._schedule_packets = None
        self._old = self._old_pin = None
        self._old_owner = self._packet_owner = None
        self.report = dict(
            phase="trial" if self.trial else "private_application",
            record=asdict(record),
            preparation=dict(attempts=0, returned=False),
            replay=None,
            replay_budget=None,
            source_binding_verified=False,
            promotions=0,
        )
        if linear_fit:
            self.report["learner_binding"] = dict(
                family="context_linear_fit",
                source_version="context-ls-12",
                history="12updates72rows, heads8/48+4/24; no Perceptron mistake counter",
                scope="typed metadata/digest binding; training provenance belongs to source owner",
            )

    @property
    def max_replays(self):
        return self.catalog.envelope.max_replay_examples // 6

    def prepare_packets(self, packets):
        self.require_pin()
        packets, _ = _validated_batches(packets, (), training_only=True)
        if (
            self._packets is not None
            or len(packets) != 2
            or any(
                b.context != context
                or b.features.shape != shape
                or b.num_classes != 6
                or b.episode is not None
                for b, context, shape in zip(
                    packets, ("new-v1", "old-v1"), ((3, 9), (6, 9)), strict=True
                )
            )
        ):
            raise ValueError("context catalog requires original two D9 TRAIN prototypes")
        self._old, self._old_pin = packets[1], deepcopy(packets[1])
        self._old_owner = self._old
        self._packets = (
            (packets[0 if self.record.context == "mistakes" else 1],) if self.trial else packets
        )
        self._packet_pin = deepcopy(self._packets)
        self._packet_owner = self._packets
        return self._packets

    def require_pin(self):
        if type(self._linear_fit) is not bool or self._linear_fit is not self._linear_fit_pin:
            raise ValueError("original context catalog learner family changed")
        if type(self.resume_wake) is not bool or self.resume_wake is not self._resume_pin:
            raise ValueError("original context wake continuation option changed")
        if self._continuation is not self._continuation_owner:
            raise ValueError("original context wake continuation owner replaced")
        if (
            self.record is not self._record_owner
            or self.catalog is not self._catalog_owner
            or self._old is not self._old_owner
            or self._packets is not self._packet_owner
        ):
            raise ValueError("original contextual catalog owner replaced")
        super().require_pin()
        if self._old is not None and not _same_value(self._old, self._old_pin):
            raise ValueError("original contextual replay TRAIN prototype changed")
        if self._packets is not None and not _same_value(self._packets, self._packet_pin):
            raise ValueError("original contextual schedule packets changed")

    def require_source(self, initial, base_snapshot, *, digest, same_state):
        self.require_pin()
        if not self._linear_fit and type(initial) is ContextLinearFitSnapshot:
            raise ValueError("context catalog LS source requires its explicit family opt-in")
        linear_pin = None
        if self._linear_fit:
            from src.core.context_linear_state import (
                ContextLinearFitPolicy,
                copy_context_linear_state,
            )

            linear_pin = copy_context_linear_state(initial)
            if linear_pin.policy != ContextLinearFitPolicy(
                9, 6, (CONTEXT_FEATURES.index("policy_v1"), CONTEXT_FEATURES.index("policy_v2"))
            ) or tuple((h.completed_updates, h.training_examples) for h in linear_pin.heads) != (
                (8, 48),
                (4, 24),
            ):
                raise ValueError("context catalog requires complete original LS head history")
        if (
            type(initial.completed_updates) is not int
            or initial.completed_updates != 12
            or type(initial.training_examples) is not int
            or initial.training_examples != 72
            or not self._linear_fit
            and (type(initial.mistake_updates) is not int or initial.mistake_updates != 16)
            or digest(initial) != self.record.state_digest
            or not same_state(base_snapshot, self._base)
            or self.trial
            and not same_state(base_snapshot, initial)
        ):
            raise ValueError("context catalog requires complete original qualified source/base")
        if self._linear_fit:
            self.require_pin()
            if not _same_value(initial, linear_pin):
                raise ValueError("original LS source changed during digest binding")
        self.report["source_binding_verified"] = True

    def prepare(self, wake, sampler):
        self.require_pin()
        if (
            not self.report["source_binding_verified"]
            or self.replay_budget is not None
            or self._old is None
        ):
            raise ValueError("context replay preparation requires original bound source/packets")
        self._wake, self._wake_limits = wake, wake.budget
        self.replay_budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=0,
                max_replay_examples=12,
                max_wall_seconds=45,
                max_process_rss_bytes=512 * 1024**2,
            ),
            monotonic,
        )
        self.replay_budget.attach_memory(sampler)

    def bind_schedule(self, packets, label_delay):
        self.before_work()
        if (
            self._schedule_packets is not None
            or label_delay != 1
            or not _same_value(packets, self._packet_pin)
        ):
            raise ValueError("context catalog requires original delay1 schedule")
        self._schedule_packets = packets
        self.report["stream"] = dict(
            natural_horizon=len(packets) + 1,
            observed_horizon=14,
            decision_cap=14,
            horizon_limited=False,
        )
        return 14

    def require_schedule_packets(self, packets):
        self.before_work()
        if packets is not self._schedule_packets or not _same_value(packets, self._packet_pin):
            raise ValueError("original contextual catalog schedule replaced or changed")

    def require_live(self, live):
        from scripts.context_head_live import live_plan
        from scripts.ticket_serving_timeline import TicketServingTimeline

        if type(live) is not TicketServingTimeline or live.plan != live_plan():
            raise ValueError("context catalog requires original fourteen-opportunity worker")

    def bind(self, runtime, source, gate):
        self.before_work()
        if (
            self.replay is not None
            or self.replay_budget is None
            or runtime._budget is not self._wake
            or self._wake.budget is not self._wake_limits
            or gate.snapshot().limits.max_admitted_updates != self.admissions
        ):
            raise ValueError("context catalog requires original budgets and shared admissions")
        if self._linear_fit:
            from src.adapters.context_linear_serving import snapshot_digest

            # Why: admission must bind the actual executor owners, not just a
            # previously checked snapshot with the same aggregate counters.
            self.require_source(
                source.snapshot_state(), self._base, digest=snapshot_digest, same_state=_same_value
            )
            if (
                runtime._base_actor_version != self.record.learner_version
                or snapshot_digest(runtime.candidate_snapshot().state) != self.record.state_digest
            ):
                raise ValueError(
                    "context catalog LS runtime/source differs from bound initial state"
                )
        self.report["preparation"]["attempts"] += 1
        self.replay = compose_context_replay(
            runtime,
            source,
            self.replay_budget,
            DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG),
            self._old,
            **(dict(linear_fit=True) if self._linear_fit else {}),
        )
        self.report["preparation"].update(
            returned=True, retained_episodes=2, retained_rows=6, retained_numeric_bytes=2608
        )
        shared = ResourceSharedRuntime(
            runtime, gate, action_envelope=self.catalog.envelope, replay_action=self.replay
        )
        if self.resume_wake:
            from scripts.context_catalog_continuation import ContextWakeContinuation

            self._continuation = ContextWakeContinuation(
                shared, runtime, gate, self.require_candidate
            )
            self._continuation_owner = self._continuation
        return shared

    def choose(self, observation, progress):
        self.require_pin()
        if self.resume_wake:
            if self._continuation is None:
                raise ValueError("context continuation requires original bound runtime")
            return self._continuation.begin(
                observation, progress, lambda: self._choose_original(observation, progress)
            )
        return self._choose_original(observation, progress)

    def _choose_original(self, observation, progress):
        context = training_context(observation)
        if self.trial and context is not None and context != self.record.context:
            raise ValueError("arrived TRAIN context differs from original private trial")
        return super().choose(observation, progress)

    def observe_poll(self, action_id, poll):
        self.require_pin()
        if not self.resume_wake or self._continuation is None:
            raise ValueError("context continuation requires explicit original opt-in")
        self._continuation.observe(action_id, poll)

    def train_action(self, action_id):
        self.require_pin()
        if not self.resume_wake or self._continuation is None:
            raise ValueError("context continuation requires explicit original opt-in")
        return self._continuation.execute(action_id)

    def capture(self):
        report = super().capture()
        if self._resume_pin:
            report["continuation"] = (
                dict(mode="finish_selected_replay_wake", not_bound=True)
                if self._continuation is None
                else self._continuation.snapshot()
            )
        return report

    def require_history(self, candidate, wake, receipts=None):
        if self._packets is None or self.replay is None:
            raise ValueError("context catalog history requires original bound packets and executor")
        self.require_candidate()
        if receipts is None or len(receipts) != wake.updates_completed:
            raise ValueError("context catalog requires original completed FIFO receipts")
        rows = 0
        for i, receipt in enumerate(receipts):
            if (
                receipt.sample_id != f"batch-{i}"
                or receipt.event_id != f"label-{i}"
                or receipt.update_number != i + 1
                or receipt.observed_at != i
                or receipt.arrived_at != i + 1
                or not receipt.arrived_at <= receipt.applied_at < 14
            ):
                raise ValueError("context catalog lost original causal TRAIN prefix")
            rows += len(self._packets[i].features)
        replay = self.replay.report()
        if (
            replay["stopped"]
            or replay["publication_unknown"]
            or replay["native_attempts"] != replay["native_completions"]
            or replay["native_completions"] != replay["committed_events"]
            or candidate.completed_updates != 12 + len(receipts) + replay["committed_events"]
            or candidate.training_examples != 72 + rows + replay["replay_examples_charged"]
        ):
            raise ValueError(
                "context catalog native history differs from observed wake/replay work"
            )
        self.report["native_history"] = dict(
            initial_updates=12,
            initial_rows=72,
            wake_updates=len(receipts),
            wake_rows=rows,
            replay_updates=replay["committed_events"],
            replay_rows=replay["replay_examples_charged"],
            completed_updates=candidate.completed_updates,
            training_rows=candidate.training_examples,
        )
