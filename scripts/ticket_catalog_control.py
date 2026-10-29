"""Opt-in catalog boundary and replay accounting for the existing ticket driver.

Reuses the original teacher compiler, fixed TRAIN bank, shared executor and cost
records. No learner algorithm, task evaluator, promotion, final/probe access or
new stream runner belongs here.
"""

from copy import deepcopy
from dataclasses import asdict
from time import monotonic
from typing import Any

from src.app.counterfactual_controller import CatalogCounterfactualRequest
from src.app.categorical_replay_action import CategoricalReplayAction
from src.app.coupled_attribution import AttributionBatch, _same_value
from src.app.resource_sharing import ResourceSharedRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.adapters.torch_feature_learner import TORCH_CE_ID
from src.core.catalog_contextual_controller import (
    ContextualArmCatalog,
    FittedCatalogController,
    CatalogTrialWork,
    choose_catalog_contextual,
)
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.experience import require_identifier, require_tick
from src.core.reference_controller import ReferenceDecision
from src.core.contextual_controller import training_context
from scripts.ticket_controller_streams import TicketStreamRecipe, LiveTrainingRecipe


def original_bank():
    from scripts.recurring_ticket_workflow_example import prepare_training
    from scripts.ticket_workflow_rehearsal import trajectory_batches

    spec, training = prepare_training()
    trajectories = trajectory_batches(spec, training)
    episode = trajectories[0].episode
    if episode is None or episode.role != "train" or len(episode.transitions) != 3:
        raise ValueError("original ticket replay requires one complete three-row TRAIN episode")
    return (episode,), dict(
        prepared_episodes=len(trajectories),
        prepared_rows=sum(len(b.episode.transitions) for b in trajectories),
        retained_episodes=1,
        retained_rows=3,
        scope="original trajectory compiler validates every TRAIN teacher transition; not native physics",
    )


class CatalogTicketControl:
    def __init__(
        self,
        trial,
        policy,
        *,
        training_context,
        base_snapshot,
        unsupported,
        batches,
        stream_recipe: TicketStreamRecipe | None = None,
        deferred_evaluation: bool = False,
        live_recipe: LiveTrainingRecipe | None = None,
    ):
        if (trial is None) == (policy is None) or unsupported:
            raise ValueError("catalog execution requires one isolated private trial or application")
        if live_recipe is not None:
            if (
                type(live_recipe) is not LiveTrainingRecipe
                or trial is not None
                or stream_recipe is not None
                or deferred_evaluation is not False
                or training_context != "new_contexts"
            ):
                raise ValueError("catalog live recipe requires its exact frozen application")
            live_recipe.__post_init__()
            if len(live_recipe.recipe.contexts) != batches:
                raise ValueError("catalog live recipe requires its original full packet window")
            stream_recipe = live_recipe.recipe
        elif stream_recipe is not None:
            if (
                trial is not None
                or type(stream_recipe) is not TicketStreamRecipe
                or deferred_evaluation is not True
                or training_context != "new_contexts"
            ):
                raise ValueError("catalog stream requires an exact deferred frozen application")
            stream_recipe.__post_init__()
            if len(stream_recipe.contexts) > batches:
                raise ValueError("catalog stream exceeds its original packet allowance")
        elif deferred_evaluation is not False:
            raise ValueError("deferred catalog application requires an explicit TRAIN recipe")
        record = trial if trial is not None else policy
        if trial is not None:
            if type(trial) is not CatalogCounterfactualRequest or base_snapshot is None:
                raise ValueError("catalog trial requires its exact request and detached base")
            if trial.role != "controller_train" or trial.context != {
                "new_contexts": "mistakes",
                "old_return": "clean",
            }.get(training_context):
                raise ValueError("catalog trial requires original controller-TRAIN context")
            require_tick(trial.started_at, "catalog trial start")
        elif type(policy) is not FittedCatalogController or base_snapshot is not None:
            raise ValueError("catalog application requires its exact frozen fit without rebasing")
        else:
            policy.__post_init__()
        if type(record.catalog) is not ContextualArmCatalog:
            raise ValueError("ticket catalog requires an exact finite catalog")
        record.catalog.__post_init__()
        require_identifier(record.state_digest, "catalog source digest")
        if record.learner_version != "qualified-96" or training_context not in (
            "new_contexts",
            "old_return",
        ):
            raise ValueError("ticket catalog requires the original qualified source/context")
        if trial is not None:
            record.catalog.envelope.select(trial.action_id)
        envelope = record.catalog.envelope
        if (
            envelope.max_decisions != batches + 1
            or envelope.max_wake_updates != batches
            or record.catalog.envelope.actions[2].replay_examples != 3
            or not 3 <= envelope.max_replay_examples <= 3 * batches
            or envelope.max_replay_examples % 3
        ):
            raise ValueError(
                "catalog envelope must preserve original horizon/wake bound and whole bank exposure"
            )
        self.record, self.catalog = record, record.catalog
        self._pin = deepcopy(record)
        self._catalog_owner = self.catalog
        self.trial = trial is not None
        self._batches = batches
        self.stream_recipe = stream_recipe
        self._stream_pin = deepcopy(stream_recipe)
        self.live_recipe = self._live_recipe_owner = live_recipe
        self._live_recipe_pin = deepcopy(live_recipe)
        self._packets: tuple[AttributionBatch, ...] | None = None
        self._packet_pin: tuple[AttributionBatch, ...] | None = None
        self._schedule_packets: tuple[AttributionBatch, ...] | None = None
        self._prepared_owner: tuple[AttributionBatch, ...] | None = None
        self._scheduled_owner: tuple[AttributionBatch, ...] | None = None
        self.replay: CategoricalReplayAction | None = None
        self.replay_budget: ToyBudgetSession | None = None
        self._episodes = None
        self._wake = None
        self._wake_limits = None
        self.report: dict[str, Any] = dict(
            phase="trial" if self.trial else "private_application",
            record=asdict(record),
            preparation=dict(attempts=0, returned=False),
            replay=None,
            replay_budget=None,
            source_binding_verified=False,
            promotions=0,
        )
        if live_recipe is not None:
            self.report["live_recipe"] = asdict(live_recipe)

    @property
    def max_replays(self):
        return self.catalog.envelope.max_replay_examples // 3

    @property
    def admissions(self):
        return self._batches + self.max_replays

    def prepare_stream(self, batches):
        """Own declared TRAIN packets before any model or bank construction."""
        from src.app.coupled_attribution import _validated_batches

        if self.stream_recipe is None:
            return batches
        self.require_pin()
        packets, _ = _validated_batches(batches, (), training_only=True)
        if (
            self._packets is not None
            or len(packets) != len(self.stream_recipe.contexts)
            or any(b.features.shape != (12, 7) or b.num_classes != 6 for b in packets)
        ):
            raise ValueError("catalog stream requires its original twelve-row TRAIN packets")
        self._packets, self._packet_pin = packets, deepcopy(packets)
        self._prepared_owner = packets
        return packets

    def require_stream_packets(self, packets):
        if self.stream_recipe is not None:
            self.before_work()
            if packets is not self._packets:
                raise ValueError("original prepared catalog packets were replaced")

    def bind_schedule(self, packets, label_delay):
        if self.stream_recipe is None:
            if label_delay != 1:
                raise ValueError("plain catalog trials require their original delay1")
            return len(packets) + label_delay
        self.before_work()
        if (
            self._schedule_packets is not None
            or self._packets is None
            or label_delay != self.stream_recipe.label_delay
            or not _same_value(packets, self._packet_pin)
        ):
            raise ValueError("catalog schedule lost its original prepared packet/delay binding")
        self._schedule_packets = packets
        self._scheduled_owner = packets
        natural = len(packets) + label_delay
        # Why: a delayed final label cannot renew the already frozen decision cap.
        horizon = min(natural, self.catalog.envelope.max_decisions)
        self.report["stream"] = dict(
            natural_horizon=natural,
            observed_horizon=horizon,
            decision_cap=self.catalog.envelope.max_decisions,
            horizon_limited=horizon < natural,
        )
        return horizon

    def require_schedule_packets(self, packets):
        if self.stream_recipe is not None:
            self.before_work()
            if packets is not self._schedule_packets:
                raise ValueError("original scheduled catalog packets were replaced")

    def require_pin(self):
        if self.record.catalog is not self._catalog_owner or not _same_value(
            self.record, self._pin
        ):
            raise ValueError("original catalog trial/application record changed or was replaced")
        self.catalog.__post_init__()
        if not _same_value(self.stream_recipe, self._stream_pin):
            raise ValueError("original catalog TRAIN recipe changed")
        if self.live_recipe is not self._live_recipe_owner or (
            self.live_recipe is not None
            and (
                self.live_recipe.recipe is not self.stream_recipe
                or not _same_value(self.live_recipe, self._live_recipe_pin)
            )
        ):
            raise ValueError("original catalog live recipe changed or was replaced")
        # Workers check immutable evidence here without mutating either budget clock.
        if (
            self.stream_recipe is not None
            and self._packets is not None
            and (
                self._packets is not self._prepared_owner
                or self._schedule_packets is not self._scheduled_owner
                or not _same_value(self._packets, self._packet_pin)
                or self._schedule_packets is not None
                and not _same_value(self._schedule_packets, self._packet_pin)
            )
        ):
            raise ValueError("original catalog TRAIN packets changed")

    def require_source(self, initial, base_snapshot, *, digest, same_state):
        self.require_pin()
        if (
            initial.completed_updates != 96
            or initial.training_examples != 576
            or digest(initial) != self.record.state_digest
            or base_snapshot is not None
            and not same_state(base_snapshot, initial)
        ):
            raise ValueError("catalog requires its original complete qualified source")
        self.report["source_binding_verified"] = True

    def prepare(self, wake, sampler):
        self.require_pin()
        if not self.report["source_binding_verified"] or self.replay_budget is not None:
            raise ValueError(
                "catalog bank preparation requires one verified source and fresh owner"
            )
        self.replay_budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=0,
                max_replay_examples=self.catalog.envelope.max_replay_examples,
                max_wall_seconds=120,
                max_process_rss_bytes=512 * 1024**2,
            ),
            monotonic,
        )
        self.replay_budget.attach_memory(sampler)
        self._wake = wake
        self._wake_limits = wake.budget
        self._episodes = None
        self.report["preparation"]["attempts"] += 1
        self._episodes, prepared = original_bank()
        self.report["preparation"].update(prepared, returned=True)
        self.replay_budget.before_final()

    def before_work(self):
        if getattr(self, "_terminal_retired", False):
            raise ValueError("catalog chooser is retired after its terminal live window")
        self.require_pin()
        if self.replay_budget is not None:
            self.replay_budget.before_final()

    def bind(self, runtime, source, gate):
        self.before_work()
        if (
            self.replay is not None
            or self.replay_budget is None
            or self._episodes is None
            or self._wake is None
            or runtime._budget is not self._wake
            or self._wake.budget is not self._wake_limits
            or gate.snapshot().limits.max_admitted_updates != self.admissions
        ):
            raise ValueError(
                "catalog requires its original prepared owners and separate admissions"
            )
        self.replay = CategoricalReplayAction(
            runtime,
            source,
            self.replay_budget,
            self._episodes,
            ReplayRetentionBudget(1, 2048),
            states_equal=_same_value,
            definition=TORCH_CE_ID,
        )
        shared = ResourceSharedRuntime(
            runtime, gate, action_envelope=self.catalog.envelope, replay_action=self.replay
        )
        self.capture()
        return shared

    def choose(self, observation, progress):
        self.before_work()
        if self.replay_budget is None or self.replay is None:
            raise ValueError("catalog choice requires its original bound replay executor")
        remaining = min(
            self.replay_budget.remaining_replay_examples,
            self.catalog.envelope.max_replay_examples - progress.replay_examples_charged,
        )
        if not self.trial:
            return choose_catalog_contextual(
                self.record,
                observation,
                envelope=self.catalog.envelope,
                remaining_replay_examples=remaining,
            )
        context = training_context(observation)
        if observation.actor_version != self.record.learner_version:
            raise ValueError("catalog trial actor version became stale")
        if context is None or observation.ready_updates == 0 or observation.remaining_wake == 0:
            return ReferenceDecision("wait", "no_arrived_context_ready_work_or_credit")
        action = self.catalog.envelope.select(self.record.action_id)
        if action.replay_examples > remaining:
            return ReferenceDecision("wait", "insufficient_declared_replay_credit")
        return ReferenceDecision(action.action_id, "catalog_private_fixed_trial")

    def require_candidate(self):
        """Reuse the original success guard before predictions/readouts, while idle."""
        self.before_work()
        replay = self.replay
        if replay is None:
            raise ValueError("catalog readout requires its original bound replay owner")
        replay._runtime.with_candidate_snapshot(lambda view: replay.before_wake_admission())

    def capture(self):
        if self.replay is not None:
            self.report["replay"] = self.replay.report()
        if self.replay_budget is not None:
            b = self.replay_budget
            self.report["replay_budget"] = dict(
                limits=asdict(b.budget),
                updates_completed=b.updates_completed,
                replay_examples_completed=b.replay_examples_completed,
                started_at=b.started_at,
                last_clock=b.last_clock,
            )
        return self.report

    def require_history(self, candidate, wake, receipts=None):
        self.before_work()
        wake_rows = 6 * wake.updates_completed
        if self.stream_recipe is not None:
            if self._packets is None or receipts is None or len(receipts) != wake.updates_completed:
                raise ValueError("catalog stream requires its original completed FIFO receipts")
            wake_rows = 0
            for index, receipt in enumerate(receipts):
                if (
                    receipt.sample_id != f"batch-{index}"
                    or receipt.event_id != f"label-{index}"
                    or receipt.update_number != index + 1
                    or receipt.observed_at != index
                    or receipt.arrived_at != index + self.stream_recipe.label_delay
                    or receipt.applied_at < receipt.arrived_at
                    or receipt.applied_at >= self.report["stream"]["observed_horizon"]
                ):
                    raise ValueError("catalog stream history lost its original causal FIFO prefix")
                wake_rows += len(self._packets[index].features)
        replay = self.capture()["replay"]
        if (
            replay is None
            or replay["stopped"]
            or replay["publication_unknown"]
            or replay["native_attempts"] != replay["native_completions"]
            or replay["native_completions"] != replay["committed_events"]
            or candidate.completed_updates
            != 96 + wake.updates_completed + replay["committed_events"]
            or candidate.training_examples != 576 + wake_rows + replay["replay_examples_charged"]
        ):
            raise ValueError("catalog native history differs from original wake plus paid replay")
        self.report["native_history"] = dict(
            initial_updates=96,
            wake_updates=wake.updates_completed,
            replay_updates=replay["committed_events"],
            completed_updates=candidate.completed_updates,
            training_rows=candidate.training_examples,
            checkpoint_axis="wake receipts; native clock includes replay",
        )
        if self.stream_recipe is not None:
            self.report["native_history"]["wake_training_rows"] = wake_rows


def case_work(result):
    """Admit complete current-case cost records before any utility read."""
    if result.get("definition") == "live_catalog_bp_ticket_publication_v1":
        raise ValueError("terminal catalog publication requires separate guard and serving costs")
    if result.get("deferred_evaluation") is True:
        raise ValueError(
            "deferred catalog TRAIN completion has no task utility before outer readout"
        )
    return _native_case_work(result)


def deferred_case_work(result):
    """Admit TRAIN costs only; task measures still require the outer frozen readout."""
    if (
        result.get("deferred_evaluation") is not True
        or result.get("definition") != "deferred_catalog_stream_v1"
        or result.get("episodes") != []
        or any(
            k in result
            for k in (
                "candidate_episodes",
                "candidate_readout_work",
                "reference_outcomes",
                "guards",
                "promotion_receipt",
            )
        )
        or any(
            result.get(k) != 0
            for k in (
                "catalog_promotions",
                "deferred_promotions",
                "final_examples",
                "reserved_probe_updates",
                "external_actions",
            )
        )
        or result.get("deferred_actor_unchanged") is not True
        or any(
            result.get("serving_work", {}).get(k) != 0
            for k in ("attempts", "returns", "native_forwards")
        )
    ):
        raise ValueError("deferred catalog costs require an unevaluated private TRAIN capture")
    return _native_case_work(result, deferred=True)


def _native_case_work(result, *, deferred=False):
    control = result.get("catalog_control", {})
    replay = control.get("replay")
    work = result.get("training_work", {})
    if (
        result.get("complete") is not True
        or result.get("source_unchanged") is not True
        or result.get("catalog_actor_unchanged") is not True
        or result.get("numpy_rng_unchanged") is not True
        or result.get("torch_rng_unchanged") is not True
        or control.get("source_binding_verified") is not True
        or not replay
        or replay["stopped"]
        or replay["publication_unknown"]
        or replay["native_attempts"] != replay["native_completions"]
        or replay["native_completions"] != replay["committed_events"]
        or work.get("attempted_updates") != work.get("returned_updates")
        or work.get("unreturned_work_unknown") is not False
    ):
        raise ValueError("incomplete/unknown catalog case has no utility")
    history = control["native_history"]
    if (
        history["completed_updates"] != 96 + work["returned_updates"]
        or history["training_rows"] != 576 + work["attempted_rows"]
        or result["replay_examples"] != replay["replay_examples_charged"]
    ):
        raise ValueError("catalog cost record differs from its completed native history")
    prediction = result["controller_work"]
    serving = result["serving_work"]
    candidate = (
        dict(native_forward_attempts=0, native_forward_returns=0)
        if deferred
        else result["candidate_readout_work"]["work"]
    )
    if (
        prediction["prediction_attempts"] != prediction["prediction_returns"]
        or serving["attempts"] != serving["returns"]
        or candidate["native_forward_attempts"] != candidate["native_forward_returns"]
    ):
        raise ValueError("unknown catalog prediction work has no utility")
    calls = (
        prediction["prediction_attempts"]
        + serving["native_forwards"]
        + candidate["native_forward_attempts"]
    )
    rows = (
        prediction["prediction_rows"]
        + serving["native_forwards"]
        + candidate["native_forward_attempts"]
    )
    return CatalogTrialWork(
        work["attempted_updates"],
        work["returned_updates"],
        work["attempted_rows"],
        work["attempted_rows"],
        replay["replay_examples_charged"],
        calls,
        rows,
    )
