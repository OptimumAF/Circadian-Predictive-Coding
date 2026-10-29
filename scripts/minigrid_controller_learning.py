"""Compose existing MiniGrid TRAIN streams, four-trial fitter and game readouts.

Inputs are the original typed A3/A4 TRAIN pairs and a caller-owned readout budget
started before preparation. Outputs retain setup, discarded-trial/application
costs and real game measures. No new learner, chooser, final/probe access, source
reader or scientific allowance belongs here.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from math import isfinite
from time import monotonic
from typing import Any, Callable

import numpy as np

from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from src.adapters.minigrid_policy_evaluation import (
    evaluate_context_linear_stream_checkpoint,
    MiniGridEvaluationFailed,
    _require_phase,
)
from src.adapters.minigrid_train_catalogue import _validate_sources, compose_arrived_cue_catalogue
from src.app.actor_shadow import StableActor
from src.app.continual_confirmation_json import state_digest
from src.app.counterfactual_controller import CounterfactualController, CounterfactualReturn
from src.app.coupled_attribution import AttributionBatch, AttributionStream, _same_value
from src.app.probe_return_phase import budget_binding, paid_report_guard
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import VersionedState
from src.core.contextual_controller import training_context
from src.core.controller_objective import ControllerMeasures, ControllerObjective
from src.core.demonstration_episode import episode_arrays
from src.core.experience import LogicalClock
from src.core.reference_controller import (
    ArrivedTrainingFeedback,
    ReferenceObservation,
    ReferencePolicy,
)

BASE_VERSION = "a3-base-2"


def describe_controller_learning(*, retain_arrived=False):
    if type(retain_arrived) is not bool:
        raise ValueError("retain_arrived must be an exact bool")
    result = dict(
        definition="minigrid_current_only_counterfactual_learning_v1",
        source="typed original A3/A4 blue/red TRAIN pairs, public296/K7, nine rows each",
        warmup="two delayed A3 fits/18 rows; full native history retained",
        trials="clean wait/wake then mistakes wait/wake, same complete native base",
        clean="A3 blue/red then A3 blue/red",
        mistakes="A4 blue/red then A3 blue/red",
        window="four packets/five opportunities/delay1/current-only; four new update credits",
        objective=asdict(ControllerObjective()),
        measures="new gain and old drop at after-task; return/recovery separate; whole-trial time/cumulative sampled RSS proxies",
        readout="four known TRAIN A3/A4 blue/red games, seed17/max16/K7; actual native counts2..6",
        application="existing source-bound fitted chooser on A4/A3, no further fitting",
        scope="controller_train development then known-TRAIN application; no final/probe/generalization/CPC/matchedcompute claim",
    )
    if retain_arrived:
        result.update(
            definition="minigrid_arrived_bank_counterfactual_learning_v1",
            window="four packets/five opportunities/delay1/joint18; four new update credits",
            readout="existing joint reference geometry, full native2..6/18..90; same four TRAIN games",
            bank="existing one-prior-packet-per-cue composer, initialized by actual warmup; independent bank per trial/application",
            feedback="outer context nine current rows; window ready feedback scores actual joint18 rows",
            replay="nine prior TRAIN rows per selected fit; native exposures, not separate consolidation events",
        )
    return result


def _packets(episodes, indices, prefix):
    return tuple(
        replace(episodes[index], episode_id=f"{prefix}-{i}") for i, index in enumerate(indices)
    )


def _batches(episodes, indices, prefix):
    batches = []
    for packet in _packets(episodes, indices, prefix):
        x, y = episode_arrays(packet)
        batches.append(
            AttributionBatch(
                "train",
                "A",
                tuple(f"{packet.episode_id}:step:{t.step}" for t in packet.transitions),
                x,
                y,
                packet,
                7,
            )
        )
    return tuple(batches)


def _bank_batches(episodes, indices, prefix, *, label_delay=1):
    """Reuse the six-arrival recipe; its warm templates are prepared, not fitted."""
    plan = compose_arrived_cue_catalogue(
        _packets(episodes, (0, 1), "warmup") + _packets(episodes, indices, prefix)
    )
    planned: dict[str, Any] = dict(
        wakes=[asdict(w) for w in plan.wakes],
        prepared_plan_rows=plan.fit_row_exposures,
        unused_warm_template_rows=18,
        future_current_rows=36,
        future_replay_rows=36,
        future_native_rows=72,
        input_array_bytes=plan.input_array_bytes,
        peak_joint_template_array_bytes=plan.peak_joint_template_array_bytes,
        peak_bank_rows=plan.peak_bank_rows,
        peak_bank_observation_target_float64_equivalent_bytes=plan.peak_bank_observation_target_float64_equivalent_bytes,
        initialization="actual completed warmup transport warmup-0/1, native2/18; no second fit",
        clock_scope="composer ticks0..5 span warmup then window; window local ticks subtract2, warmup arrivals stay in their own epoch",
        window_wakes=[
            dict(
                observed_at=w.observed_at - 2,
                arrived_at=w.arrived_at - 2,
                current_episode_id=w.current_episode_id,
                retained=[
                    dict(
                        episode_id=identity,
                        epoch="warmup" if arrival <= 2 else "window",
                        arrived_at=arrival if arrival <= 2 else arrival - 2,
                    )
                    for identity, arrival in zip(
                        w.retained_episode_ids, w.retained_arrival_ticks, strict=True
                    )
                ],
            )
            for w in plan.wakes[2:]
        ],
        accounting_scope="planned arrays include unused warm templates; actual prior rows are joint native-fit exposures, not consolidation budget.replay_examples",
    )
    if label_delay != 1:
        # Composer clocks remain original. These labels are consumed only when
        # the current packet arrives; earlier window packets arrive sooner.
        planned["window_label_delay"] = label_delay
        for wake in planned["window_wakes"]:
            wake["arrived_at"] += label_delay - 1
            for prior in wake["retained"]:
                if prior["epoch"] == "window":
                    prior["arrived_at"] += label_delay - 1
                if prior["epoch"] == "window" and prior["arrived_at"] > wake["arrived_at"]:
                    raise ValueError("prior labels have not arrived at the joint fit")
    return plan.training[2:], planned


def _goals(games, *, nuisance=False):
    _require_phase(games, **(dict(nuisance=True) if nuisance else {}))
    for row in games["episodes"]:
        env = row["environment"]
        if (
            type(row["success"]) is not bool
            or type(row["return"]) not in (int, float)
            or not isfinite(row["return"])
        ):
            raise ValueError("complete native goal/return values required")
        transitions = env.get("transitions")
        if type(transitions) is not list or len(transitions) != env["environment_steps"]:
            raise ValueError("all native action/step records required for trial utility")
        if any(
            type(t.get("action")) is not int
            or not 0 <= t["action"] < 7
            or t.get("applied") is not True
            for t in transitions
        ):
            raise ValueError("unknown/invalid native action work has no controller utility")
    rows = games["episodes"]
    return dict(
        old_A3=sum(r["success"] for r in rows[:2]),
        **{("shifted_A3" if nuisance else "new_A4"): sum(r["success"] for r in rows[2:])},
    )


class _Study:
    """One bounded composition of existing owners, with no independent authority."""

    def __init__(
        self,
        episodes,
        readout_budget,
        check_boundary,
        report,
        retain_arrived=False,
        *,
        nuisance=False,
    ):
        if type(readout_budget) is not ToyBudgetSession or (
            check_boundary is not None and not callable(check_boundary)
        ):
            raise ValueError("original readout budget and optional boundary required")
        if type(nuisance) is not bool or (nuisance and retain_arrived is not True):
            raise ValueError("nuisance study requires the exact arrived TRAIN bank")
        self.nuisance = self.nuisance_pin = nuisance
        self.budget, self.outer = readout_budget, check_boundary
        self.retain_arrived = self.retention_pin = retain_arrived
        self.binding, self.limits = budget_binding(readout_budget), asdict(readout_budget.budget)
        if not _same_value(
            self.limits,
            asdict(
                ToyExecutionBudget(
                    max_training_updates=0,
                    max_replay_examples=0,
                    max_wall_seconds=150,
                    max_process_rss_bytes=512 * 1024**2,
                )
            ),
        ):
            raise ValueError("original zero-update/replay150s512MiB readout owner required")
        self.episodes = episodes
        self.report = report
        self.witness, _ = _validate_sources(episodes, 6)
        expected = tuple(
            f"minigrid-source-{tag}-{cue}"
            for tag in (("A", "A3-redwall-train") if nuisance else ("A", "A4-train"))
            for cue in ("blue", "red")
        )
        if (
            len(episodes) != 4
            or tuple(e.episode_id for e in episodes) != expected
            or any(e.context != "A" or len(e.transitions) != 9 for e in episodes)
        ):
            raise ValueError("original ordered A3/A4 TRAIN pairs required")
        self.paid: list[tuple[Any, Any, Any]] = []
        self.links: list[tuple[Any, Any, Any]] = []
        self.link(report["contexts"], report, "contexts")
        self.link(report["trials"], report, "trials")
        self.budgets: list[tuple[Any, Any, Any, Any]] = []
        self.cold: Any = None
        self.cold_initial: Any = None
        self.warm: Any = None
        self.base: Any = None
        self.game_attempts = self.game_returns = 0
        self.pending_game = None
        self.require()
        self.cold = ContextLinearLearner()
        self.cold_initial = self.cold.snapshot_state()

    def require(self):
        self.budget.before_final()
        if self.outer is not None:
            self.outer()
        if (
            self.nuisance is not self.nuisance_pin
            or self.nuisance
            is not (self.episodes[2].episode_id == "minigrid-source-A3-redwall-train-blue")
            or type(self.retain_arrived) is not bool
            or self.retain_arrived is not self.retention_pin
            or budget_binding(self.budget) != self.binding
            or not _same_value(asdict(self.budget.budget), self.limits)
            or not _same_value(
                (self.budget.updates_completed, self.budget.replay_examples_completed), (0, 0)
            )
        ):
            raise ValueError("original controller study readout budget changed")
        self.budget.before_final()
        self.witness.check(self.episodes)
        if len(self.report["trials"]) > 4 or len(self.report["contexts"]) > 2:
            raise ValueError("finite controller trial/context bound exceeded")
        for owner, pin, limits, seal in self.budgets:
            if budget_binding(owner) != pin or not _same_value(asdict(owner.budget), limits):
                raise ValueError("original TRAIN budget changed")
            if seal is not None and not _same_value(
                (owner.updates_completed, owner.replay_examples_completed, owner.last_clock), seal
            ):
                raise ValueError("preceding TRAIN charges/clock changed")
        for value, check, _ in self.paid:
            check()
        for parent, key, value in self.links:
            present = key in parent if isinstance(parent, dict) else 0 <= key < len(parent)
            if not present:
                raise ValueError("original bounded study evidence path missing")
            if parent[key] is not value:
                raise ValueError("original study evidence was replaced")
        if self.cold is not None and not _same_value(self.cold.snapshot_state(), self.cold_initial):
            raise ValueError("cold readout source changed")
        if self.warm is not None and not _same_value(self.warm.snapshot_state(), self.base):
            raise ValueError("original complete warm source changed")

    def link(self, record, parent, key):
        if len(self.links) >= 34:
            raise ValueError("finite study owner bound exceeded")
        self.links.append((parent, key, record))

    def seal(self, record, parent=None, key=None):
        check, detached = paid_report_guard(record)
        if len(self.paid) >= 26:
            raise ValueError("finite study evidence bound exceeded")
        if parent is not None:
            self.link(record, parent, key)
        self.paid.append((record, check, detached))

    def preserve_completed_prefix(self):
        """Return known as-read successes after refusal, without clearing failure."""
        detached = {id(value): saved for value, _, saved in self.paid}
        for parent, key, value in self.links:
            saved = detached.get(id(value), value)
            if isinstance(parent, list) and key == len(parent):
                parent.append(saved)
            else:
                parent[key] = saved

    def window(
        self,
        source,
        indices,
        name,
        row,
        *,
        action=None,
        controller=None,
        policy=None,
        label_delay=1,
        acknowledge_correct=False,
    ):
        self.require()
        if type(label_delay) is not int or label_delay not in (1, 2):
            raise ValueError("declared window label delay must be1or2")
        if policy is not None:
            if type(policy) is not ReferencePolicy or action is not None or controller is not None:
                raise ValueError("one exact fixed validation policy required")
            policy.__post_init__()
        limit = len(indices)
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=limit, max_wall_seconds=90, max_process_rss_bytes=512 * 1024**2
            ),
            monotonic,
        )
        assert self.budget.process_rss_sampler is not None
        budget.attach_memory(self.budget.process_rss_sampler)
        at = len(self.budgets)
        self.budgets.append((budget, budget_binding(budget), asdict(budget.budget), None))
        base = source.snapshot_state()
        actor = StableActor(source, version=BASE_VERSION)
        original_actor = actor.snapshot_state()
        states = {}
        stream = None
        bank = self.retain_arrived and name != "warmup"
        check_planned: Callable[[], None] | None = None
        saved_planned = None
        catalogue_owner: dict[str, Any] | None = None
        row.update(
            status="running",
            window=None,
            source_episode_ids=[self.episodes[i].episode_id for i in indices],
        )

        def observe(n, state, batch):
            self.require()
            if check_planned is not None:
                check_planned()
            if (
                state.completed_updates != base.completed_updates + n
                or state.training_examples != base.training_examples + (18 if bank else 9) * n
                or state.failed_work is not None
            ):
                raise ValueError("native warm history differs from relative paid prefix")
            states[n] = state

        try:
            if bank:
                if (
                    limit != 4
                    or not _same_value(base, self.base)
                    or (base.completed_updates, base.training_examples) != (2, 18)
                ):
                    raise ValueError(
                        "bank initialization requires the original complete warm2/18 source"
                    )
                catalogue_owner = row["catalogue"] = dict(
                    preparation_attempts=1,
                    preparation_returns=0,
                    unreturned_preparation_unknown=True,
                    planned=None,
                )
                batches, planned = (
                    _bank_batches(self.episodes, indices, name)
                    if label_delay == 1
                    else _bank_batches(self.episodes, indices, name, label_delay=label_delay)
                )
                catalogue_owner.update(
                    preparation_returns=1, unreturned_preparation_unknown=False, planned=planned
                )
                guard_planned, saved_planned = paid_report_guard(planned)

                def check_plan():
                    if (
                        row.get("catalogue") is not catalogue_owner
                        or catalogue_owner.get("planned") is not planned
                        or not _same_value(
                            (
                                catalogue_owner.get("preparation_attempts"),
                                catalogue_owner.get("preparation_returns"),
                                catalogue_owner.get("unreturned_preparation_unknown"),
                            ),
                            (1, 1, False),
                        )
                    ):
                        raise ValueError("original prepared bank metadata was replaced")
                    guard_planned()

                check_planned = check_plan
            else:
                batches = _batches(self.episodes, indices, name)
            stream = AttributionStream(
                source,
                batches,
                (),
                training_only=True,
                checkpoint_updates=tuple(range(limit + 1)),
                label_delay=label_delay,
                budget=budget,
                actor=actor,
                observe_checkpoint=observe,
            )
            row["window"] = (
                stream.run_contextual_window(
                    controller, **(dict(acknowledge_correct=True) if acknowledge_correct else {})
                )
                if controller is not None
                else stream.run_reference_window(
                    policy
                    if policy is not None
                    else ReferencePolicy(
                        "no_learning" if action == "wait" else "periodic", period=1
                    ),
                    ready_feedback=name != "warmup",
                    acknowledge_correct=acknowledge_correct,
                )
            )
            result = row["window"]
            if check_planned is not None:
                check_planned()
            receipts = stream.applied_updates
            if (
                result["status"] != "complete"
                or result["window_complete"] is not True
                or not _same_value([asdict(r) for r in receipts], result["applied_updates"])
                or len(receipts) != budget.updates_completed
                or list(states) != list(range(len(receipts) + 1))
            ):
                raise ValueError("complete actual window/receipt/checkpoint prefix required")
            if (
                result["training_examples"] != len(receipts) * (18 if bank else 9)
                or not _same_value(source.snapshot_state(), base)
                or not _same_value(actor.snapshot_state(), original_actor)
            ):
                raise ValueError("window changed source/actor or original row charges")
            row.update(
                status="complete",
                source_unchanged=True,
                actor_unchanged=True,
                checkpoints=[
                    dict(
                        relative_updates=n,
                        native_updates=s.completed_updates,
                        native_rows=s.training_examples,
                    )
                    for n, s in states.items()
                ],
            )
            return states
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}"[:512])
            raise
        finally:
            if row["window"] is None and stream is not None:
                row["window"] = stream.reference_report
            row["budget"] = dict(
                limits=asdict(budget.budget),
                updates=budget.updates_completed,
                replay=budget.replay_examples_completed,
                started_at=budget.started_at,
                last_clock=budget.last_clock,
            )
            owner, pin, limits, _ = self.budgets[at]
            self.budgets[at] = (
                owner,
                pin,
                limits,
                (owner.updates_completed, owner.replay_examples_completed, owner.last_clock),
            )
            if bank and catalogue_owner is not None:
                # Why: only a verified checkpoint proves a joint-row prefix.
                # A charged but unverified/unknown call must remain separate.
                n = max(states, default=0)
                catalogue_owner["committed"] = dict(
                    verified_updates=n,
                    current_rows=9 * n,
                    replay_rows=9 * n,
                    native_rows=18 * n,
                    unverified_committed_updates=max(0, budget.updates_completed - n),
                )
                try:
                    if check_planned is not None:
                        check_planned()
                except Exception as error:
                    catalogue_owner.update(
                        capture_error=f"{type(error).__name__}: {error}"[:512],
                        planned=saved_planned,
                        preparation_attempts=1,
                        preparation_returns=1,
                        unreturned_preparation_unknown=False,
                    )
                    row.update(status="failed", catalogue=catalogue_owner)
                    raise

    def game(self, state, *, cue_prefix=None, applied_cues=None):
        self.require()
        self.game_attempts += 1
        try:
            result = evaluate_context_linear_stream_checkpoint(
                self.cold,
                state,
                budget=self.budget,
                check_boundary=self.require,
                reference_window=True,
                **(dict(reference_catalogue=True) if self.retain_arrived else {}),
                **(dict(reference_catalogue_cues=cue_prefix) if cue_prefix is not None else {}),
                **(dict(reference_applied_cues=applied_cues) if applied_cues is not None else {}),
                **(dict(reference_nuisance=True) if self.nuisance else {}),
            )
        except MiniGridEvaluationFailed as error:
            self.pending_game = error.report
            raise
        self.game_returns += 1
        self.pending_game = result
        self.require()
        _goals(result, nuisance=self.nuisance)
        self.pending_game = None
        return result

    def phases(self, states, window, baseline, baseline_games, row):
        cache: list[tuple[Any, Any, str | int]] = [
            (deepcopy(baseline), baseline_games, "shared_base")
        ]
        phases: list[dict[str, Any]] = []
        row["phases"] = phases
        self.link(phases, row, "phases")
        for tick in (2, 4):
            self.require()
            n = window["decisions"][tick]["completed_updates"]
            state = states[n]
            found = next(
                (
                    (games, origin)
                    for expected, games, origin in cache
                    if _same_value(expected, state)
                ),
                None,
            )
            games, reused = found if found is not None else (self.game(state), None)
            _goals(games)
            if found is None:
                if len(cache) >= 3:
                    raise ValueError("three readout states already retained")
                cache.append((deepcopy(state), games, tick))
            phase = dict(
                tick=tick,
                relative_updates=n,
                native_updates=state.completed_updates,
                native_rows=state.training_examples,
                games=games,
                reused_from=reused,
            )
            phases.append(phase)
            self.seal(phase, phases, len(phases) - 1)
        before, task, returned = (
            _goals(baseline_games),
            _goals(phases[0]["games"]),
            _goals(phases[1]["games"]),
        )
        row["outcomes"] = dict(
            before=before,
            after_task=task,
            after_return=returned,
            recovery=[
                None
                if not (
                    baseline_games["episodes"][i]["success"]
                    and not phases[0]["games"]["episodes"][i]["success"]
                )
                else phases[1]["games"]["episodes"][i]["success"]
                for i in range(2)
            ],
        )

    def context(self, context, index, clock, row):
        self.require()
        before = self.warm.snapshot_state()
        row["fork_attempts"] = 1
        reader = self.warm.fork()
        reader.restore_state(deepcopy(before))
        row["fork_completed"] = 1
        x, y = episode_arrays(self.episodes[index])
        row.update(predict_attempts=1, rows_attempted=len(x))
        predictions = reader.predict(x)
        row.update(predict_returned=1, rows_returned=len(x))
        if (
            predictions.shape != (9, 7)
            or not np.isfinite(predictions).all()
            or (predictions < 0).any()
            or not np.allclose(predictions.sum(axis=1), 1)
        ):
            raise ValueError("complete K7 context predictions required")
        self.require()
        if not _same_value(reader.snapshot_state(), before):
            raise ValueError("context readout changed native state")
        mistakes = int(np.sum(predictions.argmax(axis=1) != y[:, 0]))
        feedback = ArrivedTrainingFeedback(
            "train", self.episodes[index].episode_id, BASE_VERSION, 0, 0, mistakes, 9
        )
        observation = ReferenceObservation(clock.now(), 1, 1, 4, BASE_VERSION, feedback)
        if training_context(observation) != context:
            raise ValueError(
                "declared clean/mistakes TRAIN context unavailable; no search or relabeling"
            )
        row.update(
            status="complete",
            context=context,
            source_episode_id=self.episodes[index].episode_id,
            mistakes=mistakes,
            examples=9,
            scored_at=clock.now(),
            label_availability="supplied ordinary TRAIN available before fit on separate controller epoch; inner labels retain delay1",
            feedback=asdict(feedback),
        )
        return observation


def run_controller_learning(
    episodes,
    *,
    readout_budget,
    check_boundary=None,
    retain_arrived=False,
    validation_recipe=None,
    acknowledge_correct=False,
    share_readouts=False,
):
    """Run one declared development composition; caller freezes native work first."""
    report: dict[str, Any] = dict(
        protocol=describe_controller_learning(retain_arrived=retain_arrived),
        status="running",
        error=None,
        warmup={},
        baseline=None,
        contexts=[],
        trials=[],
        fitted=None,
        application=None,
    )
    if type(acknowledge_correct) is not bool or (acknowledge_correct and validation_recipe is None):
        raise ValueError(
            "correct acknowledgement requires exact bool and explicit validation recipe"
        )
    if type(share_readouts) is not bool or (share_readouts and validation_recipe is None):
        raise ValueError("readout sharing requires exact bool and explicit validation recipe")
    if validation_recipe is not None:
        from scripts.minigrid_controller_validation import describe_validation

        if retain_arrived is not True:
            raise ValueError("frozen validation requires the arrived TRAIN bank")
        report["protocol"]["validation"] = describe_validation(
            validation_recipe,
            **(dict(acknowledge_correct=True) if acknowledge_correct else {}),
            **(dict(share_readouts=True) if share_readouts else {}),
        )
        report["protocol"]["application"] = "replaced by five frozen schedule-validation windows"
        report["protocol"]["scope"] = (
            "controller_train fit then distinct known-TRAIN schedule validation; no unseen-task/final/probe/CPC/matchedcompute claim"
        )
        if validation_recipe == "nuisance_redwall_delay2":
            report["protocol"]["scope"] = (
                "original chooser fit then separate red-wall A3 TRAIN nuisance validation; no final/probe/CPC/generalization claim"
            )
        report["validation"] = None
    study = controller = None
    try:
        study = _Study(episodes, readout_budget, check_boundary, report, retain_arrived)
        started = monotonic()
        warm_states = study.window(study.cold, (0, 1), "warmup", report["warmup"], action="wake1")
        study.seal(report["warmup"], report, "warmup")
        study.base = deepcopy(warm_states[2])
        study.warm = study.cold.fork()
        study.warm.restore_state(study.base)
        if (study.base.completed_updates, study.base.training_examples) != (2, 18):
            raise ValueError("actual two-fit A3 base required")
        report["base"] = dict(
            state=asdict(study.base),
            actor_version=BASE_VERSION,
            warmup_seconds=monotonic() - started,
        )
        study.seal(report["base"], report, "base")
        report["baseline"] = study.game(study.base)
        study.seal(report["baseline"], report, "baseline")
        clock = LogicalClock()
        controller = CounterfactualController(
            lambda: VersionedState(BASE_VERSION, study.warm.snapshot_state()),
            lambda s: state_digest(asdict(s)),
            ControllerObjective(),
            clock,
        )
        for context, index, indices in (("clean", 0, (0, 1, 0, 1)), ("mistakes", 2, (2, 3, 0, 1))):
            context_row = dict(
                context=context,
                status="running",
                fork_attempts=0,
                fork_completed=0,
                predict_attempts=0,
                predict_returned=0,
                rows_attempted=0,
                rows_returned=0,
            )
            report["contexts"].append(context_row)
            observation = study.context(context, index, clock, context_row)
            study.seal(context_row, report["contexts"], len(report["contexts"]) - 1)
            for _ in range(2):
                observation = replace(observation, tick=clock.now())

                def operation(request, owned):
                    study.require()
                    if not _same_value(owned, study.base) or request.context != context:
                        raise ValueError("trial differs from original complete base/context")
                    row: dict[str, Any] = dict(
                        request=asdict(request), purpose="controller_train", phases=[]
                    )
                    report["trials"].append(row)
                    study.link(row, report["trials"], len(report["trials"]) - 1)
                    trial_start = monotonic()
                    try:
                        states = study.window(
                            study.warm,
                            indices,
                            f"trial-{context}-{request.action_id}",
                            row,
                            action=request.action_id,
                        )
                        study.seal(row["window"], row, "window")
                        study.phases(states, row["window"], study.base, report["baseline"], row)
                        study.require()
                        before, after = row["outcomes"]["before"], row["outcomes"]["after_task"]
                        measures = ControllerMeasures(
                            before["new_A4"],
                            after["new_A4"],
                            2,
                            before["old_A3"],
                            after["old_A3"],
                            2,
                            monotonic() - trial_start,
                            readout_budget.process_rss_sampler.peak_bytes,
                            0,
                            True,
                        )
                        row.update(measures=asdict(measures), status="complete")
                        study.seal(row)
                        clock.advance_to(request.started_at + 1)
                        return CounterfactualReturn(
                            request, request.started_at, clock.now(), measures
                        )
                    except BaseException as error:
                        row.update(status="failed", error=f"{type(error).__name__}: {error}"[:512])
                        raise

                controller.run(observation, operation)
        fitted = controller.freeze()
        report["fitted"] = asdict(fitted)
        study.seal(report["fitted"], report, "fitted")
        if validation_recipe is None:
            report["application"] = dict(purpose="known_TRAIN_application", phases=[])
            study.link(report["application"], report, "application")
            app_states = study.window(
                study.warm,
                (2, 3, 0, 1),
                "application",
                report["application"],
                controller=controller,
            )
            study.seal(report["application"]["window"], report["application"], "window")
            study.phases(
                app_states,
                report["application"]["window"],
                study.base,
                report["baseline"],
                report["application"],
            )
            study.seal(report["application"])
        else:
            from scripts.minigrid_controller_validation import (
                run_validation,
                prepare_nuisance_sources,
                NUISANCE_RECIPE,
            )

            validation = report["validation"] = dict(contexts=[], trials=[], candidates=[])
            study.link(validation, report, "validation")
            nuisance = validation_recipe == NUISANCE_RECIPE
            validation_episodes, boundary = (
                prepare_nuisance_sources(study, controller, validation)
                if nuisance
                else (episodes, study.require)
            )
            validation["setup_work"] = dict(attempts=1, returned=0, unreturned_work_unknown=True)
            validation_study = _Study(
                validation_episodes,
                readout_budget,
                boundary,
                validation,
                retain_arrived=True,
                **(dict(nuisance=True) if nuisance else {}),
            )
            validation["setup_work"].update(returned=1, unreturned_work_unknown=False)
            validation_study.warm = study.warm
            validation_study.base = deepcopy(study.base)
            run_validation(
                validation_study,
                controller,
                validation_recipe,
                baseline=None if nuisance else report["baseline"],
                **(dict(acknowledge_correct=True) if acknowledge_correct else {}),
                **(dict(share_readouts=True) if share_readouts else {}),
            )
            if validation["status"] != "complete":
                raise ValueError("validation capture is incomplete")
            study.seal(validation)
        study.require()
        report.update(status="complete", source_unchanged=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}"[:512])
        if validation_recipe is not None and report.get("validation") is not None:
            report["validation"].update(status="failed", error=report["error"])
        if isinstance(error, MiniGridEvaluationFailed):
            report["failed_game"] = error.report
    finally:
        if controller is not None:
            snap = controller.snapshot()
            report["controller"] = dict(
                attempts=snap["attempts"],
                completed=snap["completed"],
                discarded_trials=snap["discarded_trials"],
                stopped=snap["stopped"],
                clock=snap["clock"],
            )
        if study is not None:
            try:
                study.require()
            except Exception as error:
                report.update(
                    status="failed", capture_error=f"{type(error).__name__}: {error}"[:512]
                )
                study.preserve_completed_prefix()
            report["unadmitted_game"] = study.pending_game
            report["game_work"] = dict(
                readout_attempts=study.game_attempts,
                readout_returns=study.game_returns,
                unreturned_work_unknown=study.game_attempts != study.game_returns,
            )
            report["context_work"] = dict(
                fork_attempts=sum(r["fork_attempts"] for r in report["contexts"]),
                fork_completed=sum(r["fork_completed"] for r in report["contexts"]),
                predict_attempts=sum(r["predict_attempts"] for r in report["contexts"]),
                predict_returned=sum(r["predict_returned"] for r in report["contexts"]),
                rows_attempted=sum(r["rows_attempted"] for r in report["contexts"]),
                rows_returned=sum(r["rows_returned"] for r in report["contexts"]),
            )
            report["budgets"] = [
                dict(
                    limits=limits,
                    started_at=pin[2],
                    updates=owner.updates_completed,
                    replay=owner.replay_examples_completed,
                    last_clock=owner.last_clock,
                )
                for owner, pin, limits, _ in study.budgets
            ]
        report["readout_budget"] = dict(
            started_at=readout_budget.started_at,
            limits=asdict(readout_budget.budget),
            updates=readout_budget.updates_completed,
            replay=readout_budget.replay_examples_completed,
            last_clock=readout_budget.last_clock,
        )
    return report
