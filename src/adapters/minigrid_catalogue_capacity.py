"""Fixed TRAIN C suffix through the existing stream, from a saved after-B fit.

Inputs are admitted raw public-history TRAIN episodes and a complete saved fit.
Outputs retain both capacity arms, receipts, costs, failures and after-C scores.
No filesystem, native games, final/probe data, live publication or new runner.
"""

from dataclasses import asdict, replace
from time import monotonic
from typing import Any

from src.adapters.minigrid_context_linear_learner import (
    ContextLinearLearner,
    copy_context_linear_training_snapshot,
)
from src.adapters.minigrid_train_catalogue import _validate_sources, compose_arrived_cue_catalogue
from src.app.actor_shadow import StableActor
from src.app.coupled_attribution import AttributionStream, _same_value
from src.app.probe_return_phase import budget_binding
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_probability_scores import probability_scores
from src.core.demonstration_episode import episode_arrays
from src.core.learner_ports import TrainingUpdateFailed
from src.shared.process_memory import ProcessRssSampler


def describe_protocol():
    return dict(
        definition="minigrid_after_b_new_c_capacity_v1",
        capacities=(1, 2),
        source_paid_fit_calls=6,
        source_paid_fit_rows=90,
        selected_prefix_fit_calls=4,
        selected_prefix_fit_rows=54,
        schedule="C_train blue, C_train red; delay1; independent study forks, not resumed authority",
        suffix_observation_tick_offset=4,
        new_fit_calls=4,
        new_fit_rows=90,
        arrival_row_exposures=36,
        replay_row_exposures=54,
        score_calls=14,
        score_rows=126,
        wall_seconds=90,
        peak_rss_bytes=512 * 1024**2,
        qualification="TRAIN acquisition/retention only; unequal exposure; no native-game/generalization/CPC benefit",
    )


def _admit(episodes, state):
    saved = copy_context_linear_training_snapshot(state)
    if (
        saved.completed_updates != 4
        or saved.training_examples != 54
        or saved.failed_work is not None
        or any(head is None or head.fit_rows != 18 for head in saved.heads)
    ):
        raise ValueError("new-C study requires the retained after-B4fit54row state")
    expected = tuple(
        f"minigrid-source-{context}-{cue}"
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )
    if type(episodes) is not tuple or tuple(e.episode_id for e in episodes) != expected:
        raise ValueError("new-C study requires original ordered A/B/C_train source pairs")
    witness, _ = _validate_sources(episodes, 1)
    plans = tuple(compose_arrived_cue_catalogue(episodes, retained_per_cue=n) for n in (1, 2))
    if any(tuple(w.cue for w in plan.wakes) != ("blue", "red") * 3 for plan in plans):
        raise ValueError("new-C study requires alternating public blue/red cues")
    return saved, witness, plans


def _arm(plan):
    wakes = plan.wakes[4:]
    return dict(
        retained_per_cue=plan.retained_per_cue,
        status="running",
        stream=None,
        failure_work=None,
        committed_receipts=None,
        receipt_capture_error=None,
        last_checkpoint=None,
        source_unchanged=None,
        actor_unchanged=None,
        scores={},
        catalogue=dict(
            wakes=[asdict(wake) for wake in wakes],
            arrival_rows=sum(w.current_rows for w in wakes),
            replay_rows=sum(w.replay_rows for w in wakes),
            fit_rows=sum(w.current_rows + w.replay_rows for w in wakes),
            committed_rows=None,
            failure_work_may_be_unobserved=False,
            full_plan_input_array_bytes=plan.input_array_bytes,
            suffix_input_array_bytes=sum(
                b.features.nbytes + b.targets.nbytes for b in plan.training[4:]
            ),
            peak_joint_template_array_bytes=plan.peak_joint_template_array_bytes,
            peak_bank_rows=plan.peak_bank_rows,
            peak_bank_observation_target_float64_equivalent_bytes=plan.peak_bank_observation_target_float64_equivalent_bytes,
            storage_scope="shared episode references; current+next+target equivalents, reward/metadata/Python excluded; full allocations in RSS",
        ),
    )


def _capture_receipts(arm, stream, plan):
    if stream is None:
        return
    try:
        arm["committed_receipts"] = [asdict(receipt) for receipt in stream.applied_updates]
        count = len(arm["committed_receipts"])
        prefix = plan.wakes[4 : 4 + count]
        arm["catalogue"]["committed_rows"] = dict(
            arrivals=sum(w.current_rows for w in prefix),
            replay=sum(w.replay_rows for w in prefix),
            native=sum(w.current_rows + w.replay_rows for w in prefix),
        )
        arm["catalogue"]["failed_attempt_replay_rows_upper_bound"] = (
            plan.wakes[4 + count].replay_rows if arm["status"] == "failed" and count < 2 else 0
        )
    except Exception as error:
        arm["receipt_capture_error"] = f"{type(error).__name__}: {error}"[:512]
        arm["status"] = "failed"
        arm["catalogue"]["failure_work_may_be_unobserved"] = True
        raise


def _score_episodes(report, reader, endpoint, episodes, scores, guard):
    for episode in episodes:
        guard()
        x, y = episode_arrays(episode)
        report["score_calls_attempted"] += 1
        report["score_rows_attempted"] += len(x)
        probabilities = reader.predict(x)
        metrics = probability_scores(probabilities, y, 7)
        scores[episode.episode_id] = dict(
            metrics,
            actions=probabilities.argmax(axis=1).tolist(),
            expected=y[:, 0].astype(int).tolist(),
        )
        report["score_calls_completed"] += 1
        report["score_rows_completed"] += len(x)
        if reader.snapshot_state() != endpoint:
            raise ValueError("new-C diagnostic changed its frozen endpoint")


def compare_new_c_capacity(episodes, state, *, check_boundary=lambda: None):
    saved, witness, plans = _admit(episodes, state)
    report: dict[str, Any] = dict(
        definition=describe_protocol()["definition"],
        protocol=describe_protocol(),
        status="running",
        reason=None,
        training_complete=False,
        source_unchanged=None,
        arms=[],
        initial_new_task_scores={},
        score_calls_attempted=0,
        score_rows_attempted=0,
        score_calls_completed=0,
        score_rows_completed=0,
    )
    limits = ToyExecutionBudget(
        max_training_updates=4, max_wall_seconds=90, max_process_rss_bytes=512 * 1024**2
    )
    report["initial_study_limits"] = asdict(limits)
    started = monotonic()
    endpoints = []
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        budget = ToyBudgetSession(limits, monotonic)
        budget.attach_memory(memory)
        binding, original_limits = budget_binding(budget), replace(limits)
        original_clock = budget.clock
        source = ContextLinearLearner()
        source.restore_state(saved)

        def guard():
            if (
                budget_binding(budget) != binding
                or not _same_value(vars(budget.budget), vars(original_limits))
                or type(budget.updates_completed) is not int
                or not 0 <= budget.updates_completed <= 4
                or type(budget.replay_examples_completed) is not int
                or budget.replay_examples_completed != 0
            ):
                raise ValueError("new-C study original four-update budget changed")
            budget.before_final()
            check_boundary()
            witness.check(episodes)
            if source.snapshot_state() != saved:
                raise ValueError("new-C study original after-B source changed")

        try:
            for plan in plans:
                guard()
                arm = _arm(plan)
                report["arms"].append(arm)
                actor = StableActor(source.fork(), version="after-B-study-source")
                original_actor = actor.snapshot_state()
                stream = None

                def observe(n, current, batch):
                    # Local suffix counts start at0; native source history stays4/54.
                    expected_rows = 54 + sum(len(b.features) for b in plan.training[4 : 4 + n])
                    if (
                        current.completed_updates != 4 + n
                        or current.training_examples != expected_rows
                    ):
                        raise ValueError(
                            "new-C native counters differ from source plus suffix receipts"
                        )
                    arm["last_checkpoint"] = dict(suffix_updates=n, state=asdict(current))
                    if n == 2:
                        endpoints.append(copy_context_linear_training_snapshot(current))
                    guard()

                try:
                    stream = AttributionStream(
                        source,
                        plan.training[4:],
                        (),
                        training_only=True,
                        checkpoint_updates=(0, 1, 2),
                        label_delay=1,
                        budget=budget,
                        observe_checkpoint=observe,
                        actor=actor,
                    )
                    stream.run_until(2)
                    arm["stream"] = stream.finish()
                    if source.snapshot_state() != saved or actor.snapshot_state() != original_actor:
                        raise ValueError("new-C training changed original source or actor")
                    arm.update(status="complete", source_unchanged=True, actor_unchanged=True)
                except Exception as error:
                    arm["status"] = "failed"
                    arm["catalogue"]["failure_work_may_be_unobserved"] = True
                    if isinstance(error, TrainingUpdateFailed):
                        arm["failure_work"] = dict(definition=error.definition, work=error.work)
                    raise
                finally:
                    _capture_receipts(arm, stream, plan)
            # Why: neither TRAIN scores nor C targets may select capacity/schedule.
            # Both fixed suffix fits finish before the first diagnostic prediction.
            report["training_complete"] = True
            _score_episodes(
                report, source.fork(), saved, episodes[4:], report["initial_new_task_scores"], guard
            )
            for arm, endpoint in zip(report["arms"], endpoints, strict=True):
                reader = source.fork()
                reader.restore_state(endpoint)
                _score_episodes(report, reader, endpoint, episodes, arm["scores"], guard)
            guard()
            report.update(status="complete", source_unchanged=True)
        except Exception as error:
            report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
        finally:
            report["budget"] = dict(
                limits=asdict(budget.budget),
                updates_completed=budget.updates_completed,
                replay_examples_completed=budget.replay_examples_completed,
                started_at=budget.started_at,
                ended_at=original_clock(),
                elapsed_seconds=monotonic() - started,
                sampled_peak_rss_bytes=memory.peak_bytes,
            )
    return report
