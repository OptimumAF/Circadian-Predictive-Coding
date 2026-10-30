"""One frozen ordinary-TRAIN retention comparison through the existing stream.

No optimizer or orchestration engine lives here. Both training arms finish before
readout; task outcomes never select packets, parameters, checkpoints or updates.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from time import monotonic
from typing import Any

import numpy as np

from scripts.run_ticket_workflow import CONTEXT_FEATURES, CONTEXT_TASKS, training_cases
from src.adapters.context_linear_fit_decision import ContextLinearFitDecisionPredictor
from src.adapters.context_linear_fit_learner import ContextLinearFitLearner
from src.adapters.linear_fit_learner import _rng_signature
from src.adapters.perceptron_decision import build_decision_training_batch
from src.app.coupled_attribution import AttributionStream
from src.app.stream_rehearsal import RecentStreamReplayPolicy, StreamRehearsal
from src.app.ticket_workflow import run_workflow_episode
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.context_joint_replay import JointContextReplayPolicy
from src.core.categorical_probability_scores import validate_class_targets
from src.core.context_linear_state import ContextLinearFitPolicy
from src.core.decision_features import DecisionFeatureSpec
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    EpisodeReplaySchema,
)
from src.core.temporal_decision_sequence import VOCABULARY
from src.core.ticket_workflow import ACTION_IDS, CATALOG, WorkflowCapabilities, WorkflowState
from src.shared.process_memory import ProcessRssSampler

CONTROLS = ("joint", "current_only")
POINTS = (0, 4, 6, 10)
TASKS = (*CONTEXT_TASKS, ("v1", "legacy"), ("v2", "legacy"))
RSS_CAP = 512 * 1024**2
LIMITS = dict(
    fit_attempts=36,
    fit_rows=192,
    fit_returns=36,
    lstsq_calls=36,
    typed_delegates=576,
    affine_calls=288,
    affine_rows=288,
    unfitted_reads=288,
    route_calls=1024,
    route_rows=4096,
    teacher_traces=8,
    dispatches=312,
    task_episodes=48,
    probability_calls=0,
)


class _StudySpent(BaseException):
    """Abort beyond model Exception handlers: profiler errors disable observation.

    Why: helpers must not recover from this stop and admit unobserved work.
    """


def protocol():
    """Fresh, outcome-free declaration; preview does not generate TRAIN teachers."""
    return dict(
        definition="dynamic_joint_train_retention_v1",
        controls=list(CONTROLS),
        tasks=[list(t) for t in TASKS],
        points=list(POINTS),
        label_delay=1,
        training_order=[list(t) for t in (*CONTEXT_TASKS, *TASKS[4:], *CONTEXT_TASKS)],
        initial="two identical unfitted zero-count LS heads; no random draws",
        producer="original training_cases v1/v2 contextual TRAIN; no probe cohorts",
        readout="48 ordinary sandbox episodes, max6 steps; both arms finish first",
        direction="higher per-task success; report old and new separately",
        replay=dict(every_updates=1, rows_per_event=12, retained_rows=30, bytes=65536),
        limits=dict(LIMITS),
        wall_seconds=60,
        rss_bytes=RSS_CAP,
        qualification="known TRAIN templates/public contexts; equal rows/fits, not equal compute or unseen-task generalization",
        failure="one two-arm case; any failure is SPENT, no repair/retry or cap renewal",
    )


def episode_packet(batch, identifier):
    """Certify three original TRAIN rows as one complete retained episode."""
    if batch.role != "train" or batch.features.shape != (3, 9) or batch.targets.shape != (3, 1):
        raise ValueError("three original TRAIN rows required")
    if batch.num_classes != 6:
        raise ValueError("six-class TRAIN schema required")
    validate_class_targets(batch.targets, 3, 6)
    observations = tuple(tuple(float(v) for v in row) for row in batch.features)
    transitions = tuple(
        DemonstrationTransition(
            i,
            observations[i],
            int(batch.targets[i, 0]),
            float(i == 2),
            observations[i + 1] if i < 2 else (0.0,) * 9,
            i == 2,
            6,
        )
        for i in range(3)
    )
    episode = DemonstrationEpisode(identifier, batch.context, "train", transitions, 6)
    return replace(
        batch,
        features=batch.features.copy(),
        targets=batch.targets.copy(),
        episode=episode,
        row_ids=tuple(f"{identifier}:step:{i}" for i in range(3)),
    )


def prepare_training():
    spec = DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG)
    teachers = {
        v: training_cases(policy_version=v, include_policy_context=True) for v in ("v1", "v2")
    }
    prototypes = []
    for version, concept in TASKS:
        offset = 3 * VOCABULARY.index(concept)
        prototypes.append(
            build_decision_training_batch(
                teachers[version][offset : offset + 3],
                spec,
                f"{version}-{concept}",
                allow_subsets=True,
            )
        )
    sequence = (*prototypes, *prototypes[:4])
    return spec, tuple(episode_packet(b, f"dynamic-{i}") for i, b in enumerate(sequence))


def _sessions(sampler):
    result = tuple(
        ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=10 if i == 0 else None,
                max_replay_examples=66 if i == 1 else None,
                max_wall_seconds=30,
                max_process_rss_bytes=RSS_CAP,
            ),
            monotonic,
        )
        for i in range(2)
    )
    for session in result:
        session.attach_memory(sampler)
    return result


def train_arm(control, training, sampler, progress):
    if control not in CONTROLS:
        raise ValueError("unknown frozen comparison control")
    source = ContextLinearFitLearner(ContextLinearFitPolicy(9, 6, (4, 5)))
    before, rng = source.snapshot_state(), _rng_signature()
    wake, replay = _sessions(sampler)
    helper = StreamRehearsal(
        source,
        wake,
        RecentStreamReplayPolicy(1, 12, ReplayRetentionBudget(30, 65536)),
        replay_budget=replay,
        total_updates=10,
        input_dim=9,
        episode_schema=EpisodeReplaySchema(9, 3, 6),
        joint_context=JointContextReplayPolicy(9, 6, (4, 5), control),
    )
    stream = AttributionStream(
        source,
        training,
        (),
        checkpoint_updates=POINTS,
        label_delay=1,
        budget=wake,
        rehearsal=helper,
        training_only=True,
    )
    snapshots = []
    observations: list[dict[str, Any]] = []
    progress.update(control=control, complete=False, checkpoints=observations)
    started = monotonic()
    try:
        for point in POINTS:
            if point:
                stream.run_until(point)
            state = stream.candidate_snapshot()
            snapshots.append(state)
            observations.append(
                dict(
                    wake_updates=point,
                    fits=state.completed_updates,
                    rows=state.training_examples,
                    retention=asdict(helper.retention),
                    model_coefficient_bytes=_model_bytes(state),
                )
            )
        progress["stream"] = stream.finish()
        if (wake.updates_completed, replay.replay_examples_completed) != (10, 66):
            raise ValueError("original wake/replay charges differ from frozen recipe")
        if (state.completed_updates, state.training_examples) != (18, 96):
            raise ValueError("physical committed fit/row dose differs from frozen recipe")
        if (
            source.snapshot_state() != before
            or stream._runtime.actor.snapshot_state().state != before
        ):
            raise ValueError("source or deployed actor changed")
        if _rng_signature() != rng:
            raise ValueError("training changed global RNG")
        progress["complete"] = True
        return tuple(snapshots)
    finally:
        progress.update(
            seconds=monotonic() - started,
            wake_charges=wake.updates_completed,
            replay_charges=replay.replay_examples_completed,
            rehearsal=deepcopy(helper.report),
        )


def _model_bytes(state):
    return sum(
        8 * len(row) for h in state.heads if h.model is not None for row in h.model.coefficients
    )


def evaluate_checkpoints(control, snapshots, spec, progress):
    points: list[dict[str, Any]] = []
    progress["readouts"] = points
    for point, state in zip(POINTS, snapshots, strict=True):
        reader = ContextLinearFitDecisionPredictor(spec, state)
        rows: list[dict[str, Any]] = []
        points.append(dict(wake_updates=point, tasks=rows))
        for version, concept in TASKS:
            episode = run_workflow_episode(
                WorkflowState(
                    f"{control}-{point}-{version}-{concept}", concept, policy_version=version
                ),
                reader,
                capabilities=WorkflowCapabilities(ACTION_IDS),
                max_steps=6,
                include_policy_context=True,
            )
            rows.append(dict(version=version, concept=concept, **episode))
            reader.assert_unchanged()
            if not episode["complete"] or episode["training_updates"] != 0:
                raise ValueError("incomplete or training readout cannot support comparison")
        points[-1]["typed_work"] = asdict(reader.work)
    return points


def summarize(points):
    curves = [[bool(row["success"]) for row in p["tasks"]] for p in points]
    if len(curves) != 4 or any(len(c) != 6 for c in curves):
        raise ValueError("all four fixed six-task readouts are required")
    return dict(
        old_successes=[sum(c[:4]) for c in curves],
        new_successes=[sum(c[4:]) for c in curves],
        per_task=[
            dict(
                task=list(t),
                success=[c[i] for c in curves],
                acquired_at_new=(i >= 4 and not curves[1][i] and curves[2][i]),
                lost_after_new=(i < 4 and curves[1][i] and not curves[2][i]),
                recovered_at_return=(i < 4 and curves[1][i] and not curves[2][i] and curves[3][i]),
            )
            for i, t in enumerate(TASKS)
        ],
    )


def run_comparison(report, sampler):
    rng = _rng_signature()
    spec, training = prepare_training()
    original = tuple((b.features.copy(), b.targets.copy(), deepcopy(b.episode)) for b in training)
    snapshots = []
    arms = report["arms"] = []
    for control in CONTROLS:
        progress: dict[str, Any] = {}
        arms.append(progress)
        snapshots.append(train_arm(control, training, sampler, progress))
    # Why: even the first arm's outcomes are unavailable to the second arm.
    if not all(a["complete"] for a in arms):
        raise ValueError("both original training arms must finish before any readout")
    for b, (x, y, episode) in zip(training, original, strict=True):
        if (
            not np.array_equal(x, b.features)
            or not np.array_equal(y, b.targets)
            or b.episode != episode
        ):
            raise ValueError("original TRAIN packet changed")
    for control, states, progress in zip(CONTROLS, snapshots, arms, strict=True):
        progress["summary"] = summarize(evaluate_checkpoints(control, states, spec, progress))
    if _rng_signature() != rng:
        raise ValueError("comparison changed global RNG")
    report["complete"] = True


def observe_work(counts, started, sampler):
    """Study-local call observer; refusal before a call keeps its attempted charge."""

    def profile(frame, event, result):
        path, name = frame.f_code.co_filename.replace("\\", "/"), frame.f_code.co_name
        if (
            event == "return"
            and result is not None
            and name == "fit_categorical_linear"
            and path.endswith("src/core/categorical_linear_fit.py")
        ):
            counts["fit_returns"] += 1
        if event != "call":
            return
        add = {}
        if path.endswith("src/core/categorical_linear_fit.py"):
            if name == "fit_categorical_linear":
                add = dict(fit_attempts=1, fit_rows=len(frame.f_locals["features"]))
            elif name == "predict_categorical_linear_logits":
                add = dict(affine_calls=1, affine_rows=len(frame.f_locals["features"]))
            elif name == "predict_categorical_linear":
                add = dict(probability_calls=1)
        elif "/numpy/" in path and name == "lstsq":
            add = dict(lstsq_calls=1)
        elif path.endswith("src/core/context_perceptron.py") and name == "route_context_features":
            add = dict(route_calls=1, route_rows=len(frame.f_locals["features"]))
        elif path.endswith(
            ("src/adapters/linear_fit_decision.py", "src/adapters/context_linear_fit_decision.py")
        ):
            if name == "__call__":
                add = dict(typed_delegates=1)
            elif name == "predict_logits" and frame.f_locals["self"].model is None:
                add = dict(unfitted_reads=1)
        elif path.endswith("src/core/ticket_workflow.py") and name == "dispatch_workflow":
            add = dict(dispatches=1)
        elif path.endswith("scripts/run_ticket_workflow.py") and name == "training_cases":
            add = dict(teacher_traces=4)
        elif path.endswith("src/app/ticket_workflow.py") and name == "run_workflow_episode":
            add = dict(task_episodes=1)
        for key, value in add.items():
            counts[key] += value
            if counts[key] > LIMITS[key]:
                raise _StudySpent(f"SPENT study work cap: {key}")
        if add and (monotonic() - started > 60 or (sampler.peak_bytes or RSS_CAP + 1) > RSS_CAP):
            raise _StudySpent("SPENT study time/RSS cap")

    return profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is None:
        print(json.dumps(protocol(), indent=2))
        return 0
    report: dict[str, Any] = dict(protocol=protocol(), complete=False, disposition="FAILED/SPENT")
    counts = dict.fromkeys(LIMITS, 0)
    started = monotonic()
    # Exclusive creation spends this case even when the producer or a fit fails.
    with (
        args.output.open("x", encoding="utf-8") as output,
        ProcessRssSampler(interval_seconds=0.02) as sampler,
    ):
        try:
            sys.setprofile(observe_work(counts, started, sampler))
            run_comparison(report, sampler)
            if (
                counts["fit_attempts"],
                counts["fit_rows"],
                counts["lstsq_calls"],
                counts["fit_returns"],
            ) != (36, 192, 36, 36):
                raise ValueError("actual native work does not match frozen fit/row dose")
            report["disposition"] = "COMPLETE/SPENT"
        except (Exception, _StudySpent) as error:
            report.update(complete=False, error=f"{type(error).__name__}: {error}"[:1024])
        finally:
            sys.setprofile(None)
            sampler.sample()
            report.update(
                work=counts,
                seconds=monotonic() - started,
                peak_rss_bytes=sampler.peak_bytes,
                torch_imported="torch" in sys.modules,
                telemetry_scope="body through model/readouts; serialization/output excluded; terminal and parent receipt include closeout",
            )
            if report["seconds"] > 60 or (sampler.peak_bytes or RSS_CAP + 1) > RSS_CAP:
                report.update(
                    complete=False, disposition="FAILED/SPENT", error="study time/RSS cap"
                )
            serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
            if len(serialized.encode("utf-8")) > 8 * 1024**2:
                report.update(complete=False, disposition="FAILED/SPENT", error="study output cap")
                serialized = (
                    json.dumps({k: v for k, v in report.items() if k != "arms"}, indent=2) + "\n"
                )
            output.write(serialized)
    output_seconds, output_peak = monotonic() - started, sampler.peak_bytes
    accepted = (
        report["complete"]
        and output_seconds <= 60
        and output_peak is not None
        and output_peak > 0
        and output_peak <= RSS_CAP
    )
    print(
        json.dumps(
            dict(
                complete=accepted,
                disposition="COMPLETE/SPENT" if accepted else "FAILED/SPENT",
                work=counts,
                seconds=report["seconds"],
                peak_rss_bytes=report["peak_rss_bytes"],
                output_pipeline_seconds=output_seconds,
                peak_rss_through_output_bytes=output_peak,
            )
        )
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
