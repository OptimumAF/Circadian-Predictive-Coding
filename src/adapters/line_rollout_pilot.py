"""Live evaluation of all logged checkpoints from the existing trajectory recipe.

Owns bounded source-file reading, fixed protocol and native prediction binding.
Training stays recorded and unchanged; live actions never select or update a
learner. The app episode runner owns transition semantics, not another trainer.
"""

from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from src.adapters.frozen_torch_predictor import FrozenTorchPredictor, validate_parameters
from src.app.line_policy_evaluation import (
    EvaluationIdentity,
    LineEvaluationFailed,
    run_line_episode,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.binary_probability_scores import probability_scores
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch

METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
POINTS = (0, 2, 4, 6)
SOURCE_PROTOCOL = "native_torch_complete_demonstration_rehearsal_v1"
MAX_BYTES = 2 * 1024**2
MAX_RSS = 768 * 1024**2


def describe_protocol() -> dict[str, Any]:
    return dict(
        protocol_id="logged_torch_parameters_live_line_rollouts_v1",
        source_protocol=SOURCE_PROTOCOL,
        classification="exploratory_frozen_prediction_import_and_live_evaluation",
        methods=list(METHODS),
        replay_modes=[False, True],
        seed=17,
        checkpoint_updates=list(POINTS),
        task_sequence=["A", "B", "A"],
        checkpoint_origin="recorded_parameters_not_historical_training_state",
        actor_version="local evaluation identity, not promoted historical actor version",
        training_origin="recorded_completed_source_run; no training rerun",
        execution_origin="live_environment",
        phase="evaluation",
        learning_active=False,
        source_required="explicit --source-artifact path; bounded validated original trajectory JSON",
        episode_configs=[dict(context="A", start=-1, goal=1), dict(context="B", start=1, goal=-1)],
        policy="native class1 probability>=.5 selects right; otherwise left; no action repair",
        max_steps=4,
        success="terminal goal reached",
        return_metric="sum of actual rewards",
        parity="all64 cached final action scores before any live episode; exact accuracy/BCE atol1e-12 rtol0",
        planned_work=dict(
            frozen_predictors=32,
            cached_score_forwards=64,
            live_episodes=64,
            maximum_rollout_forwards=256,
            maximum_total_forwards=320,
            maximum_environment_steps=256,
            training_updates=0,
            replay=0,
            probes=0,
        ),
        resources=dict(
            cpu_threads=1,
            gpu=0,
            wall_seconds=60,
            process_rss_bytes=MAX_RSS,
            storage_bytes=MAX_BYTES,
            source_bytes=MAX_BYTES,
        ),
        stop_policy="one invocation; preserve partial/unreached records; unexpected failure stops all future work",
        limitations="one source seed/tiny environment; no optimizer/RNG reconstruction, publication, tuning, autonomous training, MiniGrid, generalization or circadian advantage",
    )


def _score_record(value: Any) -> None:
    if type(value) is not dict or set(value) != {"bce", "accuracy"}:
        raise ValueError("source scores require BCE and accuracy")
    for name, number in value.items():
        if type(number) not in (int, float) or not isfinite(number) or number < 0:
            raise ValueError("source scores must be finite nonnegative native numbers")
        if name == "accuracy" and number > 1:
            raise ValueError("source accuracy must be in [0,1]")


def validate_source(source: Any) -> list[dict[str, Any]]:
    """Validate only this declared source seam before Torch or RSS construction."""
    if type(source) is not dict or source.get("status") != "complete":
        raise ValueError("source must be a complete recorded trajectory report")
    protocol = source.get("protocol")
    if type(protocol) is not dict or any(
        protocol.get(k) != v
        for k, v in dict(protocol_id=SOURCE_PROTOCOL, seed=17, feature_dim=2, hidden_dim=3).items()
    ):
        raise ValueError("source protocol or fixed prediction geometry differs")
    if any(type(protocol[k]) is not int for k in ("seed", "feature_dim", "hidden_dim")):
        raise ValueError("source geometry controls must be native integers")
    for key, expected_value in dict(
        phase_contexts=["A", "B", "A"], phase_lengths=[2, 2, 2], checkpoint_updates=list(POINTS)
    ).items():
        actual = protocol.get(key)
        if type(actual) is not list or actual != expected_value:
            raise ValueError("source task sequence or checkpoint schedule differs")
        if key != "phase_contexts" and any(type(value) is not int for value in actual):
            raise ValueError("source task/checkpoint counts must be native integers")
    runs = source.get("runs")
    if type(runs) is not list or len(runs) != 8:
        raise ValueError("source must contain all eight ordered method/replay arms")
    result = []
    for index, run in enumerate(runs):
        method, enabled = METHODS[index % 4], index >= 4
        if (
            type(run) is not dict
            or run.get("status") != "complete"
            or run.get("method") != method
            or type(run.get("replay_enabled")) is not bool
            or run["replay_enabled"] is not enabled
        ):
            raise ValueError("source method/replay order or completion differs")
        observations = run.get("observations")
        stream = run.get("stream_result")
        checkpoints = stream.get("checkpoints") if type(stream) is dict else None
        if (
            type(observations) is not list
            or len(observations) != 4
            or type(checkpoints) is not list
            or len(checkpoints) != 4
        ):
            raise ValueError("source requires all four parameter and score checkpoints")
        for j, (observation, checkpoint) in enumerate(zip(observations, checkpoints, strict=True)):
            if type(observation) is not dict or type(checkpoint) is not dict:
                raise ValueError("source checkpoints must be records")
            expected = dict(
                wake_updates=POINTS[j],
                native_updates=j * (3 if enabled else 2),
                native_training_examples=j * (6 if enabled else 4),
            )
            if any(
                type(observation.get(k)) is not int or observation[k] != v
                for k, v in expected.items()
            ):
                raise ValueError("source checkpoint identity or native clocks differ")
            if type(checkpoint.get("updates")) is not int or checkpoint["updates"] != POINTS[j]:
                raise ValueError("source score checkpoint order differs")
            if checkpoint.get("training_context") != (None, "A", "B", "A")[j]:
                raise ValueError("source checkpoint training context differs")
            scores = checkpoint.get("scores")
            if type(scores) is not dict or set(scores) != {"A", "B"}:
                raise ValueError("source requires both context scores")
            for score in scores.values():
                _score_record(score)
            validate_parameters(observation.get("parameters"))
            result.append(
                dict(
                    method=method,
                    replay_enabled=enabled,
                    updates=POINTS[j],
                    native_updates=expected["native_updates"],
                    parameters=observation["parameters"],
                    scores=scores,
                )
            )
    return result


def read_source(path: str | Path) -> tuple[str, list[dict[str, Any]]]:
    location = Path(path).resolve()
    with location.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("source report exceeds 2MiB allowance")
    return str(location), validate_source(json.loads(raw.decode("utf8")))


def _identity(point: dict[str, Any]) -> EvaluationIdentity:
    return EvaluationIdentity(
        point["method"],
        17,
        f"eval:{point['method']}:replay-{point['replay_enabled']}:wake-{point['updates']}",
        ("A", "B", "A"),
        point["updates"],
        point["replay_enabled"],
    )


def _planned() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    points = [
        dict(method=method, replay_enabled=enabled, updates=updates)
        for enabled in (False, True)
        for method in METHODS
        for updates in POINTS
    ]
    checks = [
        dict(
            identity=asdict(_identity(point)),
            status="not_run",
            scores=None,
            reason="source import not reached",
            state_unchanged=None,
        )
        for point in points
    ]
    episodes = [
        dict(
            identity=asdict(_identity(point)),
            context=context,
            status="not_run",
            reason="live evaluation not reached",
            execution_origin="live_environment",
            checkpoint_origin="recorded_parameters",
            phase="evaluation",
            learning_active=False,
        )
        for point in points
        for context in ("A", "B")
    ]
    return checks, episodes


def _qualify(model: FrozenTorchPredictor, point: dict[str, Any], predict: Any) -> dict[str, Any]:
    scores = {}
    for context, inputs, target in (
        ("A", [[-0.5, 0.5], [0.0, 0.5]], 1.0),
        ("B", [[0.5, -0.5], [0.0, -0.5]], 0.0),
    ):
        values = predict(model, np.array(inputs, dtype=np.float64))
        score = probability_scores(values, np.full((2, 1), target, dtype=np.float64))
        reference = point["scores"][context]
        if (
            score["accuracy"] != reference["accuracy"]
            or abs(score["bce"] - reference["bce"]) > 1e-12
        ):
            raise ValueError("frozen parameter import does not reproduce the cached score")
        scores[context] = score
    return scores


def run_pilot(source_artifact: str | Path) -> dict[str, Any]:
    """Import once, qualify every checkpoint, then evaluate without learning."""
    started = monotonic()
    checks, episodes = _planned()
    errors: list[str] = []
    work = dict(predictors_created=0, forwards_attempted=0, forwards_returned=0)
    models: list[FrozenTorchPredictor] = []
    source_reference = None
    sampler = None
    memory = None
    active_check = None
    try:
        source_reference, points = read_source(source_artifact)
        sampler = ProcessRssSampler()
        with sampler:
            budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=0,
                    max_wall_seconds=max(0.0, 60 - (monotonic() - started)),
                    max_process_rss_bytes=MAX_RSS,
                ),
                monotonic,
            )
            budget.attach_memory(sampler)
            torch = require_torch()
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)

            def predict(model, features):
                work["forwards_attempted"] += 1
                values = model.predict(features)
                work["forwards_returned"] += 1
                budget.before_final()
                return values

            def parity_predict(model, features):
                budget.before_final()
                return predict(model, features)

            for active_check, point in zip(checks, points, strict=True):
                budget.before_final()
                model = FrozenTorchPredictor(point["method"], point["parameters"], seed=17)
                models.append(model)
                work["predictors_created"] += 1
                scores = _qualify(model, point, parity_predict)
                active_check.update(
                    status="complete", scores=scores, reason=None, state_unchanged=True
                )
            active_check = None
            for index, (point, model) in enumerate(zip(points, models, strict=True)):
                for offset, (context, start, goal) in enumerate((("A", -1, 1), ("B", 1, -1))):
                    position = 2 * index + offset
                    try:
                        episodes[position] = run_line_episode(
                            lambda features: predict(model, features),
                            identity=_identity(point),
                            context=context,
                            start=start,
                            goal=goal,
                            max_steps=4,
                            before_prediction=budget.before_final,
                        )
                    except LineEvaluationFailed as error:
                        episodes[position] = error.report
                        raise
            budget.before_final()
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        errors.append(reason)
        if active_check is not None:
            active_check.update(status="failed", reason=reason)
    if sampler is not None:
        try:
            memory = asdict(sampler.snapshot())
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
    elapsed = monotonic() - started
    complete = (
        not errors
        and all(check["status"] == "complete" for check in checks)
        and all(episode["status"] == "complete" for episode in episodes)
        and memory is not None
        and memory["peak_bytes"] <= MAX_RSS
        and elapsed <= 60
    )
    return dict(
        protocol=describe_protocol(),
        status="complete" if complete else "incomplete",
        source_reference=source_reference,
        source_training_origin="recorded_completed_run",
        checkpoints=checks,
        episodes=episodes,
        work=work,
        global_errors=errors,
        observed_process_rss=memory,
        total_seconds=elapsed,
        training_updates=0,
        policy_feedback=False,
        promotion_performed=False,
        historical_training_state_restored=False,
        scientific_advantage_verified=False,
        full_R41_complete=False,
    )
