"""One same-data TRAIN affine control; no historical model restoration or robot IO.

Uses the existing qualified198-group reader and original six-group weighting.
TS12 inputs already contain grasp, displacement/.1m and executed command history;
no fitted normalization, role conversion or task-name feature is introduced.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import monotonic
from typing import Any, Callable

import numpy as np

from scripts.diagnose_aggregated_supervision import (
    SOURCE,
    _bound_bytes,
    _require_original_unchanged,
    read_aggregated_inputs,
)
from scripts.push_cube_correction_weights import build_aggregated_correction_weights
from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import budget_binding, paid_report_guard
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.continuous_affine_fit import (
    DEFINITION,
    fit_continuous_affine,
    predict_continuous_affine,
)
from src.core.continuous_demonstration import continuous_training_pairs, copy_continuous_episode
from src.core.continuous_readout_fit import _coefficient_norm
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.shared.process_memory import ProcessRssSampler


def _source_pin(groups):
    return tuple(
        (
            group,
            episode,
            vars(episode),
            episode.spec,
            vars(episode.spec),
            episode.transitions,
            tuple(
                (row, vars(row), row.observation, row.action, row.next_observation)
                for row in episode.transitions
            ),
        )
        for group, episode in groups
    )


def _check_sources(groups, pin, copies):
    if type(groups) is not tuple or len(groups) != len(pin):
        raise ValueError("physical affine original TRAIN groups changed")
    for (name, episode), (
        original_name,
        owner,
        dictionary,
        spec,
        spec_dictionary,
        transitions,
        rows,
    ), (_, expected) in zip(groups, pin, copies, strict=True):
        if (
            name is not original_name
            or episode is not owner
            or vars(episode) is not dictionary
            or episode.spec is not spec
            or vars(episode.spec) is not spec_dictionary
            or episode.transitions is not transitions
            or copy_continuous_episode(episode) != expected
        ):
            raise ValueError("physical affine original TRAIN record changed or was replaced")
        for row, (original, dictionary, observation, action, next_observation) in zip(
            episode.transitions, rows, strict=True
        ):
            if (
                row is not original
                or vars(row) is not dictionary
                or row.observation is not observation
                or row.action is not action
                or row.next_observation is not next_observation
            ):
                raise ValueError("physical affine original TRAIN tuples changed or were replaced")


def _metrics(raw, bounded, targets, weights):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        raw_error = np.mean((raw - targets) ** 2, axis=1) * 0.5
        bounded_error = np.mean((bounded - targets) ** 2, axis=1) * 0.5
    return dict(
        raw_half_mse=float(np.mean(raw_error)),
        bounded_half_mse=float(np.mean(bounded_error)),
        weighted_raw_half_mse=float(np.dot(weights / weights.sum(), raw_error)),
        weighted_bounded_half_mse=float(np.dot(weights / weights.sum(), bounded_error)),
        clipped_per_axis=[int(np.count_nonzero(raw[:, i] != bounded[:, i])) for i in range(4)],
    )


def fit_groups(
    groups, budget: ToyBudgetSession, *, check_boundary: Callable[[], None] | None = None
):
    """Admit ordinary TRAIN and charge the single attempt before the solve."""
    if check_boundary is not None and not callable(check_boundary):
        raise ValueError("physical affine boundary must be callable")
    weights, recipe = build_aggregated_correction_weights(groups)
    copies = tuple((name, copy_continuous_episode(episode)) for name, episode in groups)
    if any(episode.spec != cube_taskspace_spec() for _, episode in copies):
        raise ValueError("physical affine TRAIN groups require original TS12 geometry")
    if (
        type(budget) is not ToyBudgetSession
        or budget.updates_completed != 0
        or budget.replay_examples_completed != 0
        or budget.budget.max_training_updates != 1
    ):
        raise ValueError("physical affine control requires its fresh single-fit budget")
    binding, limits, pin = budget_binding(budget), replace(budget.budget), _source_pin(groups)
    recipe_guard, original_recipe = paid_report_guard(recipe)
    expected_charges = 0
    pairs = [continuous_training_pairs(episode, role="train") for _, episode in copies]
    features = np.array([row for x, _ in pairs for row in x], dtype=np.float64)
    targets = np.array([row for _, y in pairs for row in y], dtype=np.float64)
    weight_array = np.array(weights, dtype=np.float64)
    work = dict(
        fit_attempts=0,
        fit_returns=0,
        fit_rows_attempted=0,
        fit_rows_returned=0,
        prediction_attempts=0,
        prediction_returns=0,
        prediction_rows=0,
        input_array_bytes=features.nbytes + targets.nbytes + weight_array.nbytes,
    )
    report: dict[str, Any] = dict(
        definition=DEFINITION,
        complete=False,
        error=None,
        model=None,
        metrics=None,
        groups=[],
        weight_recipe=recipe,
        source_inputs_unchanged=False,
        work=work,
        memory_scope="input_array_bytes counts three app arrays only; core design/guard/solver temporaries, episodes and Python overhead are additional RSS",
        qualification="weighted unclipped physical MSE solve; clipping is prediction only, not BCE/inverse-tanh/clipped-MSE optimization; TRAIN fit is not native success",
    )

    def guard():
        if check_boundary is not None:
            check_boundary()
        _check_sources(groups, pin, copies)
        recipe_guard()
        if (
            budget_binding(budget) != binding
            or not _same_value(vars(budget.budget), vars(limits))
            or type(budget.updates_completed) is not int
            or type(budget.replay_examples_completed) is not int
            or budget.updates_completed != expected_charges
            or budget.replay_examples_completed != 0
        ):
            raise ValueError("physical affine original budget/clock/charges changed")
        budget.before_final()

    try:
        guard()
        budget.before_update()
        # The generic update slot records a FIT ATTEMPT, not a returned SGD step.
        try:
            budget.record_update()
        finally:
            # Preserve a paid slot even if its acknowledgement is lost.
            expected_charges = int(
                type(budget.updates_completed) is int and budget.updates_completed == 1
            )
        if expected_charges != 1:
            raise ValueError("physical affine fit charge lacks its observed transition")
        guard()
        work.update(fit_attempts=1, fit_rows_attempted=len(features))
        model = fit_continuous_affine(features, targets, weight_array, copies[0][1].spec)
        work.update(fit_returns=1, fit_rows_returned=len(features))
        guard()
        work.update(prediction_attempts=1, prediction_rows=len(features))
        prediction = predict_continuous_affine(model, features)
        work["prediction_returns"] = 1
        guard()
        report["model"] = dict(
            asdict(model), coefficient_norm=_coefficient_norm(np.array(model.coefficients))
        )
        report["metrics"] = _metrics(prediction.raw, prediction.bounded, targets, weight_array)
        offset = 0
        for name, episode in copies:
            stop = offset + len(episode.transitions)
            region = slice(offset, stop)
            report["groups"].append(
                dict(
                    group=name,
                    episode_id=episode.episode_id,
                    role="train",
                    rows=stop - offset,
                    metrics=_metrics(
                        prediction.raw[region],
                        prediction.bounded[region],
                        targets[region],
                        weight_array[region],
                    ),
                )
            )
            offset = stop
        guard()
        report.update(complete=True, source_inputs_unchanged=True)
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}"[:512], model=None)
    finally:
        report["weight_recipe"] = original_recipe
        report["budget"] = dict(
            initial_limits=asdict(limits),
            observed_limits=asdict(budget.budget),
            original_started_at=binding[2],
            observed_started_at=budget.started_at,
            charged_fit_attempts=expected_charges,
            observed_update_slots=budget.updates_completed,
            scope="attempt slot is permanently charged before solve; not SGD/returned work",
        )
    return report


def run_control(output: Path, *, source: Path = SOURCE):
    """Exclusive one-shot capture; repeated output/marker cannot renew work."""
    if not output.parent.is_dir() or output.exists() or output.is_symlink():
        raise ValueError("affine control requires a fresh output in an existing directory")
    with output.with_name(output.name + ".SPENT").open("x"):
        pass
    started = monotonic()
    report: dict[str, Any] = dict(
        definition=DEFINITION,
        complete=False,
        error=None,
        work=None,
        sources=None,
        source_bytes_unchanged=False,
    )
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=1,
                max_replay_examples=0,
                max_wall_seconds=60,
                max_process_rss_bytes=512 * 1024**2,
            ),
            monotonic,
        )
        budget.attach_memory(memory)
        try:
            groups, receipt = read_aggregated_inputs(source)
            budget.before_final()
            report = fit_groups(groups, budget)
            report["source_bytes_unchanged"] = False
            report["sources"] = receipt
            _bound_bytes(source, work=receipt["accepted_study_read_work"])
            final_read = dict(attempts=0, returns=0, bytes=0)
            report["original_correction_final_read"] = final_read
            _require_original_unchanged(receipt["original_correction"], final_read)
            budget.before_final()
            report["source_bytes_unchanged"] = True
        except Exception as error:
            report.update(
                complete=False, error=f"{type(error).__name__}: {error}"[:512], model=None
            )
            if report.get("sources") is None:
                report["unreturned_source_read_work_unknown"] = True
        finally:
            memory.sample()
            report["elapsed_seconds"] = monotonic() - started
            report["sampled_process_peak_bytes"] = memory.peak_bytes
            report["charged_fit_attempts"] = budget.updates_completed
            report["scope"] = (
                "one cold same-data TRAIN control; no RNG/optimizer/replay/robot/probe/final/promotion or old model restoration"
            )
    payload = json.dumps(report, allow_nan=False)
    if len(payload.encode("utf8")) > 1024**2:
        raise ValueError("affine control report exceeds1MiB; attempt remains spent")
    with output.open("x", encoding="utf8") as handle:
        handle.write(payload + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    report = run_control(args.output, source=args.source)
    print(json.dumps(dict(complete=report["complete"], error=report["error"])))
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
