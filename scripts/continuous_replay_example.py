"""Fixed TRAIN rehearsal versus saved controls, using the existing stream runner.

Run only under a separately reviewed local protocol. The saved baseline is never
trained again; all new reports, including failures, are retained before checks.
Outputs are signed learning/retention tradeoffs and explicit additional costs,
not a matched-compute experiment or authority to open final/guard/native data.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from importlib import resources
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator
from types import SimpleNamespace

import numpy as np

from scripts.continuous_learnability_example import _episode, _final_state
from scripts.continuous_probe_example import _fresh_episode
from scripts.continuous_probe_control_example import _without_elapsed
from scripts.continuous_probe_direction_example import _common_probe
from scripts.continue_continuous_probe_control import (
    ORIGINAL_ERROR,
    _json,
    _metrics,
    _receipt_observations,
)
from src.adapters.continuous_learnability_demo import (
    _metrics as _benchmark_metrics,
    run_continuous_learnability,
)
from src.adapters.continuous_learnability_modes import FAMILIES
from src.adapters.continuous_probe import ContinuousProbePlan
from src.adapters.continuous_probe_control import _contrast, _native_state, _require_completed
from src.adapters.numpy_continuous_bc import ContinuousBCPolicy, NumpyContinuousBCLearner
from src.adapters.numpy_continuous_pc import ContinuousPCPolicy, NumpyContinuousPCLearner
from src.app.continuous_stream_rehearsal import ContinuousReplayPlan
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.observation_standardization import ObservationStandardization


BASELINE_PATH = "artifacts/runs/continuous-probe-continuation-20261020/remaining-arms.log"
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
BENCHMARK_DIRECTIONS = dict(
    warm_acquisition_gain=True,
    new_acquisition_gain=True,
    warm_retention_loss=False,
    warm_return_recovery=True,
    warm_return_deficit=False,
    new_retention_loss=False,
)
SETTINGS = dict(
    phase_updates=1,
    phases=["warm", "new", "warm_return"],
    max_training_updates=3,
    promotion_calls=0,
    simulator_calls=0,
    probe_reads=16,
    final_reads=0,
    guard_reads=0,
    probe_plan=dict(
        invocations=4,
        updates_each=2,
        total_training_update_budget=8,
        zero_velocity_control=True,
        inherited_invocations=4,
        zero_velocity_invocations=4,
        updates_per_scope=1,
    ),
)
SOURCES = {
    name: dict(episode_id=episode.episode_id, context=episode.context, role=episode.role)
    for name, episode in (("warm", _episode(False)), ("new", _episode(True)))
}


def _policy(family):
    base = ContinuousBCPolicy(
        _episode(False).spec,
        observation_standardization=ObservationStandardization((1.0, 0.0), (1.0, 1.0), 2),
    )
    if family == "bp":
        return asdict(base)
    chemistry = (
        None
        if family == "pc"
        else CoupledCircadianConfig.matched_pc_control()
        if family == "neutral-cpc"
        else CoupledCircadianConfig()
    )
    return asdict(ContinuousPCPolicy(base, circadian=chemistry))


def _require_equal(actual, expected, reason):
    if _json(actual) != _json(expected):
        raise ValueError(reason)


def _validate_probes(report, native_points, row_points, *, replay):
    points = report["probes"]
    _require_equal(
        [p["benchmark_updates"] for p in points], native_points, "probe native points differ"
    )
    for wake, (point, native, rows) in enumerate(
        zip(points, native_points, row_points, strict=True)
    ):
        if replay and (
            type(point["benchmark_wake_updates"]) is not int
            or point["benchmark_wake_updates"] != wake
        ):
            raise ValueError("probe wake point differs")
        pair = point["report"]
        if pair["complete"] is not True or pair["initial_scores_exact"] is not True:
            raise ValueError("paired fresh probe incomplete")
        _require_equal(
            pair["definition"],
            "continuous_zero_velocity_pair_v1",
            "paired probe definition differs",
        )
        _require_equal(
            pair["inherited"]["checkpoints"][0],
            pair["zero_velocity"]["checkpoints"][0],
            "paired initial physical scores differ",
        )
        _require_equal(
            [pair["source_updates"], pair["source_training_examples"]],
            [native, rows],
            "paired probe source differs",
        )
        for view in ("inherited", "zero_velocity"):
            nested = pair[view]
            _require_completed(
                nested,
                SimpleNamespace(completed_updates=native, training_examples=rows),
                ContinuousProbePlan(
                    _fresh_episode(False),
                    _fresh_episode(True),
                    diagnostics=True,
                    zero_velocity_control=True,
                ),
            )
            if "replay" in nested:
                raise ValueError("fresh probe imported replay authority")
            _require_equal(
                nested["plan"],
                dict(
                    updates=1,
                    training_id="fresh-train",
                    evaluation_id="fresh-test",
                    context="fresh-F",
                    diagnostics=True,
                ),
                "fresh probe recipe differs",
            )
            _require_equal(
                nested["metrics"],
                _metrics(nested["checkpoints"]),
                "probe metrics differ from saved scores",
            )
            work = nested["work"]
            _require_equal(
                [
                    work["training_rows_completed"],
                    work["prediction_returns"],
                    work["prediction_rows_returned"],
                    work["step_diagnostics"]["task_gradient_returns"],
                ],
                [2, 8, 18, 1],
                "probe selected work differs",
            )
            for checkpoint in nested["checkpoints"]:
                scores = checkpoint["scores"]
                _require_equal(
                    {
                        name: scores[name]["role"]
                        for name in ("fresh", "fresh_training", "old_0", "old_1")
                    },
                    dict(
                        fresh="probe_test",
                        fresh_training="probe_train",
                        old_0="train",
                        old_1="train",
                    ),
                    "probe score roles differ",
                )
        _require_equal(
            pair["contrast"],
            _contrast(pair["inherited"], pair["zero_velocity"]),
            "paired contrast differs",
        )


def _validate_report(family, report, *, replay):
    if report["complete"] is not True:
        raise ValueError(f"{family} incomplete report")
    _require_equal(report["policy"], _policy(family), "frozen native policy differs")
    _require_equal(report["settings"], SETTINGS, "frozen stream settings differ")
    _require_equal(report["source_episodes"], SOURCES, "frozen TRAIN sources differ")
    native_points = [0, 2, 4, 6] if replay else [0, 1, 2, 3]
    row_points = [0, 4, 9, 13] if replay else [0, 2, 5, 7]
    _require_equal(
        [p["updates"] for p in report["checkpoints"]],
        native_points,
        "benchmark native points differ",
    )
    _require_equal(
        [p["training_examples"] for p in report["checkpoints"]],
        row_points,
        "benchmark native rows differ",
    )
    if not all(
        p["state_exact"] is True and p["owners_exact"] is True for p in report["checkpoints"]
    ):
        raise ValueError("benchmark owners/readouts incomplete")
    _require_equal(
        report["metrics"],
        _benchmark_metrics(report["checkpoints"]),
        "benchmark metrics differ from checkpoint scores",
    )
    receipts = report["applied_updates"]
    if len(receipts) != 3 or len(report["events"]) != 3:
        raise ValueError("wake receipt sequence differs")
    for index, (receipt, event) in enumerate(zip(receipts, report["events"], strict=True)):
        if event["success_validated"] is not True or event["deferred_exact"] is not True:
            raise ValueError("wake event incomplete")
        _require_equal(
            [
                receipt["update_number"],
                receipt["episode_id"],
                receipt["observed_at"],
                receipt["arrived_at"],
                receipt["applied_at"],
            ],
            [
                index + 1,
                "synthetic-B" if index == 1 else "synthetic-A",
                2 * index,
                2 * index + 1,
                2 * index + 1,
            ],
            "wake receipt provenance differs",
        )
        _require_equal(event["applied"], [receipt], "wake returned receipt differs")
    work = report["work"]
    _require_equal(
        [
            work["optimizer_updates"],
            work["training_rows"],
            work["fit_calls"],
            work["fit_rows"],
            work["fit_feature_values"],
        ],
        [6 if replay else 3, 13 if replay else 7, 1, 2, 4],
        "benchmark selected work differs",
    )
    progress = dict(
        updates_completed=3, replay_examples_completed=0, next_tick=6, update_attempts=3
    )
    if replay:
        progress.update(
            native_updates_completed=6, replay_updates_completed=3, replay_rows_charged=6
        )
        bank = report["replay"]
        _require_equal(
            bank["plan"], asdict(ContinuousReplayPlan(1, 6)), "fixed replay plan differs"
        )
        _require_equal(
            [
                bank["planned_events"],
                bank["bank_rows"],
                bank["retained_numeric_bytes"],
                bank["attempted_updates"],
                bank["returned_updates"],
                bank["committed_updates"],
                bank["charged_exposures"],
            ],
            [3, 2, 112, 3, 3, 3, 6],
            "replay charges/returns differ",
        )
        _require_equal(
            [bank["bank_episode_ids"], bank["bank_contexts"]],
            [["synthetic-A"], ["context-A"]],
            "replay bank source differs",
        )
        if (
            bank["stopped"] is not False
            or bank["charge_count_unknown"] is not False
            or len(report["applied_consolidations"]) != 3
        ):
            raise ValueError("replay publication/authority incomplete")
        _require_equal(
            [work["replay_budget_rows"], work["replay_optimizer_updates"]],
            [6, 3],
            "reported replay work differs",
        )
        if len(bank["events"]) != 3:
            raise ValueError("replay event sequence differs")
        for wake, (event, receipt) in enumerate(
            zip(bank["events"], report["applied_consolidations"], strict=True), 1
        ):
            _require_equal(
                [
                    event["status"],
                    event["wake_updates"],
                    event["attempted_updates"],
                    event["returned_updates"],
                    event["committed_updates"],
                    event["charged_exposures"],
                ],
                ["complete", wake, 1, 1, 1, 2],
                "replay event incomplete",
            )
            _require_equal(event["receipt"], receipt, "replay observed receipt differs")
            _require_equal(
                {
                    key: receipt[key]
                    for key in ("event_id", "actor_version", "learner_version", "attempt_number")
                },
                dict(
                    event_id=f"continuous-replay-{wake}",
                    actor_version="actor-0",
                    learner_version="candidate-0",
                    attempt_number=wake,
                ),
                "replay receipt provenance differs",
            )
        _require_equal(
            [p["wake_updates"] for p in report["checkpoints"]],
            [0, 1, 2, 3],
            "benchmark wake labels differ",
        )
        _require_equal(
            [p["native_updates"] for p in report["checkpoints"]],
            native_points,
            "benchmark native labels differ",
        )
    elif "replay" in report or "applied_consolidations" in report:
        raise ValueError("saved control unexpectedly contains replay")
    _require_equal(report["progress"], progress, "wake/native progress differs")
    if family != "bp":
        native = report["native_updates"]
        _require_equal(
            [item["training_rows"] for item in native],
            [2, 2, 3, 2, 2, 2] if replay else [2, 3, 2],
            "native work rows differ",
        )
        if not all(item["work"]["published_updates"] == 1 for item in native):
            raise ValueError("native update unpublished")
    _validate_probes(report, native_points, row_points, replay=replay)


def _validate_baseline(baseline):
    if baseline["complete"] is not True or baseline["neutral_common_exact"] is not True:
        raise ValueError("accepted saved control cohort incomplete")
    _require_equal(sorted(baseline["reports"]), sorted(FAMILIES), "saved family set differs")
    _require_equal(
        baseline["original_invocation"]["disposition"],
        "FAILED/SPENT",
        "original failed history differs",
    )
    _require_equal(
        baseline["original_invocation"],
        dict(complete=False, disposition="FAILED/SPENT", error=ORIGINAL_ERROR),
        "original failed disposition changed",
    )
    _require_equal(baseline["reused_families"], ["bp"], "saved BP reuse differs")
    _require_equal(
        baseline["returned_new_families"],
        list(FAMILIES[1:]),
        "saved missing-arm continuation differs",
    )
    _require_equal(
        baseline["saved_original_reports_exact"],
        dict.fromkeys(FAMILIES, True),
        "saved original preservation differs",
    )
    _require_equal(
        sorted(baseline["final_states"]), sorted(FAMILIES), "saved final family set differs"
    )
    for family in FAMILIES:
        _validate_report(family, baseline["reports"][family], replay=False)
        final = baseline["final_states"][family]
        _require_equal(
            [final["completed_updates"], final["training_examples"]],
            [3, 7],
            "saved final native counters differ",
        )
        wake = final["wake_state"]
        if family in ("bp", "pc"):
            if wake is not None:
                raise ValueError("saved non-CPC wake differs")
        elif (
            type(wake) is not dict
            or type(wake.get("wake_updates")) is not int
            or wake["wake_updates"] != 3
        ):
            raise ValueError("saved CPC wake counter differs")


def _capture_owner(candidate, report):
    kind = (
        NumpyContinuousBCLearner
        if type(candidate) is NumpyContinuousBCLearner
        else NumpyContinuousPCLearner
    )
    state = _native_state(candidate, kind)
    _require_equal(
        [state.completed_updates, state.training_examples, asdict(state.policy)],
        [6, 13, report["policy"]],
        "returned full native owner differs",
    )
    return state


def _check_owner(candidate, expected):
    _native_state(candidate, type(candidate), expected)


def _neutral_exact(pc, neutral):
    first, second = (
        _native_state(result.candidate, NumpyContinuousPCLearner) for result in (pc, neutral)
    )
    return (
        first.policy.base == second.policy.base
        and all(
            np.array_equal(a, b)
            for a, b in zip(
                first.parameters + first.velocity, second.parameters + second.velocity, strict=True
            )
        )
        and pc.report["metrics"] == neutral.report["metrics"]
        and pc.report["checkpoints"] == neutral.report["checkpoints"]
        and all(
            _common_probe(a["report"][view]) == _common_probe(b["report"][view])
            for a, b in zip(pc.report["probes"], neutral.report["probes"], strict=True)
            for view in ("inherited", "zero_velocity")
        )
    )


def _difference(baseline, replay, higher):
    if any(
        type(value) not in (int, float) or not math.isfinite(value) for value in (baseline, replay)
    ):
        raise ValueError("comparison metric must be finite native numeric")
    return dict(
        baseline=baseline,
        replay=replay,
        replay_minus_baseline=replay - baseline,
        higher_is_better=higher,
    )


def _comparisons(baseline, replay):
    benchmark = {
        name: _difference(baseline["metrics"][name], replay["metrics"][name], higher)
        for name, higher in BENCHMARK_DIRECTIONS.items()
    }
    probes = []
    for wake, (old, new) in enumerate(zip(baseline["probes"], replay["probes"], strict=True)):
        views = {}
        for view in ("inherited", "zero_velocity"):
            old_metrics, new_metrics = (p["report"][view]["metrics"] for p in (old, new))
            views[view] = {
                name: _difference(old_metrics[name], new_metrics[name], True)
                for name in ("fresh_acquisition_gain", "fresh_training_gain")
            }
            views[view]["old_retention_loss"] = {
                name: _difference(
                    old_metrics["old_retention_loss"][name],
                    new_metrics["old_retention_loss"][name],
                    False,
                )
                for name in ("old_0", "old_1")
            }
        probes.append(
            dict(
                wake_updates=wake,
                baseline_native_updates=old["benchmark_updates"],
                replay_native_updates=new["benchmark_updates"],
                views=views,
            )
        )
    return dict(benchmark=benchmark, probes=probes)


def run_study(baseline, *, run_arm):
    """Fixed four-arm admission; observation failures preserve earlier paid work."""
    baseline = deepcopy(baseline)
    payload: dict[str, Any] = dict(
        complete=False,
        scenario="continuous_fixed_train_replay_vs_saved_controls_v1",
        reports={},
        final_states={},
        comparisons={},
        attempted_new_families=[],
        returned_new_families=[],
        neutral_common_exact=False,
        baseline_reused=False,
        simulator_calls=0,
        final_rows=0,
        planned_new_work=dict(
            parameter_updates=56,
            training_rows=116,
            fits=4,
            fit_rows=8,
            fit_values=16,
            initializations=4,
            normal_values=1024,
        ),
        comparison_scope="same wake information/seed/metrics; extra replay optimization/rows/compute; saved cross-artifact controls, no matched compute or benefit threshold",
    )
    returned: dict[str, Any] = {}
    raw_reports: dict[str, Any] = {}
    owners: dict[str, tuple[Any, Any]] = {}

    def check_retained():
        for family, result in returned.items():
            if result.report is not raw_reports[family]:
                raise ValueError(f"{family} preceding paid report identity changed")
        for family, (owner, expected) in owners.items():
            if returned[family].candidate is not owner:
                raise ValueError(f"{family} preceding native owner identity changed")
        # Finish private state/report checks after ALL public result getters.
        for family, raw in raw_reports.items():
            _require_equal(
                raw, payload["reports"][family], f"{family} preceding paid report changed"
            )
        for owner, expected in owners.values():
            _check_owner(owner, expected)

    try:
        _json(baseline)
        _validate_baseline(baseline)
        payload["baseline_reused"] = True
        for family in FAMILIES:
            check_retained()
            payload["attempted_new_families"].append(family)
            result = run_arm(family)
            raw_reports[family] = result.report
            payload["reports"][family] = deepcopy(raw_reports[family])
            returned[family] = result
            payload["returned_new_families"].append(family)
            check_retained()
            report = payload["reports"][family]
            _validate_report(family, report, replay=True)
            original = baseline["reports"][family]
            initial = {
                k: v
                for k, v in report["checkpoints"][0].items()
                if k not in ("native_updates", "wake_updates")
            }
            _require_equal(
                initial, original["checkpoints"][0], "initial checkpoint differs from saved control"
            )
            _require_equal(
                _without_elapsed(report["probes"][0]["report"]),
                _without_elapsed(original["probes"][0]["report"]),
                "initial probe differs from saved control",
            )
            _require_equal(
                report["applied_updates"][0],
                original["applied_updates"][0],
                "first wake differs before replay",
            )
            candidate = result.candidate
            if candidate is None:
                raise ValueError("complete replay report has no detached candidate")
            owners[family] = (candidate, _capture_owner(candidate, report))
            check_retained()
            payload["final_states"][family] = deepcopy(_final_state(candidate))
            check_retained()
            _require_equal(
                [
                    payload["final_states"][family]["completed_updates"],
                    payload["final_states"][family]["training_examples"],
                ],
                [6, 13],
                "final native metadata differs",
            )
            payload["comparisons"][family] = _comparisons(original, report)
            if family == "neutral-cpc":
                exact = _neutral_exact(returned["pc"], result)
                check_retained()
                if exact is not True:
                    raise ValueError("PC/neutral native/common parity differs before active")
                payload["neutral_common_exact"] = True
        check_retained()
        payload["complete"] = True
    except Exception as error:
        payload["error"] = f"{type(error).__name__}: {error}"
    observations = {}
    for family in payload["attempted_new_families"]:
        report = payload["reports"].get(family)
        count = _receipt_observations(report)
        consolidations = report.get("applied_consolidations") if type(report) is dict else None
        observations[family] = (
            count + len(consolidations)
            if type(count) is int and type(consolidations) is list
            else None
        )
    payload["observed_new_work"] = dict(
        retained_applied_receipts_by_family=observations,
        retained_applied_receipts=sum(observations.values())
        if all(type(v) is int for v in observations.values())
        else None,
        unobserved_attempts=[f for f in payload["attempted_new_families"] if f not in returned],
        scope="returned receipt observations only; full paid reports retain work/charges; missing or malformed work is unknown",
    )
    return payload


def _require_producer_environment():
    if (
        sys.version_info[:3] != (3, 14, 7)
        or np.__version__ != "2.4.6"
        or any(os.environ.get(name) != "1" for name in THREADS)
    ):
        raise ValueError(
            "fixed comparison requires original Windows Python3.14.7/NumPy2.4.6 and explicit single-thread environment"
        )


def _run_comparison(baseline, source_label):
    plan = ContinuousProbePlan(
        _fresh_episode(False), _fresh_episode(True), diagnostics=True, zero_velocity_control=True
    )

    def run_arm(family):
        return run_continuous_learnability(
            _episode(False),
            _episode(True),
            phase_updates=1,
            learner_family=family,
            probe_plan=plan,
            replay_plan=ContinuousReplayPlan(1, 6),
        )

    payload = run_study(baseline, run_arm=run_arm)
    payload["baseline_source"] = dict(
        path=source_label,
        original_disposition=baseline["original_invocation"]["disposition"],
        parameter_archive=False,
    )
    payload["environment"] = dict(
        python=sys.version,
        numpy=np.__version__,
        threads={name: os.environ[name] for name in THREADS},
    )
    encoded = json.dumps(payload, allow_nan=False, indent=2)
    if len(encoded.encode("utf-8")) > 6 * 1024**2:
        raise ValueError("comparison output exceeds frozen6MiB allowance")
    print(encoded)
    if payload["complete"] is not True:
        raise SystemExit(1)


MAX_BASELINE_BYTES = 6 * 1024**2


@contextmanager
def _baseline_source(explicit: Path | None, example: bool) -> Iterator[tuple[Path, str]]:
    """Select only the requested source; a missing explicit file never falls back."""
    if explicit is not None:
        yield explicit, str(explicit.resolve())
    elif not example:
        path = Path(__file__).resolve().parents[1] / BASELINE_PATH
        yield path, str(path)
    else:
        label = "package:src.infra.continuous_comparison_data/controls.json"
        local = Path(__file__).resolve().parents[1] / "examples/continuous_comparison/controls.json"
        if local.is_file():
            yield local, label
        else:
            with resources.as_file(
                resources.files("src.infra.continuous_comparison_data").joinpath("controls.json")
            ) as path:
                yield path, label


def _unique_baseline_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate saved-control key: {key}")
        result[key] = value
    return result


def _reject_baseline_constant(value):
    raise ValueError(f"nonfinite saved-control number: {value}")


def _finite_baseline_float(value):
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("nonfinite saved-control number")
    return numeric


def _load_baseline(path: Path) -> dict[str, Any]:
    """Read bounded UTF-8 JSON before admitting any model work."""
    with path.open("rb") as handle:
        payload = handle.read(MAX_BASELINE_BYTES + 1)
    if len(payload) > MAX_BASELINE_BYTES:
        raise ValueError("saved controls exceed6MiB input limit")
    try:
        baseline = json.loads(
            payload.decode("utf8"),
            object_pairs_hook=_unique_baseline_object,
            parse_constant=_reject_baseline_constant,
            parse_float=_finite_baseline_float,
        )
    except (UnicodeError, RecursionError) as error:
        raise ValueError("saved controls must be bounded UTF-8 JSON") from error
    if type(baseline) is not dict:
        raise ValueError("saved controls must be a JSON object")
    return baseline


def _preflight(baseline, source_label):
    # Why this: check the original evidence contract without granting experiment work.
    try:
        _validate_baseline(baseline)
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise ValueError("invalid saved-control structure") from error
    return dict(
        scenario="continuous_saved_control_preflight_v1",
        baseline_validated=True,
        training_executed=False,
        families=list(FAMILIES),
        baseline_source=dict(
            path=source_label,
            original_disposition=baseline["original_invocation"]["disposition"],
            parameter_archive=False,
        ),
        qualification="saved-control consistency only, not source authentication or a completed new study",
        required_execution_profile="Windows Python3.14.7/NumPy2.4.6 and explicit single-thread environment",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--baseline", type=Path, help="explicit saved control JSON; no fallback")
    source.add_argument(
        "--example-baseline", action="store_true", help="bundled original accepted controls"
    )
    parser.add_argument(
        "--validate-baseline-only",
        action="store_true",
        help="check saved controls without training; any compatible CPU Python",
    )
    args = parser.parse_args(argv)
    try:
        if not args.validate_baseline_only:
            _require_producer_environment()
        with _baseline_source(args.baseline, args.example_baseline) as (path, label):
            baseline = _load_baseline(path)
            if args.validate_baseline_only:
                print(json.dumps(_preflight(baseline, label), allow_nan=False, indent=2))
            else:
                _run_comparison(baseline, label)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
