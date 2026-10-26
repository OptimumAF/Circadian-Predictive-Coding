"""Finish only the three unexecuted fixed probe arms; reuse accepted paid BP.

This local entry point composes the existing runner, not a new runtime. Its
one-use model allowance must be separately reviewed. Original failed artifacts
remain unchanged. No probe selection, tuning, final/guard access or BP rerun.
"""

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from scripts.continuous_learnability_example import _episode, _final_state
from scripts.continuous_probe_control_example import _inherited_report, _without_elapsed
from scripts.continuous_probe_direction_example import _common_probe
from scripts.continuous_probe_example import _fresh_episode
from src.adapters.continuous_learnability_demo import run_continuous_learnability
from src.adapters.continuous_probe import ContinuousProbePlan
from src.adapters.continuous_probe_control import _contrast, _native_state
from src.adapters.numpy_continuous_pc import NumpyContinuousPCLearner


REMAINING_FAMILIES = ("pc", "neutral-cpc", "active-cpc")
ORIGINAL_ERROR = "RuntimeError: bp original saved benchmark/inherited evidence differs"


def _json(value):
    return json.dumps(value, allow_nan=False, sort_keys=True)


def _metrics(points):
    first, last = (item["scores"] for item in points)
    score = lambda point, name: point[name]["residual"]["overall"]["normalized_half_mse"]
    return dict(
        fresh_acquisition_gain=score(first, "fresh") - score(last, "fresh"),
        fresh_training_gain=score(first, "fresh_training") - score(last, "fresh_training"),
        old_retention_loss={
            name: score(last, name) - score(first, name)
            for name in first
            if name.startswith("old_")
        },
    )


def _validate_pair(pair, checkpoint):
    if (
        pair["complete"] is not True
        or pair["initial_scores_exact"] is not True
        or pair["definition"] != "continuous_zero_velocity_pair_v1"
        or _json([pair["source_updates"], pair["source_training_examples"]])
        != _json([checkpoint, (0, 2, 5, 7)[checkpoint]])
    ):
        raise ValueError("paired scope incomplete")
    preparation = dict(
        source_fork_attempts=1,
        source_fork_returns=1,
        additional_source_owners=1,
        velocity_zero_array_attempts=4,
        velocity_zero_array_returns=4,
        velocity_values_zeroed=322,
        velocity_numeric_bytes_zeroed=2576,
        restore_attempts=1,
        restore_returns=1,
        fits=0,
        parameter_initializations=0,
        seeded_normal_values=0,
        parameter_momentum_numeric_bytes_per_source=5152,
    )
    if any(
        _json(pair["preparation_work"][key]) != _json(value) for key, value in preparation.items()
    ):
        raise ValueError("paired velocity-only preparation differs from frozen work")
    inherited, control = pair["inherited"], pair["zero_velocity"]
    if _json(inherited["checkpoints"][0]) != _json(control["checkpoints"][0]):
        raise ValueError("paired initial physical checkpoint changed")
    for nested in (inherited, control):
        if (
            nested["complete"] is not True
            or nested["receipt_observation_complete"] is not True
            or type(nested["source_updates"]) is not int
            or nested["source_updates"] != checkpoint
            or type(nested["source_training_examples"]) is not int
            or nested["source_training_examples"] != inherited["source_training_examples"]
            or _json(nested["progress"])
            != _json(dict(updates_completed=1, replay_examples_completed=0, next_tick=2))
            or _json(nested["plan"]) != _json(inherited["plan"])
            or len(nested["checkpoints"]) != 2
            or any(point["complete"] is not True for point in nested["checkpoints"])
            or _json([point["updates"] for point in nested["checkpoints"]])
            != _json([checkpoint, checkpoint + 1])
            or len(nested["events"]) != 1
            or nested["events"][0]["success_validated"] is not True
            or nested["events"][0]["deferred_exact"] is not True
        ):
            raise ValueError("paired scope counters/checkpoints/authority incomplete")
        returned = nested["events"][0]["returned_receipts"]
        if len(returned) != 1 or _json(returned) != _json(nested["applied_updates"]):
            raise ValueError("paired returned/paid receipt evidence changed")
        receipt = returned[0]
        native = nested["native_updates"]
        identity = lambda point: {
            name: (value["episode_id"], value["role"]) for name, value in point["scores"].items()
        }
        if _json(identity(nested["checkpoints"][0])) != _json(identity(nested["checkpoints"][1])):
            raise ValueError("paired final score identity/role changed")
        if (
            len(native) != 1
            or type(native[0]["training_rows"]) is not int
            or native[0]["training_rows"] != 2
            or _json(native[0]["diagnostic"]) != _json(receipt["diagnostic"])
            or _json({k: v for k, v in receipt.items() if k != "diagnostic"})
            != _json(
                {k: v for k, v in inherited["applied_updates"][0].items() if k != "diagnostic"}
            )
            or _json(nested["metrics"]) != _json(_metrics(nested["checkpoints"]))
        ):
            raise ValueError("paired native/metric evidence changed")
    if _json(pair["contrast"]) != _json(_contrast(inherited, control)):
        raise ValueError("paired signed contrast changed")


def _validate_arm(family, report, final, reference):
    if report["complete"] is not True:
        raise ValueError(f"{family} incomplete returned benchmark")
    if _json(_inherited_report(report)) != _json(
        _without_elapsed(reference["reports"][family])
    ) or _json(final) != _json(reference["final_states"][family]):
        raise ValueError(f"{family} original inherited/final reference differs")
    points = report["probes"]
    if _json([item["benchmark_updates"] for item in points]) != _json([0, 1, 2, 3]):
        raise ValueError("fixed paired checkpoint sequence differs")
    for checkpoint, point in enumerate(points):
        _validate_pair(point["report"], checkpoint)


def _neutral_exact(pc, neutral):
    first_owner, second_owner = pc.candidate, neutral.candidate
    if (
        type(first_owner) is not NumpyContinuousPCLearner
        or type(second_owner) is not NumpyContinuousPCLearner
    ):
        return False
    first, second = first_owner.snapshot_state(), second_owner.snapshot_state()
    return (
        first.policy.base == second.policy.base
        and first.completed_updates == second.completed_updates == 3
        and first.training_examples == second.training_examples == 7
        and all(
            np.array_equal(a, b)
            for a, b in zip(
                first.parameters + first.velocity, second.parameters + second.velocity, strict=True
            )
        )
        and pc.report["metrics"] == neutral.report["metrics"]
        and pc.report["checkpoints"] == neutral.report["checkpoints"]
        and all(
            _common_probe(a["report"][name]) == _common_probe(b["report"][name])
            for a, b in zip(pc.report["probes"], neutral.report["probes"], strict=True)
            for name in ("inherited", "zero_velocity")
        )
    )


def _capture_owner(candidate, report):
    state = _native_state(candidate, NumpyContinuousPCLearner)
    if (
        state.completed_updates != 3
        or state.training_examples != 7
        or _json(asdict(state.policy)) != _json(report["policy"])
    ):
        raise ValueError("returned native PC owner differs from reported policy/counters")
    return state


def _check_owner(candidate, expected):
    # Native fields avoid a final public observer mutating an earlier owner.
    _native_state(candidate, NumpyContinuousPCLearner, expected)


def _receipt_observations(report):
    """Unknown malformed observations stay unknown; never infer a zero charge."""
    if type(report) is not dict or type(report.get("applied_updates")) is not list:
        return None
    count = len(report["applied_updates"])
    if type(report.get("probes")) is not list:
        return None
    for point in report["probes"]:
        if type(point) is not dict or type(point.get("report")) is not dict:
            return None
        for name in ("inherited", "zero_velocity"):
            nested = point["report"].get(name)
            if nested is None:
                continue
            if type(nested) is not dict or type(nested.get("applied_updates")) is not list:
                return None
            count += len(nested["applied_updates"])
    return count


def continue_study(reference, failed_bp, *, run_arm):
    """Retain paid reports first; admit only the fixed remaining families."""
    reference, failed_bp = deepcopy(reference), deepcopy(failed_bp)
    payload: dict[str, Any] = dict(
        complete=False,
        scenario="continuous_probe_unexecuted_arm_continuation_v1",
        reports={},
        final_states={},
        reused_families=["bp"],
        attempted_new_families=[],
        returned_new_families=[],
        saved_original_reports_exact={},
        neutral_common_exact=False,
        original_invocation=dict(
            complete=False, disposition="FAILED/SPENT", error=failed_bp.get("error")
        ),
        planned_new_work=dict(
            parameter_updates=33,
            training_rows=69,
            fits=3,
            fit_rows=6,
            fit_values=12,
            initializations=3,
            normal_values=768,
        ),
        simulator_calls=0,
        final_rows=0,
        preservation_scope="full original scalar reports/final counters-wake except elapsed and paired settings/container; no old parameter archive",
    )
    returned: dict[str, Any] = {}
    raw_reports: dict[str, Any] = {}
    owners: dict[str, tuple[Any, Any]] = {}

    def check_retained():
        for family, result in returned.items():
            if result.report is not raw_reports[family]:
                raise ValueError(f"{family} preceding returned paid report was replaced")
        for family, (owner, expected) in owners.items():
            if returned[family].candidate is not owner:
                raise ValueError(f"{family} returned candidate identity changed")
        # Finish captured reports/owners only AFTER all public result getters.
        for family, raw in raw_reports.items():
            if _json(raw) != _json(payload["reports"][family]):
                raise ValueError(f"{family} preceding returned paid evidence changed")
        for owner, expected in owners.values():
            _check_owner(owner, expected)

    try:
        _json(reference)
        _json(failed_bp)
        if (
            failed_bp["complete"] is not False
            or failed_bp["error"] != ORIGINAL_ERROR
            or set(failed_bp["reports"]) != {"bp"}
        ):
            raise ValueError("expected the original failed BP-only invocation")
        payload["reports"]["bp"] = failed_bp["reports"]["bp"]
        payload["final_states"]["bp"] = failed_bp["final_states"]["bp"]
        _validate_arm("bp", payload["reports"]["bp"], payload["final_states"]["bp"], reference)
        payload["saved_original_reports_exact"]["bp"] = True
        for family in REMAINING_FAMILIES:
            check_retained()
            payload["attempted_new_families"].append(family)
            result = run_arm(family)
            # The returned report is paid evidence, including incomplete runs.
            raw_reports[family] = result.report
            payload["reports"][family] = deepcopy(raw_reports[family])
            payload["returned_new_families"].append(family)
            returned[family] = result
            check_retained()
            candidate = result.candidate
            if payload["reports"][family]["complete"] is not True or candidate is None:
                raise ValueError(f"{family} incomplete report/candidate")
            owners[family] = (candidate, _capture_owner(candidate, payload["reports"][family]))
            check_retained()
            final = payload["final_states"][family] = deepcopy(_final_state(candidate))
            check_retained()
            _validate_arm(family, payload["reports"][family], final, reference)
            payload["saved_original_reports_exact"][family] = True
            if family == "neutral-cpc":
                neutral_exact = _neutral_exact(returned["pc"], result)
                check_retained()
                if neutral_exact is not True:
                    raise ValueError(
                        "PC/neutral current arrays/common reports differ before active"
                    )
                payload["neutral_common_exact"] = True
        check_retained()
        payload["complete"] = True
    except Exception as error:
        payload["error"] = f"{type(error).__name__}: {error}"
    observations = {
        family: _receipt_observations(report)
        for family, report in payload["reports"].items()
        if family != "bp"
    }
    payload["observed_new_work"] = dict(
        returned_reports=len(returned),
        retained_applied_receipts_by_family=observations,
        retained_applied_receipts=(
            sum(observations.values())
            if all(type(value) is int for value in observations.values())
            else None
        ),
        unobserved_attempts=[
            family for family in payload["attempted_new_families"] if family not in returned
        ],
        scope="retained returned receipt observations, not inferred charges or work from an unreturned call; full reports preserve detailed attempts/counts",
    )
    return payload


def main() -> None:
    if sys.version_info[:3] != (3, 14, 7) or np.__version__ != "2.4.6":
        raise ValueError("fixed study requires original Windows Python3.14.7/NumPy2.4.6")
    if any(
        os.environ.get(name) != "1"
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        )
    ):
        raise ValueError("fixed study requires explicit single-thread numerical environment")
    root = Path(__file__).resolve().parents[1]
    reference_path = root / "artifacts/runs/continuous-probe-direction-20261020/example-run.log"
    bp_path = root / "artifacts/runs/continuous-probe-control-20261020/example-run.log"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    failed_bp, _ = json.JSONDecoder().raw_decode(bp_path.read_text(encoding="utf-8"))
    warm, new = _episode(False), _episode(True)
    plan = ContinuousProbePlan(
        _fresh_episode(False), _fresh_episode(True), diagnostics=True, zero_velocity_control=True
    )

    def run_arm(family):
        return run_continuous_learnability(
            warm, new, phase_updates=1, learner_family=family, probe_plan=plan
        )

    payload = continue_study(reference, failed_bp, run_arm=run_arm)
    payload["sources"] = dict(reference=str(reference_path), reused_failed_bp=str(bp_path))
    payload["environment"] = dict(
        python=sys.version,
        numpy=np.__version__,
        threads={
            name: os.environ[name]
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
    )
    encoded = json.dumps(payload, allow_nan=False, indent=2)
    if len(encoded.encode("utf-8")) > 3 * 1024**2:
        raise ValueError("continuation output exceeds frozen3MiB allowance")
    print(encoded)
    if payload["complete"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
