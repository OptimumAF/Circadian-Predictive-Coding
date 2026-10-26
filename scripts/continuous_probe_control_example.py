"""Fixed paired disposable velocity control; no improvement threshold or tuning.

Run once with python -m scripts.continuous_probe_control_example after reviewing
the local protocol. Uses the accepted saved direction example as an immutable
default/inherited reference; it never reruns that earlier experiment.
"""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.continuous_learnability_example import _episode, _final_state
from scripts.continuous_probe_direction_example import _common_probe
from scripts.continuous_probe_example import _fresh_episode
from src.adapters.continuous_learnability_demo import run_continuous_learnability
from src.adapters.continuous_learnability_modes import FAMILIES
from src.adapters.continuous_probe import ContinuousProbePlan


def _without_elapsed(value):
    if isinstance(value, dict):
        return {
            key: _without_elapsed(item) for key, item in value.items() if key != "elapsed_seconds"
        }
    if isinstance(value, (list, tuple)):
        return [_without_elapsed(item) for item in value]
    return value


def _inherited_report(report):
    original = deepcopy(report)
    original["settings"]["probe_reads"] = 8
    original["settings"]["probe_plan"] = dict(
        invocations=4, updates_each=1, total_training_update_budget=4
    )
    for entry in original["probes"]:
        entry["report"] = entry["report"]["inherited"]
    return _without_elapsed(original)


def main() -> None:
    reference_path = Path(__file__).resolve().parents[1] / (
        "artifacts/runs/continuous-probe-direction-20261020/example-run.log"
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    payload: dict[str, Any] = dict(
        complete=False,
        scenario="synthetic_continuous_zero_velocity_pair_engineering",
        reference=str(reference_path),
        reports={},
        final_states={},
        saved_original_reports_exact={},
        planned_maximum_work=dict(
            benchmark_parameter_updates=12,
            probe_parameter_updates=32,
            total_training_row_exposures=92,
            parameter_initializations=4,
            seeded_normal_values=1024,
            step_direction_calls=32,
            step_direction_training_rows=64,
        ),
        simulator_calls=0,
        final_rows=0,
        preservation_scope="full original scalar reports/receipts/readouts/final counters/wake except elapsed and declared paired settings; no archived parameter matrices to compare",
    )
    try:
        warm, new = _episode(False), _episode(True)
        plan = ContinuousProbePlan(
            _fresh_episode(False),
            _fresh_episode(True),
            diagnostics=True,
            zero_velocity_control=True,
        )
        results = {}
        for family in FAMILIES:
            result = run_continuous_learnability(
                warm, new, phase_updates=1, learner_family=family, probe_plan=plan
            )
            payload["reports"][family] = result.report
            if result.report["complete"] is not True or result.candidate is None:
                raise RuntimeError(f"{family} paired integration incomplete")
            results[family] = result
            final = payload["final_states"][family] = _final_state(result.candidate)
            exact = (
                _inherited_report(result.report) == _without_elapsed(reference["reports"][family])
                and final == reference["final_states"][family]
            )
            payload["saved_original_reports_exact"][family] = exact
            if not exact:
                raise RuntimeError(f"{family} original saved benchmark/inherited evidence differs")
            pairs = result.report["probes"]
            if [item["benchmark_updates"] for item in pairs] != [0, 1, 2, 3] or not all(
                item["report"]["complete"] is True
                and item["report"]["initial_scores_exact"] is True
                for item in pairs
            ):
                raise RuntimeError(f"{family} paired scopes differ from fixed sequence")
        pc, neutral = results["pc"], results["neutral-cpc"]
        assert pc.candidate is not None and neutral.candidate is not None
        first, second = pc.candidate.snapshot_state(), neutral.candidate.snapshot_state()
        exact = (
            all(
                np.array_equal(a, b)
                for a, b in zip(
                    first.parameters + first.velocity,
                    second.parameters + second.velocity,
                    strict=True,
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
        payload["neutral_common_exact"] = exact
        if not exact:
            raise RuntimeError("PC and neutral paired common continuation differs")
        payload["complete"] = True
    except Exception as error:
        payload["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        encoded = json.dumps(payload, allow_nan=False, indent=2)
        if len(encoded.encode("utf-8")) > 3 * 1024 * 1024:
            raise RuntimeError("paired example output exceeds frozen3MiB allowance")
        print(encoded)


if __name__ == "__main__":
    main()
