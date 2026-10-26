"""Fixed opt-in read-only probe fit and native realized-step direction example."""

import json

import numpy as np

from scripts.continuous_learnability_example import _episode, _final_state
from scripts.continuous_probe_example import _fresh_episode
from src.adapters.continuous_learnability_demo import run_continuous_learnability
from src.adapters.continuous_learnability_modes import FAMILIES
from src.adapters.continuous_probe import ContinuousProbePlan
from src.adapters.numpy_continuous_pc import NumpyContinuousPCLearner


def _common_probe(report):
    extras = {
        "chemistry_attempts",
        "chemistry_returns",
        "gate_applications_attempted",
        "gate_applications_returned",
    }
    return dict(
        **{
            key: report[key]
            for key in (
                "complete",
                "source_updates",
                "source_training_examples",
                "plan",
                "events",
                "checkpoints",
                "applied_updates",
                "progress",
                "metrics",
                "update_diagnostics",
            )
        },
        native_updates=[
            dict(
                item, work={key: value for key, value in item["work"].items() if key not in extras}
            )
            for item in report["native_updates"]
        ],
    )


def main() -> None:
    warm, new = _episode(False), _episode(True)
    plan = ContinuousProbePlan(_fresh_episode(False), _fresh_episode(True), diagnostics=True)
    results = {
        family: run_continuous_learnability(
            warm, new, phase_updates=1, learner_family=family, probe_plan=plan
        )
        for family in FAMILIES
    }
    for result in results.values():
        if not result.report["complete"] or result.candidate is None:
            raise RuntimeError(
                "fixed benchmark/probe integration incomplete: "
                + result.report.get("error", "unknown")
            )
        probes = result.report["probes"]
        if (
            len(probes) != 4
            or [item["benchmark_updates"] for item in probes] != [0, 1, 2, 3]
            or not all(
                item["report"]["complete"] and item["report"]["progress"]["updates_completed"] == 1
                for item in probes
            )
        ):
            raise RuntimeError("fixed disposable scopes differ from protocol")
    pc, neutral = results["pc"], results["neutral-cpc"]
    assert isinstance(pc.candidate, NumpyContinuousPCLearner)
    assert isinstance(neutral.candidate, NumpyContinuousPCLearner)
    first, second = pc.candidate.snapshot_state(), neutral.candidate.snapshot_state()
    exact = (
        first.completed_updates == second.completed_updates == 3
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
            _common_probe(a["report"]) == _common_probe(b["report"])
            for a, b in zip(pc.report["probes"], neutral.report["probes"], strict=True)
        )
    )
    if not exact:
        raise RuntimeError("neutral common benchmark/probe continuation differs from PC")
    payload = dict(
        scenario="synthetic_continuous_probe_direction_engineering",
        neutral_common_exact=True,
        benchmark_parameter_updates=12,
        probe_parameter_updates=16,
        total_training_row_exposures=60,
        parameter_initializations=4,
        seeded_normal_values=1024,
        step_direction_calls=16,
        step_direction_training_rows=32,
        simulator_calls=0,
        final_rows=0,
        reports={name: result.report for name, result in results.items()},
        final_states={name: _final_state(result.candidate) for name, result in results.items()},
    )
    encoded = json.dumps(payload, allow_nan=False, indent=2)
    if len(encoded.encode("utf-8")) > 1536 * 1024:
        raise RuntimeError("probe example output exceeds frozen1.5MiB allowance")
    print(encoded)


if __name__ == "__main__":
    main()
