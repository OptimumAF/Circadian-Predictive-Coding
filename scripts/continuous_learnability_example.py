"""One fixed, small four-family TRAIN A/B/A integration; no benefit threshold.

Run with python -m scripts.continuous_learnability_example. This synthetic example
does not grant authority to open real-data, final/probe or native experiments.
"""

from dataclasses import asdict
import json

import numpy as np

from src.adapters.continuous_learnability_demo import run_continuous_learnability
from src.adapters.continuous_learnability_modes import FAMILIES
from src.adapters.numpy_continuous_pc import NumpyContinuousPCLearner
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)


def _episode(new: bool) -> ContinuousEpisode:
    observations = (
        ((10.0, 1.0), (11.0, 1.0), (12.0, 1.0), (13.0, 1.0))
        if new
        else ((0.0, 0.0), (2.0, 0.0), (4.0, 0.0))
    )
    actions = ((5.0, 1.0), (4.0, 0.5), (3.0, 0.0)) if new else ((-2.0, -1.0), (6.0, 1.0))
    return ContinuousEpisode(
        "synthetic-B" if new else "synthetic-A",
        "context-B" if new else "context-A",
        "train",
        ContinuousActionSpec(2, ("m", "m"), (-2.0, -1.0), (6.0, 1.0)),
        tuple(
            ContinuousTransition(
                index,
                observations[index],
                action,
                0.0,
                observations[index + 1],
                index == len(actions) - 1,
                False,
            )
            for index, action in enumerate(actions)
        ),
    )


def _final_state(learner):
    state = learner.snapshot_state()
    wake = state.wake_state if isinstance(learner, NumpyContinuousPCLearner) else None
    return dict(
        completed_updates=state.completed_updates,
        training_examples=state.training_examples,
        wake_state=(
            dict(
                asdict(wake),
                chemical_state=wake.chemical_state.tolist(),
                plasticity_state=wake.plasticity_state.tolist(),
            )
            if wake is not None
            else None
        ),
    )


def main() -> None:
    warm, new = _episode(False), _episode(True)
    results = {
        family: run_continuous_learnability(warm, new, phase_updates=1, learner_family=family)
        for family in FAMILIES
    }
    for result in results.values():
        report = result.report
        if (
            not report["complete"]
            or result.candidate is None
            or len(report["checkpoints"]) != 4
            or len(report["applied_updates"]) != 3
            or not all(
                item["success_validated"] and item["deferred_exact"] for item in report["events"]
            )
            or not all(row["owners_exact"] and row["state_exact"] for row in report["checkpoints"])
        ):
            raise RuntimeError("fixed continuous family sequence did not complete")
    pc, neutral = results["pc"], results["neutral-cpc"]
    assert isinstance(pc.candidate, NumpyContinuousPCLearner)
    assert isinstance(neutral.candidate, NumpyContinuousPCLearner)
    first, second = pc.candidate.snapshot_state(), neutral.candidate.snapshot_state()
    exact = (
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
    )
    extras = {
        "chemistry_attempts",
        "chemistry_returns",
        "gate_applications_attempted",
        "gate_applications_returned",
    }
    for a, b in zip(pc.report["native_updates"], neutral.report["native_updates"], strict=True):
        common = dict(b, work={key: value for key, value in b["work"].items() if key not in extras})
        exact = exact and a == common and all(b["work"][key] == 1 for key in extras)
    wake = second.wake_state
    if (
        not exact
        or wake is None
        or wake.wake_updates != 3
        or not np.all(wake.plasticity_state == 1)
    ):
        raise RuntimeError("neutral common-state/report continuation differs from PC")
    payload = dict(
        scenario="synthetic_four_family_train_aba_engineering",
        neutral_common_exact=True,
        total_parameter_updates=12,
        total_training_row_exposures=28,
        parameter_initializations=4,
        seeded_normal_values=1024,
        simulator_calls=0,
        final_or_probe_rows=0,
        reports={name: result.report for name, result in results.items()},
        final_states={name: _final_state(result.candidate) for name, result in results.items()},
    )
    encoded = json.dumps(payload, allow_nan=False, indent=2)
    if len(encoded.encode("utf-8")) > 512 * 1024:
        raise RuntimeError("example output exceeds frozen512KiB allowance")
    print(encoded)


if __name__ == "__main__":
    main()
