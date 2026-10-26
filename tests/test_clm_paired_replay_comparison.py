"""Fixed equal-dose consumer on the actual tiny temporal source/candidate ports."""

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import scripts.run_clm_temporal_sequence as study
import src.app.clm_replay_candidate as candidate_app
from test_clm_temporal_sequence import inputs
from test_clm_catalog_learning import parts


path = (
    Path(__file__).resolve().parents[1]
    / "artifacts/runs/clm-paired-replay-comparison-20261022/comparison.py"
)
spec = importlib.util.spec_from_file_location("paired_comparison", path)
assert spec is not None and spec.loader is not None
comparison_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison_module)


@pytest.fixture(scope="module")
def completed():
    import torch

    identity, features, (train, choices) = inputs(torch)
    state, action, scale = parts(torch)
    loaded = SimpleNamespace(
        state_head=state, action_head=action, logit_scale=scale, identity=identity
    )
    report: dict[str, Any] = dict(identity=asdict(identity), torch_version=torch.__version__)
    comparison: dict[str, Any] = {}
    observations = []
    rng = torch.get_rng_state().clone()
    reference: dict[str, Any] = {}

    def consume(source, history, plan, readout, arm, **options):
        # Match the saved native reference boundary, including public TorchVersion.
        reference.update(json.loads(comparison_module.encoded(report)))
        return comparison_module.run_pair(
            source,
            history,
            plan,
            readout,
            arm,
            source_report=report,
            reference=reference,
            comparison=comparison,
            **options,
        )

    def save():
        assert "scores" not in comparison
        if "arms" in comparison:
            observations.append(
                {
                    key: arm.get("update_budget_commits", 0)
                    for key, arm in comparison["arms"].items()
                }
            )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(candidate_app, "run_clm_replay_candidate", consume)
        study._run_head_stream(
            loaded,
            train,
            choices,
            report,
            mode="none",
            diagnostics=False,
            before_step=lambda: None,
            save=save,
            paired_replay_candidate=True,
        )
    original = comparison_module.encoded(comparison)
    scores = comparison_module.score_pair(comparison)
    assert comparison_module.encoded(comparison) == original
    assert torch.equal(torch.get_rng_state(), rng)
    return report, reference, comparison, scores, observations


def test_should_use_same_original_source_with_equal_dose_and_different_task_mix(completed):
    from torch.torch_version import TorchVersion

    report, reference, comparison, scores, observations = completed
    assert type(report["torch_version"]) is TorchVersion
    assert type(reference["torch_version"]) is str
    assert str(report["torch_version"]) == reference["torch_version"]
    assert comparison["complete"] and comparison["source_max_abs_error"] == 0.0
    assert comparison_module.require_source_match(report, reference) == 0.0
    replay, control = (comparison["arms"][name] for name in ("replay", "return_only"))
    assert replay is report["paired_replay_candidate"]
    assert replay["plan"] == [0, 1, 0, 1, 0, 1] and control["plan"] == [1] * 6
    assert replay["origins"] == control["origins"]
    for arm in (replay, control):
        assert arm["complete"] and arm["update_budget_commits"] == 6
        assert arm["replay_rows_attempted"] == arm["replay_budget_rows"] == 18
        assert arm["candidate_work"]["commits"] == arm["candidate_work"]["backward_returns"] == 6
        assert arm["candidate_work"]["prediction_rows_attempted"] == 8
        assert arm["readouts"][0]["probabilities"] == report["raw_readouts"][-1]["probabilities"]
    assert replay["candidate_work"] == control["candidate_work"]
    assert all(row.get("replay", 0) == 6 for row in observations if "return_only" in row)
    assert scores["cached_typed_decodes"] == 32
    assert set(scores["arms"]) == {"replay", "return_only"}
    assert report["stream"]["completed_updates"] == report["learner_work"]["commits"] == 12


@pytest.mark.parametrize("fault", ["loss", "clock", "probability", "work"])
def test_should_reject_changed_saved_source_before_candidate_work(completed, fault):
    report, reference, *_ = completed
    changed = deepcopy(reference)
    if fault == "loss":
        changed["rows"][0]["loss"] += 0.01
    elif fault == "clock":
        changed["stream"]["receipts"][0]["arrived_at"] += 1
    elif fault == "probability":
        changed["raw_readouts"][-1]["probabilities"][0][0] += 0.01
    else:
        changed["learner_work"]["commits"] += 1
    with pytest.raises(ValueError, match="reference differs"):
        comparison_module.require_source_match(report, changed)


def test_should_refuse_scoring_unfinished_fixed_work(completed):
    _, _, comparison, *_ = completed
    changed = deepcopy(comparison)
    changed["arms"]["return_only"]["complete"] = False
    with pytest.raises(ValueError, match="fixed paid"):
        comparison_module.score_pair(changed)
    changed["complete"] = False
    with pytest.raises(ValueError, match="complete fixed"):
        comparison_module.score_pair(changed)
