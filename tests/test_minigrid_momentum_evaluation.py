"""Distinct control readout identity keeps existing TRAIN/final boundaries."""

from dataclasses import replace
import pickle
from typing import Any

import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.core.minigrid_key_door import predefined_layouts
from test_minigrid_learning_pilot import evaluator as evaluator
from test_minigrid_warm_reward_evaluation import checkpoint, evaluate


@pytest.mark.parametrize("rows", [1160, 1173, 1280])
def test_should_score_distinct_control_endpoint_without_mutation(evaluator, rows):
    state = checkpoint(evaluator, 136, rows)
    before = pickle.dumps((state, evaluator.source.snapshot_state(), evaluator.sources))
    report = evaluate(evaluator, state, rows=rows, method="recurrent_warm_momentum_control")
    assert report["status"] == "complete" and report["state_unchanged"] is True
    assert report["method"] == "recurrent_warm_momentum_control"
    assert report["native_training_examples"] == rows
    assert len(report["scores"]) == len(report["episodes"]) == 6
    assert pickle.dumps((state, evaluator.source.snapshot_state(), evaluator.sources)) == before
    assert not evaluator.work.get("trains")


@pytest.mark.parametrize(
    "fault", ["warm", "rows", "final", "probe", "subset", "geometry", "history"]
)
def test_should_refuse_invalid_control_identity_role_or_geometry_before_native_work(
    evaluator, fault
):
    state = checkpoint(evaluator)
    options: dict[str, Any] = dict(method="recurrent_warm_momentum_control")
    if fault == "warm":
        options.update(updates=128, rows=1152)
    elif fault == "rows":
        options["rows"] = 1159
    elif fault == "final":
        options.update(layouts=predefined_layouts()[-1:], role="final_test", sources=())
    elif fault == "probe":
        options["role"] = "probe_train"
    elif fault == "subset":
        options.update(layouts=predefined_layouts()[:1], sources=evaluator.sources[:2])
    elif fault == "geometry":
        options["layouts"] = (
            replace(predefined_layouts()[0], rotation=1),
            *predefined_layouts()[1:3],
        )
    else:
        options["public_history"] = True
    with pytest.raises((ValueError, evaluation.MiniGridEvaluationFailed)):
        evaluate(evaluator, state, **options)
    assert not evaluator.work.get("forks") and not evaluator.work.get("environments")
