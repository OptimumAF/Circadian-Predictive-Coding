"""Public TorchVersion/JSON boundary only; no learner, encoding or fit work."""

import json

import pytest

from test_clm_paired_replay_comparison import comparison_module


def source_record():
    from torch.torch_version import TorchVersion

    return dict(
        rows=[],
        stream=dict(completed_updates=12),
        raw_readouts=[],
        frozen_A_probabilities=[],
        learner_work=dict(commits=12),
        identity=dict(embedding_dim=4096),
        torch_version=TorchVersion("2.14.0+cpu"),
        control_probabilities={},
    )


def test_should_compare_torch_public_version_with_its_exact_saved_json_text():
    actual = source_record()
    saved = json.loads(comparison_module.encoded(actual))
    assert type(actual["torch_version"]) is not str and type(saved["torch_version"]) is str
    assert comparison_module.require_source_match(actual, saved) == 0.0


@pytest.mark.parametrize("fault", ["version", "version_type", "counter", "counter_type"])
def test_should_still_refuse_real_version_and_counter_changes(fault):
    actual = source_record()
    saved = json.loads(comparison_module.encoded(actual))
    if fault == "version":
        actual["torch_version"] = "different"
    elif fault == "version_type":
        actual["torch_version"] = 2.14
    elif fault == "counter":
        actual["learner_work"]["commits"] = 13
    else:
        actual["learner_work"]["commits"] = 12.0
    with pytest.raises(ValueError, match="reference differs"):
        comparison_module.require_source_match(actual, saved)
