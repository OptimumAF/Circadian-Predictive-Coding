"""Legacy raw metadata remains loadable by the archived Linux native case reader."""

from dataclasses import asdict
import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.adapters.numpy_continuous_bc import ContinuousBCPolicy, NumpyContinuousBCLearner
from src.core.continuous_demonstration import ContinuousActionSpec

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="archived native harness uses Linux resource"
)


def make_source(tmp_path):
    spec = ContinuousActionSpec(
        3,
        ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper"),
        (-1.0,) * 4,
        (1.0,) * 4,
    )
    model = NumpyContinuousBCLearner(ContinuousBCPolicy(spec))
    state = model.snapshot_state()
    arrays = {f"p{i}": value for i, value in enumerate(state.parameters)}
    arrays.update({f"v{i}": value for i, value in enumerate(state.velocity)})
    np.savez(tmp_path / "bc-256-state.npz", **arrays)
    metadata = dict(
        complete=True,
        original_owner_exact=True,
        final_total_updates=256,
        final_total_training_rows=7424,
        final_state_file="bc-256-state.npz",
        policy=json.loads(json.dumps(asdict(model.policy))),
    )
    module = importlib.import_module("artifacts.runs.frozen-bc-train-rollout-20261019.native_case")
    return SimpleNamespace(spec=spec), metadata, module


@pytest.mark.parametrize("missing_optional", [True, False])
def test_should_restore_legacy_raw_metadata_without_rewriting_source(
    tmp_path, monkeypatch, missing_optional
):
    episode, metadata, module = make_source(tmp_path)
    if missing_optional:
        del metadata["policy"]["observation_standardization"]
    path = tmp_path / "train-curve.json"
    path.write_text(json.dumps(metadata))
    original = path.read_bytes()
    monkeypatch.setattr(module, "source", tmp_path)
    model = module.restore_final(episode)
    assert model.policy.observation_standardization is None
    assert model.snapshot_state().completed_updates == 256
    assert model.snapshot_state().training_examples == 7424
    assert path.read_bytes() == original


@pytest.mark.parametrize("corruption", ["non_none", "missing_required"])
def test_should_refuse_other_policy_changes_before_array_loading(tmp_path, monkeypatch, corruption):
    episode, metadata, module = make_source(tmp_path)
    if corruption == "non_none":
        metadata["policy"]["observation_standardization"] = {}
    else:
        del metadata["policy"]["hidden_dim"]
    (tmp_path / "train-curve.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(module, "source", tmp_path)
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("invalid metadata reached array loading")
    )
    with pytest.raises(AssertionError):
        module.restore_final(episode)
