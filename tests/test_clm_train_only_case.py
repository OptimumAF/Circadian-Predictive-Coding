"""Native tiny integrations for the frozen TRAIN-only artifact, no pretrained assets."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from src.adapters.clm_frozen_encoder import text_recipe
from src.core.clm_head import FrozenEncoderIdentity
from test_clm_catalog_learning import parts
from test_clm_frozen_encoder import FakeTokenizer, backbone
from test_clm_ticket_workflow import PublicTextTokens

CASE = (
    Path(__file__).resolve().parents[1] / "artifacts/runs/clm-train32-native-20261025/train_only.py"
)


def execute(*, fail=False):
    spec = importlib.util.spec_from_file_location("train_only_qualification", CASE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state, action, scale = parts(torch)
    state.eval().requires_grad_(False)
    action.eval().requires_grad_(False)
    loaded = SimpleNamespace(
        state_head=state,
        action_head=action,
        logit_scale=scale.detach(),
        model=backbone(torch),
        tokenizer=FakeTokenizer(PublicTextTokens(dummy=1)),
        identity=FrozenEncoderIdentity(
            "literal", "train32", "fixture", "last", text_recipe(2048), 2
        ),
    )
    report: dict[str, Any] = {}
    failed = False

    def save():
        nonlocal failed
        if fail and report.get("training", {}).get("committed_updates") == 1 and not failed:
            failed = True
            raise OSError("controlled first paid save failure")

    rng = torch.get_rng_state().clone()
    if fail:
        with pytest.raises(OSError, match="first paid save"):
            module.run_loaded(loaded, report, save=save)
    else:
        module.run_loaded(loaded, report, save=save)
    assert torch.equal(torch.get_rng_state(), rng)
    assert report["loaded_heads_and_caller_rng_preserved"] is True
    assert report["encoder_work"]["forward_returns"] == 1
    assert report["encoder_work"]["sequences_returned"] == 11
    assert "episodes" not in report and "binding_work" not in report
    assert all(c["role"] == "train" for c in report["training_data"]["cases"])
    return report


def test_should_complete_original_train_only_thirty_two_passes_and_guarded_reads():
    report = execute()
    assert report["complete"]
    paid = report["training"]
    assert paid["complete"] and paid["requested_passes"] == paid["committed_updates"] == 32
    assert paid["logical_clock"] == 63 and len(paid["receipts"]) == 32
    assert [r["applied_at"] for r in paid["receipts"]] == list(range(1, 64, 2))
    assert paid["permissions"] == dict(training=True, replay=False, evaluation=False)
    assert [row["completed_updates"] for row in report["train_fit"]["readouts"]] == [0, 32]
    for row in report["train_fit"]["readouts"]:
        assert row["complete"] and row["returned"] and row["validated"] and row["source_preserved"]
        assert len(row["rows"]) == len(row["probabilities"]) == 12
        assert all(len(p) == 5 for p in row["probabilities"])
        assert [r["concept"] for r in row["rows"] if r["phase"] == "ready"] == [
            "invoice",
            "legacy",
            "parcel",
            "password",
        ]
    work = report["source_work"]
    assert work["commits"] == work["optimizer_returns"] == work["scheduler_returns"] == 32
    assert work["training_rows_attempted"] == 384
    assert work["catalog_candidate_rows_attempted"] == 160
    assert work["catalog_logit_values_attempted"] == 1920
    assert work["prediction_returns"] == 2 and work["prediction_rows_attempted"] == 24
    assert work["restore_attempts"] == work["snapshot_attempts"] == 0
    assert work["unreturned_work_unknown"] is False


def test_should_stop_after_paid_save_failure_without_more_training_or_after_read():
    report = execute(fail=True)
    assert report["complete"] is False and report["training"]["complete"] is False
    assert report["source_work"]["commits"] == report["source_work"]["optimizer_returns"] == 1
    assert report["source_work"]["training_rows_attempted"] == 12
    assert report["preserved_training_prefix"]["committed_updates"] == 1
    assert len(report["preserved_training_prefix"]["receipts"]) == 1
    assert len(report["train_fit"]["readouts"]) == 1
    assert report["train_fit"]["readouts"][0]["point"] == "before"
    assert report["source_work"]["prediction_returns"] == 1
