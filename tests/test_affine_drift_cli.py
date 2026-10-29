"""Actual affine CLI/API and stream composition with literal model work only."""

from copy import deepcopy
from dataclasses import replace
from typing import Any
import json

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.core.reference_controller import ReferencePolicy
from src.app.coupled_attribution import AttributionBatch
from src.app.reference_prequential import ReferencePrequential
from src.core.demonstration_episode import episode_arrays
from tests.test_arrived_cue_catalogue import sources
from tests.test_affine_reference_cli import (
    HOOKS,
    WORK,
    LiteralAffine,
    invoke_cli,
    scripted_only as scripted_only,
)


@pytest.fixture(autouse=True)
def allow_explicit_diagnostics(scripted_only, monkeypatch):
    def predict(model, features):
        WORK["predictions"] += 1
        WORK["prediction_rows"] += len(features)
        assert WORK["predictions"] <= 128 and WORK["prediction_rows"] <= 1152
        assert features.shape == (9, 296)
        if "predict" in HOOKS:
            return HOOKS["predict"](model, features)
        return np.full((len(features), 7), 1 / 7, dtype=np.float64)

    monkeypatch.setattr(LiteralAffine, "predict", predict)


@pytest.mark.parametrize("mode", ["no_learning", "periodic", "budget_aware", "drift"])
def test_should_execute_cli_with_exact_explicit_feedback_and_honest_costs(
    monkeypatch, tmp_path, capsys, mode
):
    packets = sources()
    before = deepcopy(packets)
    monkeypatch.setattr(cli, "read_training_sources", lambda path: packets)
    output = tmp_path / "drift.json"
    work_before = dict(WORK)
    invoke_cli(
        monkeypatch,
        [
            "--context-linear-reference",
            mode,
            "--prequential-feedback",
            "--source",
            "literal.json",
            "--output",
            str(output),
        ],
    )
    result = json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out) == result
    assert packets == before and result["status"] == "complete"
    protocol, window = result["protocol"], result["stream"]
    diagnostics = window["prequential"]
    assert protocol == json.loads(
        json.dumps(
            cli.describe_context_linear_reference(ReferencePolicy(mode), prequential_feedback=True)
        )
    )
    assert (
        protocol["diagnostic_forks_upper_bound"]
        == protocol["diagnostic_prediction_calls_upper_bound"]
        == 6
    )
    assert protocol["diagnostic_prediction_rows_upper_bound"] == 54
    assert (
        diagnostics["prediction_calls_attempted"] == diagnostics["prediction_calls_completed"] == 6
    )
    assert diagnostics["prediction_rows_completed"] == 54 and diagnostics["pending_packets"] == 0
    assert WORK["predictions"] - work_before["predictions"] == 6
    assert WORK["prediction_rows"] - work_before["prediction_rows"] == 54
    assert len(diagnostics["events"]) == 6
    assert [(e["observed_at"], e["arrived_at"]) for e in diagnostics["events"]] == list(
        zip(range(6), range(1, 7))
    )
    assert window["completed_updates"] == result["budget"]["updates_completed"]
    assert len(result["committed_receipts"]) == window["completed_updates"]
    assert window["pending_unapplied"] == 6 - window["completed_updates"]
    assert (
        result["source_unchanged"]
        and result["actor_unchanged"]
        and result["receipt_capture_error"] is None
    )
    assert result["endpoints"] == result["unscored_checkpoints"] == []
    assert result["score_calls_completed"] == window["score_forward_calls"] == 0
    assert WORK["finishes"] == work_before["finishes"] and WORK["forbidden"] == 0
    if mode == "drift":
        assert [e["eligible"] for e in diagnostics["events"]] == [True, False] * 3
        assert [r["applied_at"] for r in window["applied_updates"]] == [1, 3, 5]
    if mode == "no_learning":
        assert window["completed_updates"] == 0 and all(
            e["eligible"] for e in diagnostics["events"]
        )


@pytest.mark.parametrize("mode", ["no_learning", "periodic", "budget_aware", "drift"])
def test_should_describe_feedback_bounds_without_payload_model_or_observed_work(
    monkeypatch, capsys, mode
):
    before = dict(WORK)
    monkeypatch.setattr(cli, "read_training_sources", lambda path: pytest.fail("source accessed"))
    invoke_cli(
        monkeypatch, ["--context-linear-reference", mode, "--prequential-feedback", "--describe"]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["prequential_feedback"] is True and result["reference_policy"] == cli.asdict(
        ReferencePolicy(mode)
    )
    assert (
        result["diagnostic_prediction_calls_upper_bound"] == 6
        and result["diagnostic_prediction_rows_upper_bound"] == 54
    )
    assert "prediction_calls_completed" not in result and "prediction_rows_completed" not in result
    assert result["fit_calls_upper_bound"] == 6
    assert WORK == before


@pytest.mark.parametrize("flag", [None, 0, 1, "true"])
def test_should_refuse_nonboolean_api_option_before_source_owner_or_work(flag):
    before = dict(WORK)
    with pytest.raises(ValueError, match="exact bool"):
        cli.run_context_linear_stream(
            object(), reference_policy=ReferencePolicy("drift"), prequential_feedback=flag
        )
    assert WORK == before


def test_should_refuse_feedback_api_without_reference_before_source_owner():
    before = dict(WORK)
    with pytest.raises(ValueError, match="explicit reference"):
        cli.run_context_linear_stream(object(), prequential_feedback=True)
    assert WORK == before


@pytest.mark.parametrize(
    "arguments",
    [
        ["--prequential-feedback", "--describe"],
        ["--context-linear-stream", "--prequential-feedback", "--describe"],
        ["--example-affine-published", "--prequential-feedback", "--describe"],
        ["--context-linear-reference", "drift", "--describe"],
        [
            "--context-linear-reference",
            "drift",
            "--prequential-feedback",
            "--fit-updates",
            "2",
            "--describe",
        ],
        [
            "--context-linear-reference",
            "drift",
            "--prequential-feedback",
            "--retained-source",
            "private.json",
            "--describe",
        ],
    ],
)
def test_should_refuse_cli_conflicts_before_payload_and_delegation(monkeypatch, arguments):
    before = dict(WORK)
    monkeypatch.setattr(cli, "read_training_sources", lambda path: pytest.fail("source accessed"))
    with pytest.raises(SystemExit) as raised:
        invoke_cli(monkeypatch, arguments)
    assert raised.value.code == 2 and WORK == before


def test_should_keep_original_describe_bytes_and_omit_diagnostic_metadata_by_default():
    actual = cli.describe_context_linear_reference(ReferencePolicy("periodic"))
    assert (
        not any(k.startswith("diagnostic_") for k in actual)
        and "prequential_feedback" not in actual
    )
    assert actual["definition"] == "minigrid_context_linear_reference_window_v1"


def test_should_save_failed_prequential_attempt_as_failed_cli_without_free_retry(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(cli, "read_training_sources", lambda path: sources())

    def fail(model, features):
        raise RuntimeError("lost diagnostic return")

    HOOKS["predict"] = fail
    output = tmp_path / "failed.json"
    with pytest.raises(SystemExit) as raised:
        invoke_cli(
            monkeypatch,
            [
                "--context-linear-reference",
                "drift",
                "--prequential-feedback",
                "--source",
                "literal.json",
                "--output",
                str(output),
            ],
        )
    assert raised.value.code == 2
    result = json.loads(output.read_text())
    assert result["status"] == "failed"
    diagnostic = result["stream"]["prequential"]
    assert (
        diagnostic["prediction_calls_attempted"] == 1
        and diagnostic["prediction_calls_completed"] == 0
    )
    assert diagnostic["unreturned_prediction_unknown"]
    assert result["budget"]["updates_completed"] == 0 and result["score_calls_completed"] == 0
    assert "lost diagnostic return" in result["reason"]
    before = dict(WORK)
    with pytest.raises(FileExistsError):
        invoke_cli(
            monkeypatch,
            [
                "--context-linear-reference",
                "drift",
                "--prequential-feedback",
                "--source",
                "literal.json",
                "--output",
                str(output),
            ],
        )
    assert WORK == before


@pytest.mark.parametrize("field", ["context", "row_ids", "features", "targets", "num_classes"])
def test_should_refuse_episode_metadata_mismatch_without_model_work(field):
    episode = sources()[0]
    features, targets = episode_arrays(episode)
    batch = AttributionBatch(
        "train",
        episode.context,
        tuple(f"{episode.episode_id}:step:{t.step}" for t in episode.transitions),
        features,
        targets,
        episode,
        7,
    )
    changes: dict[str, Any] = dict(
        context="foreign",
        row_ids=tuple(reversed(batch.row_ids)),
        features=features + 1.0,
        targets=(targets + 1) % 7,
        num_classes=3,
    )
    before = dict(WORK)
    with pytest.raises(ValueError, match="metadata-matching"):
        ReferencePrequential((replace(batch, **{field: changes[field]}),))
    assert WORK == before
