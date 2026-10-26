"""Existing paired temporal stream to consented replay, with literal frozen inputs."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import scripts.run_clm_temporal_sequence as study
import src.adapters.clm_replay_candidate as adapter
from src.adapters.clm_head_learner import _equal
from test_clm_temporal_sequence import inputs
from test_clm_catalog_learning import parts


@pytest.fixture(scope="module")
def streams():
    import torch

    identity, features, (train, choices) = inputs(torch)
    state, action, scale = parts(torch)
    loaded = SimpleNamespace(
        state_head=state, action_head=action, logit_scale=scale, identity=identity
    )
    original = deepcopy(state.state_dict()), deepcopy(action.state_dict()), scale.detach().clone()
    rng = torch.get_rng_state().clone()
    reports: dict[str, Any] = {}
    holder: dict[str, Any] = {}
    real = adapter.CLMReplayCandidate

    def create(source, **kwargs):
        holder.update(source=source, state=source._capture(), work=source.work)
        candidate = real(source, **kwargs)
        holder["candidate"] = candidate
        return candidate

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(adapter, "CLMReplayCandidate", create)
        for enabled in (False, True):
            report = reports["replay" if enabled else "default"] = {}
            study._run_head_stream(
                loaded,
                train,
                choices,
                report,
                mode="none",
                diagnostics=False,
                before_step=lambda: None,
                save=lambda: None,
                paired_replay_candidate=enabled,
            )
    return dict(
        reports=reports, holder=holder, loaded=loaded, original=original, rng=rng, features=features
    )


def test_should_finish_original_stream_before_fixed_replay_with_original_receipts(streams):
    report = streams["reports"]["replay"]
    child = report["paired_replay_candidate"]
    assert report["sequence_complete"] is True and child["complete"] is True
    assert report["training_replay_consent"] is True
    assert report["stream"]["completed_updates"] == report["learner_work"]["commits"] == 12
    assert child["update_budget_commits"] == child["candidate_work"]["commits"] == 6
    assert child["plan"] == [0, 1, 0, 1, 0, 1]
    assert child["replay_budget_rows"] == child["replay_rows_attempted"] == 18
    assert [v["receipt"] for v in child["origins"]] == [
        report["stream"]["receipts"][4],
        report["stream"]["receipts"][8],
    ]
    assert [v["receipt"]["applied_at"] for v in child["origins"]] == [9, 17]
    assert [v["experience"]["observed_at"] for v in child["origins"]] == [8, 16]
    assert all(v["policy_version"] == "v2" for v in child["origins"])
    assert all(v["experience"]["permissions"].replay for v in child["origins"])
    assert all(v["applied_at"] == 23 for v in child["receipts"])
    assert [v["completed_updates"] for v in child["readouts"]] == [0, 6]
    assert all(len(v["probabilities"]) == 4 for v in child["readouts"])
    assert all(len(row) == 3 for v in child["readouts"] for row in v["probabilities"])


def test_should_preserve_default_trajectory_and_source_state_after_separate_candidate(streams):
    import torch

    default, replay = (streams["reports"][key] for key in ("default", "replay"))
    assert "paired_replay_candidate" not in default and "training_replay_consent" not in default
    for key in ("rows", "stream", "raw_readouts", "frozen_A_probabilities", "learner_work"):
        assert default[key] == replay[key]
    holder, loaded = streams["holder"], streams["loaded"]
    assert _equal(vars(holder["source"]._capture()), vars(holder["state"]))
    assert holder["source"].work == holder["work"]
    assert _equal(loaded.state_head.state_dict(), streams["original"][0])
    assert _equal(loaded.action_head.state_dict(), streams["original"][1])
    assert torch.equal(loaded.logit_scale, streams["original"][2])
    assert torch.equal(torch.get_rng_state(), streams["rng"])
    assert holder["candidate"].identity == loaded.identity
    assert not any(hasattr(holder["candidate"], key) for key in ("fork", "snapshot_state"))


@pytest.mark.parametrize(
    "options",
    [
        dict(catalog_training=True),
        dict(catalog_support="repeat"),
        dict(compare_catalog_support=True),
        dict(reserved_task_probes=True),
        dict(train_evidence_memory=True),
        dict(typed_development_readout=True),
        dict(train_catalog_diagnostics=True),
        dict(probe_rehearsal=True),
        dict(probe_state_only=True),
        dict(probe_marker_control=True),
        dict(train_state_only=True),
        dict(probe_train_readouts=True),
    ],
)
def test_should_refuse_combined_protocol_before_any_loader(options, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("incompatible paired replay reached loader")

    monkeypatch.setattr(study, "load_local_clm", forbidden)
    with pytest.raises(ValueError, match="paired replay"):
        study.run_case(Path("unused"), Path("unused"), {}, paired_replay_candidate=True, **options)


def test_should_dispatch_explicit_cli_option_without_changing_default(tmp_path, monkeypatch):
    import sys

    calls = []

    def execute(*args, **kwargs):
        calls.append(kwargs)
        args[2]["complete"] = True

    monkeypatch.setattr(study, "run_case", execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--model-dir",
            "unused",
            "--head",
            "unused",
            "--output",
            str(tmp_path / "result.json"),
            "--paired-replay-candidate",
        ],
    )
    study.main()
    assert len(calls) == 1 and calls[0]["paired_replay_candidate"] is True


def test_should_refuse_cli_catalog_combination_before_output(tmp_path, monkeypatch):
    import sys

    target = tmp_path / "refused.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--model-dir",
            "unused",
            "--head",
            "unused",
            "--output",
            str(target),
            "--paired-replay-candidate",
            "--catalog-training",
        ],
    )
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2 and not target.exists()
