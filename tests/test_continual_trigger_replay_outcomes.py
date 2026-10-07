"""V14 outcomes open common final roles only after the six-trial gate."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts.run_continual_trigger_replay_outcomes import build_payload, main
from src.app import continual_trigger_replay_outcomes as outcomes
from src.app.continual_trigger_replay_schedule import fixed_trigger_replay_manifest
from src.app.continual_trigger_replay_training_study import run_trigger_replay_training_study


def test_should_score_every_fixed_cell_and_keep_common_final_roles() -> None:
    result = outcomes.run_trigger_replay_outcomes(fixed_trigger_replay_manifest())
    assert len(result.outcomes) == 6
    assert len(result.contrasts) == 18
    for seed in (47, 53):
        rows = [row for row in result.outcomes if row.seed == seed]
        assert [row.arm for row in rows] == ["periodic", "adaptive", "no_sleep"]
        assert len({row.final_role_hashes for row in rows}) == 1
        assert len({row.final_role_ids for row in rows}) == 1
        assert [row.sleep_accepted for row in rows] == [6, 0, 0]
        assert [row.sleep_attempts for row in rows] == [6, 0, 0]
        assert [row.sleep_rolled_back for row in rows] == [0, 0, 0]
        assert [row.circadian_replay_exposure.replay_updates for row in rows] == [12, 0, 0]
        assert len(rows[0].methods) == 3
        for row in rows:
            for method in row.methods:
                assert all(
                    0.0 <= value <= 1.0
                    for value in (
                        method.a_after_a_accuracy,
                        method.a_after_b_accuracy,
                        method.b_after_b_accuracy,
                        method.balanced_score,
                    )
                )
                assert all(
                    isfinite(value) and value >= 0.0
                    for value in (
                        method.a_after_a_bce,
                        method.a_after_b_bce,
                        method.b_after_b_bce,
                    )
                )
                assert method.signed_forgetting == (
                    method.a_after_a_accuracy - method.a_after_b_accuracy
                )
                assert method.balanced_score == 0.5 * (
                    method.a_after_b_accuracy + method.b_after_b_accuracy
                )
                assert method.wake_work.optimizer_updates == 24
                assert method.wake_work.examples == 1296
                assert method.replay_work.optimizer_updates == (12 if row.arm == "periodic" else 0)
        assert rows[1].methods == rows[2].methods
    assert all(
        contrast.a_after_b_accuracy_delta == 0.0
        and contrast.b_after_b_accuracy_delta == 0.0
        and contrast.replay_updates_delta == 0
        for contrast in result.contrasts
        if contrast.left_arm == "adaptive" and contrast.right_arm == "no_sleep"
    )


def test_should_stop_forged_trial_before_any_final_read(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = fixed_trigger_replay_manifest()
    study = run_trigger_replay_training_study(manifest)
    first = study.trials[0]
    offered = first.opportunities[0]
    forged = replace(offered, selected_ids=("f" * 64, *offered.selected_ids[1:]))
    trials = list(study.trials)
    trials[0] = replace(first, opportunities=(forged, *first.opportunities[1:]))
    monkeypatch.setattr(
        outcomes,
        "run_trigger_replay_training_study",
        lambda _: replace(study, trials=tuple(trials)),
    )
    monkeypatch.setattr(
        outcomes,
        "release_final_test",
        lambda *_: (_ for _ in ()).throw(AssertionError("final read before preflight")),
    )
    with pytest.raises(ValueError, match="train-only opportunity"):
        outcomes.run_trigger_replay_outcomes(manifest)


def test_should_release_all_final_roles_before_first_score(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = fixed_trigger_replay_manifest()
    original_release = outcomes.release_final_test
    original_score = outcomes._score
    releases: list[int] = []
    scores: list[int] = []

    def release(roles: Any) -> Any:
        releases.append(1)
        return original_release(roles)

    def score(model: Any, batch: Any) -> Any:
        scores.append(len(releases))
        return original_score(model, batch)

    monkeypatch.setattr(outcomes, "release_final_test", release)
    monkeypatch.setattr(outcomes, "_score", score)
    outcomes.run_trigger_replay_outcomes(manifest)
    assert len(releases) == 12
    assert len(scores) == 54
    assert set(scores) == {12}


def test_should_reject_mismatched_final_role_before_any_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_release = outcomes.release_final_test
    releases: list[int] = []

    def release(roles: Any) -> Any:
        bound = original_release(roles)
        releases.append(1)
        if len(releases) == 3:
            hashes = dict(bound.split_hashes)
            hashes["final_test"] = "f" * 64
            return replace(bound, split_hashes=hashes)
        return bound

    monkeypatch.setattr(outcomes, "release_final_test", release)
    monkeypatch.setattr(
        outcomes,
        "_score",
        lambda *_: (_ for _ in ()).throw(AssertionError("scored before final identity gate")),
    )
    with pytest.raises(ValueError, match="common final roles differ"):
        outcomes.run_trigger_replay_outcomes(fixed_trigger_replay_manifest())
    assert len(releases) == 12


def test_should_repeat_every_scored_row_and_contrast_exactly() -> None:
    first = build_payload()
    assert first == build_payload()
    parsed = json.loads(first)
    assert len(parsed["outcomes"]) == 6
    assert len(parsed["contrasts"]) == 18
    assert all(len(row["methods"]) == 3 for row in parsed["outcomes"])


def test_should_hash_actual_scored_artifact_bytes_and_refuse_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = tmp_path / "outcomes.json"
    monkeypatch.setattr(sys, "argv", ["outcomes", "--result", str(result)])
    main()
    reported = json.loads(capsys.readouterr().out)
    assert result.read_bytes() == build_payload().encode("utf-8")
    assert reported["sha256"] == sha256(result.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        main()
