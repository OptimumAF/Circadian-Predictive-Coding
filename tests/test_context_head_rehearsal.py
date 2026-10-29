"""Literal-source checks; no scientific record reads or new seed selection."""

from copy import deepcopy
from dataclasses import replace
from typing import Any
import json

import numpy as np
import pytest

from scripts import context_head_probe as head
from scripts import context_head_rehearsal as matched
from scripts import probe_ticket_policy_learning as probe
from test_context_head_transfer import literal_source as source_fixture


@pytest.fixture
def literal_source(monkeypatch):
    return source_fixture.__wrapped__(monkeypatch)


@pytest.fixture
def spec():
    return head.DecisionFeatureSpec(head.CONTEXT_FEATURES, head.CATALOG)


def test_should_certify_owned_complete_roles_and_four_original_train_tickets(spec):
    cohorts, bank = matched.prepare_cohorts(spec)
    seen: set[str] = set()
    for group, role, steps in zip(
        cohorts, ("probe_train", "probe_test", "train", "train"), (3, 3, 6, 6), strict=True
    ):
        for batch in group:
            matched._validate_batch(batch, role, seen)
            assert batch.episode.role == role
            assert len(batch.episode.transitions) == steps
            assert sum(t.terminal for t in batch.episode.transitions) == 1
    assert [(b.context, b.role, len(b.transitions)) for b in bank] == [
        (f"{concept}-{version}", "train", 3) for version, concept in probe.TASKS
    ]
    assert 4 * 8 * (matched.SCHEMA.encoded_dim + 1) == 5216
    assert len({b.episode_id for b in bank}) == 4


@pytest.mark.parametrize(
    "fault", ["final_role", "probe_role", "value", "context", "class", "order"]
)
def test_should_refuse_altered_packets_before_model_construction(spec, fault, monkeypatch):
    original = head.build_head_cohorts

    def changed(*a, **kw):
        groups = list(original(*a, **kw))
        b = groups[2][0]
        if fault in ("final_role", "probe_role"):
            b = replace(b, role="final_test" if fault == "final_role" else "probe_train")
        elif fault == "value":
            b.features[0, 0] = 0.5
        elif fault == "context":
            b = replace(b, context="foreign-old")
        elif fault == "class":
            b = replace(b, num_classes=5)
        else:
            b = replace(b, features=b.features[::-1].copy(), targets=b.targets[::-1].copy())
        groups[2] = (b, groups[2][1])
        return tuple(groups)

    monkeypatch.setattr(head, "build_head_cohorts", changed)
    with pytest.raises(ValueError):
        matched.prepare_cohorts(spec)


def test_should_match_native_rows_and_head_exposure_and_preserve_private_source(
    literal_source, monkeypatch
):
    calls = []
    original = matched.ContextPerceptronLearner.train_batch

    def counted(owner, features, targets):
        calls.append((features.copy(), targets.copy()))
        return original(owner, features, targets)

    monkeypatch.setattr(matched.ContextPerceptronLearner, "train_batch", counted)
    result = matched.run_case("unused")
    assert result["complete"], result.get("error")
    assert result["training_work"] == dict(
        attempted_updates=128,
        attempted_rows=480,
        returned_updates=128,
        unreturned_work_unknown=False,
    )
    assert (
        result["source_owners_unchanged"]
        and result["source_bytes_unchanged"]
        and result["numpy_rng_unchanged"]
    )
    assert result["final_examples"] == result["promotions"] == 0
    assert result["fixed_bank_array_bytes"] == 5216
    for offset, row in zip((0, 64), result["cases"], strict=True):
        matched.require_arm(row)
        assert [(p["native_updates"], p["native_rows"]) for p in row["readout_work"]] == [
            (12, 72),
            (60, 216),
            (76, 312),
        ]
        for checkpoint in (0, 12, 24, 48):
            fitted = calls[offset : offset + checkpoint]
            assert sum(len(x) for x, _ in fitted) == 3 * checkpoint
            for column in (4, 5):
                assert (
                    sum(int(np.count_nonzero(x[:, column])) for x, _ in fitted)
                    == 3 * checkpoint // 2
                )
        for fault in ("native", "logical", "limit", "typed", "origin", "replay"):
            bad = deepcopy(row)
            if fault == "native":
                bad["readout_work"][1]["native_updates"] -= 1
            elif fault == "logical":
                bad["probe"]["completed_updates"] = True
            elif fault == "limit":
                bad["sessions"][0]["limits"]["max_training_updates"] += 1
            elif fault == "typed":
                bad["episodes"].pop()
            elif fault == "origin":
                bad["prior_native_rows"] = 0
            elif row["control"] == "old_train_replay":
                bad["probe"]["rehearsal"]["charged_exposures"] -= 1
            else:
                bad["probe"]["rehearsal"] = {}
            with pytest.raises(ValueError):
                matched.require_arm(bad)
    replay = result["cases"][1]["probe"]["rehearsal"]
    assert (
        replay["attempted_updates"]
        == replay["returned_updates"]
        == replay["committed_updates"]
        == 32
    )
    assert all(
        e.role == "train"
        for e in matched.prepare_cohorts(
            head.DecisionFeatureSpec(head.CONTEXT_FEATURES, head.CATALOG)
        )[1]
    )


def test_should_preserve_failed_native_work_and_admit_no_second_control(
    literal_source, monkeypatch
):
    original = matched.ContextPerceptronLearner.train_batch

    def failed(owner, features, targets):
        original(owner, features, targets)
        raise RuntimeError("lost matched native return")

    monkeypatch.setattr(matched.ContextPerceptronLearner, "train_batch", failed)
    result = matched.run_case("unused")
    assert not result["complete"] and len(result["cases"]) == 1
    assert result["training_work"] == dict(
        attempted_updates=1, attempted_rows=3, returned_updates=0, unreturned_work_unknown=True
    )
    assert "lost matched native return" in result["error"]


def test_should_restore_prior_paid_evidence_and_refuse_later_control(literal_source, monkeypatch):
    original = matched.paid_report_guard
    entered = []

    def paid(row):
        check, pin = original(row)
        row["episodes"].clear()
        return check, pin

    original_arm = matched.run_arm

    def arm(*a, **kw):
        entered.append(a[7])
        return original_arm(*a, **kw)

    monkeypatch.setattr(matched, "paid_report_guard", paid)
    monkeypatch.setattr(matched, "run_arm", arm)
    result = matched.run_case("unused")
    assert not result["complete"] and "preceding matched evidence changed" in result["error"]
    assert entered == ["extra_f"]
    assert result["training_work"]["attempted_updates"] == 64
    assert len(result["cases"][0]["episodes"]) == 18
    matched.require_arm(result["cases"][0])


def test_should_keep_original_metadata_free_cohorts_and_refuse_foreign_clock_mode(spec):
    cohorts = head.build_head_cohorts(spec)
    assert all(b.episode is None for group in cohorts for b in group)
    assert [b.context for b in cohorts[1]] == ["legacy-v1", "legacy-v2"]
    with pytest.raises(ValueError):
        probe._observe_tasks(None, "after_probe", None, {}, spec, None, matched_head_exposure=True)


def test_should_keep_replay_charge_after_native_failure(spec, monkeypatch):
    from test_context_catalog_control import initial

    state = initial()
    seed = matched.ContextPerceptronLearner(state.policy)
    seed.restore_state(state)
    original = matched.ContextPerceptronLearner.train_batch
    calls = []

    def failed(owner, features, targets):
        calls.append(len(features))
        answer = original(owner, features, targets)
        if len(calls) == 3:
            raise RuntimeError("lost replay native return")
        return answer

    monkeypatch.setattr(matched.ContextPerceptronLearner, "train_batch", failed)
    cohorts, bank = matched.prepare_cohorts(spec)
    row: dict[str, Any] = {}
    with matched.ProcessRssSampler() as memory:
        with pytest.raises(ValueError, match="incomplete"):
            matched.run_arm(
                seed, state, cohorts, bank, spec, lambda: None, memory, "old_train_replay", row
            )
    assert calls == [3, 3, 3], row["probe"].get("reason")
    assert row["sessions"][2]["replay"] == 3
    assert row["probe"]["rehearsal"]["charged_exposures"] == 3
    assert row["probe"]["rehearsal"]["attempted_updates"] == 1
    assert row["probe"]["rehearsal"]["returned_updates"] == 0
    assert matched._same_value(seed.snapshot_state(), state)


def test_should_refuse_replaced_scientific_source_before_teacher(monkeypatch):
    monkeypatch.setattr(head, "_source_bytes", lambda p: b"{}")
    monkeypatch.setattr(matched, "prepare_cohorts", lambda p: pytest.fail("teacher entered"))
    report = matched.run_case("unused")
    assert not report["complete"] and "source replaced" in report["error"]
    assert report["training_work"]["attempted_updates"] == 0


def test_should_preview_without_source_or_model(tmp_path, monkeypatch):
    monkeypatch.setattr(head, "_source_bytes", lambda p: pytest.fail("source opened"))
    monkeypatch.setattr(
        matched.ContextPerceptronLearner, "__init__", lambda *a: pytest.fail("model created")
    )
    output = tmp_path / "preview.json"
    assert (
        matched.main(
            ["--input", str(tmp_path / "absent.json"), "--output", str(output), "--preview"]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert result["complete"] and not result["source_read"] and not result["model_constructed"]
    assert result["per_arm_native_updates"] == 64 and result["per_arm_rows"] == 240
    assert result["bank_array_bytes"] == 5216


@pytest.mark.parametrize("same", [False, True])
def test_should_refuse_existing_or_source_output_before_work(tmp_path, monkeypatch, same):
    output = tmp_path / "sealed.json"
    output.write_text("preserve")
    monkeypatch.setattr(matched, "run_case", lambda p: pytest.fail("model case entered"))
    with pytest.raises((ValueError, FileExistsError)):
        matched.main(
            ["--input", str(output if same else tmp_path / "input.json"), "--output", str(output)]
        )
    assert output.read_text() == "preserve"
