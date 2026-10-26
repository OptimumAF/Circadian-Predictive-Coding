"""Authentic queue metadata and fixed replay through original stream components."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import recurring_ticket_workflow_example as study
from scripts.ticket_workflow_rehearsal import trajectory_batches, require_progress
from src.core.categorical_perceptron import PerceptronPolicy, PerceptronSnapshot
from src.core.demonstration_episode import (
    EpisodeReplaySchema,
    encode_episode,
    decode_episode,
    episode_arrays,
)


@pytest.fixture(scope="module")
def packet_template():
    return study.prepare_training(context_return=True)


def test_should_compile_owned_authentic_two_ticket_queue_without_changing_training(packet_template):
    spec, original = deepcopy(packet_template)
    batches = trajectory_batches(spec, original, context_return=True)
    schema = EpisodeReplaySchema(9, 6, 6)
    assert schema.encoded_dim == 225
    for i, (old, batch) in enumerate(zip(original, batches, strict=True)):
        episode = batch.episode
        assert episode is not None and episode.role == "train"
        assert episode.context == ("v2-invoice-parcel" if 4 <= i < 8 else "v1-invoice-password")
        features, targets = episode_arrays(episode)
        assert np.array_equal(features, old.features) and np.array_equal(targets, old.targets)
        assert [x.terminal for x in episode.transitions] == [False] * 5 + [True]
        assert [x.reward for x in episode.transitions] == [0, 0, 1, 0, 0, 1]
        assert episode.transitions[2].next_observation == episode.transitions[3].observation
        assert episode.transitions[5].next_observation == (0.0,) * 9
        encoded, label = encode_episode(episode, schema)
        assert encoded.nbytes + label.nbytes == 1808
        assert decode_episode(encoded, label, schema) == episode
        assert batch.row_ids == tuple(f"{episode.episode_id}:step:{j}" for j in range(6))
    assert len({x.episode.episode_id for x in batches}) == 12


@pytest.mark.parametrize(
    "fault", ["role", "context", "features", "targets", "prior_episode", "schema"]
)
def test_should_refuse_foreign_or_wrong_teacher_before_learning(packet_template, fault):
    spec, batches = deepcopy(packet_template)
    first = batches[0]
    if fault == "role":
        first = replace(first, role="final_test")
    elif fault == "context":
        first = replace(first, context="B_v2")
    elif fault == "features":
        first.features[0, 0] += 1
    elif fault == "targets":
        first.targets[1, 0] = 0
    elif fault == "prior_episode":
        first = trajectory_batches(spec, batches, context_return=True)[0]
    else:
        spec = study.DecisionFeatureSpec(study.FEATURES, study.CATALOG)
    with pytest.raises(ValueError):
        trajectory_batches(spec, (first, *batches[1:]), context_return=True)


@pytest.fixture(scope="module", params=["recent", "coverage"])
def replay_case(request):
    captured: dict[str, Any] = {}
    original = study.AttributionStream

    def capture(*a, **kw):
        stream = original(*a, **kw)
        captured.update(stream=stream, helper=kw["rehearsal"])
        return stream

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "AttributionStream", capture)
        report = study.run_example(context_return=True, replay=request.param)
    assert report["complete"], report.get("error")
    return dict(report=report, **captured)


def test_should_charge_same_wake_and_replay_dose_with_original_clocks_and_receipts(replay_case):
    r = replay_case["report"]
    assert r["training_work"] == dict(
        attempted_updates=18,
        attempted_rows=108,
        returned_updates=18,
        unreturned_work_unknown=False,
        budget_committed_updates=12,
    )
    assert [x["source_updates"] for x in r["checkpoints"]] == [0, 4, 8, 12]
    assert [x["state"]["completed_updates"] for x in r["checkpoints"]] == [0, 6, 12, 18]
    assert [x["source_rows"] for x in r["checkpoints"]] == [0, 36, 72, 108]
    assert len(r["applied_updates"]) == 12 and len(r["published_replay_receipts"]) == 6
    assert r["replay_examples"] == r["replay_budget"]["limits"]["max_replay_examples"] == 36
    assert r["budget"]["updates_completed"] == r["budget"]["limits"]["max_training_updates"] == 12
    assert r["replay_budget"]["updates_completed"] == 0
    assert r["queue_semantics"]["advance_after_successful_commit_step"] == 2
    assert r["queue_semantics"]["terminal_step"] == 5
    assert r["source_unchanged"] and r["actor_unchanged"] and r["numpy_rng_unchanged"]
    assert len(r["episodes"]) == 24 and all(x["episode"]["complete"] for x in r["episodes"])
    assert r["summary"] == study.summarize_context_return(r["episodes"])
    assert r["costs"]["attempted_training_affine_terms"] == 108 * 10 * 6
    assert all(r[k] == 0 for k in ("final_examples", "probe_updates", "external_actions"))
    assert len(study._encode_report(r)) <= 512 * 1024


def test_should_retain_only_arrived_complete_train_queues_within_caps(replay_case):
    r = replay_case["report"]
    assert [e["updates"] for e in r["replay"]["events"]] == [2, 4, 6, 8, 10, 12]
    for event, receipt in zip(r["replay"]["events"], r["published_replay_receipts"], strict=True):
        assert event["status"] == "complete" and event["receipt"] == receipt
        assert (
            event["attempted_updates"]
            == event["returned_updates"]
            == event["committed_updates"]
            == 1
        )
        assert event["charged_exposures"] == 6 and len(event["selected_episodes"]) == 1
        assert event["retention"]["example_count"] <= 12
        assert event["retention"]["retained_bytes"] <= 3616
        episode = event["selected_episodes"][0]
        phase, repeat = episode["episode_id"].rsplit("-", 1)
        birth = {"A_v1": 0, "B_v2": 4, "A_return_v1": 8}[phase] + int(repeat) + 1
        assert birth <= event["updates"] and episode["role"] == "train"
        assert len(episode["transitions"]) == 6
    assert r["replay_retention"]["example_count"] <= 12
    assert r["replay_retention"]["retained_bytes"] <= 3616


@pytest.mark.parametrize("fault", ["receipt", "charge", "rows", "foreign_prefix"])
def test_should_refuse_corrupt_published_replay_evidence(replay_case, fault):
    r, helper, stream = replay_case["report"], replay_case["helper"], replay_case["stream"]
    raw = r["checkpoints"][-1]["state"]
    state = PerceptronSnapshot(
        PerceptronPolicy(**raw["policy"]),
        raw["coefficients"],
        raw["completed_updates"],
        raw["training_examples"],
        raw["mistake_updates"],
    )
    previous = tuple(asdict(x) for x in stream.applied_consolidations)
    assert (
        require_progress(
            stream, helper, helper._replay_budget, 12, state, previous, context_return=True
        )
        == previous
    )
    broken = deepcopy(helper.report)
    if fault == "receipt":
        broken["events"][0]["receipt"]["actor_version"] = "foreign"
    elif fault == "charge":
        broken["charged_exposures"] -= 6
    elif fault == "rows":
        state = replace(state, training_examples=107)
    else:
        previous = deepcopy(previous)
        previous[0]["learner_version"] = "foreign"
    with pytest.raises(ValueError):
        require_progress(
            stream,
            SimpleNamespace(report=broken),
            helper._replay_budget,
            12,
            state,
            previous,
            context_return=True,
        )


def test_should_dispatch_explicit_replay_without_changing_context_default(tmp_path, monkeypatch):
    seen = []

    def run(**kw):
        seen.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(study, "run_example", run)
    assert (
        study.main(
            ["--context-return", "--replay", "coverage", "--output", str(tmp_path / "case.json")]
        )
        == 0
    )
    assert seen == [dict(context_return=True, replay="coverage")]
    with pytest.raises(SystemExit):
        study.main(["--context-return", "--method", "bp", "--output", str(tmp_path / "bad.json")])
    assert not (tmp_path / "bad.json").exists()


def test_should_describe_queue_scope_before_fitting(monkeypatch):
    class StopBeforeFit:
        def __init__(self, *a, **kw):
            raise RuntimeError("metadata boundary")

    monkeypatch.setattr(study, "PerceptronLearner", StopBeforeFit)
    r = study.run_example(context_return=True, replay="coverage")
    assert not r["complete"] and "metadata boundary" in r["error"]
    assert r["definition"] == "context_return_queue_rehearsal_v1"
    assert "extra replay exposure is explicit" in r["scope"]
    assert r["queue_semantics"]["terminal_step"] == 5
    assert r["training_work"]["attempted_updates"] == 0


def test_should_preserve_saved_no_replay_case_exactly():
    prior_path = Path("artifacts/runs/context-return-stream-20261023/actual.json")
    assert (
        hashlib.sha256(prior_path.read_bytes()).hexdigest()
        == "581a68d8badbaed81fb6dbf85b78aab3e08798149de86448feb31936e3d7b4b0"
    )
    prior = json.loads(prior_path.read_bytes())
    current = study.run_example(context_return=True)
    assert current["complete"], current.get("error")

    def stable(report):
        r = deepcopy(report)
        for key in ("elapsed_seconds", "sampled_process_rss"):
            r["costs"].pop(key)
        for key in ("started_at", "last_clock"):
            r["budget"].pop(key)
        return json.loads(json.dumps(r))

    assert stable(current) == stable(prior)
