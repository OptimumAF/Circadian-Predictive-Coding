"""Actual fixed trajectory/replay composition; no new algorithm or native model."""

from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.recurring_ticket_workflow_example as study
from scripts.ticket_workflow_rehearsal import require_progress, trajectory_batches
from src.adapters.perceptron_learner import PerceptronLearner
from src.core.demonstration_episode import (
    EpisodeReplaySchema,
    decode_episode,
    encode_episode,
    episode_arrays,
)


def test_should_compile_authentic_linked_train_episodes_without_changing_native_inputs():
    spec, original = study.prepare_training()
    batches = trajectory_batches(spec, original)
    schema = EpisodeReplaySchema(7, 3, 6)
    assert schema.encoded_dim == 150
    assert [b.context for b in batches] == ["invoice"] * 4 + ["parcel"] * 4 + ["invoice"] * 4
    for old, batch in zip(original, batches, strict=True):
        e = batch.episode
        assert e.role == "train" and e.num_classes == 6 and len(e.transitions) == 3
        assert batch.row_ids == tuple(f"{e.episode_id}:step:{i}" for i in range(3))
        x, y = episode_arrays(e)
        assert np.array_equal(x, old.features) and np.array_equal(y, old.targets)
        assert e.transitions[-1].terminal and e.transitions[-1].reward == 1.0
        assert e.transitions[-1].next_observation == (0.0,) * 7
        encoded, dummy = encode_episode(e, schema)
        assert encoded.nbytes + dummy.nbytes == 1208 and decode_episode(encoded, dummy, schema) == e


@pytest.mark.parametrize("fault", ["role", "context", "target", "features"])
def test_should_refuse_foreign_or_corrupt_teacher_sources_before_training(fault):
    spec, batches = study.prepare_training()
    item = batches[0]
    if fault == "role":
        item = replace(item, role="probe_train")
    elif fault == "context":
        item = replace(item, context="other")
    elif fault == "target":
        item = replace(item, targets=np.full((3, 1), 0.0))
    else:
        item = replace(item, features=np.zeros((3, 7)))
    with pytest.raises(ValueError):
        trajectory_batches(spec, (item, *batches[1:]))


@pytest.fixture(scope="module", params=["recent", "coverage"])
def replay_case(request):
    holder: dict[str, Any] = {}
    original = study.AttributionStream

    def capture(*args, **kwargs):
        stream = original(*args, **kwargs)
        holder.update(stream=stream, helper=kwargs["rehearsal"])
        return stream

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "AttributionStream", capture)
        report = study.run_example(replay=request.param)
    assert report["complete"], report.get("error")
    return dict(report=report, **holder)


def test_should_publish_exact_bounded_replay_work_then_complete_all_task_readouts(replay_case):
    r = replay_case["report"]
    assert len(r["episodes"]) == 11 and all(v["episode"]["complete"] for v in r["episodes"])
    assert r["training_work"] == dict(
        attempted_updates=18,
        attempted_rows=54,
        returned_updates=18,
        unreturned_work_unknown=False,
        budget_committed_updates=12,
    )
    assert [p["source_updates"] for p in r["checkpoints"]] == [0, 4, 8, 12]
    assert [p["state"]["completed_updates"] for p in r["checkpoints"]] == [0, 6, 12, 18]
    assert [p["source_rows"] for p in r["checkpoints"]] == [0, 18, 36, 54]
    replay = r["replay"]
    assert (
        replay["committed_updates"]
        == replay["returned_updates"]
        == replay["attempted_updates"]
        == 6
    )
    assert replay["charged_exposures"] == r["replay_examples"] == 18
    assert r["replay_budget"]["updates_completed"] == 0 and r["budget"]["updates_completed"] == 12
    assert [e["updates"] for e in replay["events"]] == [2, 4, 6, 8, 10, 12]
    assert len(r["published_replay_receipts"]) == 6
    for e in replay["events"]:
        assert e["status"] == "complete" and e["charged_exposures"] == 3
        assert e["retention"]["example_count"] <= 6 and e["retention"]["retained_bytes"] <= 2416
        selected = e["selected_episodes"]
        assert len(selected) == 1 and selected[0]["role"] == "train"
        phase, index = selected[0]["episode_id"].rsplit("-", 1)
        birth = {"A": 0, "B": 4, "A_return": 8}[phase] + int(index) + 1
        assert birth <= e["updates"]
    assert r["replay_retention"]["retained_bytes"] <= 2416
    assert r["numpy_rng_unchanged"] and r["final_examples"] == r["probe_updates"] == 0
    assert r["summary"] == study.summarize(r["episodes"])


def test_should_return_detached_publication_receipts_and_refuse_foreign_prefix_or_counter(
    replay_case,
):
    stream, helper = replay_case["stream"], replay_case["helper"]
    receipts = stream.applied_consolidations
    previous = tuple(asdict(v) for v in receipts)
    state = replay_case["report"]["checkpoints"][-1]["state"]
    from src.core.categorical_perceptron import PerceptronSnapshot, PerceptronPolicy

    native = PerceptronSnapshot(
        PerceptronPolicy(**state["policy"]),
        state["coefficients"],
        state["completed_updates"],
        state["training_examples"],
        state["mistake_updates"],
    )
    assert require_progress(stream, helper, helper._replay_budget, 12, native, previous) == previous
    object.__setattr__(receipts[0], "actor_version", "foreign")
    assert tuple(asdict(v) for v in stream.applied_consolidations) == previous
    foreign = deepcopy(previous)
    foreign[0]["learner_version"] = "foreign"
    with pytest.raises(ValueError, match="prefix changed"):
        require_progress(stream, helper, helper._replay_budget, 12, native, foreign)
    with pytest.raises(ValueError, match="native replay progress"):
        require_progress(
            stream,
            helper,
            helper._replay_budget,
            12,
            replace(native, completed_updates=17),
            previous,
        )
    broken = deepcopy(helper.report)
    broken["events"][0]["receipt"]["actor_version"] = "foreign"
    with pytest.raises(ValueError, match="TRAIN event work"):
        require_progress(
            stream, SimpleNamespace(report=broken), helper._replay_budget, 12, native, previous
        )


def test_should_keep_paid_failed_replay_and_stop_without_renewing_wake_budget(monkeypatch):
    original = PerceptronLearner.train_batch
    calls = 0

    def fail(self, features, targets):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("replay return lost")
        return original(self, features, targets)

    monkeypatch.setattr(PerceptronLearner, "train_batch", fail)
    r = study.run_example(replay="recent")
    assert not r["complete"] and "replay return lost" in r["error"]
    assert (
        r["training_work"]["attempted_updates"] == 3 and r["training_work"]["returned_updates"] == 2
    )
    assert r["training_work"]["unreturned_work_unknown"]
    assert r["budget"]["updates_completed"] == 2 and r["replay_examples"] == 3
    assert r["replay"]["attempted_updates"] == 1 and r["replay"]["committed_updates"] == 0
    assert r["published_replay_receipts"] == [] and len(r["episodes"]) == 2


def test_should_refuse_changed_historical_replay_receipt_before_more_readouts(monkeypatch):
    from src.app.stream_rehearsal import StreamRehearsal

    original = StreamRehearsal.after_update

    def corrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self._observed == 4:
            self._events[0]["receipt"]["actor_version"] = "corrupt"
        return result

    monkeypatch.setattr(StreamRehearsal, "after_update", corrupt)
    r = study.run_example(replay="coverage")
    assert not r["complete"] and "TRAIN event work" in r["error"]
    assert r["training_work"]["returned_updates"] == 6 and len(r["episodes"]) == 2
    assert r["budget"]["updates_completed"] == 4 and r["replay_examples"] == 6


def test_should_pass_explicit_cli_mode_and_keep_default_unchanged(tmp_path, monkeypatch):
    seen = []

    def run(*, replay):
        seen.append(replay)
        return dict(complete=True)

    monkeypatch.setattr(study, "run_example", run)
    assert study.main(["--output", str(tmp_path / "replay.json"), "--replay", "coverage"]) == 0
    assert seen == ["coverage"]


def test_should_refuse_unknown_replay_mode_before_model_work():
    with pytest.raises(ValueError, match="explicit"):
        study.run_example(replay=True)
