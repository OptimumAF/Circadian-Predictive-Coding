"""Exploratory E/known-TRAIN-A phases through real refusal/FIFO native probes.

Engineering arrays and a constructed mature table verify composition and accounting;
they are not native MiniGrid performance, generalization or scientific evidence.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_action_memory_probes as pilot
from src.adapters.fifo_action_memory_learner import FifoActionMemoryLearner
from src.core.action_memory import ActionMemoryEntry, ActionMemoryPolicy, ActionMemorySnapshot
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    episode_arrays,
)
from src.core.minigrid_public_history import encode_public_history_episode
from test_minigrid_action_memory_probes import assay as assay
from test_torch_feature_rehearsal_pilot import forbidden


METHODS = ("observation_memory", "observation_memory_fifo")


def a_frame(cue: str, step: int) -> tuple[float, ...]:
    return (-0.5 if cue == "blue" else 0.5,) + (0.0,) * 146 + (step / 16,)


def a_episode(cue: str) -> DemonstrationEpisode:
    return DemonstrationEpisode(
        "minigrid-source-A-" + cue,
        "A",
        "train",
        tuple(
            DemonstrationTransition(
                step,
                a_frame(cue, step),
                (step + 2) % 7,
                0.5 if step == 8 else 0.0,
                a_frame(cue, step + 1),
                step == 8,
                7,
            )
            for step in range(9)
        ),
        7,
    )


def inputs():
    original_a = tuple(a_episode(cue) for cue in ("blue", "red"))
    entries = []
    for episode in original_a:
        x, y = episode_arrays(encode_public_history_episode(episode))
        for row, label in zip(x, y[:, 0], strict=True):
            counts = [0] * 7
            counts[int(label)] = 22
            entries.append(ActionMemoryEntry(tuple(float(v) for v in row), tuple(counts)))
    for index in range(36):
        entries.append(
            ActionMemoryEntry((0.75,) + (0.0,) * 294 + (index / 128,), (21, 0, 0, 0, 0, 0, 0))
        )
    policy = ActionMemoryPolicy(feature_dim=296, num_classes=7, max_payload_bytes=155136)
    return (
        ActionMemorySnapshot(deepcopy(policy), (), 0, 0),
        ActionMemorySnapshot(deepcopy(policy), tuple(entries), 128, 1152),
    ), original_a


@pytest.fixture
def returning(assay, monkeypatch):
    work = assay
    work["inputs"], work["return_sources"] = inputs()
    original_train = FifoActionMemoryLearner.train_batch
    original_predict = FifoActionMemoryLearner.predict
    original_fork = FifoActionMemoryLearner.fork

    def fork(self):
        child = original_fork(self)
        work["children"].append(child)
        return child

    def train(self, features, targets):
        before = self.snapshot_state()
        work["attempts"].append((self, before, features.copy(), targets.copy()))
        if work.get("fail_attempt") == len(work["attempts"]):
            raise RuntimeError("unexpected FIFO update failure")
        result = original_train(self, features, targets)
        work["trains"].append(
            (self, before, self.snapshot_state(), features.copy(), targets.copy())
        )
        callback = work["on_train"]
        if callback is not None:
            callback(self)
        return result

    def predict(self, features):
        result = original_predict(self, features)
        work["predictions"].append((self, self.snapshot_state(), features.copy(), result.copy()))
        return result

    monkeypatch.setattr(FifoActionMemoryLearner, "fork", fork)
    monkeypatch.setattr(FifoActionMemoryLearner, "train_batch", train)
    monkeypatch.setattr(FifoActionMemoryLearner, "predict", predict)
    return work


def run(work):
    return pilot.run_return_pilot(work["inputs"], work["return_sources"])


def test_should_describe_return_comparison_without_modifying_the_old_protocol_or_allocating_work(
    monkeypatch,
):
    old = pilot.describe_protocol()
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(pilot, "ToyBudgetSession", forbidden)
    monkeypatch.setattr(pilot.reserved, "_collect_probe_sources", forbidden)
    descriptor = pilot.describe_return_protocol()
    assert descriptor == pilot.describe_return_protocol() and pilot.describe_protocol() == old
    text = str(descriptor)
    assert all(method in text for method in METHODS)
    assert descriptor["checkpoint_updates"] == [0, 4, 8, 12, 16]
    assert descriptor["planned_work"]["maximum_storage_updates"] == 64
    assert descriptor["planned_work"]["maximum_training_rows"] == 576
    assert descriptor["planned_work"]["maximum_score_calls"] == 80
    assert descriptor["planned_work"]["maximum_score_rows"] == 720
    assert "known" in text.lower()


def test_should_run_four_owned_matched_phases_and_continue_only_the_accounted_capacity_negative(
    returning,
):
    work = returning
    before = pickle.dumps((work["inputs"], work["return_sources"]))
    result = run(work)
    assert result["status"] == "incomplete"
    rows = result["cohorts"]
    assert [(row["source_updates"], row["method"]) for row in rows] == [
        (0, METHODS[0]),
        (0, METHODS[1]),
        (128, METHODS[0]),
        (128, METHODS[1]),
    ]
    assert [row["status"] for row in rows] == ["complete", "complete", "failed", "complete"]
    assert [row["charged_updates"] for row in rows] == [16, 16, 1, 16]
    assert [row["report"]["attempted_updates"] for row in rows] == [16, 16, 2, 16]
    assert [row["report"]["score_forward_calls_completed"] for row in rows] == [20, 20, 4, 20]
    assert all(
        row["source_unchanged"] is True and row["budget_binding_preserved"] is True for row in rows
    )
    assert [row["fifo_retirement_metadata_bytes"] for row in rows] == [0, 16, 0, 16]
    assert len(work["helpers"]) == len(work["budgets"]) == len(work["children"]) == 4
    assert [b.updates_completed for b in work["budgets"]] == [16, 16, 1, 16]
    assert all(b.budget.max_training_updates == 16 and b.started_at == 1.0 for b in work["budgets"])
    assert (
        len(work["trains"]) == 49 and len(work["attempts"]) == 50 and len(work["predictions"]) == 64
    )
    assert pickle.dumps((work["inputs"], work["return_sources"])) == before
    assert result["return_sources"] == [asdict(episode) for episode in work["return_sources"]]
    assert result["retained_input_array_bytes"] == 642240 and len(result["probe_data"]) == 20
    assert result["work"]["benchmark_updates"] == 0
    assert (
        result["work"]["reused_TRAIN_A_episodes"] == 2
        and result["work"]["reused_TRAIN_A_new_physics_steps"] == 0
    )
    assert (
        result["work"]["teacher_episodes_observed"] == 4
        and result["work"]["teacher_steps_returned"] == 36
    )
    assert (
        result["work"]["completed_storage_updates"] == 49
        and result["work"]["attempted_storage_updates"] == 50
    )
    assert (
        result["work"]["completed_training_rows"] == 441
        and result["work"]["score_rows_completed"] == 576
    )
    for index, helper in enumerate(work["helpers"]):
        assert helper["source"].snapshot_state() == helper["state"]
        assert helper["options"]["checkpoint_updates"] == (0, 4, 8, 12, 16)
        assert helper["options"]["decision_diagnostics"] is True
        assert [p.context for p in helper["training"]] == ["E-blue", "E-red"] * 4 + [
            "A-blue",
            "A-red",
        ] * 4
        assert [p.context for p in helper["evaluation"]] == ["E-blue", "E-red", "A-blue", "A-red"]
        assert all(
            p.role == "probe_train" and p.features.shape == (9, 296) for p in helper["training"]
        )
        assert all(
            p.role == "probe_test" and p.features.shape == (9, 296) for p in helper["evaluation"]
        )
        actual_ids = []
        for packet in helper["training"][8:]:
            assert packet.episode is not None
            actual_ids.append(packet.episode.episode_id)
        assert actual_ids == [
            f"minigrid-source-A-{'blue' if i % 2 == 0 else 'red'}:return-train:{i}"
            for i in range(8, 16)
        ]
        actual_ids = []
        for packet in helper["evaluation"][2:]:
            assert packet.episode is not None
            actual_ids.append(packet.episode.episode_id)
        assert actual_ids == [f"minigrid-source-A-{cue}:return-test" for cue in ("blue", "red")]
        for packet, original in zip(helper["evaluation"][2:], work["return_sources"], strict=True):
            assert packet.episode is not None
            x, y = episode_arrays(encode_public_history_episode(original))
            np.testing.assert_array_equal(packet.features, x)
            np.testing.assert_array_equal(packet.targets, y)
        assert set(r for p in helper["training"] for r in p.row_ids).isdisjoint(
            r for p in helper["evaluation"] for r in p.row_ids
        )
        for train in helper["training"]:
            assert all(
                not np.shares_memory(train.features, test.features)
                and not np.shares_memory(train.targets, test.targets)
                for test in helper["evaluation"]
            )
        if index != 2:
            assert [p["updates"] for p in rows[index]["report"]["checkpoints"]] == [0, 4, 8, 12, 16]
    failed, fifo = rows[2], rows[3]
    assert "capacity" in failed["reason"] and [
        p["updates"] for p in failed["report"]["checkpoints"]
    ] == [0]
    assert all(value is None for value in failed["outcomes"].values())
    assert (
        failed["end_snapshot"]["completed_updates"] == 129
        and len(failed["end_snapshot"]["entries"]) == 63
    )
    end = fifo["end_snapshot"]
    assert (
        end["completed_updates"],
        end["training_examples"],
        len(end["entries"]),
        end["evictions"],
        end["retired_examples"],
    ) == (144, 1296, 64, 26, 564)
    assert (
        sum(sum(e["counts"]) for e in end["entries"]) + end["retired_examples"]
        == end["training_examples"]
    )
    curves = fifo["report"]["checkpoints"]
    assert all(curves[0]["scores"]["A-" + cue]["accuracy"] == 1 for cue in ("blue", "red"))
    assert curves[2]["scores"]["A-blue"]["accuracy"] < 1
    assert all(
        curves[-1]["scores"][context]["accuracy"] == 1
        for context in ("E-blue", "E-red", "A-blue", "A-red")
    )
    for row in (rows[0], rows[1], rows[3]):
        checkpoints = {point["updates"]: point for point in row["report"]["checkpoints"]}

        def mean(point, context):
            return (
                sum(
                    checkpoints[point]["scores"][context + "-" + cue]["ce"]
                    for cue in ("blue", "red")
                )
                / 2
            )

        assert row["outcomes"] == {
            "new_E_gain": mean(0, "E") - mean(8, "E"),
            "old_A_retention_loss": mean(8, "A") - mean(0, "A"),
            "old_A_return_gain": mean(8, "A") - mean(16, "A"),
            "new_E_retention_loss": mean(16, "E") - mean(8, "E"),
        }
    assert fifo["outcomes"]["old_A_retention_loss"] > 0
    assert fifo["outcomes"]["old_A_return_gain"] > 0


@pytest.mark.parametrize("target", ["source", "return_source", "data", "budget_limit", "clock"])
def test_should_stop_all_later_cohorts_after_unapproved_input_or_budget_changes(returning, target):
    work = returning

    def mutate(owner):
        if len(work["trains"]) != 1:
            return
        helper = work["helpers"][0]
        if target == "source":
            object.__setattr__(work["inputs"][1], "completed_updates", 127)
        elif target == "return_source":
            object.__setattr__(work["return_sources"][0], "role", "final_test")
        elif target == "data":
            helper["training"][8].features[0, 0] = 0.875
        elif target == "budget_limit":
            object.__setattr__(helper["options"]["budget"].budget, "max_training_updates", 15)
        else:
            helper["options"]["budget"].last_clock = 2.0

    work["on_train"] = mutate
    result = run(work)
    assert result["status"] == "incomplete"
    assert result["cohorts"][0]["status"] == "failed"
    assert all(row["status"] == "not_run" for row in result["cohorts"][1:])
    assert len(work["helpers"]) == len(work["trains"]) == 1
    assert result["cohorts"][0]["report"]["completed_updates"] == 1


def test_should_stop_after_unexpected_native_failure_instead_of_treating_it_as_capacity(returning):
    work = returning
    work["fail_attempt"] = 3
    result = run(work)
    first = result["cohorts"][0]
    assert result["status"] == "incomplete" and first["status"] == "failed"
    assert first["charged_updates"] == first["report"]["completed_updates"] == 2
    assert first["report"]["attempted_updates"] == 3
    assert all(row["status"] == "not_run" for row in result["cohorts"][1:])
    assert len(work["helpers"]) == 1


@pytest.mark.parametrize("target", ["failed_child", "failed_raw_report", "failed_budget"])
def test_should_revalidate_the_accepted_failed_prefix_before_the_final_fifo_cohort_can_work(
    returning, target
):
    work = returning

    def mutate_before_fourth(helper):
        if len(work["helpers"]) != 4:
            return
        previous = work["helpers"][2]
        if target == "failed_child":
            child = previous["source"].child
            assert child is not None
            state = child.snapshot_state()
            first = state.entries[0]
            counts = tuple(first.counts[(index + 1) % 7] for index in range(7))
            child.restore_state(
                replace(
                    state, entries=(ActionMemoryEntry(first.features, counts), *state.entries[1:])
                )
            )
        elif target == "failed_raw_report":
            previous["report"]["completed_updates"] = 0
        else:
            previous["options"]["budget"].updates_completed = 0

    work["before_helper"] = mutate_before_fourth
    result = run(work)
    assert result["status"] == "incomplete"
    assert result["cohorts"][2]["report"]["completed_updates"] == 1
    assert result["cohorts"][3]["status"] == "failed"
    assert len(work["trains"]) == 33 and len(work["predictions"]) == 44


@pytest.mark.parametrize(
    "fault", ["list", "one", "reversed", "role", "context", "id", "classes", "width", "steps"]
)
def test_should_refuse_nonoriginal_a_sources_before_any_resources(monkeypatch, fault):
    states, episodes = inputs()
    supplied: Any = episodes
    if fault == "list":
        supplied = list(episodes)
    elif fault == "one":
        supplied = episodes[:1]
    elif fault == "reversed":
        supplied = episodes[::-1]
    else:
        first = deepcopy(episodes[0])
        if fault == "role":
            object.__setattr__(first, "role", "probe_train")
        elif fault == "context":
            object.__setattr__(first, "context", "A-blue")
        elif fault == "id":
            object.__setattr__(first, "episode_id", "unknown-A-blue")
        elif fault == "classes":
            object.__setattr__(first, "num_classes", 5)
        elif fault == "width":
            object.__setattr__(first.transitions[0], "observation", (0.0,) * 296)
        else:
            object.__setattr__(first, "transitions", first.transitions[:-1])
        supplied = (first, episodes[1])
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(pilot.reserved, "_collect_probe_sources", forbidden)
    with pytest.raises(ValueError):
        pilot.run_return_pilot(states, supplied)


@pytest.mark.parametrize("fault", ["extra_field", "bool_count"])
def test_should_reject_malformed_native_port_returns_before_continuing(
    returning, monkeypatch, fault
):
    work = returning

    def corrupt(helper):
        if len(work["helpers"]) != 4:
            return
        child = work["helpers"][2]["source"].child
        invalid = child.snapshot_state()
        if fault == "extra_field":
            object.__setattr__(invalid.entries[0], "unexpected", 1)
        else:
            entry = next(e for e in invalid.entries if 1 in e.counts)
            counts = list(entry.counts)
            counts[counts.index(1)] = True
            object.__setattr__(entry, "counts", tuple(counts))
        monkeypatch.setattr(child, "snapshot_state", lambda: invalid)

    work["before_helper"] = corrupt
    result = run(work)
    assert result["status"] == "incomplete"
    assert result["cohorts"][2]["status"] == "failed"
    assert result["cohorts"][3]["status"] == "failed"
    assert result["cohorts"][3]["charged_updates"] == 0
    assert len(work["trains"]) == 33 and len(work["predictions"]) == 44
