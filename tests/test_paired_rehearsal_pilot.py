"""Fixed paired initializations preserve every row and never pool failures."""

from copy import deepcopy
import json
import pickle
import sys

import numpy as np
import pytest

import scripts.run_stream_rehearsal_pilot as cli
import src.adapters.stream_rehearsal_pilot as pilot
from test_recent_stream_rehearsal import retained_probe_protocol
from test_rehearsal_learnability import RETAINED_PROTOCOL, FakeLearner, fake_environment, forbidden
from test_stream_rehearsal_pilot import Parameters


SEEDS = (41, 59)
POLICIES = ("rotating", "recent_only")
ORDER = tuple(
    (seed, policy, method)
    for seed in SEEDS
    for policy in POLICIES
    for method in ("bp", "pc", "neutral_cpc", "gated_cpc")
)


def expected_single(recent, diagnose):
    result = retained_probe_protocol() if diagnose else deepcopy(RETAINED_PROTOCOL)
    if recent:
        old_id = result["protocol_id"]
        result.update(
            protocol_id="coherent_recent_rehearsal_reserved_sixteen_v1"
            if diagnose
            else "coherent_stream_recent_rehearsal_v1",
            control_for=old_id,
            replay={
                "every_updates": 3,
                "rows_per_event": 2,
                "retention": {"max_examples": 16, "max_bytes": 384},
            },
            replay_selection="arrived-only recent FIFO; latest n distinct retained rows in arrival order",
            replay_retention="recent_fifo",
            comparison_scope="matched planned work/exposure and array-memory caps per method; retention composition AND within-event order differ; no isolated reservoir effect",
            limitations="one fixed seed/data; matched exposure/native work per method versus rotating control; retention and order change together; actual memory/latency may differ; no confirmation or circadian advantage claim",
        )
    return result


@pytest.mark.parametrize("recent", [False, True])
@pytest.mark.parametrize("diagnose", [False, True])
def test_should_preserve_all_four_existing_single_protocols_exactly(recent, diagnose):
    assert pilot.describe_protocol(
        recent_only=recent, diagnose_learnability=diagnose
    ) == expected_single(recent, diagnose)


@pytest.mark.parametrize("action", ["describe", "run"])
def test_should_route_paired_cli_only_to_the_no_argument_paired_entrypoint(
    action, monkeypatch, capsys
):
    calls = []

    def selected():
        calls.append(action)
        return {"status": "complete", "selected": action}

    selected_name = "describe_paired_protocol" if action == "describe" else "run_paired_pilot"
    for name in ("describe_protocol", "run_pilot", "describe_paired_protocol", "run_paired_pilot"):
        monkeypatch.setattr(cli, name, selected if name == selected_name else forbidden)
    monkeypatch.setattr(sys, "argv", ["rehearsal", f"--{action}", "--scenario", "paired-retention"])
    assert cli.main() == 0 and calls == [action]
    assert json.loads(capsys.readouterr().out)["selected"] == action


@pytest.mark.parametrize("action", ["describe", "run"])
@pytest.mark.parametrize("flag", ["--recent-only", "--diagnose-learnability"])
def test_should_reject_single_mode_flags_on_paired_scenario_before_any_callback(
    action, flag, monkeypatch
):
    for name in ("describe_protocol", "run_pilot", "describe_paired_protocol", "run_paired_pilot"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(
        sys, "argv", ["rehearsal", f"--{action}", "--scenario", "paired-retention", flag]
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--scenario", "paired-retention"],
        ["--run", "--scenario", "arbitrary"],
        ["--run", "--scenario", "paired-retention", "--seed", "7"],
    ],
)
def test_should_require_a_fixed_valid_scenario_and_explicit_action(arguments, monkeypatch):
    for name in ("describe_protocol", "run_pilot", "describe_paired_protocol", "run_paired_pilot"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["rehearsal", *arguments])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "recent,diagnose", [(False, False), (False, True), (True, False), (True, True)]
)
def test_should_preserve_explicit_single_cli_dispatch_without_new_scenario_keywords(
    recent, diagnose, monkeypatch, capsys
):
    calls = []

    def selected(**kwargs):
        calls.append(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(cli, "describe_protocol", selected)
    monkeypatch.setattr(cli, "describe_paired_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rehearsal",
            "--describe",
            "--scenario",
            "single",
            *(["--recent-only"] if recent else []),
            *(["--diagnose-learnability"] if diagnose else []),
        ],
    )
    assert cli.main() == 0 and capsys.readouterr().out
    expected = (
        {"recent_only": True, "diagnose_learnability": diagnose}
        if recent
        else {"diagnose_learnability": True}
        if diagnose
        else {}
    )
    assert calls == [expected]


class PairedLearner(FakeLearner):
    def fork(self):
        self.control["trace"].append((self.method, self.origin, "fork", float(self.state[0])))
        return PairedLearner(self.control, self.method, self.origin, self.state[0], original=False)

    def train_batch(self, features, targets):
        diagnostic = super().train_batch(features, targets)
        # Why data-sensitive fake state: paired arithmetic must be checked on
        # actual unequal levels, without imposing a desired direction or winner.
        self.state += 0.01 * float(np.mean(features))
        if (
            self.method.startswith("row-0:")
            and len(features) == 1
            and self.control.get("fail_first")
        ):
            self.control["fail_first"] = False
            raise RuntimeError("declared paired replay mutation then failure")
        return diagnostic


def paired_environment(monkeypatch, *, fail_first=False, deadline=False):
    control = fake_environment(monkeypatch)
    control.update(initial_models=[], builds=[], fail_first=fail_first)
    seconds = [0.0]
    monkeypatch.setattr(pilot, "monotonic", lambda: seconds[0])
    monkeypatch.setattr(pilot.learnability, "create_sixteen_probe_observer", forbidden)

    class SeedModel:
        def __init__(self, input_dim, hidden_dims, seed):
            assert (input_dim, hidden_dims) == (2, (8, 8))
            self.parameters = Parameters(np.full((2, 1), seed / 1000))
            control["initial_models"].append((seed, self.parameters))
            if deadline:
                seconds[0] = 61.0

        def snapshot_parameters(self):
            return self.parameters

    def build(method, initial):
        index = len(control["builds"])
        source = PairedLearner(control, f"row-{index}:{method}", "benchmark", initial.weights[0, 0])
        control["sources"].append(source)
        control["builds"].append((method, initial, pickle.dumps(initial), source))
        return source

    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", SeedModel)
    monkeypatch.setattr(pilot, "build_learner", build)
    return control


def independent_outcomes(checkpoints):
    scores = {row["updates"]: row["scores"] for row in checkpoints}
    return {
        "new_B_gain": scores[0]["B"]["bce"] - scores[12]["B"]["bce"],
        "retention_loss": scores[12]["A"]["bce"] - scores[6]["A"]["bce"],
        "return_gain": scores[12]["A"]["bce"] - scores[18]["A"]["bce"],
    }


def assert_planned_rows(report):
    assert [(row["seed"], row["policy"], row["method"]) for row in report["runs"]] == list(ORDER)
    assert [(pair["seed"], pair["method"]) for pair in report["pairs"]] == [
        (seed, method) for seed in SEEDS for method in pilot.METHODS
    ]
    assert all(row["planned_event_updates"] == [3, 6, 9, 12, 15, 18] for row in report["runs"])


def test_should_describe_only_the_two_fixed_initializations_both_policies_and_matched_work(
    monkeypatch,
):
    for name in (
        "build_stream",
        "build_learner",
        "CoupledPredictiveCodingNetwork",
        "ProcessRssSampler",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    report = pilot.describe_paired_protocol()
    assert report["protocol_id"] == "coherent_paired_retention_initializations_v1"
    assert report["seeds"] == [41, 59] and report["policies"] == ["rotating", "recent_only"]
    assert report["methods"] == ["bp", "pc", "neutral_cpc", "gated_cpc"]
    assert report["planned_work"] == dict(
        runs=16,
        pairs=8,
        wake_updates=288,
        replay_updates=192,
        optimizer_updates=480,
        wake_row_exposures=2304,
        replay_row_exposures=192,
        total_row_exposures=2496,
        consolidation_events=96,
        final_score_forwards=128,
        probes=0,
        svd=0,
    )
    assert report["resources"]["training_updates"] == 480
    assert report["resources"]["replay_bytes"] == 384 and report["resources"]["cpu_threads"] == 1
    assert report["replay"]["rotating"] == RETAINED_PROTOCOL["replay"]
    assert report["replay"]["recent_only"] == {
        "every_updates": 3,
        "rows_per_event": 2,
        "retention": {"max_examples": 16, "max_bytes": 384},
    }
    assert report["training_rng"] == 7301 and report["final_rng"] == 9101
    assert report["checkpoint_updates"] == [0, 6, 12, 18] and report["label_delay"] == 2
    assert "order" in report["comparison_scope"] and "no pooling" in report["limitations"]


def test_should_run_each_fixed_row_once_from_two_seed_sensitive_private_initializations_and_keep_exact_pairs(
    monkeypatch,
):
    control = paired_environment(monkeypatch)
    report = pilot.run_paired_pilot()
    assert report["status"] == "complete", [row.get("reason") for row in report["runs"]]
    assert_planned_rows(report)
    assert [seed for seed, _ in control["initial_models"]] == [41, 59]
    assert report["initial_parameters"] == {
        str(seed): {"weights": [[seed / 1000], [seed / 1000]]} for seed in SEEDS
    }
    assert report["initial_parameters"]["41"] != report["initial_parameters"]["59"]
    assert len(control["builds"]) == len(control["sources"]) == 16 and control["probes"] == []
    for index, (row, (method, initial, before, source)) in enumerate(
        zip(report["runs"], control["builds"], strict=True)
    ):
        expected_initial = control["initial_models"][index // 8][1]
        assert initial is expected_initial and method == row["method"]
        assert pickle.dumps(initial) == before
        assert source.state[0] == row["seed"] / 1000 and not np.shares_memory(
            source.state, initial.weights
        )
        assert row["status"] == "complete" and row["source_learner_unchanged"] is True
        assert "observations" not in row
        events = [event for event in control["trace"] if event[0] == source.method]
        trained = [event for event in events if event[2] == "train"]
        predicted = [event for event in events if event[2] == "predict"]
        assert (
            len(trained) == 30 and sum(event[3] for event in trained) == 156 and len(predicted) == 8
        )
        assert max(position for position, event in enumerate(events) if event[2] == "train") < min(
            position for position, event in enumerate(events) if event[2] == "predict"
        )
        assert (
            len(row["stream_result"]["applied_updates"]) == 18
            and len(row["stream_result"]["applied_consolidations"]) == 6
        )
    assert len({id(source.state) for source in control["sources"]}) == 16
    assert report["wake_updates"] == 288
    assert (
        report["replay_charged_examples"]
        == report["replay_attempted_updates"]
        == report["replay_returned_updates"]
        == report["replay_committed_updates"]
        == 192
    )
    assert (
        report["native_train_calls"]
        == report["known_native_train_calls"]
        == report["total_committed_updates"]
        == 480
    )
    assert report["row_exposures_charged"] == 2496
    assert report["final_score_forwards"] == report["known_completed_score_forwards"] == 128
    indexed = {(row["seed"], row["policy"], row["method"]): row for row in report["runs"]}
    for pair in report["pairs"]:
        assert pair["status"] == "complete" and pair["reasons"] == {
            "rotating": None,
            "recent_only": None,
        }
        expected = {}
        for policy in POLICIES:
            row = indexed[pair["seed"], policy, pair["method"]]
            expected[policy] = independent_outcomes(row["stream_result"]["checkpoints"])
            assert pair["primary_levels"][policy] == row["stream_result"]["checkpoints"]
            assert pair["outcomes"][policy] == expected[policy]
        assert pair["recent_minus_rotating"] == {
            key: expected["recent_only"][key] - expected["rotating"][key]
            for key in expected["rotating"]
        }
        assert pair["initial_primary_levels_exact"] is True
    assert len(report["group_facts"]) == 4 and all(
        group["status"] == "complete"
        and group["pc_neutral_exact"] is True
        and group["shared_replay_plans_exact"] is True
        for group in report["group_facts"]
    )
    assert len(report["policy_plan_facts"]) == 2 and all(
        fact["replay_plans_exact_across_initializations"] is True
        for fact in report["policy_plan_facts"]
    )
    assert (
        report["full_R28_complete"]
        is report["full_R23_complete"]
        is report["scientific_advantage_verified"]
        is False
    )
    assert all(pickle.dumps(data) == before for data, before in control["streams"])


def test_should_preserve_first_failed_charge_and_all_other_rows_without_retry_or_incomplete_pair_averaging(
    monkeypatch,
):
    control = paired_environment(monkeypatch, fail_first=True)
    report = pilot.run_paired_pilot()
    assert report["status"] == "incomplete"
    assert_planned_rows(report)
    assert len(control["builds"]) == 16 and [seed for seed, _ in control["initial_models"]] == [
        41,
        59,
    ]
    first = report["runs"][0]
    assert first["status"] == "failed" and "declared paired replay mutation" in first["reason"]
    assert first["wake_updates"] == 3 and first["replay_charged_examples"] == 1
    assert first["rehearsal"]["attempted_updates"] == first["rehearsal"]["charged_exposures"] == 1
    assert first["rehearsal"]["returned_updates"] == first["rehearsal"]["committed_updates"] == 0
    assert [event["status"] for event in first["rehearsal"]["events"]] == ["failed"] + [
        "not_run"
    ] * 5
    assert "stream_result" not in first and all(
        row["status"] == "complete" for row in report["runs"][1:]
    )
    first_trace = [event for event in control["trace"] if event[0].startswith("row-0:")]
    assert sum(event[2] == "train" for event in first_trace) == 4
    assert not any(event[2] == "predict" for event in first_trace)
    assert report["native_train_calls"] is report["final_score_forwards"] is None
    assert (
        report["known_native_train_calls"] == 454
        and report["known_completed_score_forwards"] == 120
    )
    assert report["wake_updates"] == 273 and report["replay_charged_examples"] == 181
    assert report["replay_returned_updates"] == report["replay_committed_updates"] == 180
    assert report["total_committed_updates"] == 453 and report["row_exposures_charged"] == 2365
    pair = report["pairs"][0]
    assert pair["status"] == "incomplete"
    assert pair["reasons"]["rotating"] == first["reason"] and pair["reasons"]["recent_only"] is None
    assert (
        pair["primary_levels"]
        is pair["outcomes"]
        is pair["recent_minus_rotating"]
        is pair["initial_primary_levels_exact"]
        is None
    )
    assert all(pair["status"] == "complete" for pair in report["pairs"][1:])
    assert report["group_facts"][0]["status"] == "unavailable"
    assert (
        report["group_facts"][0]["pc_neutral_exact"]
        is report["group_facts"][0]["shared_replay_plans_exact"]
        is None
    )
    assert (
        report["policy_plan_facts"][0]["status"] == "unavailable"
        and report["policy_plan_facts"][0]["replay_plans_exact_across_initializations"] is None
    )
    assert all(
        source.state[0] == ORDER[index][0] / 1000 for index, source in enumerate(control["sources"])
    )


def assert_all_unreached(report):
    assert report["status"] == "incomplete"
    assert_planned_rows(report)
    assert all(row["status"] == "not_run" and type(row["reason"]) is str for row in report["runs"])
    for pair in report["pairs"]:
        assert pair["status"] == "incomplete" and all(
            type(reason) is str for reason in pair["reasons"].values()
        )
        assert pair["primary_levels"] is pair["outcomes"] is pair["recent_minus_rotating"] is None
    assert all(group["status"] == "unavailable" for group in report["group_facts"])
    assert (
        report["wake_updates"]
        == report["replay_charged_examples"]
        == report["known_native_train_calls"]
        == report["known_completed_score_forwards"]
        == 0
    )


def test_should_preserve_all_sixteen_rows_and_eight_unavailable_pairs_on_global_setup_failure(
    monkeypatch,
):
    def fail():
        raise RuntimeError("declared paired sampler failure")

    monkeypatch.setattr(pilot, "ProcessRssSampler", fail)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", forbidden)
    monkeypatch.setattr(pilot, "build_learner", forbidden)
    report = pilot.run_paired_pilot()
    assert_all_unreached(report)
    assert "declared paired sampler failure" in report["global_errors"][0]
    assert report["initial_parameters"] == {}


def test_should_admit_no_row_after_the_single_global_deadline_without_recreating_the_budget(
    monkeypatch,
):
    control = paired_environment(monkeypatch, deadline=True)
    report = pilot.run_paired_pilot()
    assert_all_unreached(report)
    assert control["builds"] == [] and control["trace"] == []
    assert [seed for seed, _ in control["initial_models"]] == [41]
    assert report["total_seconds"] >= 60.0
