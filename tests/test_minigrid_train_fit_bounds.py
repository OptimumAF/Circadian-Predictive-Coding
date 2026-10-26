"""TRAIN-only bounds integration with scripted runtime ports and no model work.

The full public pilot executes its ordinary publication path. Its paid learner,
score and probe ports are explicit scripts, so these checks establish reporting
and isolation behavior rather than acquisition or a scientific model advantage.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from inspect import getclosurevars
import json
import pickle
import random
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import run_coupled_attribution as public_cli
from scripts import run_minigrid_linear_fit_control as saved_cli
from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_linear_fit_control as linear
from src.adapters import minigrid_probe_support as support
from src.core.categorical_input_bounds import categorical_input_bounds
from src.core.categorical_probability_scores import probability_scores
from src.core.minigrid_public_history import encode_public_history_episode
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from test_minigrid_distinct_probe import raw_probe_data
from test_minigrid_invariant_learnability import episode as raw_episode
from test_minigrid_joint_catalogue_route import CLI_FLAGS, FLAGS
from test_minigrid_learning_pilot import POINTS
from test_minigrid_train_catalogue import episode
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub, forbidden


def histories():
    return tuple(
        episode(
            f"minigrid-source-{context}-{cue}",
            context=context,
            cue=2 if cue == "blue" else 0,
            actions=(0, 1, 2, 2, 2, 2, 3, 5, 2),
        )
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )


def raw_sources():
    return tuple(
        raw_episode(context, cue, "train", f"minigrid-source-{context}-{cue}")
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )


def test_should_reduce_exact_original_current_rows_and_preserve_inputs_and_rng():
    sources = histories()
    before = pickle.dumps(sources)
    rng = pickle.dumps((random.getstate(), np.random.get_state()))
    result = support.summarize_train_fit_bounds(sources)
    features = np.array([s.observation for e in sources for s in e.transitions])
    targets = np.array([[s.action] for e in sources for s in e.transitions], dtype=np.float64)
    canonical = np.array(
        [canonicalize_public_history(tuple(float(value) for value in row)) for row in features]
    )
    assert result["raw"] == asdict(categorical_input_bounds(features, targets, 7))
    assert result["canonical"] == asdict(categorical_input_bounds(canonical, targets, 7))
    assert result["definition"] == support.TRAIN_BOUNDS_DEFINITION
    assert result["status"] == "complete" and result["role"] == "train"
    assert result["source_episode_ids"] == [e.episode_id for e in sources]
    work = result["work"]
    assert work["native_updates"] == work["native_score_forwards"] == work["fit_calls"] == 0
    assert work["input_bound_reductions"] == 2 and work["input_bound_rows"] == 108
    assert work["admission_projection_rows"] == 108 and work["bound_projection_rows"] == 54
    assert work["projected_array_equivalent_bytes"] == 162 * 296 * 8
    assert work["input_array_bytes"] == 54 * (296 + 296 + 1) * 8
    assert work["elapsed_seconds"] >= 0
    result["raw"]["rows"] = 999
    result["source_episode_ids"].clear()
    assert pickle.dumps(sources) == before
    assert pickle.dumps((random.getstate(), np.random.get_state())) == rng
    assert support.summarize_train_fit_bounds(sources)["raw"]["rows"] == 54


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "missing",
        "extra",
        "order",
        "identity",
        "context",
        "short",
        "probe_train",
        "probe_test",
        "final_test",
        "array",
        "rogue_episode",
        "rogue_transition",
        "next_initial",
        "next_nonfinite",
        "action_bool",
        "raw148",
    ],
)
def test_should_refuse_wrong_sources_before_the_first_bound_reduction(monkeypatch, fault):
    sources: Any = histories()
    if fault == "list":
        sources = list(sources)
    elif fault == "missing":
        sources = sources[:-1]
    elif fault == "extra":
        sources = (*sources, sources[-1])
    elif fault == "order":
        sources = (sources[1], sources[0], *sources[2:])
    elif fault in ("identity", "context", "probe_train", "probe_test", "final_test"):
        field = "episode_id" if fault == "identity" else "context" if fault == "context" else "role"
        value = "wrong" if fault in ("identity", "context") else fault
        sources = (replace(sources[0], **{field: value}), *sources[1:])
    elif fault == "short":
        sources = (episode(sources[0].episode_id), *sources[1:])
    elif fault == "raw148":
        sources = raw_sources()
    elif fault == "rogue_episode":
        object.__setattr__(sources[0], "unobserved", True)
    elif fault == "rogue_transition":
        object.__setattr__(sources[0].transitions[0], "unobserved", True)
    elif fault == "array":
        object.__setattr__(sources[0].transitions[0], "observation", np.zeros(296))
    elif fault == "action_bool":
        object.__setattr__(sources[0].transitions[0], "action", True)
    else:
        step = sources[0].transitions[-1]
        values = list(step.next_observation)
        values[0] = float("nan") if fault == "next_nonfinite" else 1.0 - values[0]
        object.__setattr__(step, "next_observation", tuple(values))
    monkeypatch.setattr(support, "categorical_input_bounds", forbidden)
    with pytest.raises(ValueError):
        support.summarize_train_fit_bounds(sources)


def test_should_refuse_source_mutation_between_completed_reductions(monkeypatch):
    sources = histories()
    original = support.categorical_input_bounds
    calls = []

    def mutate(*args):
        result = original(*args)
        calls.append(result)
        object.__setattr__(sources[-1], "episode_id", "foreign")
        return result

    monkeypatch.setattr(support, "categorical_input_bounds", mutate)
    with pytest.raises(ValueError):
        support.summarize_train_fit_bounds(sources)
    assert len(calls) == 1 and sources[-1].episode_id == "foreign"


def test_should_refuse_original_source_mutation_in_the_closing_clock(monkeypatch):
    sources = histories()
    calls = []

    def clock():
        calls.append(1)
        if len(calls) == 2:
            object.__setattr__(sources[0], "context", "foreign")
        return float(len(calls))

    monkeypatch.setattr(support, "monotonic", clock)
    with pytest.raises(ValueError):
        support.summarize_train_fit_bounds(sources)
    assert len(calls) == 2 and sources[0].context == "foreign"


def test_should_publish_signed_equal_row_gaps_without_changing_original_scores():
    bounds = support.summarize_train_fit_bounds(histories())
    bounds["canonical"].update(row_accuracy_upper_bound=0.5, normalized_ce_lower_bound=0.75)
    points: list[dict[str, Any]] = [
        dict(
            updates=point,
            **{
                context: dict(accuracy=0.8, ce=0.25, exact_sequence_accuracy=0.0)
                for context in ("A", "B", "C_train")
            },
        )
        for point in POINTS
    ]
    before = pickle.dumps((bounds, points))
    gaps = support.fit_gap_outcomes(bounds, points)
    assert [g["updates"] for g in gaps] == list(POINTS)
    assert all(g["accuracy_gap"] == pytest.approx(-0.3) for g in gaps)
    assert all(g["ce_gap"] == -0.5 for g in gaps)
    assert all(g["observed_accuracy"] == pytest.approx(0.8) for g in gaps)
    assert pickle.dumps((bounds, points)) == before


def test_should_preserve_negative_ce_gap_from_admitted_normalization_roundoff():
    targets = np.tile(np.array([[0.0], [1.0]]), (27, 1))
    bound = categorical_input_bounds(np.zeros((54, 296)), targets, 2)
    scores = probability_scores(np.full((54, 2), 0.50000025), targets, 2)
    bounds = dict(
        definition=support.TRAIN_BOUNDS_DEFINITION,
        status="complete",
        role="train",
        canonical=asdict(bound),
    )
    point = dict(updates=0, **{c: scores for c in ("A", "B", "C_train")})
    gap = support.fit_gap_outcomes(bounds, [point])[0]
    assert bound.normalized_ce_lower_bound == pytest.approx(np.log(2))
    assert -1e-6 < gap["ce_gap"] < 0
    assert gap["observed_ce"] == scores["ce"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1])
@pytest.mark.parametrize("metric", ["accuracy", "ce"])
def test_should_refuse_invalid_observed_scores(value, metric):
    bounds = support.summarize_train_fit_bounds(histories())
    point: dict[str, Any] = dict(
        updates=0, **{c: dict(accuracy=0.5, ce=0.5) for c in ("A", "B", "C_train")}
    )
    for context in ("A", "B", "C_train"):
        point[context][metric] = value
    with pytest.raises(ValueError):
        support.fit_gap_outcomes(bounds, [point])


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_bool_flags_before_runtime(monkeypatch, flag):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(**FLAGS, joint_catalogue=True, train_fit_bounds=flag)
    with pytest.raises(ValueError):
        pilot.run_pilot(**FLAGS, joint_catalogue=True, train_fit_bounds=flag)


def test_should_preserve_default_descriptor_and_frozen_recipe(monkeypatch):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    options = {**FLAGS, "joint_catalogue": True}
    original = pilot.describe_protocol(**options)
    assert original == pilot.describe_protocol(**options, train_fit_bounds=False)
    updated = pilot.describe_protocol(**options, train_fit_bounds=True)
    assert updated["train_fit_bounds"] == support.describe_train_fit_bounds()
    for key in (
        "seed",
        "methods",
        "learning_rate",
        "momentum",
        "inference_steps",
        "inference_learning_rate",
        "checkpoint_updates",
        "metric_direction",
        "probe",
    ):
        assert updated[key] == original[key]
    assert (
        updated["planned_work"]["native_train_calls"]
        == original["planned_work"]["native_train_calls"]
    )
    with pytest.raises(ValueError):
        pilot.describe_protocol(**FLAGS, train_fit_bounds=True)


def scripted_pilot(monkeypatch, *, failure=None):
    """Use ordinary pilot publication/guards with no training or predictions."""
    events: list[Any] = []
    published: dict[str, Any] = {}
    original_bounds = pilot.summarize_train_fit_bounds
    monkeypatch.setattr(pilot, "monotonic", lambda: 1.0)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    for name in ("MiniGridKeyDoor", "CueKeyDoorTeacher", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(
        pilot.shared,
        "_collect_sources",
        lambda *args: tuple(pilot._packet(value) for value in raw_sources()),
    )

    def train(row, packets, sampler, deadline, check_boundary, **options):
        check_boundary()
        assert len(packets) == 128 and all(p.features.shape == (54, 296) for p in packets)
        assert options["catalogue_rows"] == 54
        events.append("benchmark_finished")
        row.update(
            status="trained",
            charged_updates=128,
            native_training_examples=6912,
            wake_calls_attempted=128,
            wake_calls_returned=128,
        )
        return dict(
            row=row,
            source=object(),
            states=tuple((p, object()) for p in POINTS),
            stream=object(),
            budget=SimpleNamespace(updates_completed=128),
            guard=lambda: None,
        )

    def bounds(sources):
        assert events == ["benchmark_finished"]
        events.append("bounds")
        return original_bounds(sources)

    def collect(check_boundary, reports, **options):
        events.append("reserved_collection")
        common = getclosurevars(check_boundary).nonlocals["common_guard"]
        report = getclosurevars(common).nonlocals["report"]
        published["report"] = report
        if failure == "bounds_value":
            report["train_fit_bounds"]["canonical"]["rows"] = 53
        elif failure == "bounds_owner":
            report["train_fit_bounds"] = deepcopy(report["train_fit_bounds"])
        check_boundary()
        return raw_probe_data()

    def score(source, state, **options):
        options["check_boundary"]()
        point = options["updates"]
        events.append(("score", point))
        return dict(
            status="complete",
            reason=None,
            updates=point,
            scores=[
                dict(
                    context=c,
                    ce=0.25 + point / 1280,
                    accuracy=0.25 + point / 256,
                    exact_sequence_accuracy=0.0,
                )
                for c in ("A", "B", "C_train")
                for _ in range(2)
            ],
            work=dict(
                score_forward_calls_attempted=0,
                score_forward_calls_completed=0,
                score_rows_completed=0,
                failure_work_may_be_unobserved=False,
            ),
        )

    def probe(slot, *args, **options):
        options["check_boundary"]()
        events.append(("probe", slot["checkpoint_updates"]))
        if failure == "probe":
            raise RuntimeError("scripted probe refusal after all paid fit scores")
        slot.update(
            status="complete",
            reason=None,
            charged_updates=0,
            work_unknown=False,
            outcomes=dict(ce_gain=0.0),
        )
        return lambda: None

    monkeypatch.setattr(pilot.shared, "_train_arm", train)
    monkeypatch.setattr(pilot.shared, "_readback", lambda *a, **k: events.append("readback"))
    monkeypatch.setattr(pilot, "summarize_train_fit_bounds", bounds)
    monkeypatch.setattr(pilot, "_collect_probe_sources", collect)
    monkeypatch.setattr(pilot, "_score_checkpoint", score)
    monkeypatch.setattr(pilot, "_run_probe", probe)
    return events, published


def test_should_publish_bounds_and_five_gaps_through_the_complete_public_pilot(monkeypatch):
    events, _ = scripted_pilot(monkeypatch)
    result = pilot.run_pilot(**FLAGS, joint_catalogue=True, train_fit_bounds=True)
    assert result["status"] == "complete", result["reason"]
    assert events[:3] == ["benchmark_finished", "bounds", "reserved_collection"]
    assert [v for v in events if type(v) is tuple] == [
        *(("score", p) for p in POINTS),
        ("probe", 0),
        ("probe", 128),
    ]
    row = result["methods"][0]
    assert len(row["fit_gap_outcomes"]) == 5
    assert row["fit_gap_outcomes"] == support.fit_gap_outcomes(
        result["train_fit_bounds"], row["outcomes"]
    )
    assert [p["updates"] for p in row["outcomes"]] == list(POINTS)
    assert row["outcomes"][-1]["A"]["accuracy"] == 0.75
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 6912
    assert result["train_fit_bounds"]["work"]["native_updates"] == 0


def test_should_skip_bounds_and_gaps_when_the_public_flag_is_false(monkeypatch):
    events, _ = scripted_pilot(monkeypatch)
    monkeypatch.setattr(pilot, "summarize_train_fit_bounds", forbidden)
    result = pilot.run_pilot(**FLAGS, joint_catalogue=True, train_fit_bounds=False)
    assert result["status"] == "complete", result["reason"]
    assert "train_fit_bounds" not in result and "bounds" not in events
    assert "fit_gap_outcomes" not in result["methods"][0]
    assert [v["updates"] for v in result["methods"][0]["outcomes"]] == list(POINTS)


@pytest.mark.parametrize("failure", ["bounds_value", "bounds_owner", "probe"])
def test_should_refuse_changed_evidence_or_later_failure_and_retain_paid_records(
    monkeypatch, failure
):
    events, _ = scripted_pilot(monkeypatch, failure=failure)
    result = pilot.run_pilot(**FLAGS, joint_catalogue=True, train_fit_bounds=True)
    assert result["status"] == "incomplete"
    assert result["train_fit_bounds"]["canonical"]["rows"] == 54
    row = result["methods"][0]
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 6912
    assert "fit_gap_outcomes" not in row
    if failure == "probe":
        assert len(row["readouts"]) == 5 and all(v["status"] == "complete" for v in row["readouts"])
        assert "scripted probe refusal" in result["reason"]
        assert ("probe", 128) not in events
    else:
        assert "source evidence" in result["reason"]
        assert not any(type(v) is tuple for v in events)


def test_should_describe_saved_bounds_without_reading_or_fitting(monkeypatch, capsys):
    monkeypatch.setattr(saved_cli, "read_training_sources", forbidden)
    monkeypatch.setattr(saved_cli, "diagnose_train_linear_fit", forbidden)
    monkeypatch.setattr(sys, "argv", ["control", "--bounds-only", "--describe"])
    saved_cli.main()
    assert json.loads(capsys.readouterr().out) == support.describe_train_fit_bounds()


def test_should_decode_saved_original_train_sources_without_fitting_or_prediction(
    tmp_path, monkeypatch, capsys
):
    source, output = tmp_path / "source.json", tmp_path / "bounds.json"
    sources = raw_sources()
    source.write_text(
        json.dumps(
            dict(
                training_sources=[asdict(e) for e in sources],
                final_test="unavailable",
                probe_data="unavailable",
            )
        )
    )
    before = source.read_bytes()
    for name in ("fit_categorical_linear", "predict_categorical_linear"):
        monkeypatch.setattr(linear, name, forbidden)
    monkeypatch.setattr(saved_cli, "diagnose_train_linear_fit", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["control", "--bounds-only", "--source", str(source), "--output", str(output)]
    )
    saved_cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(output.read_text())
    expected = support.summarize_train_fit_bounds(
        tuple(encode_public_history_episode(e) for e in sources)
    )
    assert result["raw"] == expected["raw"] and result["canonical"] == expected["canonical"]
    assert result["work"]["fit_calls"] == result["work"]["native_score_forwards"] == 0
    assert "final_test" not in result and "probe_data" not in result
    assert source.read_bytes() == before
    monkeypatch.setattr(saved_cli, "read_training_sources", forbidden)
    with pytest.raises(FileExistsError):
        saved_cli.main()


@pytest.mark.parametrize("describe", [False, True])
def test_should_forward_public_cli_bounds_option_without_native_work(monkeypatch, capsys, describe):
    calls = []

    def selected(**options):
        calls.append(options)
        return dict(status="complete")

    monkeypatch.setattr(public_cli, "describe_minigrid_learnability_protocol", selected)
    monkeypatch.setattr(public_cli, "run_minigrid_learnability_pilot", selected)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pilot",
            *CLI_FLAGS,
            "--joint-catalogue",
            "--train-fit-bounds",
            *(["--describe"] if describe else []),
        ],
    )
    public_cli.main()
    assert calls == [{**FLAGS, "joint_catalogue": True, "train_fit_bounds": True}]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize("arguments", [["--train-fit-bounds"], [*CLI_FLAGS, "--train-fit-bounds"]])
def test_should_refuse_wrong_public_cli_route_before_dispatch(monkeypatch, arguments):
    monkeypatch.setattr(public_cli, "run_minigrid_learnability_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", *arguments])
    with pytest.raises(SystemExit) as caught:
        public_cli.main()
    assert caught.value.code == 2
