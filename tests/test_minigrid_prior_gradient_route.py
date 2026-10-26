"""Existing TRAIN checkpoint/CLI integration; all learner/physics ports scripted."""

import json
import sys
from typing import Any

import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_gradient_readouts as gradients
from src.adapters import minigrid_learnability as pilot
from src.adapters.torch_recurrent_gradient_diagnostics import GradientDiagnosticFailed
from test_minigrid_learning_pilot import evaluator as evaluator
from test_minigrid_pc_qualification import FLAGS, environment
from test_torch_feature_rehearsal_pilot import forbidden


@pytest.mark.parametrize("method", [None, "bp", "pc", "neutral_cpc", "gated_cpc"])
def test_should_describe_extra_train_work_without_resources(monkeypatch, method):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", forbidden)
    options: dict[str, Any] = dict(public_history=True, invariant_history=True)
    if method is not None:
        options["qualification_method"] = method
    old = pilot.describe_protocol(**options)
    new = pilot.describe_protocol(**options, prior_gradient_geometry=True)
    assert new["methods"] == old["methods"]
    assert new["probe"] == old["probe"]
    for key, value in old["planned_work"].items():
        assert new["planned_work"][key] == value
    work = new["planned_work"]["prior_gradient_geometry"]
    pairs = 60 if method is None else 15
    assert work["pair_calls"] == pairs
    assert work["forward_rows"] == work["backward_rows"] == 18 * pairs
    assert work["projection_array_bytes"] == 2368 * 18 * pairs
    assert work["geometry_reductions"] == 5 * pairs
    assert work["optimizer_steps"] == work["published_updates"] == 0
    assert new["prior_gradient_geometry"]["methods"] == [
        m for m in old["methods"] if m.startswith("public_invariant_")
    ]
    assert "not native PC" in new["prior_gradient_geometry"]["objective"]


@pytest.mark.parametrize("flag", [1, None, "true"])
@pytest.mark.parametrize("describe", [True, False])
def test_should_refuse_nonboolean_before_resources(monkeypatch, flag, describe):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", forbidden)
    with pytest.raises(ValueError, match="prior_gradient_geometry"):
        (pilot.describe_protocol if describe else pilot.run_pilot)(
            **FLAGS, prior_gradient_geometry=flag
        )


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"public_history": True},
        {"public_history": True, "invariant_history": True, "gradient_geometry": True},
        {
            "public_history": True,
            "invariant_history": True,
            "qualification_method": "pc",
            "acquisition_only": True,
        },
    ],
)
def test_should_refuse_wrong_representation_or_recipe(extra):
    with pytest.raises(ValueError):
        pilot.describe_protocol(**extra, prior_gradient_geometry=True)


@pytest.mark.parametrize("describe", [True, False])
def test_should_forward_explicit_cli_only(monkeypatch, capsys, describe):
    calls = []

    def selected(**options):
        calls.append(options)
        return dict(status="complete")

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pilot",
            "--scenario",
            "torch-minigrid-history-learnability",
            "--invariant-history",
            "--pc-qualification",
            "--prior-gradient-geometry",
        ]
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [dict(**FLAGS, prior_gradient_geometry=True)]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--scenario", "torch-minigrid-learnability"],
        ["--scenario", "torch-minigrid-history-learnability"],
        [
            "--scenario",
            "torch-minigrid-history-learnability",
            "--invariant-history",
            "--cue-gradient-geometry",
        ],
        [
            "--scenario",
            "torch-minigrid-history-learnability",
            "--invariant-history",
            "--acquisition-only",
            "--qualification-method",
            "bp",
        ],
    ],
)
def test_should_reject_cli_before_ports(monkeypatch, arguments):
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", "--prior-gradient-geometry", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("failure", [False, True, "late_cost"])
def test_should_finish_train_gradients_before_separate_probes_or_preserve_failure(
    monkeypatch, evaluator, failure
):
    fixture = environment(monkeypatch, evaluator)
    calls = []
    holder = {}
    if failure == "late_cost":
        original_fit = pilot._score_checkpoint
        original_prior = gradients.run_prior_gradient_readouts

        def capture(report, *args, **options):
            holder["report"] = report
            return original_prior(report, *args, **options)

        def fit(*args, **options):
            original_boundary = options["check_boundary"]

            def boundary():
                original_boundary()
                report = holder.get("report")
                if report is not None:
                    work = report["work"]["prior_gradient_geometry"]
                    if work["pair_calls_completed"] == 1:
                        work["forward_rows_completed"] = 999

            return original_fit(*args, **dict(options, check_boundary=boundary))

        monkeypatch.setattr(gradients, "run_prior_gradient_readouts", capture)
        monkeypatch.setattr(pilot, "_score_checkpoint", fit)

    def read(state, pair):
        assert len(fixture.captured["finished"]) == 1
        assert not fixture.work["probes"]
        assert all(e.role == "train" and len(e.transitions[0].observation) == 296 for e in pair)
        calls.append((state.completed_updates, pair[0].context))
        work = dict.fromkeys(gradients.NATIVE_WORK_KEYS, 0)
        work.update(
            snapshot_validations_attempted=1,
            snapshot_validations_completed=1,
            forward_calls_attempted=2,
            forward_calls_completed=2,
            forward_rows_attempted=18,
            forward_rows_completed=18,
            backward_calls_attempted=2,
            backward_calls_completed=2,
            backward_rows_attempted=18,
            backward_rows_completed=18,
            projection_calls_attempted=2,
            projection_calls_completed=2,
            projection_rows_attempted=18,
            projection_rows_completed=18,
            projection_array_bytes_returned=42624,
            failure_work_may_be_unobserved=failure is True,
        )
        result: dict[str, Any] = dict(
            status="incomplete" if failure is True else "complete",
            reason="scripted fault" if failure is True else None,
            work=work,
            losses=[],
            joint_geometry=None,
            per_parameter_geometry=None,
        )
        if failure is True:
            raise GradientDiagnosticFailed(result)
        return result

    monkeypatch.setattr(gradients, "diagnose_prior_gradients", read)
    report = pilot.run_pilot(**FLAGS, prior_gradient_geometry=True)
    row = report["methods"][0]
    if failure:
        assert report["status"] == "incomplete"
        assert calls == [(0, "A")]
        slot = row["readouts"][0]
        assert slot["status"] == "failed" and len(slot["scores"]) == 6
        assert slot["prior_gradient_geometry"][0]["diagnostic"]["status"] == (
            "incomplete" if failure is True else "complete"
        )
        assert not fixture.work["probes"]
        assert report["work"]["prior_gradient_geometry"]["forward_rows_completed"] == 18
        assert report["work"]["failure_work_may_be_unobserved"]
    else:
        assert report["status"] == "complete", report["reason"]
        assert calls == [(p, c) for p in (0, 32, 64, 96, 128) for c in ("A", "B", "C_train")]
        assert all(
            len(slot["scores"]) == 6 and len(slot["prior_gradient_geometry"]) == 3
            for slot in row["readouts"]
        )
        assert all("prior_gradient_geometry" not in slot["report"] for slot in row["probes"])
        work = report["work"]["prior_gradient_geometry"]
        assert work["pair_calls_completed"] == 15
        assert work["forward_rows_completed"] == work["backward_rows_completed"] == 270
        assert work["geometry_reductions_completed"] == 75
        assert work["projection_array_bytes_returned"] == 639360
        assert (
            report["work"]["native_train_calls"] == 160
        )  # scripted only; unchanged benchmark/probe doses
        assert report["work"]["known_raw_score_forwards_completed"] == 30
        assert row["source_unchanged"] and row["actor_unchanged"]
