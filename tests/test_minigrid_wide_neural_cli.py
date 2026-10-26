"""Fixed H32 CLI admission and dispatch; all execution paths use fake diagnostics.

These cases read synthetic TRAIN records and describe recipes, but do not create
native learners, execute training, or consult final/probe data.
"""

from dataclasses import asdict
import json
import sys

import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_constructive_fit as fit
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from test_minigrid_invariant_encoding import frame


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


def forbidden(*args, **kwargs):
    raise AssertionError("CLI admission/dispatch checks must not construct or run a learner")


@pytest.fixture(autouse=True)
def no_native_work(monkeypatch):
    monkeypatch.setattr(fit, "_learner", forbidden)
    monkeypatch.setattr(fit, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(cli, "diagnose_train_fit", forbidden)
    monkeypatch.setattr(cli, "diagnose_train_linear_fit", forbidden)
    monkeypatch.setattr(cli, "summarize_train_fit_bounds", forbidden)


def saved_source(path):
    episodes = []
    for context in ("A", "B", "C_train"):
        for cue, color in (("blue", 2), ("red", 0)):
            frames = tuple(frame(direction=i, cue_color=color) for i in range(3))
            episodes.append(
                DemonstrationEpisode(
                    "minigrid-source-" + context + "-" + cue,
                    context,
                    "train",
                    tuple(
                        DemonstrationTransition(i, frames[i], i, 0.0, frames[i + 1], i == 1, 7)
                        for i in range(2)
                    ),
                    7,
                )
            )
    path.write_text(
        json.dumps(
            dict(
                training_sources=[asdict(episode) for episode in episodes],
                final_test="unavailable: never read",
                probe_data="unavailable: never read",
            )
        ),
        encoding="utf8",
    )
    return path


@pytest.mark.parametrize("method", METHODS)
def test_should_describe_each_fixed_wide_recipe_without_sources(monkeypatch, capsys, method):
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit", "--wide-neural", method, "--describe", "--fit-updates", "2"],
    )
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result == fit.describe_protocol(wide_method=method, updates=2)
    assert result["definition"] == fit.WIDE_NEURAL_DEFINITION
    assert result["wide_neural_method"] == method
    assert result["method"] == "public_invariant_" + method
    assert result["hidden_dim"] == 32 and result["seed"] == 17
    assert result["checkpoint_updates"] == [0, 2]
    assert result["probe_updates"] == result["final_episodes"] == 0


@pytest.mark.parametrize(
    "options",
    [
        ["--constructive-bp"],
        ["--perceptron"],
        ["--kernel-perceptron"],
        ["--bounds-only"],
        ["--cue-relative"],
        ["--cue-turn-gradients"],
    ],
)
def test_should_refuse_incompatible_wide_options_before_source_io(monkeypatch, options):
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(cli, "describe_constructive_protocol", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["fit", "--wide-neural", "bp", *options, "--source", "missing.json"]
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize("method", ["BP", "active_cpc", "unknown", ""])
def test_should_refuse_unknown_wide_method_before_source_io(monkeypatch, method):
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(cli, "describe_constructive_protocol", forbidden)
    monkeypatch.setattr(sys, "argv", ["fit", "--wide-neural", method, "--describe"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "options",
    [
        ["--fit-updates", "0"],
        ["--fit-updates", "513"],
        ["--fit-seconds", "0"],
        ["--fit-seconds", "91"],
    ],
)
def test_should_validate_wide_work_limits_before_source_io(monkeypatch, options):
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["fit", "--wide-neural", "bp", *options, "--source", "missing.json"]
    )
    with pytest.raises(ValueError):
        cli.main()


@pytest.mark.parametrize("method", METHODS)
def test_should_dispatch_owned_train_sources_and_preserve_output_exclusivity(
    monkeypatch, tmp_path, capsys, method
):
    source = saved_source(tmp_path / "source.json")
    output = tmp_path / "result.json"
    before = source.read_bytes()
    calls = []

    def diagnose(episodes, **options):
        calls.append((episodes, options))
        assert len(episodes) == 6 and all(episode.role == "train" for episode in episodes)
        assert all(len(step.observation) == 296 for e in episodes for step in e.transitions)
        return dict(status="complete", definition=fit.WIDE_NEURAL_DEFINITION)

    monkeypatch.setattr(cli, "diagnose_train_fit", diagnose)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit",
            "--wide-neural",
            method,
            "--fit-updates",
            "2",
            "--fit-seconds",
            "10",
            "--source",
            str(source),
            "--output",
            str(output),
        ],
    )
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert calls[0][1] == dict(updates=2, max_wall_seconds=10.0, wide_method=method)
    assert source.read_bytes() == before and json.loads(output.read_text()) == result
    assert result["definition"] == fit.WIDE_NEURAL_DEFINITION
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    with pytest.raises(FileExistsError):
        cli.main()
    assert len(calls) == 1


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_refuse_reserved_roles_before_wide_dispatch(monkeypatch, tmp_path, role):
    source = saved_source(tmp_path / "source.json")
    record = json.loads(source.read_text())
    record["training_sources"][-1]["role"] = role
    source.write_text(json.dumps(record), encoding="utf8")
    monkeypatch.setattr(sys, "argv", ["fit", "--wide-neural", "pc", "--source", str(source)])
    with pytest.raises(ValueError, match="TRAIN-source records"):
        cli.main()


@pytest.mark.parametrize("argument", ["--source", "--output"])
def test_should_refuse_describe_paths_without_reading_sources(monkeypatch, argument):
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["fit", "--wide-neural", "neutral_cpc", "--describe", argument, "unused"]
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_require_source_for_wide_execution(monkeypatch):
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(sys, "argv", ["fit", "--wide-neural", "gated_cpc"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "options,expected",
    [
        (["--constructive-bp"], dict(updates=512, max_wall_seconds=90)),
        (["--perceptron"], dict(updates=512, max_wall_seconds=90, perceptron=True)),
        (["--kernel-perceptron"], dict(updates=512, max_wall_seconds=90, kernel_perceptron=True)),
    ],
)
def test_should_omit_wide_keyword_from_existing_constructive_routes(
    monkeypatch, tmp_path, capsys, options, expected
):
    source = saved_source(tmp_path / "source.json")
    descriptions, calls = [], []

    def describe(**kwargs):
        descriptions.append(kwargs)
        return dict(status="described")

    def diagnose(episodes, **kwargs):
        calls.append(kwargs)
        return dict(status="complete")

    monkeypatch.setattr(cli, "describe_constructive_protocol", describe)
    monkeypatch.setattr(cli, "diagnose_train_fit", diagnose)
    monkeypatch.setattr(sys, "argv", ["fit", *options, "--source", str(source)])
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert descriptions == calls == [expected]


@pytest.mark.parametrize(
    "options,attribute",
    [([], "diagnose_train_linear_fit"), (["--bounds-only"], "summarize_train_fit_bounds")],
)
def test_should_preserve_nonconstructive_dispatch_without_new_options(
    monkeypatch, tmp_path, capsys, options, attribute
):
    source = saved_source(tmp_path / "source.json")
    calls = []

    def diagnose(episodes):
        calls.append(episodes)
        return dict(status="complete")

    monkeypatch.setattr(cli, "describe_constructive_protocol", forbidden)
    monkeypatch.setattr(cli, attribute, diagnose)
    monkeypatch.setattr(sys, "argv", ["fit", *options, "--source", str(source)])
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert len(calls) == 1


def test_should_save_and_report_incomplete_wide_result_before_failure_exit(
    monkeypatch, tmp_path, capsys
):
    source = saved_source(tmp_path / "source.json")
    output = tmp_path / "incomplete.json"

    def diagnose(episodes, **kwargs):
        return dict(status="incomplete", reason="injected paid failure", charged_updates=1)

    monkeypatch.setattr(cli, "diagnose_train_fit", diagnose)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit", "--wide-neural", "bp", "--source", str(source), "--output", str(output)],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "incomplete" and result["charged_updates"] == 1
    assert json.loads(output.read_text()) == result
