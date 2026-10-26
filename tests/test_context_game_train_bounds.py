"""Readonly TRAIN alignment and public-cue groups; no learner/game fixtures."""

from dataclasses import replace
import json

import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters.minigrid_probe_support import summarize_context_train_bounds
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition


def frame(cue, step=0):
    values = [-0.8, -1.0, -1.0] * 49 + [-1.0]
    values[:3] = [0.2, 2.0 * cue / 5 - 1, -1.0]
    values[4] = 2.0 * (step % 6) / 5 - 1
    return tuple(values)


def sources():
    return tuple(
        DemonstrationEpisode(
            f"minigrid-source-{context}-{cue}",
            context,
            "train",
            tuple(
                DemonstrationTransition(
                    i,
                    frame(color) + frame(color, i),
                    action,
                    0.0,
                    frame(color) + frame(color, i + 1),
                    i == 8,
                    7,
                )
                for i in range(9)
            ),
            7,
        )
        for context in ("A", "B", "C_train")
        for cue, color, action in (("blue", 2, 1), ("red", 0, 2))
    )


def test_should_preserve_current_action_alignment_and_both_head_bounds():
    episodes = sources()
    r = summarize_context_train_bounds(episodes)
    assert r["current_action_alignment"] and r["source_unchanged"]
    assert r["original_bounds"]["canonical"]["row_accuracy_upper_bound"] == 1
    assert r["work"]["input_bound_reductions"] == 4
    for cue in ("blue", "red"):
        h = r["heads"][cue]
        assert h["bounds"]["rows"] == len(h["rows"]) == 27
        assert h["bounds"]["row_accuracy_upper_bound"] == 1 and not h["conflicts"]
        assert all(row["public_cue"] == cue and row["named_cue_matches"] for row in h["rows"])
    # Returned provenance must be detached from subsequent caller mutation.
    r["heads"]["blue"]["rows"][0]["action"] = 6
    assert episodes[0].transitions[0].action == 1


def test_should_expose_same_input_conflicts_with_exact_source_row_provenance():
    episodes = sources()
    first = episodes[0]
    rows = list(first.transitions)
    rows[6] = replace(rows[6], action=2)
    episodes = (replace(first, transitions=tuple(rows)),) + episodes[1:]
    r = summarize_context_train_bounds(episodes)
    blue = r["heads"]["blue"]
    assert blue["bounds"]["min_incorrect_rows"] == 1
    assert blue["bounds"]["row_accuracy_upper_bound"] == 26 / 27
    assert any(
        row["episode_id"] == first.episode_id and row["step"] == 6 and row["action"] == 2
        for group in blue["conflicts"]
        for row in group
    )


def test_should_route_by_visible_cue_even_when_metadata_names_disagree():
    episodes = sources()
    changed = tuple(
        replace(e, episode_id=episodes[i ^ 1].episode_id) for i, e in enumerate(episodes)
    )
    # Keep the required named catalogue order but swap actual public sources.
    changed = tuple(changed[i ^ 1] for i in range(6))
    r = summarize_context_train_bounds(changed)
    for cue, action in (("blue", 1), ("red", 2)):
        assert {row["action"] for row in r["heads"][cue]["rows"]} == {action}
        assert not any(row["named_cue_matches"] for row in r["heads"][cue]["rows"])


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_nontraining_sources_before_any_bound_reduction(role, monkeypatch):
    from src.adapters import minigrid_probe_support as support

    e = sources()
    e = (replace(e[0], role=role),) + e[1:]
    monkeypatch.setattr(support, "categorical_input_bounds", lambda *a: pytest.fail("reduction"))
    with pytest.raises(ValueError):
        summarize_context_train_bounds(e)


def test_should_refuse_broken_initial_history_before_reductions(monkeypatch):
    from src.adapters import minigrid_probe_support as support

    e = sources()
    rows = list(e[0].transitions)
    # Rebuild continuous observations but the initial frame is wrong immediately.
    rows[0] = replace(rows[0], observation=frame(0) + frame(2))
    e = (replace(e[0], transitions=tuple(rows)),) + e[1:]
    monkeypatch.setattr(support, "categorical_input_bounds", lambda *a: pytest.fail("reduction"))
    with pytest.raises(ValueError, match="initial history"):
        summarize_context_train_bounds(e)


def test_should_refuse_next_view_shift_from_the_episode_array_port(monkeypatch):
    import numpy as np
    from src.adapters import minigrid_probe_support as support

    def shifted(episode):
        return (
            np.array([s.next_observation for s in episode.transitions], dtype=np.float64),
            np.array([[s.action] for s in episode.transitions], dtype=np.float64),
        )

    monkeypatch.setattr(support, "episode_arrays", shifted)
    with pytest.raises(ValueError, match="alignment"):
        summarize_context_train_bounds(sources())


@pytest.mark.parametrize("option", ["--context-bounds", "--bounds-only"])
def test_should_describe_without_source_model_or_game(option, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["fit", option, "--describe"])
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("source"))
    monkeypatch.setattr(cli.context_control, "run_control", lambda *a: pytest.fail("game"))
    cli.main()
    r = json.loads(capsys.readouterr().out)
    assert r["role"] == "train" and r["native_updates"] == r["native_score_forwards"] == 0
    assert ("context_train" in r["definition"]) == (option == "--context-bounds")


@pytest.mark.parametrize("option", ["--context-perceptron", "--bounds-only", "--fit-updates"])
def test_should_refuse_conflicting_modes_or_fit_options_before_source(option, monkeypatch):
    args = ["fit", "--context-bounds", option]
    if option == "--fit-updates":
        args.append("32")
    monkeypatch.setattr("sys.argv", args)
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("source"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_dispatch_only_to_readonly_context_summary(monkeypatch, capsys):
    e = sources()
    monkeypatch.setattr("sys.argv", ["fit", "--context-bounds", "--source", "unused"])
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: e)

    def summarize(value):
        assert value is e
        return dict(status="complete", role="train")

    monkeypatch.setattr(cli, "summarize_context_train_bounds", summarize)
    monkeypatch.setattr(cli.context_control, "run_control", lambda *a: pytest.fail("game"))
    monkeypatch.setattr(cli, "diagnose_train_linear_fit", lambda *a: pytest.fail("fit"))
    cli.main()
    assert json.loads(capsys.readouterr().out)["role"] == "train"
