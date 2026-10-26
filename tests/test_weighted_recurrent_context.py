"""Fixed weighted pilot composition on scripted models; no learned/game claims."""

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_minigrid_paired_bc_smoke as cli
from scripts.run_minigrid_linear_fit_control import read_training_sources
from src.adapters import minigrid_public_cue_recurrent_pilot as pilot
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import minigrid_public_cue_recurrent_learner as port
from src.adapters import torch_train_weighted_learner as weighted
from src.app.coupled_attribution import _same_value
from src.infra.minigrid_affine_example import example_affine_training
from test_public_cue_recurrent_learner import configure_scripted_native
from test_recurrent_recurring_control import budget, readout, state


@pytest.fixture(scope="module")
def episodes():
    with example_affine_training() as path:
        return read_training_sources(
            path,
            raw_frames=True,
            expected_sha256="5a368cfa8d4d7326d5d3ab4d6663a0e8e35e1e660968a68212e627ba9203d508",
        )


@pytest.fixture
def scripted(monkeypatch):
    configure_scripted_native(monkeypatch)
    monkeypatch.setattr(weighted, "_same_state", _same_value)


def weighted_readout(point):
    result = readout(point)
    if point in (64, 96):
        result["episodes"] = readout(32)["episodes"]
    result["class_balanced_training"] = True
    return result


def test_should_preserve_all_default_protocol_bytes_against_prechange_source():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/weighted-recurrent-context-20261025/pilot_before.py"
    )
    spec = importlib.util.spec_from_file_location("old_weighted_pilot_metadata", path)
    assert spec is not None and spec.loader is not None
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    for flags in (
        {},
        dict(typed_input=True),
        dict(typed_input=True, recurring_training=True),
        dict(typed_input=True, recurring_training=True, recurrent_replay="recent"),
        dict(typed_input=True, recurring_training=True, recurrent_replay="coverage"),
    ):
        assert pickle.dumps(pilot.describe_protocol(**flags)) == pickle.dumps(
            old.describe_protocol(**flags)
        )


def test_should_freeze_weighted_dose_allfits_before_readouts_and_fixed_observation_plan():
    p = pilot.describe_protocol(
        typed_input=True, recurring_training=True, class_balanced_training=True
    )
    assert p["protocol_id"] == "minigrid_public_cue_class_balanced_typed_abca128_v1"
    assert p["planned_work"]["updates"] == 128 and p["planned_work"]["training_rows"] == 1152
    assert p["planned_work"]["raw_calls"] == 30 and p["planned_work"]["raw_rows"] == 270
    assert (
        p["planned_work"]["live_episodes"] == 12 and p["planned_work"]["max_physics_steps"] == 192
    )
    assert p["native_readout_points"] == {
        "32": ["A"],
        "64": ["A"],
        "96": ["A"],
        "128": ["A", "B", "C_train"],
    }
    assert [
        p["limits"][k]
        for k in ("body_seconds", "parent_seconds", "training_seconds", "readout_seconds")
    ] == [420, 440, 400, 200]


@pytest.mark.parametrize("flag", [0, 1, None, "weighted"])
def test_should_refuse_nonboolean_class_mode_before_native_or_source_io(monkeypatch, flag):
    monkeypatch.setattr(pilot, "require_torch", lambda: pytest.fail("native reached"))
    monkeypatch.setattr(pilot, "_load_reference", lambda: pytest.fail("archive reached"))
    with pytest.raises(ValueError):
        pilot.run_pilot(class_balanced_training=flag)


@pytest.mark.parametrize(
    "flags",
    [
        {},
        dict(typed_input=True),
        dict(typed_input=True, recurring_training=True, recurrent_replay="recent"),
    ],
)
def test_should_refuse_unsupported_combinations_before_models(flags):
    with pytest.raises(ValueError):
        pilot.describe_protocol(class_balanced_training=True, **flags)


def test_should_admit_actual_sources_and_exact_whole_weighted_schedule_before_models(episodes):
    sources = pilot._typed_sources(episodes)
    packets = pilot.shared._packets(sources)
    owned = weighted._schedule(
        tuple(pilot.shared._episode(p) for p in packets), port.PublicCueRecurrentPolicy()
    )
    assert len(owned) == len({p.episode_id for p in owned}) == 128
    assert sum(len(p.transitions) for p in owned) == 1152
    assert [p.context for p in owned[::32]] == ["A", "B", "C_train", "A"]
    assert all(p.role == "train" and p.num_classes == 7 for p in owned)


@pytest.mark.parametrize(
    "point,contexts",
    [(0, ()), (32, ("A",)), (64, ("A",)), (96, ("A",)), (128, ("A", "B", "C_train"))],
)
def test_should_keep_strict_native_source_and_exact_fixed_readout_plan(
    scripted, episodes, monkeypatch, point, contexts
):
    source = port.PublicCueRecurrentLearner()
    seen = []

    def kernel(*a, **k):
        seen.append((a, k))
        return {"ok": True}

    monkeypatch.setattr(evaluation, "_run_checkpoint", kernel)
    assert evaluation.evaluate_public_cue_recurrent_checkpoint(
        source,
        state(source, point),
        updates=point,
        sources=pilot._typed_sources(episodes),
        budget=budget(),
        check_boundary=lambda: None,
        recurring_training=True,
        class_balanced_training=True,
    ) == {"ok": True}
    args, options = seen[0]
    assert args[0] is source and options["require_initial_source"] is True
    assert [(layout.context, cue, role) for layout, cue, role in options["live_plan"]] == [
        (c, q, "train") for c in contexts for q in ("blue", "red")
    ]
    assert len(options["sources"]) == 6 and options["return_metadata"] == {
        "class_balanced_training": True
    }
    options["extra_check"]()


@pytest.mark.parametrize("bad", ["flag", "no_recurring", "replay", "wrapper", "counter", "budget"])
def test_should_refuse_bad_readout_mode_source_clock_or_budget(
    scripted, episodes, monkeypatch, bad
):
    source: Any = port.PublicCueRecurrentLearner()
    saved = state(source, 64)
    options: dict[str, Any] = dict(recurring_training=True, class_balanced_training=True)
    limits = budget()
    if bad == "flag":
        options["class_balanced_training"] = 1
    elif bad == "no_recurring":
        options["recurring_training"] = False
    elif bad == "replay":
        options["recurrent_replay"] = "recent"
    elif bad == "wrapper":
        source = object()
    elif bad == "counter":
        saved = replace(saved, heads=(replace(saved.heads[0], training_examples=1), saved.heads[1]))
    else:
        object.__setattr__(limits.budget, "max_wall_seconds", 201)
    monkeypatch.setattr(
        evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("kernel reached")
    )
    with pytest.raises((ValueError, AttributeError)):
        evaluation.evaluate_public_cue_recurrent_checkpoint(
            source,
            saved,
            updates=64,
            sources=pilot._typed_sources(episodes),
            budget=limits,
            check_boundary=lambda: None,
            **options,
        )


@pytest.fixture
def wiring(scripted, episodes, monkeypatch):
    class Rng:
        def clone(self):
            return self

    rng = Rng()
    monkeypatch.setattr(
        pilot,
        "require_torch",
        lambda: SimpleNamespace(get_rng_state=lambda: rng, equal=lambda a, b: a is b),
    )
    captured, seen = {}, []

    def train(row, packets, memory, deadline, check, *, learner_factory):
        check()
        source = learner_factory(port.METHOD)
        assert type(source) is weighted.TorchTrainWeightedLearner
        assert len(source._episodes) == 128 and source._episodes[32].context == "B"
        row.update(
            status="trained",
            charged_updates=128,
            native_training_examples=1152,
            source_unchanged=True,
            actor_unchanged=True,
        )
        arm = dict(
            source=source,
            states=[(p, state(source, p)) for p in pilot.shared.POINTS],
            stream=object(),
            budget=object(),
            guard=check,
        )
        captured["arm"] = arm
        seen.append(("fit", 128))
        return arm

    def observe(source, saved, **options):
        assert seen[0] == ("fit", 128)
        assert type(source) is port.PublicCueRecurrentLearner
        assert source is captured["arm"]["source"]._learner
        assert options["class_balanced_training"] is options["recurring_training"] is True
        options["check_boundary"]()
        seen.append(("read", options["updates"]))
        return weighted_readout(options["updates"])

    monkeypatch.setattr(pilot.shared, "_train_arm", train)
    monkeypatch.setattr(pilot, "evaluate_public_cue_recurrent_checkpoint", observe)
    return episodes, captured, seen


def test_should_bind_original_wrapper_native_and_finish_all_fits_before_fixed_reads(wiring):
    episodes, _, seen = wiring
    result = pilot.run_pilot(episodes, recurring_training=True, class_balanced_training=True)
    assert result["status"] == "complete", result.get("reason")
    assert seen == [("fit", 128), *(("read", p) for p in pilot.shared.POINTS)]
    assert len(result["outcomes"]["native_outcomes"]) == 6
    assert (
        result["final_preservation"]
        is result["numpy_rng_unchanged"]
        is result["torch_rng_unchanged"]
        is True
    )
    assert result["readout_budget"]["updates_completed"] == 0


@pytest.mark.parametrize("fault", ["native", "wrapper", "budget", "prior", "lost"])
def test_should_keep_paid_prefix_and_original_clocks_when_reads_fail_or_are_replaced(
    wiring, monkeypatch, fault
):
    episodes, captured, seen = wiring
    original = pilot.evaluate_public_cue_recurrent_checkpoint

    def observe(source, saved, **options):
        result = original(source, saved, **options)
        if options["updates"] == 32:
            if fault == "native":
                captured["arm"]["source"]._learner = source.fork()
            elif fault == "wrapper":
                captured["arm"]["source"] = deepcopy(captured["arm"]["source"])
            elif fault == "budget":
                options["budget"].updates_completed = 1
            elif fault == "prior":
                captured["arm"]["states"] = deepcopy(captured["arm"]["states"])
            else:
                raise evaluation.MiniGridEvaluationFailed(
                    dict(result, status="failed", reason="lost readout return")
                )
        return result

    monkeypatch.setattr(pilot, "evaluate_public_cue_recurrent_checkpoint", observe)
    result = pilot.run_pilot(episodes, recurring_training=True, class_balanced_training=True)
    assert result["status"] == "incomplete" and result["outcomes"] is None
    assert result["training"]["charged_updates"] == 128
    assert seen == [("fit", 128), ("read", 0), ("read", 32)]
    assert len(result["evaluations"]) == 2
    assert result["final_preservation"] is None


def test_should_keep_negative_native_premises_separate_from_teacher_fit():
    result = pilot._recurring_outcomes(
        [weighted_readout(p) for p in pilot.shared.POINTS], class_balanced_training=True
    )
    assert [(r["updates"], r["context"]) for r in result["native_outcomes"]] == [
        (32, "A"),
        (64, "A"),
        (96, "A"),
        (128, "A"),
        (128, "B"),
        (128, "C_train"),
    ]
    assert result["native_outcomes"][0]["successes"] == 0
    assert "requires prior" in result["native64_96"]


@pytest.mark.parametrize("extra", [[], ["--class-balanced-training"]])
def test_should_describe_new_cli_without_source_or_native_work(monkeypatch, capsys, extra):
    monkeypatch.setattr(cli, "Path", Path)
    assert cli.main(["--describe", "--public-cue", "--recurring-training", *extra]) == 0
    actual = json.loads(capsys.readouterr().out)
    assert actual == pilot.describe_protocol(
        typed_input=True, recurring_training=True, class_balanced_training=bool(extra)
    )


@pytest.mark.parametrize(
    "flags",
    [
        ["--recurring-training"],
        ["--class-balanced-training"],
        ["--public-cue", "--class-balanced-training"],
        ["--public-history", "--recurring-training"],
    ],
)
def test_should_refuse_unsupported_cli_before_any_work(flags):
    with pytest.raises(SystemExit) as error:
        cli.main(["--describe", *flags])
    assert error.value.code == 2


def test_should_wire_firstparty_cli_source_flag_and_compact_exclusive_output(
    episodes, monkeypatch, tmp_path
):
    from scripts import run_minigrid_linear_fit_control as reader
    from src.shared import torch_runtime

    seen: list[Any] = []
    monkeypatch.setattr(reader, "read_training_sources", lambda path, **kw: deepcopy(episodes))
    monkeypatch.setattr(
        torch_runtime,
        "require_torch",
        lambda: SimpleNamespace(
            set_num_threads=lambda n: seen.append(("threads", n)),
            set_num_interop_threads=lambda n: seen.append(("interop", n)),
        ),
    )

    def run(values, **options):
        assert values == episodes
        seen.append(options)
        return dict(status="complete", class_balanced_training=True)

    monkeypatch.setattr(pilot, "run_pilot", run)
    output = tmp_path / "weighted.json"
    assert (
        cli.main(
            [
                "--public-cue",
                "--recurring-training",
                "--class-balanced-training",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert seen == [
        ("threads", 1),
        ("interop", 1),
        {"recurring_training": True, "class_balanced_training": True},
    ]
    assert output.read_bytes() == b'{"status":"complete","class_balanced_training":true}\n'


@pytest.mark.parametrize("point", [64, 96])
def test_should_run_real_readonly_kernel_with_fake_physics_and_genuine_scripted_sessions(
    scripted, episodes, monkeypatch, point
):
    from test_public_cue_probe_games import CueEnvironment

    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", CueEnvironment)
    source = port.PublicCueRecurrentLearner()
    result = evaluation.evaluate_public_cue_recurrent_checkpoint(
        source,
        state(source, point),
        updates=point,
        sources=pilot._typed_sources(episodes),
        budget=budget(),
        check_boundary=lambda: None,
        recurring_training=True,
        class_balanced_training=True,
    )
    assert result["status"] == "complete" and result["state_unchanged"] is True
    assert len(result["scores"]) == 6 and len(result["episodes"]) == 2
    assert all(
        e["context"] == "A" and e["source_role"] == "train" and e["environment"]["closed"] is True
        for e in result["episodes"]
    )
