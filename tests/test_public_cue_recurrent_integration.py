"""Scripted wiring/refusal checks; no real native fit or physics allowance."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_minigrid_paired_bc_smoke as cli
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import minigrid_public_cue_recurrent_pilot as pilot
from src.adapters import minigrid_public_cue_recurrent_learner as port
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from test_public_cue_recurrent_learner import data, configure_scripted_native


@pytest.fixture
def scripted(monkeypatch):
    configure_scripted_native(monkeypatch)


def sources():
    result = []
    for cue in ("blue", "red"):
        x, y = data(cue)
        steps = tuple(
            DemonstrationTransition(
                i,
                tuple(float(v) for v in x[min(i, 1)]),
                int(y[min(i, 1), 0]),
                0.0,
                tuple(float(v) for v in x[1]),
                i == 8,
                7,
            )
            for i in range(9)
        )
        result.append(
            pilot.shared._batch(
                DemonstrationEpisode(f"minigrid-source-A-{cue}", "A", "train", steps, 7)
            )
        )
    return tuple(result)


def budget():
    return ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=0,
            max_replay_examples=0,
            max_wall_seconds=100,
            max_process_rss_bytes=pilot.RSS_LIMIT,
        ),
        lambda: 0.0,
    )


def endpoint(cold, point):
    return replace(
        cold,
        heads=tuple(
            replace(h, completed_updates=point // 2, training_examples=point * 9 // 2)
            for h in cold.heads
        ),
    )


def readout(point):
    return dict(
        updates=point,
        method=port.METHOD,
        role="train",
        status="complete",
        state_unchanged=True,
        scores=[
            dict(
                context="A",
                cue=c,
                ce=2.0 if not point else 0.3,
                exact_sequence_accuracy=float(bool(point)),
            )
            for c in ("blue", "red")
        ],
        episodes=[
            dict(
                context="A",
                cue=c,
                status="complete",
                success=bool(point),
                environment=dict(closed=True),
                **{"return": 0.49375 if point else 0.0},
            )
            for c in ("blue", "red")
        ],
    )


@pytest.mark.parametrize(
    "flags", [[], ["--endpoint-first"], ["--public-history"], ["--public-cue"]]
)
def test_should_describe_each_distinct_cli_recipe_without_native_work(flags, capsys):
    import json
    from src.adapters.minigrid_paired_bc_pilot import describe_protocol

    assert cli.main(["--describe", *flags]) == 0
    actual = json.loads(capsys.readouterr().out)
    expected = (
        pilot.describe_protocol()
        if flags == ["--public-cue"]
        else describe_protocol(
            endpoint_first=flags == ["--endpoint-first"],
            public_history=flags == ["--public-history"],
        )
    )
    assert actual == expected


@pytest.mark.parametrize("other", ["--endpoint-first", "--public-history"])
def test_should_refuse_mixed_recipe_flags_before_work(other):
    with pytest.raises(SystemExit) as error:
        cli.main(["--describe", "--public-cue", other])
    assert error.value.code == 2


def test_should_refuse_replacing_an_output_before_native_work(tmp_path):
    output = tmp_path / "original.json"
    output.write_text("original")
    with pytest.raises(SystemExit):
        cli.main(["--public-cue", "--output", str(output)])
    assert output.read_text() == "original"


@pytest.mark.parametrize("point", [0, 128])
def test_should_bind_raw_recurrent_checkpoint_to_existing_kernel(scripted, monkeypatch, point):
    owner = port.PublicCueRecurrentLearner()
    state = endpoint(owner.snapshot_state(), point)
    packets = sources()
    seen: list[Any] = []

    def kernel(*a, **k):
        seen.append((a, k))
        return {"ok": True}

    monkeypatch.setattr(evaluation, "_run_checkpoint", kernel)
    assert evaluation.evaluate_public_cue_recurrent_checkpoint(
        owner, state, updates=point, sources=packets, budget=budget(), check_boundary=lambda: None
    ) == {"ok": True}
    args, options = seen[0]
    assert args[0] is owner and options["sources"] is packets
    assert options["public_history"] is False and options["require_initial_source"] is True
    assert options["method"] in evaluation._RECURRENT_METHODS
    assert [(c, r) for _, c, r in options["live_plan"]] == [("blue", "train"), ("red", "train")]
    assert options["memory_copier"](state).completed_updates == point
    options["extra_check"]()


@pytest.mark.parametrize(
    "bad", ["point_bool", "point32", "foreign", "clocks", "policy", "order", "role", "budget"]
)
def test_should_refuse_invalid_readout_before_kernel(scripted, monkeypatch, bad):
    owner: Any = port.PublicCueRecurrentLearner()
    state = endpoint(owner.snapshot_state(), 128)
    packets, limits, point = sources(), budget(), 128
    if bad == "point_bool":
        point = True
    if bad == "point32":
        point = 32
    if bad == "foreign":
        owner = object()
    if bad == "clocks":
        state = replace(
            state, heads=(replace(state.heads[0], training_examples=575), state.heads[1])
        )
    if bad == "policy":
        p = port.PublicCueRecurrentPolicy(port.native.TorchRecurrentPolicy(148, 2, 7, 17))
        state = endpoint(port.PublicCueRecurrentLearner(p).snapshot_state(), 128)
    if bad == "order":
        packets = packets[::-1]
    if bad == "role":
        packets = (replace(packets[0], role="final_test"), packets[1])
    if bad == "budget":
        limits = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    monkeypatch.setattr(
        evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("kernel must not start")
    )
    with pytest.raises(ValueError):
        evaluation.evaluate_public_cue_recurrent_checkpoint(
            owner, state, updates=point, sources=packets, budget=limits, check_boundary=lambda: None
        )


@pytest.fixture
def wiring(scripted, monkeypatch):
    from scripts import run_minigrid_history_evaluation_smoke as original

    class Rng:
        def clone(self):
            return self

    rng = Rng()
    monkeypatch.setattr(
        pilot,
        "require_torch",
        lambda: SimpleNamespace(get_rng_state=lambda: rng, equal=lambda a, b: a is b),
    )
    packets = sources()
    monkeypatch.setattr(pilot, "_load_reference", lambda: {})
    monkeypatch.setattr(
        pilot,
        "_admit_reference",
        lambda r: dict(sources=packets, provenance=dict(reference_path="literal-scripted-source")),
    )
    monkeypatch.setattr(original, "_fixed_a_packets", lambda s: tuple(s[i % 2] for i in range(128)))
    calls: list[Any] = []

    def train(row, values, memory, deadline, check, *, learner_factory):
        check()
        source = learner_factory(port.METHOD)
        cold = source.snapshot_state()
        row.update(
            status="trained",
            charged_updates=128,
            native_training_examples=1152,
            source_unchanged=True,
            actor_unchanged=True,
        )
        calls.append(("train", len(values)))
        return dict(
            source=source,
            states=[(p, endpoint(cold, p)) for p in pilot.shared.POINTS],
            stream=object(),
            budget=object(),
            guard=check,
        )

    monkeypatch.setattr(pilot.shared, "_train_arm", train)

    def observe(*args, **kw):
        calls.append(("read", kw["updates"]))
        kw["check_boundary"]()
        return deepcopy(readout(kw["updates"]))

    monkeypatch.setattr(pilot, "evaluate_public_cue_recurrent_checkpoint", observe)
    return calls


def test_should_train_first_then_read_fixed_endpoints_through_existing_ports(wiring):
    result = pilot.run_pilot()
    assert result["status"] == "complete", result.get("reason")
    assert wiring == [("train", 128), ("read", 0), ("read", 128)]
    assert result["outcomes"]["joint_acquisition"] is True
    assert result["numpy_rng_unchanged"] is result["torch_rng_unchanged"] is True
    assert result["readout_budget"]["updates_completed"] == 0


def test_should_preserve_partial_readout_failure_and_skip_later_endpoint(wiring, monkeypatch):
    def fail(*a, **k):
        raise evaluation.MiniGridEvaluationFailed(dict(readout(0), reason="literal failure"))

    monkeypatch.setattr(pilot, "evaluate_public_cue_recurrent_checkpoint", fail)
    result = pilot.run_pilot()
    assert result["status"] == "incomplete" and result["outcomes"] is None
    assert result["training"]["charged_updates"] == 128
    assert len(result["evaluations"]) == 1 and wiring == [("train", 128)]
    assert result["readout_budget"]["updates_completed"] == 0


def test_should_refuse_changed_paid_prefix_and_export_original_charges(wiring, monkeypatch):
    original_train = pilot.shared._train_arm
    captured = {}

    def train(row, *a, **k):
        captured["row"] = row
        return original_train(row, *a, **k)

    def observe(*a, **k):
        captured["row"]["charged_updates"] = 0
        return readout(k["updates"])

    monkeypatch.setattr(pilot.shared, "_train_arm", train)
    monkeypatch.setattr(pilot, "evaluate_public_cue_recurrent_checkpoint", observe)
    result = pilot.run_pilot()
    assert result["status"] == "incomplete" and result["outcomes"] is None
    assert result["training"]["charged_updates"] == 128
    assert len(result["evaluations"]) == 1


@pytest.mark.parametrize("change", ["partial", "reordered", "cleanup", "source", "state"])
def test_should_refuse_unknown_or_incomplete_acquisition_claims(change):
    records = [readout(0), readout(128)]
    if change == "partial":
        records.pop()
    if change == "reordered":
        records.reverse()
    if change == "cleanup":
        records[-1]["episodes"][0]["environment"]["closed"] = False
    if change == "source":
        records[-1]["scores"][0]["context"] = "D"
    if change == "state":
        records[-1]["state_unchanged"] = False
    with pytest.raises(ValueError):
        pilot._outcomes(records)
