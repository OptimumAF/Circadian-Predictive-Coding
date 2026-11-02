"""Sequential composition uses real stream/guards and scripted numerical ports."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import sys

import pytest

from scripts import recurrent_readout_example as example
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import minigrid_recurrent_readout as readout
from src.adapters.minigrid_learning_pilot import _batch
from src.app.coupled_attribution import _same_value
from test_recurrent_readout import scripted as scripted, fake_loop as fake_loop, nine, budget


def sources():
    return tuple(_batch(nine(cue, c)) for c in ("A", "B", "C_train") for cue in ("blue", "red"))


def fitted(point):
    packets = sources()
    owner = readout.RecurrentReadoutLearner()
    for packet in example._recurring_packets(packets)[:point]:
        owner.train_episode(packet.episode)
    return owner, owner.snapshot_state(), packets[: min(point, 6)]


def observed(point, *, lost=True, initially=True, returned=True):
    episodes = []
    current = {2: "A", 4: "B", 6: "C_train", 8: "A"}[point]
    for context in ("A", "B", "C_train")[: min(point, 6) // 2]:
        for cue in ("blue", "red"):
            success = context == current
            if context == "A":
                success = initially if point == 2 else returned if point == 8 else not lost
            episodes.append(
                dict(
                    context=context,
                    cue=cue,
                    success=success,
                    status="complete",
                    state_unchanged=True,
                    environment=dict(closed=True),
                    **{"return": 0.5 if success else 0.0},
                )
            )
    return dict(status="complete", state_unchanged=True, episodes=episodes)


@pytest.fixture
def app(scripted, monkeypatch):
    monkeypatch.setattr(example, "_require_minigrid", lambda: None)
    monkeypatch.setattr(example, "_episodes", lambda: tuple(p.episode for p in sources()))
    monkeypatch.setattr(example, "_same_state", _same_value)

    class Rng:
        def clone(self):
            return self

    rng = Rng()
    monkeypatch.setattr(
        example,
        "require_torch",
        lambda: SimpleNamespace(get_rng_state=lambda: rng, equal=lambda a, b: a is b),
    )
    return scripted


def enclosing_report(check):
    # The production guard delegates pure report checks to this one closure.
    nested = next(
        (
            c.cell_contents
            for c in check.__closure__
            if getattr(c.cell_contents, "__name__", None) == "check_evidence"
        ),
        None,
    )
    if nested is not None:
        check = nested
    return next(
        c.cell_contents
        for c in check.__closure__
        if type(c.cell_contents) is dict and "protocol" in c.cell_contents
    )


def test_recurring_describe_is_explicit_and_cli_reads_nothing(monkeypatch, capsys):
    monkeypatch.setattr(example, "_episodes", lambda: pytest.fail("source read"))
    monkeypatch.setattr(sys, "argv", ["example", "--recurring", "--describe"])
    example.main()
    assert '"fit_points"' in capsys.readouterr().out
    assert example.describe_recurring()["context_sequence"] == ["A", "B", "C_train", "A"]
    assert example.describe()["definition"] == "recurrent_readout_train_A_native_v1"


def test_recipe_has_distinct_arrivals_and_detached_released_train_packets():
    original = sources()
    before = deepcopy(original)
    packets = example._recurring_packets(original)
    assert [p.context for p in packets] == ["A", "A", "B", "B", "C_train", "C_train", "A", "A"]
    assert len({p.episode.episode_id for p in packets}) == 8
    assert all(p.role == p.episode.role == "train" for p in packets)
    packets[-1].targets[:] = 0
    assert _same_value(original, before)


@pytest.mark.parametrize("point", [2, 4, 6, 8])
def test_facade_admits_exact_fit_clock_and_arrived_roster_then_pins_state(
    scripted, monkeypatch, point
):
    owner, state, packets = fitted(point)

    def loop(actual, actual_state, **options):
        assert actual is owner and actual_state is state
        assert options["updates"] == point and options["native_training_examples"] == 9 * point
        assert options["sources"] is packets
        assert [(v.context, cue) for v, cue, role in options["live_plan"]] == [
            (c, cue) for c in ("A", "B", "C_train")[: len(packets) // 2] for cue in ("blue", "red")
        ]
        assert options["return_metadata"]["native_recurrent_updates"] == 0
        assert _same_value(options["memory_copier"](state), state)
        options["extra_check"]()
        object.__setattr__(state, "readouts", tuple(list(state.readouts)))
        with pytest.raises(ValueError, match="replaced"):
            options["extra_check"]()
        return {"status": "scripted"}

    monkeypatch.setattr(evaluation, "_run_checkpoint", loop)
    assert (
        evaluation.evaluate_recurring_recurrent_readout_checkpoint(
            owner,
            state,
            updates=point,
            sources=packets,
            budget=budget(),
            check_boundary=lambda: None,
        )["status"]
        == "scripted"
    )


@pytest.mark.parametrize("point", [True, 0, 3, 10])
def test_unsupported_fit_point_refuses_before_any_state_work(point):
    with pytest.raises(ValueError, match="fit point"):
        evaluation.evaluate_recurring_recurrent_readout_checkpoint(
            None, None, updates=point, sources=None, budget=None, check_boundary=None
        )


@pytest.mark.parametrize("fault", ["future", "order", "role", "clock", "budget"])
def test_bad_source_or_clock_or_authority_never_enters_loop(scripted, monkeypatch, fault):
    owner, state, packets = fitted(2)
    limits, point = budget(), 2
    if fault == "future":
        packets = sources()
    elif fault == "order":
        packets = packets[::-1]
    elif fault == "role":
        packets = (replace(packets[0], role="final_test"), packets[1])
    elif fault == "clock":
        point = 4
    else:
        limits = budget(1)
    monkeypatch.setattr(
        evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("readout admitted")
    )
    with pytest.raises(ValueError):
        evaluation.evaluate_recurring_recurrent_readout_checkpoint(
            owner, state, updates=point, sources=packets, budget=limits, check_boundary=lambda: None
        )


def test_sequential_facade_reuses_real_loop_and_closes_all_fake_arrived_games(scripted, fake_loop):
    owner, state, packets = fitted(4)
    result = evaluation.evaluate_recurring_recurrent_readout_checkpoint(
        owner, state, updates=4, sources=packets, budget=budget(), check_boundary=lambda: None
    )
    assert result["status"] == "complete" and result["state_unchanged"] is True
    assert len(result["scores"]) == len(result["episodes"]) == 4
    assert all(e["environment"]["closed"] for e in result["episodes"])
    assert len(fake_loop["environments"]) == 4 and len(fake_loop["steps"]) == 8
    assert owner.completed_updates == 4 and owner.training_examples == 36


def test_actual_stream_uses_eight_fits_and_one_shared_readout_birth_without_future_scoring(
    app, monkeypatch
):
    births, readers, points = [], [], []

    def games(reader, state, **options):
        point = options["updates"]
        points.append(point)
        readers.append(reader)
        births.append((id(options["budget"]), options["budget"].started_at))
        assert state.fit_updates == (point // 2, point // 2)
        assert len(options["sources"]) == min(point, 6)
        assert reader.completed_updates == point and state.native.completed_updates == 0
        assert len(app.fits) == point
        if point == 8:
            assert enclosing_report(options["check_boundary"])["training"]["completed_updates"] == 8
        options["check_boundary"]()
        return observed(point)

    monkeypatch.setattr(example, "evaluate_recurring_recurrent_readout_checkpoint", games)
    report = example.run_recurring_example()
    assert report["status"] == "complete", report["reason"]
    assert points == [2, 4, 6, 8] and len({id(r) for r in readers}) == 4
    assert len(set(births)) == 1
    assert report["training"]["completed_updates"] == report["training_budget"]["updates"] == 8
    assert report["training"]["training_examples"] == 72
    assert [r["applied_at"] for r in report["training"]["applied_updates"]] == list(range(1, 9))
    assert [len(r["training_receipts"]) for r in report["readouts"]] == [2, 4, 6, 8]
    assert report["readout_budget"]["updates"] == report["readout_budget"]["replay"] == 0
    assert all(report["preservation"].values())


@pytest.mark.parametrize(
    "fault",
    [
        "prior_replace",
        "prior_value",
        "birth",
        "final_training",
        "protocol",
        "readouts",
        "receipt",
        "final_training_before_game",
    ],
)
def test_corrupted_prior_evidence_or_birth_stops_further_learning(app, monkeypatch, fault):
    report_ref = []

    def games(reader, state, **options):
        report = enclosing_report(options["check_boundary"])
        report_ref[:] = [report]
        if options["updates"] == 8 and fault == "final_training_before_game":
            report["training"] = deepcopy(report["training"])
        if options["updates"] == 4:
            if fault == "prior_replace":
                report["readouts"][0]["games"] = deepcopy(report["readouts"][0]["games"])
            elif fault == "prior_value":
                report["readouts"][0]["games"]["episodes"][0]["success"] = False
            elif fault == "birth":
                options["budget"].started_at += 1
            elif fault == "protocol":
                report["protocol"] = deepcopy(report["protocol"])
            elif fault == "readouts":
                report["readouts"] = deepcopy(report["readouts"])
            elif fault == "receipt":
                report["readouts"][0]["training_receipts"][0]["actor_version"] = "forged"
        options["check_boundary"]()
        return observed(options["updates"])

    monkeypatch.setattr(example, "evaluate_recurring_recurrent_readout_checkpoint", games)
    if fault == "final_training":
        original = example._recurring_outcomes

        def changed(records):
            result = original(records)
            report_ref[0]["training"] = deepcopy(report_ref[0]["training"])
            return result

        monkeypatch.setattr(example, "_recurring_outcomes", changed)
    report = example.run_recurring_example()
    assert report["status"] == "incomplete" and report["preservation"] is None
    assert "changed" in report["reason"] or "replaced" in report["reason"]
    assert len(app.fits) == (8 if fault in ("final_training", "final_training_before_game") else 4)


def test_prior_evidence_changed_during_third_fit_refuses_fourth_admission(app, monkeypatch):
    parent = []

    def games(reader, state, **options):
        parent[:] = [enclosing_report(options["check_boundary"])]
        return observed(options["updates"])

    monkeypatch.setattr(example, "evaluate_recurring_recurrent_readout_checkpoint", games)

    def features(state, raw):
        result = app.feature_port(state, raw)
        if len(app.fits) == 2:
            parent[0]["readouts"][0] = deepcopy(parent[0]["readouts"][0])
        return result

    app.feature_override = features
    report = example.run_recurring_example()
    assert report["status"] == "incomplete" and "replaced" in report["reason"]
    assert len(app.fits) == report["training_budget"]["updates"] == 3


def test_late_callback_cannot_corrupt_just_returned_game_and_attest_success(app, monkeypatch):
    pending = []
    torch = example.require_torch()
    monkeypatch.setattr(example, "require_torch", lambda: torch)

    def equal(a, b):
        if pending:
            pending[0]["episodes"][0]["success"] = False
        return a is b

    torch.equal = equal

    def games(reader, state, **options):
        result = observed(options["updates"])
        pending[:] = [result]
        return result

    monkeypatch.setattr(example, "evaluate_recurring_recurrent_readout_checkpoint", games)
    report = example.run_recurring_example()
    assert report["status"] == "incomplete" and "changed" in report["reason"]
    assert report["preservation"] is None
    assert len(app.fits) == report["training_budget"]["updates"] == 2


def test_lost_native_readout_preserves_paid_prefix_and_partial_game_record(app, monkeypatch):
    def games(reader, state, **options):
        if options["updates"] == 4:
            raise evaluation.MiniGridEvaluationFailed(
                dict(status="incomplete", reason="lost returned game", episodes=[])
            )
        return observed(options["updates"])

    monkeypatch.setattr(example, "evaluate_recurring_recurrent_readout_checkpoint", games)
    report = example.run_recurring_example()
    assert report["status"] == "incomplete" and report["preservation"] is None
    assert len(app.fits) == report["training_budget"]["updates"] == 4
    assert [r["updates"] for r in report["readouts"]] == [2, 4]
    assert report["readouts"][-1]["games"]["reason"] == "lost returned game"
    assert report["outcomes"] is None


@pytest.mark.parametrize(
    "initially,lost,returned,expected",
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
    ],
)
def test_return_success_requires_observed_acquisition_and_loss_for_recovery(
    initially, lost, returned, expected
):
    records = [
        dict(updates=p, games=observed(p, initially=initially, lost=lost, returned=returned))
        for p in (2, 4, 6, 8)
    ]
    result = example._recurring_outcomes(records)
    assert len(result["acquisition"]) == 6 and len(result["retention"]) == 10
    assert all(r["observed_recovery"] is expected for r in result["returning_A"])


def test_recovery_before_return_is_distinct_from_recovery_on_return():
    records = [dict(updates=p, games=observed(p)) for p in (2, 4, 6, 8)]
    for episode in records[2]["games"]["episodes"]:
        if episode["context"] == "A":
            episode["success"] = True
    result = example._recurring_outcomes(records)
    assert all(e["observed_recovery_by_return"] for e in result["returning_A"])
    assert not any(e["observed_recovery"] for e in result["returning_A"])


@pytest.mark.parametrize("fault", ["global", "episode", "closed"])
def test_unproved_or_unclosed_game_cannot_support_recovery(app, monkeypatch, fault):
    records = [dict(updates=p, games=observed(p)) for p in (2, 4, 6, 8)]

    def corrupt(game):
        if fault == "global":
            game["state_unchanged"] = None
        elif fault == "episode":
            game["episodes"][0]["state_unchanged"] = False
        else:
            game["episodes"][0]["environment"]["closed"] = False
        return game

    corrupt(records[1]["games"])
    with pytest.raises(ValueError, match="preserved closed"):
        example._recurring_outcomes(records)
    monkeypatch.setattr(
        example,
        "evaluate_recurring_recurrent_readout_checkpoint",
        lambda *a, **k: corrupt(observed(k["updates"])),
    )
    report = example.run_recurring_example()
    assert report["status"] == "incomplete" and "preserved closed" in report["reason"]
    assert len(app.fits) == report["training_budget"]["updates"] == 2
    assert len(report["readouts"]) == 1 and report["preservation"] is None


def test_missing_game_cannot_be_summarized_as_recovery():
    records = [dict(updates=p, games=observed(p)) for p in (2, 4, 6, 8)]
    records[1]["games"]["episodes"].pop()
    with pytest.raises(ValueError, match="roster"):
        example._recurring_outcomes(records)


def test_cli_dispatches_selected_example_with_exclusive_complete_capture(monkeypatch, tmp_path):
    target = tmp_path / "actual.json"
    monkeypatch.setattr(
        example, "run_recurring_example", lambda: dict(status="complete", reason=None)
    )
    monkeypatch.setattr(example, "run_example", lambda: pytest.fail("default run"))
    monkeypatch.setattr(sys, "argv", ["example", "--recurring", "--output", str(target)])
    with pytest.raises(SystemExit) as completion:
        example.main()
    assert completion.value.code == 0 and '"complete"' in target.read_text()
    with pytest.raises(FileExistsError):
        example.main()
