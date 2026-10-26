"""From-birth orchestration with scripted ports; no numerical or native outcomes."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import sys
from typing import Any

import pytest
import numpy as np

from scripts import minigrid_affine_workflow as workflow
from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_affine_probe as original
from src.adapters import minigrid_affine_rehearsal as private
from src.adapters import minigrid_policy_evaluation as evaluation
from test_after_c_linear_games import state as reference_state
from test_affine_rehearsal_return_games import state as private_state, phase
from test_catalogue_capacity import sources
from test_minigrid_affine_probe import packets
from test_portable_affine_workflow import complete_training


def comparison():
    limits = dict(
        max_training_updates=4,
        max_wall_seconds=90,
        max_replay_examples=54,
        max_hidden_width=None,
        max_process_rss_bytes=512 * 1024**2,
    )

    def budget(n, replay):
        own = limits | dict(max_training_updates=n, max_replay_examples=replay)
        return dict(
            initial_limits=own,
            limits=deepcopy(own),
            updates_completed=n,
            replay_examples_completed=replay,
            started_at=1.0,
            observed_budget_started_at=1.0,
            ended_at=2.0,
        )

    arms = []
    for name, replay in (("current_only", 0), ("joint_ABC_F", 54)):
        arms.append(
            dict(
                arm=name,
                status="complete",
                source_native_state_unchanged=True,
                source_fork_calls=1,
                budget=budget(2, replay),
                updates_accounted_by_study=2,
                committed_old_TRAIN_reuse_rows=replay,
                committed_extra_F_rows=54 - replay,
                attempts=[
                    dict(
                        update=n,
                        fit_rows_attempted=36,
                        old_TRAIN_reuse_charged=replay // 2,
                        extra_F_exposures_charged=(54 - replay) // 2,
                    )
                    for n in (1, 2)
                ],
                probe=dict(completed_updates=2, completed_training_examples=72),
                old_task_readouts=[dict(updates=n) for n in (0, 1, 2)],
                native_endpoint=asdict(private_state()),
            )
        )
    return dict(
        definition="minigrid_private_affine_rehearsal_comparison_v1",
        status="complete",
        elapsed_seconds=1.0,
        budget=budget(4, 54),
        arms=arms,
    )


@pytest.fixture
def scripted(monkeypatch):
    box = dict(
        training=complete_training(),
        state=reference_state(),
        comparison=comparison(),
        calls=[],
        collection=None,
        selected=None,
    )
    monkeypatch.setattr(workflow, "_require_minigrid", lambda: None)
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: sources())

    def train(episodes, **options):
        box["calls"].append("training")
        assert options["retain_arrived"] is options["abc_catalogue"] is True
        options["check_boundary"]()
        options["endpoint_observer"](box["state"])
        return box["training"]

    def collect(check, reports, *, distinct_probe):
        box["calls"].append("collection")
        assert distinct_probe is True
        box["collection"] = reports
        for role in ("probe_train", "probe_test"):
            for cue in ("blue", "red"):
                check()
                reports.append(dict(status="complete", closed=True, role=role, cue=cue))
        return packets()

    def compare(episodes, state, selected, **options):
        box["calls"].append("private")
        assert state is box["state"] and state.completed_updates == 6
        assert len(episodes) == 6 and options["fresh_reference"] is True
        assert [e.role for e in selected] == ["probe_train"] * 2 + ["probe_test"] * 2
        box["selected"] = selected
        options["check_boundary"]()
        return box["comparison"]

    def games(source, state, *, arm, check_boundary, **options):
        box["calls"].append(arm)
        assert source.snapshot_state().completed_updates == 0
        assert (state.completed_updates, state.training_examples) == (8, 180)
        assert options["budget"].updates_completed == 0
        check_boundary()
        return phase(arm)

    monkeypatch.setattr(cli, "run_context_linear_stream", train)
    monkeypatch.setattr(original.reserved, "_collect_probe_sources", collect)
    monkeypatch.setattr(private, "compare_private_rehearsal", compare)
    monkeypatch.setattr(evaluation, "evaluate_context_linear_rehearsal_checkpoint", games)
    return box


def test_should_complete_both_private_arms_from_current_run_without_historical_readers(
    scripted, monkeypatch
):
    from scripts import minigrid_affine_probe_games as saved

    for module, name in (
        (cli, "_qualified_after_c_bytes"),
        (saved, "_qualified_probe_bytes"),
        (saved, "_qualified_rehearsal_bytes"),
    ):
        monkeypatch.setattr(module, name, lambda *a: pytest.fail("historical read"))
    result = workflow.run_workflow(private_adaptation=True)
    assert result["status"] == "complete", result["reason"]
    assert scripted["calls"] == ["training", "collection", "private", "current_only", "joint_ABC_F"]
    assert result["TRAIN_reference_unchanged"] is True
    assert result["attempted_phases"] == ["training", "collection", "private_comparison", "games"]
    assert len(result["collection"]) == len(result["probe_sources"]["episodes"]) == 4
    assert result["probe_sources"]["released_train_packets"] == 16
    assert result["probe_sources"]["selected_train_packets"] == 2
    assert (
        result["readout_budget"]["updates_completed"]
        == result["readout_budget"]["replay_examples_completed"]
        == 0
    )
    assert all(e["success"] is False for p in result["games"] for e in p["episodes"])


def test_should_fail_missing_runtime_before_data_owners_or_training(monkeypatch):
    def absent():
        raise RuntimeError("missing optional runtime")

    monkeypatch.setattr(workflow, "_require_minigrid", absent)
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("data"))
    monkeypatch.setattr(workflow, "ContextLinearLearner", lambda: pytest.fail("owner"))
    result = workflow.run_workflow(private_adaptation=True)
    assert result["status"] == "failed" and result["attempted_phases"] == []
    assert result["TRAIN_reference_unchanged"] is None


@pytest.mark.parametrize("fault", ["initial_role", "changed_during_collection"])
def test_should_protect_original_train_examples_through_private_collection(
    scripted, monkeypatch, fault
):
    episodes = sources()
    if fault == "initial_role":
        episodes = (replace(episodes[0], role="probe_train"), *episodes[1:])
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: episodes)

    def collect(check, reports, **options):
        reports.append(dict(status="complete", closed=True, paid_steps=9))
        object.__setattr__(episodes[0], "context", "unobserved replacement")
        check()
        pytest.fail("changed TRAIN input accepted")

    monkeypatch.setattr(original.reserved, "_collect_probe_sources", collect)
    result = workflow.run_workflow(private_adaptation=True)
    assert result["status"] == "failed" and result["private_comparison"] is None
    if fault == "initial_role":
        assert scripted["calls"] == []
    else:
        assert scripted["calls"] == ["training"]
        assert result["collection"][0]["paid_steps"] == 9


@pytest.mark.parametrize("fault", ["status", "source", "receipt", "charge", "endpoint"])
def test_should_refuse_incomplete_training_before_fresh_collection(scripted, fault):
    if fault == "status":
        scripted["training"]["status"] = "failed"
    elif fault == "source":
        scripted["training"]["source_unchanged"] = False
    elif fault == "receipt":
        scripted["training"]["committed_receipts"].pop()
    elif fault == "charge":
        scripted["training"]["budget"]["updates_completed"] = 5
    else:
        scripted["state"] = replace(scripted["state"], training_examples=90)
    result = workflow.run_workflow(private_adaptation=True)
    assert result["status"] == "failed" and scripted["calls"] == ["training"]
    assert result["training"] is scripted["training"]


@pytest.mark.parametrize("fault", ["closed", "replacement", "release", "exception"])
def test_should_stop_collection_and_keep_original_closed_or_failed_prefix(
    scripted, monkeypatch, fault
):
    saved = deepcopy(scripted["training"])

    def collect(check, reports, **options):
        reports.append(dict(status="complete", closed=True, paid_steps=9))
        check()
        if fault == "closed":
            reports.append(dict(status="failed", closed=False, paid_steps=3))
        elif fault == "replacement":
            reports[0] = deepcopy(reports[0])
        elif fault == "exception":
            raise RuntimeError("collector paid work then failed")
        else:
            reports.extend(dict(status="complete", closed=True) for _ in range(3))
            train, test = packets()
            return train[:2], test
        check()
        pytest.fail("invalid collection admitted")

    monkeypatch.setattr(original.reserved, "_collect_probe_sources", collect)
    result = workflow.run_workflow(private_adaptation=True)
    assert result["status"] == "failed" and result["private_comparison"] is None
    assert result["training"] == saved and result["collection"][0]["paid_steps"] == 9
    assert "private" not in scripted["calls"]


@pytest.mark.parametrize(
    "fault", ["failed", "rows", "attempt", "prior", "replacement", "reference"]
)
def test_should_refuse_bad_private_result_or_changed_preceding_evidence(
    scripted, monkeypatch, fault
):
    saved = deepcopy(scripted["training"])

    def compare(*args, **options):
        if fault == "failed":
            scripted["comparison"]["status"] = "failed"
        elif fault == "rows":
            scripted["comparison"]["arms"][0]["native_endpoint"]["training_examples"] = 179
        elif fault == "attempt":
            scripted["comparison"]["arms"][1]["attempts"][0]["old_TRAIN_reuse_charged"] = 0
        elif fault == "prior":
            scripted["training"]["budget"]["updates_completed"] = 0
        elif fault == "replacement":
            scripted["training"]["committed_receipts"] = deepcopy(
                scripted["training"]["committed_receipts"]
            )
        else:
            object.__setattr__(scripted["state"], "training_examples", 999)
        # Return first so failed paid comparison evidence is captured before guard.
        return scripted["comparison"]

    monkeypatch.setattr(private, "compare_private_rehearsal", compare)
    result = workflow.run_workflow(private_adaptation=True)
    assert result["status"] == "failed" and result["games"] is None
    assert result["training"] == saved and result["private_comparison"] is scripted["comparison"]
    if fault == "reference":
        assert result["TRAIN_reference_unchanged"] is False


@pytest.mark.parametrize(
    "fault", ["partial", "closed", "prior", "replacement", "source", "limits", "charge", "clock"]
)
def test_should_keep_paid_game_prefix_and_refuse_changed_authority(scripted, monkeypatch, fault):
    returned = []

    def games(source, state, *, arm, budget, check_boundary):
        returned.append(arm)
        if arm == "current_only":
            current = phase(arm)
            scripted["first_game"] = current
            return current
        if fault == "partial":
            raise workflow.MiniGridEvaluationFailed(
                dict(status="failed", reason="paid partial", episodes=[dict(environment_steps=3)])
            )
        if fault == "closed":
            result = phase(arm)
            result["episodes"][0]["environment"]["closed"] = False
            return result
        if fault == "prior":
            scripted["first_game"]["status"] = "changed"
        elif fault == "replacement":
            scripted["first_game"]["episodes"] = deepcopy(scripted["first_game"]["episodes"])
        elif fault == "source":
            source.restore_state(state)
        elif fault == "limits":
            object.__setattr__(budget.budget, "max_training_updates", 1)
        elif fault == "charge":
            budget.updates_completed = 1
        else:
            budget.started_at -= 1
        check_boundary()
        pytest.fail("changed evidence accepted")

    monkeypatch.setattr(evaluation, "evaluate_context_linear_rehearsal_checkpoint", games)
    result = workflow.run_workflow(private_adaptation=True)
    assert result["status"] == "failed" and returned == ["current_only", "joint_ABC_F"]
    assert result["games"][0]["status"] == "complete"
    assert result["private_comparison"]["budget"]["updates_completed"] == 4
    if fault in ("partial", "closed"):
        assert len(result["games"]) == 2
    if fault == "charge":
        assert result["readout_budget"]["updates_completed"] == 1


@pytest.mark.parametrize(
    "args",
    [
        ["--describe"],
        ["--source", "missing"],
        ["--retained-source", "missing"],
        ["--fit-updates", "2"],
        ["--example-affine-play"],
    ],
)
def test_should_describe_without_work_and_refuse_conflicting_adaptation_options(
    monkeypatch, capsys, args
):
    monkeypatch.setattr(workflow, "run_workflow", lambda **k: pytest.fail("work"))
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("IO"))
    monkeypatch.setattr(sys, "argv", ["control", "--example-affine-adaptation", *args])
    if args == ["--describe"]:
        cli.main()
        result = json.loads(capsys.readouterr().out)
        assert result["new_fit_calls"] == 10 and result["new_fit_rows"] == 252
        assert result["returned_layout"]["branch_row"] == 3
        assert result["private"]["upstream_paid_fits"] == 6
    else:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2


def test_should_stop_at_original_whole_workflow_deadline_before_training(scripted, monkeypatch):
    clock = iter((10.0, 181.0, 182.0, 182.0))
    monkeypatch.setattr(workflow, "monotonic", lambda: next(clock))
    result = workflow.run_workflow(private_adaptation=True)
    assert result["status"] == "failed" and "deadline" in result["reason"]
    assert scripted["calls"] == [] and result["workflow_budget"]["started_at"] == 10.0


def test_should_preserve_historical_default_provenance_and_refuse_ambiguous_flags():
    assert private.describe_protocol()["upstream_paid_fits"] == 12
    assert private.describe_protocol()["upstream_paid_rows"] == 198
    assert private.describe_protocol(fresh_reference=True)["upstream_paid_rows"] == 108
    assert workflow.describe_workflow()["definition"] == "portable_affine_train_and_play_v1"
    for function, options in (
        (private.describe_protocol, dict(fresh_reference=1)),
        (workflow.run_workflow, dict(private_adaptation=1)),
        (workflow.describe_workflow, dict(private_adaptation=1)),
    ):
        with pytest.raises(ValueError):
            function(**options)


@pytest.mark.parametrize("stop_after_first", [False, True])
def test_should_check_outer_boundary_between_absolute_stream_updates_without_renewal(
    monkeypatch, stop_after_first
):
    from dataclasses import dataclass
    from src.app import coupled_attribution as stream_module
    from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
    from src.core import categorical_probability_scores

    @dataclass
    class Receipt:
        update_number: int

    stops = []
    sessions = []

    class ScriptedStream:
        def __init__(self, source, training, tests, **options):
            self.budget = options["budget"]
            self.observe = options["observe_checkpoint"]
            self.applied_updates = []
            sessions.append(self.budget)
            self.observe(0, source.snapshot_state(), None)

        def run_until(self, stop):
            assert stop == len(self.applied_updates) + 1
            stops.append(stop)
            self.budget.record_update()
            self.applied_updates.append(Receipt(stop))
            if stop in (2, 4, 6):
                base = reference_state()
                rows = {2: (9, 18), 4: (18, 54), 6: (27, 108)}[stop]
                saved = replace(
                    base,
                    completed_updates=stop,
                    training_examples=rows[1],
                    heads=tuple(
                        replace(h, fit_rows=rows[0], singular_values=(0.0,) * rows[0])
                        for h in base.heads
                    ),
                )
                self.observe(stop, saved, None)

        def finish(self):
            return dict(scripted=True)

    def boundary():
        if stop_after_first and stops:
            raise TimeoutError("original outer deadline")

    monkeypatch.setattr(stream_module, "AttributionStream", ScriptedStream)
    monkeypatch.setattr(
        ContextLinearLearner, "predict", lambda self, x: np.full((len(x), 7), 1 / 7)
    )
    monkeypatch.setattr(categorical_probability_scores, "probability_scores", lambda *a: {})
    endpoints: list[Any] = []
    result = cli.run_context_linear_stream(
        sources(),
        retain_arrived=True,
        abc_catalogue=True,
        endpoint_observer=endpoints.append,
        check_boundary=boundary,
    )
    assert len(sessions) == 1
    if stop_after_first:
        assert result["status"] == "failed" and "outer deadline" in result["reason"]
        assert stops == [1] and endpoints == []
        assert result["budget"]["updates_completed"] == 1
        assert result["committed_receipts"] == [dict(update_number=1)]
    else:
        assert result["status"] == "complete", result["reason"]
        assert stops == list(range(1, 7)) and result["observed_checkpoints"] == [0, 2, 4, 6]
        assert len(endpoints) == 1 and endpoints[0].completed_updates == 6
