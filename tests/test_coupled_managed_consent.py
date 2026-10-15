"""Managed consent admission and handoff with real owners and fake coupled ports."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import sys
from typing import Any

import numpy as np
import pytest

import src.adapters.coupled_checkpoint_callbacks as callbacks_module
import src.adapters.coupled_managed_consent as managed
from src.adapters.coupled_learner_checkpoint import CoupledLearnerCheckpointCodec
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.managed_experience import ManagedExperienceOwner
from src.app.candidate_checkpoint import _budget_state
from src.core.experience import ExperiencePermissions
from test_coupled_repeated_handoff import FakeLearner, FakeModel
from test_managed_experience import declaration, setup
from test_experience_inbox import experience, label
from test_native_transfer_session import snapshot


_ROUTES = {
    None: "example",
    "--repeat-handoffs": "repeated_example",
    "--resource-stops": "resource_stop_example",
    "--endurance": "endurance_example",
    "--managed-consent": "managed_consent_example",
}


@pytest.mark.parametrize("option", list(_ROUTES))
@pytest.mark.parametrize("mode", ["describe", "run"])
def test_should_route_only_the_selected_managed_or_existing_cli_workflow(
    monkeypatch, capsys, option, mode
):
    import scripts.run_coupled_candidate_handoff as cli

    calls: list[Any] = []

    def forbidden():
        raise AssertionError("unselected operational workflow ran")

    def selected():
        calls.append((option, mode))
        return {"selected": (option, mode)}

    for suffix in _ROUTES.values():
        for verb in ("describe", "run"):
            monkeypatch.setattr(cli, f"{verb}_{suffix}", forbidden)
    monkeypatch.setattr(cli, f"{mode}_{_ROUTES[option]}", selected)
    monkeypatch.setattr(
        sys, "argv", ["handoff", f"--{mode}"] + ([] if option is None else [option])
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out) == json.loads(
        json.dumps({"selected": (option, mode)})
    )
    assert calls == [(option, mode)]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--managed-consent"],
        ["--describe", "--managed-consent", "--repeat-handoffs"],
        ["--run", "--managed-consent", "--resource-stops"],
        ["--run", "--managed-consent", "--endurance"],
        ["--run", "--managed-consent", "--regrant-consent"],
        ["--run", "--managed-consent", "--delete-opted-out"],
    ],
)
def test_should_refuse_conflicting_or_unfrozen_consent_controls_before_work(monkeypatch, arguments):
    import scripts.run_coupled_candidate_handoff as cli

    def forbidden():
        raise AssertionError("invalid CLI controls reached product workflow")

    for suffix in _ROUTES.values():
        for verb in ("describe", "run"):
            monkeypatch.setattr(cli, f"{verb}_{suffix}", forbidden)
    monkeypatch.setattr(sys, "argv", ["handoff", *arguments])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


class Poison:
    def __init__(self):
        self.copies = 0

    def __deepcopy__(self, memo):
        self.copies += 1
        raise AssertionError("forbidden opaque payload was copied")


def _capture(owner, runtime, gate, budget):
    return deepcopy(
        dict(
            candidate=runtime.candidate_snapshot(),
            actor=runtime.actor.snapshot_state(),
            inbox=runtime._inbox.capture_cursor(),
            budget=_budget_state(budget),
            sharing=gate.snapshot(),
            catalog=owner.declarations,
            opted_out=owner._opted_out,
            consolidation=asdict(runtime.capture_consolidation_cursor()),
        )
    )


@pytest.mark.parametrize("kind", ["source", "label"])
@pytest.mark.parametrize(
    "role", ["inner_guard", "outer_selection", "final_test", "unmanaged", "opted_out"]
)
def test_should_refuse_forbidden_metadata_before_copy_and_preserve_all_but_revision(kind, role):
    owner, _, runtime, _, gate, _, budget = setup()
    owner.declare(declaration())
    if role == "opted_out":
        owner.opt_out("person")
    poison = Poison()
    event = (
        replace(experience(), features=poison, permissions=ExperiencePermissions(True, True))
        if kind == "source"
        else replace(label(), targets=poison)
    )
    if role not in {"unmanaged", "opted_out"}:
        if kind == "source":
            event = replace(event, role=role, permissions=ExperiencePermissions(evaluation=True))
        else:
            event = replace(event, role=role)
    receiver = runtime if role == "unmanaged" else owner
    action = receiver.record_experience if kind == "source" else receiver.record_label
    before = _capture(owner, runtime, gate, budget)
    with pytest.raises(ValueError):
        action(event)
    after = _capture(owner, runtime, gate, budget)
    assert after["consolidation"]["revision"] == before["consolidation"]["revision"] + 1
    before["consolidation"]["revision"] += 1
    assert before == after
    assert poison.copies == 0 and owner._issuing is None


def test_should_describe_local_declarations_without_models_managed_ports_or_attestation(
    monkeypatch,
):
    from src.adapters.coupled_candidate_handoff import describe_example

    def forbidden(*args, **kwargs):
        raise AssertionError("description reached an operational port")

    for name in (
        "_model",
        "CoupledLearner",
        "ManagedExperienceOwner",
        "CandidateCheckpointController",
    ):
        monkeypatch.setattr(managed, name, forbidden)
    value, old = managed.describe_managed_consent_example(), describe_example()
    for key in ("families", "inference_modes", "seed", "geometry", "training", "binding", "limits"):
        assert value[key] == old[key]
    assert value["protocol_id"] == "coupled_managed_consent_engineering_v1"
    assert value["runtime_update_limit"] == 3 and value["preparation_attempt_limit"] == 1
    assert value["sharing_limits"] == dict(
        max_serving_requests=2, max_admitted_updates=3, max_updates_per_poll=1
    )
    assert value["label_arrivals"] == [3, 7, 9] and value["opt_out_at"] == 5
    assert value["lifecycle_limits"] == dict(
        max_approved_records=3, max_replay_records=3, allow_synthetic=True, allow_unverified=True
    )
    assert value["initial_poison_probes"] == 8 and value["post_optout_poison_probes"] == 2
    assert len(value["declarations"]) == 3
    for index, item in enumerate(value["declarations"]):
        assert tuple(item["key"]) == (f"e{index}", f"s{index}")
        assert item["provenance"] == dict(
            source_id="local-fixed-engineering-fixture",
            subject_id="subject-B" if index == 1 else "subject-A",
            synthetic=True,
            verified=False,
        )
        assert item["consent"] == dict(training=True, replay=True) and item["retention"] == "replay"
    assert value["native_training_calls"] == 36
    assert value["native_energy_evaluation_cap"] == 180
    assert value["native_constructor_cap"] == 48 and value["prediction_cap"] == 56
    assert value["guarded_inference_cap"] == 18
    assert value["process_launches"] == 0


def test_should_handoff_original_managed_authority_and_block_only_future_opted_out_use(monkeypatch):
    FakeLearner.instances, FakeLearner.train_calls = [], 0
    controllers: list[Any] = []
    owners: list[Any] = []
    originals: list[Any] = []

    def model(family):
        native = snapshot(family)
        if native.wake_state is not None:
            native = replace(
                native,
                wake_state=replace(
                    native.wake_state,
                    chemical_state=np.zeros(2),
                    plasticity_state=np.ones(2),
                    wake_updates=0,
                ),
            )
        return FakeModel(native)

    def restore(raw, **kwargs):
        checkpoint = CoupledLearnerCheckpointCodec().decode(raw, **kwargs)
        policy = checkpoint.policy
        return FakeLearner(
            FakeModel(checkpoint.native_state),
            learning_rate=policy.learning_rate,
            inference_steps=policy.inference_steps,
            inference_learning_rate=policy.inference_learning_rate,
            inference_policy=policy.inference_policy,
        )

    def owner(shared, **kwargs):
        actual = ManagedExperienceOwner(shared, **kwargs)
        owners.append(actual)
        return actual

    def controller(shared, **kwargs):
        assert kwargs["max_pending"] == kwargs["max_prepare_attempts"] == 1
        runtime, managed_owner = shared._runtime, owners[-1]
        originals.append(
            (
                runtime.actor,
                runtime._budget,
                runtime._inbox._clock,
                shared._sharing,
                managed_owner._catalog,
                managed_owner._opted_out,
                managed_owner._limits,
                runtime._inbox._registration_guard,
                runtime._inbox._training_guard,
            )
        )
        actual = CandidateCheckpointController(shared, **kwargs)
        controllers.append(actual)
        return actual

    monkeypatch.setattr(managed, "_model", model)
    monkeypatch.setattr(managed, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(managed, "ManagedExperienceOwner", owner)
    monkeypatch.setattr(managed, "CandidateCheckpointController", controller)
    monkeypatch.setattr(callbacks_module, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(callbacks_module, "restore_coupled_learner", restore)
    result = managed.run_managed_consent_example()
    assert (
        result["managed_consent_handoff_verified"] is result["pc_neutral_parameters_exact"] is True
    )
    assert result["full_managed_process_recovery_verified"] is False
    assert len(result["cases"]) == len(owners) == len(controllers) == 6
    assert FakeLearner.train_calls == 36
    for row, control, consent, original in zip(
        result["cases"], controllers, owners, originals, strict=True
    ):
        runtime, gate = control._shared._runtime, control._shared._sharing
        assert all(
            left is right
            for left, right in zip(
                (
                    runtime.actor,
                    runtime._budget,
                    runtime._inbox._clock,
                    gate,
                    consent._catalog,
                    consent._opted_out,
                    consent._limits,
                    runtime._inbox._registration_guard,
                    runtime._inbox._training_guard,
                ),
                original,
                strict=True,
            )
        )
        assert control._attempts == len(control._models) == 1 and len(control._pending) == 0
        assert consent._lifecycle is None and consent._opted_out == {"subject-A"}
        assert len(consent.declarations) == 3
        assert consent._limits.max_approved_records == consent._limits.max_replay_records == 3
        assert consent._limits.allow_synthetic is consent._limits.allow_unverified is True
        assert runtime._budget.updates_completed == gate.snapshot().admitted_updates == 2
        assert (
            runtime._budget.budget.max_training_updates
            == gate.snapshot().limits.max_admitted_updates
            == 3
        )
        assert not runtime._stopped and not runtime._inbox.stopped
        cursor = runtime._inbox.capture_cursor()
        assert [value.sample_id for value in cursor.experiences] == ["s0", "s1", "s2"]
        assert [value.sample_id for value in cursor.applied] == ["s0", "s1"]
        assert [value.applied_at for value in cursor.applied] == [3, 7]
        assert cursor.last_tick == 9 and cursor.completed_updates == 2
        assert len(cursor.labels) == 3 and not cursor.stopped
        assert consent._eligible(cursor.experiences[2], cursor.labels[2]) is False
        assert consent._eligible(cursor.experiences[0], cursor.labels[0]) is False
        assert consent._eligible(cursor.experiences[1], cursor.labels[1]) is True
        assert np.array_equal(cursor.experiences[2].features, [[0.2, -0.3], [-0.4, 0.6]])
        assert np.array_equal(cursor.labels[2].targets, [[1.0], [0.0]])
        for index, item in enumerate(consent.declarations):
            assert item.key == (f"e{index}", f"s{index}")
            assert item.provenance.subject_id == ("subject-B" if index == 1 else "subject-A")
            assert item.provenance.synthetic is True and item.provenance.verified is False
            assert item.provenance.source_id == "local-fixed-engineering-fixture"
            assert (
                item.consent.training is item.consent.replay is True and item.retention == "replay"
            )
        assert row["actor_unchanged"] and row["complete_continuation_exact"]
        assert row["same_managed_authority"] and row["same_actor_budget_clock_gate"]
        assert row["identity"]["original"] == row["identity"]["current"]
        assert row["builds"] == ["correct_policy"]
        assert row["preparation_attempts"] == row["prepared_models"] == 1
        assert row["pending_checkpoints"] == row["poison_copy_attempts"] == 0
        assert row["eligibility_s2"] is False and row["pending_payloads_retained"]
        assert row["waiting_at5"] == row["ineligible_at9"] == dict(updates=[], deferred_reason=None)
        assert row["before_handoff"] == row["after_handoff"]
        assert row["after_opt_out"] == row["after_stale_refusal"]
        expected_optout = deepcopy(row["before_opt_out"])
        expected_optout["managed"]["opted_out"] = ["subject-A"]
        expected_optout["consolidation_cursor"]["revision"] += 1
        assert row["after_opt_out"] == expected_optout
        assert row["stale_refusal"] and row["retired_owner_refusal"] and row["used_token_refusal"]
        assert len(row["refusals"]) == 12
        assert {item["phase"] for item in row["refusals"]} == {
            *(
                f"{role}_{kind}"
                for role in ("inner_guard", "outer_selection", "final_test")
                for kind in ("source", "label")
            ),
            "unmanaged_source",
            "unmanaged_label",
            "post_optout_source",
            "post_optout_label",
            "regrant_subject_a",
            "fourth_declaration",
        }
        for refusal in row["refusals"]:
            expected_delta = (
                0 if refusal["phase"] in {"regrant_subject_a", "fourth_declaration"} else 1
            )
            assert refusal["revision_delta"] == expected_delta
            assert refusal["revision_after"] == refusal["revision_before"] + expected_delta
            assert refusal["poison_copy_attempts"] == 0 and refusal["complete_state_preserved"]
            assert refusal["message"]
        assert [item["sample_id"] for item in row["first_updates"]] == ["s0"]
        assert [item["sample_id"] for item in row["second_updates"]] == ["s1"]
        assert row["first_updates"][0]["applied_at"] == 3
        assert row["second_updates"][0]["applied_at"] == 7
        actor, captured, final = (
            json.loads(row[key]) for key in ("actor_a_utf8", "captured_b_utf8", "continued_c_utf8")
        )
        assert actor["policy"] == final["policy"] == row["policy"]
        if row["family"] != "pc":
            assert actor["native"]["wake_state"]["wake_updates"] == 2
            assert captured["native"]["wake_state"]["wake_updates"] == 3
            assert final["native"]["wake_state"]["wake_updates"] == 4
        assert len(json.dumps(row, separators=(",", ":"), allow_nan=False).encode()) <= 65536
    for offset in (0, 3):
        for field in ("actor_a_utf8", "captured_b_utf8", "continued_c_utf8"):
            assert (
                json.loads(result["cases"][offset][field])["native"]["arrays"]
                == json.loads(result["cases"][offset + 1][field])["native"]["arrays"]
            )
