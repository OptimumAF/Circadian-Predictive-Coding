"""Fabricated complete declarations exercise preservation without experiment proof."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pytest

from src.app import continual_findings_development as development
from src.app.continual_confirmation_execution import digest_json
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.core.continual_metrics import TwoTaskAccuracy


def seal_bundle(bundle: dict[str, Any]) -> None:
    parts = bundle["parts"]
    for name in ("request", "result"):
        parts["audit"][name + "_sha256"] = canonical_body_identity(parts[name])["sha256"]
    bundle["files"] = {name: canonical_body_identity(body) for name, body in parts.items()}


def fabricated_inputs() -> dict[str, Any]:
    """All declared families/seeds/arms; values and verifier claims are fabricated."""
    manifest = json.loads(json.dumps(asdict(development.fixed_confirmation_manifest())))
    sources = {"fixture_source.py": "1" * 64}
    role_counts = {
        "a_train": 72,
        "a_inner_guard": 24,
        "a_outer_selection": 24,
        "b_train": 36,
        "b_inner_guard": 12,
        "b_outer_selection": 12,
    }
    bundles: list[dict[str, Any]] = []
    preflights: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for family in manifest["families"]:
        name = family["name"]
        scored, train_rows = [], []
        for seed_index, seed in enumerate(family["development_seeds"]):
            arms = []
            for arm_index, arm in enumerate(family["arms"]):
                a = 0.0 if name == "parent" and arm_index == 7 else 0.5 + 0.125 * seed_index
                values = TwoTaskAccuracy(a, 0.75, 0.5)
                scores: dict[str, Any] = {
                    "a_after_a": a,
                    "a_after_b": 0.75,
                    "b_after_b": 0.5,
                    "final_mean_task_accuracy": values.final_mean_task_accuracy,
                }
                if name == "gating":
                    scores["signed_forgetting"] = values.signed_forgetting_a
                else:
                    scores.update(
                        signed_forgetting_a=values.signed_forgetting_a,
                        retention_ratio_a=values.retention_ratio_a,
                    )
                arms.append({"name": arm, **scores})
            contrasts = (
                [
                    {
                        "left": left,
                        "right": right,
                        **{
                            field: next(a for a in arms if a["name"] == left)[field]
                            - next(a for a in arms if a["name"] == right)[field]
                            for field in development.PAIR_FIELDS
                        },
                    }
                    for left, right in family["contrasts"]
                ]
                if name not in ("gating", "replay")
                else []
            )
            scored.append({"seed": seed, "arms": arms, "contrasts": contrasts})
            prefix = "source_role_" if name in ("gating", "replay") else "role_"
            row = {
                "seed": seed,
                prefix + "counts": deepcopy(role_counts),
                prefix + "hashes": {role: "2" * 64 for role in role_counts},
            }
            if name in ("gating", "replay"):
                row["methods"] = [
                    {
                        "method": a["name"],
                        "development": {k: v for k, v in a.items() if k != "name"},
                        "fixture_training_fact": "preserved",
                    }
                    for a in arms
                ]
            train_rows.append(row)
        original_manifest = json.loads(family["development_manifest_json"])
        train = {
            "seed_results": train_rows,
            "manifest": original_manifest,
            "final_released": False,
            "outer_selection_scored": False,
        }
        if name in ("gating", "replay"):
            result = train
        else:
            result = {
                "scored_seeds": scored,
                "train_facts": train,
                "reference_sha256": canonical_body_identity(train)["sha256"],
                "final_released": False,
                "outer_selection_scored": True,
            }
        family_references = []
        kinds = [(development.PREFIXES[name], bundles, result)]
        if name not in ("gating", "replay"):
            kinds.append((name + "-factor-preflight", preflights, train))
        for prefix, destination, payload in kinds:
            for suffix in ("", "-repeat"):
                bundle = {
                    "family": name,
                    "directory": "artifacts/runs/p63-" + prefix + suffix,
                    "parts": {
                        "request": {
                            "manifest": original_manifest,
                            "source_sha256": sources,
                            "wall_limit_seconds": 120,
                            "adapter_sha256": "3" * 64,
                        },
                        "result": deepcopy(payload),
                        "audit": {"status": "completed", "elapsed_seconds": 0.5},
                    },
                }
                seal_bundle(bundle)
                destination.append(bundle)
                if destination is bundles:
                    family_references.append(
                        {
                            "directory": bundle["directory"],
                            "file_sha256": {k: v["sha256"] for k, v in bundle["files"].items()},
                            "result_bytes": bundle["files"]["result"]["byte_count"],
                            "source_sha256": sources,
                        }
                    )
        references.append({"family": name, "bundles": family_references})
    catalog = {
        "preflight_bundles": [
            {
                "family": b["family"],
                "directory": b["directory"],
                "prefix": b["family"] + "-factor-preflight",
                "files": b["files"],
            }
            for b in preflights
        ]
    }
    return {
        "schema_id": "p612_complete_current_development_inputs_v1",
        "input_catalog": catalog,
        "original_scope": {"manifest": manifest, "development_references": references},
        "scope_files": {},
        "source_sha256": sources,
        "source_map_sha256": digest_json(sources),
        "development_bundles": bundles,
        "preflight_bundles": preflights,
        "validation_facts": {
            "unchanged_development_bundle_validations": 12,
            "unchanged_preflight_result_validations": 8,
            "complete_bundle_files": 60,
            "current_before_and_after_byte_source_bindings": True,
            "new_training_scoring_or_final_access": False,
            "fresh_confirmation_reader_authority": False,
        },
    }


def reseal_development(inputs: dict[str, Any], family_index: int) -> None:
    for suffix_index in (0, 1):
        bundle = inputs["development_bundles"][2 * family_index + suffix_index]
        seal_bundle(bundle)
        ref = inputs["original_scope"]["development_references"][family_index]["bundles"][
            suffix_index
        ]
        ref.update(
            file_sha256={k: v["sha256"] for k, v in bundle["files"].items()},
            result_bytes=bundle["files"]["result"]["byte_count"],
        )


def test_should_preserve_every_original_body_metric_null_seed_and_pair_without_pooling() -> None:
    inputs = fabricated_inputs()
    before = deepcopy(inputs)
    result = development._derive_development_ledger(inputs)
    assert result["original_inputs"] == inputs == before
    assert result["coverage"]["cells"] == len(result["cells"]) == 168
    assert result["coverage"]["ordered_seed_pairs"] == len(result["paired_differences"]) == 174
    assert (
        sum(p["origin"] == "original_stored_contrast" for p in result["paired_differences"]) == 162
    )
    assert (
        sum(
            p["origin"] == "projection_of_original_cells_and_prospective_pair"
            for p in result["paired_differences"]
        )
        == 12
    )
    assert "signed_forgetting" in result["cells"][0]["scores"]
    assert "retention_ratio_a" not in result["cells"][0]["scores"]
    assert result["cells"][-1]["scores"]["retention_ratio_a"] is None
    assert result["cells"][-1]["scores"]["signed_forgetting_a"] == -0.75
    assert result["interpretation"]["evaluation_role"].startswith("original_outer_selection")
    assert result["fresh_confirmation_reader_authority"] is False
    assert result["original_p612_acceptance_complete"] is False
    assert result["new_training_scoring_or_final_access"] is False


@pytest.mark.parametrize(
    "change",
    [
        "last_endpoint",
        "last_forgetting",
        "last_pair",
        "last_seed",
        "last_role",
        "final_seal",
        "last_arm",
        "last_reference",
        "settings",
        "extra_input",
    ],
)
def test_should_reject_resealed_arithmetic_role_scope_or_reference_corruption(change: str) -> None:
    inputs = fabricated_inputs()
    for bundle in inputs["development_bundles"][-2:]:
        result = bundle["parts"]["result"]
        if change == "last_endpoint":
            result["scored_seeds"][-1]["arms"][-1]["a_after_b"] = 1.1
        elif change == "last_forgetting":
            result["scored_seeds"][-1]["arms"][-1]["signed_forgetting_a"] = 0.75
        elif change == "last_pair":
            result["scored_seeds"][-1]["contrasts"][-1]["final_mean_task_accuracy"] = 0.5
        elif change == "last_seed":
            result["scored_seeds"][-1]["seed"] = 9999
        elif change == "last_role":
            result["train_facts"]["seed_results"][-1]["role_counts"]["b_outer_selection"] = 40
        elif change == "final_seal":
            result["final_released"] = True
        elif change == "last_arm":
            result["scored_seeds"][-1]["arms"].pop()
        elif change == "last_reference":
            result["reference_sha256"] = "0" * 64
        elif change == "settings":
            bundle["parts"]["request"]["manifest"]["max_optimizer_updates"] += 1
    if change == "extra_input":
        inputs["selected_winner"] = "circadian"
    reseal_development(inputs, 5)
    with pytest.raises(ValueError):
        development._derive_development_ledger(inputs)


@pytest.mark.parametrize("key", ["development_bundles", "preflight_bundles"])
def test_should_reject_missing_repeat_or_detached_whole_part(key: str) -> None:
    inputs = fabricated_inputs()
    inputs[key].pop()
    with pytest.raises(ValueError):
        development._derive_development_ledger(inputs)
    inputs = fabricated_inputs()
    inputs[key][-1]["parts"]["audit"]["status"] = "completed_but_changed"
    with pytest.raises(ValueError):
        development._derive_development_ledger(inputs)


@pytest.mark.parametrize("value", [None, [], {}, {"original_scope": {}, "input_catalog": {}}])
def test_should_reject_unpinned_public_inputs(value: Any) -> None:
    with pytest.raises(ValueError):
        development.build_development_ledger(value)


def test_should_refuse_fabricated_full_scope_as_current_original_authority() -> None:
    with pytest.raises(ValueError, match="whole original prospective scope"):
        development.build_development_ledger(fabricated_inputs())


def test_should_detach_outputs_and_repeat_exactly_without_science_or_file_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = fabricated_inputs()
    from src.core.backprop_mlp import BackpropMLP
    from src.core.predictive_coding import PredictiveCodingNetwork
    from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork

    def prohibited(*args: Any, **kwargs: Any) -> None:
        pytest.fail("pure development projection performed science or IO")

    for model in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        for name in ("__init__", "train_epoch", "predict_proba", "compute_accuracy"):
            monkeypatch.setattr(model, name, prohibited)
    for name in ("read_bytes", "read_text", "write_bytes", "write_text", "open"):
        monkeypatch.setattr(Path, name, prohibited)
    first = development._derive_development_ledger(inputs)
    second = development._derive_development_ledger(deepcopy(inputs))
    assert canonical_body_identity(first) == canonical_body_identity(second)
    first["original_inputs"]["development_bundles"][-1]["parts"]["audit"]["status"] = "changed"
    first["cells"][0]["scores"]["a_after_a"] = 0.01
    assert inputs == second["original_inputs"]
