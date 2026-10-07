"""Verify the complete confirmation envelope, roles and checkpoint schemas.

This stage is pure and constructs no source/model, computes no score and owns
no IO. Raw family work/guard/supply semantics and runtime/artifact publication
remain separate unfinished gates; this is not the full scientific validator.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from src.app.continual_confirmation_checkpoints import declarations, verify_checkpoint
from src.app.continual_confirmation_json import (
    finite_json,
    hash_value,
    integer,
    object_fields,
    require,
    same_json,
)
from src.app.continual_confirmation_manifest import (
    ConfirmationFamily,
    ConfirmationManifest,
    validate_confirmation_manifest,
)
from src.app.continual_sleep_factor_preflight import SLEEP_ARMS
from src.app.continual_confirmation_parameter_links import verify_parameter_links


_SEED_FIELDS = {
    "family",
    "seed",
    "roles",
    "initial",
    "after_a",
    "after_b",
    "legacy_train_facts",
    "supplemental_guards",
    "outer_selection_scored",
    "final_released",
}
_ROLES = ("train", "inner_guard", "outer_selection")


def verify_confirmation_envelope(payload: dict[str, Any], manifest: ConfirmationManifest) -> None:
    """Require the whole frozen 560-cell scope; never accept a partial run."""
    validate_confirmation_manifest(manifest)
    try:
        finite_json(payload)
        object_fields(
            payload,
            {
                "manifest",
                "seed_results",
                "protocol_id",
                "all_a_completed_before_first_b",
                "outer_selection_scored",
                "final_released",
            },
            "joint envelope",
        )
        same_json(payload["manifest"], json.loads(json.dumps(asdict(manifest))), "joint manifest")
        require(
            payload["protocol_id"] == "continual_mechanism_confirmation_train_only_v1"
            and payload["all_a_completed_before_first_b"] is True
            and payload["outer_selection_scored"] is False
            and payload["final_released"] is False,
            "joint protocol/arrival/evaluation seal differs",
        )
        rows = payload["seed_results"]
        require(type(rows) is list, "joint seed rows differ")
        expected = [(family.name, seed) for family in manifest.families for seed in family.seeds]
        require(
            [(row["family"], row["seed"]) for row in rows] == expected,
            "complete family/seed order differs",
        )
        families = {family.name: family for family in manifest.families}
        for row in rows:
            _verify_seed_envelope(row, families[row["family"]])
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("confirmation JSON malformed or incomplete joint envelope") from error


def _verify_seed_envelope(row: dict[str, Any], family: ConfirmationFamily) -> None:
    finite_json(row)
    object_fields(row, _SEED_FIELDS, "family seed")
    seed = integer(row["seed"], "source seed")
    require(
        row["family"] == family.name
        and seed in family.seeds
        and row["outer_selection_scored"] is False
        and row["final_released"] is False,
        "family seed/role seal differs",
    )
    roles = row["roles"]
    require(type(roles) is list and len(roles) == 2, "arrived phase rows differ")
    for phase, role in zip(("a", "b"), roles, strict=True):
        _verify_role(role, phase, seed)
    hashes = [value for role in roles for value in role["hashes"].values()]
    require(len(set(hashes)) == 6, "role hashes are not distinct")
    specs = declarations(family, seed)
    for stage in ("initial", "after_a", "after_b"):
        checkpoints = object_fields(row[stage], set(family.arms), f"{stage} checkpoint inventory")
        for name in family.arms:
            verify_checkpoint(checkpoints[name], specs[name], stage)
    _verify_legacy_role_links(row, roles)
    _verify_witnesses(row, family, specs)
    verify_parameter_links(row, family)


def _verify_role(row: dict[str, Any], phase: str, seed: int) -> None:
    object_fields(
        row,
        {
            "phase",
            "seed",
            "hashes",
            "counts",
            "sample_ids",
            "expected_final_count",
            "final_released",
        },
        "arrived role",
    )
    require(
        row["phase"] == phase
        and type(row["seed"]) is int
        and row["seed"] == seed
        and row["final_released"] is False
        and type(row["expected_final_count"]) is int
        and row["expected_final_count"] == 40,
        "arrived identity/final seal differs",
    )
    hashes = object_fields(row["hashes"], set(_ROLES), "role hashes")
    counts = object_fields(row["counts"], set(_ROLES), "role counts")
    ids = object_fields(row["sample_ids"], {*_ROLES, "final_test"}, "role IDs")
    expected_counts = (72, 24, 24) if phase == "a" else (36, 12, 12)
    development = []
    for name, count in zip(_ROLES, expected_counts, strict=True):
        hash_value(hashes[name], "role")
        require(
            integer(counts[name], "role count", 1) == count
            and type(ids[name]) is list
            and len(ids[name]) == count,
            "role ID/count link differs",
        )
        indices = []
        prefix = f"phase_{phase}/seed_{seed}/development/"
        for value in ids[name]:
            require(type(value) is str and value.startswith(prefix), "role ID identity differs")
            suffix = value.removeprefix(prefix)
            require(
                suffix.isascii()
                and suffix.isdigit()
                and str(int(suffix)) == suffix
                and 0 <= int(suffix) < 120,
                "role source position differs",
            )
            indices.append(int(suffix))
        require(indices == sorted(set(indices)), "role source order/uniqueness differs")
        development.extend(indices)
    require(len(set(development)) == sum(expected_counts), "development role IDs overlap")
    if phase == "a":
        require(sorted(development) == list(range(120)), "A arrived source coverage differs")
    same_json(
        ids["final_test"],
        [f"phase_{phase}/seed_{seed}/final/{index}" for index in range(40)],
        "sealed final ID declaration",
    )


def _verify_legacy_role_links(row: dict[str, Any], roles: list[dict[str, Any]]) -> None:
    legacy = row["legacy_train_facts"]
    require(
        type(legacy) is dict and type(legacy["seed"]) is int and legacy["seed"] == row["seed"],
        "legacy seed link differs",
    )
    prefix = "source_role" if row["family"] in {"gating", "replay"} else "role"
    hashes = {
        f"{role['phase']}_{name}": value for role in roles for name, value in role["hashes"].items()
    }
    counts = {
        f"{role['phase']}_{name}": value for role in roles for name, value in role["counts"].items()
    }
    same_json(legacy[f"{prefix}_hashes"], hashes, "legacy role hash link")
    same_json(legacy[f"{prefix}_counts"], counts, "legacy role count link")


def _verify_witnesses(
    row: dict[str, Any], family: ConfirmationFamily, specs: dict[str, Any]
) -> None:
    expected = (
        [(name, "a", 12) for name in SLEEP_ARMS]
        if family.name == "sleep"
        else [
            (f"neutral_{policy}", phase, epoch)
            for phase in ("a", "b")
            for epoch in range(1, 13)
            for policy in ("periodic", "adaptive", "no_sleep")
        ]
        if family.name == "schedule"
        else []
    )
    witnesses = row["supplemental_guards"]
    require(
        type(witnesses) is list
        and [(item["name"], item["phase"], item["epoch"]) for item in witnesses] == expected,
        "complete supplemental witness inventory differs",
    )
    for witness in witnesses:
        object_fields(
            witness, {"name", "phase", "epoch", "outcome", "before", "after"}, "guard witness"
        )
        integer(witness["epoch"], "guard epoch", 1)
        require(
            witness["outcome"] in {"accepted", "rolled_back", "skipped"}, "guard outcome differs"
        )
        for endpoint in ("before", "after"):
            verify_checkpoint(witness[endpoint], specs[witness["name"]], "guard")
            require(
                witness[endpoint]["clocks"]["wake_batches"]
                == witness["epoch"] + (0 if witness["phase"] == "a" else 12),
                "guard/wake clock link differs",
            )
            require(
                witness[endpoint]["clocks"]["wake_examples"]
                == (
                    72 * witness["epoch"]
                    if witness["phase"] == "a"
                    else 864 + 36 * witness["epoch"]
                ),
                "guard/wake example link differs",
            )
        if witness["outcome"] != "accepted":
            same_json(
                witness["before"], witness["after"], "complete rejected/skipped state restoration"
            )
