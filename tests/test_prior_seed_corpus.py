from copy import deepcopy

import pytest

from src.app.prior_seed_corpus import validated_seed_contents
from seed_chronology_fixtures import saved_seed_corpus


def test_should_preserve_all_current_and_git_aliases_for_identical_contents() -> None:
    report = saved_seed_corpus()
    assert validated_seed_contents(report) == tuple(report["contents"])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_content",
        "duplicate_content",
        "missing_alias",
        "extra_alias",
        "reordered_alias",
        "wrong_size",
        "wrong_summary",
        "unknown_section",
        "unknown_top_field",
        "forged_authority",
        "boolean_count",
        "unknown_commit",
    ],
)
def test_should_reject_incomplete_or_forged_saved_corpora(mutation: str) -> None:
    report = deepcopy(saved_seed_corpus())
    if mutation == "missing_content":
        report["contents"] = []
    elif mutation == "duplicate_content":
        report["contents"].append(deepcopy(report["contents"][0]))
    elif mutation == "missing_alias":
        report["contents"][0]["aliases"].pop()
    elif mutation == "extra_alias":
        report["contents"][0]["aliases"].append("undeclared.py")
    elif mutation == "reordered_alias":
        report["contents"][0]["aliases"].reverse()
    elif mutation == "wrong_size":
        report["contents"][0]["byte_count"] += 1
    elif mutation == "wrong_summary":
        report["physical_file_count"] = 2
    elif mutation == "unknown_section":
        report["contents"][0]["omitted_section"] = []
    elif mutation == "unknown_top_field":
        report["supposed_release_proof"] = True
    elif mutation == "forged_authority":
        report["fresh_roles_authorized"] = True
    elif mutation == "boolean_count":
        report["physical_file_count"] = True
    else:
        report["history"]["objects"][0]["commit_path_aliases"][0][0] = "c" * 40
    with pytest.raises(ValueError, match="saved seed"):
        validated_seed_contents(report)
