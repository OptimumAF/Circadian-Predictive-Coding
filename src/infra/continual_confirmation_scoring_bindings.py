"""Verify current full scored sources, request bytes and reference artifacts.

Inputs are repository/request/scope paths and the complete small reference
report. Outputs bind the fixed request to current files/command/environment.
No model/source construction, score, complete reference decode or publication.
"""

from __future__ import annotations

from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_scoring_execution import (
    SOURCE_FILE_COUNT,
    encoded_identity,
    verify_reference_report,
    verify_scoring_execution_request,
)
from src.app.continual_confirmation_scoring_manifest import fixed_scoring_manifest
from src.infra.continual_confirmation_io import file_digest, read_json, verify_source_files
from src.infra.continual_confirmation_training_references import (
    stream_file_identity,
    verify_training_reference_bytes,
)


# All authoritative observer V2 pins remain unchanged; no ignored freeze file
# is needed at runtime. New component pins are filled before worker fixtures.
PRIOR_SOURCE_SHA256 = {
    "scripts/__init__.py": "ed3888e5ac0bad9b11c74c5a8055c0292ad190c6401361bd3f1bfc700ee0c8ab",
    "scripts/inspect_p67_confirmation_scope.py": "e3f96f4e284b56a2db5a21d2c636f9ca4361e4b7fbce5cd8ba8c1ca833b41a1e",
    "scripts/inspect_p67_scoring_training_references.py": "248cc2eef9290c5d051298a1a712f547c5edb04ee5940a7d7b06163452aa7b08",
    "scripts/run_p63_combined_factor_development.py": "edf68589d83033d0abdf39c737f78044938f84a97b49d86a25698edce42d8c40",
    "scripts/run_p63_combined_factor_preflight.py": "9c3d640638b48d7a2edc9cc4585e205cdb3f2d86bd1c72a6e7a8438bd0f3d404",
    "scripts/run_p63_gating_pilot.py": "919347ba9a9a38d88a83c97b5096898e13f9807f53225a2834b9927cba967c07",
    "scripts/run_p63_parent_factor_development.py": "be57cf809149af570876d75d6f924ec073ee1a35a1d96b68ae4fc7b59129e9c0",
    "scripts/run_p63_parent_factor_preflight.py": "39bfd61bd50a1abf0a00308ca40028152270b09f99eb401b4176924dd4c6a8b1",
    "scripts/run_p63_replay_factor_pilot.py": "9255ad6615b3c41953fc5f76ad4c5597389724abf57ad996f2c1945c83b8826a",
    "scripts/run_p63_schedule_factor_development.py": "94fc133317c9b0954cee47b5d5a431285bcf30da3effc9b457153b6bfe6000c1",
    "scripts/run_p63_schedule_factor_preflight.py": "4490f9201b565d3283434da786799cc3d9ac8242218cfc8f9b881de9011b7fbe",
    "scripts/run_p63_sleep_factor_development.py": "08dedd66328f41ab24403e29ca0f8bc8c4b10030528f9e4a2c745b4f0d330634",
    "scripts/run_p63_sleep_factor_preflight.py": "dae68cbdb8ef0464ff9a4971bfaedc2e50dc4d75980693a236a5a92eb38d0d3c",
    "scripts/run_p67_confirmation_training.py": "e6bafe8bc6c7e2199a578b26ff991a45fbbf4d45085e760fd61d0f9417250445",
    "src/__init__.py": "799506c6c6c8bb02ffb41f82b11e784c9b62a575d714519ae9a319494c05f3df",
    "src/app/__init__.py": "65855836756b6c2d6669b454decefedb25833ca2b406a58fda7ca0ea34254ff5",
    "src/app/circadian_checkpoint.py": "ee3c34833108007fa652f2d728b02625c168fa7b278dc939c19231628ae9e53c",
    "src/app/comparison_scope.py": "7738c5c9931d9fb835055fa2f0e94398e58ca88fb3d586164b2d45e22e6679ef",
    "src/app/continual_arrived_benchmark.py": "21b8d25f7174cf16187054bca6bd41f6ed17dbede2284eb61efd69f16b275ae7",
    "src/app/continual_arrived_checkpoint.py": "2a6502ebdb2ad7a8b8f8e29db0ce73f6bc56faaabb8391d0faf33a706affd640",
    "src/app/continual_arrived_sleep_history.py": "c47b8a31eb14056656668c1ea82b30c1d8162f6e3b664ab363fd025e10a1d6e8",
    "src/app/continual_arrived_transactions.py": "cb11cfb05ca7bfe4e073b110c3a7654a3432192f63317bf14559988fab3cb6c3",
    "src/app/continual_checkpoint.py": "4884e67839372044c49fefb4db32e427d17cea4c15e91c64c3c0f9691d9b23d6",
    "src/app/continual_combined_factor_development.py": "be4c0156c8921a9f0d4580e93695bd4464c1d6241a9bfa7c3fd531eb199994fa",
    "src/app/continual_combined_factor_manifest.py": "213d239743617c7aefe6c1848049c9ac8160c26e3c741b91807524b292a75000",
    "src/app/continual_combined_factor_preflight.py": "f19a3e33df1bdcf25c429d10f3da0e0c4edab361fe9efe513a150750363e3a78",
    "src/app/continual_combined_factor_validation.py": "3bc221199a9036ceb234621d16166a8682f209fc50caf033bcc3fad1b2004d10",
    "src/app/continual_confirmation_analysis.py": "3fed747ac5bdd61569c33a48f768b5f54c3159991de9d2e694b5c827a9f25de0",
    "src/app/continual_confirmation_analysis_contract.py": "0e3989ecfb8191ba4aea1ebb989da23581e48d7a8a919c1a21156b0505323436",
    "src/app/continual_confirmation_checkpoints.py": "4b00b454d5b317d2190ca942d36e93e09446b4f043956fd74213abc5d27f73ba",
    "src/app/continual_confirmation_execution.py": "dc45ad04c9381bee5cbee0ab7e3d70f7da6676304d056a18668be23affc16006",
    "src/app/continual_confirmation_fact_schema.py": "01f8437eb53eda763edb73c8695c6bd18dbbfa8cb6d8fef0fb7f874e3dc47862",
    "src/app/continual_confirmation_final_observation.py": "263561557dc26aa44923872d016bc406fa50b3577eb289c7b7a4c7428b354c9d",
    "src/app/continual_confirmation_json.py": "307642a05dd2b73ef3af4a3714c33f85926e205c25d71d85c9dc796d6d9ff203",
    "src/app/continual_confirmation_manifest.py": "390cc56625983f2038ea98f689c68e11e754c971a49a17fcd63a227bb35b49ec",
    "src/app/continual_confirmation_parameter_links.py": "4c1fb55e5d414ed7fbb99392ede60d05eafd8c3d025e38e76cfa137f62130262",
    "src/app/continual_confirmation_periodic.py": "a22f23ef2b3c86682b4e68f65a4c211a664b407de00f21ecd2a2039080de9b9e",
    "src/app/continual_confirmation_scoring.py": "2458044d2118203a9b94fe728a67b3cbebcf6ce417495a1abe04e08bf82c14ee",
    "src/app/continual_confirmation_scoring_manifest.py": "21c86b9f78e6fef0c475400b87f026daa6bca11e20f64f7fa369d853207efca8",
    "src/app/continual_confirmation_scoring_state.py": "cedd370866fdd7223c83ba4398cc104156d2690b98f77f6547d78f1a6aacdad3",
    "src/app/continual_confirmation_scoring_validation.py": "e81dbacc4edcdfeae7342102321a2c05abf0e3f160bae627add2bc3bc190f643",
    "src/app/continual_confirmation_simple.py": "f9f46848310e02a10a8a47caa414c21b6d36812ede3684f803282f1b3eec1073",
    "src/app/continual_confirmation_simple_work.py": "f440aa80935f49e76cb2d9876f0beb476b0df18f6ecea4e8dd56ce408065de64",
    "src/app/continual_confirmation_state.py": "2808f11b22aa78894fd322a686e423cd5052d0fea0f7508209f6f89181394cd1",
    "src/app/continual_confirmation_training.py": "372aeeba9b8661918c8137c3d4c5a4f34d26666b76cc84644c99362808c103e1",
    "src/app/continual_confirmation_validation.py": "2d220d74e26533c33b14f1a5267923abe8d6691f9478580fb4e5f67f64272733",
    "src/app/continual_confirmation_work_validation.py": "cf26b0e73c5230f033c41a25cc436e4e6faa59b76a2375f89d08d6836ec4113f",
    "src/app/continual_gating_pilot.py": "9a917fe219907a7aa9470f82a1c510863959f5bea8ff9f43db843d9c44ea8c09",
    "src/app/continual_matched_replay_schedule.py": "aa3f14f447704a19d6de7997b1aa6daaf13f7637b2fda81cceb425a6bf4229e7",
    "src/app/continual_parent_factor_development.py": "53aab115493ba2ca2dd34787e07b4deb2f246bed257aad4eb97baf9e35af1530",
    "src/app/continual_parent_factor_manifest.py": "b607a55006aeeb6f0830102fce2160b1d8b3d4e0b3d422521bc57842970c5b69",
    "src/app/continual_parent_factor_preflight.py": "8a50e08d59a63c43c3279273d967093c1904bf3322a4d8f8313c8818c6c26e29",
    "src/app/continual_parent_factor_validation.py": "4fcf68b27a85f8a7cd88f64f00f0c8da35fe858c328ad5aaf6f9adb7d97d7955",
    "src/app/continual_replay_factor_pilot.py": "f472668e0bf49b31465609880f6e2c4a7a104a6b558de99a5ba76886469a8894",
    "src/app/continual_schedule_factor_development.py": "f2d49461a05defd18b53e5e28f822dcf099405ab6ff716972d41bc765e60c48d",
    "src/app/continual_schedule_factor_preflight.py": "2b7f80dd603ea72d564feae445c8ad3168b258685e2f7a05cda046bac4397bcd",
    "src/app/continual_schedule_factor_validation.py": "739b9a23d268de79f30cec72b395f045ed75a3b745f39af8b762a469caa9374a",
    "src/app/continual_shift_benchmark.py": "53037d6cfbf4a24a807eb5a4b67d5525cc422e1d209e39ab5c54c1ec0018f5fa",
    "src/app/continual_sleep_factor_development.py": "e6a498c107d0071b822309907691e225b1a1ae828c93f8435c94ba0cfa94bf88",
    "src/app/continual_sleep_factor_preflight.py": "f5b7d8d8f071ecd7c429f87581e52033b3a99e754721b7dd01635241f75bb6ff",
    "src/app/continual_trigger_replay_schedule.py": "e4e1316fb94beff2c005dfca0acaad3e729cc9174be0fac9a9a0ede0219ab042",
    "src/app/numpy_checkpoint_validation.py": "885b6ad87dedc7999f9bda2b0ed7884ab0265f7c4b644bfbff60936668409f9e",
    "src/app/numpy_sleep_decisions.py": "085aa77b010dee217bea6467119addc3c3cca14437b669d350d404f00e5950b0",
    "src/app/sleep_schedule.py": "23cbde1e0f3c60bf35fd143e3e7ee63955862818c0b686c4ecb5eadb5a46b5c9",
    "src/core/__init__.py": "979801ce2dd7643e3d41b160fdb72e6fd2e4c199cea580b0a3e1ea985ba9c4b5",
    "src/core/activations.py": "9fdd713deff66709d75ff41a8d6844faead78fb8e9d7bcb89d39e94935f9c297",
    "src/core/backprop_mlp.py": "bb961b86a5034869356ed1f4a2e609a8b6c1d087e80c8e87b1ec26b07f17b85b",
    "src/core/circadian_predictive_coding.py": "08f36db0a5c1f71d5de198fe0cf1c1be0e6a37eaf5855dec53bbb4a299ec27aa",
    "src/core/confirmation_final_roles.py": "acfd451e68280469731c94dcf943b25bb9db4f179ffcd1ab3f4c854321aaf276",
    "src/core/continual_metrics.py": "8eb491da4221a232a32a7cc1eb7b08a3c36a96fb748543450ac068dac3ae7f78",
    "src/core/controlled_parent_selection.py": "dd19acdaeb84830c709a30cc53c96850894b901ba476082636e10808cda132fb",
    "src/core/dimension_validation.py": "33fc41ae58af6bb3d000c97b6f0acd805e74306f751bedbc7b39daddb6e8e83d",
    "src/core/neuron_adaptation.py": "a119a2dcf27fe3682fcef12b387db5dd0f1bd831eea11d0583cf823a89602493",
    "src/core/predictive_coding.py": "681eae4c48c52b4b0731d859f9947c224ff2c48f4a7ac0d593b1d4b758eeed46",
    "src/core/replay_retention.py": "f5c5e9a793ad771c1a7ba0dc29411fb36ceafc34c37ca786cbae2efeeaa1108b",
    "src/core/resnet50_variants.py": "23a4ee75f938e2cb2152458ab6586a894cc22a1e35b19693b37b83e5f011a605",
    "src/core/seed_statistics.py": "d848ea585046c0c83ed73e22953f0c5361012b73129e12c464dc5290720195cd",
    "src/core/shared_replay_schedule.py": "1961f66262e33a0b73bdaf690336f8e79f6ba5047a8df53cd61c8005f10975d0",
    "src/core/sleep_clocks.py": "c08c3e8bc662f4136a853f9a19e8f0602f1433d119ff58321223a9a47f87a598",
    "src/core/sleep_telemetry.py": "b63c33b379254a5423831de68b7ec40cbc46d7ad8c96ef2ac6bc9fd1ea058a7d",
    "src/core/training_validation.py": "e49db0ed6c66eb40803a53254e6d9e2890589fe0756b749879d4c9c132429e1c",
    "src/infra/__init__.py": "7c6f6cf33e9fbf71f6df77af377b5df00ab2893be05278c308f7e9da01be6ffc",
    "src/infra/continual_confirmation_final.py": "2270fb0c2f2ddb36fa8ae1434324643e417e4a9b54b25589d29c2c18bd479b96",
    "src/infra/continual_confirmation_final_runtime.py": "b8e656687c10275f73210661d942c453f5bdb00625d0f24fd176be3de8d8db09",
    "src/infra/continual_confirmation_io.py": "c77b6fdc58f42c924384e28acce53e9872b279213b77179293f647c4b5b86c82",
    "src/infra/continual_confirmation_runtime.py": "faf8ea73611c179eb4238138c831d0db94c8959130c37e4b8cde6efa87ba8078",
    "src/infra/continual_confirmation_training_references.py": "61ccd46c0e0fdfd7244a1915305e6bfb4753175c2c959976e821c222951b195f",
    "src/infra/continual_roles.py": "b3ea9afd00586a7e5aa2343a73f8f3ee245fa72d4162bb806f5439c86c904dd1",
    "src/infra/datasets.py": "b3ca4e1939afdcced4b22404daaec41303f71f4d3461c797b75f63cbfece747e",
    "src/shared/__init__.py": "f728c7cf4d519be569354d482d94d84bc2257b319d72b51625ec7365c7fc36c4",
    "src/shared/process_memory.py": "78662728047b0a6fdd8b29fd0e54a1c09dbbac9d90e6d8c9dfbb5225bc6e8f10",
    "src/shared/torch_runtime.py": "698074d56fdbc09e234691e780ae2ce1111c6314af9b5acb6949c3a0430455e8",
}
ADDITIONAL_SOURCE_SHA256 = {
    "src/app/continual_confirmation_scoring_execution.py": "79e8b334a2bcbe0cc1c5e3b6d30adb6624a843fd9d99dea600f0c17c111ecbdd",
    "src/infra/continual_confirmation_scoring_worker.py": "86787b7c42d3a749fc88f59d92bb58ac11dea9996ae873cea67b2821ccdf0477",
    "src/infra/continual_confirmation_scoring_artifacts.py": "7b86ecf6e01039e0f348ea880884b943053417bee0c55e9aeb2ce9a040eded3b",
}
OWN_SOURCE_PATHS = (
    "src/infra/continual_confirmation_scoring_bindings.py",
    "scripts/run_p67_confirmation_scoring.py",
)


def current_sources(root: Path) -> dict[str, str]:
    expected = dict(PRIOR_SOURCE_SHA256)
    expected.update(ADDITIONAL_SOURCE_SHA256)
    require(
        len(expected) + len(OWN_SOURCE_PATHS) == SOURCE_FILE_COUNT, "scoring source closure differs"
    )
    require(all(len(value) == 64 for value in expected.values()), "scoring source pins not frozen")
    observed = verify_source_files(root, expected)
    # Why this: include these exact bytes in every strict request without a
    # self-referential hash constant. Late/request/readback equality is mandatory.
    for name in OWN_SOURCE_PATHS:
        require((root / name).is_file(), f"scoring own source missing: {name}")
        observed[name] = file_digest(root / name)
    return observed


def worker_command(request_file: Path, scope_file: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.run_p67_confirmation_scoring",
        "--worker",
        "--request-file",
        str(request_file.resolve()),
        "--scope-file",
        str(scope_file.resolve()),
    ]


def _check_reference_files(root: Path, scope_file: Path, reference_report: dict[str, Any]) -> None:
    manifest = fixed_scoring_manifest()
    same_json(
        stream_file_identity(scope_file)["sha256"],
        manifest.scope_record_sha256,
        "scoring saved scope bytes",
    )
    actual = verify_training_reference_bytes(root, manifest)
    same_json(
        actual,
        {bundle["reference"]["name"]: bundle["files"] for bundle in reference_report["bundles"]},
        "scoring current training reference bytes",
    )


def scoring_bindings(
    root: Path, request_file: Path, scope_file: Path, reference_report: Any
) -> dict[str, Any]:
    """Current bindings only; parent complete-reader dispatch remains required."""
    manifest = fixed_scoring_manifest()
    verify_reference_report(reference_report)
    _check_reference_files(root, scope_file, reference_report)
    return {
        "manifest": manifest,
        "reference_report": reference_report,
        "source_sha256": current_sources(root),
        "command": worker_command(request_file, scope_file),
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }


def checked_scoring_request(
    root: Path,
    request_file: Path,
    scope_file: Path,
    expected_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check canonical bytes before and after every current binding."""
    before = stream_file_identity(request_file)
    if expected_identity is not None:
        same_json(before, expected_identity, "scoring request bytes changed during execution")
    request = read_json(request_file)
    same_json(encoded_identity(request), before, "scoring decoded request bytes")
    bindings = scoring_bindings(root, request_file, scope_file, request.get("reference_report"))
    verify_scoring_execution_request(request, **bindings)
    same_json(stream_file_identity(request_file), before, "scoring request changed during bindings")
    return request
