"""Fixed named S64 controls for the existing MiniGrid learner/runtime route.

Inputs are explicit recipe IDs and returned work records. Outputs are exact owned
native policies, protocol metadata and nominal inference costs. No train/update,
task selection, clock, budget, evaluation, IO or new orchestration belongs here.
"""

from dataclasses import asdict

from src.adapters.invariant_torch_feature_learner import InvariantTorchFeaturePolicy
from src.adapters.torch_feature_learner import TorchFeaturePolicy, _config

METHODS = ("pc_relax64", "neutral_cpc_relax64")
BOUND_METHODS = tuple(f"public_invariant_{name}" for name in METHODS)
INFERENCE_STEPS = 64


def base_method(method: str) -> str:
    if type(method) is not str or method not in METHODS:
        raise ValueError("relaxation control requires exact pc_relax64/neutral_cpc_relax64")
    return method.removesuffix("_relax64")


def policy(method: str, *, wide_neural: bool = False) -> InvariantTorchFeaturePolicy:
    """Bind a public recipe ID to its entire fixed native policy, not its suffix."""
    if type(wide_neural) is not bool:
        raise ValueError("fixed64 width selection requires an exact bool")
    if type(method) is not str or method not in BOUND_METHODS:
        raise ValueError("fixed64 policy requires its declared public control method")
    native = base_method(method.removeprefix("public_invariant_"))
    return InvariantTorchFeaturePolicy(
        TorchFeaturePolicy(
            native, 296, 32 if wide_neural else 8, 17, inference_steps=64, num_classes=7
        )
    )


def validate_option(method, *, joint_catalogue, train_action_margins, pc_update_diagnostics):
    if (
        type(method) is str
        and method in METHODS
        and (
            joint_catalogue is not True
            or train_action_margins is not False
            or pc_update_diagnostics is not False
        )
    ):
        raise ValueError(
            "fixed64 control requires joint_catalogue without extra readout diagnostics"
        )


def describe(protocol, method, *, wide_neural=False):
    if type(wide_neural) is not bool:
        raise ValueError("fixed64 width selection requires an exact bool")
    if type(method) is not str or method not in METHODS:
        return
    native = policy(f"public_invariant_{method}", wide_neural=wide_neural).native_policy
    if wide_neural and native.method == "neutral_cpc":
        # Why: the public control name must describe the actual native neutral
        # configuration; this does not alter the preserved H8 descriptor wire.
        protocol["cpc_configs"] = {method: asdict(_config(native))}
    protocol["inference_steps"] = 64
    protocol["relaxation_control"] = dict(
        method=method,
        native_method=native.method,
        inference_steps=64,
        original_inference_steps=2,
        native_policy=asdict(native),
        rationale="one predeclared maximum finite-relaxation sensitivity endpoint; no sweep or tuned baseline",
        comparison="distinct S64 recipe, not original S2 or equal-computation evidence; objective/data/metrics unchanged",
    )
    work = protocol["planned_work"]
    work["relaxation_control"] = dict(
        nominal_latent_iterations=64 * work["native_train_calls"],
        nominal_latent_row_iterations=64 * work["training_rows"],
        inference_steps=64,
    )
    protocol["limitations"] += (
        "; fixed64 control adds inference computation; no acquisition, convergence, cause or advantage assumed"
    )


def add_work(result, rows):
    controls = [row for row in rows if row["method"] in BOUND_METHODS]
    if not controls:
        return
    if len(rows) != 1 or len(controls) != 1:
        raise ValueError("fixed64 work requires its one explicitly selected control")
    result["relaxation_control"] = dict(
        inference_steps=64,
        native_train_calls_returned=result["native_train_calls"],
        nominal_latent_iterations_returned=64 * result["native_train_calls"],
        nominal_latent_row_iterations_returned=64 * result["training_rows"],
        failure_work_may_be_unobserved=result["failure_work_may_be_unobserved"],
        scope="derived from returned native calls/rows only; failed partial/unreturned latent work unknown; validation/copies/readouts/RSS additional",
    )
