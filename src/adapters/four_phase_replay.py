"""Fixed matched replay policy and protocol composition, without a runner.

Inputs are a declared control and the existing four-phase final protocol.
Outputs describe existing rehearsal policies and their prospective costs. No
data collection, learner, scoring, selection, budget renewal or IO occurs here.
"""

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from src.adapters import key_door_pilot as shared
from src.app.stream_rehearsal import (
    CoverageStreamReplayPolicy,
    RecentStreamReplayPolicy,
    StreamReplayPolicy,
)
from src.core.shared_replay_schedule import ReplayAllocation
from src.core.circadian_predictive_coding import ReplayRetentionBudget


def validate_control(control: str) -> str:
    if type(control) is not str or control not in (
        "recent",
        "partitioned",
        "recent48",
        "partitioned48",
        "coverage",
    ):
        raise ValueError(
            "four-phase replay control must be recent, partitioned, recent48, partitioned48 or coverage"
        )
    return control


COVERAGE_CONTEXTS = ("A-red", "A-blue", "B-red", "B-blue", "C_train-red", "C_train-blue")


def replay_policy(
    control: str,
) -> RecentStreamReplayPolicy | StreamReplayPolicy | CoverageStreamReplayPolicy:
    """Preserve old controls; new coverage comparison has a matched six-episode cap."""
    validate_control(control)
    if control == "recent":
        return RecentStreamReplayPolicy(2, 8, ReplayRetentionBudget(16, 4032))
    if control == "partitioned":
        return shared.replay_policy()
    retention = ReplayRetentionBudget(48, 12096)
    if control == "recent48":
        return RecentStreamReplayPolicy(2, 8, retention)
    if control == "partitioned48":
        return StreamReplayPolicy(
            2,
            8,
            retention,
            ReplayAllocation(ReplayRetentionBudget(24, 6048), ReplayRetentionBudget(24, 6048)),
            seed=53,
        )
    return CoverageStreamReplayPolicy(2, 8, retention, COVERAGE_CONTEXTS)


def describe_protocol(base: dict[str, Any], *, control: str) -> dict[str, Any]:
    """Detach a fixed control description from the existing final-only protocol."""
    policy = replay_policy(control)
    result = deepcopy(base)
    result.update(
        protocol_id=f"key_door_four_phase_{control}_replay_v1",
        classification="exploratory_matched_update_and_retention_replay_control",
        replay_control=control,
        comparison_readouts=dict(
            primary="pooled B/C_train live success and return at wake256 (four episodes); also report acquisition at B128/C_train192 before calling a loss forgetting",
            directions="partitioned minus recent for accuracy/success/return; recent minus partitioned for CE; positive favors partitioned",
            coverage="all five checkpoints/all layouts/both cues and all signed train metrics retained; one fixed seed, no significance or replicated advantage claim",
        ),
        replay=dict(
            policy=asdict(policy),
            selection="latest complete released episode"
            if control == "recent"
            else "existing rotating recent/reservoir complete released episode",
            wake_updates=256,
            replay_updates=128,
            replay_exposures=1024,
            native_updates=384,
            native_checkpoint_updates=[0, 96, 192, 288, 384],
            retention_scope="16transition/4032encodedarrayB ceiling; copies/events/native states included only in total time/process RSS",
            comparison="same wake/replay update counts and retention ceilings; selected content differs; no-replay256 reference is unequal compute",
        ),
        isolation="each control runs in an independent fixed CLI process; final data only after its training finishes; no cross-arm results train/select/tune",
        limitations=base["limitations"]
        + "; no matched wall-time claim; one fixed reservoir seed; retention/generalization benefit must be observed, not inferred from extra work",
    )
    if control in ("coverage", "recent48", "partitioned48"):
        result["replay"]["retention_scope"] = (
            "48transition/12096encodedarrayB ceiling; Python metadata/copies/events/native states included only in total time/process RSS"
        )
        result["comparison_readouts"]["directions"] = (
            "coverage minus each ordinary control for accuracy/success/return; ordinary control minus coverage for CE; positive favors coverage"
        )
        result["replay"]["selection"] = {
            "recent48": "latest complete released episode",
            "partitioned48": "existing rotating recent/reservoir complete released episode",
            "coverage": "latest released episode per declared training context/cue; eligible declared-order round robin after last selected key",
        }[control]
        result["replay"]["selector_information"] = (
            "coverage uses declared A/B/C_train x red/blue training context/cue metadata; ordinary controls use arrival/content retention only; not an information-matched selector comparison"
        )
        result["limitations"] += (
            "; new48 controls do not replace old16 negatives; coverage changes selector metadata and composition, not a pure age/reservoir effect"
        )
    result["planned_work"].update(
        native_train_calls=384,
        training_rows=3072,
        wake_train_calls=256,
        wake_training_rows=2048,
        replay_train_calls=128,
        replay_rows=1024,
        replay_calls=128,
        maximum_recurrent_unroll_steps=4688,
    )
    return result
