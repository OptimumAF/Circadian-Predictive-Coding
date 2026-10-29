"""One frozen-selector schedule validation through the existing MiniGrid stream.

Inputs are the owning study, its original frozen chooser and one finite recipe.
Outputs retain all candidate work and deferred TRAIN game endpoints. No fitting,
final/probe access, source loading, promotion or independent budget is granted.
"""

from copy import deepcopy
from dataclasses import asdict
from time import monotonic
from typing import Any

from src.app.coupled_attribution import _same_value
from src.core.controller_objective import ControllerMeasures, controller_utility
from src.core.reference_controller import ReferencePolicy

RECIPE = "interleaved_delay2"
NUISANCE_RECIPE = "nuisance_redwall_delay2"
MODES = ("no_learning", "periodic", "drift", "budget_aware", "fitted")


def describe_validation(recipe, *, acknowledge_correct=False, share_readouts=False):
    if type(acknowledge_correct) is not bool:
        raise ValueError("correct acknowledgement must be an exact bool")
    if type(share_readouts) is not bool:
        raise ValueError("readout sharing must be an exact bool")
    if type(recipe) is not str or recipe not in (RECIPE, NUISANCE_RECIPE):
        raise ValueError(
            "validation requires the declared interleaved_delay2 or nuisance_redwall_delay2 recipe"
        )
    result = dict(
        recipe_id=recipe,
        source_indices=[2, 0, 3, 1],
        cue_order=[0, 0, 1, 1],
        label_delay=2,
        opportunities=list(range(6)),
        update_ceiling=4,
        modes=list(MODES),
        references=[asdict(ReferencePolicy(mode)) for mode in MODES[:-1]],
        endpoint="same known TRAIN A3/A4 games after all five windows; not unseen tasks",
        utility="original objective; endpoint gain/drop, per-arm TRAIN+readout seconds and cumulative RSS proxy; shared fit cost separate, not amortized",
        recovery="not measured: interleaved endpoint has no isolated after-task/return phases",
    )
    if recipe == NUISANCE_RECIPE:
        result.update(
            source="original grey A3 pair plus fixed red-wall A3 TRAIN pair collected after chooser freeze",
            endpoint="grey/red-wall A3 TRAIN games after all five windows; separate from chooser fitting, not final/probe",
            endpoint_groups=["old_A3", "shifted_A3"],
            layout=dict(context="A", branch_row=3, rotation=0, swap_keys=False),
            wall_colors=["grey", "red"],
            seed=17,
            max_steps=16,
            baseline="fresh warm readout in validation domain, deferred until all training",
        )

    if acknowledge_correct:
        result["acknowledge_correct"] = True
        result["acknowledgement"] = (
            "fresh lease-bound correct proof; one wake ORack/tick; no optimizerstep for ack"
        )
    if share_readouts:
        result["readout_sharing"] = dict(
            max_entries=6,
            lifetime="one validation invocation after all training; discarded on exit",
            match="complete native snapshot and identical readout arguments; same guarded source",
            failure="refuse damaged/replaced prior evidence or failed calls; no cache admission",
            utility="unavailable for every reused endpoint, including shared baseline",
        )
    return result


def prepare_nuisance_sources(study, controller, report):
    """Collect one fixed TRAIN pair after freeze under the original caller clock."""
    from src.adapters.minigrid_train_collection import collect_train_pair
    from src.adapters.minigrid_train_catalogue import _validate_sources
    from src.core.minigrid_key_door import MiniGridKeyDoorLayout
    import pickle
    import random
    import numpy as np

    frozen = controller.application_snapshot()
    rng = pickle.dumps((random.getstate(), np.random.get_state()))
    records: list[dict[str, Any]] = []
    report["collection"] = records
    work = report["collection_work"] = dict(
        attempts=1, returned=0, admitted=False, unreturned_work_unknown=True
    )
    # Reuse the original study's bounded evidence store so failure closeout can
    # recover the last paid record, not merely report its corrupted replacement.
    study.link(records, report, "collection")
    study.link(work, report, "collection_work")
    paid: list[dict[str, Any]] = []
    returned, admitted = 0, False

    def require():
        study.require()
        if not _same_value(controller.application_snapshot(), frozen):
            raise ValueError("frozen chooser changed during nuisance collection")
        if pickle.dumps((random.getstate(), np.random.get_state())) != rng:
            raise ValueError("global RNG changed during nuisance validation")
        if (
            report.get("collection") is not records
            or not len(paid) <= len(records) <= 2
            or report.get("collection_work") is not work
            or (
                work.get("attempts"),
                work.get("returned"),
                work.get("admitted"),
                work.get("unreturned_work_unknown"),
            )
            != (1, returned, admitted, returned == 0)
        ):
            raise ValueError("nuisance collection evidence or charges changed")
        for i, record in enumerate(paid):
            if records[i] is not record:
                raise ValueError("preceding nuisance collection record replaced")
        for record in records[len(paid) :]:
            study.seal(record, records, len(paid))
            paid.append(record)

    require()
    packets = collect_train_pair(
        require, records, layout=MiniGridKeyDoorLayout("A", wall_color="red")
    )
    returned = 1
    work.update(returned=1, unreturned_work_unknown=False)
    require()
    if (
        len(records) != 2
        or len(packets) != 2
        or tuple(p.episode_id for p in packets)
        != tuple(f"minigrid-source-A3-redwall-train-{c}" for c in ("blue", "red"))
        or any(p.context != "A" or len(p.transitions) != 9 for p in packets)
    ):
        raise ValueError("two fixed nine-row red-wall A3 TRAIN sources required")
    episodes = study.episodes[:2] + packets
    witness, _ = _validate_sources(episodes, 6)
    admitted = True
    work.update(admitted=True)
    require()
    study.seal(work)

    def boundary():
        require()
        witness.check(episodes)

    return episodes, boundary


def run_validation(
    study, controller, recipe, *, baseline, acknowledge_correct=False, share_readouts=False
):
    """Complete every candidate before reading any validation game endpoint."""
    protocol = describe_validation(
        recipe, acknowledge_correct=acknowledge_correct, share_readouts=share_readouts
    )
    nuisance = recipe == NUISANCE_RECIPE
    if nuisance and baseline is not None:
        raise ValueError("nuisance baseline must be read in its own domain after training")
    if getattr(study, "nuisance", False) is not nuisance:
        raise ValueError("validation recipe and source/readout domain differ")
    report = study.report
    report.update(protocol=protocol, status="running", endpoint_reads_released=False)
    study.seal(protocol, report, "protocol")
    study.link(report["candidates"], report, "candidates")
    frozen = controller.application_snapshot()
    initial_source = deepcopy(study.base)
    # Why local: only this invocation shares the original source, game protocol,
    # paid evidence and lifetime owners. No cross-run authority is cached.
    cache: list[tuple[Any, Any, Any, str]] = (
        [(initial_source, None, baseline, "shared_base")] if share_readouts and not nuisance else []
    )
    game_port = study.game
    states = []
    active = None
    baseline_record = None
    candidates = report["candidates"]
    endpoints = None
    candidate_count = endpoint_count = 0
    released = False

    def require_fit():
        study.require()
        if (share_readouts or nuisance) and study.game != game_port:
            raise ValueError("original validation readout port changed")
        if getattr(study, "nuisance", False) is not nuisance:
            raise ValueError("original validation domain changed")
        if (
            report.get("candidates") is not candidates
            or len(candidates) != candidate_count
            or report.get("endpoint_reads_released") is not released
            or report["contexts"]
            or report["trials"]
            or endpoints is not None
            and (report.get("endpoints") is not endpoints or len(endpoints) != endpoint_count)
        ):
            raise ValueError("original bounded validation sequence changed")
        if not _same_value(controller.application_snapshot(), frozen):
            raise ValueError("original frozen validation fit/source changed")
        if not _same_value(study.base, initial_source):
            raise ValueError("validation warm source changed")

    try:
        for mode in MODES:
            require_fit()
            row: dict[str, Any] = dict(mode=mode, status="running")
            active = row
            report["candidates"].append(row)
            candidate_count += 1
            study.link(row, report["candidates"], len(report["candidates"]) - 1)
            started = monotonic()
            checkpoints = study.window(
                study.warm,
                tuple(protocol["source_indices"]),
                "validation-" + mode,
                row,
                controller=controller if mode == "fitted" else None,
                policy=None if mode == "fitted" else ReferencePolicy(mode),
                label_delay=protocol["label_delay"],
                **(dict(acknowledge_correct=True) if acknowledge_correct else {}),
            )
            row["training_seconds"] = monotonic() - started
            window = row["window"]
            if len(window["decisions"]) != 6 or not window["window_complete"]:
                raise ValueError("all six declared validation opportunities required")
            final = checkpoints[row["budget"]["updates"]]
            if acknowledge_correct:
                row["applied_cues"] = [
                    protocol["cue_order"][i] for i in window["applied_packet_indices"]
                ]
            states.append(deepcopy(final))
            study.seal(window, row, "window")
            # Seal paid training before another candidate. Endpoints live in a
            # different record so later reads cannot rewrite this prefix.
            row["status"] = "training_complete"
            study.seal(row)
            require_fit()
        released = True
        report["endpoint_reads_released"] = True
        endpoints = report["endpoints"] = []
        study.link(report["endpoints"], report, "endpoints")
        if nuisance:
            baseline = baseline_record = report["baseline"] = dict(status="running", games=None)
            study.link(baseline, report, "baseline")
            baseline_started = monotonic()
            baseline["games"] = study.game(initial_source, cue_prefix=tuple(protocol["cue_order"]))
            baseline["readout_seconds"] = monotonic() - baseline_started
            baseline["status"] = "complete"
            study.seal(baseline)
            baseline = baseline["games"]
            if share_readouts:
                cache.append((initial_source, None, baseline, "shared_base"))
            require_fit()
        before = _counts(baseline, nuisance=nuisance)
        for row, state in zip(report["candidates"], states, strict=True):
            require_fit()
            endpoint: dict[str, Any] = dict(mode=row["mode"], status="running", games=None)
            report["endpoints"].append(endpoint)
            endpoint_count += 1
            study.link(endpoint, report["endpoints"], len(report["endpoints"]) - 1)
            started = monotonic()
            reused = _same_value(state, initial_source)
            options = (
                dict(applied_cues=tuple(row["applied_cues"]))
                if acknowledge_correct
                else dict(cue_prefix=tuple(protocol["cue_order"]))
            )
            saved_state = deepcopy(state) if share_readouts else None
            found = (
                _find_shared_readout(cache, state, options)
                if share_readouts and not reused
                else None
            )
            games, origin = (
                (baseline, "shared_base")
                if reused
                else found
                if found is not None
                else ((game_port if share_readouts else study.game)(state, **options), None)
            )
            readout_seconds = monotonic() - started
            require_fit()
            if share_readouts and not _same_value(state, saved_state):
                raise ValueError("validation readout changed its complete candidate snapshot")
            after = _counts(games, nuisance=nuisance)
            target_group = "shifted_A3" if nuisance else "new_A4"
            reused_utility_unavailable = origin is not None and (share_readouts or nuisance)
            measures = ControllerMeasures(
                before[target_group],
                after[target_group],
                2,
                before["old_A3"],
                after["old_A3"],
                2,
                row["training_seconds"] + readout_seconds,
                study.budget.process_rss_sampler.peak_bytes,
                0,
                True,
            )
            endpoint.update(
                status="complete",
                games=games,
                reused_from=origin,
                before=before,
                after=after,
                native_updates=state.completed_updates,
                native_rows=state.training_examples,
                readout_seconds=readout_seconds,
                measures=None if reused_utility_unavailable else asdict(measures),
                utility=(
                    None
                    if reused_utility_unavailable
                    else asdict(controller_utility(frozen["policy"].objective, measures))
                ),
                recovery=None,
            )
            if share_readouts or nuisance:
                endpoint["readout_work"] = dict(
                    physical_game_calls=int(origin is None),
                    logical_episode_slots=4,
                    reused_episode_slots=4 if origin is not None else 0,
                    utility_available=origin is None,
                )
            study.seal(endpoint)
            if share_readouts and origin is None:
                if len(cache) >= 6:
                    raise ValueError("warm plus five readout states already retained")
                cache.append((saved_state, deepcopy(options), games, row["mode"]))
        require_fit()
        report.update(status="complete", frozen_fit_unchanged=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}"[:512])
        if active is not None and active.get("status") == "running":
            active.update(status="failed", error=report["error"])
        if baseline_record is not None and baseline_record.get("status") == "running":
            baseline_record.update(status="failed", error=report["error"])
        raise
    finally:
        try:
            require_fit()
        except Exception as error:
            report.update(status="failed", capture_error=f"{type(error).__name__}: {error}"[:512])
            study.preserve_completed_prefix()
        report["budgets"] = [
            dict(
                limits=limits,
                started_at=pin[2],
                updates=owner.updates_completed,
                replay=owner.replay_examples_completed,
                last_clock=owner.last_clock,
            )
            for owner, pin, limits, _ in study.budgets
        ]
        report["game_work"] = dict(
            attempts=study.game_attempts,
            returned=study.game_returns,
            unreturned_work_unknown=study.game_attempts != study.game_returns,
        )
        report["unadmitted_game"] = study.pending_game


def _counts(games, *, nuisance=False):
    # The existing game port validates complete role/layout/action records.
    if nuisance:
        from src.adapters.minigrid_policy_evaluation import _require_phase

        _require_phase(games, nuisance=True)
    rows = games["episodes"]
    return dict(
        old_A3=sum(r["success"] for r in rows[:2]),
        **{("shifted_A3" if nuisance else "new_A4"): sum(r["success"] for r in rows[2:])},
    )


def _find_shared_readout(cache, state, options):
    """Same guarded invocation only; match full state, never a score or digest."""
    return next(
        (
            (games, origin)
            for expected, arguments, games, origin in cache
            if _same_value(arguments, options) and _same_value(expected, state)
        ),
        None,
    )
