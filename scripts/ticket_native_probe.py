"""Ordinary native checkpoint handoff to the existing disposable task probe.

Owns fixed data composition and separate paid probe accounting. The ordinary
runner retains source authority; no source fitting, promotion, final data or IO.
"""

from copy import deepcopy
from dataclasses import asdict
from time import monotonic
from types import SimpleNamespace
from typing import Any

from scripts import ticket_workflow_native as native
from scripts.context_head_probe import build_head_cohorts
from scripts.context_head_serving import _episodes, PAIRED_TASKS
from scripts.run_ticket_workflow import CONTEXT_FEATURES
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeatureSnapshot
from src.adapters.torch_categorical_ce_initialization import (
    CategoricalCEInitialization,
    ce_initialization_record,
    copy_ce_initialization,
)
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import paid_report_guard
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import CATALOG


LIMITS = ToyExecutionBudget(
    max_training_updates=16, max_wall_seconds=30, max_process_rss_bytes=512 * 1024**2
)
CHECKPOINTS = (0, 4, 8, 16)


def _read_tasks(learner, phase, spec, require, row, *, boundary=None, linear=False):
    """Use the same typed evaluator on a frozen native reader, with paid work."""
    state = deepcopy(learner.snapshot_state())
    same = _same_value if linear else native.same_state
    readers: Any = native
    if linear:
        from scripts import ticket_workflow_linear

        readers = ticket_workflow_linear
    pin = deepcopy(state)
    reader: Any = None
    records: list[dict[str, Any]] = []
    row.setdefault("episodes", {})[phase] = records
    work = row.setdefault("task_readout_work", {})

    def unchanged():
        require()
        if not same(state, pin) or not same(learner.snapshot_state(), pin):
            raise ValueError("private native endpoint learner changed during readout")
        if reader is not None:
            readers.require_readout(reader, spec, pin)

    def port(read):
        if boundary is None:
            unchanged()
            read()
            unchanged()
        else:
            boundary(read, after_clocks=unchanged)

    def construct():
        nonlocal reader
        unchanged()
        reader = readers.readout(spec, state)
        unchanged()

    port(construct)

    def serve(request):
        result = None

        def read():
            nonlocal result
            unchanged()
            try:
                result = reader(request)
            finally:
                work[phase] = asdict(reader.work)
            unchanged()

        port(read)
        return SimpleNamespace(
            generation=0,
            actor_version="private-" + phase,
            completed_updates=state.completed_updates,
            result=result,
        )

    _episodes(
        SimpleNamespace(serve=serve),
        phase,
        unchanged,
        records,
        role="controller_train",
        tasks=PAIRED_TASKS,
    )
    port(unchanged)


def _require_probe_source(
    state,
    spec,
    source_report,
    source_budget,
    *,
    check_boundary,
    memory,
    row=None,
    linear_probe_update=None,
):
    from src.core.context_linear_state import ContextLinearFitPolicy, ContextLinearFitSnapshot
    from src.core.categorical_linear_state import LinearFitPolicy, LinearFitSnapshot

    shared_linear = type(state) is LinearFitSnapshot
    linear = shared_linear or type(state) is ContextLinearFitSnapshot
    valid_state = (
        state.policy
        == (
            LinearFitPolicy(9, 6)
            if shared_linear
            else ContextLinearFitPolicy(
                9, 6, (CONTEXT_FEATURES.index("policy_v1"), CONTEXT_FEATURES.index("policy_v2"))
            )
        )
        and linear_probe_update == "minimum_change"
        and type(linear_probe_update) is str
        if linear
        else type(state) is TorchFeatureSnapshot
        and state.policy.method in native.METHODS
        and state.policy == native.TorchFeaturePolicy(state.policy.method, 9, 3, 17, num_classes=6)
        and linear_probe_update is None
    )
    if (
        not valid_state
        or (state.completed_updates, state.training_examples) != (12, 72)
        or type(spec) is not DecisionFeatureSpec
        or spec != DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG)
        or type(source_report) is not dict
        or source_report.get("definition")
        != (
            "linear_context_return_ticket_workflow_v1"
            if shared_linear
            else "separate_context_linear_ticket_workflow_v1"
            if linear
            else "native_context_return_ticket_workflow_v1"
        )
        or source_report.get("method") != ("linear_fit" if linear else state.policy.method)
        or linear
        and source_report.get("reserved_probe_update") != linear_probe_update
        or linear
        and (
            source_report.get("separate_context_heads", False) is not (not shared_linear)
            or source_report.get("linear_train_update") not in (None, "minimum_change")
            or type(source_report.get("final_examples")) is not int
            or source_report["final_examples"] != 0
        )
        or shared_linear
        and (
            source_report.get("replay_mode", "none") != "none"
            or type(source_report.get("replay_examples", 0)) is not int
            or source_report.get("replay_examples", 0) != 0
        )
        or source_report.get("reserved_probe_requested") is not True
        or source_report.get("reserved_probe_source_training")
        != dict(completed_updates=12, applied_update_count=12)
        or type(source_budget) is not ToyBudgetSession
        or source_budget.updates_completed != 12
        or source_budget.process_rss_sampler is not memory
        or memory is None
        or not callable(check_boundary)
        or row is not None
        and type(row) is not dict
    ):
        raise ValueError("reserved probe requires the original ordinary native checkpoint12")


def run_owned_probe(
    state,
    spec,
    source_report,
    source_budget,
    *,
    check_boundary,
    memory,
    row=None,
    ce_initialization: CategoricalCEInitialization | None = None,
    check_initialization=None,
    linear_probe_update=None,
    probe_update_diagnostics=False,
    probe_output_only=False,
    ce_rehearsal=False,
    linear_rehearsal=None,
    source_stream=None,
    linear_return_once=False,
):
    """Keep the original source guarded while only detached descendants learn."""
    from src.core.categorical_linear_state import LinearFitSnapshot
    from src.app.coupled_attribution import AttributionStream

    if type(linear_return_once) is not bool or linear_return_once and linear_rehearsal is None:
        raise ValueError("once-return requires the explicit shared rehearsal callback")
    if (
        linear_rehearsal is not None
        and (
            type(linear_rehearsal) is not str
            or linear_rehearsal not in ("joint", "current_only")
            or type(state) is not LinearFitSnapshot
            or linear_probe_update != "minimum_change"
            or type(source_stream) is not AttributionStream
            or source_stream._budget is not source_budget
            or ce_rehearsal
            or ce_initialization is not None
            or probe_output_only
            or probe_update_diagnostics
        )
        or linear_rehearsal is None
        and source_stream is not None
    ):
        raise ValueError("shared rehearsal requires explicit ordinary shared-head source authority")
    if (
        type(ce_rehearsal) is not bool
        or ce_rehearsal
        and (
            type(state) is not TorchFeatureSnapshot
            or state.policy.method != "bp"
            or ce_initialization is None
            or linear_probe_update is not None
            or probe_output_only is not False
            or probe_update_diagnostics is not False
        )
    ):
        raise ValueError("CE rehearsal requires initialized ordinary BP without SGD diagnostics")
    if (
        type(probe_output_only) is not bool
        or probe_output_only
        and (type(state) is not TorchFeatureSnapshot or state.policy.method != "bp")
    ):
        raise ValueError("output-only probe requires an exact bool and native BP")
    if (
        type(probe_update_diagnostics) is not bool
        or probe_update_diagnostics
        and (type(state) is not TorchFeatureSnapshot or state.policy.method != "bp")
    ):
        raise ValueError("probe update diagnostics require an exact bool and native BP")
    if row is not None and (type(row) is not dict or row):
        raise ValueError("reserved probe requires a fresh report; paid work cannot be renewed")
    if ce_initialization is not None and type(ce_initialization) is not CategoricalCEInitialization:
        raise ValueError("reserved probe requires an exact CE initialization")
    if check_initialization is not None and not callable(check_initialization):
        raise ValueError("reserved probe initialization guard must be callable")
    _require_probe_source(
        state,
        spec,
        source_report,
        source_budget,
        check_boundary=check_boundary,
        memory=memory,
        row=row,
        linear_probe_update=linear_probe_update,
    )
    linear = linear_probe_update is not None
    if (
        "reserved_probe_rehearsal" in source_report
        if linear_rehearsal is None
        else source_report.get("reserved_probe_rehearsal") != linear_rehearsal
    ):
        raise ValueError("shared rehearsal intent differs from original source")
    if ("reserved_probe_return_once" in source_report) is not linear_return_once or (
        source_report.get("reserved_probe_return_once", False) is not linear_return_once
    ):
        raise ValueError("once-return intent differs from original source")
    same = _same_value if linear else native.same_state
    if linear and (ce_initialization is not None or check_initialization is not None):
        raise ValueError("linear reserved probe cannot use native CE initialization")
    prior = prior_guard = None
    if ce_initialization is not None and "probe_initialization" in source_report:
        prior = source_report["probe_initialization"]
        if type(prior) is not dict or prior.get("complete") is not True:
            raise ValueError("CE probe requires preceding complete initialization evidence")
        prior_guard, _ = paid_report_guard(prior)
    started = monotonic()
    # Why: value-equal receipt replacement during a copy must not become baseline.
    initialization_binding = (
        None
        if ce_initialization is None
        else (
            ce_initialization.fit,
            ce_initialization.state,
            ce_initialization.fit.source,
            ce_initialization.fit.work,
        )
    )
    pin = deepcopy(state)
    spec_pin = deepcopy(spec)
    initialization_pin = deepcopy(ce_initialization)
    initialization_record = None
    source_tag = (
        source_report["definition"],
        source_report["method"],
        source_report["reserved_probe_source_training"],
    )
    source_values = deepcopy(source_tag)
    source_record = asdict(state) if linear else native.snapshot_record(state)
    if source_report["checkpoints"][-1]["state"] != source_record:
        raise ValueError("reserved source checkpoint record changed")
    checkpoint = source_report["checkpoints"][-1] if linear else None
    checkpoint_pin = deepcopy(checkpoint)
    source_binding = (
        source_budget.budget,
        source_budget.started_at,
        source_budget.clock,
        source_budget.progress,
        memory,
    )
    row = {} if row is None else row
    row.update(
        definition="ordinary_linear_reserved_probe_v1"
        if linear
        else "ordinary_native_reserved_probe_v1",
        complete=False,
        method="linear_fit" if linear else state.policy.method,
        source_updates=12,
        source_rows=72,
        declared_tasks=PAIRED_TASKS,
        final_examples=0,
        replay_examples=0,
        episodes={},
        task_readout_work={},
    )
    if linear:
        row.update(
            linear_probe_update=linear_probe_update,
            source_train_update=source_report.get("linear_train_update"),
        )
    if linear_rehearsal is not None:
        row["linear_probe_rehearsal"] = linear_rehearsal
    if linear_return_once:
        row["linear_probe_return_once"] = True
    if ce_rehearsal:
        row["probe_update"] = "ce_rehearsal"
    if probe_output_only:
        row["parameter_update_scope"] = "output_only"
    fixed = deepcopy(
        {
            k: row[k]
            for k in (
                "definition",
                "method",
                "source_updates",
                "source_rows",
                "declared_tasks",
                "final_examples",
                "replay_examples",
                *(("linear_probe_update", "source_train_update") if linear else ()),
                *(("parameter_update_scope",) if probe_output_only else ()),
                *(("probe_update",) if ce_rehearsal else ()),
                *(("linear_probe_return_once",) if linear_return_once else ()),
            )
            if k != "replay_examples" or linear_rehearsal is None
        }
    )
    attempted = attempted_rows = returned = 0
    paid = row["training_work"] = dict(attempted_updates=0, attempted_rows=0, returned_updates=0)
    sessions = []
    readout_pins: dict[str, Any] = {}
    update_records: list[dict[str, Any]] = []
    update_work = dict(
        snapshot_attempts=0,
        snapshot_returns=0,
        summary_attempts=0,
        summary_returns=0,
        scalar_values_read=0,
    )
    update_table: dict[str, Any] = dict(records=update_records, work=update_work)
    if probe_output_only:
        update_table["parameter_update_scope"] = "output_only"
    update_guard = update_pin = None
    if probe_update_diagnostics:
        row["update_diagnostics"] = update_table
        update_guard, update_pin = paid_report_guard(update_table)

    ce_work = None
    linear_work = None

    def require():
        check_boundary()
        if (
            ("reserved_probe_return_once" in source_report) is not linear_return_once
            or source_report.get("reserved_probe_return_once", False) is not linear_return_once
            or ("linear_probe_return_once" in row) is not linear_return_once
            or row.get("linear_probe_return_once", False) is not linear_return_once
        ):
            raise ValueError("original once-return intent changed")
        if linear_rehearsal is not None and (
            source_report.get("reserved_probe_rehearsal") != linear_rehearsal
            or row.get("linear_probe_rehearsal") != linear_rehearsal
        ):
            raise ValueError("original shared rehearsal intent changed")
        if linear_work is not None:
            linear_work.check()
        elif linear_rehearsal is not None and row.get("replay_examples") != 0:
            raise ValueError("unobserved auxiliary exposure counter change")
        if linear:
            # Why: public mode/role headers are authority too; a late change
            # must refuse before the next private operation, not become baseline.
            _require_probe_source(
                state,
                spec,
                source_report,
                source_budget,
                check_boundary=check_boundary,
                memory=memory,
                linear_probe_update=linear_probe_update,
            )
        if ce_work is not None:
            ce_work.check()
        if ce_rehearsal and (
            type(source_report.get("reserved_probe_update")) is not str
            or source_report.get("reserved_probe_update") != "ce_rehearsal"
        ):
            raise ValueError("CE probe update intent changed")
        source_budget.before_final()
        if update_guard is not None:
            if row.get("update_diagnostics") is not update_table:
                raise ValueError("paid probe update diagnostics replaced")
            update_guard()
        if check_initialization is not None:
            check_initialization()
        if prior_guard is not None:
            if source_report.get("probe_initialization") is not prior:
                raise ValueError("CE probe initialization evidence replaced")
            prior_guard()
        if linear and any(
            row.get("episodes", {}).get(phase) is not records
            or row.get("task_readout_work", {}).get(phase) is not work
            or not same(records, values)
            or not same(work, work_values)
            for phase, (records, values, work, work_values) in readout_pins.items()
        ):
            raise ValueError("preceding linear reserved readout evidence changed")
        if (
            source_report.get("reserved_probe_requested") is not True
            or linear
            and (
                source_report.get("reserved_probe_update") != linear_probe_update
                or source_report.get("linear_train_update") != fixed["source_train_update"]
                or source_report.get("checkpoints", [])[-1:] != [checkpoint_pin]
                or source_report["checkpoints"][-1] is not checkpoint
            )
            or not same(source_tag, source_values)
            or (
                source_report.get("definition"),
                source_report.get("method"),
                source_report.get("reserved_probe_source_training"),
            )
            != source_tag
            or source_report.get("reserved_probe_source_training") is not source_tag[2]
            or source_budget.started_at != source_binding[1]
            or any(
                current is not original
                for current, original in zip(
                    (
                        source_budget.budget,
                        source_budget.clock,
                        source_budget.progress,
                        source_budget.process_rss_sampler,
                    ),
                    (source_binding[0], source_binding[2], source_binding[3], source_binding[4]),
                    strict=True,
                )
            )
            or source_budget.updates_completed != 12
            or not same(state, pin)
            or spec != spec_pin
            or row.get("training_work") is not paid
            or paid
            != dict(
                attempted_updates=attempted,
                attempted_rows=attempted_rows,
                returned_updates=returned,
            )
            or any(row.get(k) != v for k, v in fixed.items())
            or not same(ce_initialization, initialization_pin)
            or ce_initialization is not None
            and initialization_binding is not None
            and any(
                left is not right
                for left, right in zip(
                    (
                        ce_initialization.fit,
                        ce_initialization.state,
                        ce_initialization.fit.source,
                        ce_initialization.fit.work,
                    ),
                    initialization_binding,
                    strict=True,
                )
            )
            or initialization_record is not None
            and (
                row.get("initialization") is not initialization_record
                or not same(row.get("initialization"), initialization_record_pin)
            )
        ):
            raise ValueError("reserved probe source or paid work changed")

    def read_tasks(learner, phase, boundary=None):
        if not linear:
            _read_tasks(learner, phase, spec, require, row, boundary=boundary)
            return
        require()
        _read_tasks(
            learner,
            phase,
            spec,
            require,
            row,
            boundary=boundary,
            **(dict(linear=True) if linear else {}),
        )
        require()
        if linear:
            records = row["episodes"][phase]
            work = row["task_readout_work"][phase]
            readout_pins[phase] = (records, deepcopy(records), work, deepcopy(work))

    if ce_rehearsal:
        from scripts.ticket_ce_updates import CEProbeWork

        ce_work = CEProbeWork(row, require)

    def record_work(key, amount=1):
        nonlocal update_guard, update_pin
        require()
        update_work[key] += amount
        update_guard, update_pin = paid_report_guard(update_table)

    def read_update_state(learner):
        record_work("snapshot_attempts")
        value = learner.snapshot_state()
        record_work("snapshot_returns")
        return value

    class ProbeLearner(TorchFeatureLearner):
        def fork(self):
            child = ProbeLearner(self.policy)
            child.restore_state(self.snapshot_state())
            return child

        def train_batch(self, features, targets):
            nonlocal attempted, attempted_rows, returned, update_guard, update_pin
            require()
            before: Any = None
            entry: dict[str, Any] = {}
            if probe_update_diagnostics:
                if len(update_records) >= 32:
                    raise ValueError("probe update diagnostic allowance spent")
                entry = dict(status="pending", update=attempted + 1, rows=len(features))
                update_records.append(entry)
                update_guard, update_pin = paid_report_guard(update_table)
                before = read_update_state(self)
            attempted += 1
            attempted_rows += len(features)
            paid.update(attempted_updates=attempted, attempted_rows=attempted_rows)
            diagnostic = (
                super().train_batch_output_only(features, targets)
                if probe_output_only
                else super().train_batch(features, targets)
            )
            returned += 1
            paid["returned_updates"] = returned
            if probe_update_diagnostics:
                from src.adapters.torch_bp_step_norms import summarize_bp_update

                after = read_update_state(self)
                record_work("summary_attempts")
                measured = summarize_bp_update(before, after, rows=len(features))
                record_work("summary_returns")
                record_work("scalar_values_read", measured["work"]["scalar_values_read"])
                require()
                entry.update(status="complete", measurement=measured)
                update_guard, update_pin = paid_report_guard(update_table)
            return diagnostic

    # Why: source forks carry source-budget callbacks. Only detached descendants
    # use this existing explicit LS operation and its separately paid callbacks.
    if linear:
        from scripts.ticket_linear_probe import (
            MinimumChangeProbeLearner,
            MinimumChangeSingleProbeLearner,
        )
        from src.core.categorical_linear_state import LinearFitSnapshot

        # Only the disposable descendant changes type. The composed two-head
        # binder and its cold/source defaults retain their existing contract.
        probe_base: Any = (
            MinimumChangeSingleProbeLearner
            if type(pin) is LinearFitSnapshot
            else MinimumChangeProbeLearner
        )

        class LinearProbeLearner(probe_base):
            def fork(self):
                child = LinearProbeLearner(self.policy)
                child.restore_state(self.snapshot_state())
                return child

            def train_batch(self, features, targets):
                nonlocal attempted, attempted_rows, returned
                require()
                attempted += 1
                attempted_rows += len(features)
                paid.update(attempted_updates=attempted, attempted_rows=attempted_rows)
                diagnostic = super().train_batch(features, targets)
                returned += 1
                paid["returned_updates"] = returned
                return diagnostic

    try:
        require()
        initial_state = pin
        if ce_initialization is not None:
            initialization = copy_ce_initialization(ce_initialization)
            if not same(initialization.fit.source, pin) or initialization.fit.spec != spec:
                raise ValueError("CE probe initialization differs from original source/schema")
            initial_state = initialization.state
            initialization_record = ce_initialization_record(initialization)
            initialization_record_pin = deepcopy(initialization_record)
            row["initialization"] = initialization_record
            require()
        cohorts = build_head_cohorts(spec, head_task_transfer=True)
        require()
        if linear_rehearsal is not None:
            from scripts.ticket_linear_rehearsal import SourceTrainPackets, LinearProbeWork
            from scripts.ticket_linear_rehearsal import LIMITS as LINEAR_REHEARSAL_LIMITS
            from scripts.ticket_linear_rehearsal import ONCE_LIMITS

            certified = SourceTrainPackets(source_stream, pin)
            auxiliary = ToyBudgetSession(
                ONCE_LIMITS if linear_return_once else LINEAR_REHEARSAL_LIMITS, source_budget.clock
            )
            auxiliary.attach_memory(memory)
            linear_work = LinearProbeWork(
                row,
                require,
                certified,
                auxiliary,
                control=linear_rehearsal,
                **(dict(return_once=True) if linear_return_once else {}),
            )
            require()
        sessions = [ToyBudgetSession(LIMITS, monotonic) for _ in range(2)]
        for session in sessions:
            session.attach_memory(memory)
        # Why: a source's counted fork carries source callbacks. Build a detached
        # owner with the same full native state and independently paid callbacks.
        seed: Any = LinearProbeLearner(pin.policy) if linear else ProbeLearner(pin.policy)
        seed.restore_state(initial_state)
        require()
        read_tasks(seed, "before_probe")
        callback_options: dict[str, Any] = (
            {} if ce_work is None else dict(update_callback=ce_work.callback)
        )
        if linear_work is not None:
            callback_options["update_callback"] = linear_work.callback
        row["probe"] = run_fresh_task_probe(
            seed,
            initial_state,
            cohorts[0],
            cohorts[1],
            budget=sessions[0],
            checkpoint_updates=CHECKPOINTS,
            training_readouts=cohorts[2],
            return_training=cohorts[3],
            return_budget=sessions[1],
            decision_diagnostics=True,
            input_diagnostics=True,
            check_boundary=require,
            endpoint_observer=lambda learner, phase, boundary: read_tasks(learner, phase, boundary),
            **callback_options,
        )
        require()
        probe = row["probe"]
        if ce_work is not None:
            ce_work.complete(probe)
        if linear_work is not None:
            linear_work.complete(probe)
        if (
            probe["status"] != "complete"
            or probe.get("return_phase", {}).get("status") != "complete"
            or (attempted, attempted_rows, returned)
            != ((0, 0, 0) if ce_rehearsal or linear_work is not None else (32, 144, 32))
            or [s.updates_completed for s in sessions] != [16, 16]
        ):
            raise ValueError("reserved probe incomplete; paid partial work retained")
        row.update(complete=True, source_unchanged=True)
    except Exception as error:
        row.update(complete=False, error=f"{type(error).__name__}: {error}"[:512])
    finally:
        if linear_work is not None:
            row["preserved_linear_rehearsal_work"] = linear_work.preserved
        if ce_work is not None:
            row["preserved_ce_rehearsal_work"] = ce_work.preserved
        if probe_update_diagnostics:
            row["preserved_update_diagnostics"] = deepcopy(update_pin)
        row["preserved_training_work"] = dict(
            attempted_updates=attempted, attempted_rows=attempted_rows, returned_updates=returned
        )
        row["preserved_metadata"] = fixed
        if linear:
            row["preserved_readouts"] = {
                phase: dict(episodes=deepcopy(values), work=deepcopy(work_values))
                for phase, (_, values, _, work_values) in readout_pins.items()
            }
        row["sessions"] = [
            dict(
                limits=asdict(s.budget),
                started_at=s.started_at,
                last_clock=s.last_clock,
                updates=s.updates_completed,
            )
            for s in sessions
        ]
        row["elapsed_seconds"] = monotonic() - started
    return row
