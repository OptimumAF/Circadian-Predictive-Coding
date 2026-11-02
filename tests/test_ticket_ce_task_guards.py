"""Real ordinary orchestration with scripted metadata ports, never a model."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace
from typing import Any
import os

import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import ticket_ce_tasks as tasks
from src.adapters.torch_categorical_ce_readout import CEReadoutWork
from src.adapters.torch_decision import TorchDecisionWork
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from src.app.coupled_attribution import _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.learner_ports import TrainingDiagnostic
from src.shared.process_memory import ProcessRssSampler

COUNTS = dict(fake_updates=0, fake_rows=0, fake_forks=0, pure_dispatch=0)


def forbidden(*args, **kwargs):
    raise AssertionError("genuine model operation forbidden in metadata tests")


@dataclass(frozen=True)
class State:
    policy: Any
    completed_updates: int = 0
    training_examples: int = 0


@dataclass(frozen=True)
class Rng:
    value: str = "scripted, no native RNG"

    def clone(self):
        return replace(self)


@dataclass(frozen=True)
class Fit:
    source: Any
    spec: Any
    coefficients: tuple
    work: dict
    final_loss: float


class Learner:
    """Counter-only implementation of the existing forkable port."""

    def __init__(self, before, after, state=None):
        self.before, self.after = before, after
        self.state = state or State(TorchFeaturePolicy("bp", 9, 3, 17, num_classes=6))
        self._payload = object()

    @property
    def completed_updates(self):
        return self.state.completed_updates

    @property
    def training_examples(self):
        return self.state.training_examples

    def fork(self):
        COUNTS["fake_forks"] += 1
        return Learner(self.before, self.after, deepcopy(self.state))

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)

    def train_batch(self, features, targets):
        self.before(features)
        COUNTS["fake_updates"] += 1
        COUNTS["fake_rows"] += len(features)
        assert COUNTS["fake_updates"] <= int(os.environ.get("CE_FAKE_UPDATES_REMAINING", "256"))
        assert COUNTS["fake_rows"] <= int(os.environ.get("CE_FAKE_ROWS_REMAINING", "1536"))
        self.state = replace(
            self.state,
            completed_updates=self.completed_updates + 1,
            training_examples=self.training_examples + len(features),
        )
        self.after()
        return TrainingDiagnostic("scripted_counter_only_not_native", 0.0)

    predict = forbidden


class Reader:
    def __init__(self, state):
        self.state = deepcopy(state)
        self.work = TorchDecisionWork()

    def assert_unchanged(self):
        pass


class CEReader:
    def __init__(self, fit):
        self.work = CEReadoutWork()

    def assert_unchanged(self):
        pass


@pytest.fixture
def metadata(monkeypatch):
    errors = []
    sessions = []
    from scripts import run_ticket_workflow

    dispatch = run_ticket_workflow.dispatch_workflow

    def observed_dispatch(*args, **kwargs):
        COUNTS["pure_dispatch"] += 1
        assert COUNTS["pure_dispatch"] <= int(os.environ.get("CE_DISPATCH_REMAINING", "4096"))
        return dispatch(*args, **kwargs)

    monkeypatch.setattr(run_ticket_workflow, "dispatch_workflow", observed_dispatch)
    attach = ToyBudgetSession.attach_memory

    def observed_attach(session, memory):
        sessions.append(session)
        return attach(session, memory)

    monkeypatch.setattr(ToyBudgetSession, "attach_memory", observed_attach)
    factory = lambda method, dim, classes, before, after: Learner(before, after)
    monkeypatch.setattr(tasks.native, "counted_learner", factory)
    monkeypatch.setattr(tasks.native, "same_state", _same_value)
    monkeypatch.setattr(
        tasks.native,
        "require_torch",
        lambda: SimpleNamespace(get_rng_state=Rng, equal=lambda a, b: a == b),
    )
    monkeypatch.setattr(
        tasks.native,
        "checkpoint",
        lambda point, state: dict(
            source_updates=point,
            source_rows=state.training_examples,
            state=asdict(state),
            tensor_bytes=0,
        ),
    )
    monkeypatch.setattr(tasks.native, "readout", lambda spec, state: Reader(state))

    def require_readout(reader, spec, state):
        assert _same_value(reader.state, state)

    monkeypatch.setattr(tasks.native, "require_readout", require_readout)

    @contextmanager
    def unchanged_owner(source):
        state, owner = source.snapshot_state(), source._payload

        def check():
            if source._payload is not owner or not _same_value(source.snapshot_state(), state):
                raise ValueError("scripted source changed")

        try:
            yield state, check
        finally:
            check()

    monkeypatch.setattr(tasks, "_unchanged_owner", unchanged_owner)
    monkeypatch.setattr(tasks.native, "prepare_train_fit", lambda spec: ((), None))

    def fit(source, cases, spec, *, check_boundary):
        check_boundary()
        value = Fit(
            source=source.snapshot_state(),
            spec=spec,
            coefficients=((0.0,),),
            work=dict(step_returns=1, backward_rows=0),
            final_loss=0.0,
        )
        check_boundary()
        return value

    monkeypatch.setattr(tasks, "fit_categorical_ce_readout", fit)
    monkeypatch.setattr(tasks, "CategoricalCEReadoutPredictor", CEReader)

    def episode(*args, scenario="ordinary", **kwargs):
        return dict(
            complete=True,
            success=False,
            stopped_reason="scripted_no_action",
            scenario=scenario,
            steps=[],
            work={},
        )

    monkeypatch.setattr(tasks, "run_workflow_episode", episode)
    monkeypatch.setattr(entry, "run_workflow_episode", episode)
    original_run = tasks.run_train_readout

    def observed_run(*args, **kwargs):
        row, guard = original_run(*args, **kwargs)

        def observed_guard():
            try:
                guard()
            except Exception as error:
                errors.append(str(error))
                raise

        return row, observed_guard

    monkeypatch.setattr(tasks, "run_train_readout", observed_run)
    return SimpleNamespace(factory=factory, fit=fit, errors=errors, sessions=sessions)


def test_actual_ordinary_stream_finish_and_final_report_complete_with_metadata_ports(
    metadata, record_property
):
    report = entry.run_example(method="bp", context_return=True, train_readout="ce")
    record_property(
        "ordinary_metadata",
        dict(
            complete=report["complete"],
            error=report.get("error"),
            aux_complete=report.get("train_readout", {}).get("complete"),
            source=report.get("source_unchanged"),
            actor=report.get("actor_unchanged"),
            stream=report.get("stream", {}).get("completed_updates"),
            first_guard_errors=metadata.errors,
        ),
    )
    assert report["complete"] is True, report.get("error")
    assert report["stream"]["completed_updates"] == 12
    assert report["training_work"]["returned_updates"] == 12
    assert len(report["train_readout"]["episodes"]) == 4
    assert "train_readout_cohort_requested" not in report
    assert "training_cohort" not in report["train_readout"]
    assert report["source_unchanged"] is report["actor_unchanged"] is True


@pytest.fixture
def direct(metadata, monkeypatch):
    """Explicit initial metadata counters, not executed source updates."""
    policy = TorchFeaturePolicy("bp", 9, 3, 17, num_classes=6)
    source = Learner(lambda features: None, lambda: None, State(policy, 12, 72))
    state = source.snapshot_state()
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=12, max_wall_seconds=30, max_process_rss_bytes=512 * 1024**2
            ),
            lambda: 0.0,
        )
        budget.attach_memory(memory)
        budget.updates_completed = 12
        ledger = {i: dict(updates=1, rows=6) for i in range(12)}

        class Stream:
            completed_updates = 12
            _budget = budget
            _runtime = SimpleNamespace(_candidate=source, _inbox=SimpleNamespace(_applied=ledger))

            @property
            def applied_updates(self):
                return deepcopy(tuple(self._runtime._inbox._applied.values()))

        stream = Stream()
        row: dict[str, Any] = {}
        report = dict(
            definition="native_context_return_ticket_workflow_v1",
            method="bp",
            source_unchanged=True,
            actor_unchanged=True,
            final_examples=0,
            probe_updates=0,
            replay_examples=0,
            train_readout_requested="ce",
            train_readout_source_training=dict(completed_updates=12, applied_update_count=12),
            train_readout=row,
        )
        spec = entry.DecisionFeatureSpec(entry.CONTEXT_FEATURES, entry.CATALOG)
        fits = []

        def fit(*args, **kwargs):
            value = metadata.fit(*args, **kwargs)
            fits.append(value)
            return value

        monkeypatch.setattr(tasks, "fit_categorical_ce_readout", fit)
        result, guard = tasks.run_train_readout(
            stream, state, spec, report, budget, check_boundary=lambda: None, memory=memory, row=row
        )
        assert result is row and row["complete"]
        yield SimpleNamespace(
            source=source,
            state=state,
            stream=stream,
            row=row,
            report=report,
            budget=budget,
            aux=metadata.sessions[-1],
            fit=fits[0],
            spec=spec,
            guard=guard,
            ledger=ledger,
        )


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("runtime", "receipt"),
        ("candidate", "candidate owner"),
        ("budget", "source budget owner"),
        ("report", "task report owner"),
        ("aux_report", "auxiliary budget report owner"),
        ("intent", "readout intent"),
        ("source_flag", "source preservation flag"),
        ("actor_flag", "actor preservation flag"),
        ("final", "final_examples"),
        ("probe", "probe_updates"),
        ("replay", "replay_examples"),
        ("source_tag_owner", "source training record owner"),
        ("source_tag_value", "source training record value"),
        ("checkpoint", "source checkpoint value"),
        ("spec", "feature schema"),
        ("fit_value", "fit value"),
        ("fit_source", "fit source owner"),
        ("fit_spec", "fit schema owner"),
        ("fit_coefficients", "fit coefficients owner"),
        ("fit_work", "fit work owner"),
        ("source_limits_value", "source budget limits"),
        ("source_birth", "source budget birth"),
        ("source_updates", "source completed updates"),
        ("source_replay", "source replay charge"),
        ("source_limits_owner", "source budget limits owner"),
        ("source_clock", "source budget clock owner"),
        ("source_progress", "source budget progress owner"),
        ("source_memory", "source budget memory owner"),
        ("aux_limits_value", "auxiliary budget limits"),
        ("aux_birth", "auxiliary budget birth"),
        ("aux_updates", "auxiliary update charge"),
        ("aux_replay", "auxiliary replay charge"),
        ("aux_limits_owner", "auxiliary budget limits owner"),
        ("aux_clock", "auxiliary budget clock owner"),
        ("aux_progress", "auxiliary budget progress owner"),
        ("aux_memory", "auxiliary budget memory owner"),
    ],
)
def test_named_original_guard_refusals_keep_paid_prefix_and_foreign_changes(direct, fault, reason):
    d = direct
    if fault == "runtime":
        d.stream._runtime = deepcopy(d.stream._runtime)
    elif fault == "candidate":
        d.stream._runtime._candidate = d.source.fork()
    elif fault == "budget":
        d.stream._budget = object()
    elif fault == "report":
        d.report["train_readout"] = deepcopy(d.row)
    elif fault == "aux_report":
        d.report["train_readout_budget"] = deepcopy(d.report["train_readout_budget"])
    elif fault == "intent":
        d.report["train_readout_requested"] = "other"
    elif fault == "source_flag":
        d.report["source_unchanged"] = None
    elif fault == "actor_flag":
        d.report["actor_unchanged"] = None
    elif fault in ("final", "probe", "replay"):
        d.report[
            dict(final="final_examples", probe="probe_updates", replay="replay_examples")[fault]
        ] = 1
    elif fault == "source_tag_owner":
        d.report["train_readout_source_training"] = deepcopy(
            d.report["train_readout_source_training"]
        )
    elif fault == "source_tag_value":
        d.report["train_readout_source_training"]["completed_updates"] = 13
    elif fault == "checkpoint":
        object.__setattr__(d.state, "training_examples", 73)
    elif fault == "spec":
        object.__setattr__(d.spec, "vocabulary", ("foreign",))
    elif fault == "fit_value":
        object.__setattr__(d.fit, "final_loss", 1.0)
    elif fault in ("fit_source", "fit_spec", "fit_work"):
        key = dict(fit_source="source", fit_spec="spec", fit_work="work")[fault]
        object.__setattr__(d.fit, key, deepcopy(getattr(d.fit, key)))
    elif fault == "fit_coefficients":
        object.__setattr__(d.fit, "coefficients", tuple(list(d.fit.coefficients)))
    else:
        session = d.aux if fault.startswith("aux_") else d.budget
        field = fault.split("_", 1)[1]
        if field == "limits_value":
            session.budget = replace(session.budget, max_wall_seconds=31)
        elif field == "limits_owner":
            session.budget = deepcopy(session.budget)
        elif field == "birth":
            session.started_at = -1
        elif field == "updates":
            session.updates_completed += 1
        elif field == "replay":
            session.replay_examples_completed = 1
        elif field == "clock":
            session.clock = lambda: 0.0
        elif field == "progress":
            session.progress = object()
        elif field == "memory":
            session.process_rss_sampler = object()
        else:
            raise AssertionError(fault)
    with pytest.raises(ValueError, match=reason):
        d.guard()
    saved = d.report["preserved_train_readout"]
    assert not saved["complete"] and len(saved["episodes"]) == 4
    assert saved["work"]["task_attempts"] == saved["work"]["task_returns"] == 4
    assert d.report["train_readout_budget"] == d.report["preserved_train_readout_budget"]


def test_first_original_predicate_refuses_before_unsafe_later_access(direct):
    direct.stream._runtime = object()
    direct.stream._budget = object()
    with pytest.raises(ValueError, match="receipt"):
        direct.guard()


@pytest.mark.parametrize(
    "fault",
    ["row_value", "row_owner", "budget_value", "budget_owner", "receipt_owner", "receipt_value"],
)
def test_late_paid_record_and_receipt_refusals_remain_strict(direct, fault):
    d = direct
    if fault == "row_value":
        d.row["episodes"][0]["episode"]["success"] = True
    elif fault == "row_owner":
        d.row["episodes"][0] = deepcopy(d.row["episodes"][0])
    elif fault == "budget_value":
        d.report["train_readout_budget"]["elapsed_seconds"] = -1
    elif fault == "budget_owner":
        d.report["train_readout_budget"]["limits"] = deepcopy(
            d.report["train_readout_budget"]["limits"]
        )
    elif fault == "receipt_owner":
        d.ledger[0] = deepcopy(d.ledger[0])
    else:
        d.ledger[0]["rows"] = 7
    with pytest.raises(ValueError):
        d.guard()
    assert d.report["preserved_train_readout"]["episodes"][0]["episode"]["success"] is False


@pytest.fixture
def finished(metadata, monkeypatch):
    ports = []
    original = tasks._finished_snapshot

    def capture(stream, row, report):
        port = original(stream, row, report)
        ports.append(SimpleNamespace(stream=stream, row=row, report=report, port=port))
        return port

    monkeypatch.setattr(tasks, "_finished_snapshot", capture)
    report = entry.run_example(method="bp", context_return=True, train_readout="ce")
    assert report["complete"] is True
    assert len(ports) == 1
    return ports[0]


def test_finished_read_stays_bound_and_public_open_only_boundary_remains(finished):
    assert finished.port().completed_updates == 12
    with pytest.raises(ValueError, match="finished"):
        finished.stream.candidate_snapshot()


@pytest.mark.parametrize(
    "fault", ["open", "failed", "runtime", "candidate", "row_owner", "row_value", "busy"]
)
def test_finished_read_refuses_unobserved_or_changed_ownership_and_busy_stream(finished, fault):
    d = finished
    if fault == "open":
        d.stream._finished = False
    elif fault == "failed":
        d.stream._failed = True
    elif fault == "runtime":
        d.stream._runtime = object()
    elif fault == "candidate":
        d.stream._runtime._candidate = d.stream._runtime._candidate.fork()
    elif fault == "row_owner":
        d.report["stream"] = deepcopy(d.row)
    elif fault == "row_value":
        d.row["completed_updates"] = 13
    else:
        d.stream._gate.acquire()
    try:
        with pytest.raises(ValueError):
            d.port()
    finally:
        if fault == "busy":
            d.stream._gate.release()


def test_later_guard_refusal_does_not_overwrite_initiating_finish_error(metadata, monkeypatch):
    original = tasks._finished_snapshot

    def refuse_after_seal(*args):
        original(*args)

        def refuse():
            raise ValueError("initiating post-finish metadata refusal")

        return refuse

    monkeypatch.setattr(tasks, "_finished_snapshot", refuse_after_seal)
    report = entry.run_example(method="bp", context_return=True, train_readout="ce")
    assert (
        not report["complete"] and report["source_unchanged"] is report["actor_unchanged"] is None
    )
    assert report["error"] == "ValueError: initiating post-finish metadata refusal"
    assert report["preserved_train_readout"]["work"]["task_returns"] == 4


@pytest.mark.parametrize(
    "fault,reason", [("source", "fit source value"), ("schema", "fit feature schema")]
)
def test_foreign_fitter_result_refused_before_reader_admission(
    direct, metadata, monkeypatch, fault, reason
):
    d = direct
    d.report["train_readout"] = {}

    def fit(*args, **kwargs):
        value = metadata.fit(*args, **kwargs)
        if fault == "source":
            return replace(value, source=replace(value.source, training_examples=73))
        return replace(value, spec=replace(value.spec, vocabulary=("foreign",)))

    monkeypatch.setattr(tasks, "fit_categorical_ce_readout", fit)
    monkeypatch.setattr(tasks, "CategoricalCEReadoutPredictor", forbidden)
    with pytest.raises(ValueError, match=reason):
        tasks.run_train_readout(
            d.stream,
            d.state,
            d.spec,
            d.report,
            d.budget,
            check_boundary=lambda: None,
            memory=d.budget.process_rss_sampler,
            row=d.report["train_readout"],
        )
    assert d.report["preserved_train_readout"]["work"]["fit_attempts"] == 1
    assert "reader_attempts" not in d.report["preserved_train_readout"]["work"]


@pytest.mark.parametrize("field", ["final_examples", "probe_updates", "replay_examples"])
@pytest.mark.parametrize("value", [False, True, None])
def test_original_exact_zero_count_types_stay_strict(direct, field, value):
    direct.report[field] = value
    with pytest.raises(ValueError, match=field):
        direct.guard()


@pytest.mark.parametrize(
    "options",
    [
        dict(train_readout=True),
        dict(train_readout=False),
        dict(train_readout=0),
        dict(train_readout="other"),
        dict(method="pc"),
        dict(context_return=False),
        dict(replay="recent"),
        dict(reserved_probe=True),
        dict(private_adaptation=True),
        dict(paired_serving="joint"),
        dict(probe_initialization="ce"),
        dict(record_train_fit=True),
    ],
)
def test_invalid_modes_still_refuse_before_data_or_genuine_model_ports(
    metadata, monkeypatch, options
):
    monkeypatch.setattr(entry, "prepare_training", forbidden)
    request = dict(method="bp", context_return=True, train_readout="ce")
    request.update(options)
    with pytest.raises(ValueError, match="TRAIN CE readout"):
        entry.run_example(**request)


def test_ordinary_default_does_not_admit_auxiliary_fit_or_closed_ce_port(metadata, monkeypatch):
    monkeypatch.setattr(tasks, "fit_categorical_ce_readout", forbidden)
    monkeypatch.setattr(tasks, "_finished_snapshot", forbidden)
    report = entry.run_example(method="bp", context_return=True)
    assert report["complete"] is True and report["stream"]["completed_updates"] == 12
    assert "train_readout" not in report and "train_readout_requested" not in report


@pytest.mark.parametrize("value", [False, True, 0, "old", [], {}])
def test_joint_cohort_requires_exact_selection_before_data(value, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", forbidden)
    with pytest.raises(ValueError, match="TRAIN CE readout"):
        entry.run_example(
            method="bp", context_return=True, train_readout="ce", train_readout_cohort=value
        )


@pytest.mark.parametrize(
    "change",
    [
        dict(train_readout=None),
        dict(context_return=False),
        dict(method="pc"),
        dict(reserved_probe=True),
    ],
)
def test_joint_cohort_never_admits_another_mode(change, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", forbidden)
    options = dict(
        method="bp", context_return=True, train_readout="ce", train_readout_cohort="joint"
    )
    options.update(change)
    with pytest.raises(ValueError, match="TRAIN CE readout"):
        entry.run_example(**options)


@pytest.mark.parametrize("cohort", [None, "joint"])
def test_joint_cli_forwarding_preserves_default_omission_and_canonical_failure(
    cohort, tmp_path, monkeypatch
):
    from scripts import context_catalog_learning as writer

    seen, written = [], []
    paid = dict(complete=False, training_cohort="joint", training_case_count=18)
    report = dict(complete=False, train_readout={}, preserved_train_readout=paid)

    def run(**kwargs):
        seen.append(kwargs)
        return report

    def write(*args, **kwargs):
        written.append(kwargs)
        return False

    monkeypatch.setattr(entry, "run_example", run)
    monkeypatch.setattr(writer, "write_catalog_report", write)
    flags = [
        "--context-return",
        "--method",
        "bp",
        "--train-readout",
        "ce",
        "--output",
        str(tmp_path / "new.json"),
    ]
    if cohort:
        flags += ["--train-readout-cohort", cohort]
    assert entry.main(flags) == 1
    assert seen == [
        dict(
            context_return=True,
            method="bp",
            train_readout="ce",
            **(dict(train_readout_cohort=cohort) if cohort else {}),
        )
    ]
    assert written[0]["observed_work"]["train_readout"] is paid


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--context-return", "--method", "bp"],
        ["--context-return", "--method", "pc", "--train-readout", "ce"],
    ],
)
def test_joint_cli_invalid_modes_refuse_before_dispatch(flags, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "run_example", forbidden)
    output = tmp_path / "new.json"
    with pytest.raises(SystemExit):
        entry.main([*flags, "--train-readout-cohort", "joint", "--output", str(output)])
    assert not output.exists()


def test_joint_real_stream_uses_only_original_train18_and_runs_all_six_negative_tasks(
    metadata, monkeypatch
):
    from scripts.ticket_train_fit import JOINT_TRAIN_TASKS

    seen = []

    def fit(source, cases, spec, **kwargs):
        seen.append(cases)
        return metadata.fit(source, cases, spec, **kwargs)

    monkeypatch.setattr(tasks, "fit_categorical_ce_readout", fit)
    report = entry.run_example(
        method="bp", context_return=True, train_readout="ce", train_readout_cohort="joint"
    )
    assert report["complete"], report.get("error")
    row = report["train_readout"]
    assert len(seen) == 1 and len(seen[0]) == 18
    assert all(c.role == "train" for c in seen[0])
    assert [c.target_id for c in seen[0][1::3]] == [
        "route",
        "inspect",
        "inspect",
        "defer",
        "route",
        "inspect",
    ]
    assert row["training_cohort"] == report["train_readout_cohort_requested"] == "joint"
    assert row["training_case_count"] == 18
    assert row["declared_tasks"] == [list(task) for task in JOINT_TRAIN_TASKS]
    assert [(r["policy_version"], r["concept"]) for r in row["episodes"]] == list(JOINT_TRAIN_TASKS)
    assert all(not r["episode"]["success"] for r in row["episodes"])
    assert row["work"]["task_attempts"] == row["work"]["task_returns"] == 6
    assert (
        report["stream"]["completed_updates"] == report["training_work"]["returned_updates"] == 12
    )
    assert report["source_unchanged"] is report["actor_unchanged"] is True
    assert report["final_examples"] == report["probe_updates"] == report["replay_examples"] == 0


@pytest.mark.parametrize("role", ["probe_test", "final", "controller_train"])
def test_joint_nontrain_teachers_refuse_before_fitting(metadata, monkeypatch, role):
    original = tasks.training_cases

    def teacher(**kwargs):
        return tuple(replace(c, role=role) for c in original(**kwargs))

    monkeypatch.setattr(tasks, "training_cases", teacher)
    monkeypatch.setattr(tasks, "fit_categorical_ce_readout", forbidden)
    report = entry.run_example(
        method="bp", context_return=True, train_readout="ce", train_readout_cohort="joint"
    )
    assert not report["complete"]
    paid = report["preserved_train_readout"]
    assert "fit_attempts" not in paid["work"] and not paid["episodes"]


@pytest.mark.parametrize("fault", ["intent", "case_count", "task_owner"])
def test_joint_corruption_refuses_next_task_and_retains_paid_prefix(metadata, monkeypatch, fault):
    active = {}
    original = tasks.run_train_readout

    def capture(*args, **kwargs):
        active["report"] = args[3]
        return original(*args, **kwargs)

    def clock():
        report = active.get("report", {})
        row = report.get("train_readout", {})
        if row.get("work", {}).get("task_returns") == 1 and not active.get("changed"):
            active["changed"] = True
            if fault == "intent":
                report["train_readout_cohort_requested"] = "old"
            elif fault == "case_count":
                row["training_case_count"] = 12
            else:
                row["declared_tasks"][0] = list(row["declared_tasks"][0])
        return 0.0

    monkeypatch.setattr(tasks, "run_train_readout", capture)
    monkeypatch.setattr(tasks, "monotonic", clock)
    report = entry.run_example(
        method="bp", context_return=True, train_readout="ce", train_readout_cohort="joint"
    )
    assert active.get("changed") and not report["complete"]
    paid = report["preserved_train_readout"]
    assert paid["training_cohort"] == "joint" and paid["training_case_count"] == 18
    assert paid["declared_tasks"][0] == ["v1", "invoice"]
    assert (
        paid["work"]["task_attempts"] == paid["work"]["task_returns"] == len(paid["episodes"]) == 1
    )
    if fault == "case_count":
        assert report["train_readout"]["training_case_count"] == 12


@pytest.mark.parametrize("fault", ["intent", "task_owner", "cohort_value"])
def test_joint_evidence_remains_guarded_after_actual_stream_finish(metadata, monkeypatch, fault):
    original = tasks._finished_snapshot

    def changed(stream, finished, report):
        result = original(stream, finished, report)
        assert stream._finished
        if fault == "intent":
            report["train_readout_cohort_requested"] = "old"
        elif fault == "task_owner":
            row = report["train_readout"]
            row["declared_tasks"] = deepcopy(row["declared_tasks"])
        else:
            report["train_readout"]["training_cohort"] = "old"
        return result

    monkeypatch.setattr(tasks, "_finished_snapshot", changed)
    report = entry.run_example(
        method="bp", context_return=True, train_readout="ce", train_readout_cohort="joint"
    )
    assert not report["complete"] and report["stream"]["completed_updates"] == 12
    paid = report["preserved_train_readout"]
    assert paid["training_cohort"] == "joint" and len(paid["declared_tasks"]) == 6
    assert paid["work"]["task_returns"] == len(paid["episodes"]) == 6


@pytest.mark.parametrize(
    "cohort,report",
    [(False, {}), ("old", {}), ("joint", {}), (None, {"train_readout_cohort_requested": None})],
)
def test_joint_helper_refuses_invalid_or_unobserved_intent_before_boundary(cohort, report):
    with pytest.raises(ValueError, match="cohort"):
        tasks.run_train_readout(
            None,
            None,
            None,
            report,
            None,
            check_boundary=forbidden,
            memory=None,
            row={},
            cohort=cohort,
        )


def test_joint_fit_failure_keeps_exact_paid_rows_without_reader(metadata, monkeypatch):
    from src.adapters.torch_categorical_ce_readout import CategoricalCEFitFailed

    def fail(source, cases, spec, **kwargs):
        assert len(cases) == 18
        raise CategoricalCEFitFailed(
            "scripted joint failure", dict(backward_attempts=2, backward_rows=36)
        )

    monkeypatch.setattr(tasks, "fit_categorical_ce_readout", fail)
    monkeypatch.setattr(tasks, "CategoricalCEReadoutPredictor", forbidden)
    report = entry.run_example(
        method="bp", context_return=True, train_readout="ce", train_readout_cohort="joint"
    )
    assert not report["complete"]
    paid = report["preserved_train_readout"]
    assert paid["training_case_count"] == 18 and paid["training_cohort"] == "joint"
    assert paid["work"] == dict(fit_attempts=1)
    assert paid["fit_work"] == dict(backward_attempts=2, backward_rows=36)
    assert not paid["episodes"]
