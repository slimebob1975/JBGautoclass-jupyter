"""297c: a training run that also predicts, on a table where every row has a
class, trains only instead of failing at its last step (found 2026-11-10);
a finished inner progress bar stays until the stage changes."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("sqlalchemy")

from Config import Config  # noqa: E402
from JBGApplicationContracts import DataSourceKind  # noqa: E402
from JBGRunWorker import NOTHING_TO_PREDICT, train_only_without_unlabelled_rows  # noqa: E402

TRAINING_RUN = SimpleNamespace(spec={"project_run": {"project_id": "p", "dataset_id": "d"}})


def _layer(distribution, *, train=True, predict=True):
    from SQLDataLayer import DataLayer
    config = Config()
    config.mode.train, config.mode.predict = train, predict
    layer = DataLayer.__new__(DataLayer)
    layer.config = config
    asked = []

    def count():
        asked.append(True)
        if isinstance(distribution, Exception):
            raise distribution
        return distribution
    layer.count_class_distribution = count
    return layer, asked


class _Logger:
    def __init__(self):
        self.info = []

    def print_info(self, *args):
        self.info.append(" ".join(str(a) for a in args))


def test_without_rows_to_predict_the_run_trains_only():
    layer, asked = _layer({"a": 200, "b": 86})
    logger = _Logger()
    assert train_only_without_unlabelled_rows(TRAINING_RUN, layer, logger) is True
    assert (layer.config.should_train(), layer.config.should_predict()) == (True, False)
    assert asked == [True] and logger.info == [NOTHING_TO_PREDICT]


@pytest.mark.parametrize("distribution", [
    {"a": 200, "NULL": 3},           # NULL class values
    {"a": 200, "": 1},               # empty class values
])
def test_rows_without_a_class_are_still_predicted(distribution):
    layer, _ = _layer(distribution)
    logger = _Logger()
    assert train_only_without_unlabelled_rows(TRAINING_RUN, layer, logger) is False
    assert layer.config.should_predict() and logger.info == []


@pytest.mark.parametrize("record, train, predict", [
    (SimpleNamespace(spec={"project_run": {"kind": "prediction"}}), False, True),
    (SimpleNamespace(spec={"spec_version": 1}), True, True),           # not a project run
    (TRAINING_RUN, True, False),                                         # trains only already
])
def test_other_runs_are_left_alone_and_not_counted(record, train, predict):
    layer, asked = _layer({"a": 5}, train=train, predict=predict)
    assert train_only_without_unlabelled_rows(record, layer, _Logger()) is False
    assert asked == [] and layer.config.should_predict() is predict


def test_if_the_rows_cannot_be_counted_the_engine_decides(capsys):
    layer, _ = _layer(RuntimeError("timeout"))
    assert train_only_without_unlabelled_rows(TRAINING_RUN, layer, _Logger()) is False
    assert layer.config.should_predict()
    assert "rows without a class not counted: RuntimeError" in capsys.readouterr().err


def test_the_worker_checks_before_composing_an_sql_run():
    from JBGRunWorker import _compose
    layer, asked = _layer({"a": 10, "b": 4})
    record = SimpleNamespace(source=DataSourceKind.SQL, spec=TRAINING_RUN.spec)
    composed = SimpleNamespace(sql_dataset_key=lambda: "key", configuration=layer.config)
    with patch("JBGSQLSessions.open_sql_data_layer", return_value=layer), \
            patch("JBGRunComposition.compose_sql_run_on_layer", return_value=composed) as wire:
        _, _, run_config = _compose(record, layer.config, _Logger(), credentials=object(),
                                    artifacts=None)
    assert asked == [True] and wire.called
    assert run_config.should_train() and not run_config.should_predict()


def test_the_saved_configuration_is_not_changed(tmp_path):
    """Only the run's own Config copy: the spec (and the saved configuration) keep predict on."""
    from JBGRunJobStore import RunJobStore
    from JBGRunWorker import _config_from_spec
    store = RunJobStore.sqlite(tmp_path / "jobs.sqlite")
    store.create_tables()
    config = Config()
    run = store.create("owner", DataSourceKind.SQL, {
        "spec_version": 1, "configuration": config._config_to_json(username=""),
        "project_run": TRAINING_RUN.spec["project_run"]})
    record = store.get(run)
    layer, _ = _layer({"a": 1, "b": 1})
    layer.config = _config_from_spec(record.spec, None)
    train_only_without_unlabelled_rows(record, layer, _Logger())
    assert store.get(run).spec["configuration"]["mode"]["predict"] is True


# --- The inner bar stays until the stage changes ----------------------------------------------

def test_a_bar_that_ends_short_keeps_its_latest_count(tmp_path):
    from test_revision_297a_inner_progress import _proxy, _stage
    proxy, store, run = _proxy(tmp_path, interval=60.0)
    proxy.start_inline_progress("f", "Fetching data", 1000, "t")
    proxy.update_inline_progress("f", 250, "t")
    proxy.update_inline_progress("f", 600, "t")              # within the interval: not written
    assert _stage(store, run)["inner"]["done"] == 0
    proxy.end_inline_progress("f", set_100=False)            # the SQL fetch ends this way
    assert _stage(store, run)["inner"] == {"kind": "fetch_data", "done": 600, "total": 1000}


def test_a_finished_bar_is_hidden_with_the_next_stage(tmp_path):
    from test_revision_297a_inner_progress import _proxy, _stage
    proxy, store, run = _proxy(tmp_path)
    store.record_progress(run, "jbg_stage:train_model", 0.45)
    proxy.start_inline_progress("k", "Checking & Training models", 9, "t")
    proxy.update_inline_progress("k", 8, "t")
    proxy.end_inline_progress("k")
    store.record_progress(run, "Grid search: estimated ~10 s")          # same stage
    assert _stage(store, run)["inner"] == {"kind": "models", "done": 9, "total": 9}
    store.record_progress(run, "jbg_stage:evaluate_model", 0.55)
    stage = _stage(store, run)
    assert stage["name"] == "evaluate_model" and "inner" not in stage
