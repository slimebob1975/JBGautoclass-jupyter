from types import SimpleNamespace

from JBGTaskRunner import TaskRunner, get_tasks


class StubConfig:
    def __init__(self, display_mispredicted: bool, calculate_dark_numbers: bool = False, dark_number_type: str = "base"):
        self.display_mispredicted = display_mispredicted
        self.calculate_dark_numbers = calculate_dark_numbers
        self.dark_number_type = dark_number_type

    def should_display_mispredicted(self) -> bool:
        return self.display_mispredicted

    def should_calculate_dark_numbers(self) -> bool:
        return self.calculate_dark_numbers

    def get_dark_number_calculation_type(self) -> str:
        return self.dark_number_type

    def get_output_filepath(self, kind: str) -> str:
        return f"{kind}.csv"


class StubLogger:
    def __init__(self):
        self.headers = []
        self.progress = []

    def print_task_header(self, title: str) -> None:
        self.headers.append(title)

    def print_progress(self, message: str = None, percent: float = None) -> None:
        self.progress.append(message)


class StubPredictions:
    def __init__(self):
        self.mispredicted_calls = []
        self.mispredicted_evaluations = []
        self.dark_number_calls = []
        self.dark_number_evaluations = []

    def most_mispredicted(self, *args) -> None:
        self.mispredicted_calls.append(args)

    def evaluate_mispredictions(self, filepath: str) -> None:
        self.mispredicted_evaluations.append(filepath)

    def get_dark_numbers(self, **kwargs) -> None:
        self.dark_number_calls.append(kwargs)

    def evaluate_dark_numbers(self, calculations_filepath: str, confusion_filepath: str) -> None:
        self.dark_number_evaluations.append((calculations_filepath, confusion_filepath))


def make_runner(display_mispredicted: bool, regression_suite: bool, calculate_dark_numbers: bool = False, dark_number_type: str = "base"):
    logger = StubLogger()
    predictions = StubPredictions()
    runner = TaskRunner(
        datalayer=None,
        config=StubConfig(display_mispredicted, calculate_dark_numbers, dark_number_type),
        logger=logger,
        handler=None,
        regression_suite=regression_suite,
    )
    runner.dh = SimpleNamespace(
        X_original="X-original",
        X="X",
        Y="Y",
        X_train="X-train",
        Y_train="Y-train",
        X_validation="X-validation",
        Y_validation="Y-validation",
    )
    runner.ph = predictions
    return runner, logger, predictions


def test_regression_suite_skips_reclassification_but_keeps_dark_numbers():
    runner, logger, predictions = make_runner(display_mispredicted=False, regression_suite=True)

    runner.display_mispredicted__task(cross_trained_model="cross", trained_model="retrained")

    assert predictions.mispredicted_calls == []
    assert predictions.mispredicted_evaluations == []
    assert logger.headers == ["Dark numbers"]
    assert len(predictions.dark_number_calls) == 1
    assert predictions.dark_number_calls[0]["X_cv_training"] == "X-train"
    assert predictions.dark_number_calls[0]["Y_cv_training"] == "Y-train"
    assert predictions.dark_number_calls[0]["X_validation"] == "X-validation"
    assert predictions.dark_number_calls[0]["Y_validation"] == "Y-validation"
    assert predictions.dark_number_calls[0]["type"] == "all"
    assert predictions.dark_number_evaluations == [
        ("dark_numbers.csv", "dark_numb_conf_matrix.csv")
    ]


def test_regular_run_with_mispredicted_disabled_keeps_previous_behavior():
    runner, logger, predictions = make_runner(display_mispredicted=False, regression_suite=False)

    runner.display_mispredicted__task(cross_trained_model="cross", trained_model="retrained")

    assert logger.headers == []
    assert predictions.mispredicted_calls == []
    assert predictions.mispredicted_evaluations == []
    assert predictions.dark_number_calls == []
    assert predictions.dark_number_evaluations == []


class TaskListConfig:
    def get_text_column_names(self):
        return []

    def should_train(self):
        return True

    def should_predict(self):
        return False


def test_regression_suite_omits_per_profile_completion_email_task():
    tasks = get_tasks(TaskListConfig(), regression_suite=True)

    assert "send_completetion_email" not in tasks


def test_regular_run_keeps_completion_email_task():
    tasks = get_tasks(TaskListConfig())

    assert tasks[-1] == "send_completetion_email"


def test_regular_run_can_calculate_dark_numbers_without_displaying_mispredictions():
    runner, logger, predictions = make_runner(
        display_mispredicted=False,
        regression_suite=False,
        calculate_dark_numbers=True,
        dark_number_type="non_linear",
    )

    runner.display_mispredicted__task(cross_trained_model="cross", trained_model="retrained")

    assert predictions.mispredicted_calls == []
    assert logger.headers == ["Dark numbers"]
    assert predictions.dark_number_calls[0]["type"] == "non_linear"
    assert predictions.dark_number_evaluations == [
        ("dark_numbers.csv", "dark_numb_conf_matrix.csv")
    ]


def test_regular_run_can_display_mispredictions_without_calculating_dark_numbers():
    runner, logger, predictions = make_runner(
        display_mispredicted=True,
        regression_suite=False,
        calculate_dark_numbers=False,
    )

    runner.display_mispredicted__task(cross_trained_model="cross", trained_model="retrained")

    assert logger.headers == ["Calculating mispredictions"]
    assert len(predictions.mispredicted_calls) == 1
    assert predictions.mispredicted_evaluations == ["misplaced.csv"]
    assert predictions.dark_number_calls == []
    assert predictions.dark_number_evaluations == []


def test_retrain_task_uses_fresh_retrain_and_persists_it():
    class Config:
        def get_model_filename(self):
            return "model.sav"

    class ModelHandler:
        def __init__(self):
            self.model = SimpleNamespace(pipeline="cross-fitted")
            self.saved = []

        def load_pipeline_from_file(self, filename, init_dh=None):
            return "cross-snapshot"

        def retrain_picked_model(self, pipeline, X, Y):
            assert pipeline == "cross-fitted"
            assert X == "X"
            assert Y == "Y"
            return "fresh-full-data"

        def save_model_to_file(self, filename):
            self.saved.append((filename, self.model.pipeline))

    runner = TaskRunner(
        datalayer=None, config=Config(), logger=StubLogger(), handler=None, regression_suite=False
    )
    runner.dh = SimpleNamespace(X="X", Y="Y")
    runner.mh = ModelHandler()

    result = runner.retrain_model__task()

    assert result == {
        "cross_trained_model": "cross-snapshot",
        "trained_model": "fresh-full-data",
    }
    assert runner.mh.model.pipeline == "fresh-full-data"
    assert runner.mh.saved == [("model.sav", "fresh-full-data")]
