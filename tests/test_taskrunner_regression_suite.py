from types import SimpleNamespace

from JBGTaskRunner import TaskRunner


class StubConfig:
    def __init__(self, display_mispredicted: bool):
        self.display_mispredicted = display_mispredicted

    def should_display_mispredicted(self) -> bool:
        return self.display_mispredicted

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


def make_runner(display_mispredicted: bool, regression_suite: bool):
    logger = StubLogger()
    predictions = StubPredictions()
    runner = TaskRunner(
        datalayer=None,
        config=StubConfig(display_mispredicted),
        logger=logger,
        handler=None,
        regression_suite=regression_suite,
    )
    runner.dh = SimpleNamespace(X_original="X-original", X="X", Y="Y")
    runner.ph = predictions
    return runner, logger, predictions


def test_regression_suite_skips_reclassification_but_keeps_dark_numbers():
    runner, logger, predictions = make_runner(display_mispredicted=False, regression_suite=True)

    runner.display_mispredicted__task(cross_trained_model="cross", trained_model="retrained")

    assert predictions.mispredicted_calls == []
    assert predictions.mispredicted_evaluations == []
    assert logger.headers == ["Dark numbers"]
    assert len(predictions.dark_number_calls) == 1
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
