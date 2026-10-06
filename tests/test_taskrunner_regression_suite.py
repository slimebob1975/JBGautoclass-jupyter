from types import SimpleNamespace

from JBGTaskRunner import TaskRunner, get_tasks, send_email


class StubConfig:
    def __init__(self, display_mispredicted: bool, calculate_dark_numbers: bool = False, dark_number_type: str = "base", dark_number_target: str = ""):
        self.display_mispredicted = display_mispredicted
        self.calculate_dark_numbers = calculate_dark_numbers
        self.dark_number_type = dark_number_type
        self.dark_number_target = dark_number_target

    def should_display_mispredicted(self) -> bool:
        return self.display_mispredicted

    def should_calculate_dark_numbers(self) -> bool:
        return self.calculate_dark_numbers

    def get_dark_number_calculation_type(self) -> str:
        return self.dark_number_type

    def get_dark_number_target(self) -> str:
        return self.dark_number_target

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


def make_runner(display_mispredicted: bool, regression_suite: bool, calculate_dark_numbers: bool = False, dark_number_type: str = "base", dark_number_target: str = ""):
    logger = StubLogger()
    predictions = StubPredictions()
    runner = TaskRunner(
        datalayer=None,
        config=StubConfig(display_mispredicted, calculate_dark_numbers, dark_number_type, dark_number_target),
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
    assert predictions.dark_number_calls[0]["target_class"] is None
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


class MailLogger:
    def __init__(self):
        self.info = []

    def print_info(self, *args):
        self.info.append(" ".join(str(arg) for arg in args))


def test_completion_email_keeps_missing_recipient_as_non_error_state(monkeypatch):
    def smtp_must_not_be_called(*args, **kwargs):
        raise AssertionError("SMTP should not be opened without a configured recipient")

    monkeypatch.setattr("JBGTaskRunner.smtplib.SMTP", smtp_must_not_be_called)
    logger = MailLogger()
    mail = SimpleNamespace(smtp_server="smtp.example", notification_email="")

    sent = send_email(mail, logger, "subject", "text", "<p>html</p>")

    assert sent is False
    assert logger.info == [
        "Completion email skipped: no notification recipient is configured. "
        "Set DEFAULT_NOTIFICATION_EMAIL to enable completion notifications."
    ]


def test_completion_email_handles_none_recipient_without_exception(monkeypatch):
    def smtp_must_not_be_called(*args, **kwargs):
        raise AssertionError("SMTP should not be opened without a configured recipient")

    monkeypatch.setattr("JBGTaskRunner.smtplib.SMTP", smtp_must_not_be_called)
    logger = MailLogger()
    mail = SimpleNamespace(smtp_server="smtp.example", notification_email=None)

    sent = send_email(mail, logger, "subject", "text", "<p>html</p>")

    assert sent is False
    assert "no notification recipient is configured" in logger.info[0]


def test_completion_email_reports_invalid_recipient_as_configuration_state(monkeypatch):
    def smtp_must_not_be_called(*args, **kwargs):
        raise AssertionError("SMTP should not be opened with an invalid recipient")

    monkeypatch.setattr("JBGTaskRunner.smtplib.SMTP", smtp_must_not_be_called)
    logger = MailLogger()
    mail = SimpleNamespace(smtp_server="smtp.example", notification_email="not-an-email")

    sent = send_email(mail, logger, "subject", "text", "<p>html</p>")

    assert sent is False
    assert logger.info == [
        "Completion email skipped: the configured notification recipient is invalid. "
        "Check DEFAULT_NOTIFICATION_EMAIL."
    ]


def test_completion_email_reports_missing_smtp_as_configuration_state(monkeypatch):
    def smtp_must_not_be_called(*args, **kwargs):
        raise AssertionError("SMTP should not be opened without a configured server")

    monkeypatch.setattr("JBGTaskRunner.smtplib.SMTP", smtp_must_not_be_called)
    logger = MailLogger()
    mail = SimpleNamespace(smtp_server="", notification_email="runtime@example.test")

    sent = send_email(mail, logger, "subject", "text", "<p>html</p>")

    assert sent is False
    assert logger.info == [
        "Completion email skipped: no SMTP server is configured. "
        "Set DEFAULT_SMTP_SERVER to enable completion notifications."
    ]


def test_regular_run_can_calculate_dark_numbers_without_displaying_mispredictions():
    runner, logger, predictions = make_runner(
        display_mispredicted=False,
        regression_suite=False,
        calculate_dark_numbers=True,
        dark_number_type="non_linear",
        dark_number_target="positive",
    )

    runner.display_mispredicted__task(cross_trained_model="cross", trained_model="retrained")

    assert predictions.mispredicted_calls == []
    assert logger.headers == ["Dark numbers"]
    assert predictions.dark_number_calls[0]["type"] == "non_linear"
    assert predictions.dark_number_calls[0]["target_class"] == "positive"
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
