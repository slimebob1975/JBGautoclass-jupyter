# Alternate GUI for Autoclassification script using widgets from Jupyter
# Written by: Robert Granat, Jan-Feb 2022.
# Broken into module by: Marie Hogebrandt June-oct 2022

import copy
import html
import errno
import json
import os
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

from ipywidgets import Output

src_dir = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(os.getcwd(), ".env")

sys.path.append(src_dir)

from dotenv import load_dotenv

from JBGExceptions import DataLayerException
import JBGTaskRunner
from JBGStreamedLogger import JBGLogger
from AutomaticClassifier import AutomaticClassifier as autoclass
from Config import Config, DarkNumberAlpha, DarkNumberMethod
from JBGMeta import (
    AlgorithmTuple, NgramRange, Oversampling, PreprocessTuple, ReductionTuple,
    ScoreMetric, Undersampling,
)
from JBGSampling import sampling_names
from SQLDataLayer import DataLayer
from GUI.Widgets import Widgets

# Class definition for the GUI
class GUIHandler:
    LAST_RUN_STATE_FILENAME = ".jbg_last_run.json"
    LAST_RUN_STATE_VERSION = 1

    REGRESSION_SUITE_DATASETS = (
        {
            "label": "Iris",
            "slug": "iris",
            "table_name": "iris",
            "class_column": "class",
            "id_column": "id",
        },
        {
            "label": "Breast Cancer",
            "slug": "breast_cancer",
            "table_name": "breast_cancer",
            "class_column": "diagnosis",
            "id_column": "id",
        },
        {
            "label": "Wine",
            "slug": "wine",
            "table_name": "wine",
            "class_column": "class",
            "id_column": "id",
        },
        {
            "label": "Text + Category",
            "slug": "text_category",
            "table_name": "text_regression",
            "class_column": "class",
            "id_column": "id",
            # This fixture is intentionally targeted-only. Running the broad
            # numeric Cartesian profile over TF-IDF output would be both slow
            # and dominated by combinations that are unsuitable for sparse text.
            "run_base": False,
        },
    )

    # Additional targeted runs reuse a resolved dataset but override selected
    # training settings. Keep these narrower than the broad base profile: their
    # purpose is regression coverage of sampling/scoring paths, not another full
    # Cartesian product of every algorithm and transform.
    REGRESSION_SUITE_SAMPLING_ALGORITHMS = (
        "LRN", "RFCL", "LSVC", "GNB", "KNN", "DTC", "SGDE",
    )
    REGRESSION_SUITE_TEXT_ALGORITHMS = (
        "LRN", "LSVC", "SGDE", "MNB", "CNB",
    )
    REGRESSION_SUITE_PROFILES = (
        {
            "label": "Breast Cancer / Random oversampling",
            "slug": "breast_cancer_random_oversampling",
            "dataset_slug": "breast_cancer",
            "oversampler": "RND",
            "undersampler": "NUG",
            "scoring": "f1_macro",
            "algorithms": REGRESSION_SUITE_SAMPLING_ALGORITHMS,
            "preprocessors": ("NOS", "STA"),
            "reductions": ("NOR", "PCA"),
        },
        {
            "label": "Breast Cancer / Random undersampling",
            "slug": "breast_cancer_random_undersampling",
            "dataset_slug": "breast_cancer",
            "oversampler": "NOG",
            "undersampler": "RND",
            "scoring": "f1_macro",
            "algorithms": REGRESSION_SUITE_SAMPLING_ALGORITHMS,
            "preprocessors": ("NOS", "STA"),
            "reductions": ("NOR", "PCA"),
        },
        {
            "label": "Text + Category / Uni-bigrams",
            "slug": "text_category_uni_bigram",
            "dataset_slug": "text_category",
            "oversampler": "NOG",
            "undersampler": "NUG",
            "scoring": "f1_macro",
            "algorithms": REGRESSION_SUITE_TEXT_ALGORITHMS,
            "preprocessors": ("NOS", "MAX"),
            "reductions": ("NOR",),
            "use_stop_words": False,
            "ngram_range": "UNI_BI_GRAM",
            "hex_encode": False,
            "use_categorization": True,
            "category_text_columns": (),
        },
    )

    # Constructor
    def __init__(self):

        # Start persistent logging before constructing the GUI so startup warnings
        # and direct stdout/stderr output are captured as well.
        self.logger = JBGLogger(False) # Quiet is set to false here

        with self.logger.capture_console_output():
            settings = None
            settings_file = os.path.join(os.getcwd(), "settings.json")
            if os.path.isfile(settings_file):
                with open(settings_file) as f:
                    settings = json.load(f)

            self.widgets = Widgets(src_path=Path(src_dir), GUIhandler=self, settings=settings)
            self.logger.set_progress_widgets(self.widgets.progress)

            # This datalayer object is the one working with the classifier data
            self.classifier_datalayer = None

            # This datalayer object only works with the GUI
            self.gui_datalayer = None

            config = Config(
                connection=Config.Connection(
                odbc_driver=os.environ.get("DEFAULT_ODBC_DRIVER"),
                host=os.environ.get("DEFAULT_HOST"),
                class_catalog=os.environ.get("DEFAULT_CLASSIFICATION_CATALOG"),
                class_table=os.environ.get("DEFAULT_CLASSIFICATION_TABLE"),
                data_catalog=os.environ.get("DEFAULT_DATA_CATALOG"),
                data_table=os.environ.get("DEFAULT_DATA_TABLE"),
                sql_username="",
                sql_password="",
                trusted_connection=False
                )
            )
            self.gui_datalayer = DataLayer(config=config, logger=self.logger)

            self.widgets.load_contents()

    @property
    def datalayer(self) -> DataLayer:
        """ This returns the GUI datalayer """
        return self.gui_datalayer

    def set_data_catalog(self, data_catalog: str) -> None:
        """ Updates the GUI datalayer with the data catalog """
        self.gui_datalayer.config.update_connection_columns({
            "data_catalog": data_catalog
            
        })

    def get_class_distribution(self, data_settings: dict, current_class: str) -> dict:
        """ Widgets does not need to know about Classifier Datalayer """
        datalayer = self.get_classifier_datalayer(data_settings=data_settings)
        distribution = {}
        try:
            distribution = datalayer.count_class_distribution()
        except DataLayerException as e:
            self.logger.abort_cleanly(str(e))
        except Exception as e:
            message = f"Could not update summary for class: {current_class} because {e}"
            self.logger.anaconda_debug(message)

        return distribution

    def correct_mispredicted_data(self, new_class: str, index: int) -> None:
        """ Changes the original dataset """
        self.classifier_datalayer.correct_mispredicted_data(new_class, index)

    def get_classifier_datalayer(self, data_settings: dict = None, config_params: dict = None) -> DataLayer:
        """ This will create or update the layer, and should probably also include the start_classifier() stuff
        """
        if config_params:
            # This creates a new classifier_datalayer whether one exists or not, because it has all the info
            self.classifier_datalayer = DataLayer(Config(**config_params), self.logger)

            return self.classifier_datalayer

        if data_settings:
            if not self.classifier_datalayer:
                connection = Config.Connection(
                        odbc_driver = os.environ.get("DEFAULT_ODBC_DRIVER"),
                        host = os.environ.get("DEFAULT_HOST"),
                        class_catalog = os.environ.get("DEFAULT_CLASSIFICATION_CATALOG"),
                        class_table = os.environ.get("DEFAULT_CLASSIFICATION_TABLE"),
                        sql_username = data_settings["connection"]["sql_username"],
                        sql_password = data_settings["connection"]["sql_password"],
                        data_catalog = data_settings["data"]["catalog"],
                        data_table = data_settings["data"]["table"],
                        class_column = data_settings["columns"]["class"], 
                        data_text_columns =  data_settings["columns"]["data_text"], 
                        data_numerical_columns = data_settings["columns"]["data_numerical"], 
                        id_column = data_settings["columns"]["id"]
                    )

                self.classifier_datalayer = DataLayer(Config(connection), self.logger)
            else:
                updated_columns = {
                    "class_column": data_settings["columns"]["class"], 
                    "data_text_columns":  data_settings["columns"]["data_text"], 
                    "data_numerical_columns": data_settings["columns"]["data_numerical"], 
                    "id_column": data_settings["columns"]["id"]
                }
                self.classifier_datalayer.config.update_connection_columns(updated_columns)

        return self.classifier_datalayer        
        
    @property
    def last_run_state_path(self) -> Path:
        """Project-local state used to repeat the most recent manual classifier run."""
        return Path(os.getcwd()) / self.LAST_RUN_STATE_FILENAME

    @staticmethod
    def _config_params_to_last_run_snapshot(config_params: dict) -> dict:
        """Serialize a classifier configuration without persisting SQL credentials."""
        connection = config_params["connection"]
        mode = config_params["mode"]
        io = config_params["io"]
        debug = config_params["debug"]
        mail = config_params["mail"]

        return {
            "version": GUIHandler.LAST_RUN_STATE_VERSION,
            "connection": {
                "odbc_driver": connection.odbc_driver,
                "host": connection.host,
                "trusted_connection": connection.trusted_connection,
                "class_catalog": connection.class_catalog,
                "class_table": connection.class_table,
                "data_catalog": connection.data_catalog,
                "data_table": connection.data_table,
                "class_column": connection.class_column,
                "data_text_columns": list(connection.data_text_columns),
                "data_numerical_columns": list(connection.data_numerical_columns),
                "id_column": connection.id_column,
            },
            "mode": {
                "train": mode.train,
                "predict": mode.predict,
                "mispredicted": mode.mispredicted,
                "calculate_feature_importance": getattr(mode, "calculate_feature_importance", False),
                "feature_importance_repeats": getattr(mode, "feature_importance_repeats", 5),
                "feature_selection_mode": getattr(mode, "feature_selection_mode", "manual"),
                "feature_selection_profile_id": getattr(mode, "feature_selection_profile_id", ""),
                "feature_search_model_type": getattr(mode, "feature_search_model_type", "Linear"),
                "feature_search_scaler": getattr(mode, "feature_search_scaler", "STA"),
                "feature_search_oversampling": getattr(mode, "feature_search_oversampling", "NOG"),
                "feature_search_reduction": getattr(mode, "feature_search_reduction", "NOR"),
                "feature_search_trials": getattr(mode, "feature_search_trials", 4),
                "feature_search_sizes": list(getattr(mode, "feature_search_sizes", [5, 10, 20])),
                "feature_search_inner_folds": getattr(mode, "feature_search_inner_folds", 3),
                "feature_search_outer_folds": getattr(mode, "feature_search_outer_folds", 3),
                "feature_search_top_sets": getattr(mode, "feature_search_top_sets", 1),
                "feature_search_refine_rounds": getattr(mode, "feature_search_refine_rounds", 0),
                "feature_search_max_evaluations": getattr(mode, "feature_search_max_evaluations", 5),
                "feature_search_min_gain": getattr(mode, "feature_search_min_gain", 0.005),
                "feature_search_linear_c": getattr(mode, "feature_search_linear_c", 1.0),
                "feature_search_linear_max_iterations": getattr(mode, "feature_search_linear_max_iterations", 10000),
                "feature_search_non_linear_max_iterations": getattr(mode, "feature_search_non_linear_max_iterations", 2000),
                "feature_search_non_linear_alpha": getattr(mode, "feature_search_non_linear_alpha", 0.0001),
                "feature_search_non_linear_hidden_layers": list(getattr(mode, "feature_search_non_linear_hidden_layers", [100])),
                "feature_search_non_linear_learning_rate": getattr(mode, "feature_search_non_linear_learning_rate", 0.001),
                "feature_search_tree_estimator": getattr(mode, "feature_search_tree_estimator", "Extra Trees"),
                "feature_search_tree_class_weight": getattr(mode, "feature_search_tree_class_weight", "None"),
                "feature_search_tree_positive_threshold": getattr(mode, "feature_search_tree_positive_threshold", 0.5),
                "feature_search_preset": getattr(mode, "feature_search_preset", "Quick"),
                "feature_search_tree_n_estimators": getattr(mode, "feature_search_tree_n_estimators", 100),
                "feature_search_tree_max_depth": getattr(mode, "feature_search_tree_max_depth", 0),
                "feature_search_tree_min_samples_leaf": getattr(mode, "feature_search_tree_min_samples_leaf", 1),
                "use_metas": mode.use_metas,
                "use_stop_words": mode.use_stop_words,
                "ngram_range": mode.ngram_range.name,
                "hex_encode": mode.hex_encode,
                "use_categorization": mode.use_categorization,
                "category_text_columns": list(mode.category_text_columns),
                "test_size": mode.test_size,
                "calculate_dark_numbers": mode.calculate_dark_numbers,
                "dark_number_method": mode.dark_number_method.name,
                "dark_number_alpha": mode.dark_number_alpha.name,
                "dark_number_target": mode.dark_number_target,
                "dark_number_flip_fraction": mode.dark_number_flip_fraction,
                "experimental_perturbed_dark_number_fallback": mode.experimental_perturbed_dark_number_fallback,
                "oversampler": mode.oversampler.name,
                "undersampler": mode.undersampler.name,
                "oversamplers": sampling_names(getattr(mode, "oversamplers", None), Oversampling, mode.oversampler),
                "undersamplers": sampling_names(getattr(mode, "undersamplers", None), Undersampling, mode.undersampler),
                "algorithm": mode.algorithm.get_abbreviations(),
                "model_parameters": getattr(mode, "model_parameters", {}),
                "decision_policy": getattr(mode, "decision_policy", "default"),
                "probability_threshold": getattr(mode, "probability_threshold", 0.5),
                "threshold_diagnostics": getattr(mode, "threshold_diagnostics", False),
                "model_benchmark_150": getattr(mode, "model_benchmark_150", False),
                "holdout_split_mode": getattr(mode, "holdout_split_mode", "stratified_rows"),
                "validation_group_column": getattr(mode, "validation_group_column", "iafId"),
                "grouped_cross_validation": getattr(mode, "grouped_cross_validation", False),
                "excluded_predictor_columns": getattr(mode, "excluded_predictor_columns", ""),
                "preprocessor": mode.preprocessor.get_abbreviations(),
                "feature_selection": mode.feature_selection.get_abbreviations(),
                "num_selected_features": mode.num_selected_features,
                "scoring": mode.scoring.name,
                "max_iterations": mode.max_iterations,
            },
            "io": {
                "verbose": io.verbose,
                "model_path": io.model_path,
                "model_name": io.model_name,
            },
            "debug": {
                "on": debug.on,
                "data_limit": debug.data_limit,
            },
            "mail": {
                "smtp_server": mail.smtp_server,
                "notification_email": mail.notification_email,
            },
            "name": config_params["name"],
            "save": config_params["save"],
        }

    @staticmethod
    def _last_run_snapshot_to_config_params(
        snapshot: dict,
        sql_username: str,
        sql_password: str,
    ) -> dict:
        """Recreate runtime Config objects and inject current SQL credentials."""
        if snapshot.get("version") != GUIHandler.LAST_RUN_STATE_VERSION:
            raise ValueError("Saved previous-run configuration has an unsupported version")

        connection = dict(snapshot["connection"])
        mode = snapshot["mode"]

        return {
            "connection": Config.Connection(
                **connection,
                sql_username=sql_username,
                sql_password=sql_password,
            ),
            "mode": Config.Mode(
                train=mode["train"],
                predict=mode["predict"],
                mispredicted=mode["mispredicted"],
                calculate_feature_importance=mode.get("calculate_feature_importance", False),
                feature_importance_repeats=mode.get("feature_importance_repeats", 5),
                feature_selection_mode=mode.get("feature_selection_mode", "manual"),
                feature_selection_profile_id=mode.get("feature_selection_profile_id", ""),
                feature_search_model_type=mode.get("feature_search_model_type", "Linear"),
                feature_search_scaler=mode.get("feature_search_scaler", "STA"),
                feature_search_oversampling=mode.get("feature_search_oversampling", "NOG"),
                feature_search_reduction=mode.get("feature_search_reduction", "NOR"),
                feature_search_trials=mode.get("feature_search_trials", 4),
                feature_search_sizes=list(mode.get("feature_search_sizes", [5, 10, 20])),
                feature_search_inner_folds=mode.get("feature_search_inner_folds", 3),
                feature_search_outer_folds=mode.get("feature_search_outer_folds", 3),
                feature_search_top_sets=mode.get("feature_search_top_sets", 1),
                feature_search_refine_rounds=mode.get("feature_search_refine_rounds", 0),
                feature_search_max_evaluations=mode.get("feature_search_max_evaluations", 5),
                feature_search_min_gain=mode.get("feature_search_min_gain", 0.005),
                feature_search_linear_c=mode.get("feature_search_linear_c", 1.0),
                feature_search_linear_max_iterations=mode.get("feature_search_linear_max_iterations", 10000),
                feature_search_non_linear_max_iterations=mode.get("feature_search_non_linear_max_iterations", 2000),
                feature_search_non_linear_alpha=mode.get("feature_search_non_linear_alpha", 0.0001),
                feature_search_non_linear_hidden_layers=list(mode.get("feature_search_non_linear_hidden_layers", [100])),
                feature_search_non_linear_learning_rate=mode.get("feature_search_non_linear_learning_rate", 0.001),
                feature_search_preset=mode.get("feature_search_preset", "Quick"),
                feature_search_tree_estimator=mode.get("feature_search_tree_estimator", "Extra Trees"),
                feature_search_tree_class_weight=mode.get("feature_search_tree_class_weight", "None"),
                feature_search_tree_positive_threshold=mode.get("feature_search_tree_positive_threshold", 0.5),
                feature_search_tree_n_estimators=mode.get("feature_search_tree_n_estimators", 100),
                feature_search_tree_max_depth=mode.get("feature_search_tree_max_depth", 0),
                feature_search_tree_min_samples_leaf=mode.get("feature_search_tree_min_samples_leaf", 1),
                use_metas=mode["use_metas"],
                use_stop_words=mode["use_stop_words"],
                ngram_range=NgramRange[mode["ngram_range"]],
                hex_encode=mode["hex_encode"],
                use_categorization=mode["use_categorization"],
                category_text_columns=list(mode["category_text_columns"]),
                test_size=mode["test_size"],
                calculate_dark_numbers=mode.get("calculate_dark_numbers", mode["mispredicted"]),
                dark_number_method=DarkNumberMethod.from_config_value(mode.get("dark_number_method", "LINEAR")),
                dark_number_alpha=DarkNumberAlpha.from_config_value(mode.get("dark_number_alpha", "NONE")),
                dark_number_target=str(mode.get("dark_number_target", "") or ""),
                dark_number_flip_fraction=mode.get("dark_number_flip_fraction", 0.2),
                experimental_perturbed_dark_number_fallback=mode.get(
                    "experimental_perturbed_dark_number_fallback", True
                ),
                oversampler=Oversampling[mode["oversampler"]],
                undersampler=Undersampling[mode["undersampler"]],
                oversamplers=mode.get("oversamplers"),
                undersamplers=mode.get("undersamplers"),
                algorithm=AlgorithmTuple(mode["algorithm"]),
                model_parameters=mode.get("model_parameters", {}),
                decision_policy=mode.get("decision_policy", "default"),
                probability_threshold=float(mode.get("probability_threshold", 0.5)),
                threshold_diagnostics=bool(mode.get("threshold_diagnostics", False)),
                model_benchmark_150=bool(mode.get("model_benchmark_150", False)),
                holdout_split_mode=mode.get("holdout_split_mode", "stratified_rows"),
                validation_group_column=mode.get("validation_group_column", "iafId"),
                grouped_cross_validation=bool(mode.get("grouped_cross_validation", False)),
                excluded_predictor_columns=mode.get("excluded_predictor_columns", ""),
                preprocessor=PreprocessTuple(mode["preprocessor"]),
                feature_selection=ReductionTuple(mode["feature_selection"]),
                num_selected_features=mode["num_selected_features"],
                scoring=ScoreMetric[mode["scoring"]],
                max_iterations=mode["max_iterations"],
            ),
            "io": Config.IO(**snapshot["io"]),
            "debug": Config.Debug(**snapshot["debug"]),
            "mail": Config.Mail(**snapshot["mail"]),
            "name": snapshot["name"],
            "save": snapshot["save"],
        }

    def save_last_classifier_run(self, config_params: dict) -> None:
        """Persist the last manual run atomically; SQL credentials are deliberately omitted."""
        snapshot = self._config_params_to_last_run_snapshot(config_params)
        path = self.last_run_state_path
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=2)
        temporary_path.replace(path)

    def load_last_classifier_run(self, sql_username: str, sql_password: str) -> dict:
        """Load the last manual run and combine it with the current login credentials."""
        with self.last_run_state_path.open(encoding="utf-8") as handle:
            snapshot = json.load(handle)
        return self._last_run_snapshot_to_config_params(snapshot, sql_username, sql_password)

    def has_saved_classifier_run(self) -> bool:
        """Return True only when a readable, supported previous-run snapshot exists."""
        try:
            with self.last_run_state_path.open(encoding="utf-8") as handle:
                snapshot = json.load(handle)
            return snapshot.get("version") == self.LAST_RUN_STATE_VERSION
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            return False

    def repeat_last_classifier_run(
        self,
        output: Output,
        sql_username: str,
        sql_password: str,
    ):
        """Rerun the persisted manual classifier configuration with current credentials."""
        config_params = self.load_last_classifier_run(sql_username, sql_password)
        self.widgets.restore_classifier_config(config_params)
        connection = config_params["connection"]
        with output, self.logger.capture_console_output():
            self.logger.print_info(
                "Repeat previous run: using saved classifier settings for "
                f"{connection.data_catalog}.{connection.data_table}; data will be fetched again."
            )
        return self.run_classifier(config_params=config_params, output=output)

    def run_feature_selection_investigation(self, config_params: dict, options, on_event=None, row_limit=None, output=None, resume=False, pause_event=None, investigation_mode="selection_plus_diagnostics", subgroup_column=None):
        """Run the GUI-assisted feature investigation without touching classifier state."""
        from JBGFeatureSearchService import run_training_feature_search

        investigation_params = copy.deepcopy(config_params)
        investigation_params["save"] = False
        if row_limit is None or int(row_limit) < 1:
            raise ValueError("Choose at least one investigation row before starting")
        investigation_params["debug"].data_limit = int(row_limit)
        investigation_params["mode"].train = True
        investigation_params["mode"].predict = False
        investigation_params["mode"].mispredicted = False
        investigation_params["mode"].calculate_dark_numbers = False
        investigation_params["mode"].calculate_feature_importance = False
        investigation_config = Config(**investigation_params)
        return run_training_feature_search(
            investigation_config,
            self.logger,
            options=options,
            on_event=on_event,
            output=output,
            resume=resume,
            pause_event=pause_event,
            investigation_mode=investigation_mode, subgroup_column=subgroup_column,
        )

    def resume_saved_feature_selection_investigation(self, output, on_event=None, pause_event=None):
        """Resume using the immutable snapshot, without constructing GUI Config."""
        from JBGFeatureSearchService import run_training_feature_search
        return run_training_feature_search(
            None, self.logger, options=None, on_event=on_event,
            output=output, resume=True, pause_event=pause_event,
        )

    def _classifier_diagnostic(self, message: str, level: str = "INFO") -> None:
        """Write startup diagnostics to disk, independently of the GUI output widget."""
        self.logger.log_file.write_message(f"[classifier-init] {message}", level=level)

    def run_classifier(
        self,
        config_params: dict,
        output: Output,
        set_rerun: bool = True,
        regression_suite: bool = False,
    ):
        """Run classification, with file-based diagnostic boundaries for startup."""
        stage = "callback entry"
        started = time.monotonic()
        self._classifier_diagnostic("callback entered")
        try:
            if not regression_suite:
                stage = "persist last run"
                self._classifier_diagnostic("persist last run: begin")
                try:
                    self.save_last_classifier_run(config_params)
                except Exception as ex:
                    self._classifier_diagnostic(
                        f"persist last run: failed ({type(ex).__name__}: {ex})", "WARNING"
                    )
                    with output, self.logger.capture_console_output():
                        self.logger.print_warning(
                            f"Could not save previous-run configuration: {type(ex).__name__}: {ex}"
                        )
                self._classifier_diagnostic("persist last run: end")

            stage = "construct configuration and classifier data layer"
            self._classifier_diagnostic("data layer construction: begin")
            self.get_classifier_datalayer(config_params=config_params)
            self._classifier_diagnostic("data layer construction: end")

            stage = "set output verbosity"
            self.logger.set_enable_quiet(not config_params["io"].verbose)
            result = {"mispredicted": None}
            stage = "construct classifier"
            self._classifier_diagnostic("classifier construction: begin")
            with output, self.logger.capture_console_output():
                the_classifier = autoclass(
                    config=self.classifier_datalayer.get_config(),
                    logger=self.logger,
                    datalayer=self.classifier_datalayer,
                    regression_suite=regression_suite,
                )
                self._classifier_diagnostic("classifier construction: end")
                stage = "classifier.run (includes database fetch)"
                self._classifier_diagnostic("classifier.run: begin (database fetch follows in task runner)")
                result = the_classifier.run()
                self._classifier_diagnostic("classifier.run: end")
                if not result:
                    self.logger.print_info("No data was fetched from database!")

            stage = "post-run GUI"
            if result and result["mispredicted"] is not None:
                try:
                    self.widgets.handle_mispredicted(**result)
                except Exception as exc:
                    with output, self.logger.capture_console_output():
                        self.logger.print_warning(
                            f"Reclassification widget unavailable; model run completed: "
                            f"{type(exc).__name__}: {exc}"
                        )
            if set_rerun:
                self.widgets.set_rerun()
            self._classifier_diagnostic(f"callback completed; elapsed={time.monotonic()-started:.1f}s")
            return result
        except Exception:
            # Preserve the original traceback; do not log configuration values/credentials.
            self._classifier_diagnostic(
                f"failure at stage={stage}; elapsed={time.monotonic()-started:.1f}s\n"
                + traceback.format_exc(), "ERROR"
            )
            raise

    @classmethod
    def _regression_suite_dataset_labels(cls) -> str:
        return ", ".join(dataset["label"] for dataset in cls.REGRESSION_SUITE_DATASETS)

    @staticmethod
    def _table_basename(table: str) -> str:
        return str(table).rsplit(".", 1)[-1].casefold()

    def _regression_suite_catalogs(self, preferred_catalog: str) -> list[str]:
        catalogs = [catalog for catalog in self.datalayer.get_catalogs_as_options() if catalog]
        if preferred_catalog in catalogs:
            catalogs.remove(preferred_catalog)
            catalogs.insert(0, preferred_catalog)
        return catalogs

    def _find_regression_suite_dataset(self, dataset: dict, preferred_catalog: str) -> dict | None:
        original_catalog = self.datalayer.config.connection.data_catalog
        try:
            for catalog in self._regression_suite_catalogs(preferred_catalog):
                try:
                    self.set_data_catalog(catalog)
                    tables = [table for table in self.datalayer.get_tables_as_options() if table]
                except Exception as ex:
                    self.logger.print_warning(
                        f"Regression suite: could not inspect catalog {catalog}: {type(ex).__name__}: {ex}"
                    )
                    continue

                matching_tables = [
                    table for table in tables
                    if self._table_basename(table) == dataset["table_name"].casefold()
                ]
                for table in matching_tables:
                    try:
                        columns = self.datalayer.get_table_columns(catalog, table)
                        row_count = self.datalayer.count_data_rows(catalog, table)
                    except Exception as ex:
                        self.logger.print_warning(
                            f"Regression suite: could not inspect {catalog}.{table}: "
                            f"{type(ex).__name__}: {ex}"
                        )
                        continue

                    required = (dataset["class_column"], dataset["id_column"])
                    missing = [column for column in required if column not in columns]
                    if missing:
                        self.logger.print_warning(
                            f"Regression suite: skipping {dataset['label']} candidate {catalog}.{table}; "
                            f"missing required columns: {', '.join(missing)}."
                        )
                        continue

                    feature_columns = [
                        column for column in columns
                        if column not in (dataset["class_column"], dataset["id_column"])
                    ]
                    text_columns = [
                        column for column in feature_columns
                        if str(columns[column]).casefold() in Config.TEXT_DATATYPES
                    ]
                    numerical_columns = [
                        column for column in feature_columns if column not in text_columns
                    ]

                    if not feature_columns:
                        self.logger.print_warning(
                            f"Regression suite: skipping {dataset['label']} candidate {catalog}.{table}; "
                            "no feature columns were found."
                        )
                        continue

                    return {
                        **dataset,
                        "catalog": catalog,
                        "table": table,
                        "text_columns": text_columns,
                        "numerical_columns": numerical_columns,
                        "row_count": row_count,
                    }
        finally:
            self.set_data_catalog(original_catalog)

        return None

    @staticmethod
    def _apply_regression_suite_profile(config_params: dict, profile: dict) -> None:
        """Apply a targeted suite-only training profile to a copied config."""
        mode = config_params["mode"]
        mode.oversampler = Oversampling[profile["oversampler"]]
        mode.undersampler = Undersampling[profile["undersampler"]]
        mode.oversamplers = None
        mode.undersamplers = None
        mode.scoring = ScoreMetric[profile["scoring"]]
        mode.algorithm = AlgorithmTuple(profile["algorithms"])
        mode.preprocessor = PreprocessTuple(profile["preprocessors"])
        mode.feature_selection = ReductionTuple(profile["reductions"])

        # Text settings are optional so the established numeric sampling
        # profiles keep inheriting the broad suite defaults unchanged.
        if "use_stop_words" in profile:
            mode.use_stop_words = profile["use_stop_words"]
        if "ngram_range" in profile:
            mode.ngram_range = NgramRange[profile["ngram_range"]]
        if "hex_encode" in profile:
            mode.hex_encode = profile["hex_encode"]
        if "use_categorization" in profile:
            mode.use_categorization = profile["use_categorization"]
        if "category_text_columns" in profile:
            mode.category_text_columns = list(profile["category_text_columns"])

    def _build_regression_suite_config(
        self,
        base_config_params: dict,
        dataset: dict,
        profile: dict | None = None,
    ) -> dict:
        config_params = copy.deepcopy(base_config_params)
        connection = config_params["connection"]
        connection.data_catalog = dataset["catalog"]
        connection.data_table = dataset["table"]
        connection.class_column = dataset["class_column"]
        connection.id_column = dataset["id_column"]
        connection.data_text_columns = list(dataset["text_columns"])
        connection.data_numerical_columns = list(dataset["numerical_columns"])

        # Suite profiles stay bounded even when the manual GUI compares samplers.
        config_params["mode"].oversamplers = None
        config_params["mode"].undersamplers = None
        config_params["mode"].category_text_columns = []
        config_params["debug"].data_limit = dataset["row_count"]

        run_slug = dataset["slug"]
        if profile is not None:
            self._apply_regression_suite_profile(config_params, profile)
            run_slug = profile["slug"]

        base_model_name = config_params["io"].model_name or "regression"
        config_params["io"].model_name = f"{base_model_name}_{run_slug}"
        config_params["name"] = f"{config_params['name']}_{run_slug}"
        return config_params

    def get_regression_suite_configs(self, base_config_params: dict) -> tuple[list[tuple[str, dict]], list[str]]:
        preferred_catalog = base_config_params["connection"].data_catalog
        configs = []
        missing = []
        resolved_by_slug = {}

        # Keep the established broad dataset runs first. Targeted profiles are
        # appended afterwards so their addition does not change base-run order.
        for dataset in self.REGRESSION_SUITE_DATASETS:
            resolved = self._find_regression_suite_dataset(dataset, preferred_catalog)
            if resolved is None:
                missing.append(dataset["label"])
                continue

            resolved_by_slug[dataset["slug"]] = resolved
            if dataset.get("run_base", True):
                configs.append((dataset["label"], self._build_regression_suite_config(base_config_params, resolved)))

        for profile in self.REGRESSION_SUITE_PROFILES:
            resolved = resolved_by_slug.get(profile["dataset_slug"])
            if resolved is None:
                continue
            configs.append((
                profile["label"],
                self._build_regression_suite_config(base_config_params, resolved, profile=profile),
            ))

        return configs, missing

    def run_regression_suite(self, base_config_params: dict, output: Output) -> None:
        """Run known regression datasets sequentially and isolate dataset-level failures."""
        suite_started = time.monotonic()
        dataset_labels = self._regression_suite_dataset_labels()
        with output, self.logger.capture_console_output():
            self.logger.print_info(f"Regression suite: resolving configured datasets: {dataset_labels}.")
            if self.REGRESSION_SUITE_PROFILES:
                profile_labels = ", ".join(profile["label"] for profile in self.REGRESSION_SUITE_PROFILES)
                self.logger.print_info(f"Regression suite: targeted profiles: {profile_labels}.")
            self.logger.print_info(
                "Regression suite: reclassification/mispredicted output is disabled for suite runs."
            )
            configs, missing = self.get_regression_suite_configs(base_config_params)
            for label in missing:
                self.logger.print_warning(
                    f"Regression suite: {label} was not found in the accessible SQL catalogs and will be skipped."
                )

        completed = []
        failed = []
        statuses = []
        for index, (label, config_params) in enumerate(configs, start=1):
            connection = config_params["connection"]
            with output, self.logger.capture_console_output():
                self.logger.print_info(
                    f"Regression suite [{index}/{len(configs)}]: starting {label} "
                    f"from {connection.data_catalog}.{connection.data_table}."
                )

            try:
                result = self.run_classifier(
                    config_params=config_params,
                    output=output,
                    set_rerun=False,
                    regression_suite=True,
                )
                if result:
                    completed.append(label)
                    statuses.append((label, "COMPLETED"))
                    with output, self.logger.capture_console_output():
                        self.logger.print_info(f"Regression suite: completed {label}.")
                else:
                    failed.append(label)
                    statuses.append((label, "FAILED - no result"))
                    with output, self.logger.capture_console_output():
                        self.logger.print_warning(f"Regression suite: {label} returned no result.")
            except SystemExit as ex:
                failed.append(label)
                details = self._compact_regression_suite_failure(ex)
                statuses.append((label, f"FAILED - aborted: {details}"))
                with output, self.logger.capture_console_output():
                    self.logger.print_error(f"Regression suite: {label} aborted: {ex}")
            except Exception as ex:
                failed.append(label)
                details = self._compact_regression_suite_failure(ex)
                statuses.append((label, f"FAILED - {type(ex).__name__}: {details}"))
                with output, self.logger.capture_console_output():
                    self.logger.print_error(f"Regression suite: {label} failed: {type(ex).__name__}: {ex}")

        statuses.extend((label, "MISSING") for label in missing)
        elapsed = timedelta(seconds=round(time.monotonic() - suite_started))
        summary = (
            "Regression suite summary: "
            f"{len(completed)} completed, {len(failed)} failed, {len(missing)} missing; "
            f"total elapsed time {elapsed}."
        )
        with output, self.logger.capture_console_output():
            self.logger.print_info(summary)
            for label, status in statuses:
                self.logger.print_info(f"Regression suite result: {status} - {label}.")
            self.logger.print_progress(
                message=f"Regression suite finished after {elapsed}",
                percent=1.0,
            )

        self._send_regression_suite_email(base_config_params["mail"], statuses, elapsed)
        self.widgets.set_rerun()

    @staticmethod
    def _compact_regression_suite_failure(ex: BaseException, max_length: int = 160) -> str:
        """Keep suite-summary/email failure details informative without copying full trace payloads."""
        details = " ".join(str(ex).split())
        if len(details) <= max_length:
            return details
        return details[:max_length - 3] + "..."

    def _send_regression_suite_email(self, mail_config, statuses: list[tuple[str, str]], elapsed: timedelta) -> None:
        """Send one compact completion email after all suite profiles have finished."""
        text_rows = "\n".join(f"{status}: {label}" for label, status in statuses)
        html_rows = "".join(
            f"<tr><td>{html.escape(label)}</td><td>{html.escape(status)}</td></tr>"
            for label, status in statuses
        )
        subject = "JBG Regression Suite has completed"
        text_body = (
            f"The JBG Regression Suite has completed after {elapsed}.\n\n"
            f"{text_rows}\n\nSincerely yours,\nJBG\n"
        )
        html_body = f"""\
            <!doctype html>
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
                table {{ border-collapse: collapse; border: 1px solid #ccc; }}
                th, td {{ border: 1px solid #ccc; padding: 4px 6px; text-align: left; }}
                </style>
            </head>
            <body>
                <p>The JBG Regression Suite has completed after {elapsed}.</p>
                <table>
                    <thead><tr><th>Profile</th><th>Status</th></tr></thead>
                    <tbody>{html_rows}</tbody>
                </table>
                <p>Sincerely yours,<br>JBG</p>
            </body>
            </html>"""
        JBGTaskRunner.send_email(mail_config, self.logger, subject, text_body, html_body)

    def display_gui(self) -> None:
        self.widgets.display_gui()
        
        

def main():
    load_dotenv()
    gui = GUIHandler()
    gui.display_gui()


if __name__ == "__main__":
    main()
else:
    if os.path.isfile(env_path):
        load_dotenv(env_path)  # take environment variables from .env
    else:
        raise FileNotFoundError(
            errno.ENOENT, 
            os.strerror(errno.ENOENT), 
            env_path
        )
