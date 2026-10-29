"""CSV file inspection widgets; no classifier source switching."""

import csv
import tempfile
from pathlib import Path

import ipywidgets as widgets
from GUI.ButtonStates import set_action_state


class CSVSourcePanel:
    """Inspect CSV structure only; roles belong to Dataset columns."""

    def __init__(self):
        # Browsers cannot disclose the user's original local path. Upload the
        # selected CSV to a temporary server-side path for inspection only.
        self._upload_directory = tempfile.TemporaryDirectory(prefix="jbg_csv_preview_")
        self.file_picker = widgets.FileUpload(accept=".csv", multiple=False, description="Choose CSV")
        self.file_picker.observe(self._file_selected, names="value")
        self.path = widgets.Text(description="Internal CSV path:", disabled=True)
        self.selected_file = widgets.Text(description="Selected file:", placeholder="No file selected",
                                          disabled=True, layout=widgets.Layout(width="65%"))
        self.delimiter = widgets.Text(value=",", description="Delimiter:")
        self.quotechar = widgets.Text(value='"', description="Text qualifier:")
        self.encoding = widgets.Text(value="utf-8-sig", description="Encoding:")
        self.register_button = widgets.Button(description="Register dataset", disabled=True)
        self.saved_datasets = widgets.Dropdown(description="Saved CSV:", options=(("No saved datasets", ""),), value="")
        self.refresh_saved_button = widgets.Button(description="Refresh saved")
        self.load_saved_button = widgets.Button(description="Load saved", disabled=True)
        self.erase_saved_button = widgets.Button(description="Erase saved", disabled=True, button_style="danger")
        self._erase_confirmation = ""
        set_action_state(self.register_button, "blocked")
        set_action_state(self.load_saved_button, "blocked")
        set_action_state(self.erase_saved_button, "blocked")
        self.loaded_saved_artifact_id = None
        self.saved_datasets.observe(self._saved_selection_changed, names="value")
        self.saved_prediction_runs = widgets.Dropdown(description="Predictions:",
            options=(("No saved runs", ""),), value="")
        self.refresh_predictions_button = widgets.Button(description="Refresh runs")
        self.export_predictions_button = widgets.Button(description="Export CSV", disabled=True)
        self.export_joined_button = widgets.Button(description="Export joined", disabled=True)
        self.erase_predictions_button = widgets.Button(description="Erase predictions", disabled=True)
        self._erase_predictions_confirmation = ""
        set_action_state(self.erase_predictions_button, "blocked")
        self.prediction_download = widgets.HTML(value="")
        self.saved_prediction_runs.observe(self._prediction_run_changed, names="value")
        self.status = widgets.HTML(value="CSV inspection only. SQL Server remains the active execution source.")
        for control in (self.delimiter, self.quotechar, self.encoding):
            control.observe(self._format_changed, names="value")
        self.widget = widgets.VBox([
            widgets.HBox([self.file_picker, self.selected_file]), widgets.HBox([self.delimiter, self.quotechar, self.encoding]),
            widgets.HBox([self.register_button]),
            widgets.HBox([self.saved_datasets, self.refresh_saved_button, self.load_saved_button, self.erase_saved_button]),
            widgets.HBox([self.saved_prediction_runs, self.refresh_predictions_button,
                          self.export_predictions_button, self.export_joined_button,
                          self.erase_predictions_button]),
            self.prediction_download, self.status,
        ])

    def _prediction_run_changed(self, change):
        enabled = bool(change.get("new"))
        self.export_predictions_button.disabled = not enabled
        self.export_joined_button.disabled = not enabled
        self._erase_predictions_confirmation = ""
        self.erase_predictions_button.description = "Erase predictions"
        set_action_state(self.erase_predictions_button, "warning" if enabled else "blocked")
        self.prediction_download.value = ""

    def _saved_selection_changed(self, change):
        set_action_state(self.load_saved_button, "ready" if change.get("new") else "blocked")
        set_action_state(self.erase_saved_button, "warning" if change.get("new") else "blocked")
        self._erase_confirmation = ""
        self.erase_saved_button.description = "Erase saved"
        self.saved_prediction_runs.options = (("Select prediction run", ""),)
        self._erase_predictions_confirmation = ""
        self.erase_predictions_button.description = "Erase predictions"
        set_action_state(self.erase_predictions_button, "blocked")
        self.prediction_download.value = ""

    def stage_saved_csv(self, content: bytes, display_name="Registered CSV", artifact_id=None):
        """Stage already-authorized data without exposing its server path."""
        callback = getattr(self, "on_file_changed", None)
        if callback is not None:
            callback()
        self.path.value = ""
        self.selected_file.value = ""
        self.register_button.disabled = True
        target = Path(self._upload_directory.name) / "saved_dataset.csv"
        target.write_bytes(content)
        self.path.value = str(target)
        self.selected_file.value = display_name
        self.loaded_saved_artifact_id = artifact_id
        detected = self._detect_dialect(target)
        if detected:
            self._setting_detected_dialect = True
            try:
                self.delimiter.value, self.quotechar.value = detected
            finally:
                self._setting_detected_dialect = False
        self.validate()

    def _file_selected(self, change):
        """Stage a browser-selected CSV; never trust its client-side filename as a path."""
        callback = getattr(self, "on_file_changed", None)
        if callback is not None:
            callback()
        self.path.value = ""
        self.loaded_saved_artifact_id = None
        self.register_button.disabled = True
        self.selected_file.value = ""
        self.status.value = "CSV selection failed. Choose a .csv file."
        try:
            value = change.get("new")
            if not value:
                return
            # ipywidgets 8: tuple of file dictionaries; ipywidgets 7: mapping.
            if isinstance(value, dict):
                if len(value) != 1:
                    return
                name, entry = next(iter(value.items()))
                entry = dict(entry)
                name = entry.get("name", name)
            else:
                if len(value) != 1:
                    return
                entry = value[0]
                name = entry.get("name", "")
            if not isinstance(name, str) or Path(name).suffix.lower() != ".csv":
                return
            content = entry.get("content")
            if content is None:
                return
            data = bytes(content)
            if len(data) > 25 * 1024 * 1024 or not data:
                self.status.value = "CSV selection failed. File must be 1–25 MB or smaller."
                return
            target = Path(self._upload_directory.name) / "dataset.csv"
            target.write_bytes(data)
            self.path.value = str(target)
            self.selected_file.value = Path(name).name
            detected = self._detect_dialect(target)
            if detected:
                self._setting_detected_dialect = True
                try:
                    self.delimiter.value, self.quotechar.value = detected
                finally:
                    self._setting_detected_dialect = False
            self.validate()

        except (OSError, TypeError, ValueError, KeyError, AttributeError):
            self.status.value = "CSV selection failed. Choose a .csv file."

    def _format_changed(self, change):
        """Revalidate a staged file when its parsing settings change."""
        if self.path.value and not getattr(self, "_setting_detected_dialect", False):
            callback = getattr(self, "on_file_changed", None)
            if callback is not None:
                callback()
            self.validate()

    def _detect_dialect(self, path):
        """Conservative bounded dialect detection; never publish file contents/errors."""
        try:
            with path.open("r", encoding=self.encoding.value, newline="") as stream:
                sample = stream.read(65536)
            if not sample.strip():
                return None
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            delimiter = dialect.delimiter
            qualifier = dialect.quotechar or '"'
            if delimiter not in (",", ";", "\t", "|") or len(qualifier) != 1:
                return None
            # Reject single-column guesses and inconsistent sample records.
            rows = list(csv.reader(sample.splitlines(keepends=True), delimiter=delimiter, quotechar=qualifier, strict=True))
            if len(rows) < 2 or len(rows[0]) < 2 or any(len(row) != len(rows[0]) for row in rows[:-1]):
                return None
            return delimiter, qualifier
        except (OSError, UnicodeError, ValueError, csv.Error, LookupError, TypeError):
            return None

    def validate(self, _button=None):
        """Validate row shape without assigning dataset column roles or leaking data."""
        self.register_button.disabled = True
        self.status.value = "CSV validation failed. Check file and format settings."
        try:
            path = Path(self.path.value)
            delimiter = self.delimiter.value
            encoding = self.encoding.value
            quotechar = self.quotechar.value
            if not str(self.path.value).strip() or not path.is_file():
                return False
            if not isinstance(delimiter, str) or len(delimiter) != 1 or delimiter in "\r\n\x00":
                return False
            if not isinstance(encoding, str) or not encoding.strip():
                return False
            if not isinstance(quotechar, str) or len(quotechar) != 1 or quotechar in "\r\n\x00" or quotechar == delimiter:
                return False
            with path.open("r", encoding=encoding, newline="") as stream:
                reader = csv.reader(stream, delimiter=delimiter, quotechar=quotechar, strict=True)
                header = next(reader)
                if not header or any(not name or not name.strip() for name in header):
                    return False
                if len(header) != len(set(header)):
                    return False
                count = 0
                for row in reader:
                    if len(row) != len(header):
                        return False
                    count += 1
                if count == 0:
                    return False
        except (OSError, UnicodeError, ValueError, csv.Error, LookupError, StopIteration):
            return False
        self.status.value = (
            f"CSV validated: {count} rows and {len(header)} columns. "
            "Select dataset column roles and register the dataset before continuing."
        )
        callback = getattr(self, "on_validated", None)
        if callback is not None:
            callback(self)
        return True


class DataSourceSelector:
    """Mutually exclusive source settings views; classifier remains SQL-only."""

    def __init__(self, sql_widget, csv_widget):
        self.choice = widgets.ToggleButtons(
            options=[("SQL Server", "sql"), ("CSV", "csv")],
            value="sql",
            style={"description_width": "initial"},
        )
        self.sql_widget = sql_widget
        self.csv_widget = csv_widget
        self.body = widgets.VBox([sql_widget])
        self.choice.observe(self._source_changed, names="value")
        self.widget = widgets.VBox([self.choice, self.body])

    def _source_changed(self, change):
        if change.get("name") == "value":
            self.body.children = (self.csv_widget,) if change["new"] == "csv" else (self.sql_widget,)
            callback = getattr(self, "on_source_change", None)
            if callback is not None:
                callback(change["new"])
