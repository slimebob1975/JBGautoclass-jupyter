from io import StringIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from JBGDarkNumberReporting import build_dark_number_intervals, format_dark_number_interval


NAMES = ["D_cv_test - Cross-trained model", "D_cv_full - Cross-trained model",
         "D_retrained_full - Retrained model", "Combined (legacy full-data)"]
VALUES = [0.029083820006059854, 0.005065670998365261,
          0.00290572524391599, 0.009062039738029626]


def results(values=VALUES, source="direct"):
    return pd.DataFrame({"Model type": NAMES, "type": "separated_alpha", "target": "Ja",
                         "dark_number": values, "corr_source": source}, index=[0, 0, 0, 0])


def test_submitted_run_has_four_points_in_requested_order_and_no_mutation():
    matrix = results()
    before = matrix.copy(deep=True)
    interval, = build_dark_number_intervals(matrix)
    assert [p["value"] for p in interval["points"]] == [VALUES[2], VALUES[1], VALUES[3], VALUES[0]]
    assert interval["status"] == "ordered"
    plain, rich = format_dark_number_interval(interval)
    assert "[D_re_full, D_cv_full, D_comb, D_cv_test] = [0.00290573, 0.00506567, 0.00906204, 0.0290838]" in plain
    assert "<i>D</i><sub>re_full</sub>" in rich
    assert "Expected order confirmed" in rich
    pd.testing.assert_frame_equal(matrix, before)


def test_all_formulas_and_targets_recover_blank_model_block_labels():
    formulas = ["base", "single_alpha", "separated_alpha", "non_linear", "non_linear_alpha"]
    blocks = []
    for name, value in zip(NAMES, VALUES):
        block = pd.DataFrame([{"Model type": name if i == 0 else "", "type": formula,
                               "target": target, "dark_number": value, "corr_source": "direct"}
                              for i, (formula, target) in enumerate(
                                  (f, t) for f in formulas for t in ["Ja", "Nej"])])
        blocks.append(block)
    matrix = pd.concat(blocks)
    before = matrix.copy(deep=True)
    intervals = build_dark_number_intervals(matrix)
    assert len(intervals) == 10
    assert {(s["formula"], s["target"]) for s in intervals} == {(f, t) for f in formulas for t in ["Ja", "Nej"]}
    assert all(s["status"] == "ordered" for s in intervals)
    assert all([p["value"] for p in s["points"]] == [VALUES[2], VALUES[1], VALUES[3], VALUES[0]] for s in intervals)
    pd.testing.assert_frame_equal(matrix, before)


def test_order_uses_unrounded_values_and_preserves_inversions():
    interval, = build_dark_number_intervals(results([0.1, 0.100000002, 0.100000003, 0.100000001]))
    assert interval["status"] == "broken"
    assert interval["inversions"] == ["D_re_full > D_cv_full", "D_cv_full > D_comb", "D_comb > D_cv_test"]
    plain, rich = format_dark_number_interval(interval)
    assert "Expected order broken" in plain
    assert "D_re_full &gt; D_cv_full" in rich
    assert all(v == "0.1" for v in [format(p["value"], ".6g") for p in interval["points"]])


def test_ties_and_zero_fp_outputs_are_valid_but_not_estimated_corrections():
    interval, = build_dark_number_intervals(results([0.0] * 4, "not_needed_zero_fp"))
    assert interval["status"] == "ordered"
    plain, rich = format_dark_number_interval(interval)
    assert "[0, 0, 0, 0]" in plain
    assert "Correction not needed (zero observed FP)" in rich


@pytest.mark.parametrize("case", ["missing", "duplicate", "nan", "inf", "bad_value",
                                 "fallback", "unknown", "no_provenance", "inconsistent_zero_fp"])
def test_missing_invalid_or_unestimable_estimates_do_not_claim_completed_order(case):
    matrix = results()
    if case == "missing":
        matrix = matrix.iloc[[0, 1, 3]]
    elif case == "duplicate":
        matrix = pd.concat([matrix, matrix.iloc[[2]]])
    elif case in {"nan", "inf"}:
        matrix.iloc[2, matrix.columns.get_loc("dark_number")] = np.nan if case == "nan" else np.inf
    elif case == "bad_value":
        matrix["dark_number"] = matrix["dark_number"].astype(object)
        matrix.iloc[2, matrix.columns.get_loc("dark_number")] = "invalid"
    elif case == "no_provenance":
        matrix = matrix.drop(columns="corr_source")
    else:
        matrix.iloc[2, matrix.columns.get_loc("corr_source")] = {
            "fallback": "fallback/unestimable", "unknown": "unknown",
            "inconsistent_zero_fp": "not_needed_zero_fp",
        }[case]
    interval, = build_dark_number_intervals(matrix)
    assert interval["status"] == "incomplete"
    assert interval["points"][0]["value"] is None
    plain, rich = format_dark_number_interval(interval)
    assert "Order check incomplete" in plain
    assert "Expected order confirmed" not in rich
    assert ("unestimable" if case == "fallback" else "N/A") in plain


def test_incomplete_range_still_flags_visible_inversion():
    matrix = results([0.01, 0.02, 0.03, 0.04]).iloc[[0, 1, 2]]
    interval, = build_dark_number_intervals(matrix)
    assert interval["status"] == "incomplete"
    assert interval["inversions"] == ["D_re_full > D_cv_full"]
    assert "Observed inversion: D_re_full > D_cv_full" in format_dark_number_interval(interval)[0]


def test_extra_model_results_do_not_replace_any_of_the_four_scopes():
    extra = results().iloc[[0]].copy()
    extra["Model type"] = "D_model_3_full - Other model"
    extra["dark_number"] = 999.0
    assert build_dark_number_intervals(pd.concat([results(), extra])) == build_dark_number_intervals(results())


def test_target_and_formula_labels_are_html_escaped():
    matrix = results()
    matrix["target"] = '<script>alert("Ja")</script>'
    matrix["type"] = "<b>formula</b>"
    plain, rich = format_dark_number_interval(build_dark_number_intervals(matrix)[0])
    assert "<script>" in plain
    assert "<script>" not in rich and "<b>formula" not in rich
    assert "&lt;script&gt;" in rich


@pytest.mark.parametrize("matrix", [pd.DataFrame(), pd.DataFrame({"dark_number": [0.1]})])
def test_no_calculation_rows_produce_no_range(matrix):
    assert build_dark_number_intervals(matrix) == []


def test_range_is_immediately_below_table_and_leaves_csv_exports_unchanged(monkeypatch):
    import JBGHandler

    events = []
    class Logger:
        def display_matrix(self, title, matrix, **kwargs):
            events.append(("table", title))
        def print_info(self, text, **kwargs):
            assert kwargs["print_always"] is True
            events.append(("range", text, kwargs["html_function"](text)))
        def print_code(self, *args):
            events.append(("download", args[0]))
    ph = object.__new__(JBGHandler.PredictionsHandler)
    ph.handler = SimpleNamespace(logger=Logger())
    ph.dark_numbers = results()
    ph.dark_numb_conf_matrix = pd.DataFrame({"Ja": [24, 20], "Nej": [32, 2806]})
    exported = {}
    monkeypatch.setattr(JBGHandler.Helpers, "save_matrix_as_csv", lambda frame, path: exported.update({path: frame.copy(deep=True)}))
    monkeypatch.setattr(JBGHandler.Helpers, "create_download_link", lambda path, **kwargs: path)
    ph.evaluate_dark_numbers("numbers.csv", "matrix.csv")
    assert [e[0] for e in events] == ["table", "table", "range", "download", "download"]
    assert events[1][1] == "Dark numbers calculations"
    assert "<i>D</i><sub>re_full</sub>" in events[2][2]
    pd.testing.assert_frame_equal(exported["numbers.csv"], ph.dark_numbers)
    pd.testing.assert_frame_equal(exported["matrix.csv"], ph.dark_numb_conf_matrix)


@pytest.mark.parametrize("module_name", ["JBGLogger", "JBGStreamedLogger"])
def test_both_loggers_render_range_when_quiet_and_streamed_log_is_plain(module_name, monkeypatch):
    import importlib

    module = importlib.import_module(module_name)
    logger = object.__new__(module.JBGLogger)
    logger._enable_quiet = True
    logger.in_terminal = False
    logger._log_buffer = StringIO()
    logged, displayed = [], []
    logger.log_file = SimpleNamespace(write_message=logged.append)
    def capture_html(text, **kwargs):
        render = kwargs.get("html_function", lambda value: value)
        displayed.append(render(text))
    monkeypatch.setattr(module, "print_html", capture_html)
    plain, rich = format_dark_number_interval(build_dark_number_intervals(results())[0])
    logger.print_info(plain, print_always=True, html_function=lambda text: rich)
    assert displayed == [rich]
    if module_name == "JBGStreamedLogger":
        assert logged == [plain]
        assert logger._log_buffer.getvalue() == plain + "\n"


def test_terminal_logger_keeps_readable_four_point_expression():
    from JBGLogger import JBGLogger

    logger = object.__new__(JBGLogger)
    logger._enable_quiet = True
    logger.in_terminal = True
    logged = []
    logger.writeln = lambda level, text: logged.append(text)
    plain, rich = format_dark_number_interval(build_dark_number_intervals(results())[0])
    logger.print_info(plain, print_always=True, html_function=lambda text: rich)
    assert logged == [plain]
