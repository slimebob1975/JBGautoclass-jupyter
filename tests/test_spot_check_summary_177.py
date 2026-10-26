"""Revision 177: dependency-light plain-text spot-check reporting."""
import math

from Helpers import format_spot_check_summary


def row(algorithm, score, stdev=0.01):
    return ["NOG", "NUG", "STA", "PCA", algorithm, 4, score, stdev, 1.0, ""]


def test_ranking_selected_pipeline_and_csv_are_independent():
    text = format_spot_check_summary(
        [row("KERA - Keras (TensorFlow)", 0.2), row("FUTV - Future", 0.4),
         row("FAIL - Failure", math.nan)], "MCC",
        "FUTV | NOG + NUG + STA + PCA", 0.4, "/private/crossval_run.csv")
    assert "1. FUTV | NOG + NUG + STA + PCA | MCC 0.4000" in text
    assert "2. KERA | NOG + NUG + STA + PCA | MCC 0.2000" in text
    assert "FAIL" not in text
    assert "Selected pipeline:\nFUTV | NOG + NUG + STA + PCA" in text
    assert "Spot-check MCC: 0.4000" in text
    assert "Full results: crossval_run.csv" in text
    assert "/private/" not in text


def test_top_ten_and_no_finite_candidates():
    rows = [row(f"A{i}", i / 100) for i in range(12)]
    text = format_spot_check_summary(rows, "MCC", "A11", 0.11, "cv.csv")
    assert len([line for line in text.splitlines() if line[:1].isdigit()]) == 10
    empty = format_spot_check_summary([row("FAIL", math.inf)], "MCC", "NONE", 0, "cv.csv")
    assert "No candidates with a finite CV score." in empty
