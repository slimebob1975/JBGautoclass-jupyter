"""Presentation of the four existing Dark Number estimates; no recalculation."""

from html import escape
from math import isfinite

import pandas as pd


INTERVAL_SCOPES = (
    ("D_re_full", "D_retrained_full - "),
    ("D_cv_full", "D_cv_full - "),
    ("D_comb", "Combined (legacy full-data)"),
    ("D_cv_test", "D_cv_test - "),
)


def build_dark_number_intervals(matrix):
    """Group model-block rows by target/formula, retaining the requested order.

    Model names appear only on the first row of each block in the source table.
    Work on a copy, since the table and its CSV export are the calculation record.
    """
    required = {"Model type", "type", "target", "dark_number"}
    if matrix.empty or not required.issubset(matrix.columns):
        return []
    rows = matrix.copy(deep=True)
    rows["Model type"] = rows["Model type"].replace("", None).ffill()
    intervals = []
    for (formula, target), group in rows.groupby(["type", "target"], sort=False, dropna=False):
        points = []
        for label, prefix in INTERVAL_SCOPES:
            names = group["Model type"].fillna("").astype(str)
            matches = group[names.eq(prefix) if label == "D_comb" else names.str.startswith(prefix)]
            point = {"label": label, "value": None, "source": "", "reason": ""}
            if len(matches) != 1:
                point["reason"] = "missing" if matches.empty else "ambiguous"
            else:
                row = matches.iloc[0]
                source = str(row.get("corr_source", "unknown"))
                point["source"] = source
                try:
                    value = float(row["dark_number"])
                except (ValueError, TypeError, OverflowError):
                    value = float("nan")
                if not isfinite(value):
                    point["reason"] = "non-finite"
                elif "unestimable" in source or source in {"unknown", "nan", "", "None"}:
                    point["reason"] = "unestimable" if "unestimable" in source else "unknown correction"
                elif source == "not_needed_zero_fp" and value != 0:
                    point["reason"] = "inconsistent zero-FP result"
                else:
                    point["value"] = value
            points.append(point)

        # Check the original values, not their rounded display strings. Never sort.
        inversions = [
            f"{left['label']} > {right['label']}"
            for left, right in zip(points, points[1:])
            if left["value"] is not None and right["value"] is not None
            and left["value"] > right["value"]
        ]
        incomplete = [f"{point['label']}: {point['reason']}" for point in points if point["reason"]]
        status = "incomplete" if incomplete else "broken" if inversions else "ordered"
        intervals.append({
            "target": str(target), "formula": str(formula), "points": points,
            "status": status, "inversions": inversions, "incomplete": incomplete,
        })
    return intervals


def format_dark_number_interval(interval):
    """Return readable log text and a notebook-safe mathematical HTML expression.

    Native HTML subscripts and serif type avoid new widgets or a MathJax dependency.
    Values retain the proportions used in the calculation table, not percentages.
    """
    labels = [point["label"] for point in interval["points"]]
    values = [
        f"{point['value']:.6g}" if point["value"] is not None
        else "unestimable" if point["reason"] == "unestimable" else "N/A"
        for point in interval["points"]
    ]
    if interval["status"] == "ordered":
        status = "Expected order confirmed."
    elif interval["status"] == "broken":
        status = "Expected order broken: " + "; ".join(interval["inversions"]) + "."
    else:
        status = "Order check incomplete: " + "; ".join(interval["incomplete"]) + "."
        if interval["inversions"]:
            status += " Observed inversion: " + "; ".join(interval["inversions"]) + "."
    zero_fp = [point["label"] for point in interval["points"]
               if point["value"] == 0 and point["source"] == "not_needed_zero_fp"]
    note = ""
    if zero_fp:
        note = " Correction not needed (zero observed FP): " + ", ".join(zero_fp) + "."
    context = f"Target {interval['target']} / {interval['formula'].replace('_', ' ')}"
    expression = f"[{', '.join(labels)}] = [{', '.join(values)}]"
    plain = f"Dark Number estimate range — {context}: {expression}. {status}{note}"
    symbols = [f"<i>D</i><sub>{escape(label[2:])}</sub>" for label in labels]
    equation = f"[{', '.join(symbols)}] = [{', '.join(escape(value) for value in values)}]"
    color = "#8a4b00" if interval["status"] != "ordered" else "inherit"
    rich = (
        '<span style="display:block;margin:0.7em 0 1em">'
        f'<span style="display:block;font-size:0.9em">{escape(context)}</span>'
        '<span style="display:block;font-family:Cambria,Georgia,serif;'
        'font-size:1.3em;line-height:1.8;overflow-wrap:anywhere">'
        f'{equation}</span>'
        f'<span style="display:block;color:{color}">{escape(status + note)}</span>'
        '<span style="display:block;font-size:0.85em">'
        'Values are proportions. Heuristic estimate range.</span></span>'
    )
    return plain, rich
