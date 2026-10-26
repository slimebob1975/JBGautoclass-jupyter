"""Opt-in, aggregate-only diagnostics for the SQL/CSV DataFrame boundary.

Do not normalize values here: source-specific NULL, dtype and row-selection
semantics are intentionally preserved until SQL parity is proven. Never emit
column names, cell values, row identifiers, or dataset paths in diagnostics.
"""

from collections import Counter
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FrameProfile:
    rows: int
    columns: int
    null_cells: int
    empty_string_cells: int
    dtype_counts: tuple
    memory_bytes: int


def profile_dataframe(frame: pd.DataFrame) -> FrameProfile:
    """Return privacy-conscious aggregate diagnostics without changing data.

    Both `None` and pandas missing scalars count as null; empty strings remain
    a separate category. This is a diagnostic distinction, not a coercion rule.
    Memory is pandas' deep estimate, not peak process memory.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('Expected a pandas DataFrame')
    null_cells = int(frame.isna().to_numpy().sum())
    empty_cells = 0
    for series in frame.items():
        values = series[1]
        # Check only actual strings; numerical zero and missing values are not empty.
        empty_cells += int(values.map(lambda v: isinstance(v, str) and v == '').sum())
    counts = Counter(str(dtype) for dtype in frame.dtypes)
    return FrameProfile(
        rows=len(frame),
        columns=len(frame.columns),
        null_cells=null_cells,
        empty_string_cells=empty_cells,
        dtype_counts=tuple(sorted(counts.items())),
        memory_bytes=int(frame.memory_usage(index=True, deep=True).sum()),
    )


def compare_frame_profiles(sql_frame: pd.DataFrame, csv_frame: pd.DataFrame) -> dict:
    """Compare aggregate shape/type/missingness without source identifiers."""
    sql = profile_dataframe(sql_frame)
    csv = profile_dataframe(csv_frame)
    return {
        'row_count_equal': sql.rows == csv.rows,
        'column_count_equal': sql.columns == csv.columns,
        'null_cell_count_equal': sql.null_cells == csv.null_cells,
        'empty_string_count_equal': sql.empty_string_cells == csv.empty_string_cells,
        'dtype_distribution_equal': sql.dtype_counts == csv.dtype_counts,
        'sql': sql,
        'csv': csv,
    }
