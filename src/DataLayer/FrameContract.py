"""Opt-in shared DataFrame boundary for SQL and CSV row providers.

No value conversion is performed: legacy SQL row semantics must remain stable.
The ML pipeline continues to consume get_dataset() until parity is established.
"""

import pandas as pd


def dataframe_from_rows(rows, column_names):
    """Build a DataFrame with the configured column order, without coercion.

    The caller owns data retrieval and its authorization. Raises on malformed
    row shapes or duplicate column names instead of silently losing data.
    """
    names = list(column_names)
    if not names or len(names) != len(set(names)) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("Invalid dataset column contract")
    frame = pd.DataFrame(rows, columns=names)
    if frame.shape[1] != len(names):
        raise ValueError("Dataset column count mismatch")
    return frame


def load_dataframe(datalayer, config, num_rows=None):
    """Opt-in source-neutral adapter; leave get_dataset semantics untouched."""
    return dataframe_from_rows(datalayer.get_dataset(num_rows), config.get_column_names())
