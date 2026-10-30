"""Add a deterministic row identity to an unregistered CSV staging copy when needed.

The browser upload is already staged as a temporary server-side file. This
routine never modifies the user's source file or a registered dataset.
"""
import csv
import os
from pathlib import Path
import tempfile


def ensure_integer_row_id(path, *, delimiter=",", quotechar='"', encoding="utf-8-sig"):
    """Return (generated, column) after checking all rows, preserving their order.

    Unique, non-negative integer text columns are preserved. If absent, an
    unused name is generated. A temporary sibling is atomically substituted.
    """
    source = Path(path)
    with source.open("r", encoding=encoding, newline="") as stream:
        reader = csv.reader(stream, delimiter=delimiter, quotechar=quotechar, strict=True)
        header = next(reader)
        if not header or len(header) != len(set(header)) or any(not x.strip() for x in header):
            raise ValueError("Invalid CSV headers")
        rows = list(reader)
    if not rows or any(len(row) != len(header) for row in rows):
        raise ValueError("Invalid CSV row shape")
    target_names = {"label", "labels", "class", "classification", "target", "y", "sentiment"}
    for i, column in enumerate(header):
        if column.strip().lower() in target_names:
            continue
        values = [r[i].strip() for r in rows]
        if (all(v.isascii() and v.isdecimal() for v in values)
                and len(set(values)) == len(values)):
            return False, column
    name = "jbg_row_id"
    suffix = 2
    while name in header:
        name = f"jbg_row_id_{suffix}"
        suffix += 1
    fd, temp_name = tempfile.mkstemp(prefix=".jbg_id_", suffix=".csv", dir=source.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding=encoding) as stream:
            writer = csv.writer(stream, delimiter=delimiter, quotechar=quotechar)
            writer.writerow([name, *header])
            for row_number, row in enumerate(rows, 1):
                writer.writerow([str(row_number), *row])
        if Path(temp_name).stat().st_size > 25 * 1024 * 1024:
            raise ValueError("CSV exceeds 25 MB after ID generation")
        os.replace(temp_name, source)
    finally:
        if Path(temp_name).exists():
            Path(temp_name).unlink()
    return True, name
