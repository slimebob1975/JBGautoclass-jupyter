"""Source-aware prediction identifiers for the shared result pipeline.

SQL retains its historical integer conversion. CSV identifiers are opaque
strings and must not lose leading zeroes during result construction.
"""


def prediction_key(value, source_kind: str):
    if source_kind == "sql":
        return int(value)
    if source_kind != "csv":
        raise ValueError("Unsupported prediction data source")
    if isinstance(value, str) and value:
        return value
    raise ValueError("CSV prediction ID must be a nonempty string")
