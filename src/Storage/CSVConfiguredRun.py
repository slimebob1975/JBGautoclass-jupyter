"""Validated, SQL-free bridge from private CSV configuration to run preparation.

Does not execute training, perform SQL I/O or expose dataset column names.
"""
from dataclasses import dataclass
from io import BytesIO

from Storage.CSVModelConfiguration import load_csv_model_configuration
from Storage.CSVRunContext import prepare_csv_run


@dataclass(frozen=True)
class ValidatedCSVRun:
    context: object
    configuration_id: str
    row_count: int


def prepare_configured_csv_run(repository, scope, configuration_id, *,
                               delimiter=None, encoding=None, quotechar=None):
    """Verify persisted CSV bytes and selected column roles before run publication.

    Saved dialect is authoritative when present; legacy configurations may
    supply explicit settings. Caller must authorize scope separately.
    """
    configuration = load_csv_model_configuration(repository, scope, configuration_id)
    stored = configuration.csv_format or {}
    delimiter = stored.get('delimiter', ',') if delimiter is None else delimiter
    encoding = stored.get('encoding', 'utf-8-sig') if encoding is None else encoding
    quotechar = stored.get('quotechar', '"') if quotechar is None else quotechar
    if configuration.csv_format is not None and (
            delimiter != stored['delimiter'] or encoding != stored['encoding']
            or quotechar != stored['quotechar']):
        raise ValueError('CSV format does not match saved configuration')
    if (not isinstance(delimiter, str) or len(delimiter) != 1 or delimiter in '\r\n\x00'
            or not isinstance(quotechar, str) or len(quotechar) != 1
            or quotechar in '\r\n\x00' or delimiter == quotechar
            or not isinstance(encoding, str) or not encoding.strip()):
        raise ValueError('Invalid CSV format settings')
    try:
        import pandas as pd
        with repository.open_artifact(scope, configuration.artifact_id) as stream:
            data = stream.read()
        frame = pd.read_csv(BytesIO(data), sep=delimiter, encoding=encoding,
                            quotechar=quotechar, dtype=object, keep_default_na=False,
                            index_col=False, on_bad_lines='error')
        names = list(frame.columns)
        required = (configuration.id_column, configuration.class_column,
                    *configuration.data_columns, *configuration.text_columns)
        if (not len(frame) or len(names) != len(set(names))
                or any(name.rsplit('.', 1)[-1].isdigit() and name.rsplit('.', 1)[0] in names
                       for name in names)
                or any(name not in names for name in required)
                or frame[configuration.id_column].eq('').any()
                or not frame[configuration.id_column].is_unique
                or frame[configuration.class_column].eq('').all()):
            raise ValueError
    except (ValueError, OSError, UnicodeError, LookupError, KeyError, pd.errors.ParserError,
            pd.errors.EmptyDataError):
        raise ValueError('Invalid CSV data for model configuration') from None
    context = prepare_csv_run(repository, scope, configuration.artifact_id,
                              id_column=configuration.id_column,
                              class_column=configuration.class_column,
                              feature_columns=configuration.data_columns + configuration.text_columns)
    return ValidatedCSVRun(context, configuration.configuration_id, len(frame))
