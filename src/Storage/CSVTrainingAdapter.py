"""Read scoped CSV artifacts as the row tuples consumed by JBG DatasetHandler.

Opt-in preparation boundary only. No SQL access, training or result writes.
The caller must supply an authenticated/authorized storage scope.
"""
from io import BytesIO

import pandas as pd

from Storage.CSVModelConfiguration import load_csv_model_configuration
from Storage.ArtifactContract import ArtifactKind
from Storage.CSVReclassification import list_csv_corrections, read_corrected_csv


class CSVTrainingAdapter:
    """A minimal get_dataset-compatible reader for registered CSV datasets."""

    def __init__(self, repository, scope, configuration_id, config):
        self.repository = repository
        self.scope = scope
        self.configuration = load_csv_model_configuration(repository, scope, configuration_id)
        self.config = config
        # Freeze the correction head at preparation, before reading any rows.
        # None means the original CSV, even if a new correction is appended later.
        revisions = list_csv_corrections(repository, scope, configuration_id)
        self.correction_revision = revisions[-1] if revisions else None
        c = self.configuration
        # The pipeline's column order is text + numerical + ID + class.
        self.column_names = list(config.get_column_names())
        expected = (list(c.text_columns) + list(c.data_columns)
                    + [c.id_column, c.class_column])
        if (self.column_names != expected
                or config.get_id_column_name() != c.id_column
                or config.get_class_column_name() != c.class_column):
            raise ValueError('CSV model roles do not match pipeline configuration')
        source = getattr(config, 'data_source', None)
        if source is not None and getattr(source, 'kind', 'sql') == 'csv':
            if (getattr(source, 'dataset_id', None) != c.dataset_id
                    or getattr(source, 'artifact_id', None) != c.artifact_id):
                raise ValueError('CSV dataset identity mismatch')

    def load_dataframe(self, num_rows=None):
        """Opt-in common DataFrame view; existing get_dataset remains canonical."""
        from DataLayer.FrameContract import load_dataframe
        return load_dataframe(self, self.config, num_rows)

    def get_dataset(self, num_rows=None):
        """Return ordered row tuples, as SQLDataLayer.get_dataset does.

        Train-only selects known labels; predict-only selects unknown labels.
        Both modes select all rows. A positive data limit caps returned rows.
        """
        c = self.configuration
        fmt = c.csv_format or {'delimiter': ',', 'quotechar': '"', 'encoding': 'utf-8-sig'}
        ref = self.repository.get_reference(self.scope, c.artifact_id)
        if (ref.kind != ArtifactKind.DATASET or ref.content_type != 'text/csv'
                or ref.dataset_id != c.dataset_id):
            raise ValueError('CSV dataset identity mismatch')
        try:
            if self.correction_revision is None:
                with self.repository.open_artifact(self.scope, c.artifact_id) as stream:
                    raw = stream.read()
            else:
                raw = read_corrected_csv(self.repository, self.scope,
                                         c.configuration_id,
                                         revision=self.correction_revision)
            frame = pd.read_csv(BytesIO(raw), sep=fmt['delimiter'],
                                quotechar=fmt['quotechar'], encoding=fmt['encoding'],
                                dtype=object, keep_default_na=False, index_col=False,
                                on_bad_lines='error')
            names = list(frame.columns)
            if (not len(frame) or len(names) != len(set(names))
                    or any(n.rsplit('.', 1)[-1].isdigit() and n.rsplit('.', 1)[0] in names for n in names)
                    or any(name not in names for name in self.column_names)
                    or frame[c.id_column].eq('').any()
                    or not frame[c.id_column].is_unique):
                raise ValueError
            frame = frame.loc[:, self.column_names]
            # Interpret missing-label tokens only in the configured class column.
            # Keep feature values and identifiers untouched, including literal NULL.
            values = frame[c.class_column]
            stripped = values.astype('string').str.strip()
            unknown = values.isna() | stripped.eq('').fillna(True) | stripped.str.casefold().eq('null').fillna(True)
            frame.loc[unknown, c.class_column] = ''
            labelled = ~unknown
            train = self.config.should_train()
            predict = self.config.should_predict()
            if train and not predict:
                frame = frame.loc[labelled]
            elif predict and not train:
                frame = frame.loc[~labelled]
            limit = num_rows if num_rows is not None else self.config.get_data_limit()
            # JBG Config represents an unset data limit as zero.
            if limit is not None:
                if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                    raise ValueError('Invalid CSV row limit')
                if limit > 0:
                    frame = frame.iloc[:limit]
            return list(frame.itertuples(index=False, name=None))
        except (OSError, UnicodeError, LookupError, KeyError, ValueError,
                pd.errors.ParserError, pd.errors.EmptyDataError):
            raise ValueError('Unable to prepare CSV rows for training pipeline') from None
