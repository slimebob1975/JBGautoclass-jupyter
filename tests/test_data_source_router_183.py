"""Source selection regression tests without any SQL driver or server."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from CSVDataSource import CSVDataSource, CSVSourceSettings
from DataSourceRouter import DataSourceChoice, SourceKind, create_data_source


class DataSourceRouterTests(unittest.TestCase):
    def setUp(self):
        self.settings = CSVSourceSettings(
            path=Path(tempfile.gettempdir()) / "nonexistent_test_data.csv",
            id_column="row_id", class_column="target",
        )

    def test_sql_default_requires_dependencies(self):
        self.assertIs(DataSourceChoice().kind, SourceKind.SQL_SERVER)
        with self.assertRaisesRegex(ValueError, "configuration and logger"):
            create_data_source(DataSourceChoice())

    def test_csv_does_not_import_sql_layer(self):
        with patch.dict(sys.modules, {"SQLDataLayer": None}):
            source = create_data_source(DataSourceChoice(SourceKind.CSV, self.settings))
        self.assertIsInstance(source, CSVDataSource)
        self.assertIs(source.settings, self.settings)

    def test_sql_delegates_to_legacy_layer(self):
        calls = []

        class FakeLayer:
            def __init__(self, *, config, logger, validate):
                calls.append((config, logger, validate))

        fake = types.ModuleType("SQLDataLayer")
        fake.DataLayer = FakeLayer
        config, logger = object(), object()
        with patch.dict(sys.modules, {"SQLDataLayer": fake}):
            source = create_data_source(DataSourceChoice(), config=config, logger=logger, validate=False)
        self.assertIsInstance(source, FakeLayer)
        self.assertEqual(calls, [(config, logger, False)])

    def test_invalid_choices_fail_closed_and_hide_paths(self):
        for action in (
            lambda: DataSourceChoice(kind="csv", csv_settings=self.settings),
            lambda: DataSourceChoice(kind=SourceKind.CSV),
            lambda: DataSourceChoice(csv_settings=self.settings),
            lambda: create_data_source(None),
        ):
            with self.subTest(action=action):
                with self.assertRaises(ValueError) as error:
                    action()
                self.assertNotIn(str(self.settings.path), str(error.exception))
