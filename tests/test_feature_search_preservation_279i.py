"""Regression coverage for the SQL/CSV feature-investigation conversion guard."""
import pandas as pd
import pytest

from JBGFeaturePreservation import validate_investigation_feature_preservation


class Roles:
    def __init__(self, numerical):
        self.numerical = numerical

    def get_numerical_column_names(self):
        return self.numerical


def test_numeric_coercion_of_text_fails_for_either_data_source():
    rows = [('lace-up', '230'), ('slip-on', '510')]
    converted = pd.DataFrame({'title': [float('nan')] * 2, 'price': [230., 510.]})
    with pytest.raises(ValueError, match='title: 2 nonempty source values, 0 retained'):
        validate_investigation_feature_preservation(
            rows, ['title', 'price'], Roles(['title', 'price']), converted)


def test_text_correctly_declared_not_reported():
    rows = [('lace-up', '230'), ('slip-on', '510')]
    converted = pd.DataFrame({'title': ['lace-up', 'slip-on'], 'price': [230., 510.]})
    validate_investigation_feature_preservation(rows, ['title', 'price'], Roles(['price']), converted)


def test_originally_missing_column_is_not_conversion_loss():
    rows = [('', '230'), ('', '510')]
    converted = pd.DataFrame({'title': [float('nan')] * 2, 'price': [230., 510.]})
    validate_investigation_feature_preservation(rows, ['title', 'price'], Roles(['title', 'price']), converted)


def test_partial_numeric_loss_is_not_silent():
    rows = [('12',), ('not-a-number',)]
    converted = pd.DataFrame({'price': [12., float('nan')]})
    with pytest.raises(ValueError, match='price: 2 nonempty source values, 1 retained'):
        validate_investigation_feature_preservation(rows, ['price'], Roles(['price']), converted)
