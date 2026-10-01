from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from scipy import sparse

from JBGDarkNumbers import DarkNumberCalculator
from JBGHandler import PredictionsHandler


class Logger:
    def __init__(self):
        self.info = []
        self.warnings = []
    def print_info(self, message):
        self.info.append(message)
    def print_warning(self, message):
        self.warnings.append(message)


class FixedModel(BaseEstimator):
    def __init__(self, predictions):
        self.predictions = predictions
        self.predict_calls = 0
        self.probability_calls = 0
    def predict(self, X):
        self.predict_calls += 1
        return np.asarray(self.predictions)[np.asarray(X)[:, 0].astype(int)]
    def predict_proba(self, X):
        self.probability_calls += 1
        # Confidence is fixed; avoid counting a second predict() call internally.
        return np.tile([0.9, 0.1], (len(X), 1))


def predictions_handler():
    ph = object.__new__(PredictionsHandler)
    ph.handler = SimpleNamespace(
        logger=Logger(), config=SimpleNamespace(get_dark_number_flip_fraction=lambda: 0.2),
    )
    ph.correction_calls = []
    def corrections(*, model, X, Y, labels, model_name, **kwargs):
        ph.correction_calls.append((model_name, list(labels), len(Y)))
        factor = 2.0 if model_name == 'Cross' else 5.0
        return ({label: factor for label in labels}, {label: 'direct' for label in labels},
                {label: model_name for label in labels})
    ph._calculate_dark_number_corrections = corrections
    return ph


def run(ph, y, models, *, validation=None, combine=True, target='Ja', calculation_type='base'):
    X = pd.DataFrame({'row': range(len(y))})
    kwargs = {}
    if validation is not None:
        ids, labels = validation
        kwargs.update(X_validation=pd.DataFrame({'row': ids}), Y_validation=pd.Series(labels))
    ph.get_dark_numbers(
        X=X, Y=pd.Series(y), models=models,
        model_names=['Cross', 'Retrained', 'Extra'][:len(models)],
        target_class=target, combine_models=combine, type=calculation_type, **kwargs,
    )
    results = ph.dark_numbers.copy()
    results['Model type'] = results['Model type'].replace('', np.nan).ffill()
    return results


@pytest.mark.parametrize('calculation_type', [*DarkNumberCalculator.FORMULA_LATEX, 'all'])
def test_zero_fp_reproduction_skips_all_correction_work_and_keeps_every_output(calculation_type):
    y = ['Ja'] * 69 + ['Nej'] * 4511
    cross_predictions = y.copy()
    retrained_predictions = y.copy()
    for i in range(12):
        cross_predictions[i] = 'Nej'
    for i in range(9, 15):
        retrained_predictions[i] = 'Nej'
    validation_ids = [*range(7), *range(12, 19), *range(69, 971)]
    validation_y = ['Ja'] * 14 + ['Nej'] * 902
    cross = FixedModel(cross_predictions)
    retrained = FixedModel(retrained_predictions)
    ph = predictions_handler()
    def forbidden(**kwargs):
        raise AssertionError('Correction, regression and shadow fitting must all be skipped')
    ph._calculate_dark_number_corrections = forbidden
    results = run(ph, y, [cross, retrained], validation=(validation_ids, validation_y),
                  calculation_type=calculation_type)
    assert set(results['corr_source']) == {'not_needed_zero_fp'}
    assert (results['corr'] == 1.0).all()
    assert (results['dark_number'] == 0.0).all()
    assert len(results) == (20 if calculation_type == 'all' else 4)
    assert ph.dark_numb_conf_matrix[['Ja', 'Nej']].to_numpy().tolist() == [
        [7, 7], [0, 902], [57, 12], [0, 4511], [63, 6], [0, 4511], [54, 15], [0, 4511],
    ]
    assert cross.predict_calls == cross.probability_calls == 2
    assert retrained.predict_calls == retrained.probability_calls == 1
    assert ph.dark_number_fallback_events.empty
    assert len(ph.handler.logger.warnings) == 2
    assert 'despite observed false negatives' in ph.handler.logger.warnings[0]


def test_one_false_positive_prevents_skip():
    ph = predictions_handler()
    results = run(ph, ['Ja', 'Nej', 'Nej', 'Ja'], [FixedModel(['Ja', 'Ja', 'Nej', 'Nej'])], combine=False)
    assert ph.correction_calls == [('Cross', ['Ja'], 4)]
    assert results['corr_source'].tolist() == ['direct']
    assert results['dark_number'].iloc[0] == pytest.approx(1.5)


def test_holdout_fp_requires_cross_correction_even_when_full_data_fp_is_zero():
    ph = predictions_handler()
    cross = FixedModel(['Ja', 'Nej', 'Nej', 'Nej', 'Ja', 'Ja'])
    retrained = FixedModel(['Ja', 'Nej', 'Ja', 'Nej'])
    results = run(ph, ['Ja', 'Nej', 'Ja', 'Nej'], [cross, retrained],
                  validation=([4, 5], ['Ja', 'Nej']))
    assert ph.correction_calls == [('Cross', ['Ja'], 4)]
    assert results['corr_source'].tolist() == ['direct', 'direct', 'not_needed_zero_fp', 'direct']
    assert results.loc[results['Model type'].str.startswith('D_cv_test'), 'dark_number'].iloc[0] == 2.0


@pytest.mark.parametrize('combine', [True, False])
def test_combined_fp_prevents_skipping_its_correction_owner(combine):
    ph = predictions_handler()
    cross = FixedModel(['Nej'] * 4)
    retrained = FixedModel(['Ja'] * 4)
    results = run(ph, ['Ja', 'Nej', 'Ja', 'Nej'], [cross, retrained], combine=combine)
    expected = [('Cross', ['Ja'], 4), ('Retrained', ['Ja'], 4)] if combine else [('Retrained', ['Ja'], 4)]
    assert ph.correction_calls == expected
    if combine:
        combined = results.loc[results['Model type'] == 'Combined (legacy full-data)'].iloc[0]
        assert combined['corr'] == 2.0 and combined['corr_model'] == 'Cross'
        assert combined['dark_number'] > 0
    else:
        assert results['corr_source'].tolist() == ['not_needed_zero_fp', 'direct']


def test_multiclass_all_targets_only_estimates_labels_whose_factor_matters():
    ph = predictions_handler()
    results = run(ph, ['A', 'B', 'C', 'A', 'B', 'C'],
                  [FixedModel(['B', 'B', 'C', 'A', 'B', 'C'])], combine=False, target='')
    assert ph.correction_calls == [('Cross', ['B'], 6)]
    assert results.set_index('target')['corr_source'].to_dict() == {
        'A': 'not_needed_zero_fp', 'B': 'direct', 'C': 'not_needed_zero_fp',
    }


def test_zero_fp_path_preserves_sparse_estimator_input():
    class SparseModel:
        def predict(self, X):
            assert sparse.issparse(X)
            return np.array(['Ja', 'Nej', 'Nej', 'Nej'])
        def predict_proba(self, X):
            assert sparse.issparse(X)
            return np.tile([0.9, 0.1], (X.shape[0], 1))
    ph = predictions_handler()
    X = pd.DataFrame.sparse.from_spmatrix(sparse.csr_matrix(np.eye(4)))
    ph.get_dark_numbers(X=X, Y=pd.Series(['Ja', 'Nej', 'Ja', 'Nej']),
                        models=[SparseModel()], combine_models=False,
                        target_class='Ja', type='separated_alpha')
    assert ph.correction_calls == []
    assert ph.dark_numbers['corr_source'].tolist() == ['not_needed_zero_fp']


def test_combined_uses_first_probability_model_when_first_model_has_no_probabilities():
    class PredictOnlyModel:
        def predict(self, X):
            return np.array(['Nej'] * len(X))
    ph = predictions_handler()
    results = run(ph, ['Ja', 'Nej', 'Ja', 'Nej'],
                  [PredictOnlyModel(), FixedModel(['Ja'] * 4)], combine=True)
    assert ph.correction_calls == [('Retrained', ['Ja'], 4)]
    combined = results.loc[results['Model type'] == 'Combined (legacy full-data)'].iloc[0]
    assert combined['corr_model'] == 'Retrained' and combined['corr'] == 5.0
    assert len(ph.dark_numb_conf_matrix) == 6


@pytest.mark.parametrize('case', ['no_negatives', 'no_positives', 'misaligned', 'nan_labels', 'nan_probability', 'unknown_formula'])
def test_zero_fp_guard_does_not_optimize_invalid_or_undefined_scopes(case):
    real = pd.Series(['Ja', 'Nej', 'Ja', 'Nej'])
    predicted = pd.Series(['Ja', 'Nej', 'Nej', 'Nej'])
    probability = pd.Series([0.9] * 4)
    calculation_type = 'all'
    if case == 'no_negatives': real[:] = 'Ja'
    elif case == 'no_positives': real[:] = 'Nej'
    elif case == 'misaligned': predicted.index = [4, 5, 6, 7]
    elif case == 'nan_labels': real.iloc[0] = np.nan
    elif case == 'nan_probability': probability.iloc[0] = np.nan
    elif case == 'unknown_formula': calculation_type = 'unknown'
    assert not DarkNumberCalculator.correction_is_unused(real, predicted, probability, 'Ja', calculation_type)
