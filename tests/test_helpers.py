from decimal import Decimal
from datetime import datetime

import pandas
import numpy as np
from scipy import sparse as scipy_sparse
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

import Helpers

# Since this is a series of functions, seems fitting to not put the tests into a class

def test_column_operations():
    # removes single empty value
    column_names = "a,,b"
    cleaned = Helpers.clean_column_names_list(column_names)
    assert cleaned == ["a", "b"]
    
    # Removes multiple empty values
    column_names = "a,,b,"
    cleaned = Helpers.clean_column_names_list(column_names)
    assert cleaned == ["a", "b"]

    # should return an empty list
    column_names = ""
    assert Helpers.clean_column_names_list(column_names) == []

    column_names = "a,,b"
    cleaned = Helpers.get_from_string_or_list(column_names)
    assert cleaned == ["a", "b"]

    column_names = ["a", "", "b"]
    cleaned = Helpers.get_from_string_or_list(column_names)
    assert cleaned == ["a", "b"]

    column_names = ["a", "b"]
    cleaned = Helpers.get_from_string_or_list(column_names)
    assert cleaned == ["a", "b"]
    
def test_positive_int_or_none():
        """ Help function to test input params"""
        # 1. None
        assert Helpers.positive_int_or_none(None)
        # 2. 3
        assert Helpers.positive_int_or_none(3)
        # 3. -2
        assert not Helpers.positive_int_or_none(-2)
        # 4. 3.5
        assert not Helpers.positive_int_or_none(3.5)
    
def test_set_none_or_int():
    """ Help function for loading input params from file """
    # 1: "None"
    assert Helpers.set_none_or_int("None") is None
    # 2. "5"
    value = Helpers.set_none_or_int("5")
    assert value == 5 and isinstance(value, int)
    # 3. 4
    assert Helpers.set_none_or_int(4) == 4
    # 4. -3
    assert Helpers.set_none_or_int(-3) is None
    # 5. -3.5
    assert Helpers.set_none_or_int(-3.5) is None

def test_is_float():
    """ Check a few different values to see what it can and can't handler"""
    
    # 1. String in a "float format" can be read as a float
    assert Helpers.is_float("1.5")

    # 2. General string can not
    assert not Helpers.is_float("abc")

    # 3. int can be read as a float
    assert Helpers.is_float(5)

    # 4. Float can be read as a float
    assert Helpers.is_float(1.5)

    # 5. Unlikely, but Decimal can be read as a float
    assert Helpers.is_float(Decimal(1.5))

    # 5. Unlikely, but Decimal can be read as a float
    assert Helpers.is_float(Decimal(1))

    # 6. Boolean values can be read as floats
    assert Helpers.is_float(False)
    assert Helpers.is_float(True)

def test_is_int():
    """ Valid floats can be read as int, but the value is cut off """
    # 1. String in a "float format" can't be read as an int
    assert not Helpers.is_int("1.5")

    # 2. General string can not
    assert not Helpers.is_int("abc")

    # 3. String in "int format" can
    assert Helpers.is_int("5")

    # 4. int can be read as an int
    assert Helpers.is_int(5)

    # 5. Float can be read as an int
    assert Helpers.is_int(1.5)

    # 6. Decimal with "float value" can be read as an int
    assert Helpers.is_int(Decimal(1.5))

    # 7.Decimal "int value" can be read as an int
    assert Helpers.is_int(Decimal(1))

    # 8. Boolean values can be read as int
    assert Helpers.is_int(False)
    assert Helpers.is_int(True)

def test_is_str():
    """ As the value you input is likely a string, needs to check value """
    assert not Helpers.is_str("1.5") # This is float

    assert Helpers.is_str("abc")

    assert not Helpers.is_str("5") # This is float or int

    assert not Helpers.is_str(5) # This is float or int

    assert not Helpers.is_str(1.5) # This is float or int

    assert not Helpers.is_str(Decimal(1.5)) # This is float or int

    assert not Helpers.is_str(Decimal(1)) # This is float or int

    assert not Helpers.is_str(False) # boolean
    
    assert not Helpers.is_str(True) # boolean

    assert not Helpers.is_str("2004-05-09") # Datetime


def test_get_datetime():
    """ This checks both type and some various forms of formats """
    now = datetime.now()

    assert isinstance(Helpers.get_datetime(now), datetime)

    assert isinstance(Helpers.get_datetime("2004-04-12 12:34:17"), datetime)
    assert isinstance(Helpers.get_datetime("2005-03-19"), datetime)
    assert isinstance(Helpers.get_datetime("2005-03-19 12:34:17.000001"), datetime)
    assert isinstance(Helpers.get_datetime("2005-03-19 12:34:17,000001"), datetime)
    assert isinstance(Helpers.get_datetime("24/11/2024 15:18:11"), datetime)
    assert isinstance(Helpers.get_datetime("19/03/2022"), datetime)
    assert isinstance(Helpers.get_datetime("19/03/2022 12:34:17.000001"), datetime)
    assert isinstance(Helpers.get_datetime("19/03/2022 12:34:17,000001"), datetime)
    assert isinstance(Helpers.get_datetime("11/2/2024 15:18:11"), datetime)
    assert isinstance(Helpers.get_datetime("12/3/2022"), datetime)
    assert isinstance(Helpers.get_datetime("12/3/2022 12:34:17.000001"), datetime)
    assert isinstance(Helpers.get_datetime("12/3/2022 12:34:17,000001"), datetime)
    assert isinstance(Helpers.get_datetime("24/11/2024 15:18:11"), datetime)
    assert isinstance(Helpers.get_datetime("19/3/2022"), datetime)
    assert isinstance(Helpers.get_datetime("19/3/2022 12:34:17.000001"), datetime)
    assert isinstance(Helpers.get_datetime("19/3/2022 12:34:17,000001"), datetime)
    assert isinstance(Helpers.get_datetime("19/03/2022"), datetime)
    assert isinstance(Helpers.get_datetime("19/03/2022 12:34:17.000001"), datetime)
    assert isinstance(Helpers.get_datetime("19/03/2022 12:34:17,000001"), datetime)

    # A few with invalid values
    assert Helpers.get_datetime("35/3/2022") is None # Day outside of valid days


    # Testdata for Validation
    # airline_tweets_mini, breast_cancer, creditcard_fraud, stroke_data och stars är alla vettiga testset

def test_save_matrix_as_csv_creates_parent_directory(tmp_path):
    matrix = pandas.DataFrame({"value": [1, 2]})
    filepath = tmp_path / "output" / "csvs" / "result.csv"

    Helpers.save_matrix_as_csv(matrix, filepath)

    assert filepath.is_file()


def test_prepare_estimator_input_preserves_sparse_representation():
    frame = pandas.DataFrame({
        "dense": [1.0, 2.0, 3.0],
        "sparse": pandas.arrays.SparseArray([0.0, 4.0, 0.0], fill_value=0.0),
    })

    prepared = Helpers.prepare_estimator_input(frame)

    assert scipy_sparse.isspmatrix_csr(prepared)
    np.testing.assert_allclose(
        prepared.toarray(),
        np.array([[1.0, 0.0], [2.0, 4.0], [3.0, 0.0]]),
    )


def test_prepare_estimator_input_keeps_dense_behavior():
    frame = pandas.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    assert Helpers.prepare_estimator_input(frame) is frame
    np.testing.assert_allclose(
        Helpers.prepare_estimator_input(frame, prefer_numpy=True),
        frame.to_numpy(),
    )


def test_contains_nan_handles_sparse_data_without_densifying():
    clean = pandas.DataFrame({
        "dense": [1.0, 2.0],
        "sparse": pandas.arrays.SparseArray([0.0, 3.0], fill_value=0.0),
    })
    dirty = clean.copy()
    dirty.loc[dirty.index[0], "dense"] = np.nan

    assert not Helpers.contains_nan(clean)
    assert Helpers.contains_nan(dirty)


def test_ensure_float64_preserves_dense_dataframe_shape_and_values():
    frame = pandas.DataFrame({"a": [1, 2], "b": [3, 4]}, dtype=np.int64)

    converted = Helpers.ensure_float64(frame)

    assert isinstance(converted, pandas.DataFrame)
    assert converted.index.equals(frame.index)
    assert converted.columns.equals(frame.columns)
    assert all(dtype == np.dtype("float64") for dtype in converted.dtypes)
    np.testing.assert_allclose(converted.to_numpy(), frame.to_numpy(dtype=float))


def test_ensure_float64_preserves_sparse_representation():
    matrix = scipy_sparse.csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.int64))

    converted = Helpers.ensure_float64(matrix)

    assert scipy_sparse.isspmatrix_csr(converted)
    assert converted.dtype == np.float64
    np.testing.assert_allclose(converted.toarray(), matrix.toarray())


def test_float_normalization_preserves_fractional_smote_samples():
    features = np.array(
        [[0, 0], [3, 3], [10, 10], [11, 11], [12, 12], [13, 13]],
        dtype=np.int64,
    )
    labels = np.array([1, 1, 0, 0, 0, 0])
    imputed = SimpleImputer(strategy="constant", fill_value=0).fit_transform(features)

    normalized = Helpers.ensure_float64(imputed)
    resampled, _ = SMOTE(k_neighbors=1, random_state=1).fit_resample(normalized, labels)

    synthetic = resampled[len(features):]
    assert resampled.dtype == np.float64
    assert synthetic.size > 0
    assert np.any(np.modf(synthetic)[0] != 0.0)


def test_model_performance_matrix_labels_holdout_as_diagnostic_and_does_not_rank_by_it():
    results = [
        ["MAX", "PCA", "A", 4, 0.95, 0.02, 0.70, 1.0, ""],
        ["MAX", "PCA", "B", 4, 0.95, 0.02, 0.99, 1.0, ""],
        ["MAX", "PCA", "C", 4, 0.94, 0.01, 1.00, 1.0, ""],
    ]

    matrix = Helpers.build_model_performance_matrix(results)

    assert "Holdout (diagnostic)" in matrix.columns
    assert "Test data" not in matrix.columns
    assert matrix["Algorithm (Library)"].tolist() == ["A", "B", "C"]
    assert matrix["Holdout (diagnostic)"].tolist() == [0.70, 0.99, 1.00]


def test_model_performance_matrix_uses_cv_stdev_as_only_tiebreaker():
    results = [
        ["MAX", "PCA", "wider", 4, 0.95, 0.03, 1.00, 1.0, ""],
        ["MAX", "PCA", "tighter", 4, 0.95, 0.01, 0.50, 1.0, ""],
    ]

    matrix = Helpers.build_model_performance_matrix(results)

    assert matrix["Algorithm (Library)"].tolist() == ["tighter", "wider"]
