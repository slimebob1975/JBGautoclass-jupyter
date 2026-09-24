from datetime import datetime
from pickle import PicklingError
from types import SimpleNamespace
import warnings
import numpy as np
import pandas
import pytest
from conftest import get_fixture_path

import JBGHandler as handler_module
from JBGExceptions import DatasetException, HandlerException

from JBGHandler import JBGHandler, DatasetHandler, Model, _SpotCheckState
from JBGMeta import Algorithm, Oversampling, Preprocess, Reduction, Undersampling

# One class per class in the module
class TestHandler():
    """ The main class """

    def add_handlers(self, handler: JBGHandler) -> None:
        handler.add_handler("dataset")
        handler.add_handler("predictions")
        handler.add_handler("model")

    def test_add_handler(self, default_handler):
        """ This has two cases: either it returns a valid <type>Handler or throws an exception"""
    
        with pytest.raises(HandlerException):
            default_handler.add_handler("does_not_exist")

        assert isinstance(default_handler.add_handler("dataset"), DatasetHandler)

        # This checks that the capitalization of the handler is irrelevant
        assert isinstance(default_handler.add_handler("dAtaSet"), DatasetHandler)

    def test_get_handler(self, default_handler):
        """ Two cases: returns a valid <type>Handler or throws an exception """

        with pytest.raises(HandlerException):
            default_handler.get_handler("does_not_exist")

        default_handler.add_handler("dataset")
        
        assert isinstance(default_handler.get_handler("dataset"), DatasetHandler)

        # This checks that the capitalization of the handler is irrelevant
        default_handler.add_handler("daTaseT")
        
        assert isinstance(default_handler.get_handler("dAtasEt"), DatasetHandler)

        
        

class TestDatasetHandler():
    """ Tests the dataset handler """
    # This has a lot of complex calculations so will postpone most of it

    def test_load_data(self, default_dataset_handler):
        """ This doesn't test the content of the items, but tests that they're generally proper """

        # Before anything runs, dataset, keys and classes should not be set
        assert not hasattr(default_dataset_handler, "dataset")
        assert not hasattr(default_dataset_handler, "keys")
        assert not hasattr(default_dataset_handler, "classes")
        data = [
            ["Karan",23, "odd", 1],
            ["Rohit",22, "even", 2],
            ["Sahil",21, "odd", 3],
            ["Aryan",24, "even", 4]
        ]
        default_dataset_handler.load_data(data)

        
        # After it runs, dataset, keys and classes should be set
        assert hasattr(default_dataset_handler, "dataset")
        assert hasattr(default_dataset_handler, "keys")
        assert hasattr(default_dataset_handler, "classes")
        
        assert isinstance(default_dataset_handler.dataset, pandas.DataFrame)
        assert isinstance(default_dataset_handler.keys, pandas.Series)
        assert isinstance(default_dataset_handler.classes, list)
        
        # Check that the dataset's index is int64
        assert default_dataset_handler.dataset.index.inferred_type == "integer", "must have an integer index"

        # CHeck that the keys are only ints
        assert all(isinstance(x, int) for x in default_dataset_handler.keys)

        # Check that the classes are all strings
        assert all(isinstance(x, str) for x in default_dataset_handler.classes)
        

    def test_set_unpredicted_keys(self, default_dataset_handler):
        keys = pandas.Series(dtype="object")

        default_dataset_handler.set_unpredicted_keys(keys)
        
        assert default_dataset_handler.unpredicted_keys.equals(keys)

    def test_sanitize_value(self, default_dataset_handler):
        """ Checks that the right values/types are returned """
        # sanitize_value(self, item, column_is_text: bool) -> Union[str, int, float]
        # 1. column_is_text = True
        # 1a. Empty string if the value is None/NoneType
        value = None
        assert default_dataset_handler.sanitize_value(value, True) == ""

        # 1b. Linebreaks & superfluous blanke spaces needs to be removed
        value = "\nthis has\r many strange \n linebreaks and \r stuff"
        expected_value = "this has many strange linebreaks and stuff"
        assert default_dataset_handler.sanitize_value(value, True) == expected_value

        value = "\nthis has\rmany strange\nlinebreaks and\rstuff"
        assert default_dataset_handler.sanitize_value(value, True) == expected_value

        value = "  this  is  a  string  "
        expected_value = "this is a string"
        assert default_dataset_handler.sanitize_value(value, True) == expected_value

        # 1c. A non-string is given to a text column
        value = 14.9
        assert default_dataset_handler.sanitize_value(value, True) == ""

        value = False
        assert default_dataset_handler.sanitize_value(value, True) == ""

        # 1d. Unchanged string value is unchanged
        value = "foo"
        assert default_dataset_handler.sanitize_value(value, True) == "foo"

        # 2. column_is_text = False (numeric column)
        # 2a. Empty string if the value is None/NoneType
        value = None
        assert default_dataset_handler.sanitize_value(value, False) == 0

        # 2b. Datetime should be turned into ordinals
        ordinal = 732289
        date = "2005-12-09"
        # Datetime
        value = datetime.strptime(str(date), "%Y-%m-%d")
        assert default_dataset_handler.sanitize_value(value, False) == ordinal

        # Using string
        assert default_dataset_handler.sanitize_value(date, False) == ordinal

        # 2c. Values that cannot be read as int or float should be 0
        value = "foo"
        assert default_dataset_handler.sanitize_value(value, False) == 0

        # 2d. Numerical values are numerical
        value = "1.75"
        assert default_dataset_handler.sanitize_value(value, False) == 1.75

        value = 10
        assert default_dataset_handler.sanitize_value(value, False) == 10

    def test_validate_dataset(self, default_dataset_handler):
        """ Tests that given data X with given column names, it returns the right dataset """
        
        # This is the absolutely simplest case, with no changes to the data needed
        column_names = default_dataset_handler.handler.config.get_column_names()
        class_column = default_dataset_handler.handler.config.get_class_column_name()
        #column_names =  ["name", "age", "test_class", "test_id"]
        data = [
            ["Karan",23.0, "odd", 1.0],
            ["Rohit",22.0, "even", 2.0],
            ["Sahil",21.0, "odd", 3.0],
            ["Aryan",24.0, "even", 4.0]
        ]
        expected_dataset = pandas.DataFrame(data, columns = column_names)
        dataset = default_dataset_handler.validate_dataset(data, column_names, class_column)

        assert isinstance(dataset, pandas.DataFrame)
        pandas.testing.assert_frame_equal(dataset, expected_dataset)

        data = [
            ["Karan",23, "odd", 1],
            [True, 22, "even", 2],
            ["Sahil",21, "odd", 3],
            ["Aryan",24, "even", 4]
        ]

        cleaned_data = [
            ["Karan",23.0, "odd", 1.0],
            ["", 22.0, "even", 2.0],
            ["Sahil",21.0, "odd", 3.0],
            ["Aryan",24.0, "even", 4.0]
        ]
        expected_dataset = pandas.DataFrame(cleaned_data, columns = column_names)

        dataset = default_dataset_handler.validate_dataset(data, column_names, class_column)
        assert isinstance(dataset, pandas.DataFrame)
        pandas.testing.assert_frame_equal(dataset, expected_dataset)

    def test_validate_dataset_normalizes_known_class_labels_without_losing_unknowns(self, default_dataset_handler):
        column_names = default_dataset_handler.handler.config.get_column_names()
        class_column = default_dataset_handler.handler.config.get_class_column_name()

        rows = [
            ["Karan", 23, 1, 1],
            ["Rohit", 22, 2, 2],
            ["Sahil", 21, None, 3],
            ["Aryan", 24, "", 4],
        ]

        dataset = default_dataset_handler.validate_dataset(rows, column_names, class_column)

        assert dataset[class_column].tolist() == ["1", "2", None, ""]
        assert default_dataset_handler.get_num_unpredicted_rows(dataset) == 2

    def test_concat_with_index(self, default_dataset_handler):
        """ This function takes two dataframes and one int64 index """
        index = pandas.Int64Index(data=[1, 2, 3, 4], dtype="int64", name="test_id")
        
        X = pandas.DataFrame()
        data_concat = [
            [23],
            [22],
            [21],
            [24]
        ]
        concat = pandas.DataFrame(data_concat)

        # 1. Empty first dataframe, it is equal to the 2nd
        expected_dataset = pandas.DataFrame(data_concat)
        expected_dataset.set_index(index, drop=False, append=False, inplace=True)

        concatted = default_dataset_handler.concat_with_index(X, concat, index)
        pandas.testing.assert_frame_equal(concatted, expected_dataset)
        
        # 2. Empty second dataframe, it is equal to the 1st
        expected_dataset = pandas.DataFrame(data_concat)
        expected_dataset.set_index(index, drop=False, append=False, inplace=True)
        
        concatted = default_dataset_handler.concat_with_index(concat, X, index)
        pandas.testing.assert_frame_equal(concatted, expected_dataset)

        # 3. Put X and Concat together to data
        data_X = [
            ["Karan", "odd"],
            ["Rohit", "even"],
            ["Sahil", "odd"],
            ["Aryan", "even"]
        ]
        X3 = pandas.DataFrame(data_X, columns=["name", "test_class"])

        data_concat = [
            [23],
            [22],
            [21],
            [24]
        ]
        concat3 = pandas.DataFrame(data_concat, columns=["age"])

        data = [
            ["Karan", "odd", 23],
            ["Rohit", "even", 22],
            ["Sahil", "odd", 21],
            ["Aryan","even", 24]
        ]

        expected_dataset3 = pandas.DataFrame(data, columns=["name", "test_class", "age"])
        expected_dataset3.set_index(index, drop=False, append=False, inplace=True)
        
        # The concatted dataframes need to share the index
        X3.set_index(index, drop=False, append=False, inplace=True)
        
        concatted3 = default_dataset_handler.concat_with_index(X3, concat3, index)
        pandas.testing.assert_frame_equal(concatted3, expected_dataset3, check_like=True)
        
    def test_create_X(self, default_dataset_handler):
        """ This returns the concat of text, numerical and binary columns. 
            At the moment it requires text data/numerical data to be set as bools, but perhaps rather check lengths?
        """
        index = pandas.Int64Index(data=[1], dtype="int64", name="test_id")

        # 1. All sets have data
        text_set = [
            ["activation"]
        ]
        text = pandas.DataFrame(text_set, columns=["status"])
        numerical_set = [
            [25, 19]
        ]
        numerical = pandas.DataFrame(numerical_set, columns=["age", "code"])
        binary_set = [
            "something"
        ]
        binary = pandas.DataFrame(binary_set, columns=["binarised"])
        
        expected_data = [
            ["activation", 25, 19, "something"]
        ]
        expected_columns = ["status", "age", "code", "binarised"]
        expected_dataframe = pandas.DataFrame(expected_data, columns = expected_columns)
        expected_dataframe.set_index(index, drop=False, append=False, inplace=True)

        actual_dataframe = default_dataset_handler.create_X([text, numerical, binary], index=index)
        pandas.testing.assert_frame_equal(actual_dataframe, expected_dataframe, check_like=True)

        # 2. No numerical data
        expected_data = [
            ["activation", "something"]
        ]
        expected_columns = ["status", "binarised"]
        expected_dataframe = pandas.DataFrame(expected_data, columns = expected_columns)
        expected_dataframe.set_index(index, drop=False, append=False, inplace=True)

        actual_dataframe = default_dataset_handler.create_X([text, binary],  index=index)
        pandas.testing.assert_frame_equal(actual_dataframe, expected_dataframe, check_like=True)

        # 3. No text data
        expected_data = [
            [25, 19, "something"]
        ]
        expected_columns = ["age", "code", "binarised"]
        expected_dataframe = pandas.DataFrame(expected_data, columns = expected_columns)
        expected_dataframe.set_index(index, drop=False, append=False, inplace=True)

        actual_dataframe = default_dataset_handler.create_X([numerical, binary], index=index)
        pandas.testing.assert_frame_equal(actual_dataframe, expected_dataframe, check_like=True)

        # 4. No binary data
        expected_data = [
            ["activation", 25, 19]
        ]
        expected_columns = ["status", "age", "code"]
        expected_dataframe = pandas.DataFrame(expected_data, columns = expected_columns)
        expected_dataframe.set_index(index, drop=False, append=False, inplace=True)

        actual_dataframe = default_dataset_handler.create_X([text, numerical], index=index)
        pandas.testing.assert_frame_equal(actual_dataframe, expected_dataframe, check_like=True)

        # 5. No sets at all
        expected_dataframe = pandas.DataFrame()
        
        actual_dataframe = default_dataset_handler.create_X([], index=index)
        pandas.testing.assert_frame_equal(actual_dataframe, expected_dataframe, check_like=True)
        
    def test_is_categorical_data(self, default_dataset_handler):
        """ 
            1. If either should_train or use_categorization is False, this is False (will be True in these tests)
            2. If either the count is less-or-equal to 30 _or_ the column is stated as is_categorical, this    
        """
        series_longer = pandas.Series(range(1,32))
        series_shorter = pandas.Series(range(1, 5))
        # Because I want to make sure my premises are correct
        assert series_longer.value_counts().count() > 30
        assert series_shorter.value_counts().count() <= 30
        assert default_dataset_handler.handler.config.is_categorical("is_categorical")
        assert not default_dataset_handler.handler.config.is_categorical("is_not_categorical")
        
        # 1. Series longer than 30 (LIMIT_IS_CATEGORICAL) [False], is_categorical [True] = True
        case_1 = series_longer.rename("is_categorical")
        assert default_dataset_handler.is_categorical_data(column=case_1)
        
        # 2. Series longer than 30 (LIMIT_IS_CATEGORICAL) [False], is_categorical [False] = False
        case_2 = series_longer.rename("is_not_categorical")
        assert not default_dataset_handler.is_categorical_data(column=case_2)
        
        # 3. Series shorter than 30 (LIMIT_IS_CATEGORICAL) [True], is_categorical [False] = True
        case_3 = series_shorter.rename("is_not_categorical")
        assert default_dataset_handler.is_categorical_data(column=case_3)

        # 4. Series short than 30 (LIMIT_IS_CATEGORICAL) [True], is_categorical [True] = True
        case_4 = series_shorter.rename("is_categorical")
        assert default_dataset_handler.is_categorical_data(column=case_4)
        
    def test_split_keys_from_dataset(self, default_dataset_handler):
        """ Tests that given data X with given column names, it returns the right dataset and keys """
        # This is the absolutely simplest case, with no changes to the data needed
        column_names = default_dataset_handler.handler.config.get_column_names()
        id_column = default_dataset_handler.handler.config.get_id_column_name()
        # ["name", "age", "test_class", "test_id"]
        data = [
            ["Karan",23, "odd", 1],
            ["Rohit",22, "even", 2],
            ["Sahil",21, "odd", 3],
            ["Aryan",24, "even", 4]
        ]

        expected_keys = set([1, 2, 3, 4])
        input_dataset = pandas.DataFrame(data, columns = column_names)
        
        dataset, keys = default_dataset_handler.split_keys_from_dataset(input_dataset, id_column)

        # First return, the dataset (minus keys)
        assert isinstance(dataset, pandas.DataFrame)
        
        # Second return, the keys (id_column)
        assert isinstance(keys, pandas.Series)
        difference = set(keys) ^ expected_keys
        assert not difference

        # Keys cannot be turned into integers, exception
        data = [
            ["Karan",23, "odd", "foo"],
            ["Rohit",22, "even", 2],
            ["Sahil",21, "odd", 3],
            ["Aryan",24, "even", 4]
        ]

        input_dataset = pandas.DataFrame(data, columns = column_names)

        with pytest.raises(DatasetException):
            default_dataset_handler.split_keys_from_dataset(input_dataset, id_column)

        
    
    def test_get_num_unpredicted_rows(self, default_dataset_handler):
        """ This can be tested with either a given dataset or one in the handler """
        #get_num_unpredicted_rows(self, dataset: pandas.DataFrame = None) -> int:
        # 1. Start by giving a dataset to the function
        column_names = default_dataset_handler.handler.config.get_column_names()
        data = [
            ["Karan",23, "odd", "1"],
            ["Rohit",22, "even", "2"],
            ["Sahil",21, None, "3"],
            ["Aryan",24, "", "4"]
        ]

        df = pandas.DataFrame(data, columns=column_names)
        assert default_dataset_handler.get_num_unpredicted_rows(df) == 2

        # Just making sure our data isn't accidentally saved
        assert not hasattr(default_dataset_handler, "dataset")

        # 2. Same dataset, but this time given to the handler beforehand
        default_dataset_handler.dataset = df
        assert default_dataset_handler.get_num_unpredicted_rows() == 2


class TestModel():
    """ Tests the model class"""

    def test_update_fields(self):
        """ Updates fields based on a list of fields and values returned by a callable """
        model = Model()

        values = self._fake_update_fields_values() 

        model.update_fields({"text_converter"}, self._fake_update_fields_values)

        assert model.text_converter == values[0]

    def _fake_update_fields_values(self, model: Model = None) -> list:
        """ Help function since update_fields takes a callable with the parameter model"""
        return ["item"]

    def test_update_field(self):
        """ Updates a single field (if it exists) """
        model = Model()

        model.update_field("text_converter", {"valid": "dict"})

        assert model.text_converter == {"valid": "dict"}

    def test_get_name(self):
        """ Tests the get name, which returns an empty string or a combination of algorith and preprocessor """

        model = Model()

        assert model.get_name() == "Empty model"


class TestModelHandler():
    """ Tests the Model Handler """

    def test_load_empty_model(self, default_model_handler):
        """ Confirm that the empty model loading works as it should """
        model = Model()

        assert default_model_handler.load_empty_model() == model

    def test_load_model_from_file(self, default_model_handler, default_model):
        """ Loads from model-save.sav """
        # 1. Ensure that None is returned if the path is wrong
        assert default_model_handler.load_model_from_file("does-not-exist") == None

        # 2. This exists, but is a bare .sav without the proper values, so still None
        filename = get_fixture_path() / "config-save.sav"
        assert default_model_handler.load_model_from_file(filename) == default_model
        
        path = get_fixture_path() / "model-save.sav"
        assert default_model_handler.load_model_from_file(path) == default_model

    def test_load_pipeline_from_file(self, default_model_handler):
        """ This is almost identical to the one above, except only returning the Pipeline (here None)"""
        path = get_fixture_path() / "model-save.sav"
        assert default_model_handler.load_pipeline_from_file(path) == None

    @pytest.mark.parametrize("algorithm", [Algorithm.MNB, Algorithm.BNB, Algorithm.CNB])
    def test_rfe_skips_naive_bayes_estimators_without_feature_importances(
        self, default_model_handler, algorithm
    ):
        assert default_model_handler.should_run_computation(Reduction.RFE, algorithm) is False

    def test_preflight_skips_degenerate_binarized_candidates(self, default_model_handler):
        X = pandas.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [5.0, 6.0, 7.0, 8.0],
        })
        scaler = Preprocess.BIN.call_preprocess()

        reason = default_model_handler.get_preflight_skip_reason(
            Preprocess.BIN, scaler, Reduction.NOR, Algorithm.LDA, X
        )

        assert reason == "no feature variance after preprocessing BIN"

    def test_preflight_keeps_dummy_without_reduction_as_baseline(self, default_model_handler):
        X = pandas.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [5.0, 6.0, 7.0, 8.0],
        })
        scaler = Preprocess.BIN.call_preprocess()

        reason = default_model_handler.get_preflight_skip_reason(
            Preprocess.BIN, scaler, Reduction.NOR, Algorithm.DUMY, X
        )

        assert reason is None

    def test_preflight_skips_dummy_when_reduction_would_receive_constant_data(self, default_model_handler):
        X = pandas.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [5.0, 6.0, 7.0, 8.0],
        })
        scaler = Preprocess.BIN.call_preprocess()

        reason = default_model_handler.get_preflight_skip_reason(
            Preprocess.BIN, scaler, Reduction.PCA, Algorithm.DUMY, X
        )

        assert reason == "no feature variance after preprocessing BIN"

    def test_preflight_preserves_sparse_text_input(self, default_model_handler):
        X = pandas.DataFrame({
            "text_token": pandas.arrays.SparseArray([0.0, 1.0, 0.0, 1.0], fill_value=0.0),
            "priority": [1.0, 2.0, 1.0, 2.0],
        })
        scaler = Preprocess.MAX.call_preprocess()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reason = default_model_handler.get_preflight_skip_reason(
                Preprocess.MAX, scaler, Reduction.NOR, Algorithm.LRN, X
            )

        sparse_warnings = [
            warning for warning in caught
            if "pandas.DataFrame with sparse columns found" in str(warning.message)
        ]
        assert reason is None
        assert sparse_warnings == []

    def test_preflight_skips_minmax_for_sparse_text_input(self, default_model_handler):
        X = pandas.DataFrame({
            "text_token": pandas.arrays.SparseArray([0.0, 1.0, 0.0, 1.0], fill_value=0.0),
            "priority": [1.0, 2.0, 1.0, 2.0],
        })
        scaler = Preprocess.MIX.call_preprocess()

        reason = default_model_handler.get_preflight_skip_reason(
            Preprocess.MIX, scaler, Reduction.NOR, Algorithm.DUMY, X
        )

        assert reason == "MinMaxScaler does not support sparse input"

    def test_preflight_allows_informative_binarized_features(self, default_model_handler):
        X = pandas.DataFrame({
            "a": [-1.0, 1.0, -2.0, 2.0],
            "b": [2.0, -2.0, 3.0, -3.0],
        })
        scaler = Preprocess.BIN.call_preprocess()

        reason = default_model_handler.get_preflight_skip_reason(
            Preprocess.BIN, scaler, Reduction.PCA, Algorithm.LDA, X
        )

        assert reason is None

    def test_preflight_skips_lda_when_sparse_input_reaches_estimator(self, default_model_handler):
        X = pandas.DataFrame({
            "text_token": pandas.arrays.SparseArray([0.0, 1.0, 0.0, 1.0], fill_value=0.0),
            "priority": [1.0, 2.0, 1.0, 2.0],
        })
        scaler = Preprocess.MAX.call_preprocess()

        reason = default_model_handler.get_preflight_skip_reason(
            Preprocess.MAX, scaler, Reduction.NOR, Algorithm.LDA, X
        )

        assert reason == "LinearDiscriminantAnalysis requires dense input"

    def test_preflight_allows_lda_after_dense_text_reduction(self, default_model_handler):
        X = pandas.DataFrame({
            "text_token": pandas.arrays.SparseArray([0.0, 1.0, 0.0, 1.0], fill_value=0.0),
            "priority": [1.0, 2.0, 1.0, 2.0],
        })
        scaler = Preprocess.MAX.call_preprocess()

        reason = default_model_handler.get_preflight_skip_reason(
            Preprocess.MAX, scaler, Reduction.TSVD, Algorithm.LDA, X
        )

        assert reason is None

    def test_preflight_skips_fastica_for_sparse_text_input(self, default_model_handler):
        X = pandas.DataFrame({
            "text_token": pandas.arrays.SparseArray([0.0, 1.0, 0.0, 1.0], fill_value=0.0),
            "priority": [1.0, 2.0, 1.0, 2.0],
        })
        scaler = Preprocess.MAX.call_preprocess()

        reason = default_model_handler.get_preflight_skip_reason(
            Preprocess.MAX, scaler, Reduction.FICA, Algorithm.LRN, X
        )

        assert reason == "FastICA requires dense input"

    def test_nystroem_components_are_capped_to_smallest_cv_training_fold(
        self, default_model_handler
    ):
        reducer = Reduction.NYS.get_function(num_samples=160, num_features=641)
        kfold = default_model_handler._create_spot_check_kfold(10)

        capped = default_model_handler._cap_nystroem_components_for_cv(
            feature_reducer=reducer,
            kfold=kfold,
            n_train=128,
        )

        assert reducer.n_components == 160
        assert capped is not reducer
        assert capped.n_components == 115

    def test_fastica_components_are_capped_to_smallest_cv_training_fold(
        self, default_model_handler
    ):
        reducer = Reduction.FICA.get_function(num_samples=160, num_features=641)
        kfold = default_model_handler._create_spot_check_kfold(10)

        capped = default_model_handler._cap_fastica_components_for_cv(
            feature_reducer=reducer,
            kfold=kfold,
            n_train=128,
            n_features=641,
        )

        assert reducer.n_components == 160
        assert capped is not reducer
        assert capped.n_components == 115
        assert capped.max_iter == 1000
        assert capped.random_state == 1

    def test_preflight_skip_is_kept_in_results_without_per_candidate_info_output(
        self, default_model_handler
    ):
        dh = SimpleNamespace(
            X=pandas.DataFrame(np.zeros((8, 4))),
            X_train=pandas.DataFrame(np.zeros((8, 4))),
            X_validation=None,
            Y_validation=None,
        )
        state = _SpotCheckState(best_num_components=4, best_rfe_feature_selection=4)
        info_messages = []

        default_model_handler.get_preflight_skip_reason = \
            lambda *args, **kwargs: "known incompatible combination"
        default_model_handler.handler.logger.print_info = \
            lambda message, *args, **kwargs: info_messages.append(message)

        results, success = default_model_handler._evaluate_spot_check_candidate(
            dh=dh,
            preprocessor=Preprocess.MIX,
            preprocessor_callable=None,
            reduction=Reduction.TSVD,
            reduction_callable=None,
            algorithm=Algorithm.LDA,
            algorithm_callable=None,
            oversampler=None,
            undersampler=None,
            kfold=None,
            state=state,
        )

        assert success is False
        assert results[0][-1] == "SKIPPED: known incompatible combination"
        assert info_messages == []

    def test_spot_check_result_schema(self, default_model_handler):
        result = default_model_handler._build_spot_check_result(
            preprocessor=Preprocess.NOS,
            reduction=Reduction.NOR,
            algorithm=Algorithm.DUMY,
            components=4,
            cv_score=0.8,
            cv_stdev=0.1,
            test_score=0.75,
            elapsed_time=1.25,
            failure="",
        )

        assert result == [
            Preprocess.NOS.name,
            Reduction.NOR.name,
            f"{Algorithm.DUMY.name} - {Algorithm.DUMY.full_name} ({Algorithm.DUMY.lib.full_name})",
            4,
            0.8,
            0.1,
            0.75,
            1.25,
            "",
        ]

    def test_spot_check_kfold_configuration(self, default_model_handler):
        kfold = default_model_handler._create_spot_check_kfold(7)

        assert kfold.n_splits == 7
        assert kfold.shuffle is True
        assert kfold.random_state == 1

    def test_rfe_search_round_limit_scales_with_feature_count(self, default_model_handler):
        assert default_model_handler._get_rfe_search_round_limit(1) == 1
        assert default_model_handler._get_rfe_search_round_limit(30) == 6
        assert default_model_handler._get_rfe_search_round_limit(230) == 9
        assert default_model_handler._get_rfe_search_round_limit(1 << 40) == \
            default_model_handler.MAX_RFE_SEARCH_ROUNDS

    def test_rfe_search_finishes_normal_binary_search_without_limit_warning(
        self, default_model_handler
    ):
        dh = SimpleNamespace(
            X=pandas.DataFrame(np.zeros((8, 230))),
            X_train=pandas.DataFrame(np.zeros((8, 230))),
            X_validation=None,
            Y_validation=None,
        )
        state = _SpotCheckState(
            best_num_components=230,
            best_rfe_feature_selection=230,
        )
        targets = []
        warning_messages = []

        default_model_handler.get_preflight_skip_reason = lambda *args, **kwargs: None
        default_model_handler.create_pipeline_and_cv = lambda *args, **kwargs: (
            targets.append(args[-1]) or object(),
            np.array([0.8]),
            "",
        )
        default_model_handler.get_components_from_pipeline = \
            lambda reduction, pipeline, num_features: num_features
        default_model_handler.handler.logger.print_warning = \
            lambda message, *args, **kwargs: warning_messages.append(message)

        results, success = default_model_handler._evaluate_spot_check_candidate(
            dh=dh,
            preprocessor=Preprocess.STA,
            preprocessor_callable=None,
            reduction=Reduction.RFE,
            reduction_callable=None,
            algorithm=Algorithm.LRN,
            algorithm_callable=None,
            oversampler=None,
            undersampler=None,
            kfold=None,
            state=state,
        )

        assert success is True
        assert targets == [230, 115, 58, 29, 15, 8, 4, 2, 1]
        assert len(results) == 9
        assert warning_messages == []

    def test_rfe_search_uses_transient_progress_and_stops_if_interval_does_not_shrink(
        self, default_model_handler
    ):
        dh = SimpleNamespace(
            X=pandas.DataFrame(np.zeros((8, 230))),
            X_train=pandas.DataFrame(np.zeros((8, 230))),
            X_validation=None,
            Y_validation=None,
        )
        state = _SpotCheckState(
            best_num_components=230,
            best_rfe_feature_selection=230,
        )
        targets = []
        info_messages = []
        progress_messages = []
        warning_messages = []

        default_model_handler.get_preflight_skip_reason = lambda *args, **kwargs: None
        default_model_handler.create_pipeline_and_cv = lambda *args, **kwargs: (
            targets.append(args[-1]) or object(),
            np.array([0.8]),
            "",
        )
        default_model_handler.get_components_from_pipeline = \
            lambda reduction, pipeline, num_features: num_features
        default_model_handler.calculate_current_features = \
            lambda current_score, best_score, num_features, max_features, min_features: (
                best_score, max_features, min_features
            )
        default_model_handler.handler.logger.print_info = \
            lambda message, *args, **kwargs: info_messages.append(message)
        default_model_handler.handler.logger.print_progress = \
            lambda message=None, percent=None, *args, **kwargs: progress_messages.append(message)
        default_model_handler.handler.logger.print_warning = \
            lambda message, *args, **kwargs: warning_messages.append(message)

        results, success = default_model_handler._evaluate_spot_check_candidate(
            dh=dh,
            preprocessor=Preprocess.STA,
            preprocessor_callable=None,
            reduction=Reduction.RFE,
            reduction_callable=None,
            algorithm=Algorithm.LRN,
            algorithm_callable=None,
            oversampler=None,
            undersampler=None,
            kfold=None,
            state=state,
        )

        assert success is True
        assert targets == [230, 115]
        assert len(results) == 2
        assert any(
            "RFE search STA-RFE-LRN: round 2/9, target features 115/230" in message
            for message in progress_messages
        )
        assert not any(
            message.startswith("RFE search ")
            for message in info_messages
        )
        assert any(
            "Stopping RFE search STA-RFE-LRN: search interval did not shrink" in message
            for message in warning_messages
        )

    def test_spot_check_candidate_updates_best_state(self, default_model_handler):
        state = _SpotCheckState(best_num_components=4, best_rfe_feature_selection=4)
        pipeline = object()

        failure, stable = default_model_handler._consider_spot_check_candidate(
            state=state,
            pipeline=pipeline,
            preprocessor=Preprocess.NOS,
            reduction=Reduction.NOR,
            algorithm=Algorithm.DUMY,
            cv_score=0.8,
            cv_stdev=0.1,
            test_score=0.8,
            num_features=3,
            num_components=2,
            failure="",
        )

        assert failure == ""
        assert stable is True
        assert state.trained_pipeline is pipeline
        assert state.best_preprocessor == Preprocess.NOS
        assert state.best_reduction == Reduction.NOR
        assert state.best_algorithm == Algorithm.DUMY
        assert state.best_cv_score == 0.8
        assert state.best_stdev == 0.1
        assert state.best_test_score == 0.8
        assert state.best_rfe_feature_selection == 3
        assert state.best_num_components == 2

    def test_spot_check_candidate_prefers_higher_cv_over_holdout(self, default_model_handler):
        state = _SpotCheckState(best_num_components=4, best_rfe_feature_selection=4)
        first_pipeline = object()
        second_pipeline = object()

        default_model_handler._consider_spot_check_candidate(
            state=state,
            pipeline=first_pipeline,
            preprocessor=Preprocess.STA,
            reduction=Reduction.PCA,
            algorithm=Algorithm.LRN,
            cv_score=0.974178,
            cv_stdev=0.030762,
            test_score=1.0,
            num_features=30,
            num_components=30,
            failure="",
        )

        failure, candidate_success = default_model_handler._consider_spot_check_candidate(
            state=state,
            pipeline=second_pipeline,
            preprocessor=Preprocess.STA,
            reduction=Reduction.RFE,
            algorithm=Algorithm.LRN,
            cv_score=0.978843,
            cv_stdev=0.022639,
            test_score=0.90,
            num_features=22,
            num_components=22,
            failure="",
        )

        assert failure == ""
        assert candidate_success is True
        assert state.trained_pipeline is second_pipeline
        assert state.best_reduction == Reduction.RFE
        assert state.best_cv_score == 0.978843
        assert state.best_test_score == 0.90

    def test_spot_check_selection_uses_cv_stdev_as_tiebreaker(self, default_model_handler):
        assert default_model_handler.is_best_run_yet(0.95, 0.02, 0.95, 0.03) is True
        assert default_model_handler.is_best_run_yet(0.95, 0.04, 0.95, 0.03) is False
        assert default_model_handler.is_best_run_yet(np.nan, 0.01, 0.95, 0.03) is False

    def test_apply_spot_check_state_uses_winning_reduction(self, default_model_handler):
        captured_updates = {}

        def capture_updates(updates: dict, type: str = None) -> None:
            captured_updates.update(updates)

        default_model_handler.handler.config.update_attributes = capture_updates
        default_model_handler.model = Model(
            text_converter=None,
            preprocess=Preprocess.NOS,
            reduction=Reduction.NOR,
            algorithm=Algorithm.DUMY,
            pipeline=None,
            n_features_out=4,
        )
        pipeline = object()
        state = _SpotCheckState(
            best_num_components=2,
            best_rfe_feature_selection=3,
            trained_pipeline=pipeline,
            best_algorithm=Algorithm.LRN,
            best_preprocessor=Preprocess.STA,
            best_reduction=Reduction.PCA,
        )

        best_model = default_model_handler._apply_spot_check_state(state)

        assert captured_updates["feature_selection"] == Reduction.PCA
        assert captured_updates["algorithm"] == Algorithm.LRN
        assert captured_updates["preprocessor"] == Preprocess.STA
        assert captured_updates["num_selected_features"] == 3
        assert best_model.reduction == Reduction.PCA
        assert best_model.algorithm == Algorithm.LRN
        assert best_model.preprocess == Preprocess.STA
        assert best_model.pipeline is pipeline
        assert best_model.n_features_out == 2

    def test_create_pipeline_and_cv_returns_none_pipeline_when_build_fails(self, default_model_handler):
        def fail_build(*args, **kwargs):
            raise RuntimeError("pipeline build failed")

        default_model_handler._build_spot_check_pipeline = fail_build

        pipe, cv_results, exception = default_model_handler.create_pipeline_and_cv(
            reduction=None,
            algorithm=None,
            preprocessor=None,
            feature_reducer=None,
            estimator=None,
            scaler=None,
            oversampler=None,
            undersampler=None,
            kfold=None,
            dh=None,
            num_features=1,
        )

        assert pipe is None
        assert np.isnan(cv_results).all()
        assert exception == "RuntimeError: pipeline build failed"

    def test_execute_n_job_preserves_typeerror_and_context(
        self, monkeypatch, default_model_handler
    ):
        monkeypatch.setattr(handler_module.psutil, "cpu_count", lambda logical=True: 8)
        default_model_handler.handler.STANDARD_DESIRED_N_JOBS = -1

        def fail_with_typeerror(*, n_jobs):
            raise TypeError(f"unsupported input with {n_jobs} workers")

        with pytest.raises(TypeError, match="unsupported input with 3 workers") as exc_info:
            default_model_handler.execute_n_job(
                fail_with_typeerror, n_jobs_desired=3
            )

        notes = getattr(exc_info.value, "__notes__", [])
        assert any("func=fail_with_typeerror" in note for note in notes)
        assert any("n_jobs=3" in note for note in notes)

    def test_execute_n_job_retries_resource_errors_with_fewer_workers(
        self, monkeypatch, default_model_handler
    ):
        monkeypatch.setattr(handler_module.psutil, "cpu_count", lambda logical=True: 8)
        default_model_handler.handler.STANDARD_DESIRED_N_JOBS = -1
        attempts = []

        def flaky(*, n_jobs):
            attempts.append(n_jobs)
            if len(attempts) < 3:
                raise MemoryError("temporary memory pressure")
            return "ok"

        result = default_model_handler.execute_n_job(flaky, n_jobs_desired=8)

        assert result == "ok"
        assert attempts == [8, 4, 2]

    def test_execute_n_job_re_raises_original_pickling_error_at_one_worker(
        self, monkeypatch, default_model_handler
    ):
        monkeypatch.setattr(handler_module.psutil, "cpu_count", lambda logical=True: 2)
        default_model_handler.handler.STANDARD_DESIRED_N_JOBS = -1
        attempts = []

        def never_pickles(*, n_jobs):
            attempts.append(n_jobs)
            raise PicklingError("cannot serialize estimator")

        with pytest.raises(PicklingError, match="cannot serialize estimator"):
            default_model_handler.execute_n_job(never_pickles, n_jobs_desired=2)

        assert attempts == [2, 1]

    def test_execute_n_job_treats_negative_global_limit_as_unlimited(
        self, monkeypatch, default_model_handler
    ):
        monkeypatch.setattr(handler_module.psutil, "cpu_count", lambda logical=True: 8)
        default_model_handler.handler.STANDARD_DESIRED_N_JOBS = -1
        observed = []

        def record_workers(*, n_jobs):
            observed.append(n_jobs)
            return n_jobs

        result = default_model_handler.execute_n_job(
            record_workers, n_jobs_desired=3
        )

        assert result == 3
        assert observed == [3]

    def test_execute_n_job_honors_positive_global_worker_cap(
        self, monkeypatch, default_model_handler
    ):
        monkeypatch.setattr(handler_module.psutil, "cpu_count", lambda logical=True: 8)
        default_model_handler.handler.STANDARD_DESIRED_N_JOBS = 2

        def record_workers(*, n_jobs):
            return n_jobs

        assert default_model_handler.execute_n_job(
            record_workers, n_jobs_desired=6
        ) == 2

    def test_cross_val_score_serial_fallback_uses_sklearn_n_jobs_keyword(
        self, monkeypatch, default_model_handler
    ):
        parallel_attempts = []

        def fail_parallel(*args, **kwargs):
            parallel_attempts.append(kwargs)
            if len(parallel_attempts) == 1:
                raise TypeError("NumPy input not supported")
            raise RuntimeError("parallel execution failed")

        serial_kwargs = {}

        def fake_cross_val_score(*args, **kwargs):
            serial_kwargs.update(kwargs)
            return np.array([0.5, 0.5])

        default_model_handler.execute_n_job = fail_parallel
        monkeypatch.setattr(handler_module, "cross_val_score", fake_cross_val_score)

        dh = type("Dataset", (), {})()
        dh.X_train = pandas.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        dh.Y_train = pandas.Series([0, 1, 0, 1])
        kfold = type("KFold", (), {"get_n_splits": lambda self: 2})()

        cv_results = default_model_handler.get_cross_val_score(
            pipeline=object(),
            dh=dh,
            kfold=kfold,
            algorithm=Algorithm.DUMY,
        )

        assert len(parallel_attempts) == 2
        assert np.array_equal(cv_results, np.array([0.5, 0.5]))
        assert serial_kwargs["n_jobs"] == 1
        assert "n_jobs_desired" not in serial_kwargs

    def test_spot_check_candidate_marks_unstable_without_overwriting_failure(self, default_model_handler):
        state = _SpotCheckState(best_num_components=4, best_rfe_feature_selection=4)

        failure, stable = default_model_handler._consider_spot_check_candidate(
            state=state,
            pipeline=object(),
            preprocessor=Preprocess.NOS,
            reduction=Reduction.NOR,
            algorithm=Algorithm.DUMY,
            cv_score=1.0,
            cv_stdev=0.0,
            test_score=0.9,
            num_features=4,
            num_components=4,
            failure="existing failure",
        )

        assert failure == "existing failure"
        assert stable is False
        assert state.trained_pipeline is None

    def test_spot_check_cv_failure_is_not_overwritten_by_validation(self, default_model_handler):
        dh = SimpleNamespace(
            X=pandas.DataFrame(np.zeros((8, 4))),
            X_train=pandas.DataFrame(np.zeros((8, 4))),
            X_validation=pandas.DataFrame(np.zeros((2, 4))),
            Y_validation=pandas.Series([0, 1]),
        )
        state = _SpotCheckState(best_num_components=4, best_rfe_feature_selection=4)
        validation_calls = []

        default_model_handler.get_preflight_skip_reason = lambda *args, **kwargs: None
        default_model_handler.create_pipeline_and_cv = lambda *args, **kwargs: (
            object(),
            np.array([np.nan]),
            "ValueError: original CV failure",
        )
        default_model_handler.train_and_evaluate_picked_model = \
            lambda *args, **kwargs: validation_calls.append(True)
        default_model_handler.get_components_from_pipeline = \
            lambda reduction, pipeline, num_features: num_features

        results, success = default_model_handler._evaluate_spot_check_candidate(
            dh=dh,
            preprocessor=Preprocess.MAX,
            preprocessor_callable=None,
            reduction=Reduction.PCA,
            reduction_callable=None,
            algorithm=Algorithm.RFCL,
            algorithm_callable=None,
            oversampler=None,
            undersampler=None,
            kfold=None,
            state=state,
        )

        assert success is False
        assert validation_calls == []
        assert results[0][-1] == "ValueError: original CV failure"
        assert state.trained_pipeline is None

    def test_smote_pipeline_normalizes_integer_features_to_float_before_sampling(
        self, default_model_handler
    ):
        pipeline = default_model_handler.get_pipeline(
            reduction=Reduction.NOR,
            feature_reducer=Reduction.NOR.call_reduction(num_samples=12, num_features=2),
            algorithm=Algorithm.MLPC,
            estimator=Algorithm.MLPC.call_algorithm(max_iterations=20, size=12),
            preprocessor=Preprocess.NOS,
            scaler=Preprocess.NOS.call_preprocess(),
            oversampler=Oversampling.SME,
            undersampler=Undersampling.NUG,
            max_features=2,
        )

        step_names = [name for name, _ in pipeline.steps]
        assert step_names[:4] == ["IMP", "FLT", "SME", "NUG"]

        integer_features = np.array([[0, 0], [1, 1], [10, 10], [11, 11]], dtype=np.int64)
        transformed = pipeline.named_steps["FLT"].transform(integer_features)
        assert transformed.dtype == np.float64

    def test_random_oversampling_pipeline_does_not_add_float_normalization(
        self, default_model_handler
    ):
        pipeline = default_model_handler.get_pipeline(
            reduction=Reduction.NOR,
            feature_reducer=Reduction.NOR.call_reduction(num_samples=12, num_features=2),
            algorithm=Algorithm.MLPC,
            estimator=Algorithm.MLPC.call_algorithm(max_iterations=20, size=12),
            preprocessor=Preprocess.NOS,
            scaler=Preprocess.NOS.call_preprocess(),
            oversampler=Oversampling.RND,
            undersampler=Undersampling.NUG,
            max_features=2,
        )

        assert "FLT" not in dict(pipeline.steps)

    def test_validation_fit_falls_back_to_numpy(self, default_model_handler):
        class DataFrameRejectingPipeline:
            def __init__(self):
                self.fit_inputs = []

            def fit(self, X, Y):
                self.fit_inputs.append((X, Y))
                if isinstance(X, pandas.DataFrame):
                    raise TypeError("DataFrame not supported")
                return self

        dh = type("Dataset", (), {})()
        dh.X_train = pandas.DataFrame({"a": [1.0, 2.0]})
        dh.Y_train = pandas.Series([0, 1])
        pipeline = DataFrameRejectingPipeline()

        default_model_handler._fit_pipeline_for_validation(pipeline, dh)

        assert len(pipeline.fit_inputs) == 2
        assert isinstance(pipeline.fit_inputs[0][0], pandas.DataFrame)
        assert isinstance(pipeline.fit_inputs[1][0], np.ndarray)
        assert isinstance(pipeline.fit_inputs[1][1], np.ndarray)

    def test_validation_scorer_falls_back_to_numpy(self, default_model_handler):
        score_inputs = []

        def scorer(pipeline, X, Y):
            score_inputs.append((X, Y))
            if isinstance(X, pandas.DataFrame):
                raise TypeError("DataFrame not supported")
            return 0.75

        default_model_handler.handler.config.get_scoring_mechanism = lambda: scorer
        dh = type("Dataset", (), {})()
        dh.X_validation = pandas.DataFrame({"a": [1.0, 2.0]})
        dh.Y_validation = pandas.Series([0, 1])

        score = default_model_handler._score_validation_pipeline(object(), dh)

        assert score == 0.75
        assert len(score_inputs) == 2
        assert isinstance(score_inputs[0][0], pandas.DataFrame)
        assert isinstance(score_inputs[1][0], np.ndarray)
        assert isinstance(score_inputs[1][1], np.ndarray)

    # Series of functions calling each other
    # train_model calls get_model_from
    # get_model_from calls spot_check_ml_algorithms
    # None of them easy to test, so will postpone


class TestPredictionsHandler:
    """ Tests functions in the predictions handler """

    @pytest.mark.parametrize("correction", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_dark_number_correction_rejects_nonfinite_or_nonpositive_values(
        self, default_predictions_handler, correction
    ):
        with pytest.raises(ValueError, match="Invalid dark-number correction factor"):
            default_predictions_handler._validate_dark_number_correction(correction)

    @pytest.mark.parametrize("correction", [1.0, 1.25, np.float64(2.0)])
    def test_dark_number_correction_accepts_finite_positive_values(
        self, default_predictions_handler, correction
    ):
        result = default_predictions_handler._validate_dark_number_correction(correction)

        assert result == float(correction)
        assert isinstance(result, float)

    def test_dark_numbers_skip_models_without_predict_proba(self, default_predictions_handler):
        class PredictOnlyModel:
            def __init__(self, predictions):
                self.predictions = predictions

            def predict(self, X):
                return np.array(self.predictions)

        class WarningLogger:
            def __init__(self):
                self.warnings = []

            def print_warning(self, message):
                self.warnings.append(message)

        logger = WarningLogger()
        default_predictions_handler.handler.logger = logger
        X = pandas.DataFrame({"feature": [0.0, 1.0, 2.0]})
        Y = pandas.Series(["B", "M", "B"])
        models = [
            PredictOnlyModel(["B", "M", "B"]),
            PredictOnlyModel(["B", "B", "B"]),
        ]

        default_predictions_handler.get_dark_numbers(
            X=X,
            Y=Y,
            models=models,
            model_names=["Cross-trained", "Retrained"],
        )

        assert default_predictions_handler.dark_numbers.empty
        assert not default_predictions_handler.dark_numb_conf_matrix.empty
        assert len(logger.warnings) == 2
        assert all("does not support predict_proba()" in warning for warning in logger.warnings)


    def test_dark_numbers_report_three_estimates_with_model_specific_corrections(
        self, default_predictions_handler, monkeypatch
    ):
        from sklearn.base import BaseEstimator

        class FixedProbabilityModel(BaseEstimator):
            def __init__(self, threshold=2.5):
                self.threshold = threshold

            def predict(self, X):
                values = np.asarray(X)[:, 0]
                return np.where(values > self.threshold, "M", "B")

            def predict_proba(self, X):
                predicted = self.predict(X)
                return np.array([
                    [0.9, 0.1] if value == "B" else [0.1, 0.9]
                    for value in predicted
                ])

        class FakeCorrectionEstimator:
            def __init__(self, estimator, **kwargs):
                self.estimator = estimator
                self.correction_factor_ = None

            def fit(self, X, Y):
                # Cross-trained correction data has four rows; retrained has six.
                self.correction_factor_ = 1.25 if len(Y) == 4 else 1.75
                return self

            def score(self, X=None, Y=None):
                return self.correction_factor_

        class FakeModelHandler:
            def execute_n_job(self, func, *args, n_jobs_desired=None, **kwargs):
                return func(*args, n_jobs=1, **kwargs)

        monkeypatch.setattr(handler_module, "DarkNumberCorrectionFactorEstimator", FakeCorrectionEstimator)
        monkeypatch.setattr(
            default_predictions_handler.handler,
            "get_handler",
            lambda name: FakeModelHandler(),
        )

        X = pandas.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
        Y = pandas.Series(["B", "B", "B", "M", "M", "M"])
        X_cv_training = X.iloc[:4]
        Y_cv_training = Y.iloc[:4]
        X_validation = X.iloc[4:]
        Y_validation = Y.iloc[4:]

        default_predictions_handler.get_dark_numbers(
            X=X,
            Y=Y,
            type="base",
            models=[FixedProbabilityModel(2.5), FixedProbabilityModel(3.5)],
            model_names=["Cross", "Retrained"],
            combine_models=False,
            X_validation=X_validation,
            Y_validation=Y_validation,
            X_cv_training=X_cv_training,
            Y_cv_training=Y_cv_training,
        )

        results = default_predictions_handler.dark_numbers.copy()
        results["Model type"] = results["Model type"].replace("", np.nan).ffill()

        assert set(results["Model type"]) == {
            "D_cv_test - Cross",
            "D_cv_full - Cross",
            "D_retrained_full - Retrained",
        }
        assert set(results.loc[results["Model type"].str.startswith("D_cv_"), "corr"]) == {1.25}
        assert set(results.loc[results["Model type"].str.startswith("D_retrained_"), "corr"]) == {1.75}
        assert set(results["corr_source"]) == {"direct"}

    def test_make_predictions_without_predict_proba_uses_single_warning_and_precision_fallback(
        self, default_predictions_handler
    ):
        class PredictOnlyModel:
            def predict(self, X):
                return np.array([0, 1, 0, 1])

        class WarningLogger:
            def __init__(self):
                self.warnings = []

            def print_warning(self, message):
                self.warnings.append(message)

        logger = WarningLogger()
        default_predictions_handler.handler.logger = logger
        X = pandas.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
        Y = pandas.Series([0, 1, 0, 0])
        classes = pandas.Series([0, 1])

        could_predict_proba = default_predictions_handler.make_predictions(
            PredictOnlyModel(), X=X, classes=classes, Y=Y
        )

        assert could_predict_proba is False
        assert default_predictions_handler.probabilites == [1.0, 0.5, 1.0, 0.5]
        assert len(logger.warnings) == 1
        assert "does not support predict_proba()" in logger.warnings[0]

    def test_get_prediction_results(self, default_predictions_handler):
        """ Two cases: An appropriate list or an empty list """

        keys = pandas.Series(data=[1, 2, 3])
        # 1. This is currently empty, so will get an AttributeError, which returns []
        assert default_predictions_handler.get_prediction_results(keys) == []

        # 2. Need to set up so that the handler has probabilities, predictions and rates
        default_predictions_handler.predictions = np.array(["pred1", "pred2", "pred3"])
        default_predictions_handler.rates = np.array([1.0, 1.0, 1.0])
        default_predictions_handler.probabilites = np.array([["prob1a", "prob1b"], ["prob2a", "prob2b"], ["prob3a", "prob3b"]])

        expected_list = [
            {
                "key": 1,
                "prediction": "pred1",
                "rate": 1.0,
                "probabilities": "prob1a,prob1b"
            },
            {
                "key": 2,
                "prediction": "pred2",
                "rate": 1.0,
                "probabilities": "prob2a,prob2b"
            },
            {
                "key": 3,
                "prediction": "pred3",
                "rate": 1.0,
                "probabilities": "prob3a,prob3b"
            }
        ]

        assert default_predictions_handler.get_prediction_results(keys) == expected_list

    # Again, somewhat too complicated to test, so will postpone