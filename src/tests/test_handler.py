from datetime import datetime
import numpy as np
import pandas
import pytest
from conftest import get_fixture_path

from JBGExceptions import DatasetException, HandlerException

from JBGHandler import JBGHandler, DatasetHandler, Model

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
        expected_dataset.set_index(index, drop=False, append=False, inplace=True, verify_integrity=False)

        concatted = default_dataset_handler.concat_with_index(X, concat, index)
        pandas.testing.assert_frame_equal(concatted, expected_dataset)
        
        # 2. Empty second dataframe, it is equal to the 1st
        expected_dataset = pandas.DataFrame(data_concat)
        expected_dataset.set_index(index, drop=False, append=False, inplace=True, verify_integrity=False)
        
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
        expected_dataset3.set_index(index, drop=False, append=False, inplace=True, verify_integrity=False)
        
        # The concatted dataframes need to share the index
        X3.set_index(index, drop=False, append=False, inplace=True, verify_integrity=False)
        
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
        expected_dataframe.set_index(index, drop=False, append=False, inplace=True, verify_integrity=False)

        actual_dataframe = default_dataset_handler.create_X([text, numerical, binary], index=index)
        pandas.testing.assert_frame_equal(actual_dataframe, expected_dataframe, check_like=True)

        # 2. No numerical data
        expected_data = [
            ["activation", "something"]
        ]
        expected_columns = ["status", "binarised"]
        expected_dataframe = pandas.DataFrame(expected_data, columns = expected_columns)
        expected_dataframe.set_index(index, drop=False, append=False, inplace=True, verify_integrity=False)

        actual_dataframe = default_dataset_handler.create_X([text, binary],  index=index)
        pandas.testing.assert_frame_equal(actual_dataframe, expected_dataframe, check_like=True)

        # 3. No text data
        expected_data = [
            [25, 19, "something"]
        ]
        expected_columns = ["age", "code", "binarised"]
        expected_dataframe = pandas.DataFrame(expected_data, columns = expected_columns)
        expected_dataframe.set_index(index, drop=False, append=False, inplace=True, verify_integrity=False)

        actual_dataframe = default_dataset_handler.create_X([numerical, binary], index=index)
        pandas.testing.assert_frame_equal(actual_dataframe, expected_dataframe, check_like=True)

        # 4. No binary data
        expected_data = [
            ["activation", 25, 19]
        ]
        expected_columns = ["status", "age", "code"]
        expected_dataframe = pandas.DataFrame(expected_data, columns = expected_columns)
        expected_dataframe.set_index(index, drop=False, append=False, inplace=True, verify_integrity=False)

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

    # Series of functions calling each other
    # train_model calls get_model_from
    # get_model_from calls spot_check_ml_algorithms
    # None of them easy to test, so will postpone


class TestPredictionsHandler:
    """ Tests functions in the predictions handler """

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