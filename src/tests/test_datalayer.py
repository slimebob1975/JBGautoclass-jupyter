import pytest

import DataLayer

from Config import Config, Algorithm, Preprocess, Reduction, Scoretype

class MockLogger():
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""
    def print_info(self, *args) -> None:
        """printing info"""

    def print_progress(self, message: str = None, percent: float = None) -> None:
        """Printing progress"""

    def print_components(self, component, components, exception = None) -> None:
        """ Printing Reduction components"""

    def print_formatted_info(self, message: str) -> None:
        """ Printing info with """

    def investigate_dataset(self, dataset, show_class_distribution: bool = True, show_statistics: bool = True) -> bool:
        """ Print information about dataset """
    
    def print_warning(self, *args) -> None:
        """ print warning """

    def print_table_row(self, items: list[str], divisor: str = None) -> None:
        """ Prints row of items, with optional divisor"""

    def abort_cleanly(self, message: str) -> None:
        """ Exits the process """

    def print_percentage_checked(self, text: str, old_percent, percent_checked) -> None:
        """ Prints using the normal print() """

@pytest.fixture
def valid_iris_config() -> Config:
    config = Config(
        Config.Connection(
            odbc_driver="Mock Server",
            host="tcp:database.iaf.mock",
            trusted_connection=True,
            class_catalog="DatabaseOne",
            class_table="ResultTable",
            class_table_script="createtable.sql.txt",
            class_username="some_fake_name",
            class_password="",
            data_catalog="DatabaseTwo",
            data_table="InputTable",
            class_column="class",
            data_text_columns=["data_text_column"],
            data_numerical_columns=["sepal-length","sepal-width","petal-length", "petal-width"],
            id_column="id",
            data_username="some_fake_name",
            data_password=""
        ),
        Config.Mode(
            train=False,
            predict=True,
            mispredicted=False,
            use_stop_words=False,
            specific_stop_words_threshold=1.0,
            hex_encode=True,
            use_categorization=True,
            category_text_columns=["category_text_column"],
            test_size=0.2,
            smote=False,
            undersample=False,
            algorithm=Algorithm.LDA,
            preprocessor=Preprocess.STA,
            feature_selection=Reduction.PCA,
            num_selected_features=None,
            scoring=Scoretype.accuracy,
            max_iterations=20000
        ),
        Config.IO(
            verbose=True,
            model_path="./model",
            model_name="test"
        ),
        Config.Debug(
            on=True,
            num_rows=150
        ),
        name="test",
        filename="autoclassconfig_test_some_fake_name.py"
    )

    return config

@pytest.fixture
def default_datalayer(valid_iris_config) -> DataLayer.DataLayer:
    return DataLayer.DataLayer(valid_iris_config.connection, MockLogger())


# From GUI--perhaps instead test_connection()
#try:
#    self.data_layer.get_sql_connection() # This checks so that the SQL connection works
#except Exception as ex:
#    sys.exit("GUI class could not connect to SQL Server: {0}".format(str(ex)))

class TestDataLayer():
    """ The main class """

    def test_classified_data_query(self, default_datalayer) -> None:
        """ This is a query in string format """
        query = default_datalayer.get_sql_command_for_recently_classified_data(10)
        expectedQuery = "SELECT TOP(10),A.[id],A.[sepal-length],A.[sepal-width],A.[petal-length],A.[petal-width],A.[data_text_column],B.[class_result],B.[class_rate],B.[class_time],B.[class_algorithm] FROM [DatabaseTwo].[InputTable] A  INNER JOIN [DatabaseOne].[ResultTable] B  ON A.[id] = B.[unique_key] WHERE B.[class_user] = 'Mht0202' AND B.[table_name] = 'InputTable'' ORDER BY B.[class_time] DESC"
        
        assert query == expectedQuery