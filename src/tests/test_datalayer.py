from collections import OrderedDict
from pathlib import Path
import pytest
import SQLDataLayer

from Config import Config, Algorithm, Preprocess, Reduction, ScoreMetric


class MockLogger():
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""

    def print_info(self, *args) -> None:
        """printing info"""

    def print_progress(self, message: str = None, percent: float = None) -> None:
        """Printing progress"""

    def print_components(self, component, components, exception=None) -> None:
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
            class_catalog="Schema.DatabaseOne",
            class_table="ResultTable",
            class_table_script="./sql/autoClassCreateTable.sql.txt",
            class_username="some_fake_name",
            class_password="",
            data_catalog="DatabaseTwo",
            data_table="InputTable",
            class_column="class",
            data_text_columns=["data_text_column"],
            data_numerical_columns=[
                "sepal-length", "sepal-width", "petal-length", "petal-width"],
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
            scoring=ScoreMetric.accuracy,
            max_iterations=20000
        ),
        Config.IO(
            verbose=True,
            model_path="./model",
            model_name="test"
        ),
        Config.Debug(
            on=True,
            data_limit=150
        ),
        name="test",
        _filename="autoclassconfig_test_some_fake_name.py"
    )

    return config


@pytest.fixture
def default_sqldatalayer(valid_iris_config) -> SQLDataLayer.DataLayer:
    return SQLDataLayer.DataLayer(config=valid_iris_config, logger=MockLogger(), validate=False)
    

    


class TestDataLayer():
    """ The main class """

    def test_validate_parameters(self, default_sqldatalayer) -> None:
        """ Test that the validation works, since we told it to ignore it in creation """
        
        default_sqldatalayer.validate = True
        with pytest.raises(ValueError) as e:
            default_sqldatalayer.validate_parameters()
            assert "Given ODBC driver (Mock Server) cannot be found" in str(e.value)

        # Now we need to change the connection's odbc_driver to SQL Server, but the connection will still fail
        default_sqldatalayer.config.update_attribute("connection.odbc_driver", "SQL Server")
        
        with pytest.raises(ValueError) as e:
            default_sqldatalayer.validate_parameters()
            assert "Connection to server failed" in str(e.value)

    def test_get_connection(self, default_sqldatalayer) -> None:
        con = default_sqldatalayer.get_connection()
        expected_con_str = "DRIVER=Mock Server;SERVER=tcp:database.iaf.mock;DATABASE=Schema.DatabaseOne;TRUSTED_CONNECTION=yes"
        
        # While this will also hit an error if the IAFSqlHelper changes it's str-method
        # that isn't necessarily a bug
        assert str(con) == expected_con_str


    def test_classification_table_query(self, default_sqldatalayer) -> None:
        query = default_sqldatalayer.create_classification_table_query()
        expected_query = [
            "IF OBJECT_ID('[Schema].[DatabaseOne].[ResultTable]','U') IS NULL",
            'CREATE TABLE [Schema].[DatabaseOne].[ResultTable](', 'class_id INTEGER IDENTITY(1,1) PRIMARY KEY,     /* The primary key */', 'catalog_name VARCHAR(255) NOT NULL,             /* Database namne */', 'table_name VARCHAR(255) NOT NULL,               /* Table name */', 'column_names NVARCHAR(1024) NOT NULL,             /* Column names, CSV-style */', 'unique_key BIGINT NOT NULL,                     /* Unique key for classified row in table_name*/', 'class_result VARCHAR(255) NOT NULL,              /* Classification result as string */', 'class_rate FLOAT,                               /* Estimated correctness of rate */', "class_rate_type CHAR DEFAULT 'U',               /* How rate was calculcated: (U)nknown, (A)verage or (I)ndividual */",
            'class_labels VARCHAR(255) NOT NULL,\t\t\t\t/* The possible class labels for the classification */',
            'class_probabilities VARCHAR(255) NOT NULL,      /* The different probabilities corresponding to class labels, CSV-style */', 'class_algorithm VARCHAR(255),                   /* Name of classification algorithm */', 'class_time DATETIME DEFAULT CURRENT_TIMESTAMP,  /* A timestamp when the record was inserted */', 'class_script VARCHAR(255),                      /* The full path to the classification program */', 'class_user VARCHAR(255),                        /* Who executed the classification program */', '', 'CONSTRAINT Check_catalog_name CHECK (DB_ID(catalog_name) IS NOT NULL),', "CONSTRAINT Check_table_name CHECK (OBJECT_ID(CONCAT(catalog_name,'.',table_name)) IS NOT NULL),",
            'CONSTRAINT Check_class_rate CHECK ((class_rate >= 0.0 AND class_rate <= 1.0) OR class_rate = -1.0), /* -1 means N/A */', "CONSTRAINT Check_class_rate_type CHECK ( CHARINDEX(class_rate_type, 'UAI') > 0 ),",
            ');'
        ]
        #print(query)
        assert query == expected_query

    def test_classified_data_query(self, default_sqldatalayer) -> None:
        """ This is a query in string format """
        query = default_sqldatalayer.get_sql_command_for_recently_classified_data(
            10)
        expectedQuery = "SELECT TOP(10),A.[id],A.[data_text_column],A.[sepal-length],A.[sepal-width],A.[petal-length],A.[petal-width],B.[class_result],B.[class_rate],B.[class_time],B.[class_algorithm] FROM [DatabaseTwo].[InputTable] A INNER JOIN [Schema].[DatabaseOne].[ResultTable] B ON A.[id] = B.[unique_key] WHERE B.[class_user] = 'Mht0202' AND B.[table_name] = 'InputTable' ORDER BY B.[class_time] DESC"

        assert query == expectedQuery

    def test_class_distribution_query(self, default_sqldatalayer) -> None:
        """ This is a query in string format """
        query = default_sqldatalayer.get_class_distribution_query()

        expectedQuery = "SELECT [class], COUNT(*) FROM [DatabaseTwo].[InputTable] GROUP BY [class] ORDER BY [class] DESC"

        assert query == expectedQuery

    def test_data_query(self, default_sqldatalayer) -> None:
        """ This is a query in string format """
        #self.config.get_max_limit(), self.config.should_train(), self.config.should_predict()
        num_rows = 15

        # Per default, this is ! train && predict
        query = default_sqldatalayer.get_data_query(num_rows)
        expectedQuery = "SELECT TOP(15) [data_text_column],[sepal-length],[sepal-width],[petal-length],[petal-width],[id],[class] FROM (SELECT TOP(15) [data_text_column],[sepal-length],[sepal-width],[petal-length],[petal-width],[id],[class] FROM [DatabaseTwo].[InputTable] WHERE [class] IS NULL OR CAST([class] AS VARCHAR) = '' ORDER BY NEWID()) A"
        assert query == expectedQuery

        # Case 2: train && ! predict
        default_sqldatalayer.config.mode.train = True
        default_sqldatalayer.config.mode.predict = False

        query = default_sqldatalayer.get_data_query(num_rows)
        expectedQuery = "SELECT TOP(15) [data_text_column],[sepal-length],[sepal-width],[petal-length],[petal-width],[id],[class] FROM (SELECT TOP(15) [data_text_column],[sepal-length],[sepal-width],[petal-length],[petal-width],[id],[class] FROM [DatabaseTwo].[InputTable] WHERE [class] IS NOT NULL AND CAST([class] AS VARCHAR) != '' ORDER BY NEWID()) A"
        assert query == expectedQuery        

        # Case 3: train and predict
        default_sqldatalayer.config.mode.predict = True

        query = default_sqldatalayer.get_data_query(num_rows)
        expectedQuery = "SELECT TOP(15) [data_text_column],[sepal-length],[sepal-width],[petal-length],[petal-width],[id],[class] FROM (SELECT TOP(15) [data_text_column],[sepal-length],[sepal-width],[petal-length],[petal-width],[id],[class] FROM [DatabaseTwo].[InputTable] ORDER BY NEWID()) A"
        assert query == expectedQuery

        # Case 4: ! (train and predict), same expected as Case 3
        default_sqldatalayer.config.mode.train = False
        default_sqldatalayer.config.mode.predict = False

        query = default_sqldatalayer.get_data_query(num_rows)
        
        assert query == expectedQuery 

    def test_save_data(self, default_sqldatalayer) -> None:
        """ This is a query in string format """
        # TODO: fix test_handler/test_save_classification_data
        columns = OrderedDict({
            "catalog_name": default_sqldatalayer.config.get_data_catalog(),
            "table_name": default_sqldatalayer.config.get_data_table(),
            "column_names": ",".join(default_sqldatalayer.config.get_data_column_names()), 
            "unique_key": "", # dict["key"]
            "class_result": "", # dict["prediction"]
            "class_rate": "", # dict["rate"]
            "class_rate_type": "A name",
            "class_labels": ["a", "b"],
            "class_probabilities": "", # dict["probabilities"]
            "class_algorithm": "ALGO-PRE",
            "class_script": "//script/path",
            "class_user": default_sqldatalayer.config.get_data_username()
        })
        
        query = default_sqldatalayer.get_save_classification_insert(columns)
        
        expectedQuery = "INSERT INTO [Schema].[DatabaseOne].[ResultTable] (catalog_name,table_name,column_names,unique_key,class_result,class_rate,class_rate_type,class_labels,class_probabilities,class_algorithm,class_script,class_user)"

        assert query == expectedQuery

    
    def test_mispredicted(self, default_sqldatalayer) -> None:
        """ This is a query in string format """

        query = default_sqldatalayer.get_mispredicted_query("new_class", 1)
        
        expectedQuery = "UPDATE [DatabaseTwo].[InputTable] SET class = 'new_class' WHERE id = 1"
        
        assert query == expectedQuery
