from collections import OrderedDict
import numpy as np
import pytest

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
        expected_con_str = "DRIVER=Mock Server;SERVER=tcp:database.jbg.mock;DATABASE=DatabaseOne;TRUSTED_CONNECTION=yes"
        
        # While this will also hit an error if the JBGSqlHelper changes it's str-method
        # that isn't necessarily a bug
        assert str(con) == expected_con_str


    def test_run_query(self, default_sqldatalayer) -> None:
        """ This is a query in string format """
        query = default_sqldatalayer.get_run_query(run_id=10)
        
        expectedQuery = "SELECT A.[id], A.[sepal-length], A.[sepal-width], A.[petal-length], A.[petal-width], RR.[class_result], RR.[class_rate], RH.[class_time], RH.[class_algorithm] FROM [DatabaseTwo].[InputTable] A INNER JOIN [DatabaseOne].[ResultTableRow] RR ON A.[id] = RR.[unique_key] INNER JOIN [DatabaseOne].[ResultTableHeader] RH ON RR.[run_id] = RH.[run_id] WHERE RH.[run_id] = 10 ORDER BY A.[id]"

        assert query == expectedQuery

    def test_unique_int_id_columns_filter_non_unique_and_nullable_candidates(
        self, default_sqldatalayer, monkeypatch
    ) -> None:
        queries = []

        def get_data(query):
            queries.append(query)
            # total; id nonnull/distinct; priority nonnull/distinct; nullable nonnull/distinct
            return [(160, 160, 160, 160, 3, 159, 159)]

        monkeypatch.setattr(default_sqldatalayer, "get_data_list_from_query", get_data)

        result = default_sqldatalayer.get_unique_int_id_columns(
            "DatabaseTwo", "InputTable", ["id", "priority", "nullable_key"]
        )

        assert result == ["id"]
        assert len(queries) == 1
        assert "COUNT(DISTINCT [id])" in queries[0]
        assert "COUNT(DISTINCT [priority])" in queries[0]
        assert "COUNT(DISTINCT [nullable_key])" in queries[0]

    def test_run_query_uses_configured_unique_id_in_join(self, default_sqldatalayer) -> None:
        default_sqldatalayer.config.connection.id_column = "case_key"
        default_sqldatalayer.config.connection.data_numerical_columns = ["sepal-length"]

        query = default_sqldatalayer.get_run_query(run_id=10)

        assert "A.[case_key] = RR.[unique_key]" in query
        assert "ORDER BY A.[case_key]" in query
        assert "A.[id] = RR.[unique_key]" not in query

    def test_class_distribution_query(self, default_sqldatalayer) -> None:
        """ This is a query in string format """
        query = default_sqldatalayer.get_class_distribution_query()

        expectedQuery = "SELECT [class], COUNT(*) FROM [DatabaseTwo].[InputTable] GROUP BY [class] ORDER BY [class] DESC"

        assert query == expectedQuery

    def test_data_query(self, default_sqldatalayer) -> None:
        """ This is a query in string format """
        num_rows = 15

        # Per default, this is ! train && predict
        query = default_sqldatalayer.get_data_query(num_rows)
        
        expectedQuery = "SELECT TOP(15) [sepal-length],[sepal-width],[petal-length],[petal-width],[id],[class] FROM (SELECT TOP(15) [sepal-length],[sepal-width],[petal-length],[petal-width],[id],[class] FROM [DatabaseTwo].[InputTable] WHERE [class] IS NULL OR CAST([class] AS VARCHAR) = '' ORDER BY NEWID()) A"
        assert query == expectedQuery

        # Case 2: train && ! predict
        default_sqldatalayer.config.mode.train = True
        default_sqldatalayer.config.mode.predict = False

        query = default_sqldatalayer.get_data_query(num_rows)
        expectedQuery = "SELECT TOP(15) [sepal-length],[sepal-width],[petal-length],[petal-width],[id],[class] FROM (SELECT TOP(15) [sepal-length],[sepal-width],[petal-length],[petal-width],[id],[class] FROM [DatabaseTwo].[InputTable] WHERE [class] IS NOT NULL AND CAST([class] AS VARCHAR) != '' ORDER BY NEWID()) A"
        assert query == expectedQuery        

        # Case 3: train and predict
        default_sqldatalayer.config.mode.predict = True

        query = default_sqldatalayer.get_data_query(num_rows)
        expectedQuery = "SELECT TOP(15) [sepal-length],[sepal-width],[petal-length],[petal-width],[id],[class] FROM (SELECT TOP(15) [sepal-length],[sepal-width],[petal-length],[petal-width],[id],[class] FROM [DatabaseTwo].[InputTable] ORDER BY NEWID()) A"
        assert query == expectedQuery

        # Case 4: ! (train and predict), same expected as Case 3
        default_sqldatalayer.config.mode.train = False
        default_sqldatalayer.config.mode.predict = False

        query = default_sqldatalayer.get_data_query(num_rows)
        
        assert query == expectedQuery 


    def test_count_rows_available_for_mode(self, default_sqldatalayer) -> None:
        data_dist = {"setosa": 50, "versicolor": 50, "virginica": 35, "NULL": 15}

        counter = default_sqldatalayer._count_rows_available_for_mode
        assert counter(data_dist, train=True, predict=False) == 135
        assert counter(data_dist, train=False, predict=True) == 15
        assert counter(data_dist, train=True, predict=True) == 150
        assert counter(data_dist, train=False, predict=False) == 150

    def test_prediction_only_data_limit_is_an_upper_bound(self, default_sqldatalayer, monkeypatch) -> None:
        class FakeSqlHelper:
            def __init__(self):
                self.queries = []

            def execute_query(self, query, get_data=False):
                self.queries.append(query)
                return True

            def read_next(self, chunksize=None):
                return []

            def disconnect(self):
                return None

        fake_sql = FakeSqlHelper()
        default_sqldatalayer.config.mode.train = False
        default_sqldatalayer.config.mode.predict = True
        monkeypatch.setattr(
            default_sqldatalayer,
            "count_class_distribution",
            lambda: {"setosa": 50, "versicolor": 50, "virginica": 35, "NULL": 15},
        )
        monkeypatch.setattr(default_sqldatalayer, "get_connection", lambda: fake_sql)
        monkeypatch.setattr(
            default_sqldatalayer,
            "parse_dataset",
            lambda num_rows, use_chunks, read_data_func: np.arange(15).reshape(15, 1),
        )

        data = default_sqldatalayer.get_dataset(num_rows=150)

        assert len(data) == 15
        assert len(fake_sql.queries) == 1
        assert "TOP(15)" in fake_sql.queries[0]

    def test_mispredicted(self, default_sqldatalayer) -> None:
        """ This is a query in string format """

        query = default_sqldatalayer.get_mispredicted_query("new_class", 1)
        
        expectedQuery = "UPDATE [DatabaseTwo].[InputTable] SET class = 'new_class' WHERE id = 1"
        
        assert query == expectedQuery


def test_get_table_columns_queries_explicit_data_catalog(default_sqldatalayer, monkeypatch) -> None:
    captured = {}

    def fake_query(query):
        captured["query"] = query
        return [("id", "int"), ("feature", "float"), ("class", "varchar")]

    monkeypatch.setattr(default_sqldatalayer, "get_data_list_from_query", fake_query)

    columns = default_sqldatalayer.get_table_columns("OtherDatabase", "dbo.iris")

    assert columns == {"id": "int", "feature": "float", "class": "varchar"}
    assert "FROM [OtherDatabase].INFORMATION_SCHEMA.COLUMNS" in captured["query"]
    assert "TABLE_CATALOG = 'OtherDatabase'" in captured["query"]
    assert "CONCAT(CONCAT(TABLE_SCHEMA,'.'),TABLE_NAME) = 'dbo.iris'" in captured["query"]
