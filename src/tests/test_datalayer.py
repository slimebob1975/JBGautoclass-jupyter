from collections import OrderedDict
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

    def test_mispredicted(self, default_sqldatalayer) -> None:
        """ This is a query in string format """

        query = default_sqldatalayer.get_mispredicted_query("new_class", 1)
        
        expectedQuery = "UPDATE [DatabaseTwo].[InputTable] SET class = 'new_class' WHERE id = 1"
        
        assert query == expectedQuery
