
from dataclasses import dataclass, field
from pathlib import Path
import sys
from collections import OrderedDict
from typing import Protocol
import numpy as np
import pyodbc
from Config import RateType, Config as Cf, to_quoted_string, period_to_brackets
from JBGHandler import Model
import SqlHelper.JBGSqlHelper as sql
from JBGExceptions import DataLayerException
from DataLayer.Base import DataLayerBase

class Logger(Protocol):
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""
    def print_info(self, *args) -> None:
        """printing info"""

    def print_warning(self, *args) -> None:
        """ print warning """

    def print_query(self, type: str, query: str) -> None:
        """ Prints a query """

    def print_dragon(self, exception: Exception) -> None:
        """ Type of Unhandled Exceptions, to handle them for the future """

class Config(Protocol):
    # Methods to hide implementation of Config
    def get_attribute(self, attribute: str):
        """ Gets an attribute from a attribute.subattribute string """
    
    def get_class_catalog_params(self) -> dict:
        """ Gets params to connect to class database """

    def get_classification_script_path(self) -> Path:
        """ Gives a calculated path based on config"""

    

@dataclass
class DataLayer(DataLayerBase):
    SQL_CHUNKSIZE = 1000 #TODO: Decide how many
    SQL_USE_CHUNKS = True

    config: Config
    logger: Logger
    scriptpath: str = field(init=False)

    def __post_init__(self) -> None:
        self.validate_parameters()
    

    def validate_parameters(self) -> None:
        """ Validates the DataLayer in whichever way is needed for the specific one """
        if not self.validate:
            return
        
        """ Checks that the ODBC-driver is valid """
        drivers = pyodbc.drivers()
        
        driver = self.config.get_attribute("connection.odbc_driver")
        if driver not in drivers:
            raise ValueError(f"Given ODBC driver ({driver}) cannot be found")

        """ Checks that the server connection is valid """
        if not self.can_connect():
            raise ValueError(f"Connection to server failed")
        
    # TODO: This *explicitly* only uses class_catalog_params--it never connects to the data_catalog
    def get_connection(self) -> sql.JBGSqlHelper:
        """ Get an odbc-based sql handler """
        return sql.JBGSqlHelper(**self.config.get_class_catalog_params())
        # TODO: Should work if needed 
        # return sql.JBGSqlHelper(**self.config.get_data_catalog_params())

    def get_data_list_from_query(self, query: str) -> list:
        """ Gets a list from the data source, based on the query """
        return self.get_connection().get_data_from_query(query)

    def get_gui_list(self, type: str, list_format: str, first_empty: bool = True) -> list:
        """ For queries used in the GUI """

        available_queries = {
            "databases": "SELECT name FROM sys.databases ORDER BY name",
            "tables": "SELECT TABLE_SCHEMA,TABLE_NAME FROM " + \
                self.config.get_data_catalog_params()["catalog"] + \
                ".INFORMATION_SCHEMA.TABLES ORDER BY TABLE_SCHEMA,TABLE_NAME"
        }

        if type in available_queries:
            data = self.get_data_list_from_query(available_queries[type])
            formatted = [list_format.format(*x) for x in data]

            if first_empty:
                formatted.insert(0, "")
            
            return formatted

        raise DataLayerException(f"Defined query type {type} does not exist")

    def get_catalogs_as_options(self) -> list:
        """ Used in the GUI, to get the databases (only return those we can access without exceptions) """

        catalogs = self.get_gui_list("databases", "{}")
        default_catalog = self.config.connection.data_catalog
        for catalog in list(catalogs): 
            if catalog != "":
                self.config.connection.data_catalog = catalog
                try:
                    _ = self.get_gui_list("tables", "{}.{}")
                except Exception:
                    catalogs.remove(catalog)
        self.config.connection.data_catalog = default_catalog
        
        return catalogs

    def get_tables_as_options(self) -> list:
        """ Used in the GUI, to get the tables """
        return self.get_gui_list("tables", "{}.{}")

    def get_id_columns(self, database: str, table: str) -> list:
        """ Gets name and type for columns in specified <database> and <table> """

        database = to_quoted_string(database)
        table = to_quoted_string(table)
        select = "COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS"
        where = f"TABLE_CATALOG = {database} AND CONCAT(CONCAT(TABLE_SCHEMA,'.'),TABLE_NAME) = {table}"
        order_by = "DATA_TYPE DESC"
        query = f"SELECT {select} WHERE {where} ORDER BY {order_by}"
        
        data = self.get_data_list_from_query(query)

        if (len(data) < 3):
            raise ValueError(f"Selected data table has too few columns ({len(data)}) . Minimum is three (3).")
        
        return {column[0]:column[1] for column in data}
    
    def prepare_for_classification(self) -> bool:
        """ Setting up tables or similar things in preparation for the classifictions """
        query = "\n".join(self.create_classification_table_query())
        self.create_classification_table(query)

    def create_classification_table(self, query: str) -> None:
        """ Create the classification table """
        self.logger.print_progress(message="Create the classification table")
        
        
        # Get a sql handler and connect to data database
        sqlHelper = self.get_connection()

        # Execute query
        sqlHelper.execute_query(query, get_data=False, commit = True)

        sqlHelper.disconnect(commit=False)

    def create_classification_table_query(self, path: str = None) -> list:
        """ Creates a table query from a given text file """
        path = path if path else self.config.get_classification_script_path()
        
        if not path.is_file():
            raise DataLayerException(f"File {path} does not exist.")
        
        query_list = []
        class_catalog = self.config.get_connection().get_formatted_class_catalog()
        class_table = self.config.get_connection().get_formatted_class_table(include_database=False)
        with open(path, mode="rt") as f:
            for line in f:
                transformed_line = line.strip().replace("<class_catalog>", class_catalog)
                transformed_line = transformed_line.replace("<class_table>", class_table)
                
                query_list.append(transformed_line)
        
        return query_list

    def get_sql_command_for_recently_classified_data(self, num_rows: int) -> str:
        """ Produces an SQL command for fetching the recently classified data elements """
        connection = self.config.get_connection()
        id_column = self.config.get_id_column_name()
        class_column = self.config.get_class_column_name()
        
        dataCols = [
            id_column
        ] + self.config.get_data_column_names()

        if self.config.should_use_metas():
            metaCols = [
                col for col in self.get_id_columns(self.config.get_data_catalog(), self.config.get_data_table()).keys()
                if col not in dataCols and col != class_column
                ]
            dataCols += metaCols

        classCols = [
            "class_result",
            "class_rate",
            "class_time",
            "class_algorithm"
        ]

        columns = f"TOP({num_rows}) " + ", ".join([f"A.[{a}]" for a in dataCols] + [f"B.[{b}]" for b in classCols])

        classTable = connection.get_formatted_class_table() + " B"
        dataTable = connection.get_formatted_data_table() + " A"
        
        class_user = self.config.get_quoted_attribute("connection.class_username")
        table_name = self.config.get_quoted_attribute("connection.data_table")
        
        join = f"A.[{id_column}] = B.[unique_key]"
        where = f"B.[class_user] = {class_user} AND B.[table_name] = {table_name}"
        query = f"SELECT {columns} FROM {dataTable} INNER JOIN {classTable} ON {join} WHERE {where} ORDER BY B.[class_time] DESC"

        return query


    def count_data_rows(self, data_catalog: str, data_table: str) -> int:
        """ Function for counting the number of rows in data to classify """
        count = -1
        try:
            endQuery = f"[{period_to_brackets(data_catalog)}].[{period_to_brackets(data_table)}]"
            count = self.get_connection().get_count(endQuery=endQuery)

        except Exception as e:
            self.logger.print_dragon(e)
            raise DataLayerException(f"Count of dataset failed: {str(e)}")

        # Return 
        return count

    def get_class_distribution_query(self) -> str:
        """ Builds the query for class=>rows """
        c_column = self.config.get_class_column_name()
        
        select = f"SELECT [{c_column}], COUNT(*) FROM {self.config.connection.get_formatted_data_table()}"

        query = f"{select} GROUP BY [{c_column}] ORDER BY [{c_column}] DESC"

        return query

    def count_class_distribution(self) -> dict:
        """ Function for counting the number of rows in data corresponding to each class """

        query = self.get_class_distribution_query()
        data = []
        try:
            data = self.get_data_list_from_query(query)
        except Exception as e:
            self.logger.print_dragon(e)
            raise DataLayerException(f"Count of dataset distribution failed: {str(e)}")

        dict = {}
        for key,value in data:
            if key == None:
                key = "NULL"
            dict[key] = value
        
        return dict

    def get_save_classification_insert(self, column_keys) -> str:
        # Build up the base query outside of the range
        insert = f"INSERT INTO {self.config.connection.get_formatted_class_table()} "
        column_names = ",".join([column for column in column_keys])
        insert += f"({column_names})"

        return insert

    def save_data(self, results: list, class_rate_type: RateType, model: Model)-> int:
        """ Save the classifications in the database """

        # Loop through the data
        num_lines = 0
        percent_fetched = 0.0
        result_num = len(results)

        # Get class labels separately
        try:
            class_labels = ",".join(model.model.classes_)
        except AttributeError:
            self.logger.print_info(f"No classes_ attribute in model")
            class_labels = "N/A"

        columns = OrderedDict({
            "catalog_name": self.config.get_data_catalog(),
            "table_name": self.config.get_data_table(),
            "column_names": ",".join(self.config.get_data_column_names()), 
            "unique_key": "", # dict["key"]
            "class_result": "", # dict["prediction"]
            "class_rate": "", # dict["rate"]
            "class_rate_type": class_rate_type.name,
            "class_labels": class_labels,
            "class_probabilities": "", # dict["probabilities"]
            "class_algorithm": model.get_name(),
            "class_script": self.config.script_path,
            "class_user": self.config.get_data_username()
        })
        
        
        # Build up the base query outside of the range
        insert = self.get_save_classification_insert(columns.keys())
        
        try:
            # Get a sql handler and connect to data database
            sqlHelper = self.get_connection()
        
            for row in results:
                columns["unique_key"] = row["key"]
                columns["class_result"] = row["prediction"]
                columns["class_rate"] = row["rate"]
                columns["class_probabilities"] = row["probabilities"]

                values = ",".join([to_quoted_string(elem) for elem in columns.values()])
                query = insert + f" VALUES ({values})"
                
                # Execute a query without getting any data
                # Delay the commit until the connection is closed
                if sqlHelper.execute_query(query, get_data=False, commit=False):
                    num_lines += 1
                    percent_fetched = round(100.0 * float(num_lines) / float(result_num))
                    self.logger.print_percentage("Part of data saved", percent_fetched)
                

            self.logger.print_linebreak()
            
            if num_lines > 0:
                # Disconnect from database and commit all inserts
                sqlHelper.disconnect(commit=True)
            
            # Return the number of inserted rows
            return num_lines
        except KeyError as ke:
            raise DataLayerException(f"Something went wrong when saving data ({ke})")
        except Exception as e:
            self.logger.print_dragon(e)
            raise DataLayerException(f"Something went wrong when saving data ({e})")

    def get_data_query(self, num_rows: int) -> str:
        """ Prepares the query for getting data from the database """
        train = self.config.should_train()
        predict = self.config.should_predict()
        columns = ",".join([ f"[{column}]" for column in self.config.get_column_names()])
        class_column = f"[{self.config.get_class_column_name()}]"
        query = f"SELECT TOP({num_rows}) {columns} FROM "
        
        outer_query = query
        
        query += self.config.connection.get_formatted_data_table()
        
        cast_where = None
        # Take care of the special case of only training or only predictions
        if train and not predict:
            cast_where = f"NOT NULL AND CAST({class_column} AS VARCHAR) != \'\'"

        if not train and predict:
            cast_where = f"NULL OR CAST({class_column} AS VARCHAR) = \'\'"

        if cast_where:
            query += f" WHERE {class_column} IS {cast_where}"

        # Since sorting the DataFrames directly does not seem to work right now (see below)
        # we sort the data in retreiving in directly in SQL. The "DESC" keyword makes sure
        # all NULL values (unclassified data) is placed last.
        # If num_rows is less than the number of available records, this query will fetch
        # num_rows randomly selected rows (NEWID does this trick)
        query += " ORDER BY NEWID()"
        query = outer_query + "(" + query + ") A"
        
        return query
    
    def parse_dataset(self, num_rows: int, use_chunks: bool, read_data_func):
        """ Given a number of rows, whether to use chunks or rows, and a function for reading data
            parse and return a list
        """
        num_lines = 0
        percent_fetched = 0
        data = []
        
        data_section = read_data_func[0](**read_data_func[1])
        data =  np.asarray(data_section)
        num_lines = len(data_section) if use_chunks else 1
        percent_fetched = round(100.0 * float(num_lines) / float(num_rows))
        self.logger.print_percentage("Data fetched of available", percent_fetched)
        
        while data_section:
            if num_lines >= num_rows:
                break
            
            data_section = read_data_func[0](**read_data_func[1])
            if data_section:
                data = np.append(data, data_section, axis = 0)
                num_lines += len(data_section) if use_chunks else 1
                old_percent_fetched = percent_fetched
                percent_fetched = round(100.0 * float(num_lines) / float(num_rows))
                self.logger.print_percentage("Data fetched of available", percent_fetched, old_percent_fetched)

        
        if not use_chunks:
            # Rearrange the long 1-dim array into a 2-dim numpy array which resembles the
            # database table in question
            data = np.asarray(data).reshape(num_lines,int(len(data)/num_lines)) 

        self.logger.print_linebreak()
        return data

    def get_dataset(self, num_rows: int = None) -> list:
        """ Gets the needed data from the database """
        num_rows = num_rows if num_rows else self.config.get_max_limit()
        query = self.get_data_query(num_rows)
        
        query += f" ORDER BY [{self.config.get_class_column_name()}] DESC"
        
        self.logger.print_query(type="Classification data", query=query)

        try:
            # Get a sql handler and connect to data database
            sqlHelper = self.get_connection()

             # Setup and execute a query to get the wanted data. 
            # 
            # The query has the form 
            # 
            #   SELECT columns FROM ( SELECT columns FROM table ORDER BY NEWID()) ORDER BY class_column
            #
            # to ensure a random selection (if not all rows are chosen) and ordering of data, and a 
            # subsequent sorting by class_column to put the NULL-classification last
            #
            
            # Now we are ready to execute the sql query
            # By default, all fetched data is placed in one long 1-dim list. The alternative is to read by chunks.
            num_lines = 0
            
            data = []
            if sqlHelper.execute_query(query, get_data=True):
                read_data_function = (
                    sqlHelper.read_next,
                    {
                        "chunksize": self.SQL_CHUNKSIZE if self.SQL_USE_CHUNKS else None
                    }
                )
                data = self.parse_dataset(num_rows, use_chunks=self.SQL_USE_CHUNKS, read_data_func=read_data_function)
                
                num_lines = len(data)
            self.logger.print_formatted_info(f"Totally fetched {num_lines} data rows")
            
            # Disconnect from database
            sqlHelper.disconnect()

            # Quick return if no data was fetched
            if num_lines == 0:
                return None
            
            return data

        except Exception as e:
            self.logger.print_dragon(e)
            raise DataLayerException(f"Something went wrong when fetching dataset ({e})")

    def get_mispredicted_query(self, new_class: str, index: int) -> str:
        """ Set together SQL code for the insert """
        query_strings = [
            f"UPDATE {self.config.get_connection().get_formatted_data_table()}",
            f"SET {self.config.get_class_column_name()} = {to_quoted_string(new_class)}",
            f"WHERE {self.config.get_id_column_name()} = {index}"
        ]
        
        return " ".join(query_strings)
        
    def correct_mispredicted_data(self, new_class: str, index: int) -> int:
        """ Corrects data in the original dataset """
        num_lines = 0

        self.logger.print_info(f"Changing data row {index} to {new_class}: ")
        query = self.get_mispredicted_query(new_class, index)
        self.logger.print_query("mispredicted", query)
        
        try:
            # Get a sql handler and connect to data database
            sqlHelper = self.get_connection()

            # Execute a query without getting any data
            # Delay the commit until the connection is closed
            if sqlHelper.execute_query(query, get_data=False, commit=False):
                num_lines = 1
            
            # Disconnect from database and commit all inserts
            sqlHelper.disconnect(commit=True)

            # Return the number of inserted rows
            return num_lines

        except Exception as e:
            self.logger.print_dragon(e)
            raise DataLayerException(f"Correction of mispredicted data: {query} failed: {e}")
    
    def save_data_revised(self, results: list, class_rate_type: RateType, model: Model)-> int:
        """ Save the classifications in the database """
        # This is rewriting save_data--once it works the save_data can be deleted and this renamed to save_data
        # Once I have a working config fil, ta bort staticmethod och börja använd self igen

        return self.get_insert_many_statement(self.config.connection.get_formatted_class_table(), ["a", "b", "c"])
        # Loop through the data
        num_lines = 0
        percent_fetched = 0.0
        result_num = len(results)

        # Get class labels separately
        """
        try:
            class_labels = ",".join(model.model.classes_)
        except AttributeError:
            self.logger.print_info(f"No classes_ attribute in model")
            class_labels = "N/A"

        columns = OrderedDict({
            "catalog_name": self.config.get_data_catalog(),
            "table_name": self.config.get_data_table(),
            "column_names": ",".join(self.config.get_data_column_names()), 
            "unique_key": "", # dict["key"]
            "class_result": "", # dict["prediction"]
            "class_rate": "", # dict["rate"]
            "class_rate_type": class_rate_type.name,
            "class_labels": class_labels,
            "class_probabilities": "", # dict["probabilities"]
            "class_algorithm": model.get_name(),
            "class_script": self.config.script_path,
            "class_user": self.config.get_data_username()
        })
        
        
        # Build up the base query outside of the range
        insert = self.get_save_classification_insert(columns.keys())
        
        try:
            # Get a sql handler and connect to data database
            sqlHelper = self.get_connection()
        
            for row in results:
                columns["unique_key"] = row["key"]
                columns["class_result"] = row["prediction"]
                columns["class_rate"] = row["rate"]
                columns["class_probabilities"] = row["probabilities"]

                values = ",".join([to_quoted_string(elem) for elem in columns.values()])
                query = insert + f" VALUES ({values})"
                
                # Execute a query without getting any data
                # Delay the commit until the connection is closed
                if sqlHelper.execute_query(query, get_data=False, commit=False):
                    num_lines += 1
                    percent_fetched = round(100.0 * float(num_lines) / float(result_num))
                    self.logger.print_percentage("Part of data saved", percent_fetched)
                

            self.logger.print_linebreak()
            
            if num_lines > 0:
                # Disconnect from database and commit all inserts
                sqlHelper.disconnect(commit=True)
            
            # Return the number of inserted rows
            return num_lines
        except KeyError as ke:
            raise DataLayerException(f"Something went wrong when saving data ({ke})")
        except Exception as e:
            self.logger.print_dragon(e)
            raise DataLayerException(f"Something went wrong when saving data ({e})")
        """

    @staticmethod
    def get_insert_many_statement(table: str, columns) -> str:
        """ Given a table and an iterable of column names returns an insert statement with placeholders """
        columns_string = ",".join([column for column in columns])
        values = ",".join(["?" for x in range(len(columns))]) # A string of ?, ?, ?, with the number of ? being based on the number of columns
        insert = f"INSERT INTO {table} ({columns_string}) VALUES({values})"
        
        return insert

# Main method
def main():
    from JBGLogger import JBGLogger

    # python SQLDataLayer.py -f .\config\test_iris.py
    config = Cf.load_config_from_module(sys.argv)
    logger = JBGLogger(False)
    dl = DataLayer(config, logger)
    table = 'aterkommande_automat.iris'
    database = 'Arbetsdatabas'
    #insert = DataLayer.get_insert_many_statement("table", ["a", "b", "c"])
    #print(insert)
    print(dl.save_data_revised([], "", ""))
    #dataset, query = dl.get_dataset(15)
    #print(dataset)


    #print(dl.get_connection())
    
    

    # database_list = [''] + [database[0] for database in data]
    # self.database_dropdown.options = database_list
    #self.database_dropdown.value = database_list[0]
    # get_tables
    # tables_list = [""] + [table[0]+'.'+table[1] for table in tables]
    # self.tables_dropdown.options = tables_list
    # self.tables_dropdown.value = tables_list[0]
    # get_id_columns
    # This is used for testing get_data_from_query: get_simple_query
    #tbls = dl.get_tables()
    #print(tbls)
    # columns_list = [column[0] for column in columns]
    # self.datatype_dict = {column[0]:column[1] for column in columns}
    

# Start main
if __name__ == "__main__":
    main()

"""
columns = self.gui_datalayer.get_id_columns(self.database_dropdown.value, self.tables_dropdown.value)
columns_list = [column[0] for column in columns]
self.datatype_dict = {column[0]:column[1] for column in columns}

id_dicts = dl.get_id_columns(database=database, table=table)
    
    columns = {
        "class": {
            "start": 0,
            "default": 0,
            "observer": "class"
        },
        "id": {
            "start": 1,
            "default": 1,
            "observer": "id"
        },
        "data": {
            "start": 2,
            "default": None,
            "observer": "text"
        }
    }
    #testing = list(id_dicts.keys())
    #s = slice(0, -1)
    #print(testing[s])

    columns_list = list(id_dicts.keys())
    for key, values in columns.items():
        #print("options: 0:?, 1:, 2:")
        print(key)
        s = slice(values["start"], -1)
        print(columns_list[s])
        default = [] if values["default"] is None else columns_list[values["default"]]
        print(default)
        print("disabled: False")
        print(values["observer"])
"""