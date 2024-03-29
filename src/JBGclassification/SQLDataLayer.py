
import sys
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Callable, Protocol
import hashlib

import numpy as np
import pandas as pd
import pyodbc
import SqlHelper.JBGSqlHelper as sql
from Config import Config as Cf
from Config import period_to_brackets, to_quoted_string
from JBGMeta import  RateType
from DataLayer.Base import DataLayerBase
from JBGExceptions import DataLayerException, SQLException


class Logger(Protocol):
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""
    def print_query(self, type: str, query: str) -> None:
        """ Prints a query """

    def print_dragon(self, exception: Exception) -> None:
        """ Type of Unhandled Exceptions, to handle them for the future """

    def start_inline_progress(self, key: str, description: str, final_count: int, tooltip: str) -> None:
        """ This will overwrite any prior bars with the same key """

    def update_inline_progress(self, key: str, current_count: int, terminal_text: str) -> None:
        """ Updates progress bars within the script"""
    
    def end_inline_progress(self, key: str, set_100: bool = True) -> None:
        """ Ensures that any loose ends are tied up after the progress is done """

    def parse_dataset_progress(self, key: str, num_lines: int, num_rows: int) -> None:
        """ Groups the start of the parse_dataset functions print-outs """

    def print_progress(self, message: str = None, percent: float = None) -> None:
        """ Updates the progress bar and prints out a value in the terminal of no bar"""

    def print_formatted_info(self, message: str) -> None:
        """ Prints information with a bit of markup """

    def print_correcting_mispredicted(self, new_class: str, index: int, query: str) -> None:
        """ Prints out notice about correcting mispredicted class in class_catalog """

        
    

class Connection(Protocol):
    def update_catalogs(self, type: str, catalogs: list, checking_func: Callable) -> list:
        """ Updates the config to only contain accessible catalogs """

    def get_formatted_data_table(self, include_database: bool = True) -> str:
            """ Gets the data table as a formatted string for the correct driver
                In the type of [schema].[catalog].[table]
            """

    

class Config(Protocol):
    script_path: Path
    
    # Methods to hide implementation of Config
    def get_attribute(self, attribute: str):
        """ Gets an attribute from a attribute.subattribute string """
    
    def get_class_catalog_params(self) -> dict:
        """ Gets params to connect to class database """

    def get_classification_script_path(self) -> Path:
        """ Gives a calculated path based on config"""
    
    def get_prediction_tables(self, include_database: bool = True) -> dict:
        """ Gets a dict with 'header' and 'row', based on class_table"""

    def get_data_catalog_params(self) -> dict:
        """ Gets params to connect to data database """

    def get_class_column_name(self) -> str:
        """ Gets the name of the column"""

    def should_train(self) -> bool:
        """ Returns if this is a training config """
        
    def should_predict(self) -> bool:
        """ Returns if this is a prediction config """
        
    def get_column_names(self) -> list[str]:
        """ Gets the column names based on connection columns """

    def get_id_column_name(self) -> str:
        """ Gets the name of the ID column"""

    def get_data_limit(self) -> int:
        """ Get the data limit"""

    def get_connection(self)  -> Connection:
        """ Returns the connection object """

    def get_data_catalog(self) -> str:
        """ Gets the data catalog """
    
    def get_data_table(self, include_database: bool = False) -> str:
        """ Gets the data table """
    
    def get_data_username(self) -> str:
        """ Gets the data user name """

    def get_data_column_names(self) -> list[str]:
        """ Gets data columns, so not Class or ID """

    def should_use_metas(self) -> bool:
        """ Returns if this is a use metas config """
    

@dataclass
class DataLayer(DataLayerBase):
    SQL_CHUNKSIZE = 2500
    SQL_USE_CHUNKS = True
    SQL_STALLED_FETCHES_LIMIT = 3
    SQL_MIN_PERCENT_FETCH = 0.025

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
        #print(self.config)
        if self.config.connection.trusted_connection or \
            (self.config.connection.sql_username and self.config.connection.sql_password):
            if not self.can_connect():
                raise ValueError(f"Connection to server failed")
        
    def get_connection(self) -> sql.JBGSqlHelper:
        """ Get an odbc-based sql handler using the class catalog from the config """
        return sql.JBGSqlHelper(**self.config.get_class_catalog_params())

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
        return self.config.get_connection().update_catalogs("data", catalogs, checking_func = self.get_gui_list)
    

    def get_tables_as_options(self) -> list:
        """ Used in the GUI, to get the tables """
        return self.get_gui_list("tables", "{}.{}")

    def get_id_columns(self, database: str, table: str) -> dict:
        """ Gets name and type for columns in specified <database> and <table> """

        database = to_quoted_string(database)
        table = to_quoted_string(table)
        select = "COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS"
        where = f"TABLE_CATALOG = {database} AND CONCAT(CONCAT(TABLE_SCHEMA,'.'),TABLE_NAME) = {table}"
        order_by = "DATA_TYPE DESC"
        query = f"SELECT {select} WHERE {where} ORDER BY {order_by}"
        
        data = self.get_data_list_from_query(query)

        return {column[0]:column[1] for column in data}


    def create_predictions_table(self, query: str) -> None:
        """ Create the predictions table """
        self.logger.print_progress(message="Create the predictions table")
        
        
        # Get a sql handler and connect to data database
        sqlHelper = self.get_connection()

        # Execute query
        sqlHelper.execute_query(query, get_data=False, commit = True)

        sqlHelper.disconnect(commit=False)

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

        table = self.config.get_connection().get_formatted_data_table()
        #print(table)
        
        select = f"SELECT [{c_column}], COUNT(*) FROM {table}"

        query = f"{select} GROUP BY [{c_column}] ORDER BY [{c_column}] DESC"

        return query

    def count_class_distribution(self) -> dict:
        """ Function for counting the number of rows in data corresponding to each class """

        query = self.get_class_distribution_query()
        #print(query)
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


    def get_data_query(self, num_rows: int) -> str:
        """ Prepares the query for getting data from the database """
        train = self.config.should_train()
        predict = self.config.should_predict()
        columns = ",".join([ f"[{column}]" for column in self.config.get_column_names()])
        class_column = f"[{self.config.get_class_column_name()}]"
        query = f"SELECT TOP({num_rows}) {columns} FROM "
        
        outer_query = query
        
        query += self.config.get_connection().get_formatted_data_table()
        
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
        data = []
        
        data_section = read_data_func[0](**read_data_func[1])
        data =  np.asarray(data_section)
        num_lines = len(data_section) if use_chunks else 1
        progress_key = "fetch_dataset"
        self.logger.parse_dataset_progress(progress_key, num_lines, num_rows)
        
        while data_section:
            if num_lines >= num_rows:
                break
            
            data_section = read_data_func[0](**read_data_func[1])
            if data_section:
                data = np.append(data, data_section, axis = 0)
                num_lines += len(data_section) if use_chunks else 1
                self.logger.update_inline_progress(progress_key, num_lines, "Data fetched of available")
        
        if not use_chunks:
            # Rearrange the long 1-dim array into a 2-dim numpy array which resembles the
            # database table in question
            data = np.asarray(data).reshape(num_lines,int(len(data)/num_lines)) 

        self.logger.end_inline_progress(progress_key, set_100 = False) # This can have a progress less than 100
        return data

    def get_dataset(self, num_rows: int = None) -> list:
        """ Gets the needed data from the database """
        
        # Calculate the maximum number of rows that can be fetched from the database
        num_rows = num_rows if num_rows else self.config.get_data_limit()
        
        # To be able to fetch the correct number of data rows, we need to find the 
        # data distribution for each class
        data_dist = self.count_class_distribution()

        # Remove the unpredicted data counts if no predictions should be performed and sum up
        if self.config.mode.predict == False:
            for key in ["", "NULL", None]:
                data_dist.pop(key, None)
            max_num_rows = sum(data_dist.values())
            num_rows = min(num_rows, max_num_rows)
        
        # Start fetching the data from the database
        try:
        
            # Setup and execute a query to get the wanted data. 
            # 
            # The query has the form 
            # 
            #   SELECT TOP(columns) FROM ( SELECT TOP(columns) FROM table ORDER BY NEWID()) ORDER BY class_column
            #
            # to ensure a random selection (if not all rows are chosen) and ordering of data, and a 
            # subsequent sorting by class_column to put the NULL-classification last
            #
            # Now we are ready to execute the sql query
            # By default, all fetched data is placed in one long 1-dim list. The alternative is to read by chunks.
            num_lines = 0
            num_added_lines = 0
            num_stalled_fetches = 0
            data = None

            # Get a sql handler and connect to data database
            sqlHelper = self.get_connection()

            # If network issues comes up, we might need to read by several queries
            num_queries = 0
            loc_num_rows = num_rows
            while num_lines < num_rows:

                query = self.get_data_query_with_order(loc_num_rows, True, True)
                if num_queries == 0:
                    self.logger.print_query(type="Classification data", query=query)
                num_queries += 1

                # Try and possibly catch SQLException because of network issues
                try:
                    if sqlHelper.execute_query(query, get_data=True):
                        
                        # Get the data from the current query
                        read_data_function = (
                            sqlHelper.read_next,
                            {
                                "chunksize": self.SQL_CHUNKSIZE if self.SQL_USE_CHUNKS else None
                            }
                        )
                        _data = self.parse_dataset(loc_num_rows, use_chunks=self.SQL_USE_CHUNKS, read_data_func=read_data_function)
                        
                        # Take care of retreived data
                        if data is None:
                            old_num_lines = 0
                        else:
                            old_num_lines = len(data)
                        data = self.add_data_without_duplicates(old_data = data, new_data = _data)

                        # Keep track of the progress
                        num_lines = len(data)
                        num_added_lines = num_lines - old_num_lines
                        num_lines_missing = num_rows - num_lines
                        if num_added_lines > 0:
                            self.logger.print_formatted_info(f"Last query added {str(num_added_lines)} to dataset. (Still missing {str(num_lines_missing)}.)")
                        
                        # If we are done, we are done
                        if num_lines_missing == 0:
                            break
                
                # In case of SQLException, divide the fetched number of rows by 2 and try again
                except SQLException as e:
                    self.logger.print_formatted_info(f"An SQL Exception occured {str(e)}.")
                    if num_queries < 2:
                        self.logger.print_formatted_info(f"Using multiple queries for dataset...")
                    sqlHelper.disconnect()
                    loc_num_rows = int(loc_num_rows / 2)
                    self.logger.print_formatted_info(f"Scaling down number of rows per query to: {str(loc_num_rows)}")
                    sqlHelper = self.get_connection()
                    
                    # In case of an exception, do not proceed further but try again from the beginning of the loop
                    continue
                
                # Keep track of stalling, adjust number of rows and break the while loop when fetching has stalled
                if data is not None and num_added_lines == 0:
                    num_stalled_fetches += 1
                    self.logger.print_warning(f"Warning! Last {str(num_stalled_fetches)} queries added no rows to dataset")
                    if num_stalled_fetches > self.SQL_STALLED_FETCHES_LIMIT:
                        self.logger.print_formatted_info(f"Data fetching stalled. Breaking the fetch loop...")
                        break
                    else:
                        # Increase number of lines per query
                        loc_num_rows += self.SQL_CHUNKSIZE
                else:
                    
                    # We did fetch something, didn't we?
                    num_stalled_fetches = 0
                
                    # If we added very litte data last time, add one chuncksize to number of fetched lines.
                    if data is not None and num_added_lines <= int(loc_num_rows * self.SQL_MIN_PERCENT_FETCH):
                        loc_num_rows += self.SQL_CHUNKSIZE
                        self.logger.print_formatted_info(f"Scaling up number of rows per query to: {str(loc_num_rows)}")
                    
                    # Next time, do not fetch more than necessary.
                    else:
                        loc_num_rows = min(loc_num_rows, num_rows - num_lines)
                        
                    # This condiction ensures that we break the loop when finished (maybe not necessary)
                    if loc_num_rows < 1:
                        break

                # In case of debug mode, wait five seconds before continuing
                if self.config.debug:
                    time.sleep(5)

            self.logger.print_formatted_info(f"Fetched {num_lines} data rows in total")
            
            # Disconnect from database
            sqlHelper.disconnect()

            # Quick return if no data was fetched
            if num_lines == 0:
                return None
            
            # If multiple querires were used, make sure to sort the data before returning since the SQL sorting most likely was corrupted
            if num_queries > 1:
                data = self.sort_data_on_class_column(data)
            return data
        
        except Exception as e:
            self.logger.print_dragon(e)
            raise DataLayerException(f"Something went wrong when fetching dataset ({e})")

    def add_data_without_duplicates(self, old_data: list = None, new_data: list = None) -> list:
        
        # Trivial cases where there is no old data or no new data
        if old_data is None:
            return new_data
        elif new_data is None:
            return old_data
        else:

           # Get column names and unique index of the dataset
            column_names = self.config.get_column_names()
            unique_index = self.config.get_id_column_name()

            # Convert both new and old data into panda Dataframes
            old_df = pd.DataFrame(old_data, columns = column_names)
            new_df = pd.DataFrame(new_data, columns = column_names)

            # Merge the two DataFrames
            concatenated_df = pd.concat([old_df, new_df], axis = 0)

            # Get rid of duplicate data rows with the same index (keep the old)
            concatenated_df.drop_duplicates(subset = [unique_index], keep = 'first', inplace=True)
            
            # Return the resulting data in numpy format
            return concatenated_df.to_numpy()

    def sort_data_on_class_column(self, data, asc = False):
        
        class_column = self.config.get_class_column_name()
        column_names = self.config.get_column_names()

        df = pd.DataFrame(data, columns = column_names)
        df.sort_values(by = class_column, axis=0, ascending=asc, inplace=True, \
            kind='quicksort', na_position='last', ignore_index=False, key=None)

        return df.to_numpy()

    def get_data_query_with_order(self, num_rows: int, order_by_class: bool = True, descending: bool = True ) -> str:
        
        # Build the query for getting data from the database
        query = self.get_data_query(num_rows)

        # Add the order by class column if needed
        if order_by_class:
            query += f" ORDER BY [{self.config.get_class_column_name()}] "
            if descending: 
                query += "DESC"
            else:
                query += "ASC"

        return query
    
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

        query = self.get_mispredicted_query(new_class, index)
        self.logger.print_correcting_mispredicted(new_class, index, query)
    
        
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

    @staticmethod
    def get_insert_many_statement(table: str, columns) -> str:
        """ Given a table and an iterable of column names returns an insert statement with placeholders """
        columns_string = ",".join([column for column in columns])
        values = ",".join(["?" for x in range(len(columns))]) # A string of ?, ?, ?, with the number of ? being based on the number of columns
        insert = f"INSERT INTO {table} ({columns_string}) VALUES({values})"
        
        return insert
    
    def save_prediction_data(self, results: list, class_rate_type: RateType, model_name: str, class_labels: list, test_run: int = 0, commit: bool = True) -> dict:
        """ Saves prediction data, with the side-effect of creating the tables if necessary """
        # 1: Check if the main table exists. If not it will either create them, or fail gracefully
        
        tables = self.config.get_prediction_tables()
        if not self.prepare_predictions(tables, commit):
            error = "Database" if commit else f"Table '{tables['header']}' does not exist"
            return {"error": error}
    
        # 2: Insert information into the tables in two sets
        # a) Insert header info, get the ID
        header_columns = [
            "catalog_name",
            "table_name",
            "column_names",
            "test_run",
            "class_rate_type",
            "class_labels",
            "class_algorithm",
            "class_script",
            "class_user"
        ]
        
        header_values = (
            self.config.get_data_catalog(),
            self.config.get_data_table(),
            ",".join(self.config.get_data_column_names()),
            test_run,
            class_rate_type.name,
            ",".join(class_labels),
            model_name,
            str(self.config.script_path),
            self.config.get_data_username()
        )
        
        try:
            run_id = self.get_connection().insert_row(
                table=tables['header'], columns=header_columns, values=header_values, return_id=True, commit=commit
            )
        except Exception as e:
            self.logger.print_dragon(e)
            raise DataLayerException(f"Something went wrong when creating meta header row for predictions: ({e})")
        
        # b) Rows, run_id = id 
        row_columns = [
            "run_id",
            "unique_key",
            "class_result",
            "class_rate",
            "class_probabilities"
        ]

        row_values = [
            [
                run_id, 
                row["key"], 
                row["prediction"], 
                row["rate"], 
                row["probabilities"]
            ] for row in results
        ]

        try:
            row_count = self.get_connection().insert_many_rows(
                tables['row'], columns=row_columns, params=row_values, commit=commit
            )
        except Exception as e:
            self.logger.print_dragon(e)
            raise DataLayerException(f"Something went wrong when creating rows for predictions: ({e})")
        
        return {
            "error": None,
            "results": {
                "run_id": int(run_id),
                "row_count": row_count,
                "query": self.get_run_query(run_id)
            }
        }

    def get_run_query(self, run_id: int) -> str:
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
        
        metaCols = [
            "RR.[class_result]",
            "RR.[class_rate]",
            "RH.[class_time]",
            "RH.[class_algorithm]"
        ]
        columns = ", ".join([f"A.[{a}]" for a in dataCols] + [r for r in metaCols])
        
        dataTable = connection.get_formatted_data_table() + " A"
        predictionTables = connection.get_formatted_prediction_tables()

        query_strings = [
            f"SELECT {columns} FROM {dataTable}",
            f"INNER JOIN {predictionTables['row']} RR ON A.[id] = RR.[unique_key]",
            f"INNER JOIN {predictionTables['header']} RH ON RR.[run_id] = RH.[run_id]",
            f"WHERE RH.[run_id] = {run_id}",
            f"ORDER BY A.[{id_column}]"
        ]
        return " ".join(query_strings)
        

    def prepare_predictions(self, tables: dict, create_tables: bool) -> bool:
        header_table = tables["header"]
        if not self.get_connection().check_table_exists(header_table):
            if not create_tables:
                return False

            self.create_prediction_tables(tables)

        return True

    def create_prediction_tables(self, tables: dict) -> None:
        """ Creates tables for prediction data """
        query = "\n".join(self.create_predictions_tables_query(tables))
        self.create_predictions_table(query)

    def create_predictions_tables_query(self, tables: dict, path: str = None) -> list:
        """ Creates a table query from a given text file """
        path = path if path else self.config.get_classification_script_path()

        if not path.is_file():
            raise DataLayerException(f"File {path} does not exist.")

        query_list = []
        
        unique_run_header = hashlib.md5(tables["header"].encode('utf-8')).hexdigest()
        unique_run_row = hashlib.md5(tables["row"].encode('utf-8')).hexdigest()

        with open(path, mode="rt") as f:
            for line in f:
                transformed_line = line.strip().replace("<header_table>", tables["header"])
                transformed_line = transformed_line.replace("<row_table>", tables["row"])
                transformed_line = transformed_line.replace("<run_header>", unique_run_header)
                transformed_line = transformed_line.replace("<run_row>", unique_run_row)
                
                query_list.append(transformed_line)
        
        return query_list

# Main method
def main():
    from JBGLogger import JBGLogger

    # python SQLDataLayer.py -f .\config\test_iris.py
    config = Cf.load_config_from_module(sys.argv)
    logger = JBGLogger(False)
    dl = DataLayer(config, logger)
    table = 'aterkommande_automat.iris'
    database = 'Arbetsdatabas'
    model_name = "name"
    class_labels=["a", "b", "c"]
    
    results = [
        {
            "key": 60, 
            "prediction": "Iris-versicolor", 
            "rate": 1,
            "probabilities": "0.0,1.0,0.0"
        }, {
            "key": 40, 
            "prediction": "Iris-setosa", 
            "rate": 1,
            "probabilities": "1.0,0.0,0.0"
        }, {
            "key": 140, 
            "prediction": "Iris-virginica", 
            "rate": 1,
            "probabilities": "0.0,0.0,1.0"
        }, {
            "key": 50, 
            "prediction": "Iris-setosa", 
            "rate": 1,
            "probabilities": "1.0,0.0,0.0"
        }, {
            "key": 120, 
            "prediction": "Iris-versicolor", 
            "rate": 1,
            "probabilities": "0.0,1.0,0.0"
        }, {
            "key": 130, 
            "prediction": "Iris-versicolor", 
            "rate": 1,
            "probabilities": "0.0,1.0,0.0"
        }, {
            "key": 20, 
            "prediction": "Iris-setosa", 
            "rate": 1,
            "probabilities": "1.0,0.0,0.0"
        }, {
            "key": 90, 
            "prediction": "Iris-versicolor", 
            "rate": 1,
            "probabilities": "0.0,1.0,0.0"
        }, {
            "key": 30, 
            "prediction": "Iris-setosa", 
            "rate": 1,
            "probabilities": "1.0,0.0,0.0"
        }, {
            "key": 150, 
            "prediction": "Iris-virginica", 
            "rate": 1,
            "probabilities": "0,6;0.0,0.4,0.6"
        }, {
            "key": 10, 
            "prediction": "Iris-setosa", 
            "rate": 1,
            "probabilities": "1.0,0.0,0.0"
        }, {
            "key": 100, 
            "prediction": "Iris-versicolor", 
            "rate": 1,
            "probabilities": "0.0,1.0,0.0"
        }, {
            "key": 110, 
            "prediction": "Iris-virginica", 
            "rate": 1,
            "probabilities": "0.0,0.0,1.0"
        }, {
            "key": 70, 
            "prediction": "Iris-versicolor", 
            "rate": 1,
            "probabilities": "0.0,1.0,0.0"
        }, {
            "key": 80, 
            "prediction": "Iris-versicolor", 
            "rate": 1,
            "probabilities": "0.0,1.0,0.0"
        }
    ]
    rate_type = RateType.I

    print(dl.save_prediction_data(
        commit=False, 
        test_run=1,
        model_name = model_name,
        class_labels=class_labels,
        results=results,
        class_rate_type=rate_type))
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