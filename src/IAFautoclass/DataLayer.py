
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from collections import OrderedDict
import numpy as np
import pyodbc
from Config import Config, RateType
from IAFHandler import Model
from IAFLogger import IAFLogger
import SqlHelper.IAFSqlHelper as sql
from IAFExceptions import DataLayerException

@dataclass
class DataLayer:
    SQL_CHUNKSIZE = 1000 #TODO: Decide how many
    SQL_USE_CHUNKS = True

    connection: Config.Connection
    logger: IAFLogger
    scriptpath: str = field(init=False)
    text_data: bool = field(init=False)
    numerical_data: bool = field(init=False)

    def __post_init__(self) -> None:
        # Extract parameters that are not textual
        self.text_data = self.connection.data_text_columns != ""
        self.numerical_data = self.connection.data_numerical_columns != ""
        self.scriptpath = os.path.dirname(os.path.realpath(__file__))
    
        if drivers().find(self.connection.odbc_driver) == -1:
            raise ValueError("Specified ODBC driver cannot be found!")
    
    def update_connection(self, connection: Config.Connection) -> None:
        """ Allows us to update the connection to match the Config """
        self.connection = connection
        self.__post_init__()
    
    # get the SQL handler
    def get_sql_connection(self) -> sql.IAFSqlHelper:
        # Get a sql handler and connect to data database
        sqlHelper = sql.IAFSqlHelper(driver = self.connection.odbc_driver, \
            host = self.connection.host, catalog = self.connection.class_catalog, \
            trusted_connection = self.connection.trusted_connection, \
            username = self.connection.class_username, \
            password = self.connection.class_password)
        
        return sqlHelper

    def can_connect(self, verbose: bool = False) -> bool:
        try:
            self.get_sql_connection() # This checks so that the SQL connection works
        except Exception as e:
            if verbose:
                self.logger.print_warning(f"Exception when connectiong to SQL server: {e}")
            
            return False
        
        return True

# Update the text_data if not set when the DataLayer is created
    def update_text_data(self, text_data:bool) -> None:
        self.text_data = text_data

    # Update the numerical_data if not set when the DataLayer is created
    def update_numerical_data(self, numerical_data:bool) -> None:
        self.numerical_data = numerical_data
    
    # generic function to get data from any query
    def get_data_from_query(self, query: str) -> list:
        sqlHelper = self.get_sql_connection()
        try:
            sqlHelper.execute_query(query, get_data = True)
        except Exception as ex:
            sys.exit("Query \"{0}\" failed: {1}".format(query,str(ex)))
        
        return sqlHelper.read_all_data()

    # Used in the GUI, to get the databases to choose from
    def get_databases(self):
        query = "SELECT name FROM sys.databases"
        
        return self.get_data_from_query(query)

    # Used in the GUI, to get the tables to choose from
    def get_tables(self):
        query = "SELECT TABLE_SCHEMA,TABLE_NAME FROM INFORMATION_SCHEMA.TABLES ORDER BY TABLE_SCHEMA,TABLE_NAME"
        
        return self.get_data_from_query(query)

    def get_id_columns(self, database: str, table: str):
        query = "SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_CATALOG = " \
            + "\'" + database + "\' AND CONCAT(CONCAT(TABLE_SCHEMA,'.'),TABLE_NAME) = " \
            + "\'" + table + "\'" + " ORDER BY DATA_TYPE DESC"
        
        return self.get_data_from_query(query)
        

    # Get a list of pretrained models
    def get_trained_models(self, model_path: str, model_file_extension: str):
        models = []
        for file in os.listdir(model_path):
            if file[-len(model_file_extension):] == model_file_extension:
                models.append(file)

        return models

    # Create the classification table
    def create_classification_table(self) -> None:
        self.logger.print_progress(message="Create the classification table")
        
        # Read in the sql-query from file
        sql_file = open(self.scriptpath + "\\" + str(Path(self.connection.class_table_script)), mode="rt")
        
        #sql_file = open(self.scriptpath + self.connection.class_table_script, mode="rt")
        query = ""
        nextline = sql_file.readline()
        while nextline:
            query += nextline.strip() + "\n"
            nextline = sql_file.readline()
        sql_file.close()
        
        # Make sure we create a classification table in the right place with the right name
        #print("query=",query)
        query = query.replace("<class_catalog>", self.connection.class_catalog)
        #print("query2=",query)
        query = query.replace("<class_table>", self.connection.class_table.replace(".", "].["))
        #print("query3=",query)
        
        # Get a sql handler and connect to data database
        sqlHelper = self.get_sql_connection()

        # Execute query
        sqlHelper.execute_query(query, get_data=False, commit = True)

        sqlHelper.disconnect(commit=False)

    # Produces an SQL command that can be executed to get a hold of the recently classified
    # data elements
    def get_sql_command_for_recently_classified_data(self, num_rows: int) -> str:
        
        selcols = ("A.[" + \
            "], A.[".join((self.connection.id_column + "," + \
            self.connection.data_numerical_columns + "," + \
            self.connection.data_text_columns).split(',')) + \
            "], ").replace("A.[]", "").replace(",,",",").replace(", ,",",")
        
        query = \
            "SELECT TOP(" + str(num_rows) + ") " + selcols +  \
            "B.[class_result], B.[class_rate],  B.[class_time], B.[class_algorithm] " + \
            "FROM [" + self.connection.data_catalog.replace('.',"].[") + "].[" + \
            self.connection.data_table.replace('.',"].[") + "] A " + \
            " INNER JOIN [" + self.connection.class_catalog.replace('.',"].[") + "].[" + \
            self.connection.class_table.replace('.',"].[") + "] B " + \
            "ON A." + self.connection.id_column + " = B.[unique_key] " + \
            "WHERE B.[class_user] = \'" + self.connection.class_username + "\' AND " + \
            "B.[table_name] = \'" + self.connection.data_table + "\' " + \
            "ORDER BY B.[class_time] DESC "
        return query

    # Function to set a new row in classification database which marks that execution has started.
    # If this new row persists when exeuction has ended, this signals that something went wrong.
    # This new row should be deleted before the program ends, which signals all is okay.
    #
    def mark_execution_started(self) -> int:
        self.logger.print_progress(message="Marking execution started in database...-please wait!")

        try:
           # Get a sql handler and connect to data database
            sqlHelper = self.get_sql_connection()

            # Mark execution started by setting unique key to -1 and algorithm to "Not set"

            # Set together SQL code for the insert
            query = "INSERT INTO " + self.connection.class_catalog + "." 
            query +=  self.connection.class_table + " (catalog_name,table_name,column_names," 
            query +=  "unique_key,class_result,class_rate,class_rate_type,"
            query += "class_labels,class_probabilities,class_algorithm," 
            query +=  "class_script,class_user) VALUES(\'" + self.connection.data_catalog + "\'," 
            query +=  "\'" + self.connection.data_table + "\',\'" 
            if self.text_data:
                query += self.connection.data_text_columns
            if self.text_data and self.numerical_data:
                query += ","
            if self.numerical_data:
                query +=  self.connection.data_numerical_columns  
            query +=  "\',-1" + ",\' N/A \',0.0," 
            query +=  "" + "\'U\',\' N/A \',\' N/A \',\'" + "Not set" + "\',\'" 
            query += self.scriptpath + "\',\'" + self.connection.data_username + "\')"

            # Execute a query without getting any data
            sqlHelper.execute_query(query, get_data=False, commit=True)   

            # Disconnect from database and commit all inserts
            sqlHelper.disconnect(commit=False)

            # Return the number of inserted rows
            return 1
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DataLayerException(f"Marking execution start failed: {e}")

    def mark_execution_ended(self) -> int:
        self.logger.print_info("Marking execution ended in database...")

        try:
            # Get a sql handler and connect to data database
            sqlHelper = self.get_sql_connection()

            # Mark execution ended by removing first row with unique key as -1 
            # for the current user and currect script and so forth

            # Set together SQL code for the deletion operation
            query = "DELETE FROM " + self.connection.class_catalog + "." 
            query +=  self.connection.class_table + " WHERE "
            query += "catalog_name = \'" + self.connection.data_catalog + "\' AND "
            query += "table_name = \'" + self.connection.data_table + "\' AND "
            query += "unique_key = -1 AND "
            query += "class_algorithm = \'" + "Not set" + "\' AND " 
            query += "class_script = \'" + self.scriptpath + "\' AND "
            query += "class_user = \'" + self.connection.data_username + "\'"

            # Execute a query without getting any data
            sqlHelper.execute_query(query, get_data=False, commit=True)   

            # Disconnect from database and commit all inserts
            sqlHelper.disconnect(commit=False)

            # Return the number of inserted rows
            return 1
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DataLayerException(f"Marking execution end failed: {e}")

    # Function for counting the number of rows in data to classify
    def count_data_rows(self, data_catalog=str, data_table=str) -> int:
        try:
            self.connection.data_catalog = data_catalog
            self.connection.data_table = data_table
            # Get a sql handler and connect to data database
            sqlHelper = self.get_sql_connection()
            
            #
            query = "SELECT COUNT(*) FROM "
            query += "[" + self.connection.data_catalog + "].[" + self.connection.data_table.replace(".","].[") + "]"

            if sqlHelper.execute_query(query, get_data=True):
                count = sqlHelper.read_data()[0]

            # Disconnect from database
            sqlHelper.disconnect()

        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DataLayerException(f"Count of dataset failed: {str(e)}")

        # Return 
        return count

     # Function for counting the number of rows in data corresponding to each class
    def count_class_distribution(self, class_column: str, data_catalog: str, data_table: str) -> dict:
        try:
            self.connection.class_column = class_column
            self.connection.data_catalog = data_catalog
            self.connection.data_table = data_table

            # Get a sql handler and connect to data database
            sqlHelper = self.get_sql_connection()
            
            # Construct the query
            query = "SELECT " + self.connection.class_column + ", COUNT(*) FROM "
            query += "[" + self.connection.data_catalog + "].[" + self.connection.data_table.replace(".","].[") + "] "
            query += "GROUP BY " + self.connection.class_column + " ORDER BY " + self.connection.class_column + " DESC"
            
            # Now we are ready to execute the sql query
            # By default, all fetched data is placed in one long 1-dim list. The alternative is to read by chunks.
            data = []
            if sqlHelper.execute_query(query, get_data=True):
                data = sqlHelper.read_all_data()
            dict = {}
            for pair in data:
                key, value = pair
                if key == None:
                    key = "NULL"
                dict[key] = value
            
            # Disconnect from database
            sqlHelper.disconnect()

        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DataLayerException(f"Count of dataset distribution failed: {str(e)}")

        # Return 
        return dict

    # For saving results in database
    def save_data(self,
        results: list,
        class_rate_type: RateType,
        model: Model,
        config: Config)-> int:

        try:
            # Get a sql handler and connect to data database
            sqlHelper = self.get_sql_connection()
            
            # Loop through the data
            num_lines = 0
            percent_fetched = 0.0
            result_num = len(results)
            
            columns = OrderedDict({
                "catalog_name": config.get_data_catalog(),
                "table_name": config.get_data_table(),
                "column_names": ",".join(config.get_data_column_names()), 
                "unique_key": "", # keys[i]
                "class_result": "", # Y[i]
                "class_rate": "", # rates[i]
                "class_rate_type": class_rate_type.name,
                "class_labels": ",".join(model.model.classes_),
                "class_probabilities": "", #probabilities[i]
                "class_algorithm": model.get_name(),
                "class_script": self.scriptpath,
                "class_user": config.get_data_username()
            })
            
            
            # Build up the base query outside of the range
            base_query = f"INSERT INTO {config.get_class_table()} "
            column_names = ",".join([column for column in columns.keys()])
            base_query += f"({column_names})"
            
            for row in results:
                columns["unique_key"] = row["key"]
                columns["class_result"] = row["prediction"]
                columns["class_rate"] = row["rate"]
                columns["class_probabilities"] = row["probabilities"]

                values = ",".join([to_quoted_string(elem) for elem in columns.values()])
                query = base_query + f" VALUES ({values})"

                # Execute a query without getting any data
                # Delay the commit until the connection is closed
                if sqlHelper.execute_query(query, get_data=False, commit=False):
                    num_lines += 1
                    percent_fetched = round(100.0 * float(num_lines) / float(result_num))
                    # TODO: clean up
                    if not self.logger.is_quiet(): print("Part of data saved: " + str(percent_fetched) + " %", end='\r')

            if not self.logger.is_quiet(): print("\n")
            
            if num_lines > 0:
                # Disconnect from database and commit all inserts
                sqlHelper.disconnect(commit=True)
            
            # Return the number of inserted rows
            return num_lines
        except KeyError as ke:
            print(ke)
            raise DataLayerException(f"Something went wrong when saving data ({ke})")
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DataLayerException(f"Something went wrong when saving data ({e})")

    def get_data_query(self, num_rows, train, predict) -> str:
        query = "SELECT "
        query += "TOP(" + str(num_rows) + ") "
        column_groups = (self.connection.data_text_columns, self.connection.data_numerical_columns, self.connection.id_column, self.connection.class_column)
        for column_group in column_groups:
                columns = column_group.split(',')
                for column in columns:
                    if column != "":
                        query += "[" + column + "],"

        # Remove last comma sign
        # Remove last comma sign from last statement above
        query = query[:-1]
        query += " FROM "
        outer_query = query
        query += "[" + self.connection.data_catalog + "].[" + self.connection.data_table.replace(".", "].[") + "]"

        # Take care of the special case of only training or only predictions
        if train and not predict:
            query += " WHERE [" + self.connection.class_column + "] IS NOT NULL AND CAST([" + \
                self.connection.class_column + "] AS VARCHAR) != \'\' "

        elif not train and predict:
            query += " WHERE [" + self.connection.class_column + "] IS NULL OR CAST([" + \
                self.connection.class_column + "] AS VARCHAR) = \'\' "



        # Since sorting the DataFrames directly does not seem to work right now (see below)
        # we sort the data in retreiving in directly in SQL. The "DESC" keyword makes sure
        # all NULL values (unclassified data) is placed last.
        # TODO: If num_rows is less than the number of available records, this query will fetch
        # num_rows randomly selected rows (NEWID does this trick)
        query += " ORDER BY NEWID() "
        query = outer_query + "(" + query + ") A "
        
        return query

    def get_dataset(self, num_rows: int, train: bool, predict: bool):
        try:
            # Get a sql handler and connect to data database
            sqlHelper = self.get_sql_connection()

             # Setup and execute a query to get the wanted data. 
            # 
            # The query has the form 
            # 
            #   SELECT columns FROM ( SELECT columns FROM table ORDER BY NEWID()) ORDER BY class_column
            #
            # to ensure a random selection (if not all rows are chosen) and ordering of data, and a 
            # subsequent sorting by class_column to put the NULL-classification last
            #
            query = self.get_data_query(num_rows, train, predict)
            non_ordered_query = query
            query += " ORDER BY [" +  str(self.connection.class_column) + "] DESC"

            self.logger.print_query("Classification data: ", query)

            # Now we are ready to execute the sql query
            # By default, all fetched data is placed in one long 1-dim list. The alternative is to read by chunks.
            num_lines = 0
            percent_fetched = 0
            data = []
            if sqlHelper.execute_query(query, get_data=True):

                # Read line by line from the query
                if not self.SQL_USE_CHUNKS:
                    row = sqlHelper.read_data()
                    data = [elem for elem in row]
                    num_lines = 1
                    percent_fetched = round(100.0 * float(num_lines) / float(num_rows))
                    
                    # TODO: Samma som i huvudklassen, \r
                    print("Data fetched of available: " + str(percent_fetched) + " %", end='\r')
                    while row:
                        if num_lines >= num_rows:
                            break;
                        row = sqlHelper.read_data()
                        if row:
                            data = data + [elem for elem in row]
                            num_lines += 1
                            old_percent_fetched = percent_fetched
                            percent_fetched = round(100.0 * float(num_lines) / float(num_rows))
                            if percent_fetched > old_percent_fetched:
                                # TODO: Samma som i huvudklassen, \r
                                print("Data fetched of available: " + str(percent_fetched) + " %", end='\r')

                    # Rearrange the long 1-dim array into a 2-dim numpy array which resembles the
                    # database table in question
                    data = np.asarray(data).reshape(num_lines,int(len(data)/num_lines)) 

                # Read data lines in chunks from the query
                else:
                    data_chunk = sqlHelper.read_many_data(chunksize=self.SQL_CHUNKSIZE)
                    data = np.asarray(data_chunk)
                    num_lines = len(data_chunk)
                    percent_fetched = round(100.0 * float(num_lines) / float(num_rows))
                    if not self.logger.is_quiet():
                        # TODO: Samma som i huvudklassen, \r
                        print("Data fetched of available: " + str(percent_fetched) + " %", end='\r')
                    while data_chunk:
                        if num_lines >= num_rows:
                            break;
                        data_chunk = sqlHelper.read_many_data(chunksize=self.SQL_CHUNKSIZE)
                        if data_chunk:
                            data = np.append(data, data_chunk, axis = 0)
                            num_lines += len(data_chunk)
                            old_percent_fetched = percent_fetched
                            percent_fetched = round(100.0 * float(num_lines) / float(num_rows))
                            if not self.logger.is_quiet() and percent_fetched > old_percent_fetched:
                                # TODO: Samma som i huvudklassen, \r
                                print("Data fetched of available: " + str(percent_fetched) + " %", end='\r')

            self.logger.print_formatted_info(f"Totally fetched {num_lines} data rows")
            
            # Disconnect from database
            sqlHelper.disconnect()

            # Quick return if no data was fetched
            if num_lines == 0:
                return None, None
            
            return data, non_ordered_query # list, string

        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DataLayerException(f"Something went wrong when fetching dataset ({e})")

    def correct_mispredicted_data(self, new_class: str, index: int) -> int:
        try:
            self.logger.print_info(f"Changing data row {index} to {new_class}: ")

            # Get a sql handler and connect to data database
            sqlHelper = self.get_sql_connection()

            # Set together SQL code for the insert
            query =  "UPDATE [" + self.connection.data_catalog + "]."
            query += "[" + "].[".join(self.connection.data_table.split('.')) + "]"
            query += " SET " + self.connection.class_column + " = \'" + new_class + "\'"  
            query += " WHERE " + self.connection.id_column + " = " + str(index)
            
            self.logger.print_query("mispredicted", query)
            
            # Execute a query without getting any data
            # Delay the commit until the connection is closed
            if sqlHelper.execute_query(query, get_data=False, commit=False):
                num_lines = 1
            else:
                num_lines = 0

            # Disconnect from database and commit all inserts
            sqlHelper.disconnect(commit=True)

            # Return the number of inserted rows
            return num_lines, query

        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DataLayerException(f"Correction of mispredicted data: {query} failed: {e}")

def drivers():
        return str(pyodbc.drivers())

def to_quoted_string(x, quotes:str = '\'') -> str:
    value = str(x)

    return quotes + value + quotes