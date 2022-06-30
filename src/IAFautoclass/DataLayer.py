
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pyodbc
from Config import Config
import SqlHelper.IAFSqlHelper as sql

@dataclass
class DataLayer:
    SQL_CHUNKSIZE = 1000 #TODO: Decide how many
    SQL_USE_CHUNKS = True

    connection: Config.Connection
    scriptpath: str
    verbose: bool = False
    text_data: bool = field(init=False)
    numerical_data: bool = field(init=False)

    def __post_init__(self) -> None:
        # Extract parameters that are not textual
        self.text_data = self.connection.data_text_columns != ""
        self.numerical_data = self.connection.data_numerical_columns != ""

        if drivers().find(self.connection.odbc_driver) == -1:
            raise ValueError("Specified ODBC driver cannot be found!")
        
    # get the SQL handler
    def get_sql_connection(self) -> sql.IAFSqlHelper:
        # Get a sql handler and connect to data database
        sqlHelper = sql.IAFSqlHelper(driver = self.connection.odbc_driver, \
            host = self.connection.host, catalog = self.connection.class_catalog, \
            trusted_connection = self.connection.trusted_connection, \
            username = self.connection.class_username, \
            password = self.connection.class_password)
        sqlHelper.connect()

        return sqlHelper

    
    # Create the classification table
    def create_classification_table(self) -> None:

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
        except Exception:
            raise

    def mark_execution_ended(self) -> int:
        
        try:
            if self.verbose: print("Marking execution ended in database...")

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
        except Exception:
            raise

    # Function for counting the number of rows in data to classify
    def count_data_rows(self) -> int:

        try:

            # Get a sql handler and connect to data database
            sqlHelper = self.get_sql_connection()()
            
            #
            query = "SELECT COUNT(*) FROM "
            query += "[" + self.connection.data_catalog + "].[" + self.connection.data_table.replace(".","].[") + "]"

            if sqlHelper.execute_query(query, get_data=True):
                count = sqlHelper.read_data()[0]

            # Disconnect from database
            sqlHelper.disconnect()

        except Exception:
            raise
            # TODO: throw error and let GUI catch
            #print("Count of dataset failed: " + str(e))
            #if self.ProgressBar: self.ProgressBar.value = 1.0
            #sys.exit("Program aborted.")

        # Return 
        return count

     # Function for counting the number of rows in data corresponding to each class
    def count_class_distribution(self) -> dict:
        try:

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

        except Exception:
            raise
            # TODO: throw error and let GUI catch
            #print("Count of dataset distribution failed: " + str(e))
            #if self.ProgressBar: self.ProgressBar.value = 1.0
            #sys.exit("Program aborted.")

        # Return 
        return dict

    # For saving results in database
    def save_data(self, keys, Y, rates, prob_mode = 'U', labels='N/A', probabilities='N/A', alg = "Unknown") -> int:

        try:
            print(alg)
            # Get a sql handler and connect to data database
            sqlHelper = self.get_sql_connection()
            
            # Loop through the data
            num_lines = 0
            percent_fetched = 0.0
            for i in range(len(keys)):

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
                query += "\'," + str(keys[i]) + ",\'" + str(Y[i]) + "\'," + str(rates[i]) + "," 
                query += "\'" + prob_mode + "\'," + "\'" + ','.join(labels) + "\'," + "\'" + \
                    ','.join([str(elem) for elem in probabilities[i].tolist()]) + "\',"
                query += "\'" + alg + "\',\'" + self.scriptpath + "\',\'" + self.connection.data_username + "\')"

                # Execute a query without getting any data
                # Delay the commit until the connection is closed
                if sqlHelper.execute_query(query, get_data=False, commit=False):
                    num_lines += 1
                    percent_fetched = round(100.0 * float(num_lines) / float(len(keys)))
                    if self.verbose: print("Part of data saved: " + str(percent_fetched) + " %", end='\r')

            if self.verbose: print("\n")
                
            # Disconnect from database and commit all inserts
            sqlHelper.disconnect(commit=True)

            # Return the number of inserted rows
            return num_lines

        except Exception:
            raise

    def get_data_query(self, num_rows, train, predict) -> str:
        query = "SELECT "
        query += "TOP(" + str(num_rows) + ") "
        column_groups = (self.connection.id_column, self.connection.class_column,
                            self.connection.data_text_columns, self.connection.data_numerical_columns)
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

    def get_data(self, num_rows, debug, redirect_output, train, predict):
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

            if self.verbose: print("Query for classification data: ", query)

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
                    if not redirect_output:
                        print("Data fetched of available: " + str(percent_fetched) + " %", end='\r')
                    while row:
                        if debug and num_lines >= num_rows:
                            break;
                        row = sqlHelper.read_data()
                        if row:
                            data = data + [elem for elem in row]
                            num_lines += 1
                            old_percent_fetched = percent_fetched
                            percent_fetched = round(100.0 * float(num_lines) / float(num_rows))
                            if not redirect_output and percent_fetched > old_percent_fetched:
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
                    if self.verbose and not redirect_output:
                        print("Data fetched of available: " + str(percent_fetched) + " %", end='\r')
                    while data_chunk:
                        if debug and num_lines >= num_rows:
                            break;
                        data_chunk = sqlHelper.read_many_data(chunksize=self.SQL_CHUNKSIZE)
                        if data_chunk:
                            data = np.append(data, data_chunk, axis = 0)
                            num_lines += len(data_chunk)
                            old_percent_fetched = percent_fetched
                            percent_fetched = round(100.0 * float(num_lines) / float(num_rows))
                            if self.verbose and not redirect_output and percent_fetched > old_percent_fetched:
                                print("Data fetched of available: " + str(percent_fetched) + " %", end='\r')

            if self.verbose: print("\n--- Totally fetched {0} data rows ---".format(num_lines))
            
            # Disconnect from database
            sqlHelper.disconnect()

            # Quick return if no data was fetched
            if num_lines == 0:
                return None, None, None
            
            return data, non_ordered_query, num_lines

        except Exception:
            raise
    
    def mispredicted_data_query(self, new_class: str, index: str) -> int:
        try:
            if self.verbose: print("Changing data row {0} to {1}: ".format(index, new_class))

            # Get a sql handler and connect to data database
            sqlHelper = self.get_sql_connection

            # Set together SQL code for the insert
            query =  "UPDATE [" + self.connection.data_catalog + "]."
            query += "[" + "].[".join(self.connection.data_table.split('.')) + "]"
            query += " SET " + self.connection.class_column + " = \'" + new_class + "\'"  
            query += " WHERE " + self.connection.id_column + " = " + index
            
            if self.verbose: print("Executing SQL code: {0}".format(query))
            
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

        except Exception:
            raise #GUI's responsibility
            #print("Correction of mispredicted data: {0} failed: {1}".format(query,str(e)))

           



def drivers():
        return str(pyodbc.drivers())
