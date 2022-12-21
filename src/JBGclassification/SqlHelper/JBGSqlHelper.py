import sys
import platform
from typing import Generator, Iterator, List, Sequence, Union
from JBGExceptions import SQLException

if platform.system() == 'Windows':
    import pyodbc
else:
    sys.exit("Aborting! This application cannot run on platform: " + str(platform.system()))      

class JBGSqlHelper():
    """JBG SQL helper class, using ODBC to connect to database"""
    standard_chunksize = 10000
    
    # Constructor with arguments
    def __init__(self, driver = "", host = "", catalog = "", \
                 trusted_connection = False, username = "", password = "", \
                 ignore_errors = True ):
        
        self.Driver = driver
        self.Host = host
        self.Catalog = catalog
        if trusted_connection:
            self.Trusted_connection = True
            self.Username = None
            self.Password = None
        else:
            self.Trusted_connection = False
            self.Username = username
            self.Password = password
        self.Connection = None
        self.Cursor = None
        self.ignore_errors = ignore_errors

    # Destructor
    def __del__(self):
        if self.Cursor:
            self.Cursor.close()
            self.Cursor = None
        if self.Connection:
            self.Connection.close()
            self.Connection = None
    
    # Print the class 
    def __str__(self):
        return self.build_connection()

    def build_connection(self) -> str:
        fields = {
            "driver": self.Driver,
            "server": self.Host,
            "database": self.Catalog
        }

        if self.Trusted_connection:
            fields["trusted_connection"] = "yes"

        else:
            fields["uid"] = self.Username
            fields["pwd"] = self.Password
        
        connection = [ f"{key.upper()}={value}" for (key,value) in fields.items()]
        
        return ";".join(connection)

    def build_connection_string(self) -> str:
        if not self.Trusted_connection:
            connect_string = \
                "DRIVER=" + self.Driver + ";" + \
                "SERVER=" + self.Host + ";" + \
                "DATABASE=" + self.Catalog + ";" + \
                "UID=" + self.Username +";" + \
                "PWD=" + self.Password + ";"
        else:
            connect_string = \
                "DRIVER=" + self.Driver + ";" + \
                "SERVER=" + self.Host + ";" + \
                "DATABASE=" + self.Catalog + ";" + \
                "TRUSTED_CONNECTION=yes;" 
        
        return connect_string

    def can_connect(self) -> bool:
        connect_string = self.build_connection()
        connection = self.connect(connect_string, False)

        if (type(connection) == pyodbc.Connection):
            return True

        return False

    # Make a connection to database
    def connect(self, connect: str = None, print_fail: bool = True) -> pyodbc.Connection:
        connect_string = connect
        if not connect_string:
            connect_string = self.build_connection()
        
        try:
            return pyodbc.connect(connect_string)
        except Exception as ex:
            raise SQLException(str(ex))
    
    # Disconnect from SQL server and close cursor
    def disconnect(self, commit = False):
        if commit and self.Connection:
            self.Connection.commit()
        if self.Cursor:
            self.Cursor.close()
            self.Cursor = None
        if self.Connection:
            self.Connection.close()
            self.Connection = None

    # Execute query with/without data retreival 
    def execute_query(self, query, get_data = False, commit = True):

        if self.Connection == None:
            self.Connection = self.connect()
            self.Cursor = None
        if self.Cursor == None:
            self.Cursor = self.Connection.cursor()

        try:
            self.Cursor.execute(query)
            if not get_data:
                if commit:
                    self.Connection.commit()
                return 1
            else:
                return self.Cursor
        except Exception as ex:
            raise SQLException(str(ex))    
    
    def execute_query_with_params(self, query: str, params: tuple, get_data: bool = False, commit: bool = True):
        """ Uses Cursor.execute with a parameterised query """
        if self.Connection == None:
            self.Connection = self.connect()
            self.Cursor = None
        if self.Cursor == None:
            self.Cursor = self.Connection.cursor()

        try:
            self.Cursor.execute(query, params)
            if not get_data:
                if commit:
                    self.Connection.commit()
                return 1
            else:
                return self.Cursor
        except Exception as ex:
            raise SQLException(str(ex))

    def execute_many_queries(self, query: str, params: Union[Sequence, Iterator, Generator], commit: bool = True) -> None:
        """ Uses Cursor.executemany with a parameterised query """
        if self.Connection == None:
            self.Connection = self.connect()
            self.Cursor = None
        if self.Cursor == None:
            self.Cursor = self.Connection.cursor()

        try:
            self.Cursor.fast_executemany = True
            self.Cursor.executemany(query, params)
        
            if commit:
                self.Connection.commit()
            return
        except Exception as ex:
            print(ex)
            raise SQLException(str(ex))
    
    def read_next(self, chunksize: int = None):
        if chunksize:
            return self.read_many_data(chunksize)
        
        return self.read_data()

    @staticmethod
    def columns_to_parameters(columns) -> str:
        """ Returns a string on the form of VALUES(?,?,?---), 
            the number of ? based on number of columns 
        """
        values = ",".join(["?" for x in range(len(columns))])
        return f"VALUES({values})"

    def check_table_exists(self, table: str) -> bool:
        """" Runs a query to check if the given table exists """
        query_strings = [
            f"IF OBJECT_ID('{table}','U') IS NULL SELECT 0;",
            "ELSE SELECT 1;"
        ]
        
        query = " ".join(query_strings)
        
        try:
            value = self.get_data_from_query(query)
            return value[0][0] == 1
        except Exception as e:
            raise SQLException(str(e))

    def insert_row(self, table: str, columns: list, values: tuple, return_id: bool = True, commit: bool = True):
        """" Inserts row and, if wanted returns id """
        id = None
        query = JBGSqlHelper.get_insert_statement(table, columns)
        
        if return_id:
            queries = [
                "SET NOCOUNT ON",
                query,
                "SELECT @@IDENTITY AS table_id"
            ]

            query = ";".join(queries)

        cur = self.execute_query_with_params(query, params=values, get_data=True, commit=commit)
        
        if return_id:
            id = cur.fetchone().table_id
        
        self.disconnect(commit=commit)
        return id

    def insert_many_rows(self, table: str, columns: list, params: Union[Sequence, Iterator, Generator], commit: bool = True) -> int:
        """ Inserts several rows into a single table """
        query = JBGSqlHelper.get_insert_statement(table, columns)
        #print(query)
        self.execute_many_queries(query, params, commit=commit)

        return len(params)

    # Read next line of data, if possible
    def read_data(self) -> pyodbc.Row:

        try:
            row = self.Cursor.fetchone()
            return row
        except Exception as ex:
            raise SQLException(str(ex))
            

    # Read many (chunksize) lines of data, if possible
    def read_many_data(self, chunksize=standard_chunksize)  -> List[pyodbc.Row]:

        try:
            data = self.Cursor.fetchmany(chunksize)
            return data
        except Exception as ex:
            raise SQLException(str(ex))
            

    # Read all (remaining) lines of data, if possible
    def read_all_data(self) -> List[pyodbc.Row]:

        try:
            data = self.Cursor.fetchall()
            return data
        except Exception as ex:
            raise SQLException(str(ex))
            
    
    def get_count(self, endQuery: str) -> int:
        """ Creates a COUNT query and returns the count
            endQuery is on the form of "[database].[table] WHERE x = y"
        """

        query = f"SELECT COUNT(*) FROM {endQuery}"
        count = -1

        if self.execute_query(query, get_data=True):
            count = self.read_data()[0]

        # Disconnect from database
        self.disconnect()

        return count


    def get_data_from_query(self, query: str) -> list:
        """ Gets the full set of data from a query """
        try:
            self.execute_query(query, get_data = True)
        except Exception as ex:
            raise SQLException(str(ex))
        
        data = self.read_all_data()
        self.disconnect()

        return data
    
    # Print available drivers
    @staticmethod
    def drivers():
        return str(pyodbc.drivers())

    @staticmethod
    def get_insert_statement(table: str, columns) -> str:
        """ Given a table and an iterable of column names returns an insert statement with placeholders """
        columns_string = ",".join([f"[{column}]" for column in columns])
        insert = f"INSERT INTO {table} ({columns_string}) {JBGSqlHelper.columns_to_parameters(columns)}"
        
        return insert


# Main method
def main():
    print(JBGSqlHelper.drivers())

# Start main
if __name__ == "__main__":
    main()