import sys
import platform
from typing import List

if platform.system() == 'Windows':
    import pyodbc
else:
    sys.exit("Aborting! This application cannot run on platform: " + str(platform.system()))      

class IAFSqlHelper():
    """IAF SQL helper class, using ODBC to connect to database"""
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
            func = str(sys._getframe().f_code.co_name)
            if (print_fail):
                print("Connection to database failed: " + connect_string)
            if not self.ignore_errors:
                self.end_program( func, str(ex) )
    
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
            func = str(sys._getframe().f_code.co_name)
            print("Execution of query failed: " + query)
            if not self.ignore_errors:
                self.end_program( func, str(ex) )    
    
    # Read next line of data, if possible
    def read_data(self):

        try:
            row = self.Cursor.fetchone()
            return row
        except Exception as ex:
            func = str(sys._getframe().f_code.co_name)
            print("Execution failed!")
            if not self.ignore_errors:
                self.end_program( func, str(ex) )

    # Read many (chunksize) lines of data, if possible
    def read_many_data(self, chunksize=standard_chunksize):

        try:
            data = self.Cursor.fetchmany(chunksize)
            return data
        except Exception as ex:
            func = str(sys._getframe().f_code.co_name)
            print("Execution failed!")
            if not self.ignore_errors:
                self.end_program( func, str(ex) )

    # Read all (remaining) lines of data, if possible
    def read_all_data(self) -> List[pyodbc.Row]:

        try:
            data = self.Cursor.fetchall()
            return data
        except Exception as ex:
            func = str(sys._getframe().f_code.co_name)
            print("Execution failed!")
            if not self.ignore_errors:
                self.end_program( func, str(ex) )
            
    # Print available drivers
    @staticmethod
    def drivers():
        return str(pyodbc.drivers())

    # End program if something went wrong
    @staticmethod
    def end_program(func = "Unknown function", mess = "Unknown error"):
        sys.exit("Program malfunctioned in: " + func + \
                 " with error: " + mess)

# Main method
def main():
    print(IAFSqlHelper.drivers())

# Start main
if __name__ == "__main__":
    main()