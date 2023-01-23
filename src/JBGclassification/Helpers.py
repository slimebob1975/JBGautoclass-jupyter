import base64
from datetime import datetime
import getopt
import importlib
import os
import sys
from typing import Iterable


import numpy as np
import pandas
import IPython.display

def config_dict_to_list(order: int, config_dict: dict[str, str]) -> list[str]:
    """ Helps simplify the __str__ of the Config.* classes """
    title = config_dict.pop("title") # As all the places it's using right now has a title, it's a breaking error if not
    str_list = [f"{order}. {title}"] + [f" * {key}: {value}" for key, value in config_dict.items()]
    
    return "\n".join(str_list)


def html_wrapper(element: str, text: str, attributes: dict[str, str] = None) -> str:
    """ Wraps the text in HTML tags """
    opening_tag = f"<{element}"
    if attributes:
        for attribute, value in attributes.items():
            if value:
                opening_tag += f' {attribute}="{value}"'
    opening_tag += ">"
    
    return f"{opening_tag}{text}</{element}>"

def print_html(*args, **kwargs) -> None:
        """ Uses Display and HTML to display a text, possibly wrapped in HTML-tags """
        
        if kwargs.get("terminal_only"):
            return # This is an element that is only written in the terminal
        
        html_function = kwargs.get("html_function")
        text = " ".join([str(arg) for arg in args])
        if callable(html_function):
            text = html_function(text)
        else:
            text = html_wrapper("p", text)
        
        IPython.display.display(IPython.display.HTML(text))

def positive_int_or_none(value: int) -> bool:
    if value is None:
        return True
    
    if isinstance(value, int) and value >= 0:
        return True

    return False

def set_none_or_int(value) -> int:
    if value == "None":
        return None

    if int(value) < 0:
        return None

    return int(value)

# Returns a list
def get_from_string_or_list(value) -> list:
    if isinstance(value, str):
        return clean_column_names_list(value)

    if isinstance(value, list):
        return [ elem for elem in value if elem != ""]
    

# Turns a comma-delinated text string into a list of strings
def clean_column_names_list(column_names: str) -> list[str]:
        """ This takes a comma-delimeted string and returns the list with no empty values"""
        splitted = column_names.split(",")

        return [ elem for elem in splitted if elem != ""] # Removes all empty column names

# Help routines for determining consistency of input data
def is_float(val) -> bool:
    """ In DatasetHandler.read_data() """
    try:
        float(val)
    except ValueError:
        return False
    return True

def is_int(val) -> bool:
    """ In DatasetHandler.read_data() """
    try:
        int(val)
    except ValueError:
        return False
    return True

def is_str(val) -> bool:
    """ In DatasetHandler.read_data() """
    val_is_datetime = get_datetime(val)
    is_other_type = \
        is_float(val) or \
        is_int(val) or \
        isinstance(val, bool) or \
        val_is_datetime is not None
    return not is_other_type 
    
def get_datetime(val) -> datetime:
    """ In DatasetHandler.read_data() """
    try:
        if isinstance(val, datetime): #Simple, is already instance of datetime
            return val
    except ValueError:
        pass
    # Harder: test the value for many different datetime formats and see if any is correct.
    # If so, return it, otherwise, return None.
    the_formats = [
        '%Y-%m-%d %H:%M:%S','%Y-%m-%d','%Y-%m-%d %H:%M:%S.%f','%Y-%m-%d %H:%M:%S,%f', 
        '%d/%m/%Y %H:%M:%S','%d/%m/%Y','%d/%m/%Y %H:%M:%S.%f', '%d/%m/%Y %H:%M:%S,%f',
        '%m/%d/%Y %H:%M:%S','%m/%d/%Y','%m/%d/%Y %H:%M:%S.%f', '%m/%d/%Y %H:%M:%S,%f'
    ]
    for the_format in the_formats:
        try:
            date_time = datetime.strptime(str(val), the_format)
            if isinstance(date_time, datetime):
                return date_time
        except ValueError:
            pass
    return None

# Convert dataset to unreadable hex code
def do_hex_base64_encode_on_data(X):

    XX = X.copy()

    with np.nditer(XX, op_flags=['readwrite'], flags=["refs_ok"]) as iterator:
        for x in iterator:
            xval = str(x)
            xhex = cipher_encode_string(xval)
            x[...] = xhex 

    return XX

def cipher_encode_string(a):

    aa = a.split()
    b = ""
    for i in range(len(aa)):
        b += (str(
                base64.b64encode(
                    bytes(aa[i].encode("utf-8").hex(),"utf-8")
                )
            ) + " ")

    return b.strip()

def get_rid_of_decimals(x) -> int:
    return int(round(float(x)))

def find_smallest_class_number(Y: pandas.DataFrame) -> int:
    class_count = {}
    for elem in Y:
        if elem not in class_count:
            class_count[elem] = 1
        else:
            class_count[elem] += 1
    return max(1, min(class_count.values()))

def count_value_distr_as_dict(list: Iterable) -> dict:
    if isinstance(list, Iterable):
        return {key: list.count(key) for key in set(list)}
    else:
        raise ValueError("Input must be an iterable")

def add_prefix_to_dict_keys(prefix: str, the_dict: dict):
    new_dict = {}
    for key in the_dict.keys():
        new_dict[prefix+key] = the_dict[key]
    return new_dict

# In case the user has specified some input arguments to command line calls
# As written, you need to call on the class in the src\JBGclassification dir, with
# the configfilename on the format of ".\config\filename.py", where it has to be
# a subfolder in the src\JBGclassification dir
# This complicates testing a Config loaded from a file 
def check_input_arguments(argv: list):
    command_line_instructions = \
        f"Usage: {argv[0] } [-h/--help] [-f/--file <configfilename>]"

    try:
        short_options = "hf:"
        long_options = ["help", "file"]
        opts, args = getopt.getopt(argv[1:], short_options, long_options)
    except getopt.GetoptError:
        print(command_line_instructions)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h' or opt == '--help':
            print(command_line_instructions)
            sys.exit()
        elif opt == '-f' or opt == '--file':
            if arg.count("..") > 0:
                # This throws an error if you write the filename in the form of ..\
                print(
                    "Configuration file must be in a subfolder to {0}".format(argv[0]))
                sys.exit()
            print("Importing specified configuration file:", arg)
            
            # TODO: This does nothing as it currently stands
            if not arg[0] == '.': 
                arg = os.path.relpath(arg)
            
            file = arg.split('\\')[-1]
            filename = file.split('.')[0]
            filepath = '\\'.join(arg.split('\\')[:-1])
            paths = arg.split('\\')[:-1] # This is expected to be [".", "directory"]
            try:
                # This removes the "." from the above list
                paths.pop(paths.index('.'))
            except Exception as e:
                # Why is this an error? 
                # If there is no "." in the above list, then it's done without needing to pop it
                print("Filepath {0} does not seem to be relative (even after conversion)".format(
                    filepath))
                sys.exit()
            pack = '.'.join(paths)
            sys.path.insert(0, filepath)
            try:
                # This expects ex "config.filename"
                module = importlib.import_module(pack+"."+filename)
                
                return module
            except Exception as e:
                print("Filename {0} and pack {1} could not be imported dynamically".format(
                    filename, pack))
                sys.exit(str(e))
        else:
            print("Illegal argument to " + argv[0] + "!")
            print(command_line_instructions)
            sys.exit()
