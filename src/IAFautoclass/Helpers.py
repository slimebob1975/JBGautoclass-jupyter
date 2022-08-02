import base64
from datetime import datetime


import numpy as np

# Help routines for determining consistency of input data
def is_float(val):
    try:
        float(val)
    except ValueError:
        return False
    return True

def is_int(val):
    try:
        int(val)
    except ValueError:
        return False
    return True

def is_str(val):
    is_other_type = \
        is_float(val) or \
        is_int(val) or \
        isinstance(val, bool) or \
        is_datetime(val)
    return not is_other_type 
    
def is_datetime(val):
    try:
        if isinstance(val, datetime): #Simple, is already instance of datetime
            return True
    except ValueError:
        pass
    # Harder: test the value for many different datetime formats and see if any is correct.
    # If so, return true, otherwise, return false.
    the_formats = ['%Y-%m-%d %H:%M:%S','%Y-%m-%d','%Y-%m-%d %H:%M:%S.%f','%Y-%m-%d %H:%M:%S,%f', \
                    '%d/%m/%Y %H:%M:%S','%d/%m/%Y','%d/%m/%Y %H:%M:%S.%f', '%d/%m/%Y %H:%M:%S,%f']
    for the_format in the_formats:
        try:
            date_time = datetime.strptime(str(val), the_format)
            return isinstance(date_time, datetime)
        except ValueError:
            pass
    return False

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
