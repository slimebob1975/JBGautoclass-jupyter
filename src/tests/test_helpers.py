from decimal import Decimal
from datetime import datetime
import pytest

import Helpers

# Since this is a series of functions, seems fitting to not put the tests into a class
def test_is_float():
    """ Check a few different values to see what it can and can't handler"""
    
    # 1. String in a "float format" can be read as a float
    assert Helpers.is_float("1.5") is True

    # 2. General string can not
    assert Helpers.is_float("abc") is False

    # 3. int can be read as a float
    assert Helpers.is_float(5) is True

    # 4. Float can be read as a float
    assert Helpers.is_float(1.5) is True

    # 5. Unlikely, but Decimal can be read as a float
    assert Helpers.is_float(Decimal(1.5)) is True

    # 5. Unlikely, but Decimal can be read as a float
    assert Helpers.is_float(Decimal(1)) is True

    # 6. Boolean values can be read as floats
    assert Helpers.is_float(False) is True
    assert Helpers.is_float(True) is True

def test_is_int():
    """ Valid floats can be read as int, but the value is cut off """
    # 1. String in a "float format" can't be read as an int
    assert Helpers.is_int("1.5") is False

    # 2. General string can not
    assert Helpers.is_int("abc") is False

    # 3. String in "int format" can
    assert Helpers.is_int("5") is True

    # 4. int can be read as an int
    assert Helpers.is_int(5) is True

    # 5. Float can be read as an int
    assert Helpers.is_int(1.5) is True

    # 6. Decimal with "float value" can be read as an int
    assert Helpers.is_int(Decimal(1.5)) is True

    # 7.Decimal "int value" can be read as an int
    assert Helpers.is_int(Decimal(1)) is True

    # 8. Boolean values can be read as int
    assert Helpers.is_int(False) is True
    assert Helpers.is_int(True) is True

def test_is_str():
    """ As the value you input is likely a string, needs to check value """
    assert Helpers.is_str("1.5") is False # This is float

    assert Helpers.is_str("abc") is True

    assert Helpers.is_str("5") is False # This is float or int

    assert Helpers.is_str(5) is False # This is float or int

    assert Helpers.is_str(1.5) is False # This is float or int

    assert Helpers.is_str(Decimal(1.5)) is False # This is float or int

    assert Helpers.is_str(Decimal(1)) is False # This is float or int

    assert Helpers.is_str(False) is False # boolean
    
    assert Helpers.is_str(True) is False # boolean

    assert Helpers.is_str("2004-05-09") is False # Datetime


def test_get_datetime():
    """ This checks both type and some various forms of formats """
    now = datetime.now()

    assert isinstance(Helpers.get_datetime(now), datetime)

    assert isinstance(Helpers.get_datetime("2004-04-12 12:34:17"), datetime)
    assert isinstance(Helpers.get_datetime("2005-03-19"), datetime)
    assert isinstance(Helpers.get_datetime("2005-03-19 12:34:17.000001"), datetime)
    assert isinstance(Helpers.get_datetime("2005-03-19 12:34:17,000001"), datetime)
    assert isinstance(Helpers.get_datetime("24/11/2024 15:18:11"), datetime)
    assert isinstance(Helpers.get_datetime("19/03/2022"), datetime)
    assert isinstance(Helpers.get_datetime("19/03/2022 12:34:17.000001"), datetime)
    assert isinstance(Helpers.get_datetime("19/03/2022 12:34:17,000001"), datetime)
    assert isinstance(Helpers.get_datetime("11/2/2024 15:18:11"), datetime)
    assert isinstance(Helpers.get_datetime("12/3/2022"), datetime)
    assert isinstance(Helpers.get_datetime("12/3/2022 12:34:17.000001"), datetime)
    assert isinstance(Helpers.get_datetime("12/3/2022 12:34:17,000001"), datetime)
    assert isinstance(Helpers.get_datetime("24/11/2024 15:18:11"), datetime)
    assert isinstance(Helpers.get_datetime("19/3/2022"), datetime)
    assert isinstance(Helpers.get_datetime("19/3/2022 12:34:17.000001"), datetime)
    assert isinstance(Helpers.get_datetime("19/3/2022 12:34:17,000001"), datetime)
    assert isinstance(Helpers.get_datetime("19/03/2022"), datetime)
    assert isinstance(Helpers.get_datetime("19/03/2022 12:34:17.000001"), datetime)
    assert isinstance(Helpers.get_datetime("19/03/2022 12:34:17,000001"), datetime)

    # A few with invalid values
    assert Helpers.get_datetime("35/3/2022") is None # Day outside of valid days


    # Testdata for Validation
    # airline_tweets_mini, breast_cancer, creditcard_fraud, stroke_data och stars är alla vettiga testset