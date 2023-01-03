from decimal import Decimal
from datetime import datetime

import Helpers

# Since this is a series of functions, seems fitting to not put the tests into a class

def test_column_operations():
    # removes single empty value
    column_names = "a,,b"
    cleaned = Helpers.clean_column_names_list(column_names)
    assert cleaned == ["a", "b"]
    
    # Removes multiple empty values
    column_names = "a,,b,"
    cleaned = Helpers.clean_column_names_list(column_names)
    assert cleaned == ["a", "b"]

    # should return an empty list
    column_names = ""
    assert Helpers.clean_column_names_list(column_names) == []

    column_names = "a,,b"
    cleaned = Helpers.get_from_string_or_list(column_names)
    assert cleaned == ["a", "b"]

    column_names = ["a", "", "b"]
    cleaned = Helpers.get_from_string_or_list(column_names)
    assert cleaned == ["a", "b"]

    column_names = ["a", "b"]
    cleaned = Helpers.get_from_string_or_list(column_names)
    assert cleaned == ["a", "b"]
    
def test_positive_int_or_none():
        """ Help function to test input params"""
        # 1. None
        assert Helpers.positive_int_or_none(None)
        # 2. 3
        assert Helpers.positive_int_or_none(3)
        # 3. -2
        assert not Helpers.positive_int_or_none(-2)
        # 4. 3.5
        assert not Helpers.positive_int_or_none(3.5)
    
def test_set_none_or_int():
    """ Help function for loading input params from file """
    # 1: "None"
    assert Helpers.set_none_or_int("None") is None
    # 2. "5"
    value = Helpers.set_none_or_int("5")
    assert value == 5 and isinstance(value, int)
    # 3. 4
    assert Helpers.set_none_or_int(4) == 4
    # 4. -3
    assert Helpers.set_none_or_int(-3) is None
    # 5. -3.5
    assert Helpers.set_none_or_int(-3.5) is None

def test_is_float():
    """ Check a few different values to see what it can and can't handler"""
    
    # 1. String in a "float format" can be read as a float
    assert Helpers.is_float("1.5")

    # 2. General string can not
    assert not Helpers.is_float("abc")

    # 3. int can be read as a float
    assert Helpers.is_float(5)

    # 4. Float can be read as a float
    assert Helpers.is_float(1.5)

    # 5. Unlikely, but Decimal can be read as a float
    assert Helpers.is_float(Decimal(1.5))

    # 5. Unlikely, but Decimal can be read as a float
    assert Helpers.is_float(Decimal(1))

    # 6. Boolean values can be read as floats
    assert Helpers.is_float(False)
    assert Helpers.is_float(True)

def test_is_int():
    """ Valid floats can be read as int, but the value is cut off """
    # 1. String in a "float format" can't be read as an int
    assert not Helpers.is_int("1.5")

    # 2. General string can not
    assert not Helpers.is_int("abc")

    # 3. String in "int format" can
    assert Helpers.is_int("5")

    # 4. int can be read as an int
    assert Helpers.is_int(5)

    # 5. Float can be read as an int
    assert Helpers.is_int(1.5)

    # 6. Decimal with "float value" can be read as an int
    assert Helpers.is_int(Decimal(1.5))

    # 7.Decimal "int value" can be read as an int
    assert Helpers.is_int(Decimal(1))

    # 8. Boolean values can be read as int
    assert Helpers.is_int(False)
    assert Helpers.is_int(True)

def test_is_str():
    """ As the value you input is likely a string, needs to check value """
    assert not Helpers.is_str("1.5") # This is float

    assert Helpers.is_str("abc")

    assert not Helpers.is_str("5") # This is float or int

    assert not Helpers.is_str(5) # This is float or int

    assert not Helpers.is_str(1.5) # This is float or int

    assert not Helpers.is_str(Decimal(1.5)) # This is float or int

    assert not Helpers.is_str(Decimal(1)) # This is float or int

    assert not Helpers.is_str(False) # boolean
    
    assert not Helpers.is_str(True) # boolean

    assert not Helpers.is_str("2004-05-09") # Datetime


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