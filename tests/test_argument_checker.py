import pytest

from Helpers import check_input_arguments

class TestArgumentChecker:
    """ This can have more tests, but I need to understand why the function is written as is"""
    def test_bad_arguments(self):
        """ Valid arguments are -h or -f <filename>"""
        argv = [
            "script.py",
            "-i"
        ]
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            check_input_arguments(argv)

        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == 2

    def test_missing_filename(self):
        """ Needs <filename> """
        argv = [
            "script.py",
            "-f"
        ]
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            check_input_arguments(argv)

        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == 2

    def test_not_in_subfolder(self):
        """ Exits properly with a message if the file is in the wrong place
            It should be in a subfolder to the class you're calling (as written)
        """
        argv = [
            "script.py",
            "-f",
            "../filepath/name.py"
        ]
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            check_input_arguments(argv)

        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code is None

    def test_helpfile(self):
        """ Prints an instruction and then exits properly """
        argv = [
            "script.py",
            "-h"
        ]
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            check_input_arguments(argv)

        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code is None

        argv = [
            "script.py",
            "--help"
        ]
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            check_input_arguments(argv)

        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code is None
    
    #def test_argument_checker(self):
    #    # Valid arguments: name (argv[0] -h/--help, -f/--file <configgilename>
    #    # ['Config.py', '-f', '.\\config\\some-file-name']
    #    argv = [
    #        "test_script.py",
    #        "-f",
    #        "configfilename"
    #    ]
    #
    #    # Valid flags: 
    #    # -h/--help
    #    # -f/--file <configfile
        