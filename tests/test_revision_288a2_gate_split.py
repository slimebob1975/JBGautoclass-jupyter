"""288a2: classification of native end-to-end tests for the gate split."""
from conftest import NATIVE_FIXTURE_CLASS, is_native_test_class


def test_tests_built_on_the_native_fixture_are_native():
    Fixture = type(NATIVE_FIXTURE_CLASS, (), {})
    Derived = type("ProjectRunNativeExample", (Fixture,), {})
    Further = type("FurtherDerived", (Derived,), {})
    assert all(is_native_test_class(cls) for cls in (Fixture, Derived, Further))


def test_other_tests_are_not_native():
    Other = type("CSVApplicationEndToEndLookalike", (), {})
    assert not is_native_test_class(Other)
    assert not is_native_test_class(None)            # plain test functions


def test_the_shared_fixture_keeps_its_name():
    # Renaming the fixture class would silently empty the native group; the
    # gate also fails then (pytest exit code 5), but this names the cause.
    import test_revision_281c_csv_application_e2e as revision_281c
    assert hasattr(revision_281c, NATIVE_FIXTURE_CLASS)
