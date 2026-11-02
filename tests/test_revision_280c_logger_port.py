"""280c: the reduction logger contract stays stable without GUI dependencies."""

import inspect

from JBGMeta import Logger as LegacyLogger
from JBGMetaProtocols import Logger


def test_legacy_logger_import_is_same_protocol():
    assert LegacyLogger is Logger
    assert inspect.isclass(Logger)
    assert getattr(Logger, "_is_protocol", False)


def test_logger_port_has_same_required_methods_and_signatures():
    assert set(Logger.__dict__).issuperset(
        {"print_info", "print_progress", "print_components"}
    )
    assert str(inspect.signature(Logger.print_info)) == "(self, *args) -> None"
    assert str(inspect.signature(Logger.print_progress)) == (
        "(self, message: str = None, percent: float = None) -> None"
    )
    assert str(inspect.signature(Logger.print_components)) == (
        "(self, component, components, exception=None) -> None"
    )


def test_port_does_not_import_gui_or_ml_engine():
    from pathlib import Path
    source = Path(inspect.getfile(Logger)).read_text(encoding="utf-8")
    assert "from GUI" not in source
    assert "import ipywidgets" not in source
    assert "import JBGMeta" not in source
