"""Source-neutral logging port used by metadata reduction operations.

No concrete logger, GUI package, or runtime logging backend is imported here.
The historical JBGMeta.Logger import remains available as an alias.
"""

from typing import Protocol


class Logger(Protocol):
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""
    def print_info(self, *args) -> None:
        """printing info"""

    def print_progress(self, message: str = None, percent: float = None) -> None:
        """Printing progress"""

    def print_components(self, component, components, exception = None) -> None:
        """ Printing Reduction components"""
