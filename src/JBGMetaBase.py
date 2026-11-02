"""Source-neutral base types for JBG metadata registries.

The public compatibility surface remains ``JBGMeta``. This module contains only
registry mechanics so later revisions can separate metadata from estimator
construction without changing existing ``JBGMeta`` imports.
"""

from __future__ import annotations

import enum
from functools import total_ordering
from typing import Iterable, Type, TypeVar

META_ENUM_DEFAULT_TERMS = ("ALL", "DUMY", "NOR", "NOS", "NOG", "NUG")

@total_ordering
class MetaEnum(enum.Enum):
    @property
    def full_name(self):
        if isinstance(self.value, dict):
            return self.value["full_name"]

        return self.value

    @classmethod
    def get_plain_list(cls) -> list[tuple[str, str]]:

        return [(item.full_name, item.name) for item in cls]

    @classmethod
    def get_sorted_list(cls, default_terms_first: bool = True) -> list[tuple[str, str]]:
        if not default_terms_first:
            return sorted([(item.full_name, item.name) for item in cls])

        excluded_enums = []
        for name in META_ENUM_DEFAULT_TERMS:
            try:
                item = cls[name]
            except KeyError:
                """ Left blank on purpose """
            else:
                excluded_enums.insert(0, item)

        listed_enums = sorted([(item.full_name, item.name) for item in cls if item not in excluded_enums])

        default_list = [(item.full_name, item.name) for item in excluded_enums]

        return default_list + listed_enums

    def get_function_name(self, do_or_get: str = 'do') -> str:
        if do_or_get == 'get':
            return f"get_{self.name}"
        elif do_or_get == 'do':
            return f"do_{self.name}"
        else:
            return None

    def has_function(self, do_or_get: str = 'do') -> bool:
        do_or_get = self.get_function_name(do_or_get)
        return hasattr(self, do_or_get) and callable(getattr(self, do_or_get))

    def call_function(self, do_or_get: str = 'do', **kwargs):
        do_or_get = self.get_function_name(do_or_get)
        if hasattr(self, do_or_get) and callable(func := getattr(self, do_or_get)):
            return func(**kwargs)

        return None

    def get_function(self, **kwargs):
        return self.call_function('get', **kwargs)

    def __eq__(self, other: MetaEnum) -> bool:
        """
        This bases the comparisons on the name
        """
        if (type(self) == type(other)):
            return self.name == other.name

        return False

    def __lt__(self, other: MetaEnum) -> bool:
        """
        This bases the comparisons on the name
        """
        return self.name < other.name


T = TypeVar('T', bound='MetaTuple')

class MetaTuple:
    """ Takes a list of MetaEnums and turns it into a Tuple"""

    @property
    def type(self) -> MetaEnum:
        if hasattr(self, "_type"):
            return self._type

        return MetaEnum

    @property
    def full_name(self):
        return self.get_full_names()

    @classmethod
    def from_string(cls: Type[T], commaseparated_list_as_string: str) -> T:
        """
        Creates a MetaEnum based on a (comma-separated) list in string format
        Strips spaces around each string value
        """
        return cls([x.strip() for x in commaseparated_list_as_string.split(",")])

    @classmethod
    def createTuple(cls, initiating_values: Iterable) -> None:
        """
        Overloaded initiation function
        """
        return cls(initiating_values)

    def __init__(self, initiating_values: Iterable) -> None:
        if isinstance(initiating_values, Iterable):
            metaEnums = []
            for metaEnum in initiating_values:
                if isinstance(metaEnum, self.type):
                    metaEnums.append(metaEnum)
                elif isinstance(metaEnum, str):
                    metaEnums.append(self.type[metaEnum])
                else:
                    raise ValueError(f"Each element in input list must be an instance of {self.type.__name__}")
            self.metaEnums = tuple(metaEnums)
        else:
            raise ValueError(f"Input to {self.__class__.__name__} must be an iterable")


    def __eq__(self, other: MetaTuple) -> bool:
        """
        This compares that the list of metaEnums contains the same metaEnums, does not care about order
        """
        return sorted(self.metaEnums) == sorted(other.metaEnums)

    def get_abbreviations(self) -> list[str]:
        """
        Returns a tuple of the abbreviations
        The list is the same order as the values were initiated in
        """
        return [x.name for x in self.metaEnums]

    def __str__(self) -> str:
        """
        Returns a comma-separated list of the abbreviations
        The list is the same order as the values were initiated in
        """
        return ", ".join([x.name for x in self.metaEnums])

    def get_full_names(self) -> str:
        """
        Returns a comma-separated list of the full names
        The list is the same order as the values were initiated in
        """
        return ", ".join([x.full_name for x in self.metaEnums])

    def get_full_names_and_abbrevs(self) -> str:
        """
        Returns a comma-separated list of the full names + abbreviations
        The list is the same order as the values were initiated in
        """
        list = [f"{x.full_name} ({x})" for x in self.metaEnums]
        return ", ".join(list)


    def list_callables(self, do_or_get: str, **kwargs) -> list[tuple]:
        """ Gets a list of metaEnums that are callable
            in the form (metaEnum, called function)
        """
        # TODO: Semantically speaking, it should be "get" for all MetaEnum
        # Tuples, but that requires a larger operation to rewrite the function
        # names in the MetaEnums. Needs done, but not right now
        metaEnums = [(
            metaEnum,
            metaEnum.call_function(do_or_get, **kwargs))
                for metaEnum in self.metaEnums if metaEnum.has_function(do_or_get)
            ]
        metaEnums.sort(key=lambda enumtuple: enumtuple[0].name)
        return metaEnums
