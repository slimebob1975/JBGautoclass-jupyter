"""Resolve historical model module names without retaining the old directory."""

import importlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys
from types import ModuleType


_SOURCE_DIR = Path(__file__).resolve().parent
_LEGACY_PACKAGES = (
    "src.JBGclassification", "JBGclassification",
    "src.JBGClassification", "JBGClassification",
)


class _LegacyModuleLoader(importlib.abc.Loader):
    def __init__(self, canonical=None):
        self.canonical = canonical

    def create_module(self, spec):
        if self.canonical is not None:
            # Share the real class/function objects, especially stored enums.
            return importlib.import_module(self.canonical)
        module = ModuleType(spec.name)
        module.__path__ = [str(_SOURCE_DIR)]
        return module

    def exec_module(self, module):
        pass


class _LegacyModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for prefix in (*_LEGACY_PACKAGES, "src"):
            if fullname == prefix and prefix != "src":
                return importlib.util.spec_from_loader(
                    fullname, _LegacyModuleLoader(), is_package=True,
                )
            if fullname.startswith(prefix + "."):
                canonical = fullname[len(prefix) + 1:]
                candidate = _SOURCE_DIR.joinpath(*canonical.split("."))
                if candidate.with_suffix(".py").is_file() or candidate.is_dir():
                    return importlib.util.spec_from_loader(
                        fullname, _LegacyModuleLoader(canonical),
                        is_package=candidate.is_dir(),
                    )
        return None


def install_legacy_import_aliases():
    """Enable old pickle/dill/joblib globals using the current source modules."""
    for directory in (_SOURCE_DIR.parent, _SOURCE_DIR):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    if not any(isinstance(finder, _LegacyModuleFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LegacyModuleFinder())
