"""Revision 176c scoring-default regression contract.

This file intentionally uses only the standard library so its source/default-settings
contract can also be checked in dependency-light maintenance environments.
"""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _score_metric_default_member() -> str:
    tree = ast.parse((ROOT / "src" / "JBGMeta.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ScoreMetric":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "defaultScoreMetric":
                    for statement in item.body:
                        if isinstance(statement, ast.Return):
                            value = statement.value
                            if (
                                isinstance(value, ast.Attribute)
                                and isinstance(value.value, ast.Name)
                                and value.value.id == "cls"
                            ):
                                return value.attr
    raise AssertionError("ScoreMetric.defaultScoreMetric() return member not found")


def test_backend_default_score_metric_is_mcc():
    assert _score_metric_default_member() == "mcc"


def test_gui_default_score_metric_is_mcc():
    settings = json.loads((ROOT / "src" / "GUI" / "default_settings.json").read_text(encoding="utf-8"))
    assert settings["widgets"]["scoremetric_dropdown"]["params"]["value"] == "mcc"


def test_legacy_local_gui_default_migration_is_present():
    source = (ROOT / "src" / "GUI" / "Widgets.py").read_text(encoding="utf-8")
    assert 'if params.get("value") == "accuracy":' in source
    assert 'params["value"] = "mcc"' in source


if __name__ == "__main__":
    test_backend_default_score_metric_is_mcc()
    test_gui_default_score_metric_is_mcc()
    test_legacy_local_gui_default_migration_is_present()
