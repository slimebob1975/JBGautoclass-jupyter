"""Revision 172b: formatted-info helpers must emit complete bold HTML."""
from pathlib import Path
import ast
import unittest


ROOT = Path(__file__).resolve().parents[1]
LOGGER_FILES = (
    ROOT / "src" / "JBGLogger.py",
    ROOT / "src" / "JBGStreamedLogger.py",
)


def _formatted_info_source(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "JBGLogger":
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "print_formatted_info":
                    return ast.get_source_segment(source, member) or ""
    raise AssertionError(f"print_formatted_info not found in {path.name}")


class RuntimeLogPrivacy172bTests(unittest.TestCase):
    def test_both_logger_variants_emit_complete_bold_closing_tag(self):
        for path in LOGGER_FILES:
            with self.subTest(logger=path.name):
                function_source = _formatted_info_source(path)
                self.assertIn('f"<b>{message}</b>"', function_source)
                self.assertNotIn('f"<b>{message}</b"', function_source)

    def test_fixed_markup_survives_default_path_redaction(self):
        from JBGLogPrivacy import redact_local_paths

        message = "<b>Fetched 10000 data rows in total</b>"
        self.assertEqual(redact_local_paths(message), message)


if __name__ == "__main__":
    unittest.main()
