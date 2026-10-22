"""Revision 172a: path redaction must preserve HTML closing tags."""
import os
import unittest
from unittest.mock import patch

from JBGLogPrivacy import redact_local_paths


class RuntimeLogPrivacy172aTests(unittest.TestCase):
    def test_html_closing_tags_are_not_treated_as_posix_paths(self):
        message = (
            "<b>Last query added 10 rows.</b> "
            "<em>Dataset memory usage</em>: 1.5 MB "
            "<pre>diagnostic</pre> <span>ok</span>"
        )
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(redact_local_paths(message), message)

    def test_html_is_preserved_while_real_posix_path_is_redacted(self):
        message = "<em>Artifact</em>: /home/example/private/model.sav"
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                redact_local_paths(message),
                "<em>Artifact</em>: <local-path>",
            )

    def test_angle_bracketed_real_path_is_still_redacted(self):
        message = "artifact: </home/example/private/model.sav>"
        with patch.dict(os.environ, {}, clear=True):
            redacted = redact_local_paths(message)
        self.assertNotIn("home/example/private", redacted)
        self.assertIn("<local-path>", redacted)

    def test_sensitive_opt_in_still_bypasses_all_path_redaction(self):
        message = "<b>Artifact</b>: /home/example/private/model.sav"
        with patch.dict(os.environ, {"JBG_LOG_SENSITIVE_DATA": "1"}, clear=True):
            self.assertEqual(redact_local_paths(message), message)


if __name__ == "__main__":
    unittest.main()
