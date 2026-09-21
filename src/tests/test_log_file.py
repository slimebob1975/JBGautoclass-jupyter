import re
import sys
import warnings

import pytest

from JBGLogFile import TimestampedLogFile, capture_console_output


_LOG_PREFIX = r"\[[^\]]+\]"


def read_log(log_file: TimestampedLogFile) -> str:
    log_file.flush()
    return log_file.path.read_text(encoding="utf-8")


def test_timestamped_log_file_assigns_and_preserves_levels(tmp_path):
    log_file = TimestampedLogFile(log_dir=tmp_path, filename="levels.log")

    log_file.write("plain info\n")
    log_file.write("[DEBUG] debug detail\n")
    log_file.write("WARNING: legacy warning\n")
    log_file.write_message("table snapshot", level="TABLE")

    text = read_log(log_file)
    assert re.search(rf"^{_LOG_PREFIX} \[INFO\] plain info$", text, flags=re.MULTILINE)
    assert re.search(rf"^{_LOG_PREFIX} \[DEBUG\] debug detail$", text, flags=re.MULTILINE)
    assert re.search(rf"^{_LOG_PREFIX} \[WARNING\] legacy warning$", text, flags=re.MULTILINE)
    assert re.search(rf"^{_LOG_PREFIX} \[INFO\] \[TABLE\] table snapshot$", text, flags=re.MULTILINE)

    log_file.close()


def test_timestamped_log_file_references_server_log_from_environment(tmp_path, monkeypatch):
    server_log = tmp_path / "jbg-server.log"
    monkeypatch.setenv("JBG_SERVER_LOG", str(server_log))

    log_file = TimestampedLogFile(log_dir=tmp_path, filename="application.log")
    text = read_log(log_file)

    assert f"[INFO] Server log: {server_log}" in text

    log_file.close()


def test_capture_console_output_logs_stdout_stderr_warnings_and_exceptions(tmp_path):
    log_file = TimestampedLogFile(log_dir=tmp_path, filename="capture.log")

    with capture_console_output(log_file):
        print("stdout message")
        print("stderr message", file=sys.stderr)
        warnings.warn("runtime warning", RuntimeWarning)

    with pytest.raises(ValueError):
        with capture_console_output(log_file):
            raise ValueError("boom")

    text = read_log(log_file)
    assert "[INFO] stdout message" in text
    assert "[ERROR] stderr message" in text
    assert "[WARNING] RuntimeWarning: runtime warning" in text
    assert "[EXCEPTION] Exception escaped captured application scope" in text
    assert "[ERROR] Traceback (most recent call last):" in text
    assert "[ERROR] ValueError: boom" in text

    log_file.close()
