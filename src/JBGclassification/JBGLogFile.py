"""Small, dependency-free helpers for persistent application logging."""

import atexit
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
import os
from pathlib import Path
import re
import sys
import threading
import traceback
from typing import Optional, TextIO, Union
import warnings


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_EXPLICIT_LEVEL_RE = re.compile(
    r"^\[(DEBUG|INFO|WARNING|ERROR|CRITICAL|EXCEPTION)\]\s*",
    flags=re.IGNORECASE,
)
_LEGACY_LEVEL_RE = re.compile(
    r"^(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|EXCEPTION)\s*:\s*",
    flags=re.IGNORECASE,
)
_LEVEL_ALIASES = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "WARN": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
    "EXCEPTION": "EXCEPTION",
}


def normalize_log_level(level: Optional[str], default: str = "INFO") -> str:
    """Return a stable, uppercase application log level."""
    if level is None:
        return default

    normalized = str(level).strip().upper()
    if normalized in ("ALWAYS", "UNFORMATTED"):
        return "INFO"
    return _LEVEL_ALIASES.get(normalized, default)


class TeeStream:
    """Write the same text to multiple file-like streams."""

    def __init__(self, *streams: TextIO):
        self.streams = tuple(stream for stream in streams if stream is not None)

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return bool(self.streams and getattr(self.streams[0], "isatty", lambda: False)())

    def fileno(self) -> int:
        if not self.streams:
            raise OSError("TeeStream has no underlying stream")
        return self.streams[0].fileno()

    @property
    def encoding(self):
        if not self.streams:
            return None
        return getattr(self.streams[0], "encoding", None)


class LeveledLogStream:
    """File-like adapter that assigns a default level to streamed text."""

    def __init__(self, log_file: "TimestampedLogFile", level: str):
        self.log_file = log_file
        self.level = normalize_log_level(level)

    def write(self, data: str) -> int:
        return self.log_file.write_with_level(data, level=self.level)

    def flush(self) -> None:
        self.log_file.flush()

    @property
    def encoding(self):
        return "utf-8"


class TimestampedLogFile:
    """Line-oriented UTF-8 log file with local timestamps and log levels."""

    def __init__(self, log_dir: Optional[Union[str, Path]] = None, filename: Optional[str] = None):
        if log_dir is None:
            log_dir = Path(__file__).resolve().parent / "output" / "logs"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"jbg-autoclass_{timestamp}_pid{os.getpid()}.log"

        self.path = self.log_dir / filename
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)
        self._at_line_start = True
        self._lock = threading.RLock()
        self.write_message("Log started", level="INFO")
        self.write_message(f"Process ID: {os.getpid()}", level="INFO")
        self.write_message(f"Python: {sys.version.split()[0]}", level="INFO")
        self.write_message(f"Working directory: {Path.cwd()}", level="INFO")
        atexit.register(self.close)

    def _timestamp(self) -> str:
        return datetime.now().astimezone().isoformat(sep=" ", timespec="milliseconds")

    def _extract_level(self, chunk: str, default_level: str) -> tuple[str, str]:
        """Extract an explicit/legacy level prefix, otherwise use the default."""
        stripped = chunk.lstrip()
        indentation = chunk[:len(chunk) - len(stripped)]

        match = _EXPLICIT_LEVEL_RE.match(stripped)
        if match:
            level = normalize_log_level(match.group(1), default=default_level)
            return level, indentation + stripped[match.end():]

        match = _LEGACY_LEVEL_RE.match(stripped)
        if match:
            level = normalize_log_level(match.group(1), default=default_level)
            return level, indentation + stripped[match.end():]

        return default_level, chunk

    def write(self, data: str) -> int:
        """Write streamed stdout-like text, treating it as INFO by default."""
        return self.write_with_level(data, level="INFO")

    def write_with_level(self, data: str, level: str = "INFO") -> int:
        """Write streamed text with a timestamp and level on each non-empty line."""
        if data is None:
            return 0

        text = _ANSI_ESCAPE_RE.sub("", str(data)).replace("\r", "\n")
        if not text:
            return 0

        default_level = normalize_log_level(level)
        with self._lock:
            for chunk in text.splitlines(keepends=True):
                if self._at_line_start and chunk not in ("\n", "\r"):
                    line_level, chunk = self._extract_level(chunk, default_level)
                    self._stream.write(f"[{self._timestamp()}] [{line_level}] ")
                self._stream.write(chunk)
                self._at_line_start = chunk.endswith("\n")
            self._stream.flush()

        return len(data)

    def write_message(self, *args, level: str = "INFO") -> None:
        message = " ".join(str(arg) for arg in args)
        requested_level = str(level).strip().upper() if level is not None else "INFO"
        normalized_level = normalize_log_level(level)
        if requested_level not in _LEVEL_ALIASES and requested_level not in ("ALWAYS", "UNFORMATTED", "INFO"):
            message = f"[{requested_level}] {message}"
        self.write_with_level(message + "\n", level=normalized_level)

    def flush(self) -> None:
        with self._lock:
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            if not self._stream.closed:
                self.write_message("Log closed", level="INFO")
                self._stream.close()


def _show_warning(message, category, filename, lineno, file=None, line=None) -> None:
    """Render Python warnings as a single structured WARNING line."""
    del file, line
    print(
        f"[WARNING] {category.__name__}: {message} ({filename}:{lineno})",
        file=sys.stderr,
    )


@contextmanager
def capture_console_output(log_file: TimestampedLogFile):
    """Duplicate console output and warnings to the persistent session log."""

    stdout_tee = TeeStream(sys.stdout, LeveledLogStream(log_file, "INFO"))
    stderr_tee = TeeStream(sys.stderr, LeveledLogStream(log_file, "ERROR"))

    # Do not globally suppress warning categories. During an application run,
    # show each warning once per source location and route it through stderr so
    # it receives a stable WARNING level in the persistent log.
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        warnings.showwarning = _show_warning

        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            try:
                yield
            except Exception:
                log_file.write_message("Exception escaped captured application scope", level="EXCEPTION")
                traceback.print_exc(file=LeveledLogStream(log_file, "ERROR"))
                raise
