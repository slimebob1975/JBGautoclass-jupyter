"""Runtime policy for optional local GUI SQL credential prefill.

Credentials remain runtime-only.  JBG never writes them to configuration files or
back to ``.env``; a developer may opt in to local prefill by setting the documented
environment flag and credential variables themselves.
"""

from __future__ import annotations

from collections.abc import Mapping
import os

GUI_PREFILL_ENV = "JBG_GUI_PREFILL_SQL_CREDENTIALS"
SQL_USERNAME_ENV = "JBG_SQL_USERNAME"
SQL_PASSWORD_ENV = "JBG_SQL_PASSWORD"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def gui_sql_credential_prefill_enabled(environment: Mapping[str, str] | None = None) -> bool:
    """Return whether the GUI may prefill SQL credentials from the environment."""
    env = os.environ if environment is None else environment
    return str(env.get(GUI_PREFILL_ENV, "")).strip().lower() in _TRUTHY


def gui_sql_credential_defaults(environment: Mapping[str, str] | None = None) -> tuple[str, str]:
    """Return GUI SQL username/password defaults without persisting either value."""
    env = os.environ if environment is None else environment
    if not gui_sql_credential_prefill_enabled(env):
        return "", ""

    username = str(env.get(SQL_USERNAME_ENV, "") or "").strip()
    password = str(env.get(SQL_PASSWORD_ENV, "") or "")
    return username, password
