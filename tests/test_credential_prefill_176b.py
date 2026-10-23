from pathlib import Path

from JBGCredentialPolicy import (
    GUI_PREFILL_ENV,
    SQL_PASSWORD_ENV,
    SQL_USERNAME_ENV,
    gui_sql_credential_defaults,
    gui_sql_credential_prefill_enabled,
)


ROOT = Path(__file__).resolve().parents[1]


def test_prefill_is_disabled_by_default_and_ignores_credentials():
    env = {
        SQL_USERNAME_ENV: "developer-user",
        SQL_PASSWORD_ENV: "developer-password",
    }

    assert gui_sql_credential_prefill_enabled(env) is False
    assert gui_sql_credential_defaults(env) == ("", "")


def test_prefill_requires_explicit_truthy_opt_in():
    for value in ("1", "true", "TRUE", "yes", "on"):
        env = {
            GUI_PREFILL_ENV: value,
            SQL_USERNAME_ENV: " developer-user ",
            SQL_PASSWORD_ENV: " developer-password ",
        }
        assert gui_sql_credential_prefill_enabled(env) is True
        assert gui_sql_credential_defaults(env) == (
            "developer-user",
            " developer-password ",
        )


def test_false_like_prefill_values_keep_credentials_hidden_from_gui_defaults():
    for value in ("", "0", "false", "no", "off", "unexpected"):
        env = {
            GUI_PREFILL_ENV: value,
            SQL_USERNAME_ENV: "developer-user",
            SQL_PASSWORD_ENV: "developer-password",
        }
        assert gui_sql_credential_prefill_enabled(env) is False
        assert gui_sql_credential_defaults(env) == ("", "")


def test_guihandler_uses_runtime_prefill_without_persisting_values():
    source = (ROOT / "src" / "GUIHandler.py").read_text(encoding="utf-8")
    assert "gui_sql_credential_defaults()" in source
    assert "gui_sql_credential_prefill_enabled()" in source
    assert "self.widgets.sql_username.value = sql_username" in source
    assert "self.widgets.sql_password.value = sql_password" in source


def test_env_template_is_production_safe_by_default():
    template = (ROOT / ".env_template").read_text(encoding="utf-8")
    assert 'JBG_GUI_PREFILL_SQL_CREDENTIALS="0"' in template
    assert 'JBG_SQL_USERNAME=""' in template
    assert 'JBG_SQL_PASSWORD=""' in template
