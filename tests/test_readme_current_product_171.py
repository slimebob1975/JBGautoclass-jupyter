from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ENV_TEMPLATE = ROOT / ".env_template"


def readme_text() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_is_current_product_guide_not_numbered_revision_history():
    text = readme_text()
    required_headings = (
        "## Main capabilities",
        "## Requirements",
        "## Configuration",
        "## Starting the GUI",
        "## Core workflow",
        "## Feature selection modelling",
        "## Files and runtime output",
        "## Troubleshooting",
    )
    for heading in required_headings:
        assert heading in text

    assert not re.search(r"\bRevision\s+\d+\b", text, flags=re.IGNORECASE)
    assert not re.search(r"\(\d{2,3}(?:[–-]\d{2,3})?\)", text)
    assert "docs/" not in text


def test_readme_documents_current_persistence_and_log_privacy_contracts():
    text = readme_text()
    assert "requirements-legacy-models.txt" in text
    assert "standard-library `pickle`" in text
    assert "JBG_LOG_SENSITIVE_DATA" in text
    assert "Feature search parallelism:" in text
    assert "every target value set to `NULL`" in text


def test_environment_template_keeps_parallelism_and_privacy_settings_separate():
    lines = ENV_TEMPLATE.read_text(encoding="utf-8").splitlines()
    n_jobs = [line for line in lines if line.startswith("DEFAULT_N_JOBS_DESIRED=")]
    privacy = [line for line in lines if line.startswith("JBG_LOG_SENSITIVE_DATA=")]
    assert len(n_jobs) == 1
    assert len(privacy) == 1
    assert "JBG_LOG_SENSITIVE_DATA" not in n_jobs[0]
