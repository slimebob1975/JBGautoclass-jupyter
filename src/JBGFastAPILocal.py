"""Local, offline and read-only FastAPI composition root (revision 282b).

Only binds to loopback; does not configure datasets or training. The in-memory
job registry starts empty until a trusted composition root adds job creation.
"""
from __future__ import annotations

from JBGDatasetApplicationService import DatasetApplicationService
from JBGIdentity import AuthenticationMode, IdentitySettings, LocalIdentityProvider
from JBGFastAPI import create_app
from JBGRunAPIPresentation import RunAPIPresentationService
from JBGRunExecutionService import RunExecutionService
from JBGRunJobService import RunJobService
from JBGRunPreparationService import RunPreparationService


def create_local_app(*, host: str = "127.0.0.1", subject: str = "developer"):
    settings = IdentitySettings(AuthenticationMode.LOCAL, "development", host)
    identity = LocalIdentityProvider(settings, subject)
    datasets = DatasetApplicationService()
    preparation = RunPreparationService(datasets)
    execution = RunExecutionService(preparation)
    jobs = RunJobService(execution)
    presentation = RunAPIPresentationService(jobs)
    return create_app(presentation, current_principal=identity.current_principal)


# Uvicorn CLI imports this module as JBGFastAPILocal:app.
app = create_local_app()
