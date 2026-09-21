import opik

from app.config import get_settings


def configure_opik() -> None:
    settings = get_settings()

    kwargs = {
        "project_name": settings.opik_project_name,
    }

    if settings.opik_url_override:
        kwargs["url_override"] = settings.opik_url_override

    if settings.opik_workspace:
        kwargs["workspace"] = settings.opik_workspace

    if settings.opik_api_key:
        kwargs["api_key"] = settings.opik_api_key

    opik.configure(**kwargs)
