import opik

from app.config import get_settings


def get_opik_client():
    settings = get_settings()

    kwargs = {}

    if settings.opik_url_override:
        kwargs["url"] = settings.opik_url_override

    if settings.opik_workspace:
        kwargs["workspace"] = settings.opik_workspace

    if settings.opik_api_key:
        kwargs["api_key"] = settings.opik_api_key

    return opik.Opik(**kwargs)
