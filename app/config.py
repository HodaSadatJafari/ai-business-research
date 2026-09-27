from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Business Research"
    environment: str = "development"

    openai_api_key: str
    openai_base_url: str | None = None
    openai_model: str = "gpt-5.6-luna"

    opik_project_name: str = "ai-business-research"
    opik_api_key: str | None = None
    opik_workspace: str | None = None
    opik_url_override: str | None = None

    knowledge_base_dir: Path = Path(__file__).resolve().parents[1] / "knowledge_base"
    retrieval_k: int = 4
    retrieval_method: str = "bm25"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
