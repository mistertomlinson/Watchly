from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.version import __version__


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    TMDB_API_KEY: str | None = None
    PORT: int = 8000
    ADDON_ID: str = "com.bimal.watchly"
    ADDON_NAME: str = "Watchly"
    REDIS_URL: str = "redis://redis:6379/0"
    REDIS_MAX_CONNECTIONS: int = 20
    REDIS_CONNECTIONS_THRESHOLD: int = 100
    REDIS_TOKEN_KEY: str = "watchly:token:"
    TOKEN_SALT: str = "change-me"
    TOKEN_TTL_SECONDS: int = 0
    ANNOUNCEMENT_HTML: str = ""
    AUTO_UPDATE_CATALOGS: bool = True
    CATALOG_REFRESH_INTERVAL_SECONDS: int = 21600
    MANIFEST_CACHE_TTL_SECONDS: int = 21600
    APP_ENV: Literal["development", "production", "vercel"] = "production"
    HOST_NAME: str = "https://1ccea4301587-watchly.baby-beamup.club"

    RECOMMENDATION_SOURCE_ITEMS_LIMIT: int = 10
    LIBRARY_ITEMS_LIMIT: int = 20
    CATALOG_CACHE_TTL: int = 43200
    CATALOG_STALE_TTL: int = 604800

    DEFAULT_GEMINI_MODEL: str = "gemma-3-27b-it"
    GEMINI_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None

    TRAKT_CLIENT_ID: str | None = None
    TRAKT_CLIENT_SECRET: str | None = None

    # Simkl OAuth. SIMKL_CLIENT_ID is also the API key sent in the
    # simkl-api-key header for both public and authenticated requests.
    SIMKL_CLIENT_ID: str | None = None
    SIMKL_CLIENT_SECRET: str | None = None


settings = Settings()
APP_VERSION = __version__
