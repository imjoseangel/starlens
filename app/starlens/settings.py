"""StarLens — Centralized configuration via pydantic-settings.

All tuneable values live here. Override via environment variables or a `.env` file.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeminiSettings(BaseSettings):
    """Google AI Studio (Gemini API) connection and Gemma 4 model configuration."""

    model_config = SettingsConfigDict(env_prefix="STARLENS_GEMINI_")

    api_key: str = Field(
        default="",
        description="Google AI Studio API key (https://aistudio.google.com/apikey)",
    )
    model_identify: str = Field(
        default="gemma-4-26b-a4b-it",
        description="Gemma 4 MoE model for multimodal identification (4B active params — fast & efficient)",
    )
    model_reason: str = Field(
        default="gemma-4-31b-it",
        description="Gemma 4 Dense 31B for deep reasoning, planning, and explanation (256K context)",
    )
    available_models: list[str] = Field(
        default=["gemma-4-26b-a4b-it", "gemma-4-31b-it"],
        description="Gemma 4 models shown in the UI dropdown",
    )
    max_output_tokens: int = Field(
        default=8192,
        description="Maximum tokens to generate per response",
    )


class ModelOptions(BaseSettings):
    """Inference options — temperature per task type."""

    model_config = SettingsConfigDict(env_prefix="STARLENS_OPTS_")

    temperature_identify: float = Field(default=0.2)
    temperature_explain: float = Field(default=0.5)
    temperature_plan: float = Field(default=0.4)
    temperature_narrate: float = Field(default=0.7)
    temperature_tour: float = Field(default=0.6)
    temperature_chat: float = Field(default=0.5)
    temperature_chart: float = Field(default=0.3)
    temperature_why: float = Field(default=0.4)


class AppSettings(BaseSettings):
    """Gradio application settings."""

    model_config = SettingsConfigDict(env_prefix="STARLENS_APP_")

    port: int = Field(default=8000, description="Server port")
    host: str = Field(default="0.0.0.0", description="Server bind address")
    share: bool = Field(default=False, description="Create Gradio share link")
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    default_city: str = Field(
        default="Madrid, Spain",
        description="Default location shown in UI",
    )
    fallback_lat: float = Field(default=40.4168)
    fallback_lon: float = Field(default=-3.7038)


class RedisSettings(BaseSettings):
    """Redis caching configuration."""

    model_config = SettingsConfigDict(env_prefix="STARLENS_REDIS_")

    url: str = Field(
        default="",
        description="Redis URL (empty = caching disabled)",
    )
    ttl_sky: int = Field(
        default=300,
        description="TTL in seconds for sky computation cache",
    )
    ttl_llm: int = Field(
        default=600,
        description="TTL in seconds for LLM response cache",
    )


class Settings(BaseSettings):
    """Root settings aggregating all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    options: ModelOptions = Field(default_factory=ModelOptions)
    app: AppSettings = Field(default_factory=AppSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)


# Module-level singleton — import this everywhere
settings = Settings()
