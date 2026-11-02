from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)
    database_url: str = "sqlite:///runtime/studio.db"
    allow_live_api: bool = False
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4.1-mini"
    input_usd_per_million: float | None = Field(default=None, ge=0)
    output_usd_per_million: float | None = Field(default=None, ge=0)
    mlflow_tracking_uri: str = ""
    studio_api_token: SecretStr | None = None
