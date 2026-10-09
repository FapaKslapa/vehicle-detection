from __future__ import annotations

import sys
from typing import Literal, TypeVar

from pydantic import AnyHttpUrl, Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_S = TypeVar("_S", bound=BaseSettings)
_BUCKET_PATTERN = r"^[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]$"


class ConfigError(RuntimeError):
    pass


class _Base(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)


class StorageSettings(_Base):
    s3_endpoint_url: AnyHttpUrl
    s3_bucket: str = Field(pattern=_BUCKET_PATTERN)
    s3_region: str = Field(default="garage", min_length=1)
    aws_access_key_id: SecretStr = Field(min_length=8)
    aws_secret_access_key: SecretStr = Field(min_length=16)

    @field_validator("aws_access_key_id", "aws_secret_access_key")
    @classmethod
    def _no_whitespace(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value()
        if secret != secret.strip() or any(c.isspace() for c in secret):
            raise ValueError("contiene spazi o a capo (controlla il copia e incolla)")
        return value

    @property
    def endpoint(self) -> str:
        return str(self.s3_endpoint_url).rstrip("/")


class TrackingSettings(_Base):
    wandb_api_key: SecretStr | None = None
    wandb_entity: str | None = None
    wandb_project: str = Field(default="vehicle-detection", min_length=1)
    wandb_mode: Literal["online", "offline", "disabled"] = "online"

    @field_validator("wandb_api_key")
    @classmethod
    def _api_key_shape(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 40:
            raise ValueError("troppo corta per una chiave W&B (attese almeno 40 caratteri)")
        return value

    def require_online_credentials(self) -> None:
        if self.wandb_mode == "online" and self.wandb_api_key is None:
            raise ConfigError(
                "WANDB_API_KEY mancante con WANDB_MODE=online: "
                "impostala oppure usa WANDB_MODE=offline o disabled"
            )


def _format(error: ValidationError) -> str:
    lines = []
    for item in error.errors():
        name = ".".join(str(part) for part in item["loc"]).upper() or "(generale)"
        problem = "mancante" if item["type"] == "missing" else item["msg"]
        lines.append(f"  {name}: {problem}")
    return "\n".join(lines)


def _load(cls: type[_S], label: str) -> _S:
    try:
        return cls()
    except ValidationError as error:
        raise ConfigError(f"Configurazione {label} non valida:\n{_format(error)}") from None


def load_storage_settings() -> StorageSettings:
    return _load(StorageSettings, "storage S3")


def load_tracking_settings() -> TrackingSettings:
    settings = _load(TrackingSettings, "tracking W&B")
    settings.require_online_credentials()
    return settings


def main() -> int:
    try:
        storage = load_storage_settings()
        tracking = load_tracking_settings()
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"storage: {storage.endpoint} bucket={storage.s3_bucket} region={storage.s3_region}")
    print(f"tracking: project={tracking.wandb_project} mode={tracking.wandb_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
