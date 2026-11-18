from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    environment: str = os.getenv("TOPTA_ENV", "MOCK").upper()
    database_path: str = os.getenv("TOPTA_DATABASE", "data/t-opta.sqlite3")
    log_level: str = os.getenv("TOPTA_LOG_LEVEL", "INFO").upper()

    def validate(self):
        if self.environment not in {"MOCK", "STAGING", "PRODUCTION"}:
            raise ValueError(f"Unsupported TOPTA_ENV: {self.environment}")

settings = Settings()
settings.validate()
