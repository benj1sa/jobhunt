from pydantic import Field
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    database_url: str = Field(
        default="sqlite:///./jobhunt.db",
        description="SQLAlchemy database URL. SQLite is the supported default; PostgreSQL also works.",
    )
