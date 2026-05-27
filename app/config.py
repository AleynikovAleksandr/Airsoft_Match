import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


class Settings(BaseSettings):
    SECRET_KEY: str = Field(...)
    ALGORITHM: str = Field(...)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(..., gt=0)

    DATABASE_URL: str = Field(...)

    MODEL_DIR: str = Field(...)
    TEXT_MODEL_NAME: str = Field(...)
    IMAGE_MODEL_NAME: str = Field(...)

    DATA_DIR: str = Field(...)
    POSTS_FILE: str = Field(...)
    PHOTOS_FILE: str = Field(...)
    SUBCATEGORY_IMAGES_DIR: str = Field(...)

    N_JOBS: int = Field(...)
    N_ESTIMATORS: int = Field(..., gt=0)
    IMAGE_LOAD_WORKERS: int = Field(..., gt=0)

    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env_multimodal"),
        extra="ignore",
    )


settings = Settings()
