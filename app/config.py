import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


class Settings(BaseSettings):
    SECRET_KEY: str = Field(..., min_length=1)
    ALGORITHM: str = Field(..., min_length=1)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(..., gt=0)

    DATABASE_URL: str = Field(..., min_length=1)

    MODEL_DIR: str = Field(..., min_length=1)
    TEXT_MODEL_NAME: str = Field(..., min_length=1)
    IMAGE_MODEL_NAME: str = Field(..., min_length=1)

    DATA_DIR: str = Field(..., min_length=1)
    POSTS_FILE: str = Field(..., min_length=1)
    PHOTOS_FILE: str = Field(..., min_length=1)
    SUBCATEGORY_IMAGES_DIR: str = Field(..., min_length=1)

    N_JOBS: int = Field(...)
    N_ESTIMATORS: int = Field(..., gt=0)
    IMAGE_LOAD_WORKERS: int = Field(..., gt=0)

    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env_multimodal"),
        extra="ignore",
    )


settings = Settings()
