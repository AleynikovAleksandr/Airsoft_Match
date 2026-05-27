import os
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


class Settings(BaseSettings):
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    DATABASE_URL: str

    MODEL_DIR: str
    TEXT_MODEL_NAME: str
    IMAGE_MODEL_NAME: str

    DATA_DIR: str
    POSTS_FILE: str
    PHOTOS_FILE: str
    SUBCATEGORY_IMAGES_DIR: str

    N_JOBS: int
    N_ESTIMATORS: int
    IMAGE_LOAD_WORKERS: int

    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env_multimodal"),
        extra="ignore",
    )


settings = Settings()
