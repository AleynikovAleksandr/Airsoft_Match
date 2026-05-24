from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    # JWT
    SECRET_KEY: str = "super-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 часа

    # База данных
    DATABASE_URL: str = "sqlite:///./database/users.db"

    # Модели
    MODEL_DIR: str = "model"
    TEXT_MODEL_NAME: str = "ai-forever/ruBert-base"
    IMAGE_MODEL_NAME: str = "google/vit-base-patch16-224"

    # Данные
    DATA_DIR: str = "data/raw"
    POSTS_FILE: str = "data/raw/posts.parquet"
    PHOTOS_FILE: str = "data/raw/photos.parquet"
    SUBCATEGORY_IMAGES_DIR: str = "data/raw/subcategory_images"

    # Обучение
    N_JOBS: int = -1          # все ядра для Random Forest
    N_ESTIMATORS: int = 200   # деревьев в лесу
    IMAGE_LOAD_WORKERS: int = 4

    class Config:
        env_file = ".env"


settings = Settings()
