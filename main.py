"""
main.py
=======
Точка входа приложения.

Логика запуска:
  1. Создаём таблицы в SQLite (если не существуют).
  2. Загружаем мультимодальную модель.
  3. Если классификаторы НЕ найдены в model/ — запускаем обучение.
  4. Стартуем FastAPI сервер.
"""

import logging
import os

import uvicorn
from fastapi import FastAPI

from src.api.routes import router
from src.db.database import create_tables
from src.models.inference import get_model
from src.models.trainer import ModelTrainer

# --------------------------------------------------------------------------- #
#  Настройка логирования                                                       #
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  FastAPI приложение                                                          #
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Airsoft Multimodal API",
    description=(
        "API для классификации страйкбольного снаряжения "
        "по тексту и фотографиям объявления.\n\n"
        "Используемые модели:\n"
        "- **ruBert-base** (ai-forever) — текстовый энкодер\n"
        "- **ViT-base-patch16-224** (google) — визуальный энкодер\n"
        "- **Random Forest** (sklearn) — финальный классификатор\n\n"
        "Аутентификация: Bearer JWT токен."
    ),
    version="1.0.0",
)

app.include_router(router)


# --------------------------------------------------------------------------- #
#  Жизненный цикл                                                              #
# --------------------------------------------------------------------------- #

@app.on_event("startup")
def startup_event():
    logger.info("Запуск приложения...")

    # 1. БД
    create_tables()
    logger.info("Таблицы SQLite инициализированы")

    # 2. Модель
    model = get_model()

    # 3. Первоначальное обучение (только если нет сохранённых классификаторов)
    if not model.is_ready():
        logger.info("Классификаторы не найдены — запускаем обучение...")
        trainer = ModelTrainer(model)
        trainer.train_all()
    else:
        logger.info("Классификаторы загружены из model/ — обучение пропускается")

    logger.info("Приложение готово к работе")


@app.get("/health", tags=["System"])
def health_check():
    model = get_model()
    return {
        "status": "ok",
        "model_ready": model.is_ready(),
        "categories": model.categories,
        "subcategories": model.subcategories,
    }


# --------------------------------------------------------------------------- #
#  Прямой запуск                                                               #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
