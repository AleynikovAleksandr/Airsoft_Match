"""
inference.py
============
Обёртка для инференса: принимает входные данные запроса API
и возвращает структурированный ответ.
"""

import logging
from typing import Optional

from app.ml.multimodal_model import AirsoftMultimodalModel

logger = logging.getLogger(__name__)

# Синглтон — модель загружается один раз при старте приложения
_model: Optional[AirsoftMultimodalModel] = None


def get_model() -> AirsoftMultimodalModel:
    """Возвращает глобальный экземпляр модели (ленивая инициализация)."""
    global _model
    if _model is None:
        logger.info("Инициализация модели (первый вызов)...")
        _model = AirsoftMultimodalModel()
    return _model


def run_inference(post_id: str, text: str, photos: list[dict]) -> dict:
    """
    Запускает предсказание для одного запроса.

    Args:
        post_id: Идентификатор поста.
        text:    Текст объявления.
        photos:  Список {"photo_id": str, "url": str}.

    Returns:
        Словарь формата:
        {
            "post_id": "...",
            "predictions": [
                {
                    "object_id": "...",
                    "category": "...",
                    "subcategory": "...",
                    "confidence": 0.9,
                    "photo_ids": [...]
                }, ...
            ]
        }
    """
    model = get_model()

    if not model.is_ready():
        logger.warning(
            "Модель не обучена — возвращаются placeholder предсказания"
        )

    predictions = model.predict(text, photos)

    return {
        "post_id": post_id,
        "predictions": predictions,
    }
